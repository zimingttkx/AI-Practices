"""
文本转语音模型 (Text-to-Speech, TTS)

本模块实现端到端的文本转语音系统，包括：
- 文本编码器
- 声学模型 (Tacotron 风格)
- 声码器 (HiFi-GAN 风格)

=== TTS 系统架构 ===

文本 → [文本编码器] → 文本特征
                ↓
        [声学模型/解码器]
                ↓
        Mel 频谱图
                ↓
          [声码器]
                ↓
           音频波形

=== 核心组件 ===

1. 文本编码器 (Text Encoder)
   - 字符/音素嵌入
   - 卷积层提取局部特征
   - Transformer 编码器

2. 声学模型 (Acoustic Model)
   - 自回归解码器 (Tacotron 风格)
   - 或非自回归解码器 (FastSpeech 风格)
   - 预测 Mel 频谱图

3. 声码器 (Vocoder)
   - 将 Mel 频谱转换为波形
   - HiFi-GAN: 基于 GAN 的高保真声码器

=== 参考文献 ===

1. Tacotron 2:
   Shen et al. "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" 2018

2. FastSpeech 2:
   Ren et al. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" 2020

3. HiFi-GAN:
   Kong et al. "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" 2020
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTSConfig:
    """TTS 模型配置"""

    # 文本配置
    vocab_size: int = 256  # 字符词表大小
    max_text_length: int = 512

    # 音频配置
    n_mels: int = 80
    max_mel_length: int = 1024
    sample_rate: int = 22050

    # 编码器配置
    encoder_dim: int = 256
    encoder_conv_layers: int = 3
    encoder_conv_kernel: int = 5
    encoder_heads: int = 4
    encoder_layers: int = 4
    encoder_ff_dim: int = 1024

    # 解码器配置
    decoder_dim: int = 256
    decoder_heads: int = 4
    decoder_layers: int = 4
    decoder_ff_dim: int = 1024
    prenet_dim: int = 256
    postnet_channels: int = 512
    postnet_kernel: int = 5
    postnet_layers: int = 5

    # 声码器配置
    vocoder_upsample_rates: Tuple[int, ...] = (8, 8, 2, 2)
    vocoder_upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4)
    vocoder_resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11)
    vocoder_resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
        (1, 3, 5), (1, 3, 5), (1, 3, 5)
    )
    vocoder_initial_channel: int = 512

    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConvBlock(nn.Module):
    """卷积块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class TextEncoder(nn.Module):
    """文本编码器"""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config

        # 字符嵌入
        self.embedding = nn.Embedding(config.vocab_size, config.encoder_dim)

        # 卷积层
        self.conv_layers = nn.ModuleList([
            ConvBlock(
                config.encoder_dim,
                config.encoder_dim,
                config.encoder_conv_kernel,
                config.dropout
            )
            for _ in range(config.encoder_conv_layers)
        ])

        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.encoder_dim, config.max_text_length, config.dropout
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_ff_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.encoder_layers)

    def forward(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text: 输入文本 token [batch, seq_len]
            text_mask: 填充掩码 [batch, seq_len]
        Returns:
            编码器输出 [batch, seq_len, encoder_dim]
        """
        # 嵌入
        x = self.embedding(text)

        # 卷积层 (需要转换维度)
        x = x.transpose(1, 2)  # [batch, dim, seq_len]
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)  # [batch, seq_len, dim]

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer
        if text_mask is not None:
            x = self.transformer(x, src_key_padding_mask=~text_mask)
        else:
            x = self.transformer(x)

        return x


class Prenet(nn.Module):
    """预网络 - 解码器输入处理"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Postnet(nn.Module):
    """后网络 - Mel 频谱精修"""

    def __init__(self, config: TTSConfig):
        super().__init__()

        layers = []
        for i in range(config.postnet_layers):
            in_ch = config.n_mels if i == 0 else config.postnet_channels
            out_ch = config.n_mels if i == config.postnet_layers - 1 else config.postnet_channels

            layers.append(nn.Conv1d(
                in_ch, out_ch, config.postnet_kernel,
                padding=(config.postnet_kernel - 1) // 2
            ))
            if i < config.postnet_layers - 1:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(config.dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel 频谱 [batch, n_mels, time]
        Returns:
            精修后的 Mel 频谱
        """
        return self.layers(mel)


class MelDecoder(nn.Module):
    """Mel 频谱解码器 (Tacotron 风格)"""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config

        # 预网络
        self.prenet = Prenet(config.n_mels, config.prenet_dim, config.decoder_dim)

        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.decoder_dim, config.max_mel_length, config.dropout
        )

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.decoder_layers)

        # Mel 输出投影
        self.mel_linear = nn.Linear(config.decoder_dim, config.n_mels)

        # 停止预测
        self.stop_linear = nn.Linear(config.decoder_dim, 1)

        # 后网络
        self.postnet = Postnet(config)

    def forward(
        self,
        encoder_output: torch.Tensor,
        mel_target: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_output: 编码器输出 [batch, text_len, encoder_dim]
            mel_target: 目标 Mel 频谱 [batch, n_mels, mel_len] (训练时)
            encoder_mask: 编码器掩码
        Returns:
            mel_output: Mel 频谱输出
            mel_postnet: 后网络精修后的 Mel 频谱
            stop_tokens: 停止 token 预测
        """
        if mel_target is not None:
            # 训练模式: Teacher forcing
            mel_input = mel_target.transpose(1, 2)  # [batch, mel_len, n_mels]
            mel_input = F.pad(mel_input, (0, 0, 1, 0))[:, :-1, :]  # 右移一位

            # 预网络
            decoder_input = self.prenet(mel_input)
            decoder_input = self.pos_encoding(decoder_input)

            # 因果掩码
            mel_len = decoder_input.size(1)
            causal_mask = torch.triu(
                torch.ones(mel_len, mel_len, device=decoder_input.device), diagonal=1
            ).bool()

            # Transformer 解码
            decoder_output = self.transformer(
                decoder_input, encoder_output,
                tgt_mask=causal_mask,
                memory_key_padding_mask=~encoder_mask if encoder_mask is not None else None
            )

            # Mel 输出
            mel_output = self.mel_linear(decoder_output)
            mel_output = mel_output.transpose(1, 2)  # [batch, n_mels, mel_len]

            # 停止 token
            stop_tokens = self.stop_linear(decoder_output).squeeze(-1)

            # 后网络
            mel_postnet = mel_output + self.postnet(mel_output)

            return mel_output, mel_postnet, stop_tokens
        else:
            # 推理模式: 自回归生成
            return self._inference(encoder_output, encoder_mask)

    def _inference(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_len: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """自回归推理"""
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        # 初始化
        mel_outputs = []
        stop_tokens = []

        # 起始帧 (全零)
        current_mel = torch.zeros(batch_size, 1, self.config.n_mels, device=device)

        for _ in range(max_len):
            # 预网络
            decoder_input = self.prenet(current_mel)
            decoder_input = self.pos_encoding(decoder_input)

            # Transformer 解码
            decoder_output = self.transformer(
                decoder_input, encoder_output,
                memory_key_padding_mask=~encoder_mask if encoder_mask is not None else None
            )

            # 取最后一帧
            last_output = decoder_output[:, -1:, :]

            # Mel 输出
            mel_frame = self.mel_linear(last_output)
            mel_outputs.append(mel_frame)

            # 停止 token
            stop_token = torch.sigmoid(self.stop_linear(last_output).squeeze(-1))
            stop_tokens.append(stop_token)

            # 更新输入
            current_mel = torch.cat([current_mel, mel_frame], dim=1)

            # 检查停止条件
            if (stop_token > 0.5).all():
                break

        # 合并输出
        mel_output = torch.cat(mel_outputs, dim=1).transpose(1, 2)
        stop_tokens = torch.cat(stop_tokens, dim=1)

        # 后网络
        mel_postnet = mel_output + self.postnet(mel_output)

        return mel_output, mel_postnet, stop_tokens


class ResBlock(nn.Module):
    """HiFi-GAN 残差块"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5)
    ):
        super().__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            padding = (kernel_size * dilation - dilation) // 2
            self.convs1.append(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=dilation, padding=padding
            ))
            self.convs2.append(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=1, padding=(kernel_size - 1) // 2
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv1(x)
            x = F.leaky_relu(x, 0.1)
            x = conv2(x)
            x = x + residual
        return x


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN 声码器生成器"""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config

        # 初始卷积
        self.conv_pre = nn.Conv1d(
            config.n_mels, config.vocoder_initial_channel,
            kernel_size=7, padding=3
        )

        # 上采样层
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = config.vocoder_initial_channel
        for i, (u, k) in enumerate(zip(
            config.vocoder_upsample_rates,
            config.vocoder_upsample_kernel_sizes
        )):
            self.ups.append(nn.ConvTranspose1d(
                ch, ch // 2, k, stride=u, padding=(k - u) // 2
            ))

            ch = ch // 2

            for j, (k_r, d_r) in enumerate(zip(
                config.vocoder_resblock_kernel_sizes,
                config.vocoder_resblock_dilation_sizes
            )):
                self.resblocks.append(ResBlock(ch, k_r, d_r))

        # 输出卷积
        self.conv_post = nn.Conv1d(ch, 1, kernel_size=7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel 频谱 [batch, n_mels, time]
        Returns:
            波形 [batch, 1, time * prod(upsample_rates)]
        """
        x = self.conv_pre(mel)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            # 应用残差块
            n_resblocks = len(self.config.vocoder_resblock_kernel_sizes)
            xs = None
            for j in range(n_resblocks):
                if xs is None:
                    xs = self.resblocks[i * n_resblocks + j](x)
                else:
                    xs += self.resblocks[i * n_resblocks + j](x)
            x = xs / n_resblocks

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class TextToSpeech(nn.Module):
    """端到端 TTS 模型"""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config

        self.encoder = TextEncoder(config)
        self.decoder = MelDecoder(config)
        self.vocoder = HiFiGANGenerator(config)

    def forward(
        self,
        text: torch.Tensor,
        mel_target: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        训练前向传播
        Args:
            text: 输入文本 [batch, text_len]
            mel_target: 目标 Mel 频谱 [batch, n_mels, mel_len]
            text_mask: 文本掩码
        Returns:
            包含各种输出的字典
        """
        # 编码文本
        encoder_output = self.encoder(text, text_mask)

        # 解码 Mel 频谱
        mel_output, mel_postnet, stop_tokens = self.decoder(
            encoder_output, mel_target, text_mask
        )

        return {
            "mel_output": mel_output,
            "mel_postnet": mel_postnet,
            "stop_tokens": stop_tokens
        }

    @torch.no_grad()
    def synthesize(
        self,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        合成语音
        Args:
            text: 输入文本 [batch, text_len]
        Returns:
            音频波形 [batch, 1, audio_len]
        """
        # 编码
        encoder_output = self.encoder(text, text_mask)

        # 解码 Mel
        _, mel_postnet, _ = self.decoder(encoder_output, None, text_mask)

        # 声码器
        waveform = self.vocoder(mel_postnet)

        return waveform


def tts_loss(
    mel_output: torch.Tensor,
    mel_postnet: torch.Tensor,
    mel_target: torch.Tensor,
    stop_tokens: torch.Tensor,
    stop_target: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """
    计算 TTS 损失
    Args:
        mel_output: 解码器 Mel 输出
        mel_postnet: 后网络 Mel 输出
        mel_target: 目标 Mel 频谱
        stop_tokens: 停止 token 预测
        stop_target: 停止 token 目标
    Returns:
        总损失和各项损失字典
    """
    # Mel 损失
    mel_loss = F.mse_loss(mel_output, mel_target)
    mel_postnet_loss = F.mse_loss(mel_postnet, mel_target)

    # 停止 token 损失
    stop_loss = F.binary_cross_entropy_with_logits(stop_tokens, stop_target)

    # 总损失
    total_loss = mel_loss + mel_postnet_loss + stop_loss

    return total_loss, {
        "mel_loss": mel_loss,
        "mel_postnet_loss": mel_postnet_loss,
        "stop_loss": stop_loss
    }


def create_tts_model(size: str = "base") -> TextToSpeech:
    """
    创建预定义大小的 TTS 模型
    Args:
        size: 模型大小 ("tiny", "base", "large")
    Returns:
        TextToSpeech 模型实例
    """
    configs = {
        "tiny": TTSConfig(
            encoder_dim=128,
            encoder_layers=2,
            decoder_dim=128,
            decoder_layers=2,
            vocoder_initial_channel=256
        ),
        "base": TTSConfig(
            encoder_dim=256,
            encoder_layers=4,
            decoder_dim=256,
            decoder_layers=4,
            vocoder_initial_channel=512
        ),
        "large": TTSConfig(
            encoder_dim=512,
            encoder_layers=6,
            decoder_dim=512,
            decoder_layers=6,
            vocoder_initial_channel=512
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")

    return TextToSpeech(configs[size])
