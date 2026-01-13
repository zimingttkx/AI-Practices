"""
FastSpeech2 非自回归文本转语音模型 (Non-Autoregressive TTS)

本模块实现 FastSpeech 2 风格的非自回归 TTS 系统，包括：
- 文本编码器 (Transformer)
- 变分适配器 (时长、音高、能量预测)
- 长度调节器
- Mel 解码器

=== FastSpeech2 核心优势 ===

相比自回归 TTS (如 Tacotron):
1. 并行生成: 所有 Mel 帧同时生成，速度快
2. 可控性: 显式控制时长、音高、能量
3. 稳定性: 避免自回归的累积误差

=== 模型架构 ===

文本 → [文本编码器] → 隐藏表示
              ↓
    [变分适配器: 时长/音高/能量预测]
              ↓
    [长度调节器: 扩展到帧级别]
              ↓
    [Mel 解码器] → Mel 频谱图

=== 参考文献 ===

1. FastSpeech:
   Ren et al. "FastSpeech: Fast, Robust and Controllable Text to Speech" 2019

2. FastSpeech 2:
   Ren et al. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" 2020
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FastSpeech2Config:
    """FastSpeech2 模型配置"""

    # 文本配置
    vocab_size: int = 256
    max_seq_len: int = 1000

    # 编码器配置
    encoder_hidden: int = 256
    encoder_layers: int = 4
    encoder_heads: int = 2
    encoder_ff_dim: int = 1024
    encoder_conv_kernel: int = 9

    # 解码器配置
    decoder_hidden: int = 256
    decoder_layers: int = 4
    decoder_heads: int = 2
    decoder_ff_dim: int = 1024
    decoder_conv_kernel: int = 9

    # 变分适配器配置
    variance_predictor_kernel: int = 3
    variance_predictor_filter: int = 256
    variance_predictor_dropout: float = 0.5

    # 音高配置
    pitch_feature_level: str = "phoneme"  # "phoneme" or "frame"
    pitch_quantization: str = "linear"  # "linear" or "log"
    n_bins: int = 256
    pitch_min: float = 50.0
    pitch_max: float = 600.0

    # 能量配置
    energy_feature_level: str = "phoneme"
    energy_quantization: str = "linear"
    energy_min: float = 0.0
    energy_max: float = 1.0

    # 音频配置
    n_mels: int = 80
    max_mel_len: int = 2000

    # 其他配置
    dropout: float = 0.2


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

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


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(context)


class ConvFFN(nn.Module):
    """
    卷积前馈网络

    FastSpeech 使用卷积而非全连接层，
    以更好地捕捉局部依赖关系。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        kernel_size: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            d_model, d_ff, kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.conv2 = nn.Conv1d(
            d_ff, d_model, kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        return self.dropout(x)


class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer 块

    FastSpeech 的核心构建块，包含:
    - 多头自注意力
    - 卷积前馈网络
    - 残差连接和层归一化
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        kernel_size: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = ConvFFN(d_model, d_ff, kernel_size, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = self.attn_norm(residual + x)

        residual = x
        x = self.ffn(x)
        x = self.ffn_norm(residual + x)

        return x


class TextEncoder(nn.Module):
    """
    文本编码器

    将输入文本序列编码为隐藏表示。
    """

    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.encoder_hidden)
        self.pos_encoding = PositionalEncoding(
            config.encoder_hidden, config.max_seq_len, config.dropout
        )

        self.layers = nn.ModuleList([
            FFTBlock(
                config.encoder_hidden,
                config.encoder_heads,
                config.encoder_ff_dim,
                config.encoder_conv_kernel,
                config.dropout
            )
            for _ in range(config.encoder_layers)
        ])

    def forward(
        self,
        text: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(text)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class VariancePredictor(nn.Module):
    """
    变分预测器

    用于预测时长、音高、能量等变分信息。
    使用卷积网络捕捉局部模式。
    """

    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        self.conv1 = nn.Conv1d(
            config.encoder_hidden,
            config.variance_predictor_filter,
            config.variance_predictor_kernel,
            padding=(config.variance_predictor_kernel - 1) // 2
        )
        self.ln1 = nn.LayerNorm(config.variance_predictor_filter)
        self.dropout1 = nn.Dropout(config.variance_predictor_dropout)

        self.conv2 = nn.Conv1d(
            config.variance_predictor_filter,
            config.variance_predictor_filter,
            config.variance_predictor_kernel,
            padding=(config.variance_predictor_kernel - 1) // 2
        )
        self.ln2 = nn.LayerNorm(config.variance_predictor_filter)
        self.dropout2 = nn.Dropout(config.variance_predictor_dropout)

        self.linear = nn.Linear(config.variance_predictor_filter, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.linear(x).squeeze(-1)

        if mask is not None:
            x = x.masked_fill(mask == 0, 0.0)

        return x


class LengthRegulator(nn.Module):
    """
    长度调节器

    根据预测的时长将音素级表示扩展到帧级表示。
    这是 FastSpeech 实现并行生成的关键组件。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        duration: torch.Tensor,
        max_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入序列 [batch, seq_len, hidden]
            duration: 每个位置的时长 [batch, seq_len]
            max_len: 最大输出长度

        Returns:
            expanded: 扩展后的序列 [batch, mel_len, hidden]
            mel_lens: 每个样本的实际长度 [batch]
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # 确保 duration 非负并转为整数
        duration = torch.clamp(duration, min=0).long()

        mel_lens = duration.sum(dim=1)
        if max_len is None:
            max_len = mel_lens.max().item()
        
        # 确保 max_len 至少为 1
        max_len = max(int(max_len), 1)

        # 使用 repeat_interleave 进行向量化扩展 (更高效)
        expanded = torch.zeros(batch_size, max_len, hidden_dim, device=x.device, dtype=x.dtype)

        for i in range(batch_size):
            # 获取当前样本的有效时长
            dur = duration[i]
            total_dur = dur.sum().item()
            
            if total_dur == 0:
                continue
                
            # 使用 repeat_interleave 进行高效扩展
            expanded_seq = torch.repeat_interleave(x[i], dur, dim=0)
            
            # 截断或填充到 max_len
            actual_len = min(expanded_seq.size(0), max_len)
            expanded[i, :actual_len] = expanded_seq[:actual_len]

        return expanded, mel_lens


class VarianceAdaptor(nn.Module):
    """
    变分适配器

    FastSpeech2 的核心组件，包含:
    - 时长预测器
    - 音高预测器
    - 能量预测器
    - 长度调节器
    """

    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        self.config = config

        self.duration_predictor = VariancePredictor(config)
        self.pitch_predictor = VariancePredictor(config)
        self.energy_predictor = VariancePredictor(config)

        self.length_regulator = LengthRegulator()

        # 音高嵌入
        self.pitch_embedding = nn.Embedding(config.n_bins, config.encoder_hidden)

        # 能量嵌入
        self.energy_embedding = nn.Embedding(config.n_bins, config.encoder_hidden)

        # 音高和能量的量化边界
        self.register_buffer(
            "pitch_bins",
            torch.linspace(config.pitch_min, config.pitch_max, config.n_bins - 1)
        )
        self.register_buffer(
            "energy_bins",
            torch.linspace(config.energy_min, config.energy_max, config.n_bins - 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        duration_target: Optional[torch.Tensor] = None,
        pitch_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0
    ) -> dict:
        """
        变分适配器前向传播

        训练时使用真实的时长/音高/能量目标
        推理时使用预测值
        """
        # 时长预测
        log_duration_pred = self.duration_predictor(x, src_mask)

        if duration_target is not None:
            duration = duration_target
        else:
            # 使用 clamp 防止 exp 溢出，并确保时长非负
            clamped_log_dur = torch.clamp(log_duration_pred, max=10.0)  # exp(10) ≈ 22026
            duration = torch.clamp(
                torch.round(torch.exp(clamped_log_dur) - 1) * duration_control,
                min=0
            )

        # 音高预测
        pitch_pred = self.pitch_predictor(x, src_mask)

        if pitch_target is not None:
            pitch = pitch_target
        else:
            pitch = pitch_pred * pitch_control

        pitch_embedding = self._get_pitch_embedding(pitch)
        x = x + pitch_embedding

        # 能量预测
        energy_pred = self.energy_predictor(x, src_mask)

        if energy_target is not None:
            energy = energy_target
        else:
            energy = energy_pred * energy_control

        energy_embedding = self._get_energy_embedding(energy)
        x = x + energy_embedding

        # 长度调节
        x, mel_lens = self.length_regulator(x, duration, max_len)

        return {
            "output": x,
            "mel_lens": mel_lens,
            "log_duration_pred": log_duration_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
            "duration": duration
        }

    def _get_pitch_embedding(self, pitch: torch.Tensor) -> torch.Tensor:
        pitch_idx = torch.bucketize(pitch, self.pitch_bins)
        return self.pitch_embedding(pitch_idx)

    def _get_energy_embedding(self, energy: torch.Tensor) -> torch.Tensor:
        energy_idx = torch.bucketize(energy, self.energy_bins)
        return self.energy_embedding(energy_idx)


class MelDecoder(nn.Module):
    """
    Mel 解码器

    将帧级隐藏表示解码为 Mel 频谱图。
    """

    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        self.pos_encoding = PositionalEncoding(
            config.decoder_hidden, config.max_mel_len, config.dropout
        )

        self.layers = nn.ModuleList([
            FFTBlock(
                config.decoder_hidden,
                config.decoder_heads,
                config.decoder_ff_dim,
                config.decoder_conv_kernel,
                config.dropout
            )
            for _ in range(config.decoder_layers)
        ])

        self.mel_linear = nn.Linear(config.decoder_hidden, config.n_mels)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        mel = self.mel_linear(x)
        return mel


class PostNet(nn.Module):
    """
    后处理网络

    对 Mel 频谱进行精修，提高音质。
    """

    def __init__(self, config: FastSpeech2Config, n_layers: int = 5, kernel_size: int = 5):
        super().__init__()
        channels = 512

        layers = []
        for i in range(n_layers):
            in_ch = config.n_mels if i == 0 else channels
            out_ch = config.n_mels if i == n_layers - 1 else channels

            layers.append(nn.Conv1d(
                in_ch, out_ch, kernel_size,
                padding=(kernel_size - 1) // 2
            ))
            if i < n_layers - 1:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(config.dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(1, 2)
        residual = self.layers(mel)
        return residual.transpose(1, 2)


class FastSpeech2(nn.Module):
    """
    FastSpeech2 非自回归 TTS 模型

    主要组件:
    1. 文本编码器: 编码输入文本
    2. 变分适配器: 预测时长/音高/能量并调节长度
    3. Mel 解码器: 生成 Mel 频谱
    4. 后处理网络: 精修 Mel 频谱
    """

    def __init__(self, config: FastSpeech2Config):
        super().__init__()
        self.config = config

        self.encoder = TextEncoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = MelDecoder(config)
        self.postnet = PostNet(config)

    def forward(
        self,
        text: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        mel_mask: Optional[torch.Tensor] = None,
        duration_target: Optional[torch.Tensor] = None,
        pitch_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
        max_mel_len: Optional[int] = None
    ) -> dict:
        """
        训练前向传播

        Args:
            text: 输入文本 [batch, seq_len]
            src_mask: 源序列掩码
            mel_mask: Mel 序列掩码
            duration_target: 时长目标
            pitch_target: 音高目标
            energy_target: 能量目标
            max_mel_len: 最大 Mel 长度

        Returns:
            包含 Mel 输出和预测值的字典
        """
        encoder_output = self.encoder(text, src_mask)

        variance_output = self.variance_adaptor(
            encoder_output,
            src_mask,
            duration_target,
            pitch_target,
            energy_target,
            max_mel_len
        )

        decoder_output = self.decoder(variance_output["output"], mel_mask)
        postnet_output = decoder_output + self.postnet(decoder_output)

        return {
            "mel_output": decoder_output,
            "mel_postnet": postnet_output,
            "mel_lens": variance_output["mel_lens"],
            "log_duration_pred": variance_output["log_duration_pred"],
            "pitch_pred": variance_output["pitch_pred"],
            "energy_pred": variance_output["energy_pred"]
        }

    @torch.no_grad()
    def synthesize(
        self,
        text: torch.Tensor,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0
    ) -> torch.Tensor:
        """
        合成语音 (推理模式)

        Args:
            text: 输入文本
            duration_control: 时长控制因子 (>1 变慢, <1 变快)
            pitch_control: 音高控制因子
            energy_control: 能量控制因子

        Returns:
            Mel 频谱图
        """
        encoder_output = self.encoder(text)

        variance_output = self.variance_adaptor(
            encoder_output,
            duration_control=duration_control,
            pitch_control=pitch_control,
            energy_control=energy_control
        )

        decoder_output = self.decoder(variance_output["output"])
        mel_output = decoder_output + self.postnet(decoder_output)

        return mel_output


def fastspeech2_loss(
    mel_output: torch.Tensor,
    mel_postnet: torch.Tensor,
    mel_target: torch.Tensor,
    log_duration_pred: torch.Tensor,
    duration_target: torch.Tensor,
    pitch_pred: torch.Tensor,
    pitch_target: torch.Tensor,
    energy_pred: torch.Tensor,
    energy_target: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    mel_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, dict]:
    """
    计算 FastSpeech2 损失

    Args:
        mel_output: 解码器 Mel 输出
        mel_postnet: 后处理 Mel 输出
        mel_target: 目标 Mel 频谱
        log_duration_pred: 预测的对数时长
        duration_target: 目标时长
        pitch_pred: 预测的音高
        pitch_target: 目标音高
        energy_pred: 预测的能量
        energy_target: 目标能量
        src_mask: 源序列掩码
        mel_mask: Mel 序列掩码

    Returns:
        总损失和各项损失字典
    """
    log_duration_target = torch.log(duration_target.float() + 1)

    if src_mask is not None:
        log_duration_pred = log_duration_pred.masked_select(src_mask.bool())
        log_duration_target = log_duration_target.masked_select(src_mask.bool())
        pitch_pred = pitch_pred.masked_select(src_mask.bool())
        pitch_target = pitch_target.masked_select(src_mask.bool())
        energy_pred = energy_pred.masked_select(src_mask.bool())
        energy_target = energy_target.masked_select(src_mask.bool())

    if mel_mask is not None:
        mel_mask = mel_mask.unsqueeze(-1)
        mel_output = mel_output.masked_select(mel_mask.bool())
        mel_postnet = mel_postnet.masked_select(mel_mask.bool())
        mel_target = mel_target.masked_select(mel_mask.bool())

    mel_loss = F.mse_loss(mel_output, mel_target)
    mel_postnet_loss = F.mse_loss(mel_postnet, mel_target)
    duration_loss = F.mse_loss(log_duration_pred, log_duration_target)
    pitch_loss = F.mse_loss(pitch_pred, pitch_target)
    energy_loss = F.mse_loss(energy_pred, energy_target)

    total_loss = mel_loss + mel_postnet_loss + duration_loss + pitch_loss + energy_loss

    return total_loss, {
        "mel_loss": mel_loss,
        "mel_postnet_loss": mel_postnet_loss,
        "duration_loss": duration_loss,
        "pitch_loss": pitch_loss,
        "energy_loss": energy_loss
    }


def create_fastspeech2_model(size: str = "base") -> FastSpeech2:
    """
    创建预定义大小的 FastSpeech2 模型

    Args:
        size: 模型大小 ("tiny", "base", "large")

    Returns:
        FastSpeech2 模型实例
    """
    configs = {
        "tiny": FastSpeech2Config(
            encoder_hidden=128,
            encoder_layers=2,
            encoder_heads=2,
            encoder_ff_dim=512,
            decoder_hidden=128,
            decoder_layers=2,
            decoder_heads=2,
            decoder_ff_dim=512,
            variance_predictor_filter=128,
        ),
        "base": FastSpeech2Config(
            encoder_hidden=256,
            encoder_layers=4,
            encoder_heads=2,
            encoder_ff_dim=1024,
            decoder_hidden=256,
            decoder_layers=4,
            decoder_heads=2,
            decoder_ff_dim=1024,
            variance_predictor_filter=256,
        ),
        "large": FastSpeech2Config(
            encoder_hidden=512,
            encoder_layers=6,
            encoder_heads=4,
            encoder_ff_dim=2048,
            decoder_hidden=512,
            decoder_layers=6,
            decoder_heads=4,
            decoder_ff_dim=2048,
            variance_predictor_filter=512,
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")

    return FastSpeech2(configs[size])
