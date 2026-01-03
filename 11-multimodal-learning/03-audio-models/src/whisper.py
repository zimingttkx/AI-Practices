"""
Whisper 语音识别模型 (Whisper Speech Recognition)

本模块实现 OpenAI Whisper 风格的语音识别模型。

=== 核心架构 ===

Whisper 是一个编码器-解码器 Transformer 模型：

1. 音频编码器 (Audio Encoder)
   - 输入: Log-Mel 频谱图 [batch, n_mels, n_frames]
   - 两层卷积进行下采样
   - Transformer 编码器层
   - 输出: 音频特征 [batch, n_frames/2, d_model]

2. 文本解码器 (Text Decoder)
   - 自回归 Transformer 解码器
   - 带交叉注意力连接编码器输出
   - 输出: 文本 token 概率

=== 多任务能力 ===

Whisper 支持多种任务，通过特殊 token 控制：
- <|transcribe|>: 语音转录
- <|translate|>: 语音翻译 (转为英语)
- <|language|>: 语言检测
- <|notimestamps|>: 不输出时间戳

=== 参考文献 ===

Radford et al. "Robust Speech Recognition via Large-Scale Weak Supervision" 2022
https://arxiv.org/abs/2212.04356
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WhisperConfig:
    """Whisper 模型配置"""

    # 音频配置
    n_mels: int = 80
    n_audio_ctx: int = 1500  # 最大音频上下文长度 (30秒 @ 16kHz)

    # 文本配置
    vocab_size: int = 51865
    n_text_ctx: int = 448  # 最大文本上下文长度

    # 模型配置
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.0

    # 特殊 token
    sot_token: int = 50258  # Start of transcript
    eot_token: int = 50257  # End of transcript
    transcribe_token: int = 50359
    translate_token: int = 50358
    no_timestamps_token: int = 50363


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.is_causal = is_causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 重塑为多头
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # 应用掩码
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        # 因果掩码
        if self.is_causal:
            seq_len = query.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device), diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, v)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """编码器层"""

    def __init__(self, config: WhisperConfig):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.self_attn_norm = nn.LayerNorm(config.d_model)

        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, x, x)
        x = residual + x

        # 前馈网络
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class DecoderLayer(nn.Module):
    """解码器层"""

    def __init__(self, config: WhisperConfig):
        super().__init__()

        # 自注意力 (因果)
        self.self_attn = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout, is_causal=True
        )
        self.self_attn_norm = nn.LayerNorm(config.d_model)

        # 交叉注意力
        self.cross_attn = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.cross_attn_norm = nn.LayerNorm(config.d_model)

        # 前馈网络
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 自注意力
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, x, x)
        x = residual + x

        # 交叉注意力
        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, encoder_output, encoder_output, encoder_mask)
        x = residual + x

        # 前馈网络
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class AudioEncoder(nn.Module):
    """Whisper 音频编码器"""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config

        # 卷积下采样
        self.conv1 = nn.Conv1d(config.n_mels, config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)

        # 位置编码
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.n_audio_ctx
        )

        # Transformer 编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_encoder_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Log-Mel 频谱图 [batch, n_mels, n_frames]
        Returns:
            编码器输出 [batch, n_frames/2, d_model]
        """
        # 卷积下采样
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))

        # 转换维度 [batch, d_model, time] -> [batch, time, d_model]
        x = x.transpose(1, 2)

        # 位置编码
        x = self.positional_encoding(x)

        # Transformer 层
        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)
        return x


class TextDecoder(nn.Module):
    """Whisper 文本解码器"""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config

        # Token 嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 位置编码
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.n_text_ctx
        )

        # Transformer 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_decoder_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: 输入 token [batch, seq_len]
            encoder_output: 编码器输出 [batch, audio_len, d_model]
        Returns:
            解码器输出 [batch, seq_len, d_model]
        """
        # Token 嵌入 + 位置编码
        x = self.token_embedding(tokens)
        x = self.positional_encoding(x)

        # Transformer 层
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask)

        x = self.layer_norm(x)
        return x


class Whisper(nn.Module):
    """Whisper 语音识别模型"""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config

        self.encoder = AudioEncoder(config)
        self.decoder = TextDecoder(config)

        # 输出投影 (共享嵌入权重)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        训练前向传播
        Args:
            mel: Log-Mel 频谱图 [batch, n_mels, n_frames]
            tokens: 目标 token [batch, seq_len]
        Returns:
            logits [batch, seq_len, vocab_size]
        """
        encoder_output = self.encoder(mel)
        decoder_output = self.decoder(tokens, encoder_output)
        logits = self.proj_out(decoder_output)
        return logits

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """编码音频"""
        return self.encoder(mel)

    def decode(
        self,
        tokens: torch.Tensor,
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """解码一步"""
        decoder_output = self.decoder(tokens, encoder_output)
        logits = self.proj_out(decoder_output)
        return logits

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        max_length: int = 224,
        temperature: float = 0.0,
        task: str = "transcribe"
    ) -> torch.Tensor:
        """
        自回归生成文本
        Args:
            mel: Log-Mel 频谱图
            max_length: 最大生成长度
            temperature: 采样温度 (0 = greedy)
            task: 任务类型 ("transcribe" 或 "translate")
        Returns:
            生成的 token 序列
        """
        batch_size = mel.size(0)
        device = mel.device

        # 编码音频
        encoder_output = self.encode(mel)

        # 初始化 token 序列
        task_token = self.config.transcribe_token if task == "transcribe" else self.config.translate_token
        tokens = torch.tensor([[self.config.sot_token, task_token, self.config.no_timestamps_token]], device=device)
        tokens = tokens.expand(batch_size, -1)

        # 自回归生成
        for _ in range(max_length):
            logits = self.decode(tokens, encoder_output)
            next_token_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            # 检查是否生成了结束 token
            if (next_token == self.config.eot_token).all():
                break

        return tokens


def create_whisper_model(size: str = "base") -> Whisper:
    """
    创建预定义大小的 Whisper 模型
    Args:
        size: 模型大小 ("tiny", "base", "small", "medium")
    Returns:
        Whisper 模型实例
    """
    configs = {
        "tiny": WhisperConfig(
            d_model=384,
            n_heads=6,
            n_encoder_layers=4,
            n_decoder_layers=4,
            d_ff=1536
        ),
        "base": WhisperConfig(
            d_model=512,
            n_heads=8,
            n_encoder_layers=6,
            n_decoder_layers=6,
            d_ff=2048
        ),
        "small": WhisperConfig(
            d_model=768,
            n_heads=12,
            n_encoder_layers=12,
            n_decoder_layers=12,
            d_ff=3072
        ),
        "medium": WhisperConfig(
            d_model=1024,
            n_heads=16,
            n_encoder_layers=24,
            n_decoder_layers=24,
            d_ff=4096
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")

    return Whisper(configs[size])
