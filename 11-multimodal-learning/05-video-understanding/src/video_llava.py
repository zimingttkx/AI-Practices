"""
Video-LLaVA: 视频理解多模态模型

扩展 LLaVA 处理视频时序信息，支持视频问答、描述生成等任务。

架构:
    视频 → 帧采样 → ViT编码 → 时序建模 → 投影 → LLaMA → 输出

参考:
    - Video-LLaVA (Lin et al., 2024): https://arxiv.org/abs/2312.00731
    - LLaVA (Liu et al., 2023): https://arxiv.org/abs/2304.08485
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplingStrategy(Enum):
    """视频帧采样策略"""
    UNIFORM = "uniform"          # 均匀采样
    RANDOM = "random"            # 随机采样
    KEYFRAME = "keyframe"        # 关键帧采样（基于场景变化）
    ADAPTIVE = "adaptive"        # 自适应采样


@dataclass
class VideoLLaVAConfig:
    """Video-LLaVA 模型配置

    参数：
        image_size: 单帧图像大小 (默认224)
        patch_size: 图像分块大小 (默认14)
        vision_layers: 视觉编码器层数
        vision_width: 视觉编码器隐藏层维度
        vision_heads: 视觉编码器注意力头数
        vocab_size: 词汇表大小
        max_seq_length: 最大序列长度
        hidden_size: LLM 隐藏层维度
        num_layers: LLM 层数
        num_heads: LLM 注意力头数
        intermediate_size: FFN 中间层维度
        num_frames: 采样帧数 (默认8帧)
        temporal_mode: 时序建模模式 ("pooling", "transformer", "lstm")
        temporal_layers: 时序 Transformer 层数
        temporal_heads: 时序 Transformer 注意力头数
        projector_type: 投影层类型 ("linear" 或 "mlp2x_gelu")
        dropout: Dropout 概率
    """

    # 视觉编码器配置
    image_size: int = 224
    patch_size: int = 14
    vision_layers: int = 24
    vision_width: int = 1024
    vision_heads: int = 16

    # 语言模型配置
    vocab_size: int = 32000
    max_seq_length: int = 2048
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    intermediate_size: int = 11008

    # 视频配置
    num_frames: int = 8  # 采样帧数

    # 时序建模配置
    temporal_mode: str = "transformer"  # "pooling", "transformer", "lstm"
    temporal_layers: int = 4
    temporal_heads: int = 8
    temporal_hidden_size: int = 1024

    # 投影层配置
    projector_type: str = "mlp2x_gelu"

    # 共享配置
    dropout: float = 0.0

    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
        valid_modes = ["pooling", "transformer", "lstm"]
        assert self.temporal_mode in valid_modes, \
            f"temporal_mode must be one of {valid_modes}, got {self.temporal_mode}"


class PatchEmbedding(nn.Module):
    """图像分块嵌入"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=config.vision_width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_width))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.vision_width)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量 [batch_size, 3, height, width]
        Returns:
            分块嵌入 [batch_size, num_patches + 1, vision_width]
        """
        batch_size = x.shape[0]

        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.position_embedding

        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


class MLP(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate='tanh')
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 块"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        causal: bool = False
    ):
        super().__init__()
        self.causal = causal

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask, causal=self.causal)
        x = x + self.mlp(self.ln2(x))
        return x


class VisionEncoder(nn.Module):
    """CLIP 风格视觉编码器"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(config)
        self.ln_pre = nn.LayerNorm(config.vision_width)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.vision_width,
                num_heads=config.vision_heads,
                mlp_ratio=4.0,
                dropout=config.dropout,
                causal=False
            )
            for _ in range(config.vision_layers)
        ])

        self.ln_post = nn.LayerNorm(config.vision_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量 [batch_size, 3, height, width]
        Returns:
            视觉特征 [batch_size, num_patches + 1, vision_width]
        """
        x = self.patch_embed(x)
        x = self.ln_pre(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TemporalPositionalEncoding(nn.Module):
    """时序位置编码 - 标记帧的时序关系"""

    def __init__(self, d_model: int, max_frames: int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_frames = max_frames

        # 创建时序位置编码
        position = torch.arange(max_frames).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_frames, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 时序特征 [batch_size, num_frames, d_model]
        Returns:
            添加位置编码后的特征
        """
        num_frames = x.size(1)
        return x + self.pe[:, :num_frames, :]


class TemporalTransformer(nn.Module):
    """时序 Transformer - 跨帧注意力"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.config = config

        # 时序位置编码
        self.temporal_pe = TemporalPositionalEncoding(
            config.temporal_hidden_size,
            max_frames=config.num_frames
        )

        # 时序 Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.temporal_hidden_size,
            nhead=config.temporal_heads,
            dim_feedforward=config.temporal_hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.temporal_layers
        )

        self.norm = nn.LayerNorm(config.temporal_hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 帧级特征 [batch_size, num_frames, d_model]
            attention_mask: 注意力掩码 [batch_size, num_frames]
        Returns:
            时序聚合后的特征 [batch_size, num_frames, d_model]
        """
        # 添加时序位置编码
        x = self.temporal_pe(x)

        # Transformer 编码
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)

        x = self.norm(x)

        return x


class TemporalLSTM(nn.Module):
    """时序 LSTM - 替代方案"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.temporal_hidden_size,
            hidden_size=config.temporal_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout if config.temporal_layers > 1 else 0.0,
            bidirectional=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 帧级特征 [batch_size, num_frames, d_model]
        Returns:
            LSTM 处理后的特征 [batch_size, num_frames, d_model]
        """
        output, _ = self.lstm(x)
        return output


class TemporalPooling(nn.Module):
    """时序池化 - 简单的帧特征聚合"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.pooling_type = "mean"  # 可选: "mean", "max", "attention"

        if self.pooling_type == "attention":
            self.attention_weights = nn.Sequential(
                nn.Linear(config.vision_width, config.vision_width // 2),
                nn.Tanh(),
                nn.Linear(config.vision_width // 2, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 帧级特征 [batch_size, num_frames, num_patches, vision_width]
        Returns:
            聚合后的特征 [batch_size, num_patches, vision_width]
        """
        if self.pooling_type == "mean":
            return x.mean(dim=1)
        elif self.pooling_type == "max":
            return x.max(dim=1)[0]
        elif self.pooling_type == "attention":
            # 注意力加权池化
            batch_size, num_frames, num_patches, vision_width = x.shape
            x_flat = x.permute(0, 2, 1, 3)
            x_flat = x_flat.reshape(batch_size * num_patches, num_frames, vision_width)

            attn_scores = self.attention_weights(x_flat).squeeze(-1)  # [B*P, T]
            attn_weights = F.softmax(attn_scores, dim=1)

            pooled = (x_flat * attn_weights.unsqueeze(-1)).sum(dim=1)
            return pooled.view(batch_size, num_patches, vision_width)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class VideoProjector(nn.Module):
    """视频特征投影层 - 将视觉特征映射到 LLM 空间"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.config = config

        if config.projector_type == "linear":
            self.projector = nn.Linear(config.vision_width, config.hidden_size)
        elif config.projector_type == "mlp2x_gelu":
            self.projector = nn.Sequential(
                nn.Linear(config.vision_width, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 视频特征 [batch_size, num_tokens, vision_width]
        Returns:
            投影后的特征 [batch_size, num_tokens, hidden_size]
        """
        return self.projector(x)


class RMSNorm(nn.Module):
    """RMS 归一化 (LLaMA 风格)"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转一半的维度"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LLaMAAttention(nn.Module):
    """LLaMA 风格的多头注意力"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_seq_length)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 计算注意力
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)


class LLaMAMLP(nn.Module):
    """LLaMA 风格的 MLP (SwiGLU)"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaMADecoderLayer(nn.Module):
    """LLaMA 解码器层"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.self_attn = LLaMAAttention(config)
        self.mlp = LLaMAMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class LLaMAModel(nn.Module):
    """简化的 LLaMA 语言模型"""

    def __init__(self, config: VideoLLaVAConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LLaMADecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


class VideoProcessor:
    """视频预处理工具"""

    @staticmethod
    def sample_frames(
        video: torch.Tensor,
        num_frames: int,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    ) -> torch.Tensor:
        """
        从视频采样帧

        Args:
            video: 视频张量 [T, 3, H, W] 或 [B, T, 3, H, W]
            num_frames: 采样帧数
            strategy: 采样策略

        Returns:
            采样后的帧 [num_frames, 3, H, W] 或 [B, num_frames, 3, H, W]
        """
        squeeze_batch = False
        if video.dim() == 4:
            video = video.unsqueeze(0)
            squeeze_batch = True

        batch_size, total_frames, c, h, w = video.shape

        if total_frames <= num_frames:
            # 如果帧数不足，重复最后一帧
            indices = torch.cat([
                torch.arange(total_frames),
                torch.full((num_frames - total_frames,), total_frames - 1)
            ])
        else:
            if strategy == SamplingStrategy.UNIFORM:
                indices = torch.linspace(0, total_frames - 1, num_frames).long()
            elif strategy == SamplingStrategy.RANDOM:
                indices = torch.randperm(total_frames)[:num_frames].sort()[0]
            else:
                # 默认使用均匀采样
                indices = torch.linspace(0, total_frames - 1, num_frames).long()

        sampled = video[:, indices, :, :, :]

        if squeeze_batch:
            sampled = sampled.squeeze(0)

        return sampled

    @staticmethod
    def resize_frames(frames: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """
        调整帧大小

        Args:
            frames: 帧张量 [T, 3, H, W] 或 [B, T, 3, H, W]
            size: 目标大小 (height, width)

        Returns:
            调整大小后的帧
        """
        squeeze_batch = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_batch = True

        batch_size, num_frames, c, h, w = frames.shape
        frames_flat = frames.view(batch_size * num_frames, c, h, w)

        # 使用插值调整大小
        resized = F.interpolate(
            frames_flat,
            size=size,
            mode='bilinear',
            align_corners=False
        )

        resized = resized.view(batch_size, num_frames, c, size[0], size[1])

        if squeeze_batch:
            resized = resized.squeeze(0)

        return resized


class VideoLLaVA(nn.Module):
    """Video-LLaVA 视频理解模型

    扩展 LLaVA 以处理视频时序信息。

    架构流程：
        视频 → 帧采样 → VisionEncoder → TemporalModel → VideoProjector → [视频tokens]
                                                                              ↓
        文本 → TokenEmbedding → [文本tokens] → 拼接 → LLaMA → 输出

    示例：
        >>> config = VideoLLaVAConfig(num_frames=8)
        >>> model = VideoLLaVA(config)
        >>> videos = torch.randn(2, 16, 3, 224, 224)  # 16帧
        >>> input_ids = torch.randint(0, 32000, (2, 50))
        >>> output = model(input_ids, videos)
        >>> logits = output["logits"]
    """

    def __init__(self, config: VideoLLaVAConfig):
        """初始化 Video-LLaVA 模型

        参数：
            config: VideoLLaVA 配置
        """
        super().__init__()
        self.config = config

        # 视觉编码器
        self.vision_encoder = VisionEncoder(config)

        # 时序建模
        if config.temporal_mode == "transformer":
            self.temporal_model = TemporalTransformer(config)
        elif config.temporal_mode == "lstm":
            self.temporal_model = TemporalLSTM(config)
        else:  # pooling
            self.temporal_model = TemporalPooling(config)

        # 视频投影层
        self.video_projector = VideoProjector(config)

        # 语言模型
        self.language_model = LLaMAModel(config)

        # 视频处理器
        self.video_processor = VideoProcessor()

        # 特殊 token ID
        self.video_token_id = -200  # 占位符

    def encode_video(
        self,
        video: torch.Tensor,
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    ) -> torch.Tensor:
        """
        编码视频为时序特征

        Args:
            video: 视频张量 [batch_size, total_frames, 3, H, W]
            sampling_strategy: 帧采样策略

        Returns:
            视频特征 [batch_size, num_tokens, hidden_size]
        """
        batch_size = video.shape[0]

        # 采样帧
        sampled_frames = self.video_processor.sample_frames(
            video,
            self.config.num_frames,
            sampling_strategy
        )  # [B, T, 3, H, W]

        # 调整大小
        target_size = (self.config.image_size, self.config.image_size)
        resized_frames = self.video_processor.resize_frames(sampled_frames, target_size)

        # 展平为 [B*T, 3, H, W]
        num_frames = resized_frames.shape[1]
        h, w = target_size
        frames_flat = resized_frames.view(batch_size * num_frames, 3, h, w)

        # 视觉编码
        frame_features = self.vision_encoder(frames_flat)
        # [B*T, num_patches+1, vision_width]

        # 提取 CLS token (全局帧特征)
        cls_features = frame_features[:, 0:1, :]  # [B*T, 1, vision_width]
        patch_features = frame_features[:, 1:, :]  # [B*T, num_patches, vision_width]

        # 时序建模
        if self.config.temporal_mode == "pooling":
            # 对于池化模式，需要将 patch 维度展开
            patch_features = patch_features.view(
                batch_size, num_frames, -1, self.config.vision_width
            )  # [B, T, P, D]
            aggregated_patches = self.temporal_model(patch_features)
            # [B, P, D]

            # 拼接 CLS 特征
            cls_features = cls_features.view(batch_size, num_frames, self.config.vision_width)
            aggregated_cls = cls_features.mean(dim=1, keepdim=True)  # [B, 1, D]

            video_features = torch.cat([aggregated_cls, aggregated_patches], dim=1)
        else:
            # Transformer/LSTM 模式
            cls_features = cls_features.view(batch_size, num_frames, self.config.vision_width)

            if self.config.temporal_mode == "transformer":
                temporal_features = self.temporal_model(cls_features)
            else:  # lstm
                temporal_features = self.temporal_model(cls_features)

            # 使用第一帧的 patch 特征
            first_frame_patches = patch_features[::num_frames]  # [B, num_patches, D]
            video_features = torch.cat([temporal_features, first_frame_patches], dim=1)

        # 投影到语言空间
        video_tokens = self.video_projector(video_features)

        return video_tokens

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 文本 token IDs [batch_size, text_seq_len]
            videos: 视频张量 [batch_size, total_frames, 3, H, W]
            attention_mask: 注意力掩码 [batch_size, total_seq_len]
            labels: 标签 [batch_size, text_seq_len]

        Returns:
            包含 logits 和 loss 的字典
        """
        # 编码视频
        if videos is not None:
            video_tokens = self.encode_video(videos)
            num_video_tokens = video_tokens.shape[1]
        else:
            video_tokens = None
            num_video_tokens = 0

        # 准备文本嵌入
        if input_ids is not None:
            text_embeds = self.language_model.embed_tokens(input_ids)
        else:
            text_embeds = None

        # 拼接视频和文本嵌入
        if video_tokens is not None and text_embeds is not None:
            inputs_embeds = torch.cat([video_tokens, text_embeds], dim=1)

            # 更新注意力掩码
            if attention_mask is not None:
                batch_size = video_tokens.shape[0]
                video_mask = torch.ones(
                    batch_size,
                    num_video_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([video_mask, attention_mask], dim=1)
        elif video_tokens is not None:
            inputs_embeds = video_tokens
        else:
            inputs_embeds = text_embeds

        # 语言模型生成 logits
        logits = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        # 计算损失
        loss = None
        if labels is not None:
            # Shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {
            "logits": logits,
            "loss": loss,
            "video_tokens": video_tokens
        }

    def generate(
        self,
        videos: torch.Tensor,
        prompt: str = "",
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True
    ) -> str:
        """
        生成响应

        Args:
            videos: 视频张量 [batch_size, total_frames, 3, H, W]
            prompt: 文本提示
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            do_sample: 是否采样

        Returns:
            生成的文本
        """
        self.eval()
        with torch.no_grad():
            # 编码视频
            video_tokens = self.encode_video(videos)

            # TODO: 实现完整的生成逻辑
            # 这里需要实现 tokenization 和 autoregressive generation

            return "Video understanding output (placeholder)"


def create_video_llava(
    model_size: str = "base",
    num_frames: int = 8,
    temporal_mode: str = "transformer"
) -> VideoLLaVA:
    """
    创建预定义大小的 Video-LLaVA 模型

    参数：
        model_size: 模型大小 ("tiny", "base", "large")
        num_frames: 采样帧数
        temporal_mode: 时序建模模式

    返回：
        Video-LLaVA 模型
    """
    if model_size == "tiny":
        config = VideoLLaVAConfig(
            image_size=224,
            patch_size=16,
            vision_layers=6,
            vision_width=384,
            vision_heads=6,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=1024,
            num_frames=num_frames,
            temporal_mode=temporal_mode,
            temporal_layers=2,
            temporal_heads=4,
            temporal_hidden_size=384
        )
    elif model_size == "base":
        config = VideoLLaVAConfig(
            num_frames=num_frames,
            temporal_mode=temporal_mode
        )
    elif model_size == "large":
        config = VideoLLaVAConfig(
            image_size=336,
            vision_layers=32,
            vision_width=1280,
            vision_heads=16,
            hidden_size=5120,
            num_layers=40,
            num_heads=40,
            intermediate_size=13824,
            num_frames=num_frames,
            temporal_mode=temporal_mode,
            temporal_layers=6,
            temporal_heads=16,
            temporal_hidden_size=1280
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    return VideoLLaVA(config)
