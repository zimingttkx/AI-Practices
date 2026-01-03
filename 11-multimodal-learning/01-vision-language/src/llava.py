"""
LLaVA (Large Language and Vision Assistant) 实现

LLaVA 是一个多模态对话模型，通过视觉投影层将视觉编码器与大语言模型连接。

=== 核心思想 ===

LLaVA 采用简洁高效的架构设计，将预训练的视觉编码器与大语言模型连接：

1. 视觉编码器 (Vision Encoder)
   - 使用 CLIP 预训练的 ViT 提取图像特征
   - 保持视觉编码器参数冻结或微调
   - 输出 patch-level 的视觉表示

2. 视觉投影层 (Vision Projector)
   - 将视觉特征映射到语言模型的嵌入空间
   - 支持线性投影或 MLP 投影
   - 是连接视觉和语言的关键桥梁

3. 大语言模型 (LLM)
   - 使用 LLaMA 架构的自回归语言模型
   - 接收视觉 token 和文本 token 的混合序列
   - 生成多模态对话响应

=== 数学基础 ===

视觉特征投影:
    H_v = Projector(VisionEncoder(I))
    
    其中:
    - I: 输入图像 [B, 3, H, W]
    - VisionEncoder: CLIP ViT 编码器
    - Projector: 线性层或 MLP
    - H_v: 视觉 token [B, N_patches, D_llm]

旋转位置编码 (RoPE):
    q' = q * cos(mθ) + rotate(q) * sin(mθ)
    k' = k * cos(mθ) + rotate(k) * sin(mθ)
    
    其中:
    - m: 位置索引
    - θ_i = 10000^(-2i/d): 频率
    - rotate: 将向量分成两半并旋转

RMSNorm 归一化:
    RMSNorm(x) = x / RMS(x) * γ
    RMS(x) = √(1/n * Σ x_i²)

SwiGLU 激活:
    SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)
    Swish(x) = x * sigmoid(x)

自回归语言建模损失:
    L = -Σ log P(x_t | x_{<t}, H_v)

=== 算法流程 ===

训练阶段:
    输入: 图像 I, 文本指令 T, 目标响应 R
      ↓
    视觉编码: H_v = Projector(VisionEncoder(I))  # [B, N, D]
    文本嵌入: H_t = Embed(T)                      # [B, L, D]
      ↓
    拼接序列: H = [H_v; H_t]                      # [B, N+L, D]
      ↓
    语言模型: logits = LLM(H)
      ↓
    计算损失: L = CrossEntropy(logits, R)
      ↓
    反向传播更新参数

推理阶段 (多模态对话):
    输入: 图像 I, 用户问题 Q
      ↓
    视觉编码: H_v = Projector(VisionEncoder(I))
    问题嵌入: H_q = Embed(Q)
      ↓
    拼接序列: H = [H_v; H_q]
      ↓
    自回归生成: A = LLM.generate(H)
      ↓
    输出: 模型回答 A

=== 参考文献 ===

1. LLaVA 原始论文:
   Liu et al. "Visual Instruction Tuning" NeurIPS 2023
   https://arxiv.org/abs/2304.08485

2. LLaVA-1.5:
   Liu et al. "Improved Baselines with Visual Instruction Tuning" 2023
   https://arxiv.org/abs/2310.03744

3. LLaMA:
   Touvron et al. "LLaMA: Open and Efficient Foundation Language Models" 2023

4. RoPE 位置编码:
   Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" 2021

=== 核心组件 ===

    - LLaVAConfig: LLaVA 模型配置
    - PatchEmbedding: 图像分块嵌入
    - VisionEncoder: CLIP 风格视觉编码器
    - VisionProjector: 视觉特征投影层
    - RMSNorm: RMS 归一化层
    - RotaryEmbedding: 旋转位置编码
    - LLaMAAttention: LLaMA 风格多头注意力 (带 RoPE)
    - LLaMAMLP: LLaMA 风格 MLP (SwiGLU)
    - LLaMADecoderLayer: LLaMA 解码器层
    - LLaMAModel: 简化的 LLaMA 语言模型
    - LLaVA: 完整的多模态对话模型
    - create_llava_model: 创建预定义大小的 LLaVA 模型
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLaVAConfig:
    """LLaVA 模型配置。

    参数：
        image_size: 输入图像大小 (默认224)
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

    # 投影层配置
    projector_type: str = "mlp2x_gelu"  # linear, mlp2x_gelu

    # 共享配置
    dropout: float = 0.0

    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"


class PatchEmbedding(nn.Module):
    """图像分块嵌入"""

    def __init__(self, config: LLaVAConfig):
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

    def __init__(self, config: LLaVAConfig):
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


class VisionProjector(nn.Module):
    """视觉特征投影层 - 将视觉特征映射到 LLM 空间。

    这是连接视觉编码器和语言模型的关键桥梁。

    支持的投影类型：
        - linear: 单层线性投影
        - mlp2x_gelu: 两层 MLP + GELU 激活
    """

    def __init__(self, config: LLaVAConfig):
        """初始化视觉投影层。

        参数：
            config: LLaVA 配置
        """
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
            x: 视觉特征 [batch_size, num_patches, vision_width]
        Returns:
            投影后的特征 [batch_size, num_patches, hidden_size]
        """
        return self.projector(x)


class RMSNorm(nn.Module):
    """RMS 归一化 (LLaMA 风格)。

    数学原理：
        RMSNorm(x) = x / RMS(x) * γ
        RMS(x) = √(1/n * Σ x_i²)

    相比 LayerNorm，RMSNorm 不计算均值，计算效率更高。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """初始化 RMS 归一化层。

        参数：
            hidden_size: 隐藏层维度
            eps: 数值稳定性的小常数
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)。

    数学原理：
        q' = q * cos(mθ) + rotate(q) * sin(mθ)
        k' = k * cos(mθ) + rotate(k) * sin(mθ)

    优势：
        - 相对位置编码，支持外推到更长序列
        - 计算高效，无需额外参数
        - 保持注意力分数的平移不变性
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        """初始化旋转位置编码。

        参数：
            dim: 编码维度
            max_seq_len: 最大序列长度
            base: 频率基数
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转一半的维度"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
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

    def __init__(self, config: LLaVAConfig):
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
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
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
    """LLaMA 风格的 MLP (SwiGLU)。

    数学原理：
        SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)
        output = SwiGLU(x) @ W_down

    相比标准 FFN，SwiGLU 具有更好的性能。
    """

    def __init__(self, config: LLaVAConfig):
        """初始化 LLaMA MLP。

        参数：
            config: LLaVA 配置
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LLaMADecoderLayer(nn.Module):
    """LLaMA 解码器层"""

    def __init__(self, config: LLaVAConfig):
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

    def __init__(self, config: LLaVAConfig):
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

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class LLaVA(nn.Module):
    """LLaVA 多模态对话模型。

    将视觉编码器与大语言模型连接，实现多模态理解和生成。

    架构：
        图像 → VisionEncoder → VisionProjector → [视觉tokens]
                                                      ↓
        文本 → TokenEmbedding → [文本tokens] → 拼接 → LLaMA → 输出

    示例：
        >>> config = LLaVAConfig()
        >>> model = LLaVA(config)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> input_ids = torch.randint(0, 32000, (2, 20))
        >>> output = model(input_ids, images)
        >>> logits = output["logits"]
    """

    def __init__(self, config: LLaVAConfig):
        """初始化 LLaVA 模型。

        参数：
            config: LLaVA 配置
        """
        super().__init__()
        self.config = config

        # 视觉编码器
        self.vision_encoder = VisionEncoder(config)

        # 视觉投影层
        self.vision_projector = VisionProjector(config)

        # 语言模型
        self.language_model = LLaMAModel(config)

        # 特殊 token ID
        self.image_token_id = -200  # 占位符

    def get_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        获取视觉特征
        Args:
            images: [batch_size, 3, H, W]
        Returns:
            视觉特征 [batch_size, num_patches, hidden_size]
        """
        vision_outputs = self.vision_encoder(images)
        # 移除 CLS token，只保留 patch tokens
        vision_features = vision_outputs[:, 1:, :]
        vision_features = self.vision_projector(vision_features)
        return vision_features

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        准备输入嵌入，将图像特征插入到文本嵌入中
        Args:
            input_ids: [batch_size, seq_len]
            images: [batch_size, 3, H, W] 或 None
            image_positions: 图像 token 的位置 [batch_size] 或 None
        Returns:
            inputs_embeds: [batch_size, total_seq_len, hidden_size]
        """
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        if images is None:
            return text_embeds

        batch_size = input_ids.shape[0]
        vision_features = self.get_vision_features(images)
        num_image_tokens = vision_features.shape[1]

        # 简化实现: 将图像特征插入到序列开头
        if image_positions is None:
            # 默认在开头插入
            inputs_embeds = torch.cat([vision_features, text_embeds], dim=1)
        else:
            # 根据指定位置插入
            inputs_embeds_list = []
            for i in range(batch_size):
                pos = image_positions[i].item()
                before = text_embeds[i, :pos, :]
                after = text_embeds[i, pos:, :]
                combined = torch.cat([before, vision_features[i], after], dim=0)
                inputs_embeds_list.append(combined)
            inputs_embeds = torch.stack(inputs_embeds_list)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            input_ids: token IDs [batch_size, seq_len]
            images: 图像张量 [batch_size, 3, H, W]
            attention_mask: 注意力掩码
            image_positions: 图像插入位置
            labels: 标签 (用于计算损失)
        Returns:
            包含 logits 和可选 loss 的字典
        """
        inputs_embeds = self.prepare_inputs_embeds(input_ids, images, image_positions)

        # 调整 attention_mask 以适应新的序列长度
        if attention_mask is not None and images is not None:
            num_image_tokens = (self.config.image_size // self.config.patch_size) ** 2
            image_mask = torch.ones(
                attention_mask.shape[0], num_image_tokens,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        logits = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        output = {"logits": logits}

        if labels is not None:
            # 移位标签计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output["loss"] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int = 2
    ) -> torch.Tensor:
        """
        自回归生成
        Args:
            input_ids: 输入 token IDs
            images: 图像张量
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus 采样参数
            eos_token_id: 结束 token ID
        Returns:
            生成的 token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 准备初始输入
        inputs_embeds = self.prepare_inputs_embeds(input_ids, images)
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.language_model(inputs_embeds=inputs_embeds)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-p 采样
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for batch_idx in range(batch_size):
                indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                next_token_logits[batch_idx, indices_to_remove] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_tokens], dim=-1)

            # 检查是否生成了 EOS
            if (next_tokens == eos_token_id).all():
                break

            # 更新 inputs_embeds
            next_embeds = self.language_model.get_input_embeddings()(next_tokens)
            inputs_embeds = torch.cat([inputs_embeds, next_embeds], dim=1)

        return generated


def create_llava_model(model_size: str = "small") -> LLaVA:
    """
    创建预定义大小的 LLaVA 模型

    Args:
        model_size: 模型大小 ("tiny", "small", "base")

    Returns:
        LLaVA 模型实例
    """
    configs = {
        "tiny": LLaVAConfig(
            image_size=224, patch_size=14,
            vision_layers=6, vision_width=384, vision_heads=6,
            vocab_size=32000, max_seq_length=512,
            hidden_size=512, num_layers=4, num_heads=8,
            intermediate_size=1376,
            projector_type="mlp2x_gelu"
        ),
        "small": LLaVAConfig(
            image_size=224, patch_size=14,
            vision_layers=12, vision_width=768, vision_heads=12,
            vocab_size=32000, max_seq_length=1024,
            hidden_size=1024, num_layers=8, num_heads=16,
            intermediate_size=2752,
            projector_type="mlp2x_gelu"
        ),
        "base": LLaVAConfig(
            image_size=224, patch_size=14,
            vision_layers=24, vision_width=1024, vision_heads=16,
            vocab_size=32000, max_seq_length=2048,
            hidden_size=2048, num_layers=16, num_heads=32,
            intermediate_size=5504,
            projector_type="mlp2x_gelu"
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return LLaVA(configs[model_size])
