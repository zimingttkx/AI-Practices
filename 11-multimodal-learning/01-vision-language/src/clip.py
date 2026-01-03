"""
CLIP (Contrastive Language-Image Pre-training) 实现

CLIP 是 OpenAI 提出的多模态模型，通过对比学习将图像和文本映射到共享的嵌入空间。

=== 核心思想 ===

CLIP 通过在大规模图像-文本对上进行对比学习，学习将图像和文本映射到统一的嵌入空间。
在嵌入空间中，匹配的图像-文本对的距离更近，不匹配的距离更远。

核心特性:
    1. 对比学习: 使用 InfoNCE 损失函数拉近正样本对，推远负样本对
    2. 双塔架构: 图像编码器和文本编码器独立编码，便于推理
    3. 零样本迁移: 通过自然语言描述实现零样本分类

=== 数学基础 ===

InfoNCE 对比损失函数:
    L = -1/N * Σ log exp(sim(vi, ti)/τ) / Σ exp(sim(vi, tj)/τ)
    
    其中:
    - sim(vi, ti) = vi·ti / (||vi|| * ||ti||)  # 余弦相似度
    - τ: 温度参数 (默认 0.07)
    - N: 批次大小
    - vi, ti: 第 i 个图像和文本的归一化嵌入

双向对比损失:
    L = (L_image2text + L_text2image) / 2
    
    L_image2text: 以图像为查询，匹配正确文本
    L_text2image: 以文本为查询，匹配正确图像

=== 算法流程 ===

训练阶段:
    输入: 图像-文本对 [(I1, T1), (I2, T2), ..., (IN, TN)]
      ↓
    图像编码: vi = VisionEncoder(Ii)  # [batch, embed_dim]
    文本编码: ti = TextEncoder(Ti)    # [batch, embed_dim]
      ↓
    L2 归一化: vi = vi / ||vi||, ti = ti / ||ti||
      ↓
    计算相似度矩阵: S = vi * ti^T  # [batch, batch]
      ↓
    计算对比损失: L = InfoNCE(S)
      ↓
    反向传播更新参数

推理阶段 (零样本分类):
    输入: 图像 I, 类别文本描述 [T1, T2, ..., Tk]
      ↓
    编码图像: v = VisionEncoder(I)
    编码文本: [t1, t2, ..., tk] = TextEncoder([T1, T2, ..., Tk])
      ↓
    计算相似度: scores = [v·t1, v·t2, ..., v·tk]
      ↓
    预测类别: argmax(scores)

=== 参考文献 ===

1. CLIP 原始论文:
   Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" ICML 2021
   https://arxiv.org/abs/2103.00020

2. 对比学习:
   Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" ICML 2020
   https://arxiv.org/abs/2002.05709

3. Vision Transformer:
   Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ICLR 2021
   https://arxiv.org/abs/2010.11929

=== 核心组件 ===

    - CLIPConfig: CLIP 模型配置
    - PatchEmbedding: 图像分块嵌入
    - MultiHeadAttention: 多头自注意力机制
    - MLP: 前馈神经网络
    - TransformerBlock: Transformer 编码器块
    - VisionEncoder: 基于 ViT 的图像编码器
    - TextEncoder: 基于 Transformer 的文本编码器
    - CLIP: 完整的对比学习模型
    - clip_loss: InfoNCE 对比损失函数
    - create_clip_model: 创建预定义大小的 CLIP 模型
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CLIPConfig:
    """CLIP 模型配置。

    参数：
        image_size: 输入图像大小 (默认224)
        patch_size: 图像分块大小 (默认16)
        vision_layers: 视觉编码器层数
        vision_width: 视觉编码器隐藏层维度
        vision_heads: 视觉编码器注意力头数
        vocab_size: 词汇表大小
        context_length: 文本最大序列长度
        text_layers: 文本编码器层数
        text_width: 文本编码器隐藏层维度
        text_heads: 文本编码器注意力头数
        embed_dim: 投影后的嵌入维度
        dropout: Dropout 概率
    """

    # 视觉编码器配置
    image_size: int = 224
    patch_size: int = 16
    vision_layers: int = 12
    vision_width: int = 768
    vision_heads: int = 12

    # 文本编码器配置
    vocab_size: int = 49408
    context_length: int = 77
    text_layers: int = 12
    text_width: int = 512
    text_heads: int = 8

    # 共享配置
    embed_dim: int = 512  # 投影后的嵌入维度
    dropout: float = 0.0

    def __post_init__(self):
        """验证配置参数。"""
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"


class PatchEmbedding(nn.Module):
    """将图像分割为 patches 并嵌入。

    实现细节：
        - 使用卷积层将图像分割为 patches
        - 添加可学习的 [CLS] token
        - 添加可学习的位置嵌入
    """

    def __init__(self, config: CLIPConfig):
        """初始化图像分块嵌入。

        参数：
            config: CLIP 配置
        """
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # 使用卷积实现 patch 嵌入
        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=config.vision_width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False
        )

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_width))

        # 位置嵌入
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.vision_width)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数：
            x: 图像张量 [batch_size, 3, height, width]

        返回：
            patch 嵌入 [batch_size, num_patches + 1, vision_width]
        """
        batch_size = x.shape[0]

        # [B, 3, H, W] -> [B, vision_width, H/P, W/P] -> [B, vision_width, num_patches]
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, vision_width]

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, vision_width]

        # 添加位置嵌入
        x = x + self.position_embedding

        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制。

    数学原理：
        Attention(Q, K, V) = softmax(QK^T / √d_k) V

    特性：
        - 支持因果掩码 (用于文本解码器)
        - 支持注意力掩码 (用于 padding)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        """初始化多头注意力。

        参数：
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout 概率
        """
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
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            causal: 是否使用因果掩码
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用因果掩码
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
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
    """Transformer 编码器块"""

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
    """基于 ViT 的视觉编码器"""

    def __init__(self, config: CLIPConfig):
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
        self.projection = nn.Linear(config.vision_width, config.embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量 [batch_size, 3, height, width]
        Returns:
            图像嵌入 [batch_size, embed_dim]
        """
        x = self.patch_embed(x)
        x = self.ln_pre(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x[:, 0, :])  # 取 [CLS] token
        x = self.projection(x)

        return x


class TextEncoder(nn.Module):
    """基于 Transformer 的文本编码器"""

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.context_length, config.text_width)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.text_width,
                num_heads=config.text_heads,
                mlp_ratio=4.0,
                dropout=config.dropout,
                causal=True  # 文本使用因果注意力
            )
            for _ in range(config.text_layers)
        ])

        self.ln_final = nn.LayerNorm(config.text_width)
        self.projection = nn.Linear(config.text_width, config.embed_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        Returns:
            文本嵌入 [batch_size, embed_dim]
        """
        seq_len = input_ids.shape[1]

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :seq_len, :]

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_final(x)

        # 取 [EOS] token 的表示 (最后一个非 padding token)
        if attention_mask is not None:
            eos_indices = attention_mask.sum(dim=1).long() - 1
            x = x[torch.arange(x.shape[0], device=x.device), eos_indices]
        else:
            x = x[:, -1, :]

        x = self.projection(x)

        return x


class CLIP(nn.Module):
    """CLIP 对比学习模型"""

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        self.vision_encoder = VisionEncoder(config)
        self.text_encoder = TextEncoder(config)

        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像"""
        return self.vision_encoder(images)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """编码文本"""
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images: 图像张量 [batch_size, 3, height, width]
            input_ids: token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        Returns:
            (image_features, text_features, logit_scale)
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)

        # L2 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp()

    def get_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """计算图像-文本相似度矩阵"""
        return self.logit_scale.exp() * image_features @ text_features.T


def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor
) -> torch.Tensor:
    """
    CLIP 对比损失函数

    Args:
        image_features: 归一化的图像特征 [batch_size, embed_dim]
        text_features: 归一化的文本特征 [batch_size, embed_dim]
        logit_scale: 温度参数的指数

    Returns:
        对比损失值
    """
    batch_size = image_features.shape[0]
    device = image_features.device

    # 计算相似度矩阵
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    # 标签：对角线为正样本
    labels = torch.arange(batch_size, device=device)

    # 双向对比损失
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2


def create_clip_model(model_size: str = "base") -> CLIP:
    """
    创建预定义大小的 CLIP 模型

    Args:
        model_size: 模型大小 ("small", "base", "large")

    Returns:
        CLIP 模型实例
    """
    configs = {
        "small": CLIPConfig(
            vision_layers=6, vision_width=384, vision_heads=6,
            text_layers=6, text_width=256, text_heads=4,
            embed_dim=256
        ),
        "base": CLIPConfig(
            vision_layers=12, vision_width=768, vision_heads=12,
            text_layers=12, text_width=512, text_heads=8,
            embed_dim=512
        ),
        "large": CLIPConfig(
            vision_layers=24, vision_width=1024, vision_heads=16,
            text_layers=12, text_width=768, text_heads=12,
            embed_dim=768
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return CLIP(configs[model_size])
