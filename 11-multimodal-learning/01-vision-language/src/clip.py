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

SigLIP 损失函数 (Sigmoid Loss for Language-Image Pre-training):
    L = -1/N * Σ_i Σ_j [ y_ij * log(σ(z_ij)) + (1 - y_ij) * log(1 - σ(z_ij)) ]
    
    其中:
    - z_ij = logit_scale * (vi · tj) - bias
    - y_ij = 1 if i == j else 0 (正样本标签)
    - σ: Sigmoid 函数
    
    优势:
    - 不需要全局 softmax 归一化，支持更大 batch size
    - 训练更稳定，收敛更快
    - 支持分布式训练时的局部计算

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
    计算对比损失: L = InfoNCE(S) 或 SigLIP(S)
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

2. SigLIP:
   Zhai et al. "Sigmoid Loss for Language Image Pre-Training" ICCV 2023
   https://arxiv.org/abs/2303.15343

3. OpenCLIP:
   Ilharco et al. "OpenCLIP" 2021
   https://github.com/mlfoundations/open_clip

4. 对比学习:
   Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" ICML 2020
   https://arxiv.org/abs/2002.05709

5. Vision Transformer:
   Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ICLR 2021
   https://arxiv.org/abs/2010.11929

=== 核心组件 ===

    - CLIPConfig: CLIP 模型配置
    - PatchEmbedding: 图像分块嵌入 (支持多尺度)
    - MultiHeadAttention: 多头自注意力机制
    - MLP: 前馈神经网络
    - TransformerBlock: Transformer 编码器块
    - VisionEncoder: 基于 ViT 的图像编码器 (支持梯度检查点)
    - TextEncoder: 基于 Transformer 的文本编码器
    - CLIP: 完整的对比学习模型
    - clip_loss: InfoNCE 对比损失函数
    - siglip_loss: SigLIP Sigmoid 对比损失函数
    - CLIPFineTuner: CLIP 微调工具 (Linear Probe, Adapter, Full)
    - ZeroShotClassifier: 零样本分类器
    - create_clip_model: 创建预定义大小的 CLIP 模型
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Literal, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LossType(Enum):
    """对比损失类型"""
    INFONCE = "infonce"      # 标准 CLIP InfoNCE 损失
    SIGLIP = "siglip"        # SigLIP Sigmoid 损失


class FineTuneMode(Enum):
    """微调模式"""
    LINEAR_PROBE = "linear_probe"  # 仅训练分类头
    ADAPTER = "adapter"            # 训练适配器层
    FULL = "full"                  # 全参数微调
    LORA = "lora"                  # LoRA 低秩适配


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
        use_gradient_checkpointing: 是否使用梯度检查点
        loss_type: 损失函数类型 (infonce/siglip)
        siglip_bias: SigLIP 损失的偏置项
        multi_scale_training: 是否启用多尺度训练
        image_sizes: 多尺度训练的图像尺寸列表
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

    # 高级训练配置
    use_gradient_checkpointing: bool = False  # 梯度检查点，节省显存
    loss_type: str = "infonce"  # 损失类型: "infonce" 或 "siglip"
    siglip_bias: float = -10.0  # SigLIP 损失的偏置项
    
    # 多尺度训练配置
    multi_scale_training: bool = False
    image_sizes: Tuple[int, ...] = (224,)  # 多尺度训练的图像尺寸

    def __post_init__(self):
        """验证配置参数。"""
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
        assert self.loss_type in ("infonce", "siglip"), \
            f"loss_type must be 'infonce' or 'siglip', got {self.loss_type}"


class PatchEmbedding(nn.Module):
    """将图像分割为 patches 并嵌入。

    实现细节：
        - 使用卷积层将图像分割为 patches
        - 添加可学习的 [CLS] token
        - 添加可学习的位置嵌入
        - 支持多尺度输入 (通过位置嵌入插值)
    """

    def __init__(self, config: CLIPConfig):
        """初始化图像分块嵌入。

        参数：
            config: CLIP 配置
        """
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
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

    def _interpolate_pos_encoding(
        self, 
        x: torch.Tensor, 
        h: int, 
        w: int
    ) -> torch.Tensor:
        """插值位置编码以支持不同分辨率的输入。
        
        Args:
            x: patch 嵌入 [batch_size, num_patches + 1, vision_width]
            h: 图像高度对应的 patch 数
            w: 图像宽度对应的 patch 数
            
        Returns:
            添加位置编码后的嵌入
        """
        num_patches = h * w
        num_positions = self.position_embedding.shape[1] - 1  # 减去 CLS token
        
        # 如果 patch 数量匹配，直接使用原始位置编码
        if num_patches == num_positions and h == w:
            return x + self.position_embedding
        
        # 分离 CLS token 的位置编码和 patch 的位置编码
        cls_pos_embed = self.position_embedding[:, :1, :]
        patch_pos_embed = self.position_embedding[:, 1:, :]
        
        # 计算原始位置编码的空间尺寸
        orig_size = int(num_positions ** 0.5)
        
        # 重塑为 2D 并插值
        patch_pos_embed = patch_pos_embed.reshape(
            1, orig_size, orig_size, -1
        ).permute(0, 3, 1, 2)  # [1, dim, orig_size, orig_size]
        
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h, w),
            mode='bicubic',
            align_corners=False
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(
            1, h * w, -1
        )  # [1, h*w, dim]
        
        # 合并 CLS 和 patch 位置编码
        pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
        
        return x + pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数：
            x: 图像张量 [batch_size, 3, height, width]

        返回：
            patch 嵌入 [batch_size, num_patches + 1, vision_width]
        """
        batch_size, _, height, width = x.shape
        
        # 计算实际的 patch 数量
        h = height // self.patch_size
        w = width // self.patch_size

        # [B, 3, H, W] -> [B, vision_width, H/P, W/P] -> [B, vision_width, num_patches]
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, vision_width]

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, vision_width]

        # 添加位置嵌入 (支持多尺度)
        x = self._interpolate_pos_encoding(x, h, w)

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
    """基于 ViT 的视觉编码器
    
    支持特性:
        - 梯度检查点 (节省显存)
        - 多尺度输入 (通过位置编码插值)
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

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

    def set_gradient_checkpointing(self, enable: bool = True):
        """启用或禁用梯度检查点"""
        self.use_gradient_checkpointing = enable

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
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_post(x[:, 0, :])  # 取 [CLS] token
        x = self.projection(x)

        return x


class TextEncoder(nn.Module):
    """基于 Transformer 的文本编码器
    
    支持特性:
        - 梯度检查点 (节省显存)
        - 因果注意力掩码
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

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

    def set_gradient_checkpointing(self, enable: bool = True):
        """启用或禁用梯度检查点"""
        self.use_gradient_checkpointing = enable

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
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
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
    """CLIP 对比学习模型
    
    支持特性:
        - InfoNCE 和 SigLIP 两种损失函数
        - 梯度检查点 (节省显存)
        - 多尺度训练
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        self.vision_encoder = VisionEncoder(config)
        self.text_encoder = TextEncoder(config)

        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
        # SigLIP 偏置参数
        if config.loss_type == "siglip":
            self.logit_bias = nn.Parameter(torch.ones([]) * config.siglip_bias)
        else:
            self.register_buffer("logit_bias", torch.zeros([]))

    def set_gradient_checkpointing(self, enable: bool = True):
        """启用或禁用梯度检查点"""
        self.vision_encoder.set_gradient_checkpointing(enable)
        self.text_encoder.set_gradient_checkpointing(enable)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images: 图像张量 [batch_size, 3, height, width]
            input_ids: token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        Returns:
            (image_features, text_features, logit_scale, logit_bias)
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)

        # L2 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp(), self.logit_bias

    def get_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """计算图像-文本相似度矩阵"""
        return self.logit_scale.exp() * image_features @ text_features.T + self.logit_bias


def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    CLIP InfoNCE 对比损失函数

    Args:
        image_features: 归一化的图像特征 [batch_size, embed_dim]
        text_features: 归一化的文本特征 [batch_size, embed_dim]
        logit_scale: 温度参数的指数
        logit_bias: 偏置项 (可选，用于兼容性)

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


def siglip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor
) -> torch.Tensor:
    """
    SigLIP Sigmoid 对比损失函数
    
    与 InfoNCE 相比的优势:
        - 不需要全局 softmax 归一化
        - 支持更大的 batch size
        - 训练更稳定
        - 支持分布式训练时的局部计算
    
    数学公式:
        L = -1/N * Σ_i Σ_j [ y_ij * log(σ(z_ij)) + (1 - y_ij) * log(1 - σ(z_ij)) ]
        其中 z_ij = logit_scale * (vi · tj) + logit_bias

    Args:
        image_features: 归一化的图像特征 [batch_size, embed_dim]
        text_features: 归一化的文本特征 [batch_size, embed_dim]
        logit_scale: 温度参数的指数
        logit_bias: 偏置项 (SigLIP 特有)

    Returns:
        Sigmoid 对比损失值
        
    Reference:
        Zhai et al. "Sigmoid Loss for Language Image Pre-Training" ICCV 2023
        https://arxiv.org/abs/2303.15343
    """
    batch_size = image_features.shape[0]
    device = image_features.device

    # 计算相似度矩阵 (带偏置)
    logits = logit_scale * image_features @ text_features.T + logit_bias

    # 创建标签矩阵: 对角线为 1 (正样本), 其他为 -1 (负样本)
    # 使用 -1 和 1 是为了配合 sigmoid 损失的数学形式
    labels = 2 * torch.eye(batch_size, device=device) - 1

    # Sigmoid 损失: -log(sigmoid(label * logit))
    # 等价于: log(1 + exp(-label * logit))
    # 这是二元交叉熵的一种形式
    loss = -F.logsigmoid(labels * logits).mean()

    return loss


def contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    loss_type: str = "infonce"
) -> torch.Tensor:
    """
    统一的对比损失函数接口
    
    Args:
        image_features: 归一化的图像特征 [batch_size, embed_dim]
        text_features: 归一化的文本特征 [batch_size, embed_dim]
        logit_scale: 温度参数的指数
        logit_bias: 偏置项
        loss_type: 损失类型 ("infonce" 或 "siglip")
        
    Returns:
        对比损失值
    """
    if loss_type == "siglip":
        return siglip_loss(image_features, text_features, logit_scale, logit_bias)
    else:
        return clip_loss(image_features, text_features, logit_scale, logit_bias)


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


# =============================================================================
# 微调工具和零样本分类器
# =============================================================================


class Adapter(nn.Module):
    """适配器模块 - 用于高效微调
    
    适配器是一种参数高效的微调方法，在原始模型中插入小型可训练模块。
    
    结构:
        x -> LayerNorm -> Down Projection -> GELU -> Up Projection -> x + residual
    
    Reference:
        Houlsby et al. "Parameter-Efficient Transfer Learning for NLP" ICML 2019
    """
    
    def __init__(self, input_dim: int, bottleneck_dim: int = 64):
        """
        Args:
            input_dim: 输入维度
            bottleneck_dim: 瓶颈层维度 (通常远小于 input_dim)
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        
        # 初始化为近似恒等映射
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = F.gelu(x)
        x = self.up_proj(x)
        return x + residual


class CLIPWithAdapters(nn.Module):
    """带适配器的 CLIP 模型
    
    在视觉编码器的每个 Transformer 块后添加适配器。
    """
    
    def __init__(self, clip_model: CLIP, bottleneck_dim: int = 64):
        super().__init__()
        self.clip = clip_model
        self.config = clip_model.config
        
        # 冻结原始 CLIP 参数
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # 为视觉编码器添加适配器
        self.vision_adapters = nn.ModuleList([
            Adapter(self.config.vision_width, bottleneck_dim)
            for _ in range(self.config.vision_layers)
        ])
        
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 视觉编码 (带适配器)
        x = self.clip.vision_encoder.patch_embed(images)
        x = self.clip.vision_encoder.ln_pre(x)
        
        for block, adapter in zip(self.clip.vision_encoder.blocks, self.vision_adapters):
            x = block(x)
            x = adapter(x)
        
        x = self.clip.vision_encoder.ln_post(x[:, 0, :])
        image_features = self.clip.vision_encoder.projection(x)
        
        # 文本编码 (不变)
        text_features = self.clip.encode_text(input_ids, attention_mask)
        
        # L2 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features, self.clip.logit_scale.exp(), self.clip.logit_bias


class ZeroShotClassifier:
    """零样本分类器
    
    使用 CLIP 进行零样本图像分类。
    
    使用方法:
        classifier = ZeroShotClassifier(clip_model)
        classifier.set_classes(["cat", "dog", "bird"])
        predictions = classifier.predict(images)
    """
    
    def __init__(
        self, 
        model: CLIP,
        prompt_template: str = "a photo of a {}"
    ):
        """
        Args:
            model: CLIP 模型
            prompt_template: 提示模板，{} 会被替换为类别名
        """
        self.model = model
        self.prompt_template = prompt_template
        self.text_features: Optional[torch.Tensor] = None
        self.class_names: List[str] = []
        
    def set_classes(
        self,
        class_names: List[str],
        tokenizer_fn,
        device: Optional[torch.device] = None
    ):
        """设置分类类别
        
        Args:
            class_names: 类别名称列表
            tokenizer_fn: 分词函数，接受文本列表返回 (input_ids, attention_mask)
            device: 设备
        """
        self.class_names = class_names
        
        if device is None:
            device = next(self.model.parameters()).device
            
        # 生成类别文本描述
        prompts = [self.prompt_template.format(name) for name in class_names]
        
        # 编码文本
        input_ids, attention_mask = tokenizer_fn(prompts)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(input_ids, attention_mask)
            self.text_features = F.normalize(text_features, dim=-1)
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """预测图像类别
        
        Args:
            images: 图像张量 [batch_size, 3, H, W]
            return_probs: 是否返回概率分布
            
        Returns:
            predictions: 预测类别索引 [batch_size]
            probs: (可选) 概率分布 [batch_size, num_classes]
        """
        if self.text_features is None:
            raise RuntimeError("Please call set_classes() first")
        
        # 编码图像
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # 计算相似度
        similarity = self.model.logit_scale.exp() * image_features @ self.text_features.T
        
        # 预测
        probs = F.softmax(similarity, dim=-1)
        predictions = probs.argmax(dim=-1)
        
        if return_probs:
            return predictions, probs
        return predictions
    
    @torch.no_grad()
    def evaluate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """评估分类准确率
        
        Args:
            images: 图像张量 [batch_size, 3, H, W]
            labels: 真实标签 [batch_size]
            
        Returns:
            包含评估指标的字典
        """
        predictions, probs = self.predict(images, return_probs=True)
        
        # Top-1 准确率
        top1_correct = (predictions == labels).float().mean().item()
        
        # Top-5 准确率
        _, top5_preds = probs.topk(min(5, probs.size(1)), dim=-1)
        top5_correct = (top5_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        
        return {
            "top1_accuracy": top1_correct,
            "top5_accuracy": top5_correct
        }


class LinearProbe(nn.Module):
    """线性探测分类器
    
    冻结 CLIP 编码器，仅训练线性分类头。
    这是最简单的迁移学习方法。
    
    使用方法:
        probe = LinearProbe(clip_model, num_classes=1000)
        logits = probe(images)
    """
    
    def __init__(self, clip_model: CLIP, num_classes: int):
        """
        Args:
            clip_model: 预训练的 CLIP 模型
            num_classes: 分类类别数
        """
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Linear(clip_model.config.embed_dim, num_classes)
        
        # 冻结 CLIP 参数
        for param in self.clip.parameters():
            param.requires_grad = False
            
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: 图像张量 [batch_size, 3, H, W]
        Returns:
            分类 logits [batch_size, num_classes]
        """
        with torch.no_grad():
            features = self.clip.encode_image(images)
            features = F.normalize(features, dim=-1)
        return self.classifier(features)


class CLIPFineTuner:
    """CLIP 微调工具类
    
    提供多种微调策略的统一接口。
    
    支持的微调模式:
        - linear_probe: 仅训练分类头
        - adapter: 训练适配器层
        - full: 全参数微调
    """
    
    def __init__(self, clip_model: CLIP):
        self.clip_model = clip_model
        
    def create_linear_probe(self, num_classes: int) -> LinearProbe:
        """创建线性探测分类器"""
        return LinearProbe(self.clip_model, num_classes)
    
    def create_adapter_model(self, bottleneck_dim: int = 64) -> CLIPWithAdapters:
        """创建带适配器的模型"""
        return CLIPWithAdapters(self.clip_model, bottleneck_dim)
    
    def prepare_for_full_finetune(
        self,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        freeze_text_encoder: bool = True
    ) -> Tuple[CLIP, List[Dict]]:
        """准备全参数微调
        
        Args:
            learning_rate: 学习率
            weight_decay: 权重衰减
            freeze_text_encoder: 是否冻结文本编码器
            
        Returns:
            (model, param_groups) 用于优化器
        """
        model = self.clip_model
        
        if freeze_text_encoder:
            for param in model.text_encoder.parameters():
                param.requires_grad = False
        
        # 分组参数，对不同层使用不同学习率
        no_decay = ["bias", "LayerNorm.weight", "ln"]
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": learning_rate
            }
        ]
        
        return model, param_groups
    
    @staticmethod
    def get_trainable_params(model: nn.Module) -> Tuple[int, int]:
        """获取可训练参数数量
        
        Returns:
            (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total


# =============================================================================
# 辅助函数
# =============================================================================


def create_siglip_model(model_size: str = "base") -> CLIP:
    """
    创建使用 SigLIP 损失的 CLIP 模型
    
    Args:
        model_size: 模型大小 ("small", "base", "large")
        
    Returns:
        配置为 SigLIP 损失的 CLIP 模型
    """
    configs = {
        "small": CLIPConfig(
            vision_layers=6, vision_width=384, vision_heads=6,
            text_layers=6, text_width=256, text_heads=4,
            embed_dim=256,
            loss_type="siglip",
            siglip_bias=-10.0
        ),
        "base": CLIPConfig(
            vision_layers=12, vision_width=768, vision_heads=12,
            text_layers=12, text_width=512, text_heads=8,
            embed_dim=512,
            loss_type="siglip",
            siglip_bias=-10.0
        ),
        "large": CLIPConfig(
            vision_layers=24, vision_width=1024, vision_heads=16,
            text_layers=12, text_width=768, text_heads=12,
            embed_dim=768,
            loss_type="siglip",
            siglip_bias=-10.0
        ),
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    return CLIP(configs[model_size])


def create_multiscale_clip_model(
    model_size: str = "base",
    image_sizes: Tuple[int, ...] = (224, 336, 448)
) -> CLIP:
    """
    创建支持多尺度训练的 CLIP 模型
    
    Args:
        model_size: 模型大小
        image_sizes: 训练时使用的图像尺寸列表
        
    Returns:
        配置为多尺度训练的 CLIP 模型
    """
    base_configs = {
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
    
    if model_size not in base_configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config = base_configs[model_size]
    config.multi_scale_training = True
    config.image_sizes = image_sizes
    
    return CLIP(config)
