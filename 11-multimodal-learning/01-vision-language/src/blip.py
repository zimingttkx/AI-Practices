"""
BLIP (Bootstrapping Language-Image Pre-training) 实现

BLIP 是一个统一的视觉-语言预训练框架，支持多任务学习。

=== 核心思想 ===

BLIP 通过三个互补的预训练任务实现视觉-语言理解与生成的统一：

1. 图像-文本对比学习 (ITC - Image-Text Contrastive)
   - 学习图像和文本的对齐表示
   - 使用动量蒸馏提升负样本质量
   - 适用于图像-文本检索任务

2. 图像-文本匹配 (ITM - Image-Text Matching)
   - 二分类任务，判断图像-文本是否匹配
   - 使用硬负样本挖掘提升判别能力
   - 适用于细粒度理解任务

3. 语言建模 (LM - Language Modeling)
   - 基于图像生成文本描述
   - 使用因果注意力的自回归解码
   - 适用于图像描述生成任务

=== 数学基础 ===

ITC 对比损失 (InfoNCE):
    L_itc = -1/2 * [log(exp(s_ii/τ) / Σ exp(s_ij/τ)) + log(exp(s_ii/τ) / Σ exp(s_ji/τ))]
    
    其中:
    - s_ij = f_v(I_i)^T · f_t(T_j)  # 图像-文本相似度
    - τ: 温度参数
    - f_v, f_t: 视觉和文本编码器

ITM 匹配损失:
    L_itm = CrossEntropy(ITM_head(h_cls), y)
    
    其中:
    - h_cls: 融合后的 [CLS] 表示
    - y ∈ {0, 1}: 匹配标签

LM 语言建模损失:
    L_lm = -Σ log P(w_t | w_{<t}, I)
    
    其中:
    - w_t: 第 t 个词
    - I: 输入图像
    - P: 条件概率

总损失:
    L = L_itc + L_itm + L_lm

=== 算法流程 ===

预训练阶段:
    输入: 图像-文本对 (I, T)
      ↓
    ┌─────────────────────────────────────┐
    │ 视觉编码: v = VisionEncoder(I)      │
    │ 文本编码: t = TextEncoder(T)        │
    └─────────────────────────────────────┘
      ↓
    ┌─────────────┬─────────────┬─────────────┐
    │    ITC      │    ITM      │     LM      │
    │ 对比学习    │ 匹配判断    │ 描述生成    │
    └─────────────┴─────────────┴─────────────┘
      ↓
    计算总损失并更新参数

推理阶段 (图像描述生成):
    输入: 图像 I
      ↓
    视觉编码: v = VisionEncoder(I)
      ↓
    自回归解码: T = TextDecoder(v, [BOS])
      ↓
    输出: 生成的文本描述 T

=== 参考文献 ===

1. BLIP 原始论文:
   Li et al. "BLIP: Bootstrapping Language-Image Pre-training for Unified
   Vision-Language Understanding and Generation" ICML 2022
   https://arxiv.org/abs/2201.12086

2. 对比学习基础:
   Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" 2021

3. Vision Transformer:
   Dosovitskiy et al. "An Image is Worth 16x16 Words" ICLR 2021

=== 核心组件 ===

    - BLIPConfig: BLIP 模型配置
    - PatchEmbedding: 图像分块嵌入
    - MultiHeadAttention: 多头注意力机制
    - TransformerEncoderBlock: Transformer 编码器块
    - TransformerDecoderBlock: Transformer 解码器块 (带交叉注意力)
    - VisionEncoder: ViT 图像编码器
    - TextEncoder: BERT 风格文本编码器
    - TextDecoder: 自回归文本解码器
    - BLIP: 完整的多任务模型
    - itc_loss: 图像-文本对比损失
    - itm_loss: 图像-文本匹配损失
    - lm_loss: 语言建模损失
    - create_blip_model: 创建预定义大小的 BLIP 模型
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BLIPConfig:
    """BLIP 模型配置。

    参数：
        image_size: 输入图像大小 (默认224)
        patch_size: 图像分块大小 (默认16)
        vision_layers: 视觉编码器层数
        vision_width: 视觉编码器隐藏层维度
        vision_heads: 视觉编码器注意力头数
        vocab_size: 词汇表大小 (BERT词汇表)
        max_text_length: 文本最大序列长度
        text_layers: 文本编码器层数
        text_width: 文本编码器隐藏层维度
        text_heads: 文本编码器注意力头数
        embed_dim: 投影后的嵌入维度
        dropout: Dropout 概率
        decoder_layers: 解码器层数
        decoder_width: 解码器隐藏层维度
        decoder_heads: 解码器注意力头数
    """

    # 视觉编码器配置
    image_size: int = 224
    patch_size: int = 16
    vision_layers: int = 12
    vision_width: int = 768
    vision_heads: int = 12

    # 文本编码器配置
    vocab_size: int = 30522  # BERT vocab size
    max_text_length: int = 512
    text_layers: int = 12
    text_width: int = 768
    text_heads: int = 12

    # 共享配置
    embed_dim: int = 256  # 投影后的嵌入维度
    dropout: float = 0.1

    # 解码器配置
    decoder_layers: int = 12
    decoder_width: int = 768
    decoder_heads: int = 12

    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"


class PatchEmbedding(nn.Module):
    """图像分块嵌入。

    将输入图像分割为固定大小的 patches，并通过卷积层映射到嵌入空间。

    实现细节：
        - 使用卷积层实现高效的 patch 分割和嵌入
        - 添加可学习的 [CLS] token 作为全局表示
        - 添加可学习的位置嵌入编码空间信息
    """

    def __init__(self, config: BLIPConfig):
        """初始化图像分块嵌入。

        参数：
            config: BLIP 配置
        """
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
    """多头注意力机制。

    数学原理：
        Attention(Q, K, V) = softmax(QK^T / √d_k) V

    特性：
        - 支持自注意力和交叉注意力
        - 支持因果掩码 (用于解码器)
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
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        kv_seq_len = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
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


class TransformerEncoderBlock(nn.Module):
    """Transformer 编码器块"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x = self.attn(x, x, x, attention_mask)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer 解码器块 (带交叉注意力)"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 自注意力 (因果)
        residual = x
        x = self.ln1(x)
        x = self.self_attn(x, x, x, attention_mask, causal=True)
        x = residual + x

        # 交叉注意力
        residual = x
        x = self.ln2(x)
        x = self.cross_attn(x, encoder_hidden_states, encoder_hidden_states, encoder_attention_mask)
        x = residual + x

        # FFN
        residual = x
        x = self.ln3(x)
        x = self.mlp(x)
        x = residual + x

        return x


class VisionEncoder(nn.Module):
    """ViT 视觉编码器。

    基于 Vision Transformer 架构的图像编码器，将图像编码为序列表示。

    架构：
        图像 → PatchEmbedding → TransformerBlocks → LayerNorm → 输出
    """

    def __init__(self, config: BLIPConfig):
        """初始化视觉编码器。

        参数：
            config: BLIP 配置
        """
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(config)
        self.ln_pre = nn.LayerNorm(config.vision_width)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.vision_width,
                num_heads=config.vision_heads,
                mlp_ratio=4.0,
                dropout=config.dropout
            )
            for _ in range(config.vision_layers)
        ])

        self.ln_post = nn.LayerNorm(config.vision_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量 [batch_size, 3, height, width]
        Returns:
            图像特征 [batch_size, num_patches + 1, vision_width]
        """
        x = self.patch_embed(x)
        x = self.ln_pre(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextEncoder(nn.Module):
    """BERT 风格文本编码器。

    基于 Transformer 编码器架构，将文本编码为上下文相关的表示。

    架构：
        Token IDs → Embedding → TransformerBlocks → LayerNorm → 输出
    """

    def __init__(self, config: BLIPConfig):
        """初始化文本编码器。

        参数：
            config: BLIP 配置
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_text_length, config.text_width)
        )

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.text_width,
                num_heads=config.text_heads,
                mlp_ratio=4.0,
                dropout=config.dropout
            )
            for _ in range(config.text_layers)
        ])

        self.ln_final = nn.LayerNorm(config.text_width)
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
            文本特征 [batch_size, seq_len, text_width]
        """
        seq_len = input_ids.shape[1]

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :seq_len, :]

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_final(x)
        return x


class TextDecoder(nn.Module):
    """自回归文本解码器 (用于图像描述生成)。

    基于 Transformer 解码器架构，支持交叉注意力机制。

    架构：
        Token IDs → Embedding → DecoderBlocks(自注意力+交叉注意力) → LM Head → logits
    """

    def __init__(self, config: BLIPConfig):
        """初始化文本解码器。

        参数：
            config: BLIP 配置
        """
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.decoder_width)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_text_length, config.decoder_width)
        )

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=config.decoder_width,
                num_heads=config.decoder_heads,
                mlp_ratio=4.0,
                dropout=config.dropout
            )
            for _ in range(config.decoder_layers)
        ])

        self.ln_final = nn.LayerNorm(config.decoder_width)
        self.lm_head = nn.Linear(config.decoder_width, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: token IDs [batch_size, seq_len]
            encoder_hidden_states: 视觉特征 [batch_size, num_patches, vision_width]
            attention_mask: 解码器注意力掩码
            encoder_attention_mask: 编码器注意力掩码
        Returns:
            logits [batch_size, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :seq_len, :]

        for block in self.blocks:
            x = block(x, encoder_hidden_states, attention_mask, encoder_attention_mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits


class BLIP(nn.Module):
    """BLIP 多任务视觉-语言模型。

    支持三种预训练任务：
        - ITC (Image-Text Contrastive): 图像-文本对比学习
        - ITM (Image-Text Matching): 图像-文本匹配
        - LM (Language Modeling): 图像描述生成

    示例：
        >>> config = BLIPConfig()
        >>> model = BLIP(config)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> input_ids = torch.randint(0, 30522, (4, 20))
        >>> # ITC
        >>> image_feat, text_feat, logit_scale = model.forward_itc(images, input_ids)
        >>> # ITM
        >>> itm_logits = model.forward_itm(images, input_ids)
        >>> # LM
        >>> logits = model.forward_lm(images, input_ids)
    """

    def __init__(self, config: BLIPConfig):
        """初始化 BLIP 模型。

        参数：
            config: BLIP 配置
        """
        super().__init__()
        self.config = config

        # 视觉编码器
        self.vision_encoder = VisionEncoder(config)

        # 文本编码器 (用于 ITC 和 ITM)
        self.text_encoder = TextEncoder(config)

        # 文本解码器 (用于图像描述生成)
        self.text_decoder = TextDecoder(config)

        # 投影层 (用于对比学习)
        self.vision_proj = nn.Linear(config.vision_width, config.embed_dim)
        self.text_proj = nn.Linear(config.text_width, config.embed_dim)

        # ITM 分类头
        self.itm_head = nn.Linear(config.text_width, 2)

        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        Returns:
            image_embeds: [batch_size, num_patches + 1, vision_width]
        """
        return self.vision_encoder(images)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码文本
        Returns:
            text_embeds: [batch_size, seq_len, text_width]
        """
        return self.text_encoder(input_ids, attention_mask)

    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """获取归一化的图像特征 (用于对比学习)"""
        image_embeds = self.encode_image(images)
        image_feat = self.vision_proj(image_embeds[:, 0, :])  # CLS token
        image_feat = F.normalize(image_feat, dim=-1)
        return image_feat

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """获取归一化的文本特征 (用于对比学习)"""
        text_embeds = self.encode_text(input_ids, attention_mask)
        text_feat = self.text_proj(text_embeds[:, 0, :])  # CLS token
        text_feat = F.normalize(text_feat, dim=-1)
        return text_feat

    def forward_itc(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        图像-文本对比学习 (ITC)
        Returns:
            (image_features, text_features, logit_scale)
        """
        image_feat = self.get_image_features(images)
        text_feat = self.get_text_features(input_ids, attention_mask)
        return image_feat, text_feat, self.logit_scale.exp()

    def forward_itm(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        图像-文本匹配 (ITM)
        Returns:
            itm_logits: [batch_size, 2]
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)

        # 融合图像和文本特征
        # 简化实现: 使用 CLS token 的拼接
        fused = text_embeds[:, 0, :] + image_embeds[:, 0, :].detach()
        itm_logits = self.itm_head(fused)

        return itm_logits

    def forward_lm(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        图像描述生成 (Language Modeling)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        image_embeds = self.encode_image(images)
        logits = self.text_decoder(input_ids, image_embeds, attention_mask)
        return logits

    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 30,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        自回归生成图像描述
        Args:
            images: 图像张量 [batch_size, 3, H, W]
            max_length: 最大生成长度
            bos_token_id: 开始 token ID
            eos_token_id: 结束 token ID
            pad_token_id: 填充 token ID
        Returns:
            generated_ids: [batch_size, seq_len]
        """
        batch_size = images.shape[0]
        device = images.device

        # 编码图像
        image_embeds = self.encode_image(images)

        # 初始化生成序列
        generated = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits = self.text_decoder(generated, image_embeds)
            next_token_logits = logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)

            # 已完成的序列填充 pad token
            next_tokens = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tokens, pad_token_id),
                next_tokens
            )

            generated = torch.cat([generated, next_tokens], dim=-1)

            # 检查是否生成了 EOS
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        return generated


def itc_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor
) -> torch.Tensor:
    """
    图像-文本对比损失 (InfoNCE)
    """
    batch_size = image_features.shape[0]
    device = image_features.device

    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T

    labels = torch.arange(batch_size, device=device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2


def itm_loss(itm_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    图像-文本匹配损失
    Args:
        itm_logits: [batch_size, 2]
        labels: [batch_size] (0: 不匹配, 1: 匹配)
    """
    return F.cross_entropy(itm_logits, labels)


def lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    语言模型损失
    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
        ignore_index: 忽略的标签索引
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index
    )


def create_blip_model(model_size: str = "base") -> BLIP:
    """
    创建预定义大小的 BLIP 模型

    Args:
        model_size: 模型大小 ("small", "base", "large")

    Returns:
        BLIP 模型实例
    """
    configs = {
        "small": BLIPConfig(
            vision_layers=6, vision_width=384, vision_heads=6,
            text_layers=6, text_width=384, text_heads=6,
            decoder_layers=6, decoder_width=384, decoder_heads=6,
            embed_dim=256
        ),
        "base": BLIPConfig(
            vision_layers=12, vision_width=768, vision_heads=12,
            text_layers=12, text_width=768, text_heads=12,
            decoder_layers=12, decoder_width=768, decoder_heads=12,
            embed_dim=256
        ),
        "large": BLIPConfig(
            vision_layers=24, vision_width=1024, vision_heads=16,
            text_layers=12, text_width=768, text_heads=12,
            decoder_layers=12, decoder_width=768, decoder_heads=12,
            embed_dim=512
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return BLIP(configs[model_size])
