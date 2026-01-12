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

=== BLIP-2 Q-Former ===

Q-Former (Querying Transformer) 是 BLIP-2 的核心创新：
    - 使用可学习的查询向量从冻结的视觉编码器中提取特征
    - 通过交叉注意力桥接视觉和语言模态
    - 大幅减少训练参数，实现高效预训练

Q-Former 架构:
    查询向量 Q → [自注意力] → [交叉注意力(视觉)] → [自注意力(文本)] → 输出
    
    其中:
    - Q: 可学习的查询向量 [num_queries, query_dim]
    - 交叉注意力: 从视觉特征中提取信息
    - 输出: 固定长度的多模态表示

=== 数学基础 ===

ITC 对比损失 (InfoNCE):
    L_itc = -1/2 * [log(exp(s_ii/τ) / Σ exp(s_ij/τ)) + log(exp(s_ii/τ) / Σ exp(s_ji/τ))]

ITM 匹配损失:
    L_itm = CrossEntropy(ITM_head(h_cls), y)

LM 语言建模损失:
    L_lm = -Σ log P(w_t | w_{<t}, I)

总损失:
    L = L_itc + L_itm + L_lm

=== 生成策略 ===

1. Greedy Search: 每步选择概率最高的 token
2. Beam Search: 维护 k 个最优候选序列
3. Nucleus Sampling (Top-p): 从累积概率 >= p 的 token 中采样
4. Top-k Sampling: 从概率最高的 k 个 token 中采样

=== 参考文献 ===

1. BLIP 原始论文:
   Li et al. "BLIP: Bootstrapping Language-Image Pre-training for Unified
   Vision-Language Understanding and Generation" ICML 2022
   https://arxiv.org/abs/2201.12086

2. BLIP-2:
   Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
   Image Encoders and Large Language Models" ICML 2023
   https://arxiv.org/abs/2301.12597

3. InstructBLIP:
   Dai et al. "InstructBLIP: Towards General-purpose Vision-Language Models
   with Instruction Tuning" NeurIPS 2023
   https://arxiv.org/abs/2305.06500

=== 核心组件 ===

    - BLIPConfig: BLIP 模型配置
    - QFormerConfig: Q-Former 配置
    - PatchEmbedding: 图像分块嵌入
    - MultiHeadAttention: 多头注意力机制
    - TransformerEncoderBlock: Transformer 编码器块
    - TransformerDecoderBlock: Transformer 解码器块 (带交叉注意力)
    - VisionEncoder: ViT 图像编码器
    - TextEncoder: BERT 风格文本编码器
    - TextDecoder: 自回归文本解码器
    - QFormer: BLIP-2 查询 Transformer
    - BLIP: 完整的多任务模型
    - BLIP2: 带 Q-Former 的 BLIP-2 模型
    - VQAHead: 视觉问答分类头
    - itc_loss, itm_loss, lm_loss: 损失函数
    - create_blip_model, create_blip2_model: 模型工厂函数
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union

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


@dataclass
class QFormerConfig:
    """Q-Former 配置 (BLIP-2)
    
    Q-Former 是 BLIP-2 的核心组件，用于桥接视觉和语言模态。
    """
    
    # 查询配置
    num_query_tokens: int = 32  # 可学习查询向量数量
    query_dim: int = 768  # 查询向量维度
    
    # Transformer 配置
    num_layers: int = 12
    num_heads: int = 12
    hidden_dim: int = 768
    ff_dim: int = 3072
    dropout: float = 0.1
    
    # 视觉输入配置
    vision_width: int = 768  # 视觉编码器输出维度
    
    # 文本配置
    vocab_size: int = 30522
    max_text_length: int = 512
    
    # 输出配置
    cross_attention_freq: int = 2  # 每隔几层添加交叉注意力


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


# =============================================================================
# Q-Former (BLIP-2 核心组件)
# =============================================================================


class QFormerBlock(nn.Module):
    """Q-Former Transformer 块
    
    包含自注意力和可选的交叉注意力层。
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        has_cross_attention: bool = False,
        vision_width: int = 768
    ):
        super().__init__()
        self.has_cross_attention = has_cross_attention
        
        # 自注意力
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # 交叉注意力 (可选)
        if has_cross_attention:
            self.ln_cross = nn.LayerNorm(hidden_dim)
            self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
            # 视觉特征投影 (如果维度不匹配)
            if vision_width != hidden_dim:
                self.vision_proj = nn.Linear(vision_width, hidden_dim)
            else:
                self.vision_proj = nn.Identity()
        
        # FFN
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, ff_dim, dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 自注意力
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.self_attn(hidden_states, hidden_states, hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # 交叉注意力
        if self.has_cross_attention and vision_embeds is not None:
            residual = hidden_states
            hidden_states = self.ln_cross(hidden_states)
            vision_embeds_proj = self.vision_proj(vision_embeds)
            hidden_states = self.cross_attn(hidden_states, vision_embeds_proj, vision_embeds_proj)
            hidden_states = residual + hidden_states
        
        # FFN
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class QFormer(nn.Module):
    """Q-Former: BLIP-2 的查询 Transformer
    
    使用可学习的查询向量从冻结的视觉编码器中提取固定长度的特征表示。
    
    架构:
        1. 可学习查询向量 [num_queries, hidden_dim]
        2. 自注意力层处理查询
        3. 交叉注意力层从视觉特征中提取信息
        4. 输出固定长度的多模态表示
    """
    
    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.config = config
        
        # 可学习的查询向量
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.hidden_dim)
        )
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # 文本嵌入 (用于多模态理解)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_text_length, config.hidden_dim)
        )
        
        # Transformer 层
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            has_cross_attn = (i % config.cross_attention_freq == 0)
            self.layers.append(QFormerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                has_cross_attention=has_cross_attn,
                vision_width=config.vision_width
            ))
        
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)
    
    def forward(
        self,
        vision_embeds: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            vision_embeds: 视觉特征 [batch_size, num_patches, vision_width]
            input_ids: 文本 token IDs [batch_size, seq_len] (可选)
            attention_mask: 注意力掩码 (可选)
            
        Returns:
            query_output: 查询输出 [batch_size, num_queries, hidden_dim]
        """
        batch_size = vision_embeds.shape[0]
        
        # 扩展查询向量
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # 如果有文本输入，拼接查询和文本
        if input_ids is not None:
            text_embeds = self.token_embedding(input_ids)
            seq_len = input_ids.shape[1]
            text_embeds = text_embeds + self.position_embedding[:, :seq_len, :]
            hidden_states = torch.cat([query_tokens, text_embeds], dim=1)
        else:
            hidden_states = query_tokens
        
        # 通过 Transformer 层
        for layer in self.layers:
            hidden_states = layer(hidden_states, vision_embeds, attention_mask)
        
        hidden_states = self.ln_final(hidden_states)
        
        # 只返回查询部分
        query_output = hidden_states[:, :self.config.num_query_tokens, :]
        
        return query_output


# =============================================================================
# 高级生成方法
# =============================================================================


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float('Inf')
) -> torch.Tensor:
    """Top-k 和 Top-p (Nucleus) 采样过滤
    
    Args:
        logits: 预测 logits [batch_size, vocab_size]
        top_k: 保留概率最高的 k 个 token (0 表示不使用)
        top_p: 保留累积概率 >= p 的 token (1.0 表示不使用)
        filter_value: 被过滤 token 的值
        
    Returns:
        过滤后的 logits
    """
    # Top-k 过滤
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    # Top-p (Nucleus) 过滤
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过 top_p 的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 保留第一个超过阈值的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 恢复原始顺序
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    return logits


class GenerationMixin:
    """生成方法混入类
    
    提供多种文本生成策略：
    - Greedy Search
    - Beam Search  
    - Nucleus Sampling (Top-p)
    - Top-k Sampling
    """
    
    def _prepare_generation(
        self,
        images: torch.Tensor,
        bos_token_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备生成所需的编码和初始序列"""
        batch_size = images.shape[0]
        device = images.device
        image_embeds = self.encode_image(images)
        generated = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )
        return image_embeds, generated
    
    @torch.no_grad()
    def generate_greedy(
        self,
        images: torch.Tensor,
        max_length: int = 30,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """贪婪搜索生成"""
        image_embeds, generated = self._prepare_generation(images, bos_token_id)
        batch_size = images.shape[0]
        device = images.device
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            logits = self.text_decoder(generated, image_embeds)
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_tokens = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tokens, pad_token_id),
                next_tokens
            )
            generated = torch.cat([generated, next_tokens], dim=-1)
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        return generated
    
    @torch.no_grad()
    def generate_sample(
        self,
        images: torch.Tensor,
        max_length: int = 30,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        pad_token_id: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ) -> torch.Tensor:
        """采样生成 (支持 Top-k 和 Nucleus Sampling)
        
        Args:
            temperature: 温度参数，越高越随机
            top_k: Top-k 采样参数
            top_p: Nucleus 采样参数
        """
        image_embeds, generated = self._prepare_generation(images, bos_token_id)
        batch_size = images.shape[0]
        device = images.device
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            logits = self.text_decoder(generated, image_embeds)
            next_token_logits = logits[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            next_tokens = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tokens, pad_token_id),
                next_tokens
            )
            generated = torch.cat([generated, next_tokens], dim=-1)
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        return generated
    
    @torch.no_grad()
    def generate_beam(
        self,
        images: torch.Tensor,
        max_length: int = 30,
        num_beams: int = 5,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        pad_token_id: int = 0,
        length_penalty: float = 1.0
    ) -> torch.Tensor:
        """Beam Search 生成
        
        Args:
            num_beams: beam 数量
            length_penalty: 长度惩罚因子 (>1 鼓励长序列, <1 鼓励短序列)
        """
        batch_size = images.shape[0]
        device = images.device
        image_embeds = self.encode_image(images)
        
        # 扩展 image_embeds 以适应 beam search
        image_embeds = image_embeds.unsqueeze(1).expand(-1, num_beams, -1, -1)
        image_embeds = image_embeds.reshape(batch_size * num_beams, -1, image_embeds.size(-1))
        
        # 初始化 beam
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9  # 初始只有第一个 beam 有效
        beam_scores = beam_scores.view(-1)
        
        generated = torch.full(
            (batch_size * num_beams, 1), bos_token_id, dtype=torch.long, device=device
        )
        
        for step in range(max_length - 1):
            logits = self.text_decoder(generated, image_embeds)
            next_token_logits = logits[:, -1, :]
            vocab_size = next_token_logits.size(-1)
            
            next_scores = F.log_softmax(next_token_logits, dim=-1)
            next_scores = next_scores + beam_scores.unsqueeze(-1)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            
            # 选择 top-k
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=-1, largest=True, sorted=True
            )
            
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # 重组 beam
            beam_outputs = []
            beam_scores_new = []
            
            for batch_idx in range(batch_size):
                beam_idx = 0
                for score, token, idx in zip(
                    next_scores[batch_idx], next_tokens[batch_idx], next_indices[batch_idx]
                ):
                    if beam_idx >= num_beams:
                        break
                    beam_outputs.append(
                        torch.cat([
                            generated[batch_idx * num_beams + idx],
                            token.unsqueeze(0)
                        ])
                    )
                    beam_scores_new.append(score)
                    beam_idx += 1
            
            generated = torch.stack(beam_outputs).view(batch_size * num_beams, -1)
            beam_scores = torch.tensor(beam_scores_new, device=device)
            
            # 检查是否所有 beam 都结束
            if (generated[:, -1] == eos_token_id).all():
                break
        
        # 应用长度惩罚并选择最佳序列
        final_scores = beam_scores.view(batch_size, num_beams)
        lengths = (generated != pad_token_id).sum(dim=-1).float().view(batch_size, num_beams)
        final_scores = final_scores / (lengths ** length_penalty)
        
        best_indices = final_scores.argmax(dim=-1)
        best_sequences = []
        for batch_idx, beam_idx in enumerate(best_indices):
            best_sequences.append(generated[batch_idx * num_beams + beam_idx])
        
        return torch.stack(best_sequences)


# =============================================================================
# VQA 和 BLIP-2 模型
# =============================================================================


class VQAHead(nn.Module):
    """视觉问答分类头
    
    用于 VQA 任务的答案预测。支持开放式和多选式问答。
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_answers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_answers)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: 融合后的多模态特征 [batch_size, hidden_dim]
        Returns:
            logits: 答案预测 [batch_size, num_answers]
        """
        return self.classifier(hidden_states)


class BLIP2(nn.Module):
    """BLIP-2: 带 Q-Former 的视觉语言模型
    
    BLIP-2 使用冻结的视觉编码器和 Q-Former 实现高效的视觉语言预训练。
    
    架构:
        冻结的视觉编码器 → Q-Former → LLM/任务头
    """
    
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        qformer_config: QFormerConfig,
        freeze_vision: bool = True
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.qformer = QFormer(qformer_config)
        
        # 冻结视觉编码器
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # 投影层 (用于对比学习)
        self.vision_proj = nn.Linear(qformer_config.hidden_dim, qformer_config.hidden_dim)
        self.text_proj = nn.Linear(qformer_config.hidden_dim, qformer_config.hidden_dim)
        
        # ITM 头
        self.itm_head = nn.Linear(qformer_config.hidden_dim, 2)
        
        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像并通过 Q-Former"""
        with torch.no_grad():
            vision_embeds = self.vision_encoder(images)
        query_output = self.qformer(vision_embeds)
        return query_output
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """获取归一化的图像特征"""
        query_output = self.encode_image(images)
        image_feat = self.vision_proj(query_output.mean(dim=1))
        return F.normalize(image_feat, dim=-1)
    
    def get_text_features(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """获取归一化的文本特征 (需要图像上下文)"""
        with torch.no_grad():
            vision_embeds = self.vision_encoder(images)
        query_output = self.qformer(vision_embeds, input_ids, attention_mask)
        text_feat = self.text_proj(query_output.mean(dim=1))
        return F.normalize(text_feat, dim=-1)
    
    def forward_itc(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """图像-文本对比学习"""
        image_feat = self.get_image_features(images)
        text_feat = self.get_text_features(images, input_ids, attention_mask)
        return image_feat, text_feat, self.logit_scale.exp()
    
    def forward_itm(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """图像-文本匹配"""
        with torch.no_grad():
            vision_embeds = self.vision_encoder(images)
        query_output = self.qformer(vision_embeds, input_ids, attention_mask)
        itm_logits = self.itm_head(query_output[:, 0, :])
        return itm_logits


class InstructBLIP(BLIP2):
    """InstructBLIP: 指令微调的 BLIP-2
    
    在 BLIP-2 基础上添加指令理解能力。
    """
    
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        qformer_config: QFormerConfig,
        freeze_vision: bool = True
    ):
        super().__init__(vision_encoder, qformer_config, freeze_vision)
        
        # 指令嵌入
        self.instruction_embedding = nn.Embedding(
            qformer_config.vocab_size, qformer_config.hidden_dim
        )
    
    def forward_with_instruction(
        self,
        images: torch.Tensor,
        instruction_ids: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """带指令的前向传播
        
        Args:
            images: 输入图像
            instruction_ids: 指令 token IDs
            input_ids: 输入文本 token IDs (可选)
            attention_mask: 注意力掩码
        """
        with torch.no_grad():
            vision_embeds = self.vision_encoder(images)
        
        # 将指令嵌入添加到查询中
        instruction_embeds = self.instruction_embedding(instruction_ids)
        
        # 拼接指令和输入
        if input_ids is not None:
            combined_ids = torch.cat([instruction_ids, input_ids], dim=1)
        else:
            combined_ids = instruction_ids
        
        query_output = self.qformer(vision_embeds, combined_ids, attention_mask)
        return query_output


def create_blip2_model(
    model_size: str = "base",
    num_query_tokens: int = 32,
    freeze_vision: bool = True
) -> BLIP2:
    """创建 BLIP-2 模型
    
    Args:
        model_size: 模型大小 ("small", "base", "large")
        num_query_tokens: Q-Former 查询向量数量
        freeze_vision: 是否冻结视觉编码器
    """
    vision_configs = {
        "small": {"layers": 6, "width": 384, "heads": 6},
        "base": {"layers": 12, "width": 768, "heads": 12},
        "large": {"layers": 24, "width": 1024, "heads": 16},
    }
    
    if model_size not in vision_configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    vc = vision_configs[model_size]
    
    # 创建视觉编码器
    vision_encoder = VisionEncoder(
        image_size=224,
        patch_size=16,
        width=vc["width"],
        layers=vc["layers"],
        heads=vc["heads"],
        dropout=0.1
    )
    
    # 创建 Q-Former 配置
    qformer_config = QFormerConfig(
        num_query_tokens=num_query_tokens,
        hidden_dim=vc["width"],
        num_layers=12,
        num_heads=12,
        ff_dim=vc["width"] * 4,
        vision_width=vc["width"]
    )
    
    return BLIP2(vision_encoder, qformer_config, freeze_vision)
