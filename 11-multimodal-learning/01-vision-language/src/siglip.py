"""
SigLIP (Sigmoid Loss for Language-Image Pre-training) 独立实现

SigLIP 是 Google 提出的改进版 CLIP，使用 Sigmoid 损失替代 Softmax 损失。

=== 核心思想 ===

SigLIP 的核心创新是使用 Sigmoid 损失函数替代传统的 InfoNCE (Softmax) 损失：

1. Sigmoid 损失优势:
   - 不需要全局 softmax 归一化，支持更大 batch size
   - 每个图像-文本对独立计算损失，便于分布式训练
   - 训练更稳定，收敛更快
   - 在相同计算预算下性能更好

2. 与 CLIP 的区别:
   - CLIP: L = -log(exp(s_ii/τ) / Σ_j exp(s_ij/τ))  # Softmax
   - SigLIP: L = -Σ_ij [y_ij * log(σ(z_ij)) + (1-y_ij) * log(1-σ(z_ij))]  # Sigmoid

=== 数学基础 ===

Sigmoid 对比损失:
    L = -1/(N*N) * Σ_i Σ_j [ y_ij * log(σ(z_ij)) + (1 - y_ij) * log(1 - σ(z_ij)) ]
    
    其中:
    - z_ij = t * (vi · tj) + b  # t: 温度, b: 偏置
    - y_ij = 1 if i == j else 0 (正样本标签)
    - σ: Sigmoid 函数
    
    等价形式 (数值稳定):
    L = -1/(N*N) * Σ_ij log(σ(label_ij * z_ij))
    其中 label_ij = 2*y_ij - 1 ∈ {-1, 1}

分块计算 (Chunked Loss):
    对于超大 batch size，可以分块计算损失:
    L = Σ_chunks L_chunk / num_chunks

=== 架构特点 ===

1. 视觉编码器:
   - 基于 ViT (Vision Transformer)
   - 支持多种尺寸: So400m, B/16, L/16, So400m/14
   - 使用 SwiGLU 激活函数 (可选)

2. 文本编码器:
   - 基于 Transformer
   - 使用 [EOS] token 作为句子表示
   - 支持更长的上下文长度

3. 训练策略:
   - 使用较大的 batch size (32k+)
   - 可学习的温度和偏置参数
   - 支持渐进式分辨率训练

=== 参考文献 ===

1. SigLIP 原始论文:
   Zhai et al. "Sigmoid Loss for Language Image Pre-Training" ICCV 2023
   https://arxiv.org/abs/2303.15343

2. CLIP:
   Radford et al. "Learning Transferable Visual Models From Natural Language Supervision"
   https://arxiv.org/abs/2103.00020

3. OpenCLIP:
   https://github.com/mlfoundations/open_clip

=== 核心组件 ===

    - SigLIPConfig: SigLIP 模型配置
    - SwiGLU: SwiGLU 激活函数
    - SigLIPPatchEmbedding: 图像分块嵌入
    - SigLIPAttention: 多头自注意力
    - SigLIPMLP: 前馈网络 (支持 SwiGLU)
    - SigLIPEncoderBlock: Transformer 编码器块
    - SigLIPVisionEncoder: 视觉编码器
    - SigLIPTextEncoder: 文本编码器
    - SigLIP: 完整模型
    - siglip_loss: Sigmoid 对比损失
    - chunked_siglip_loss: 分块 Sigmoid 损失
    - create_siglip_model: 模型工厂函数
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Literal
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SigLIPModelSize(Enum):
    """SigLIP 模型尺寸"""
    SMALL = "small"      # So400m 风格
    BASE = "base"        # B/16
    LARGE = "large"      # L/16
    SO400M = "so400m"    # So400m/14


@dataclass
class SigLIPConfig:
    """SigLIP 模型配置。

    参数：
        image_size: 输入图像大小 (默认224)
        patch_size: 图像分块大小 (默认16)
        vision_layers: 视觉编码器层数
        vision_width: 视觉编码器隐藏层维度
        vision_heads: 视觉编码器注意力头数
        vision_mlp_ratio: 视觉 MLP 扩展比例
        vocab_size: 词汇表大小
        context_length: 文本最大序列长度
        text_layers: 文本编码器层数
        text_width: 文本编码器隐藏层维度
        text_heads: 文本编码器注意力头数
        text_mlp_ratio: 文本 MLP 扩展比例
        embed_dim: 投影后的嵌入维度
        dropout: Dropout 概率
        use_swiglu: 是否使用 SwiGLU 激活
        init_logit_scale: 初始温度参数 (log scale)
        init_logit_bias: 初始偏置参数
        use_gradient_checkpointing: 是否使用梯度检查点
        layer_norm_eps: LayerNorm epsilon
    """

    # 视觉编码器配置
    image_size: int = 224
    patch_size: int = 16
    vision_layers: int = 12
    vision_width: int = 768
    vision_heads: int = 12
    vision_mlp_ratio: float = 4.0

    # 文本编码器配置
    vocab_size: int = 32000
    context_length: int = 64
    text_layers: int = 12
    text_width: int = 768
    text_heads: int = 12
    text_mlp_ratio: float = 4.0

    # 共享配置
    embed_dim: int = 768
    dropout: float = 0.0
    
    # SigLIP 特有配置
    use_swiglu: bool = True  # 使用 SwiGLU 激活
    init_logit_scale: float = 10.0  # 初始温度 (非 log)
    init_logit_bias: float = -10.0  # 初始偏置
    
    # 训练配置
    use_gradient_checkpointing: bool = False
    layer_norm_eps: float = 1e-6

    def __post_init__(self):
        """验证配置参数。"""
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
        assert self.vision_width % self.vision_heads == 0, \
            f"vision_width ({self.vision_width}) must be divisible by vision_heads ({self.vision_heads})"
        assert self.text_width % self.text_heads == 0, \
            f"text_width ({self.text_width}) must be divisible by text_heads ({self.text_heads})"


class SwiGLU(nn.Module):
    """SwiGLU 激活函数
    
    SwiGLU(x) = Swish(xW) * (xV)
    
    其中 Swish(x) = x * sigmoid(x)
    
    Reference:
        Shazeer "GLU Variants Improve Transformer" 2020
        https://arxiv.org/abs/2002.05202
    """
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.w = nn.Linear(in_features, hidden_features, bias=bias)
        self.v = nn.Linear(in_features, hidden_features, bias=bias)
        self.out = nn.Linear(hidden_features, out_features, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.w(x)) * self.v(x))


class SigLIPPatchEmbedding(nn.Module):
    """SigLIP 图像分块嵌入
    
    特点:
        - 使用卷积实现 patch 嵌入
        - 可学习的位置嵌入
        - 不使用 [CLS] token，使用全局平均池化
    """

    def __init__(self, config: SigLIPConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # Patch 嵌入
        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=config.vision_width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True
        )

        # 位置嵌入 (不包含 CLS token)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, config.vision_width)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量 [batch_size, 3, height, width]
        Returns:
            patch 嵌入 [batch_size, num_patches, vision_width]
        """
        batch_size = x.shape[0]
        
        # [B, 3, H, W] -> [B, vision_width, H/P, W/P] -> [B, num_patches, vision_width]
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加位置嵌入
        x = x + self.position_embedding

        return x


class SigLIPAttention(nn.Module):
    """SigLIP 多头自注意力
    
    特点:
        - 支持因果掩码
        - 支持注意力掩码
        - QKV 合并投影 (可选)
    """

    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV 投影
        self.q_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

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


class SigLIPMLP(nn.Module):
    """SigLIP 前馈网络
    
    支持标准 GELU 和 SwiGLU 两种激活函数。
    """

    def __init__(
        self, 
        d_model: int, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0,
        use_swiglu: bool = True
    ):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        
        if use_swiglu:
            # SwiGLU: 2/3 的隐藏维度以保持参数量
            swiglu_hidden = int(hidden_dim * 2 / 3)
            self.mlp = SwiGLU(d_model, swiglu_hidden, d_model)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout)
            )
        
        self.use_swiglu = use_swiglu
        self.dropout = nn.Dropout(dropout) if use_swiglu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.dropout(x)
        return x


class SigLIPEncoderBlock(nn.Module):
    """SigLIP Transformer 编码器块
    
    Pre-LN 架构: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_swiglu: bool = True,
        causal: bool = False,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        self.causal = causal

        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = SigLIPAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = SigLIPMLP(d_model, mlp_ratio, dropout, use_swiglu)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask, causal=self.causal)
        x = x + self.mlp(self.ln2(x))
        return x


class SigLIPVisionEncoder(nn.Module):
    """SigLIP 视觉编码器
    
    特点:
        - 基于 ViT 架构
        - 使用全局平均池化 (无 CLS token)
        - 支持梯度检查点
        - 支持 SwiGLU 激活
    """

    def __init__(self, config: SigLIPConfig):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        self.patch_embed = SigLIPPatchEmbedding(config)
        
        self.blocks = nn.ModuleList([
            SigLIPEncoderBlock(
                d_model=config.vision_width,
                num_heads=config.vision_heads,
                mlp_ratio=config.vision_mlp_ratio,
                dropout=config.dropout,
                use_swiglu=config.use_swiglu,
                causal=False,
                layer_norm_eps=config.layer_norm_eps
            )
            for _ in range(config.vision_layers)
        ])

        self.ln_post = nn.LayerNorm(config.vision_width, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.vision_width, config.embed_dim, bias=False)

    def set_gradient_checkpointing(self, enable: bool = True):
        self.use_gradient_checkpointing = enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量 [batch_size, 3, height, width]
        Returns:
            图像嵌入 [batch_size, embed_dim]
        """
        x = self.patch_embed(x)

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # 全局平均池化 (SigLIP 特有)
        x = x.mean(dim=1)
        x = self.ln_post(x)
        x = self.projection(x)

        return x


class SigLIPTextEncoder(nn.Module):
    """SigLIP 文本编码器
    
    特点:
        - 使用 [EOS] token 作为句子表示
        - 支持梯度检查点
        - 因果注意力掩码
    """

    def __init__(self, config: SigLIPConfig):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        self.token_embedding = nn.Embedding(config.vocab_size, config.text_width)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.context_length, config.text_width)
        )

        self.blocks = nn.ModuleList([
            SigLIPEncoderBlock(
                d_model=config.text_width,
                num_heads=config.text_heads,
                mlp_ratio=config.text_mlp_ratio,
                dropout=config.dropout,
                use_swiglu=config.use_swiglu,
                causal=True,
                layer_norm_eps=config.layer_norm_eps
            )
            for _ in range(config.text_layers)
        ])

        self.ln_final = nn.LayerNorm(config.text_width, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.text_width, config.embed_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.01)

    def set_gradient_checkpointing(self, enable: bool = True):
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

        # 取 [EOS] token 的表示
        if attention_mask is not None:
            eos_indices = attention_mask.sum(dim=1).long() - 1
            x = x[torch.arange(x.shape[0], device=x.device), eos_indices]
        else:
            x = x[:, -1, :]

        x = self.projection(x)

        return x


class SigLIP(nn.Module):
    """SigLIP 完整模型
    
    使用 Sigmoid 损失的视觉-语言对比学习模型。
    """

    def __init__(self, config: SigLIPConfig):
        super().__init__()
        self.config = config

        self.vision_encoder = SigLIPVisionEncoder(config)
        self.text_encoder = SigLIPTextEncoder(config)

        # 可学习的温度和偏置参数
        self.logit_scale = nn.Parameter(torch.tensor(config.init_logit_scale))
        self.logit_bias = nn.Parameter(torch.tensor(config.init_logit_bias))

    def set_gradient_checkpointing(self, enable: bool = True):
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

        return image_features, text_features, self.logit_scale, self.logit_bias

    def get_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """计算图像-文本相似度矩阵"""
        return self.logit_scale * image_features @ text_features.T + self.logit_bias


# =============================================================================
# 损失函数
# =============================================================================


def siglip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor
) -> torch.Tensor:
    """
    SigLIP Sigmoid 对比损失函数
    
    数学公式:
        L = -1/(N*N) * Σ_ij [ y_ij * log(σ(z_ij)) + (1 - y_ij) * log(1 - σ(z_ij)) ]
        
        等价形式 (数值稳定):
        L = -1/(N*N) * Σ_ij log(σ(label_ij * z_ij))
        其中 label_ij = 2*y_ij - 1 ∈ {-1, 1}

    Args:
        image_features: 归一化的图像特征 [batch_size, embed_dim]
        text_features: 归一化的文本特征 [batch_size, embed_dim]
        logit_scale: 温度参数
        logit_bias: 偏置参数

    Returns:
        Sigmoid 对比损失值
    """
    batch_size = image_features.shape[0]
    device = image_features.device

    # 计算相似度矩阵 (带偏置)
    logits = logit_scale * image_features @ text_features.T + logit_bias

    # 创建标签矩阵: 对角线为 1 (正样本), 其他为 -1 (负样本)
    labels = 2 * torch.eye(batch_size, device=device) - 1

    # Sigmoid 损失: -log(sigmoid(label * logit))
    loss = -F.logsigmoid(labels * logits).mean()

    return loss


def chunked_siglip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    chunk_size: int = 1024
) -> torch.Tensor:
    """
    分块 SigLIP 损失 - 用于超大 batch size
    
    将大 batch 分成小块计算，减少显存占用。
    
    Args:
        image_features: 归一化的图像特征 [batch_size, embed_dim]
        text_features: 归一化的文本特征 [batch_size, embed_dim]
        logit_scale: 温度参数
        logit_bias: 偏置参数
        chunk_size: 每块的大小

    Returns:
        Sigmoid 对比损失值
    """
    batch_size = image_features.shape[0]
    device = image_features.device
    
    if batch_size <= chunk_size:
        return siglip_loss(image_features, text_features, logit_scale, logit_bias)
    
    total_loss = 0.0
    num_chunks = 0
    
    # 分块计算
    for i in range(0, batch_size, chunk_size):
        for j in range(0, batch_size, chunk_size):
            img_chunk = image_features[i:i+chunk_size]
            txt_chunk = text_features[j:j+chunk_size]
            
            # 计算相似度
            logits = logit_scale * img_chunk @ txt_chunk.T + logit_bias
            
            # 创建标签 (只有对角块有正样本)
            chunk_i_size = img_chunk.shape[0]
            chunk_j_size = txt_chunk.shape[0]
            
            if i == j:
                # 对角块
                labels = 2 * torch.eye(chunk_i_size, chunk_j_size, device=device) - 1
            else:
                # 非对角块，全是负样本
                labels = -torch.ones(chunk_i_size, chunk_j_size, device=device)
            
            chunk_loss = -F.logsigmoid(labels * logits).sum()
            total_loss += chunk_loss
            num_chunks += chunk_i_size * chunk_j_size
    
    return total_loss / num_chunks


# =============================================================================
# 零样本分类器
# =============================================================================


class SigLIPZeroShotClassifier:
    """SigLIP 零样本分类器"""
    
    def __init__(
        self, 
        model: SigLIP,
        prompt_template: str = "{}"
    ):
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
        """设置分类类别"""
        self.class_names = class_names
        
        if device is None:
            device = next(self.model.parameters()).device
            
        prompts = [self.prompt_template.format(name) for name in class_names]
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
    ):
        """预测图像类别"""
        if self.text_features is None:
            raise RuntimeError("Please call set_classes() first")
        
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        similarity = self.model.logit_scale * image_features @ self.text_features.T
        probs = F.softmax(similarity, dim=-1)
        predictions = probs.argmax(dim=-1)
        
        if return_probs:
            return predictions, probs
        return predictions


# =============================================================================
# 模型工厂函数
# =============================================================================


def create_siglip_model(model_size: str = "base") -> SigLIP:
    """
    创建预定义大小的 SigLIP 模型

    Args:
        model_size: 模型大小 ("small", "base", "large", "so400m")

    Returns:
        SigLIP 模型实例
    """
    configs = {
        "small": SigLIPConfig(
            vision_layers=6, vision_width=384, vision_heads=6,
            text_layers=6, text_width=384, text_heads=6,
            embed_dim=384, patch_size=16, image_size=224
        ),
        "base": SigLIPConfig(
            vision_layers=12, vision_width=768, vision_heads=12,
            text_layers=12, text_width=768, text_heads=12,
            embed_dim=768, patch_size=16, image_size=224
        ),
        "large": SigLIPConfig(
            vision_layers=24, vision_width=1024, vision_heads=16,
            text_layers=12, text_width=768, text_heads=12,
            embed_dim=768, patch_size=14, image_size=224
        ),
        "so400m": SigLIPConfig(
            vision_layers=27, vision_width=1152, vision_heads=16,
            text_layers=12, text_width=1024, text_heads=16,
            embed_dim=1152, patch_size=14, image_size=224,
            use_swiglu=True
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return SigLIP(configs[model_size])


def create_siglip_config(
    model_size: str = "base",
    image_size: int = 224,
    **kwargs
) -> SigLIPConfig:
    """
    创建 SigLIP 配置
    
    Args:
        model_size: 基础模型大小
        image_size: 图像尺寸
        **kwargs: 额外配置参数
        
    Returns:
        SigLIPConfig 实例
    """
    base_configs = {
        "small": dict(
            vision_layers=6, vision_width=384, vision_heads=6,
            text_layers=6, text_width=384, text_heads=6,
            embed_dim=384, patch_size=16
        ),
        "base": dict(
            vision_layers=12, vision_width=768, vision_heads=12,
            text_layers=12, text_width=768, text_heads=12,
            embed_dim=768, patch_size=16
        ),
        "large": dict(
            vision_layers=24, vision_width=1024, vision_heads=16,
            text_layers=12, text_width=768, text_heads=12,
            embed_dim=768, patch_size=14
        ),
    }
    
    if model_size not in base_configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config_dict = base_configs[model_size]
    config_dict["image_size"] = image_size
    config_dict.update(kwargs)
    
    return SigLIPConfig(**config_dict)
