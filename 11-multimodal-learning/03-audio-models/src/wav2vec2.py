"""
Wav2Vec2 自监督语音表示学习 (Self-Supervised Speech Representation Learning)

本模块实现 Wav2Vec 2.0 风格的自监督语音模型，包括：
- 特征编码器 (CNN)
- 上下文网络 (Transformer)
- 量化模块 (Gumbel-Softmax)
- 对比学习损失

=== Wav2Vec2 核心思想 ===

Wav2Vec2 通过自监督学习从大量无标注音频中学习语音表示：

1. 特征编码器: 将原始波形转换为潜在表示
2. 量化模块: 将连续表示离散化为有限的码本向量
3. 上下文网络: 通过 Transformer 建模上下文信息
4. 对比学习: 预测被遮蔽位置的量化表示

=== 训练流程 ===

原始波形 → [特征编码器] → 潜在表示 z
                ↓
        [随机遮蔽部分位置]
                ↓
        [上下文网络] → 上下文表示 c
                ↓
        [对比损失: 预测被遮蔽位置的量化目标 q]

=== 参考文献 ===

1. Wav2Vec 2.0:
   Baevski et al. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" 2020

2. HuBERT:
   Hsu et al. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units" 2021
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Wav2Vec2Config:
    """Wav2Vec2 模型配置"""

    # 特征编码器配置
    conv_dim: Tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: Tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)

    # Transformer 配置
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # 量化配置
    num_codevector_groups: int = 2
    num_codevectors_per_group: int = 320
    codevector_dim: int = 256

    # 遮蔽配置
    mask_time_prob: float = 0.065
    mask_time_length: int = 10
    mask_time_min_masks: int = 2

    # 对比学习配置
    num_negatives: int = 100
    contrastive_logits_temperature: float = 0.1
    diversity_loss_weight: float = 0.1

    # 其他配置
    feat_extract_norm: str = "group"  # "group" or "layer"
    final_dropout: float = 0.1


class ConvLayerBlock(nn.Module):
    """特征编码器的卷积层块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: str = "group"
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False
        )

        if norm_type == "group":
            self.norm = nn.GroupNorm(out_channels, out_channels, affine=True)
        else:
            self.norm = nn.LayerNorm(out_channels)

        self.norm_type = norm_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm_type == "group":
            x = self.norm(x)
        else:
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        x = F.gelu(x)
        return x


class FeatureEncoder(nn.Module):
    """
    特征编码器 (Feature Encoder)

    将原始波形转换为潜在表示序列。
    使用多层 1D 卷积进行下采样。
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config

        conv_layers = []
        in_channels = 1

        for i, (out_channels, kernel, stride) in enumerate(zip(
            config.conv_dim, config.conv_kernel, config.conv_stride
        )):
            norm_type = config.feat_extract_norm if i == 0 else "group"
            conv_layers.append(ConvLayerBlock(
                in_channels, out_channels, kernel, stride, norm_type
            ))
            in_channels = out_channels

        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: 原始波形 [batch, time] 或 [batch, 1, time]
        Returns:
            特征序列 [batch, time', hidden]
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        x = waveform
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.transpose(1, 2)
        return x


class FeatureProjection(nn.Module):
    """特征投影层"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):
    """
    位置卷积嵌入

    使用分组卷积来编码相对位置信息，
    这比正弦位置编码更适合语音任务。
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=128,
            padding=64,
            groups=16
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.gelu(x[:, :, :-1])
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # 扩展 attention_mask 维度: [batch, seq_len] -> [batch, 1, 1, seq_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            # 将 0/1 掩码转换为加性掩码 (0 -> 0, 1 -> -inf 或反过来)
            attention_mask = (1.0 - attention_mask.float()) * -10000.0
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intermediate(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm(x)

        residual = x
        x = self.feed_forward(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x


class TransformerEncoder(nn.Module):
    """Transformer 编码器 (上下文网络)"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        position_embeddings = self.pos_conv_embed(x)
        x = x + position_embeddings
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return x


class GumbelVectorQuantizer(nn.Module):
    """
    Gumbel-Softmax 向量量化器

    将连续的特征表示离散化为码本向量。
    使用 Gumbel-Softmax 技巧实现可微分的离散化。
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group
        self.codevector_dim = config.codevector_dim

        self.weight_proj = nn.Linear(
            config.conv_dim[-1],
            self.num_groups * self.num_vars
        )

        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        nn.init.uniform_(self.codevectors)

        self.temperature = 2.0

    def forward(
        self,
        x: torch.Tensor,
        mask_time_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        x = self.weight_proj(x)
        x = x.view(batch_size * seq_len * self.num_groups, -1)

        if self.training:
            codevector_probs = F.gumbel_softmax(
                x.float(), tau=self.temperature, hard=True
            ).type_as(x)
        else:
            codevector_idx = x.argmax(dim=-1)
            codevector_probs = F.one_hot(codevector_idx, self.num_vars).type_as(x)

        codevector_probs = codevector_probs.view(batch_size * seq_len, self.num_groups, -1)
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors.view(
            1, self.num_groups, self.num_vars, -1
        )
        codevectors = codevectors_per_group.sum(dim=2).view(batch_size, seq_len, -1)

        perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        return codevectors, perplexity

    def _compute_perplexity(
        self,
        probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算码本使用的困惑度，用于衡量码本利用率"""
        if mask is not None:
            mask = mask.flatten()
            probs = probs[mask.bool()]

        # 避免空张量
        if probs.numel() == 0:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)

        marginal_probs = probs.mean(dim=0)
        # 使用更稳定的 epsilon 和 clamp 防止数值问题
        marginal_probs = torch.clamp(marginal_probs, min=1e-10)
        perplexity = torch.exp(-torch.sum(
            marginal_probs * torch.log(marginal_probs), dim=-1
        )).sum()

        return perplexity


def compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    min_masks: int = 2,
    device: torch.device = None
) -> torch.Tensor:
    """
    计算遮蔽索引 (向量化实现)

    Args:
        shape: (batch_size, seq_len)
        mask_prob: 遮蔽概率
        mask_length: 每个遮蔽区域的长度
        min_masks: 最小遮蔽数量
        device: 设备

    Returns:
        遮蔽索引 [batch_size, seq_len]
    """
    batch_size, seq_len = shape

    # 边界检查
    if seq_len <= 0:
        return torch.zeros(batch_size, max(seq_len, 0), dtype=torch.bool, device=device)

    mask_length = min(mask_length, seq_len)
    if mask_length <= 0:
        mask_length = 1

    num_masked_spans = max(int(mask_prob * seq_len / mask_length + 0.5), min_masks)
    num_masked_spans = min(num_masked_spans, seq_len // mask_length)

    if num_masked_spans <= 0:
        return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    # 向量化实现：为每个 batch 生成随机起始位置
    max_start = seq_len - mask_length + 1
    if max_start <= 0:
        max_start = 1

    # 生成随机起始索引 [batch_size, num_masked_spans]
    rand_indices = torch.rand(batch_size, num_masked_spans, device=device)
    start_indices = (rand_indices * max_start).long()

    # 创建遮蔽区域
    for i in range(batch_size):
        for idx in start_indices[i]:
            end_idx = min(idx.item() + mask_length, seq_len)
            mask[i, idx.item():end_idx] = True

    return mask


class Wav2Vec2Model(nn.Module):
    """
    Wav2Vec2 自监督语音模型

    核心组件:
    1. 特征编码器: CNN 将波形转换为潜在表示
    2. 特征投影: 线性层投影到 Transformer 维度
    3. 量化器: Gumbel-Softmax 离散化
    4. 上下文网络: Transformer 编码器
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config

        self.feature_extractor = FeatureEncoder(config)
        self.feature_projection = FeatureProjection(config)
        self.quantizer = GumbelVectorQuantizer(config)
        self.encoder = TransformerEncoder(config)

        self.project_q = nn.Linear(config.codevector_dim, config.hidden_size)
        self.project_hid = nn.Linear(config.hidden_size, config.codevector_dim)

        self.dropout_features = nn.Dropout(config.final_dropout)
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.Tensor] = None
    ) -> dict:
        """
        前向传播

        Args:
            waveform: 原始波形 [batch, time]
            attention_mask: 注意力掩码
            mask_time_indices: 时间遮蔽索引 (训练时)

        Returns:
            包含各种输出的字典
        """
        extract_features = self.feature_extractor(waveform)
        quantized_features, perplexity = self.quantizer(extract_features, mask_time_indices)

        hidden_states = self.feature_projection(extract_features)

        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        hidden_states = self.dropout_features(hidden_states)
        encoder_output = self.encoder(hidden_states, attention_mask)

        return {
            "last_hidden_state": encoder_output,
            "extract_features": extract_features,
            "quantized_features": quantized_features,
            "perplexity": perplexity
        }

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """仅提取特征 (用于下游任务)"""
        extract_features = self.feature_extractor(waveform)
        hidden_states = self.feature_projection(extract_features)
        encoder_output = self.encoder(hidden_states)
        return encoder_output


class Wav2Vec2ForPreTraining(nn.Module):
    """
    Wav2Vec2 预训练模型

    实现对比学习预训练目标:
    - 对比损失: 区分正确的量化目标和负样本
    - 多样性损失: 鼓励码本的均匀使用
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.Tensor] = None
    ) -> dict:
        """
        预训练前向传播

        Args:
            waveform: 原始波形 [batch, time]
            attention_mask: 注意力掩码
            mask_time_indices: 时间遮蔽索引

        Returns:
            包含损失和输出的字典
        """
        if mask_time_indices is None:
            batch_size = waveform.size(0)
            extract_features = self.wav2vec2.feature_extractor(waveform)
            seq_len = extract_features.size(1)
            mask_time_indices = compute_mask_indices(
                (batch_size, seq_len),
                self.config.mask_time_prob,
                self.config.mask_time_length,
                self.config.mask_time_min_masks,
                waveform.device
            )

        outputs = self.wav2vec2(waveform, attention_mask, mask_time_indices)

        transformer_features = self.wav2vec2.project_hid(outputs["last_hidden_state"])
        quantized_features = self.wav2vec2.project_q(outputs["quantized_features"])

        contrastive_loss = self._compute_contrastive_loss(
            transformer_features,
            quantized_features,
            mask_time_indices
        )

        num_codevectors = self.config.num_codevector_groups * self.config.num_codevectors_per_group
        diversity_loss = (num_codevectors - outputs["perplexity"]) / num_codevectors

        loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss,
            "diversity_loss": diversity_loss,
            "perplexity": outputs["perplexity"],
            "last_hidden_state": outputs["last_hidden_state"]
        }

    def _compute_contrastive_loss(
        self,
        transformer_features: torch.Tensor,
        quantized_features: torch.Tensor,
        mask_time_indices: torch.Tensor
    ) -> torch.Tensor:
        """计算对比损失，区分正确的量化目标和负样本"""
        batch_size, seq_len, hidden_size = transformer_features.shape

        transformer_features = transformer_features[mask_time_indices]
        quantized_features = quantized_features[mask_time_indices]

        num_masked = transformer_features.size(0)
        if num_masked == 0:
            # 返回需要梯度的零张量以保持计算图
            return transformer_features.sum() * 0.0

        negative_indices = torch.randint(
            0, num_masked, (num_masked, self.config.num_negatives),
            device=transformer_features.device
        )
        negatives = quantized_features[negative_indices]

        targets = torch.cat([quantized_features.unsqueeze(1), negatives], dim=1)

        logits = F.cosine_similarity(
            transformer_features.unsqueeze(1),
            targets,
            dim=-1
        ) / self.config.contrastive_logits_temperature

        labels = torch.zeros(num_masked, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss


class Wav2Vec2ForCTC(nn.Module):
    """
    Wav2Vec2 用于 CTC 语音识别

    在预训练的 Wav2Vec2 基础上添加 CTC 头，
    用于端到端语音识别任务。
    """

    def __init__(self, config: Wav2Vec2Config, vocab_size: int = 32):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None
    ) -> dict:
        """
        CTC 前向传播

        Args:
            waveform: 原始波形 [batch, time]
            attention_mask: 注意力掩码
            labels: 目标标签 [batch, label_len]
            label_lengths: 标签长度 [batch]

        Returns:
            包含 logits 和可选损失的字典
        """
        outputs = self.wav2vec2(waveform, attention_mask)
        hidden_states = outputs["last_hidden_state"]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1),
                dtype=torch.long, device=logits.device
            )

            loss = F.ctc_loss(
                log_probs, labels, input_lengths, label_lengths,
                blank=0, reduction="mean", zero_infinity=True
            )

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": hidden_states
        }

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        贪婪解码转录

        Args:
            waveform: 原始波形

        Returns:
            预测的 token 序列
        """
        outputs = self.forward(waveform)
        logits = outputs["logits"]
        predicted_ids = torch.argmax(logits, dim=-1)

        # 简单的 CTC 解码 (去除重复和空白)
        decoded = []
        for seq in predicted_ids:
            prev = -1
            result = []
            for token in seq:
                if token != 0 and token != prev:  # 0 是空白符
                    result.append(token.item())
                prev = token
            decoded.append(result)

        return decoded


class Wav2Vec2ForSequenceClassification(nn.Module):
    """
    Wav2Vec2 用于序列分类

    用于音频分类任务，如情感识别、说话人识别等。
    """

    def __init__(self, config: Wav2Vec2Config, num_labels: int = 2):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        分类前向传播

        Args:
            waveform: 原始波形 [batch, time]
            attention_mask: 注意力掩码
            labels: 分类标签 [batch]

        Returns:
            包含 logits 和可选损失的字典
        """
        outputs = self.wav2vec2(waveform, attention_mask)
        hidden_states = outputs["last_hidden_state"]

        # 平均池化
        pooled_output = hidden_states.mean(dim=1)
        pooled_output = self.projector(pooled_output)
        pooled_output = torch.tanh(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "pooled_output": pooled_output
        }


def create_wav2vec2_model(size: str = "base") -> Wav2Vec2Model:
    """
    创建预定义大小的 Wav2Vec2 模型

    Args:
        size: 模型大小 ("tiny", "base", "large")

    Returns:
        Wav2Vec2Model 实例
    """
    configs = {
        "tiny": Wav2Vec2Config(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            conv_dim=(256, 256, 256, 256, 256, 256, 256),
        ),
        "base": Wav2Vec2Config(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        ),
        "large": Wav2Vec2Config(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")

    return Wav2Vec2Model(configs[size])


def create_wav2vec2_for_ctc(
    size: str = "base",
    vocab_size: int = 32
) -> Wav2Vec2ForCTC:
    """创建用于 CTC 的 Wav2Vec2 模型"""
    configs = {
        "tiny": Wav2Vec2Config(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            conv_dim=(256, 256, 256, 256, 256, 256, 256),
        ),
        "base": Wav2Vec2Config(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        ),
        "large": Wav2Vec2Config(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        ),
    }

    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")

    return Wav2Vec2ForCTC(configs[size], vocab_size)
