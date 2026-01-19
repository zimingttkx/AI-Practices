"""
4M: Massively Multimodal Masked Modeling

本模块实现 4M 风格的多模态统一模型。

=== 核心架构 ===

4M 通过统一的 tokenization 将所有模态转换为离散 token，实现任意模态到任意模态的生成。

1. 模态 Tokenizer
   - VQ-VAE 将连续信号编码为离散 token
   - 支持 RGB、Depth、Semantic、Normal 等模态

2. Transformer 编码器-解码器
   - 编码器处理 masked input
   - 解码器生成任意目标模态

=== 参考文献 ===

Bachmann et al. "4M: Massively Multimodal Masked Modeling" NeurIPS 2023
https://arxiv.org/abs/2312.06647
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FourMConfig:
    """4M 模型配置"""

    # 图像配置
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # VQ-VAE 配置
    codebook_size: int = 8192
    codebook_dim: int = 256
    vq_commitment_cost: float = 0.25
    vq_decay: float = 0.99

    # Transformer 配置
    d_model: int = 768
    n_heads: int = 12
    n_encoder_layers: int = 12
    n_decoder_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1

    # 模态配置
    modalities: list[str] = field(default_factory=lambda: ["rgb", "depth", "semantic", "normal"])
    num_semantic_classes: int = 150

    # 训练配置
    mask_ratio: float = 0.75

    def __post_init__(self):
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
        self.num_patches = (self.image_size // self.patch_size) ** 2


class VectorQuantizer(nn.Module):
    """向量量化器 - EMA 更新的 Codebook"""

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook embeddings
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # EMA 更新的统计量
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding_avg", self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        向量量化

        Args:
            z: 输入特征 [B, H, W, D] 或 [B, N, D]

        Returns:
            quantized: 量化后的特征
            indices: codebook 索引
            loss: VQ 损失
        """
        input_shape = z.shape

        # 展平为 [B*N, D]
        flat_z = z.reshape(-1, self.codebook_dim)

        # 计算距离: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        # 找最近的 codebook entry
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view(input_shape)

        # EMA 更新 (仅训练时)
        if self.training:
            self._ema_update(flat_z, indices)

        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, indices.view(input_shape[:-1]), loss

    def _ema_update(self, flat_z: torch.Tensor, indices: torch.Tensor):
        """EMA 更新 codebook"""
        with torch.no_grad():
            # 统计每个 cluster 的样本数
            encodings = F.one_hot(indices, self.codebook_size).float()
            cluster_size = encodings.sum(0)

            # 更新 EMA cluster size
            self.ema_cluster_size.data.mul_(self.decay).add_(
                cluster_size, alpha=1 - self.decay
            )

            # 更新 EMA embedding average
            embedding_sum = encodings.t() @ flat_z
            self.ema_embedding_avg.data.mul_(self.decay).add_(
                embedding_sum, alpha=1 - self.decay
            )

            # 归一化得到新的 embedding
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon) * n
            )
            self.embedding.weight.data.copy_(
                self.ema_embedding_avg / cluster_size.unsqueeze(1)
            )

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """根据索引获取 codebook entry"""
        return self.embedding(indices)


class VQVAEEncoder(nn.Module):
    """VQ-VAE 编码器 - CNN 下采样"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        codebook_dim: int = 256,
        num_downsamples: int = 4
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]

        ch = hidden_channels
        for _i in range(num_downsamples):
            out_ch = min(ch * 2, 512)
            layers.extend([
                nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            ch = out_ch

        layers.append(nn.Conv2d(ch, codebook_dim, kernel_size=1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码图像为特征图"""
        return self.encoder(x).permute(0, 2, 3, 1)  # [B, H, W, D]


class VQVAEDecoder(nn.Module):
    """VQ-VAE 解码器 - CNN 上采样"""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 128,
        codebook_dim: int = 256,
        num_upsamples: int = 4
    ):
        super().__init__()

        ch = min(hidden_channels * (2 ** num_upsamples), 512)
        layers = [nn.Conv2d(codebook_dim, ch, kernel_size=1)]

        for _i in range(num_upsamples):
            out_ch = max(ch // 2, hidden_channels)
            layers.extend([
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ])
            ch = out_ch

        layers.append(nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """从特征图解码图像"""
        z = z.permute(0, 3, 1, 2)  # [B, D, H, W]
        return self.decoder(z)


class ModalityTokenizer(nn.Module):
    """多模态统一分词器"""

    def __init__(self, config: FourMConfig):
        super().__init__()
        self.config = config
        self.modalities = config.modalities

        # 为每种模态创建 VQ-VAE
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.quantizers = nn.ModuleDict()

        for modality in self.modalities:
            in_ch = self._get_modality_channels(modality)
            out_ch = in_ch

            self.encoders[modality] = VQVAEEncoder(
                in_channels=in_ch,
                codebook_dim=config.codebook_dim
            )
            self.decoders[modality] = VQVAEDecoder(
                out_channels=out_ch,
                codebook_dim=config.codebook_dim
            )
            self.quantizers[modality] = VectorQuantizer(
                codebook_size=config.codebook_size,
                codebook_dim=config.codebook_dim,
                commitment_cost=config.vq_commitment_cost,
                decay=config.vq_decay
            )

    def _get_modality_channels(self, modality: str) -> int:
        """获取模态的通道数"""
        channels_map = {
            "rgb": 3,
            "depth": 1,
            "normal": 3,
            "semantic": self.config.num_semantic_classes,
        }
        return channels_map.get(modality, 3)

    def tokenize(
        self,
        x: torch.Tensor,
        modality: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将输入转换为离散 token

        Args:
            x: 输入数据 [B, C, H, W]
            modality: 模态名称

        Returns:
            tokens: 离散 token 索引 [B, h, w]
            vq_loss: VQ 损失
        """
        z = self.encoders[modality](x)
        quantized, tokens, vq_loss = self.quantizers[modality](z)
        return tokens, vq_loss

    def detokenize(
        self,
        tokens: torch.Tensor,
        modality: str
    ) -> torch.Tensor:
        """
        将 token 转换回连续信号

        Args:
            tokens: 离散 token 索引 [B, h, w]
            modality: 模态名称

        Returns:
            重建的数据 [B, C, H, W]
        """
        quantized = self.quantizers[modality].get_codebook_entry(tokens)
        return self.decoders[modality](quantized)

    def encode(
        self,
        x: torch.Tensor,
        modality: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码并量化"""
        z = self.encoders[modality](x)
        quantized, tokens, vq_loss = self.quantizers[modality](z)
        return quantized, tokens, vq_loss

    def decode(
        self,
        quantized: torch.Tensor,
        modality: str
    ) -> torch.Tensor:
        """从量化特征解码"""
        return self.decoders[modality](quantized)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
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
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        if self.is_causal:
            seq_len = query.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device), diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
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


class FourMEncoderLayer(nn.Module):
    """4M 编码器层"""

    def __init__(self, config: FourMConfig):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.self_attn_norm = nn.LayerNorm(config.d_model)

        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 自注意力
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x

        # 前馈网络
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class FourMDecoderLayer(nn.Module):
    """4M 解码器层"""

    def __init__(self, config: FourMConfig):
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

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 自注意力
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x

        # 交叉注意力
        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, encoder_output, encoder_output, encoder_mask)
        x = self.dropout(x)
        x = residual + x

        # 前馈网络
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class FourMEncoder(nn.Module):
    """4M Transformer 编码器"""

    def __init__(self, config: FourMConfig):
        super().__init__()
        self.config = config

        # Token embedding (从 codebook indices)
        self.token_embedding = nn.Embedding(config.codebook_size, config.d_model)

        # 模态 embedding
        self.modality_embedding = nn.Embedding(len(config.modalities), config.d_model)

        # 位置 embedding
        max_seq_len = config.num_patches * len(config.modalities)
        self.position_embedding = nn.Embedding(max_seq_len, config.d_model)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Transformer 层
        self.layers = nn.ModuleList([
            FourMEncoderLayer(config) for _ in range(config.n_encoder_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        tokens: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """
        编码多模态 token

        Args:
            tokens: 每种模态的 token {modality: [B, N]}
            mask_dict: 每种模态的 mask {modality: [B, N]} (True = masked)

        Returns:
            编码器输出 [B, total_seq_len, d_model]
        """
        batch_size = list(tokens.values())[0].size(0)
        device = list(tokens.values())[0].device

        all_embeddings = []
        position_offset = 0

        for i, modality in enumerate(self.config.modalities):
            if modality not in tokens:
                continue

            modal_tokens = tokens[modality]  # [B, N]
            seq_len = modal_tokens.size(1)

            # Token embedding
            token_emb = self.token_embedding(modal_tokens)  # [B, N, D]

            # 应用 mask
            if mask_dict is not None and modality in mask_dict:
                mask = mask_dict[modality].unsqueeze(-1)  # [B, N, 1]
                token_emb = torch.where(
                    mask,
                    self.mask_token.expand(batch_size, seq_len, -1),
                    token_emb
                )

            # 模态 embedding
            modality_emb = self.modality_embedding(
                torch.full((batch_size, seq_len), i, device=device)
            )

            # 位置 embedding
            positions = torch.arange(
                position_offset, position_offset + seq_len, device=device
            ).unsqueeze(0).expand(batch_size, -1)
            position_emb = self.position_embedding(positions)

            # 组合
            embeddings = token_emb + modality_emb + position_emb
            all_embeddings.append(embeddings)

            position_offset += seq_len

        # 拼接所有模态
        x = torch.cat(all_embeddings, dim=1)  # [B, total_seq_len, D]
        x = self.dropout(x)

        # Transformer 层
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class FourMDecoder(nn.Module):
    """4M Transformer 解码器"""

    def __init__(self, config: FourMConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.codebook_size, config.d_model)

        # 模态 embedding
        self.modality_embedding = nn.Embedding(len(config.modalities), config.d_model)

        # 位置 embedding
        self.position_embedding = nn.Embedding(config.num_patches, config.d_model)

        # Start token
        self.start_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Transformer 层
        self.layers = nn.ModuleList([
            FourMDecoderLayer(config) for _ in range(config.n_decoder_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # 输出投影
        self.output_proj = nn.Linear(config.d_model, config.codebook_size)

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_tokens: torch.Tensor,
        target_modality_idx: int
    ) -> torch.Tensor:
        """
        解码生成目标模态

        Args:
            encoder_output: 编码器输出 [B, enc_seq_len, D]
            target_tokens: 目标 token (teacher forcing) [B, N]
            target_modality_idx: 目标模态索引

        Returns:
            logits [B, N, codebook_size]
        """
        batch_size = target_tokens.size(0)
        seq_len = target_tokens.size(1)
        device = target_tokens.device

        # Token embedding (shifted right for autoregressive)
        token_emb = self.token_embedding(target_tokens)  # [B, N, D]

        # 添加 start token
        start = self.start_token.expand(batch_size, -1, -1)
        token_emb = torch.cat([start, token_emb[:, :-1]], dim=1)

        # 模态 embedding
        modality_emb = self.modality_embedding(
            torch.full((batch_size, seq_len), target_modality_idx, device=device)
        )

        # 位置 embedding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(positions)

        # 组合
        x = token_emb + modality_emb + position_emb
        x = self.dropout(x)

        # Transformer 层
        for layer in self.layers:
            x = layer(x, encoder_output)

        x = self.norm(x)
        logits = self.output_proj(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        target_modality_idx: int,
        max_len: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """自回归生成"""
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        # 从 start token 开始
        generated = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        for _ in range(max_len):
            if generated.size(1) == 0:
                # 第一步：只用 start token
                x = self.start_token.expand(batch_size, -1, -1)
            else:
                # 后续步骤
                token_emb = self.token_embedding(generated)
                start = self.start_token.expand(batch_size, -1, -1)
                x = torch.cat([start, token_emb], dim=1)

            seq_len = x.size(1)

            # 添加 embedding
            modality_emb = self.modality_embedding(
                torch.full((batch_size, seq_len), target_modality_idx, device=device)
            )
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            position_emb = self.position_embedding(positions)

            x = x + modality_emb + position_emb

            # Transformer
            for layer in self.layers:
                x = layer(x, encoder_output)

            x = self.norm(x)
            logits = self.output_proj(x[:, -1:, :])  # 只取最后一个位置

            # 采样
            if temperature == 0:
                next_token = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated


class FourMLoss(nn.Module):
    """4M 损失函数"""

    def __init__(self, config: FourMConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vq_loss: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        计算总损失

        Args:
            logits: 预测 logits [B, N, codebook_size]
            targets: 目标 token [B, N]
            vq_loss: VQ 损失
            mask: 只计算 masked 位置的损失 [B, N]

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失
        """
        # 重建损失 (交叉熵)
        recon_loss = F.cross_entropy(
            logits.view(-1, self.config.codebook_size),
            targets.view(-1),
            reduction="none"
        ).view(targets.shape)

        if mask is not None:
            recon_loss = (recon_loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            recon_loss = recon_loss.mean()

        # 总损失
        total_loss = recon_loss + vq_loss

        loss_dict = {
            "total": total_loss,
            "recon": recon_loss,
            "vq": vq_loss
        }

        return total_loss, loss_dict


class FourM(nn.Module):
    """4M: Massively Multimodal Masked Modeling"""

    def __init__(self, config: FourMConfig):
        super().__init__()
        self.config = config

        # 模态 Tokenizer
        self.tokenizer = ModalityTokenizer(config)

        # Transformer 编码器-解码器
        self.encoder = FourMEncoder(config)
        self.decoder = FourMDecoder(config)

        # 损失函数
        self.loss_fn = FourMLoss(config)

    def tokenize(
        self,
        inputs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        将多模态输入转换为 token

        Args:
            inputs: {modality: [B, C, H, W]}

        Returns:
            tokens: {modality: [B, N]}
            total_vq_loss: 总 VQ 损失
        """
        tokens = {}
        total_vq_loss = 0.0

        for modality, x in inputs.items():
            modal_tokens, vq_loss = self.tokenizer.tokenize(x, modality)
            tokens[modality] = modal_tokens.flatten(1)  # [B, h*w]
            total_vq_loss = total_vq_loss + vq_loss

        return tokens, total_vq_loss

    def detokenize(
        self,
        tokens: dict[str, torch.Tensor],
        spatial_shape: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """
        将 token 转换回多模态输出

        Args:
            tokens: {modality: [B, N]}
            spatial_shape: (h, w)

        Returns:
            outputs: {modality: [B, C, H, W]}
        """
        outputs = {}
        h, w = spatial_shape

        for modality, modal_tokens in tokens.items():
            modal_tokens = modal_tokens.view(-1, h, w)
            outputs[modality] = self.tokenizer.detokenize(modal_tokens, modality)

        return outputs

    def create_random_mask(
        self,
        tokens: dict[str, torch.Tensor],
        mask_ratio: float = None
    ) -> dict[str, torch.Tensor]:
        """创建随机 mask"""
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio

        mask_dict = {}
        for modality, modal_tokens in tokens.items():
            batch_size, seq_len = modal_tokens.shape
            num_mask = int(seq_len * mask_ratio)

            # 随机选择要 mask 的位置
            noise = torch.rand(batch_size, seq_len, device=modal_tokens.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=modal_tokens.device)

            for b in range(batch_size):
                mask[b, ids_shuffle[b, :num_mask]] = True

            mask_dict[modality] = mask

        return mask_dict

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        target_modality: str,
        mask_ratio: float = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            inputs: 多模态输入 {modality: [B, C, H, W]}
            target_modality: 目标模态
            mask_ratio: mask 比例

        Returns:
            loss: 总损失
            loss_dict: 各项损失
        """
        # Tokenize
        tokens, vq_loss = self.tokenize(inputs)

        # 创建 mask
        mask_dict = self.create_random_mask(tokens, mask_ratio)

        # 编码
        encoder_output = self.encoder(tokens, mask_dict)

        # 解码目标模态
        target_idx = self.config.modalities.index(target_modality)
        target_tokens = tokens[target_modality]
        logits = self.decoder(encoder_output, target_tokens, target_idx)

        # 计算损失 (只在 masked 位置)
        target_mask = mask_dict[target_modality].float()
        loss, loss_dict = self.loss_fn(logits, target_tokens, vq_loss, target_mask)

        return loss, loss_dict

    def encode(
        self,
        inputs: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """编码多模态输入"""
        tokens, _ = self.tokenize(inputs)
        return self.encoder(tokens, mask_dict)

    @torch.no_grad()
    def generate(
        self,
        inputs: dict[str, torch.Tensor],
        target_modality: str,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        生成目标模态

        Args:
            inputs: 条件输入 {modality: [B, C, H, W]}
            target_modality: 目标模态
            temperature: 采样温度

        Returns:
            生成的图像 [B, C, H, W]
        """
        # Tokenize 输入
        tokens, _ = self.tokenize(inputs)

        # 编码
        encoder_output = self.encoder(tokens, None)

        # 生成
        target_idx = self.config.modalities.index(target_modality)
        h = w = self.config.image_size // self.config.patch_size // 16  # VQ-VAE 下采样
        max_len = h * w

        generated_tokens = self.decoder.generate(
            encoder_output, target_idx, max_len, temperature
        )

        # Detokenize
        generated_tokens = generated_tokens.view(-1, h, w)
        output = self.tokenizer.detokenize(generated_tokens, target_modality)

        return output

    @torch.no_grad()
    def transfer(
        self,
        source: torch.Tensor,
        source_modality: str,
        target_modality: str,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        模态转换

        Args:
            source: 源模态数据 [B, C, H, W]
            source_modality: 源模态名称
            target_modality: 目标模态名称
            temperature: 采样温度

        Returns:
            目标模态数据 [B, C, H, W]
        """
        inputs = {source_modality: source}
        return self.generate(inputs, target_modality, temperature)


def create_fourm_model(model_size: str = "base") -> FourM:
    """
    创建预定义大小的 4M 模型

    Args:
        model_size: 模型大小 ("tiny", "small", "base")

    Returns:
        4M 模型实例
    """
    configs = {
        "tiny": FourMConfig(
            image_size=64,
            patch_size=8,
            codebook_size=4096,
            codebook_dim=128,
            d_model=256,
            n_heads=4,
            n_encoder_layers=4,
            n_decoder_layers=4,
            d_ff=1024
        ),
        "small": FourMConfig(
            image_size=128,
            patch_size=16,
            codebook_size=8192,
            codebook_dim=256,
            d_model=512,
            n_heads=8,
            n_encoder_layers=6,
            n_decoder_layers=6,
            d_ff=2048
        ),
        "base": FourMConfig(
            image_size=224,
            patch_size=16,
            codebook_size=8192,
            codebook_dim=256,
            d_model=768,
            n_heads=12,
            n_encoder_layers=12,
            n_decoder_layers=12,
            d_ff=3072
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return FourM(configs[model_size])
