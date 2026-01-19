"""
ImageBind: One Embedding Space To Bind Them All

核心思想:
- 以图像为锚点，将 6 种模态对齐到统一嵌入空间
- 通过图像-X 配对数据训练，实现跨模态涌现能力
- 支持零样本跨模态检索和分类

支持模态:
- 图像 (Image)
- 文本 (Text)
- 音频 (Audio)
- 深度图 (Depth)
- 热力图 (Thermal)
- IMU 传感器 (IMU)

参考文献:
1. ImageBind: One Embedding Space To Bind Them All
   https://arxiv.org/abs/2305.05665
2. Learning Transferable Visual Models From Natural Language Supervision (CLIP)
   https://arxiv.org/abs/2103.00020
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityType(str, Enum):
    """支持的模态类型"""

    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    DEPTH = "depth"
    THERMAL = "thermal"
    IMU = "imu"


@dataclass
class ImageBindConfig:
    """ImageBind 配置"""

    embed_dim: int = 768
    vision_embed_dim: int = 1024
    text_embed_dim: int = 768
    audio_embed_dim: int = 768

    # Vision (ViT)
    image_size: int = 224
    patch_size: int = 14
    vision_layers: int = 12
    vision_heads: int = 16

    # Text
    text_vocab_size: int = 32000
    text_max_length: int = 77
    text_layers: int = 12
    text_heads: int = 12

    # Audio
    audio_num_mel_bins: int = 128
    audio_target_length: int = 204
    audio_patch_size: int = 16
    audio_stride: int = 10
    audio_layers: int = 12
    audio_heads: int = 12

    # Depth / Thermal (same as vision)
    depth_patch_size: int = 14
    thermal_patch_size: int = 14

    # IMU
    imu_input_dim: int = 6
    imu_seq_length: int = 2000
    imu_patch_size: int = 50
    imu_layers: int = 6
    imu_heads: int = 8

    # Training
    dropout: float = 0.0
    temperature: float = 0.07
    learnable_temperature: bool = True

    # Modality-specific projection dims
    modality_embed_dims: dict[str, int] = field(default_factory=lambda: {
        "image": 1024,
        "text": 768,
        "audio": 768,
        "depth": 1024,
        "thermal": 1024,
        "imu": 512,
    })


class PatchEmbedding(nn.Module):
    """通用 Patch 嵌入层"""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if stride is None:
            stride = patch_size
        elif isinstance(stride, int):
            stride = (stride, stride)

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 块"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoder(nn.Module):
    """ViT 图像编码器"""

    def __init__(self, config: ImageBindConfig):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.patch_size) ** 2

        self.patch_embed = PatchEmbedding(
            in_channels=3,
            embed_dim=config.vision_embed_dim,
            patch_size=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.vision_embed_dim)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.vision_embed_dim,
                config.vision_heads,
                dropout=config.dropout,
            )
            for _ in range(config.vision_layers)
        ])
        self.norm = nn.LayerNorm(config.vision_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


class TextEncoder(nn.Module):
    """Transformer 文本编码器"""

    def __init__(self, config: ImageBindConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.text_vocab_size, config.text_embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.text_max_length, config.text_embed_dim)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.text_embed_dim,
                config.text_heads,
                dropout=config.dropout,
            )
            for _ in range(config.text_layers)
        ])
        self.norm = nn.LayerNorm(config.text_embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :x.shape[1]]

        if attention_mask is not None:
            causal_mask = torch.triu(
                torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1
            ).bool()
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_mask = causal_mask | padding_mask
        else:
            attn_mask = None

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.norm(x)

        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(x.shape[0], device=x.device)
            return x[batch_indices, seq_lens]
        return x[:, -1]


class AudioEncoder(nn.Module):
    """音频频谱编码器"""

    def __init__(self, config: ImageBindConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(
            in_channels=1,
            embed_dim=config.audio_embed_dim,
            patch_size=(config.audio_patch_size, config.audio_patch_size),
            stride=(config.audio_stride, config.audio_stride),
        )

        num_patches_freq = (config.audio_num_mel_bins - config.audio_patch_size) // config.audio_stride + 1
        num_patches_time = (config.audio_target_length - config.audio_patch_size) // config.audio_stride + 1
        num_patches = num_patches_freq * num_patches_time

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.audio_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.audio_embed_dim)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.audio_embed_dim,
                config.audio_heads,
                dropout=config.dropout,
            )
            for _ in range(config.audio_layers)
        ])
        self.norm = nn.LayerNorm(config.audio_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :x.shape[1]]
        else:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
            ).transpose(1, 2)
            x = x + pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


class DepthEncoder(nn.Module):
    """深度图编码器 (与图像编码器结构相同)"""

    def __init__(self, config: ImageBindConfig):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.depth_patch_size) ** 2

        self.patch_embed = PatchEmbedding(
            in_channels=1,
            embed_dim=config.vision_embed_dim,
            patch_size=config.depth_patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.vision_embed_dim)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.vision_embed_dim,
                config.vision_heads,
                dropout=config.dropout,
            )
            for _ in range(config.vision_layers)
        ])
        self.norm = nn.LayerNorm(config.vision_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


class ThermalEncoder(nn.Module):
    """热力图编码器"""

    def __init__(self, config: ImageBindConfig):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.thermal_patch_size) ** 2

        self.patch_embed = PatchEmbedding(
            in_channels=1,
            embed_dim=config.vision_embed_dim,
            patch_size=config.thermal_patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.vision_embed_dim)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.vision_embed_dim,
                config.vision_heads,
                dropout=config.dropout,
            )
            for _ in range(config.vision_layers)
        ])
        self.norm = nn.LayerNorm(config.vision_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


class IMUEncoder(nn.Module):
    """IMU 传感器编码器"""

    def __init__(self, config: ImageBindConfig):
        super().__init__()
        self.config = config
        imu_embed_dim = config.modality_embed_dims.get("imu", 512)
        num_patches = config.imu_seq_length // config.imu_patch_size

        self.patch_embed = nn.Conv1d(
            config.imu_input_dim, imu_embed_dim,
            kernel_size=config.imu_patch_size,
            stride=config.imu_patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, imu_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, imu_embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(
                imu_embed_dim,
                config.imu_heads,
                dropout=config.dropout,
            )
            for _ in range(config.imu_layers)
        ])
        self.norm = nn.LayerNorm(imu_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.transpose(1, 2)
        x = self.patch_embed(x)
        x = x.transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :x.shape[1]]
        else:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
            ).transpose(1, 2)
            x = x + pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]


class ModalityProjector(nn.Module):
    """模态投影到共享嵌入空间"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class ImageBindLoss(nn.Module):
    """ImageBind 对比学习损失 (InfoNCE)"""

    def __init__(self, temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        if learnable:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        else:
            self.register_buffer(
                "log_temperature", torch.log(torch.tensor(1.0 / temperature))
            )
        self.learnable = learnable

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.log_temperature)

    def forward(
        self,
        anchor_embeds: torch.Tensor,
        positive_embeds: torch.Tensor,
    ) -> torch.Tensor:
        anchor_embeds = F.normalize(anchor_embeds, dim=-1)
        positive_embeds = F.normalize(positive_embeds, dim=-1)

        logits = anchor_embeds @ positive_embeds.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss_a2p = F.cross_entropy(logits, labels)
        loss_p2a = F.cross_entropy(logits.T, labels)

        return (loss_a2p + loss_p2a) / 2


class ImageBind(nn.Module):
    """ImageBind 多模态统一嵌入模型"""

    def __init__(self, config: ImageBindConfig):
        super().__init__()
        self.config = config

        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        self.thermal_encoder = ThermalEncoder(config)
        self.imu_encoder = IMUEncoder(config)

        self.image_proj = ModalityProjector(config.vision_embed_dim, config.embed_dim)
        self.text_proj = ModalityProjector(config.text_embed_dim, config.embed_dim)
        self.audio_proj = ModalityProjector(config.audio_embed_dim, config.embed_dim)
        self.depth_proj = ModalityProjector(config.vision_embed_dim, config.embed_dim)
        self.thermal_proj = ModalityProjector(config.vision_embed_dim, config.embed_dim)
        imu_embed_dim = config.modality_embed_dims.get("imu", 512)
        self.imu_proj = ModalityProjector(imu_embed_dim, config.embed_dim)

        self.loss_fn = ImageBindLoss(
            temperature=config.temperature,
            learnable=config.learnable_temperature,
        )

        self._encoders: dict[str, nn.Module] = {
            ModalityType.IMAGE: self.image_encoder,
            ModalityType.TEXT: self.text_encoder,
            ModalityType.AUDIO: self.audio_encoder,
            ModalityType.DEPTH: self.depth_encoder,
            ModalityType.THERMAL: self.thermal_encoder,
            ModalityType.IMU: self.imu_encoder,
        }
        self._projectors: dict[str, nn.Module] = {
            ModalityType.IMAGE: self.image_proj,
            ModalityType.TEXT: self.text_proj,
            ModalityType.AUDIO: self.audio_proj,
            ModalityType.DEPTH: self.depth_proj,
            ModalityType.THERMAL: self.thermal_proj,
            ModalityType.IMU: self.imu_proj,
        }

    def encode(
        self,
        modality: ModalityType | str,
        x: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(modality, str):
            modality = ModalityType(modality)

        encoder = self._encoders[modality]
        projector = self._projectors[modality]

        features = encoder(x, **kwargs) if modality == ModalityType.TEXT else encoder(x)

        embeddings = projector(features)

        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.encode(ModalityType.IMAGE, images, normalize=normalize)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.encode(
            ModalityType.TEXT, input_ids,
            normalize=normalize, attention_mask=attention_mask
        )

    def encode_audio(self, audio: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.encode(ModalityType.AUDIO, audio, normalize=normalize)

    def encode_depth(self, depth: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.encode(ModalityType.DEPTH, depth, normalize=normalize)

    def encode_thermal(self, thermal: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.encode(ModalityType.THERMAL, thermal, normalize=normalize)

    def encode_imu(self, imu: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.encode(ModalityType.IMU, imu, normalize=normalize)

    def compute_similarity(
        self,
        embeds_a: torch.Tensor,
        embeds_b: torch.Tensor,
    ) -> torch.Tensor:
        embeds_a = F.normalize(embeds_a, dim=-1)
        embeds_b = F.normalize(embeds_b, dim=-1)
        return embeds_a @ embeds_b.T

    def forward(
        self,
        anchor_modality: ModalityType | str,
        anchor_input: torch.Tensor,
        positive_modality: ModalityType | str,
        positive_input: torch.Tensor,
        anchor_kwargs: dict | None = None,
        positive_kwargs: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        anchor_kwargs = anchor_kwargs or {}
        positive_kwargs = positive_kwargs or {}

        anchor_embeds = self.encode(
            anchor_modality, anchor_input, normalize=False, **anchor_kwargs
        )
        positive_embeds = self.encode(
            positive_modality, positive_input, normalize=False, **positive_kwargs
        )

        loss = self.loss_fn(anchor_embeds, positive_embeds)

        return {
            "loss": loss,
            "anchor_embeds": F.normalize(anchor_embeds, dim=-1),
            "positive_embeds": F.normalize(positive_embeds, dim=-1),
        }

    @torch.no_grad()
    def zero_shot_classify(
        self,
        modality: ModalityType | str,
        inputs: torch.Tensor,
        class_embeddings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        embeddings = self.encode(modality, inputs, normalize=True, **kwargs)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        logits = embeddings @ class_embeddings.T
        return logits

    @torch.no_grad()
    def retrieve(
        self,
        query_modality: ModalityType | str,
        query_input: torch.Tensor,
        gallery_modality: ModalityType | str,
        gallery_inputs: torch.Tensor,
        top_k: int = 5,
        query_kwargs: dict | None = None,
        gallery_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_kwargs = query_kwargs or {}
        gallery_kwargs = gallery_kwargs or {}

        query_embeds = self.encode(
            query_modality, query_input, normalize=True, **query_kwargs
        )
        gallery_embeds = self.encode(
            gallery_modality, gallery_inputs, normalize=True, **gallery_kwargs
        )

        similarities = query_embeds @ gallery_embeds.T
        top_k = min(top_k, similarities.shape[1])
        scores, indices = similarities.topk(top_k, dim=1)

        return scores, indices


def create_imagebind_model(model_size: str = "base") -> ImageBind:
    """工厂函数"""
    configs = {
        "tiny": ImageBindConfig(
            embed_dim=256,
            vision_embed_dim=384,
            text_embed_dim=256,
            audio_embed_dim=256,
            vision_layers=6,
            vision_heads=6,
            text_layers=6,
            text_heads=4,
            audio_layers=6,
            audio_heads=4,
            imu_layers=4,
            imu_heads=4,
            modality_embed_dims={
                "image": 384, "text": 256, "audio": 256,
                "depth": 384, "thermal": 384, "imu": 256,
            },
        ),
        "small": ImageBindConfig(
            embed_dim=512,
            vision_embed_dim=768,
            text_embed_dim=512,
            audio_embed_dim=512,
            vision_layers=8,
            vision_heads=8,
            text_layers=8,
            text_heads=8,
            audio_layers=8,
            audio_heads=8,
            imu_layers=4,
            imu_heads=4,
            modality_embed_dims={
                "image": 768, "text": 512, "audio": 512,
                "depth": 768, "thermal": 768, "imu": 384,
            },
        ),
        "base": ImageBindConfig(),
        "large": ImageBindConfig(
            embed_dim=1024,
            vision_embed_dim=1280,
            text_embed_dim=1024,
            audio_embed_dim=1024,
            vision_layers=24,
            vision_heads=16,
            text_layers=24,
            text_heads=16,
            audio_layers=24,
            audio_heads=16,
            imu_layers=12,
            imu_heads=8,
            modality_embed_dims={
                "image": 1280, "text": 1024, "audio": 1024,
                "depth": 1280, "thermal": 1280, "imu": 768,
            },
        ),
    }
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}, choose from {list(configs.keys())}")
    return ImageBind(configs[model_size])
