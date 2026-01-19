"""
Unified-IO 2: 多模态统一输入输出模型

核心思想:
- 任意模态输入 (文本/图像/音频/视频)
- 统一表示空间与统一解码接口
- 统一任务入口 (分类/检索/生成)

参考文献:
1. Unified-IO 2: Scaling Autoregressive Multimodal Models with a Unified Vocabulary
   https://arxiv.org/abs/2312.17172
2. Unified-IO: A Unified Model for Vision, Language, and Multi-Task Learning
   https://arxiv.org/abs/2206.08916
3. ImageBind: One Embedding Space to Bind Them All
   https://arxiv.org/abs/2305.05665
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class Modality(str, Enum):
    """支持的模态类型"""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class TaskType(str, Enum):
    """统一任务类型"""

    CLASSIFICATION = "classification"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"


@dataclass
class UnifiedIOConfig:
    """Unified-IO 配置"""

    # 模型维度
    hidden_size: int = 768
    num_layers: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # 统一词表
    text_vocab_size: int = 32000
    max_text_length: int = 128

    # 图像 patch
    image_size: int = 224
    image_patch_size: int = 16
    image_channels: int = 3

    # 音频帧
    audio_channels: int = 1
    audio_patch_size: int = 16
    max_audio_length: int = 2048

    # 视频 patch
    video_frames: int = 8
    video_patch_size: int = 16
    video_channels: int = 3

    # 任务相关
    num_labels: int = 10
    pooling: str = "cls"  # "cls" or "mean"
    pad_token_id: int = 0
    eos_token_id: int = 2

    # 生成参数
    max_generate_length: int = 128


@dataclass
class MultimodalBatch:
    """统一输入批次"""

    text_input_ids: torch.Tensor | None = None  # [B, L]
    text_attention_mask: torch.Tensor | None = None  # [B, L]
    images: torch.Tensor | None = None  # [B, C, H, W]
    audio: torch.Tensor | None = None  # [B, T]
    video: torch.Tensor | None = None  # [B, C, T, H, W]
    labels: torch.Tensor | None = None  # [B] or [B, L]


class PatchEmbed2D(nn.Module):
    """图像 patch embedding"""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class PatchEmbed1D(nn.Module):
    """音频 patch embedding"""

    def __init__(self, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(1, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.proj(x)  # [B, D, T/P]
        x = x.transpose(1, 2)  # [B, N, D]
        return x


class PatchEmbed3D(nn.Module):
    """视频 patch embedding"""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.proj(x)  # [B, D, T, H/P, W/P]
        B, D, T, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H/P, W/P, D]
        x = x.reshape(B, T * Hp * Wp, D)  # [B, T*N, D]
        return x


class TextEmbedding(nn.Module):
    """文本嵌入"""

    def __init__(self, vocab_size: int, hidden_size: int, max_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        return x


class ModalityEmbedding(nn.Module):
    """模态类型嵌入"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(len(Modality), hidden_size)

    def forward(self, modality: Modality, length: int, batch_size: int) -> torch.Tensor:
        modality_id = list(Modality).index(modality)
        emb = self.embedding.weight[modality_id].view(1, 1, -1)
        return emb.expand(batch_size, length, -1)


class UnifiedEncoder(nn.Module):
    """统一编码器"""

    def __init__(self, config: UnifiedIOConfig):
        super().__init__()
        self.config = config
        self.text_embed = TextEmbedding(
            config.text_vocab_size, config.hidden_size, config.max_text_length
        )
        self.image_embed = PatchEmbed2D(
            config.image_channels, config.hidden_size, config.image_patch_size
        )
        self.audio_embed = PatchEmbed1D(config.hidden_size, config.audio_patch_size)
        self.video_embed = PatchEmbed3D(
            config.video_channels, config.hidden_size, config.video_patch_size
        )
        self.modality_embed = ModalityEmbedding(config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_size * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_size)

    def _build_tokens(
        self, batch: MultimodalBatch
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, tuple[int, int]]]:
        token_list = []
        mask_list = []
        spans: dict[str, tuple[int, int]] = {}

        batch_size = None
        if batch.text_input_ids is not None:
            batch_size = batch.text_input_ids.shape[0]
        elif batch.images is not None:
            batch_size = batch.images.shape[0]
        elif batch.audio is not None:
            batch_size = batch.audio.shape[0]
        elif batch.video is not None:
            batch_size = batch.video.shape[0]
        else:
            raise ValueError("至少提供一种模态输入")

        cls = self.cls_token.expand(batch_size, -1, -1)
        token_list.append(cls)
        mask_list.append(torch.ones(batch_size, 1, device=cls.device))
        spans["cls"] = (0, 1)
        offset = 1

        if batch.text_input_ids is not None:
            text_tokens = self.text_embed(batch.text_input_ids)
            if batch.text_attention_mask is None:
                text_mask = torch.ones(
                    batch.text_input_ids.shape[0],
                    batch.text_input_ids.shape[1],
                    device=text_tokens.device,
                )
            else:
                text_mask = batch.text_attention_mask.float()
            text_tokens = text_tokens + self.modality_embed(
                Modality.TEXT, text_tokens.shape[1], batch_size
            )
            token_list.append(text_tokens)
            mask_list.append(text_mask)
            spans["text"] = (offset, offset + text_tokens.shape[1])
            offset += text_tokens.shape[1]

        if batch.images is not None:
            image_tokens = self.image_embed(batch.images)
            image_tokens = image_tokens + self.modality_embed(
                Modality.IMAGE, image_tokens.shape[1], batch_size
            )
            token_list.append(image_tokens)
            mask_list.append(torch.ones(batch_size, image_tokens.shape[1], device=image_tokens.device))
            spans["image"] = (offset, offset + image_tokens.shape[1])
            offset += image_tokens.shape[1]

        if batch.audio is not None:
            audio_tokens = self.audio_embed(batch.audio)
            audio_tokens = audio_tokens + self.modality_embed(
                Modality.AUDIO, audio_tokens.shape[1], batch_size
            )
            token_list.append(audio_tokens)
            mask_list.append(torch.ones(batch_size, audio_tokens.shape[1], device=audio_tokens.device))
            spans["audio"] = (offset, offset + audio_tokens.shape[1])
            offset += audio_tokens.shape[1]

        if batch.video is not None:
            video_tokens = self.video_embed(batch.video)
            video_tokens = video_tokens + self.modality_embed(
                Modality.VIDEO, video_tokens.shape[1], batch_size
            )
            token_list.append(video_tokens)
            mask_list.append(torch.ones(batch_size, video_tokens.shape[1], device=video_tokens.device))
            spans["video"] = (offset, offset + video_tokens.shape[1])
            offset += video_tokens.shape[1]

        tokens = torch.cat(token_list, dim=1)
        attention_mask = torch.cat(mask_list, dim=1)
        return tokens, attention_mask, spans

    def forward(self, batch: MultimodalBatch) -> dict[str, torch.Tensor]:
        tokens, attention_mask, spans = self._build_tokens(batch)
        key_padding_mask = attention_mask == 0
        hidden = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        hidden = self.norm(hidden)
        return {
            "hidden_states": hidden,
            "attention_mask": attention_mask,
            "spans": spans,
        }


class UnifiedDecoder(nn.Module):
    """统一解码器 (文本生成为主)"""

    def __init__(self, config: UnifiedIOConfig):
        super().__init__()
        self.config = config
        self.embedding = TextEmbedding(
            config.text_vocab_size, config.hidden_size, config.max_text_length
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_size * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.lm_head = nn.Linear(config.hidden_size, config.text_vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(input_ids)
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
        ).bool()
        memory_key_padding = encoder_attention_mask == 0
        out = self.decoder(
            x,
            encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding,
        )
        return self.lm_head(out)


class UnifiedIO(nn.Module):
    """Unified-IO 统一模型"""

    def __init__(self, config: UnifiedIOConfig):
        super().__init__()
        self.config = config
        self.encoder = UnifiedEncoder(config)
        self.decoder = UnifiedDecoder(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

    def forward(
        self, batch: MultimodalBatch, task: TaskType = TaskType.CLASSIFICATION
    ) -> dict[str, torch.Tensor]:
        encoder_out = self.encoder(batch)
        hidden = encoder_out["hidden_states"]
        attention_mask = encoder_out["attention_mask"]

        if task == TaskType.CLASSIFICATION:
            pooled = self._pool(hidden, attention_mask)
            logits = self.classifier(pooled)
            loss = None
            if batch.labels is not None:
                loss = F.cross_entropy(logits, batch.labels)
            return {
                "logits": logits,
                "loss": loss,
                "pooled_embedding": pooled,
                "attention_mask": attention_mask,
            }

        if task == TaskType.RETRIEVAL:
            pooled = self._pool(hidden, attention_mask)
            pooled = F.normalize(pooled, dim=-1)
            return {
                "embeddings": pooled,
                "attention_mask": attention_mask,
            }

        if task == TaskType.GENERATION:
            if batch.text_input_ids is None:
                raise ValueError("生成任务需要 text_input_ids")
            logits = self.decoder(
                batch.text_input_ids,
                hidden,
                attention_mask,
            )
            loss = None
            if batch.labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch.labels.view(-1),
                    ignore_index=self.config.pad_token_id,
                )
            return {
                "logits": logits,
                "loss": loss,
                "attention_mask": attention_mask,
            }

        raise ValueError(f"未知任务类型: {task}")

    @torch.no_grad()
    def generate(
        self,
        batch: MultimodalBatch,
        max_length: int | None = None,
    ) -> torch.Tensor:
        encoder_out = self.encoder(batch)
        hidden = encoder_out["hidden_states"]
        attention_mask = encoder_out["attention_mask"]
        if batch.text_input_ids is None:
            raise ValueError("生成需要 text_input_ids 作为前缀")
        max_length = max_length or self.config.max_generate_length
        input_ids = batch.text_input_ids

        for _ in range(max_length - input_ids.shape[1]):
            logits = self.decoder(input_ids, hidden, attention_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if (next_token == self.config.eos_token_id).all():
                break
        return input_ids


def create_unified_io_model(model_size: str = "base") -> UnifiedIO:
    """工厂函数"""
    configs = {
        "tiny": UnifiedIOConfig(hidden_size=256, num_layers=4, num_heads=4),
        "small": UnifiedIOConfig(hidden_size=384, num_layers=6, num_heads=6),
        "base": UnifiedIOConfig(),
        "large": UnifiedIOConfig(hidden_size=1024, num_layers=12, num_heads=16),
    }
    if model_size not in configs:
        raise ValueError(f"未知模型大小: {model_size}")
    return UnifiedIO(configs[model_size])
