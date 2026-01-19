"""
Unified-IO 单元测试

测试覆盖:
- 配置与初始化
- 各模态 Patch Embedding
- 统一编码器
- 统一解码器
- 分类/检索/生成任务
- 工厂函数
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unified_io import (
    Modality,
    ModalityEmbedding,
    MultimodalBatch,
    PatchEmbed1D,
    PatchEmbed2D,
    PatchEmbed3D,
    TaskType,
    TextEmbedding,
    UnifiedDecoder,
    UnifiedEncoder,
    UnifiedIO,
    UnifiedIOConfig,
    create_unified_io_model,
)


class TestUnifiedIOConfig:
    """配置测试"""

    def test_default_config(self):
        config = UnifiedIOConfig()
        assert config.hidden_size == 768
        assert config.num_layers == 8
        assert config.num_heads == 12
        assert config.image_size == 224
        assert config.image_patch_size == 16

    def test_custom_config(self):
        config = UnifiedIOConfig(hidden_size=512, num_layers=6)
        assert config.hidden_size == 512
        assert config.num_layers == 6


class TestPatchEmbeddings:
    """Patch Embedding 测试"""

    @pytest.fixture
    def config(self):
        return UnifiedIOConfig(hidden_size=64, image_patch_size=16, audio_patch_size=16)

    def test_patch_embed_2d_shape(self, config):
        embed = PatchEmbed2D(3, config.hidden_size, config.image_patch_size)
        x = torch.randn(2, 3, 224, 224)
        out = embed(x)
        num_patches = (224 // 16) ** 2
        assert out.shape == (2, num_patches, config.hidden_size)

    def test_patch_embed_1d_shape(self, config):
        embed = PatchEmbed1D(config.hidden_size, config.audio_patch_size)
        x = torch.randn(2, 1024)
        out = embed(x)
        num_patches = 1024 // 16
        assert out.shape == (2, num_patches, config.hidden_size)

    def test_patch_embed_3d_shape(self, config):
        embed = PatchEmbed3D(3, config.hidden_size, config.video_patch_size)
        x = torch.randn(2, 3, 4, 64, 64)
        out = embed(x)
        num_patches_per_frame = (64 // 16) ** 2
        total_patches = 4 * num_patches_per_frame
        assert out.shape == (2, total_patches, config.hidden_size)


class TestTextEmbedding:
    """文本嵌入测试"""

    def test_text_embedding_shape(self):
        embed = TextEmbedding(vocab_size=1000, hidden_size=64, max_length=128)
        input_ids = torch.randint(0, 1000, (2, 20))
        out = embed(input_ids)
        assert out.shape == (2, 20, 64)


class TestModalityEmbedding:
    """模态嵌入测试"""

    def test_modality_embedding_shape(self):
        embed = ModalityEmbedding(hidden_size=64)
        out = embed(Modality.TEXT, length=10, batch_size=2)
        assert out.shape == (2, 10, 64)

    def test_different_modalities(self):
        embed = ModalityEmbedding(hidden_size=64)
        text_emb = embed(Modality.TEXT, 10, 2)
        image_emb = embed(Modality.IMAGE, 10, 2)
        assert not torch.allclose(text_emb, image_emb)


class TestUnifiedEncoder:
    """统一编码器测试"""

    @pytest.fixture
    def config(self):
        return UnifiedIOConfig(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            image_size=64,
            image_patch_size=16,
            audio_patch_size=16,
            video_patch_size=16,
            text_vocab_size=1000,
            max_text_length=32,
        )

    @pytest.fixture
    def encoder(self, config):
        return UnifiedEncoder(config)

    def test_text_only(self, encoder):
        batch = MultimodalBatch(
            text_input_ids=torch.randint(0, 1000, (2, 10)),
            text_attention_mask=torch.ones(2, 10),
        )
        out = encoder(batch)
        assert "hidden_states" in out
        assert "attention_mask" in out
        assert "spans" in out
        assert out["hidden_states"].shape[0] == 2
        assert out["hidden_states"].shape[2] == 64

    def test_image_only(self, encoder):
        batch = MultimodalBatch(images=torch.randn(2, 3, 64, 64))
        out = encoder(batch)
        num_patches = (64 // 16) ** 2
        assert out["hidden_states"].shape[1] == 1 + num_patches

    def test_audio_only(self, encoder):
        batch = MultimodalBatch(audio=torch.randn(2, 256))
        out = encoder(batch)
        num_patches = 256 // 16
        assert out["hidden_states"].shape[1] == 1 + num_patches

    def test_video_only(self, encoder, config):
        batch = MultimodalBatch(video=torch.randn(2, 3, 4, 64, 64))
        out = encoder(batch)
        num_patches_per_frame = (64 // 16) ** 2
        total_patches = 4 * num_patches_per_frame
        assert out["hidden_states"].shape[1] == 1 + total_patches

    def test_multimodal_input(self, encoder):
        batch = MultimodalBatch(
            text_input_ids=torch.randint(0, 1000, (2, 10)),
            images=torch.randn(2, 3, 64, 64),
        )
        out = encoder(batch)
        text_len = 10
        image_patches = (64 // 16) ** 2
        expected_len = 1 + text_len + image_patches
        assert out["hidden_states"].shape[1] == expected_len

    def test_spans_tracking(self, encoder):
        batch = MultimodalBatch(
            text_input_ids=torch.randint(0, 1000, (2, 10)),
            images=torch.randn(2, 3, 64, 64),
        )
        out = encoder(batch)
        spans = out["spans"]
        assert "cls" in spans
        assert "text" in spans
        assert "image" in spans
        assert spans["cls"] == (0, 1)
        assert spans["text"][0] == 1
        assert spans["image"][0] == spans["text"][1]


class TestUnifiedDecoder:
    """统一解码器测试"""

    @pytest.fixture
    def config(self):
        return UnifiedIOConfig(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            text_vocab_size=1000,
            max_text_length=32,
        )

    @pytest.fixture
    def decoder(self, config):
        return UnifiedDecoder(config)

    def test_decoder_forward(self, decoder, config):
        input_ids = torch.randint(0, 1000, (2, 10))
        encoder_hidden = torch.randn(2, 20, 64)
        encoder_mask = torch.ones(2, 20)
        logits = decoder(input_ids, encoder_hidden, encoder_mask)
        assert logits.shape == (2, 10, config.text_vocab_size)


class TestUnifiedIO:
    """完整模型测试"""

    @pytest.fixture
    def config(self):
        return UnifiedIOConfig(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            image_size=64,
            image_patch_size=16,
            audio_patch_size=16,
            video_patch_size=16,
            text_vocab_size=1000,
            max_text_length=32,
            num_labels=5,
        )

    @pytest.fixture
    def model(self, config):
        return UnifiedIO(config)

    def test_classification_task(self, model, config):
        batch = MultimodalBatch(
            images=torch.randn(2, 3, 64, 64),
            labels=torch.randint(0, 5, (2,)),
        )
        out = model(batch, task=TaskType.CLASSIFICATION)
        assert "logits" in out
        assert "loss" in out
        assert out["logits"].shape == (2, config.num_labels)
        assert out["loss"] is not None

    def test_classification_without_labels(self, model, config):
        batch = MultimodalBatch(images=torch.randn(2, 3, 64, 64))
        out = model(batch, task=TaskType.CLASSIFICATION)
        assert out["logits"].shape == (2, config.num_labels)
        assert out["loss"] is None

    def test_retrieval_task(self, model):
        batch = MultimodalBatch(images=torch.randn(2, 3, 64, 64))
        out = model(batch, task=TaskType.RETRIEVAL)
        assert "embeddings" in out
        assert out["embeddings"].shape == (2, 64)
        norms = out["embeddings"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_generation_task(self, model, config):
        batch = MultimodalBatch(
            images=torch.randn(2, 3, 64, 64),
            text_input_ids=torch.randint(0, 1000, (2, 10)),
            labels=torch.randint(0, 1000, (2, 10)),
        )
        out = model(batch, task=TaskType.GENERATION)
        assert "logits" in out
        assert "loss" in out
        assert out["logits"].shape == (2, 10, config.text_vocab_size)

    def test_generate_method(self, model, config):
        batch = MultimodalBatch(
            images=torch.randn(2, 3, 64, 64),
            text_input_ids=torch.randint(0, 1000, (2, 5)),
        )
        generated = model.generate(batch, max_length=15)
        assert generated.shape[0] == 2
        assert generated.shape[1] <= 15
        assert generated.shape[1] >= 5

    def test_multimodal_classification(self, model, config):
        batch = MultimodalBatch(
            text_input_ids=torch.randint(0, 1000, (2, 10)),
            images=torch.randn(2, 3, 64, 64),
            labels=torch.randint(0, 5, (2,)),
        )
        out = model(batch, task=TaskType.CLASSIFICATION)
        assert out["logits"].shape == (2, config.num_labels)

    def test_audio_classification(self, model, config):
        batch = MultimodalBatch(
            audio=torch.randn(2, 256),
            labels=torch.randint(0, 5, (2,)),
        )
        out = model(batch, task=TaskType.CLASSIFICATION)
        assert out["logits"].shape == (2, config.num_labels)

    def test_video_classification(self, model, config):
        batch = MultimodalBatch(
            video=torch.randn(2, 3, 4, 64, 64),
            labels=torch.randint(0, 5, (2,)),
        )
        out = model(batch, task=TaskType.CLASSIFICATION)
        assert out["logits"].shape == (2, config.num_labels)


class TestFactoryFunction:
    """工厂函数测试"""

    @pytest.mark.parametrize("size", ["tiny", "small", "base", "large"])
    def test_create_model_sizes(self, size):
        model = create_unified_io_model(size)
        assert isinstance(model, UnifiedIO)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            create_unified_io_model("invalid")

    def test_tiny_model_forward(self):
        model = create_unified_io_model("tiny")
        batch = MultimodalBatch(
            text_input_ids=torch.randint(0, 1000, (2, 10)),
        )
        out = model(batch, task=TaskType.CLASSIFICATION)
        assert "logits" in out


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def model(self):
        config = UnifiedIOConfig(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            image_size=64,
            image_patch_size=16,
            text_vocab_size=1000,
            max_text_length=32,
        )
        return UnifiedIO(config)

    def test_empty_batch_raises(self, model):
        batch = MultimodalBatch()
        with pytest.raises(ValueError):
            model(batch)

    def test_generation_without_text_raises(self, model):
        batch = MultimodalBatch(images=torch.randn(2, 3, 64, 64))
        with pytest.raises(ValueError):
            model(batch, task=TaskType.GENERATION)

    def test_batch_size_one(self, model):
        batch = MultimodalBatch(images=torch.randn(1, 3, 64, 64))
        out = model(batch, task=TaskType.CLASSIFICATION)
        assert out["logits"].shape[0] == 1

    def test_text_without_mask(self, model):
        batch = MultimodalBatch(text_input_ids=torch.randint(0, 1000, (2, 10)))
        out = model(batch, task=TaskType.CLASSIFICATION)
        assert out["logits"].shape[0] == 2


class TestPoolingMethods:
    """池化方法测试"""

    def test_cls_pooling(self):
        config = UnifiedIOConfig(hidden_size=64, num_layers=2, num_heads=4, pooling="cls")
        model = UnifiedIO(config)
        batch = MultimodalBatch(text_input_ids=torch.randint(0, 1000, (2, 10)))
        out = model(batch, task=TaskType.RETRIEVAL)
        assert out["embeddings"].shape == (2, 64)

    def test_mean_pooling(self):
        config = UnifiedIOConfig(hidden_size=64, num_layers=2, num_heads=4, pooling="mean")
        model = UnifiedIO(config)
        batch = MultimodalBatch(text_input_ids=torch.randint(0, 1000, (2, 10)))
        out = model(batch, task=TaskType.RETRIEVAL)
        assert out["embeddings"].shape == (2, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
