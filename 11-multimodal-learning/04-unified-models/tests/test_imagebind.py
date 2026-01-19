"""
ImageBind 单元测试

测试覆盖:
- 配置与初始化
- 各模态编码器
- 模态投影器
- 对比学习损失
- 完整模型前向传播
- 跨模态检索
- 零样本分类
- 工厂函数
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagebind import (
    AudioEncoder,
    DepthEncoder,
    ImageBind,
    ImageBindConfig,
    ImageBindLoss,
    ImageEncoder,
    IMUEncoder,
    ModalityProjector,
    ModalityType,
    PatchEmbedding,
    TextEncoder,
    ThermalEncoder,
    TransformerBlock,
    create_imagebind_model,
)


class TestImageBindConfig:
    """配置测试"""

    def test_default_config(self):
        config = ImageBindConfig()
        assert config.embed_dim == 768
        assert config.vision_embed_dim == 1024
        assert config.image_size == 224
        assert config.patch_size == 14
        assert config.temperature == 0.07

    def test_custom_config(self):
        config = ImageBindConfig(embed_dim=512, vision_layers=6)
        assert config.embed_dim == 512
        assert config.vision_layers == 6

    def test_modality_embed_dims(self):
        config = ImageBindConfig()
        assert "image" in config.modality_embed_dims
        assert "text" in config.modality_embed_dims
        assert "audio" in config.modality_embed_dims


class TestPatchEmbedding:
    """Patch 嵌入测试"""

    def test_2d_patch_embed(self):
        embed = PatchEmbedding(3, 768, patch_size=16)
        x = torch.randn(2, 3, 224, 224)
        out = embed(x)
        num_patches = (224 // 16) ** 2
        assert out.shape == (2, num_patches, 768)

    def test_patch_embed_with_stride(self):
        embed = PatchEmbedding(1, 512, patch_size=16, stride=10)
        x = torch.randn(2, 1, 128, 204)
        out = embed(x)
        h_patches = (128 - 16) // 10 + 1
        w_patches = (204 - 16) // 10 + 1
        assert out.shape == (2, h_patches * w_patches, 512)


class TestTransformerBlock:
    """Transformer 块测试"""

    def test_forward(self):
        block = TransformerBlock(embed_dim=256, num_heads=4)
        x = torch.randn(2, 10, 256)
        out = block(x)
        assert out.shape == x.shape

    def test_with_attention_mask(self):
        block = TransformerBlock(embed_dim=256, num_heads=4)
        x = torch.randn(2, 10, 256)
        mask = torch.zeros(10, 10).bool()
        mask[0, 5:] = True
        out = block(x, attn_mask=mask)
        assert out.shape == x.shape


class TestImageEncoder:
    """图像编码器测试"""

    @pytest.fixture
    def config(self):
        return ImageBindConfig(
            vision_embed_dim=256,
            vision_layers=2,
            vision_heads=4,
            image_size=112,
            patch_size=14,
        )

    def test_forward(self, config):
        encoder = ImageEncoder(config)
        x = torch.randn(2, 3, 112, 112)
        out = encoder(x)
        assert out.shape == (2, config.vision_embed_dim)

    def test_output_is_cls_token(self, config):
        encoder = ImageEncoder(config)
        x = torch.randn(1, 3, 112, 112)
        out = encoder(x)
        assert out.dim() == 2


class TestTextEncoder:
    """文本编码器测试"""

    @pytest.fixture
    def config(self):
        return ImageBindConfig(
            text_embed_dim=256,
            text_layers=2,
            text_heads=4,
            text_vocab_size=1000,
            text_max_length=32,
        )

    def test_forward(self, config):
        encoder = TextEncoder(config)
        input_ids = torch.randint(0, 1000, (2, 20))
        out = encoder(input_ids)
        assert out.shape == (2, config.text_embed_dim)

    def test_with_attention_mask(self, config):
        encoder = TextEncoder(config)
        input_ids = torch.randint(0, 1000, (2, 20))
        attention_mask = torch.ones(2, 20)
        attention_mask[0, 15:] = 0
        out = encoder(input_ids, attention_mask)
        assert out.shape == (2, config.text_embed_dim)

    def test_different_seq_lengths(self, config):
        encoder = TextEncoder(config)
        input_ids_short = torch.randint(0, 1000, (2, 10))
        input_ids_long = torch.randint(0, 1000, (2, 30))
        out_short = encoder(input_ids_short)
        out_long = encoder(input_ids_long)
        assert out_short.shape == out_long.shape


class TestAudioEncoder:
    """音频编码器测试"""

    @pytest.fixture
    def config(self):
        return ImageBindConfig(
            audio_embed_dim=256,
            audio_layers=2,
            audio_heads=4,
            audio_num_mel_bins=128,
            audio_target_length=204,
            audio_patch_size=16,
            audio_stride=10,
        )

    def test_forward_4d(self, config):
        encoder = AudioEncoder(config)
        x = torch.randn(2, 1, 128, 204)
        out = encoder(x)
        assert out.shape == (2, config.audio_embed_dim)

    def test_forward_3d(self, config):
        encoder = AudioEncoder(config)
        x = torch.randn(2, 128, 204)
        out = encoder(x)
        assert out.shape == (2, config.audio_embed_dim)


class TestDepthEncoder:
    """深度图编码器测试"""

    @pytest.fixture
    def config(self):
        return ImageBindConfig(
            vision_embed_dim=256,
            vision_layers=2,
            vision_heads=4,
            image_size=112,
            depth_patch_size=14,
        )

    def test_forward_4d(self, config):
        encoder = DepthEncoder(config)
        x = torch.randn(2, 1, 112, 112)
        out = encoder(x)
        assert out.shape == (2, config.vision_embed_dim)

    def test_forward_3d(self, config):
        encoder = DepthEncoder(config)
        x = torch.randn(2, 112, 112)
        out = encoder(x)
        assert out.shape == (2, config.vision_embed_dim)


class TestThermalEncoder:
    """热力图编码器测试"""

    @pytest.fixture
    def config(self):
        return ImageBindConfig(
            vision_embed_dim=256,
            vision_layers=2,
            vision_heads=4,
            image_size=112,
            thermal_patch_size=14,
        )

    def test_forward(self, config):
        encoder = ThermalEncoder(config)
        x = torch.randn(2, 1, 112, 112)
        out = encoder(x)
        assert out.shape == (2, config.vision_embed_dim)


class TestIMUEncoder:
    """IMU 编码器测试"""

    @pytest.fixture
    def config(self):
        return ImageBindConfig(
            imu_input_dim=6,
            imu_seq_length=2000,
            imu_patch_size=50,
            imu_layers=2,
            imu_heads=4,
            modality_embed_dims={"imu": 256},
        )

    def test_forward(self, config):
        encoder = IMUEncoder(config)
        x = torch.randn(2, 2000, 6)
        out = encoder(x)
        assert out.shape == (2, 256)

    def test_different_seq_length(self, config):
        encoder = IMUEncoder(config)
        x = torch.randn(2, 1000, 6)
        out = encoder(x)
        assert out.shape[0] == 2


class TestModalityProjector:
    """模态投影器测试"""

    def test_forward(self):
        proj = ModalityProjector(512, 768)
        x = torch.randn(2, 512)
        out = proj(x)
        assert out.shape == (2, 768)

    def test_normalization(self):
        proj = ModalityProjector(256, 256)
        x = torch.randn(2, 256)
        out = proj(x)
        assert out.shape == (2, 256)


class TestImageBindLoss:
    """对比学习损失测试"""

    def test_forward(self):
        loss_fn = ImageBindLoss(temperature=0.07)
        anchor = torch.randn(4, 256)
        positive = torch.randn(4, 256)
        loss = loss_fn(anchor, positive)
        assert loss.dim() == 0
        assert loss >= 0

    def test_learnable_temperature(self):
        loss_fn = ImageBindLoss(temperature=0.07, learnable=True)
        assert hasattr(loss_fn, "log_temperature")
        assert loss_fn.log_temperature.requires_grad

    def test_fixed_temperature(self):
        loss_fn = ImageBindLoss(temperature=0.07, learnable=False)
        assert not loss_fn.log_temperature.requires_grad

    def test_perfect_alignment(self):
        loss_fn = ImageBindLoss(temperature=0.07)
        embeds = torch.randn(4, 256)
        embeds = torch.nn.functional.normalize(embeds, dim=-1)
        loss = loss_fn(embeds, embeds)
        assert loss < 0.1

    def test_symmetric_loss(self):
        loss_fn = ImageBindLoss(temperature=0.07)
        a = torch.randn(4, 256)
        b = torch.randn(4, 256)
        loss_ab = loss_fn(a, b)
        loss_ba = loss_fn(b, a)
        assert torch.allclose(loss_ab, loss_ba, atol=1e-5)


class TestImageBind:
    """完整模型测试"""

    @pytest.fixture
    def config(self):
        return ImageBindConfig(
            embed_dim=128,
            vision_embed_dim=192,
            text_embed_dim=128,
            audio_embed_dim=128,
            vision_layers=2,
            vision_heads=4,
            text_layers=2,
            text_heads=4,
            audio_layers=2,
            audio_heads=4,
            imu_layers=2,
            imu_heads=4,
            image_size=56,
            patch_size=14,
            text_vocab_size=1000,
            text_max_length=32,
            audio_num_mel_bins=64,
            audio_target_length=100,
            audio_patch_size=8,
            audio_stride=8,
            imu_seq_length=500,
            imu_patch_size=25,
            modality_embed_dims={
                "image": 192, "text": 128, "audio": 128,
                "depth": 192, "thermal": 192, "imu": 128,
            },
        )

    @pytest.fixture
    def model(self, config):
        return ImageBind(config)

    def test_encode_image(self, model, config):
        images = torch.randn(2, 3, 56, 56)
        embeds = model.encode_image(images)
        assert embeds.shape == (2, config.embed_dim)
        norms = embeds.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_text(self, model, config):
        input_ids = torch.randint(0, 1000, (2, 20))
        embeds = model.encode_text(input_ids)
        assert embeds.shape == (2, config.embed_dim)

    def test_encode_text_with_mask(self, model, config):
        input_ids = torch.randint(0, 1000, (2, 20))
        attention_mask = torch.ones(2, 20)
        attention_mask[0, 15:] = 0
        embeds = model.encode_text(input_ids, attention_mask)
        assert embeds.shape == (2, config.embed_dim)

    def test_encode_audio(self, model, config):
        audio = torch.randn(2, 64, 100)
        embeds = model.encode_audio(audio)
        assert embeds.shape == (2, config.embed_dim)

    def test_encode_depth(self, model, config):
        depth = torch.randn(2, 56, 56)
        embeds = model.encode_depth(depth)
        assert embeds.shape == (2, config.embed_dim)

    def test_encode_thermal(self, model, config):
        thermal = torch.randn(2, 1, 56, 56)
        embeds = model.encode_thermal(thermal)
        assert embeds.shape == (2, config.embed_dim)

    def test_encode_imu(self, model, config):
        imu = torch.randn(2, 500, 6)
        embeds = model.encode_imu(imu)
        assert embeds.shape == (2, config.embed_dim)

    def test_encode_generic(self, model, config):
        images = torch.randn(2, 3, 56, 56)
        embeds = model.encode(ModalityType.IMAGE, images)
        assert embeds.shape == (2, config.embed_dim)

    def test_encode_with_string_modality(self, model, config):
        images = torch.randn(2, 3, 56, 56)
        embeds = model.encode("image", images)
        assert embeds.shape == (2, config.embed_dim)

    def test_compute_similarity(self, model):
        embeds_a = torch.randn(3, 128)
        embeds_b = torch.randn(5, 128)
        sim = model.compute_similarity(embeds_a, embeds_b)
        assert sim.shape == (3, 5)
        assert sim.min() >= -1.0
        assert sim.max() <= 1.0

    def test_forward_image_text(self, model, config):
        images = torch.randn(4, 3, 56, 56)
        input_ids = torch.randint(0, 1000, (4, 20))
        outputs = model(
            ModalityType.IMAGE, images,
            ModalityType.TEXT, input_ids,
        )
        assert "loss" in outputs
        assert "anchor_embeds" in outputs
        assert "positive_embeds" in outputs
        assert outputs["loss"] >= 0
        assert outputs["anchor_embeds"].shape == (4, config.embed_dim)

    def test_forward_image_audio(self, model, config):
        images = torch.randn(4, 3, 56, 56)
        audio = torch.randn(4, 64, 100)
        outputs = model(
            ModalityType.IMAGE, images,
            ModalityType.AUDIO, audio,
        )
        assert outputs["loss"] >= 0


class TestCrossModalRetrieval:
    """跨模态检索测试"""

    @pytest.fixture
    def model(self):
        config = ImageBindConfig(
            embed_dim=128,
            vision_embed_dim=192,
            text_embed_dim=128,
            vision_layers=2,
            vision_heads=4,
            text_layers=2,
            text_heads=4,
            image_size=56,
            patch_size=14,
            text_vocab_size=1000,
            text_max_length=32,
            modality_embed_dims={
                "image": 192, "text": 128, "audio": 128,
                "depth": 192, "thermal": 192, "imu": 128,
            },
        )
        return ImageBind(config)

    def test_retrieve_text_from_image(self, model):
        query_image = torch.randn(1, 3, 56, 56)
        gallery_texts = torch.randint(0, 1000, (10, 20))
        scores, indices = model.retrieve(
            ModalityType.IMAGE, query_image,
            ModalityType.TEXT, gallery_texts,
            top_k=5,
        )
        assert scores.shape == (1, 5)
        assert indices.shape == (1, 5)
        assert (indices >= 0).all() and (indices < 10).all()

    def test_retrieve_image_from_text(self, model):
        query_text = torch.randint(0, 1000, (2, 20))
        gallery_images = torch.randn(20, 3, 56, 56)
        scores, indices = model.retrieve(
            ModalityType.TEXT, query_text,
            ModalityType.IMAGE, gallery_images,
            top_k=3,
        )
        assert scores.shape == (2, 3)
        assert indices.shape == (2, 3)

    def test_retrieve_top_k_limit(self, model):
        query = torch.randn(1, 3, 56, 56)
        gallery = torch.randn(3, 3, 56, 56)
        scores, indices = model.retrieve(
            ModalityType.IMAGE, query,
            ModalityType.IMAGE, gallery,
            top_k=10,
        )
        assert scores.shape == (1, 3)


class TestZeroShotClassify:
    """零样本分类测试"""

    @pytest.fixture
    def model(self):
        config = ImageBindConfig(
            embed_dim=128,
            vision_embed_dim=192,
            text_embed_dim=128,
            vision_layers=2,
            vision_heads=4,
            text_layers=2,
            text_heads=4,
            image_size=56,
            patch_size=14,
            text_vocab_size=1000,
            text_max_length=32,
            modality_embed_dims={
                "image": 192, "text": 128, "audio": 128,
                "depth": 192, "thermal": 192, "imu": 128,
            },
        )
        return ImageBind(config)

    def test_zero_shot_image_classification(self, model):
        images = torch.randn(4, 3, 56, 56)
        class_embeds = torch.randn(10, 128)
        logits = model.zero_shot_classify(
            ModalityType.IMAGE, images, class_embeds
        )
        assert logits.shape == (4, 10)

    def test_zero_shot_audio_classification(self, model):
        config = ImageBindConfig(
            embed_dim=128,
            audio_embed_dim=128,
            audio_layers=2,
            audio_heads=4,
            audio_num_mel_bins=64,
            audio_target_length=100,
            audio_patch_size=8,
            audio_stride=8,
            modality_embed_dims={"audio": 128},
        )
        model = ImageBind(config)
        audio = torch.randn(4, 64, 100)
        class_embeds = torch.randn(5, 128)
        logits = model.zero_shot_classify(
            ModalityType.AUDIO, audio, class_embeds
        )
        assert logits.shape == (4, 5)


class TestFactoryFunction:
    """工厂函数测试"""

    @pytest.mark.parametrize("size", ["tiny", "small", "base", "large"])
    def test_create_model_sizes(self, size):
        model = create_imagebind_model(size)
        assert isinstance(model, ImageBind)

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            create_imagebind_model("invalid")

    def test_tiny_model_forward(self):
        model = create_imagebind_model("tiny")
        images = torch.randn(2, 3, 224, 224)
        embeds = model.encode_image(images)
        assert embeds.shape[0] == 2


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def model(self):
        config = ImageBindConfig(
            embed_dim=128,
            vision_embed_dim=192,
            text_embed_dim=128,
            vision_layers=2,
            vision_heads=4,
            text_layers=2,
            text_heads=4,
            image_size=56,
            patch_size=14,
            text_vocab_size=1000,
            text_max_length=32,
            modality_embed_dims={
                "image": 192, "text": 128, "audio": 128,
                "depth": 192, "thermal": 192, "imu": 128,
            },
        )
        return ImageBind(config)

    def test_batch_size_one(self, model):
        images = torch.randn(1, 3, 56, 56)
        embeds = model.encode_image(images)
        assert embeds.shape == (1, 128)

    def test_no_normalize(self, model):
        images = torch.randn(2, 3, 56, 56)
        embeds = model.encode_image(images, normalize=False)
        norms = embeds.norm(dim=-1)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_gradient_flow(self, model):
        images = torch.randn(2, 3, 56, 56, requires_grad=True)
        input_ids = torch.randint(0, 1000, (2, 20))
        outputs = model(
            ModalityType.IMAGE, images,
            ModalityType.TEXT, input_ids,
        )
        outputs["loss"].backward()
        assert images.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
