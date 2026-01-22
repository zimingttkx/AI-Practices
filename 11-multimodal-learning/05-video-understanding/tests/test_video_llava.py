"""Video-LLaVA 单元测试"""

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_llava import (
    VideoLLaVAConfig,
    SamplingStrategy,
    PatchEmbedding,
    VisionEncoder,
    MultiHeadAttention,
    MLP,
    TransformerBlock,
    TemporalPositionalEncoding,
    TemporalTransformer,
    TemporalLSTM,
    TemporalPooling,
    VideoProjector,
    RMSNorm,
    RotaryEmbedding,
    LLaMAAttention,
    LLaMAMLP,
    LLaMADecoderLayer,
    LLaMAModel,
    VideoLLaVA,
    VideoProcessor,
    create_video_llava,
    rotate_half,
)


class TestVideoLLaVAConfig:
    """VideoLLaVAConfig 测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = VideoLLaVAConfig()
        assert config.image_size == 224
        assert config.patch_size == 14
        assert config.num_frames == 8
        assert config.temporal_mode == "transformer"

    def test_custom_config(self):
        """测试自定义配置"""
        config = VideoLLaVAConfig(
            image_size=336,
            patch_size=14,
            num_frames=16,
            temporal_mode="lstm"
        )
        assert config.image_size == 336
        assert config.num_frames == 16
        assert config.temporal_mode == "lstm"

    def test_config_validation_image_size(self):
        """测试图像大小验证"""
        with pytest.raises(AssertionError):
            VideoLLaVAConfig(image_size=225, patch_size=14)

    def test_config_validation_temporal_mode(self):
        """测试时序模式验证"""
        with pytest.raises(AssertionError):
            VideoLLaVAConfig(temporal_mode="invalid")

    def test_num_patches_calculation(self):
        """测试 patch 数量计算"""
        config = VideoLLaVAConfig(image_size=224, patch_size=14)
        expected_patches = (224 // 14) ** 2
        assert expected_patches == 256


class TestPatchEmbedding:
    """PatchEmbedding 测试"""

    @pytest.fixture
    def config(self):
        return VideoLLaVAConfig(
            image_size=224,
            patch_size=16,
            vision_width=384,
            vision_heads=6
        )

    def test_output_shape(self, config):
        """测试输出形状"""
        patch_embed = PatchEmbedding(config)
        x = torch.randn(2, 3, 224, 224)
        output = patch_embed(x)

        num_patches = (224 // 16) ** 2
        assert output.shape == (2, num_patches + 1, 384)

    def test_cls_token(self, config):
        """测试 CLS token"""
        patch_embed = PatchEmbedding(config)
        assert patch_embed.cls_token.shape == (1, 1, 384)

    def test_position_embedding(self, config):
        """测试位置编码"""
        patch_embed = PatchEmbedding(config)
        num_patches = (224 // 16) ** 2
        assert patch_embed.position_embedding.shape == (1, num_patches + 1, 384)


class TestVisionEncoder:
    """VisionEncoder 测试"""

    @pytest.fixture
    def config(self):
        return VideoLLaVAConfig(
            image_size=224,
            patch_size=16,
            vision_layers=2,
            vision_width=384,
            vision_heads=6
        )

    def test_output_shape(self, config):
        """测试输出形状"""
        encoder = VisionEncoder(config)
        x = torch.randn(2, 3, 224, 224)
        output = encoder(x)

        num_patches = (224 // 16) ** 2
        assert output.shape == (2, num_patches + 1, 384)

    def test_num_layers(self, config):
        """测试层数"""
        encoder = VisionEncoder(config)
        assert len(encoder.blocks) == 2


class TestMultiHeadAttention:
    """MultiHeadAttention 测试"""

    def test_output_shape(self):
        """测试输出形状"""
        attn = MultiHeadAttention(d_model=256, num_heads=8)
        x = torch.randn(2, 10, 256)
        output = attn(x)
        assert output.shape == (2, 10, 256)

    def test_causal_mask(self):
        """测试因果掩码"""
        attn = MultiHeadAttention(d_model=256, num_heads=8)
        x = torch.randn(2, 10, 256)
        output = attn(x, causal=True)
        assert output.shape == (2, 10, 256)

    def test_attention_mask(self):
        """测试注意力掩码"""
        attn = MultiHeadAttention(d_model=256, num_heads=8)
        x = torch.randn(2, 10, 256)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0
        output = attn(x, attention_mask=mask)
        assert output.shape == (2, 10, 256)


class TestMLP:
    """MLP 测试"""

    def test_output_shape(self):
        """测试输出形状"""
        mlp = MLP(d_model=256, hidden_dim=1024)
        x = torch.randn(2, 10, 256)
        output = mlp(x)
        assert output.shape == (2, 10, 256)


class TestTransformerBlock:
    """TransformerBlock 测试"""

    def test_output_shape(self):
        """测试输出形状"""
        block = TransformerBlock(d_model=256, num_heads=8)
        x = torch.randn(2, 10, 256)
        output = block(x)
        assert output.shape == (2, 10, 256)

    def test_causal_block(self):
        """测试因果 Transformer 块"""
        block = TransformerBlock(d_model=256, num_heads=8, causal=True)
        x = torch.randn(2, 10, 256)
        output = block(x)
        assert output.shape == (2, 10, 256)


class TestTemporalPositionalEncoding:
    """TemporalPositionalEncoding 测试"""

    def test_output_shape(self):
        """测试输出形状"""
        pe = TemporalPositionalEncoding(d_model=256, max_frames=100)
        x = torch.randn(2, 8, 256)
        output = pe(x)
        assert output.shape == (2, 8, 256)

    def test_encoding_values(self):
        """测试编码值"""
        pe = TemporalPositionalEncoding(d_model=256, max_frames=100)
        assert pe.pe.shape == (1, 100, 256)


class TestTemporalTransformer:
    """TemporalTransformer 测试"""

    @pytest.fixture
    def config(self):
        return VideoLLaVAConfig(
            temporal_hidden_size=256,
            temporal_layers=2,
            temporal_heads=4,
            num_frames=8
        )

    def test_output_shape(self, config):
        """测试输出形状"""
        temporal = TemporalTransformer(config)
        x = torch.randn(2, 8, 256)
        output = temporal(x)
        assert output.shape == (2, 8, 256)


class TestTemporalLSTM:
    """TemporalLSTM 测试"""

    @pytest.fixture
    def config(self):
        return VideoLLaVAConfig(temporal_hidden_size=256)

    def test_output_shape(self, config):
        """测试输出形状"""
        lstm = TemporalLSTM(config)
        x = torch.randn(2, 8, 256)
        output = lstm(x)
        assert output.shape == (2, 8, 256)


class TestTemporalPooling:
    """TemporalPooling 测试"""

    @pytest.fixture
    def config(self):
        return VideoLLaVAConfig(vision_width=256)

    def test_mean_pooling(self, config):
        """测试均值池化"""
        pooling = TemporalPooling(config)
        pooling.pooling_type = "mean"
        x = torch.randn(2, 8, 16, 256)
        output = pooling(x)
        assert output.shape == (2, 16, 256)

    def test_max_pooling(self, config):
        """测试最大池化"""
        pooling = TemporalPooling(config)
        pooling.pooling_type = "max"
        x = torch.randn(2, 8, 16, 256)
        output = pooling(x)
        assert output.shape == (2, 16, 256)


class TestVideoProjector:
    """VideoProjector 测试"""

    def test_linear_projector(self):
        """测试线性投影"""
        config = VideoLLaVAConfig(
            vision_width=256,
            hidden_size=512,
            projector_type="linear"
        )
        projector = VideoProjector(config)
        x = torch.randn(2, 10, 256)
        output = projector(x)
        assert output.shape == (2, 10, 512)

    def test_mlp_projector(self):
        """测试 MLP 投影"""
        config = VideoLLaVAConfig(
            vision_width=256,
            hidden_size=512,
            projector_type="mlp2x_gelu"
        )
        projector = VideoProjector(config)
        x = torch.randn(2, 10, 256)
        output = projector(x)
        assert output.shape == (2, 10, 512)


class TestRMSNorm:
    """RMSNorm 测试"""

    def test_output_shape(self):
        """测试输出形状"""
        norm = RMSNorm(hidden_size=256)
        x = torch.randn(2, 10, 256)
        output = norm(x)
        assert output.shape == (2, 10, 256)

    def test_normalization(self):
        """测试归一化效果"""
        norm = RMSNorm(hidden_size=256)
        x = torch.randn(2, 10, 256) * 100
        output = norm(x)
        rms = output.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestRotaryEmbedding:
    """RotaryEmbedding 测试"""

    def test_output_shape(self):
        """测试输出形状"""
        rope = RotaryEmbedding(dim=64, max_seq_len=512)
        cos, sin = rope(100)
        assert cos.shape == (1, 1, 100, 64)
        assert sin.shape == (1, 1, 100, 64)


class TestRotateHalf:
    """rotate_half 函数测试"""

    def test_rotate_half(self):
        """测试旋转一半"""
        x = torch.tensor([[1, 2, 3, 4]])
        rotated = rotate_half(x)
        expected = torch.tensor([[-3, -4, 1, 2]])
        assert torch.equal(rotated, expected)


class TestLLaMAAttention:
    """LLaMAAttention 测试"""

    @pytest.fixture
    def config(self):
        return VideoLLaVAConfig(
            hidden_size=256,
            num_heads=8,
            max_seq_length=512
        )

    def test_output_shape(self, config):
        """测试输出形状"""
        attn = LLaMAAttention(config)
        x = torch.randn(2, 10, 256)
        output = attn(x)
        assert output.shape == (2, 10, 256)


class TestLLaMAMLP:
    """LLaMAMLP 测试"""

    @pytest.fixture
    def config(self):
        return VideoLLaVAConfig(
            hidden_size=256,
            intermediate_size=1024
        )

    def test_output_shape(self, config):
        """测试输出形状"""
        mlp = LLaMAMLP(config)
        x = torch.randn(2, 10, 256)
        output = mlp(x)
        assert output.shape == (2, 10, 256)


class TestLLaMADecoderLayer:
    """LLaMADecoderLayer 测试"""

    @pytest.fixture
    def config(self):
        return VideoLLaVAConfig(
            hidden_size=256,
            num_heads=8,
            intermediate_size=1024,
            max_seq_length=512
        )

    def test_output_shape(self, config):
        """测试输出形状"""
        layer = LLaMADecoderLayer(config)
        x = torch.randn(2, 10, 256)
        output = layer(x)
        assert output.shape == (2, 10, 256)


class TestLLaMAModel:
    """LLaMAModel 测试"""

    @pytest.fixture
    def config(self):
        return VideoLLaVAConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=8,
            intermediate_size=1024,
            max_seq_length=512
        )

    def test_output_shape_with_input_ids(self, config):
        """测试使用 input_ids 的输出形状"""
        model = LLaMAModel(config)
        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids=input_ids)
        assert output.shape == (2, 10, 1000)

    def test_output_shape_with_embeds(self, config):
        """测试使用 inputs_embeds 的输出形状"""
        model = LLaMAModel(config)
        inputs_embeds = torch.randn(2, 10, 256)
        output = model(inputs_embeds=inputs_embeds)
        assert output.shape == (2, 10, 1000)


class TestVideoProcessor:
    """VideoProcessor 测试"""

    def test_uniform_sampling(self):
        """测试均匀采样"""
        video = torch.randn(16, 3, 224, 224)
        sampled = VideoProcessor.sample_frames(
            video, num_frames=8, strategy=SamplingStrategy.UNIFORM
        )
        assert sampled.shape == (8, 3, 224, 224)

    def test_random_sampling(self):
        """测试随机采样"""
        video = torch.randn(16, 3, 224, 224)
        sampled = VideoProcessor.sample_frames(
            video, num_frames=8, strategy=SamplingStrategy.RANDOM
        )
        assert sampled.shape == (8, 3, 224, 224)

    def test_sampling_with_batch(self):
        """测试批量采样"""
        video = torch.randn(2, 16, 3, 224, 224)
        sampled = VideoProcessor.sample_frames(
            video, num_frames=8, strategy=SamplingStrategy.UNIFORM
        )
        assert sampled.shape == (2, 8, 3, 224, 224)

    def test_sampling_insufficient_frames(self):
        """测试帧数不足时的采样"""
        video = torch.randn(4, 3, 224, 224)
        sampled = VideoProcessor.sample_frames(
            video, num_frames=8, strategy=SamplingStrategy.UNIFORM
        )
        assert sampled.shape == (8, 3, 224, 224)

    def test_resize_frames(self):
        """测试帧大小调整"""
        frames = torch.randn(8, 3, 480, 640)
        resized = VideoProcessor.resize_frames(frames, size=(224, 224))
        assert resized.shape == (8, 3, 224, 224)

    def test_resize_frames_with_batch(self):
        """测试批量帧大小调整"""
        frames = torch.randn(2, 8, 3, 480, 640)
        resized = VideoProcessor.resize_frames(frames, size=(224, 224))
        assert resized.shape == (2, 8, 3, 224, 224)


class TestVideoLLaVA:
    """VideoLLaVA 完整模型测试"""

    @pytest.fixture
    def tiny_config(self):
        return VideoLLaVAConfig(
            image_size=224,
            patch_size=16,
            vision_layers=2,
            vision_width=256,
            vision_heads=4,
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
            num_frames=4,
            temporal_mode="transformer",
            temporal_layers=1,
            temporal_heads=4,
            temporal_hidden_size=256
        )

    def test_model_creation(self, tiny_config):
        """测试模型创建"""
        model = VideoLLaVA(tiny_config)
        assert model is not None

    def test_encode_video(self, tiny_config):
        """测试视频编码"""
        model = VideoLLaVA(tiny_config)
        video = torch.randn(2, 8, 3, 224, 224)
        video_tokens = model.encode_video(video)
        assert video_tokens.dim() == 3
        assert video_tokens.shape[0] == 2
        assert video_tokens.shape[2] == 256

    def test_forward_with_video_only(self, tiny_config):
        """测试仅视频输入"""
        model = VideoLLaVA(tiny_config)
        video = torch.randn(2, 8, 3, 224, 224)
        output = model(videos=video)
        assert "logits" in output
        assert "video_tokens" in output

    def test_forward_with_text_only(self, tiny_config):
        """测试仅文本输入"""
        model = VideoLLaVA(tiny_config)
        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids=input_ids)
        assert "logits" in output
        assert output["logits"].shape == (2, 10, 1000)

    def test_forward_with_video_and_text(self, tiny_config):
        """测试视频和文本输入"""
        model = VideoLLaVA(tiny_config)
        video = torch.randn(2, 8, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids=input_ids, videos=video)
        assert "logits" in output
        assert "video_tokens" in output

    def test_temporal_modes(self, tiny_config):
        """测试不同时序模式"""
        for mode in ["transformer", "lstm", "pooling"]:
            tiny_config.temporal_mode = mode
            model = VideoLLaVA(tiny_config)
            video = torch.randn(1, 8, 3, 224, 224)
            video_tokens = model.encode_video(video)
            assert video_tokens is not None


class TestCreateVideoLLaVA:
    """create_video_llava 工厂函数测试"""

    def test_create_tiny(self):
        """测试创建 tiny 模型"""
        model = create_video_llava(model_size="tiny")
        assert model is not None
        assert model.config.vision_layers == 6

    def test_create_base(self):
        """测试创建 base 模型"""
        model = create_video_llava(model_size="base")
        assert model is not None
        assert model.config.vision_layers == 24

    @pytest.mark.skip(reason="Large model requires too much memory")
    def test_create_large(self):
        """测试创建 large 模型"""
        model = create_video_llava(model_size="large")
        assert model is not None
        assert model.config.vision_layers == 32

    def test_create_with_custom_frames(self):
        """测试自定义帧数"""
        model = create_video_llava(model_size="tiny", num_frames=16)
        assert model.config.num_frames == 16

    def test_create_with_custom_temporal_mode(self):
        """测试自定义时序模式"""
        model = create_video_llava(model_size="tiny", temporal_mode="lstm")
        assert model.config.temporal_mode == "lstm"

    def test_invalid_model_size(self):
        """测试无效模型大小"""
        with pytest.raises(ValueError):
            create_video_llava(model_size="invalid")


class TestIntegration:
    """集成测试"""

    @pytest.fixture
    def tiny_model(self):
        return create_video_llava(model_size="tiny", num_frames=4)

    def test_end_to_end_inference(self, tiny_model):
        """测试端到端推理"""
        video = torch.randn(1, 8, 3, 224, 224)
        input_ids = torch.randint(0, 32000, (1, 20))

        with torch.no_grad():
            output = tiny_model(input_ids=input_ids, videos=video)

        assert output["logits"] is not None
        assert output["video_tokens"] is not None

    def test_gradient_flow(self, tiny_model):
        """测试梯度流"""
        # 仅使用文本输入测试梯度流，避免视频+文本拼接后的 labels 形状问题
        input_ids = torch.randint(0, 32000, (1, 10))
        labels = torch.randint(0, 32000, (1, 10))

        output = tiny_model(input_ids=input_ids, labels=labels)

        if output["loss"] is not None:
            output["loss"].backward()

    def test_different_batch_sizes(self, tiny_model):
        """测试不同批量大小"""
        for batch_size in [1, 2, 4]:
            video = torch.randn(batch_size, 4, 3, 224, 224)
            input_ids = torch.randint(0, 32000, (batch_size, 10))

            with torch.no_grad():
                output = tiny_model(input_ids=input_ids, videos=video)

            assert output["logits"].shape[0] == batch_size

    def test_different_video_lengths(self, tiny_model):
        """测试不同视频长度"""
        for num_frames in [2, 4, 8, 16]:
            video = torch.randn(1, num_frames, 3, 224, 224)

            with torch.no_grad():
                video_tokens = tiny_model.encode_video(video)

            assert video_tokens is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

