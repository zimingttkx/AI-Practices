"""
LLaVA 模型单元测试

测试覆盖:
    - LLaVAConfig 配置验证
    - VisionEncoder 视觉编码器
    - VisionProjector 视觉投影层
    - LLaMAModel 语言模型
    - LLaVA 完整模型
"""

import unittest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llava import (
    LLaVAConfig,
    PatchEmbedding,
    MultiHeadAttention,
    VisionEncoder,
    VisionProjector,
    RMSNorm,
    RotaryEmbedding,
    LLaMAAttention,
    LLaMAMLP,
    LLaMADecoderLayer,
    LLaMAModel,
    LLaVA,
    create_llava_model,
)


class TestLLaVAConfig(unittest.TestCase):
    """测试 LLaVA 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = LLaVAConfig()
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.patch_size, 14)
        self.assertEqual(config.hidden_size, 4096)

    def test_custom_config(self):
        """测试自定义配置"""
        config = LLaVAConfig(image_size=336, patch_size=14, hidden_size=2048)
        self.assertEqual(config.image_size, 336)
        self.assertEqual(config.hidden_size, 2048)

    def test_invalid_patch_size(self):
        """测试无效的 patch_size"""
        with self.assertRaises(AssertionError):
            LLaVAConfig(image_size=224, patch_size=15)


class TestVisionEncoder(unittest.TestCase):
    """测试视觉编码器"""

    def setUp(self):
        self.config = LLaVAConfig(
            image_size=224, patch_size=14,
            vision_layers=2, vision_width=384, vision_heads=6
        )
        self.encoder = VisionEncoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        output = self.encoder(images)
        num_patches = (224 // 14) ** 2 + 1  # 257
        self.assertEqual(output.shape, (batch_size, num_patches, 384))


class TestVisionProjector(unittest.TestCase):
    """测试视觉投影层"""

    def test_linear_projector(self):
        """测试线性投影"""
        config = LLaVAConfig(
            vision_width=384, hidden_size=512,
            projector_type="linear"
        )
        projector = VisionProjector(config)
        x = torch.randn(2, 256, 384)
        output = projector(x)
        self.assertEqual(output.shape, (2, 256, 512))

    def test_mlp_projector(self):
        """测试 MLP 投影"""
        config = LLaVAConfig(
            vision_width=384, hidden_size=512,
            projector_type="mlp2x_gelu"
        )
        projector = VisionProjector(config)
        x = torch.randn(2, 256, 384)
        output = projector(x)
        self.assertEqual(output.shape, (2, 256, 512))


class TestRMSNorm(unittest.TestCase):
    """测试 RMS 归一化"""

    def test_output_shape(self):
        """测试输出形状"""
        norm = RMSNorm(512)
        x = torch.randn(2, 10, 512)
        output = norm(x)
        self.assertEqual(output.shape, x.shape)

    def test_normalization(self):
        """测试归一化效果"""
        norm = RMSNorm(512)
        x = torch.randn(2, 10, 512) * 100
        output = norm(x)
        # 输出的 RMS 应该接近 1
        rms = output.pow(2).mean(-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.5))


class TestRotaryEmbedding(unittest.TestCase):
    """测试旋转位置编码"""

    def test_output_shape(self):
        """测试输出形状"""
        rope = RotaryEmbedding(dim=64, max_seq_len=512)
        cos, sin = rope(100)
        self.assertEqual(cos.shape, (100, 64))
        self.assertEqual(sin.shape, (100, 64))


class TestLLaMAComponents(unittest.TestCase):
    """测试 LLaMA 组件"""

    def setUp(self):
        self.config = LLaVAConfig(
            hidden_size=512, num_heads=8, num_layers=2,
            intermediate_size=1376, max_seq_length=256
        )

    def test_llama_attention(self):
        """测试 LLaMA 注意力"""
        attn = LLaMAAttention(self.config)
        x = torch.randn(2, 20, 512)
        output = attn(x)
        self.assertEqual(output.shape, x.shape)

    def test_llama_mlp(self):
        """测试 LLaMA MLP"""
        mlp = LLaMAMLP(self.config)
        x = torch.randn(2, 20, 512)
        output = mlp(x)
        self.assertEqual(output.shape, x.shape)

    def test_llama_decoder_layer(self):
        """测试 LLaMA 解码器层"""
        layer = LLaMADecoderLayer(self.config)
        x = torch.randn(2, 20, 512)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)


class TestLLaMAModel(unittest.TestCase):
    """测试 LLaMA 模型"""

    def setUp(self):
        self.config = LLaVAConfig(
            vocab_size=1000, max_seq_length=256,
            hidden_size=512, num_layers=2, num_heads=8,
            intermediate_size=1376
        )
        self.model = LLaMAModel(self.config)

    def test_forward_with_input_ids(self):
        """测试使用 input_ids 前向传播"""
        input_ids = torch.randint(0, 1000, (2, 20))
        output = self.model(input_ids=input_ids)
        self.assertEqual(output.shape, (2, 20, 1000))

    def test_forward_with_inputs_embeds(self):
        """测试使用 inputs_embeds 前向传播"""
        inputs_embeds = torch.randn(2, 20, 512)
        output = self.model(inputs_embeds=inputs_embeds)
        self.assertEqual(output.shape, (2, 20, 1000))


class TestLLaVA(unittest.TestCase):
    """测试完整 LLaVA 模型"""

    def setUp(self):
        self.config = LLaVAConfig(
            image_size=224, patch_size=14,
            vision_layers=2, vision_width=384, vision_heads=6,
            vocab_size=1000, max_seq_length=512,  # 需要足够大以容纳图像+文本
            hidden_size=384, num_layers=2, num_heads=6,
            intermediate_size=1024,
            projector_type="mlp2x_gelu"
        )
        self.model = LLaVA(self.config)

    def test_get_vision_features(self):
        """测试获取视觉特征"""
        images = torch.randn(2, 3, 224, 224)
        features = self.model.get_vision_features(images)
        num_patches = (224 // 14) ** 2  # 256 (不含 CLS)
        self.assertEqual(features.shape, (2, num_patches, 384))

    def test_prepare_inputs_embeds_without_images(self):
        """测试不带图像的输入嵌入"""
        input_ids = torch.randint(0, 1000, (2, 20))
        embeds = self.model.prepare_inputs_embeds(input_ids)
        self.assertEqual(embeds.shape, (2, 20, 384))

    def test_prepare_inputs_embeds_with_images(self):
        """测试带图像的输入嵌入"""
        input_ids = torch.randint(0, 1000, (2, 20))
        images = torch.randn(2, 3, 224, 224)
        embeds = self.model.prepare_inputs_embeds(input_ids, images)
        num_patches = (224 // 14) ** 2  # 256
        expected_len = num_patches + 20
        self.assertEqual(embeds.shape, (2, expected_len, 384))

    def test_forward_text_only(self):
        """测试纯文本前向传播"""
        input_ids = torch.randint(0, 1000, (2, 20))
        output = self.model(input_ids)
        self.assertIn("logits", output)
        self.assertEqual(output["logits"].shape, (2, 20, 1000))

    def test_forward_with_images(self):
        """测试带图像的前向传播"""
        input_ids = torch.randint(0, 1000, (2, 20))
        images = torch.randn(2, 3, 224, 224)
        output = self.model(input_ids, images)
        self.assertIn("logits", output)
        num_patches = (224 // 14) ** 2
        expected_len = num_patches + 20
        self.assertEqual(output["logits"].shape, (2, expected_len, 1000))

    def test_forward_with_labels(self):
        """测试带标签的前向传播"""
        input_ids = torch.randint(0, 1000, (2, 20))
        labels = torch.randint(0, 1000, (2, 20))
        output = self.model(input_ids, labels=labels)
        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"].item(), float)


class TestCreateLLaVAModel(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_tiny(self):
        """测试创建 tiny 模型"""
        model = create_llava_model("tiny")
        self.assertIsInstance(model, LLaVA)
        self.assertEqual(model.config.hidden_size, 512)

    def test_create_small(self):
        """测试创建 small 模型"""
        model = create_llava_model("small")
        self.assertIsInstance(model, LLaVA)
        self.assertEqual(model.config.hidden_size, 1024)

    def test_invalid_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_llava_model("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
