"""
CLIP 模型单元测试

测试覆盖:
    - CLIPConfig 配置验证
    - PatchEmbedding 图像分块
    - MultiHeadAttention 注意力机制
    - VisionEncoder 视觉编码器
    - TextEncoder 文本编码器
    - CLIP 完整模型
    - clip_loss 对比损失
"""

import unittest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clip import (
    CLIPConfig,
    PatchEmbedding,
    MultiHeadAttention,
    MLP,
    TransformerBlock,
    VisionEncoder,
    TextEncoder,
    CLIP,
    clip_loss,
    create_clip_model,
)


class TestCLIPConfig(unittest.TestCase):
    """测试 CLIP 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = CLIPConfig()
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.patch_size, 16)
        self.assertEqual(config.embed_dim, 512)

    def test_custom_config(self):
        """测试自定义配置"""
        config = CLIPConfig(image_size=384, patch_size=32, embed_dim=768)
        self.assertEqual(config.image_size, 384)
        self.assertEqual(config.patch_size, 32)
        self.assertEqual(config.embed_dim, 768)

    def test_invalid_patch_size(self):
        """测试无效的 patch_size"""
        with self.assertRaises(AssertionError):
            CLIPConfig(image_size=224, patch_size=15)  # 224 不能被 15 整除


class TestPatchEmbedding(unittest.TestCase):
    """测试图像分块嵌入"""

    def setUp(self):
        self.config = CLIPConfig(image_size=224, patch_size=16, vision_width=768)
        self.patch_embed = PatchEmbedding(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        output = self.patch_embed(x)

        num_patches = (224 // 16) ** 2  # 196
        expected_shape = (batch_size, num_patches + 1, 768)  # +1 for CLS token
        self.assertEqual(output.shape, expected_shape)

    def test_cls_token_position(self):
        """测试 CLS token 位置"""
        x = torch.randn(2, 3, 224, 224)
        output = self.patch_embed(x)

        # CLS token 应该在第一个位置
        self.assertEqual(output.shape[1], 197)  # 196 patches + 1 CLS


class TestMultiHeadAttention(unittest.TestCase):
    """测试多头注意力"""

    def setUp(self):
        self.attn = MultiHeadAttention(d_model=512, num_heads=8)

    def test_output_shape(self):
        """测试输出形状"""
        batch_size, seq_len = 4, 77
        x = torch.randn(batch_size, seq_len, 512)
        output = self.attn(x)
        self.assertEqual(output.shape, x.shape)

    def test_causal_mask(self):
        """测试因果掩码"""
        x = torch.randn(2, 10, 512)
        output_causal = self.attn(x, causal=True)
        output_normal = self.attn(x, causal=False)

        # 输出形状应该相同
        self.assertEqual(output_causal.shape, output_normal.shape)

    def test_attention_mask(self):
        """测试注意力掩码"""
        x = torch.randn(2, 10, 512)
        mask = torch.ones(2, 10)
        mask[:, 5:] = 0  # 后半部分被掩码

        output = self.attn(x, attention_mask=mask)
        self.assertEqual(output.shape, x.shape)


class TestMLP(unittest.TestCase):
    """测试前馈网络"""

    def test_output_shape(self):
        """测试输出形状"""
        mlp = MLP(d_model=512, hidden_dim=2048)
        x = torch.randn(4, 77, 512)
        output = mlp(x)
        self.assertEqual(output.shape, x.shape)


class TestTransformerBlock(unittest.TestCase):
    """测试 Transformer 块"""

    def test_output_shape(self):
        """测试输出形状"""
        block = TransformerBlock(d_model=512, num_heads=8)
        x = torch.randn(4, 77, 512)
        output = block(x)
        self.assertEqual(output.shape, x.shape)

    def test_causal_block(self):
        """测试因果 Transformer 块"""
        block = TransformerBlock(d_model=512, num_heads=8, causal=True)
        x = torch.randn(4, 77, 512)
        output = block(x)
        self.assertEqual(output.shape, x.shape)


class TestVisionEncoder(unittest.TestCase):
    """测试视觉编码器"""

    def setUp(self):
        self.config = CLIPConfig(
            image_size=224, patch_size=16,
            vision_layers=2, vision_width=384, vision_heads=6,
            embed_dim=256
        )
        self.encoder = VisionEncoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        output = self.encoder(images)
        self.assertEqual(output.shape, (batch_size, 256))

    def test_different_batch_sizes(self):
        """测试不同批次大小"""
        for batch_size in [1, 2, 8]:
            images = torch.randn(batch_size, 3, 224, 224)
            output = self.encoder(images)
            self.assertEqual(output.shape, (batch_size, 256))


class TestTextEncoder(unittest.TestCase):
    """测试文本编码器"""

    def setUp(self):
        self.config = CLIPConfig(
            vocab_size=1000, context_length=77,
            text_layers=2, text_width=256, text_heads=4,
            embed_dim=256
        )
        self.encoder = TextEncoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        batch_size, seq_len = 4, 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        output = self.encoder(input_ids)
        self.assertEqual(output.shape, (batch_size, 256))

    def test_with_attention_mask(self):
        """测试带注意力掩码"""
        batch_size, seq_len = 4, 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 15:] = 0

        output = self.encoder(input_ids, attention_mask)
        self.assertEqual(output.shape, (batch_size, 256))


class TestCLIP(unittest.TestCase):
    """测试完整 CLIP 模型"""

    def setUp(self):
        self.config = CLIPConfig(
            image_size=224, patch_size=16,
            vision_layers=2, vision_width=384, vision_heads=6,
            vocab_size=1000, context_length=77,
            text_layers=2, text_width=256, text_heads=4,
            embed_dim=256
        )
        self.model = CLIP(self.config)

    def test_encode_image(self):
        """测试图像编码"""
        images = torch.randn(4, 3, 224, 224)
        features = self.model.encode_image(images)
        self.assertEqual(features.shape, (4, 256))

    def test_encode_text(self):
        """测试文本编码"""
        input_ids = torch.randint(0, 1000, (4, 20))
        features = self.model.encode_text(input_ids)
        self.assertEqual(features.shape, (4, 256))

    def test_forward(self):
        """测试前向传播"""
        images = torch.randn(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (4, 20))

        image_features, text_features, logit_scale = self.model(images, input_ids)

        self.assertEqual(image_features.shape, (4, 256))
        self.assertEqual(text_features.shape, (4, 256))
        self.assertIsInstance(logit_scale.item(), float)

    def test_features_normalized(self):
        """测试特征是否归一化"""
        images = torch.randn(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (4, 20))

        image_features, text_features, _ = self.model(images, input_ids)

        # 检查 L2 范数是否接近 1
        image_norms = torch.norm(image_features, dim=-1)
        text_norms = torch.norm(text_features, dim=-1)

        self.assertTrue(torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-5))
        self.assertTrue(torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-5))

    def test_get_similarity(self):
        """测试相似度计算"""
        images = torch.randn(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (4, 20))

        image_features, text_features, _ = self.model(images, input_ids)
        similarity = self.model.get_similarity(image_features, text_features)

        self.assertEqual(similarity.shape, (4, 4))


class TestCLIPLoss(unittest.TestCase):
    """测试 CLIP 损失函数"""

    def test_loss_computation(self):
        """测试损失计算"""
        batch_size = 8
        embed_dim = 256

        image_features = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_features = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        logit_scale = torch.tensor(14.0).exp()

        loss = clip_loss(image_features, text_features, logit_scale)

        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)

    def test_perfect_alignment(self):
        """测试完美对齐时的损失"""
        batch_size = 4
        embed_dim = 256

        # 使用相同的特征
        features = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        logit_scale = torch.tensor(14.0).exp()

        loss = clip_loss(features, features, logit_scale)

        # 完美对齐时损失应该很小
        self.assertLess(loss.item(), 1.0)


class TestCreateCLIPModel(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_small(self):
        """测试创建小模型"""
        model = create_clip_model("small")
        self.assertIsInstance(model, CLIP)
        self.assertEqual(model.config.embed_dim, 256)

    def test_create_base(self):
        """测试创建基础模型"""
        model = create_clip_model("base")
        self.assertIsInstance(model, CLIP)
        self.assertEqual(model.config.embed_dim, 512)

    def test_create_large(self):
        """测试创建大模型"""
        model = create_clip_model("large")
        self.assertIsInstance(model, CLIP)
        self.assertEqual(model.config.embed_dim, 768)

    def test_invalid_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_clip_model("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
