"""
BLIP 模型单元测试

测试覆盖:
    - BLIPConfig 配置验证
    - VisionEncoder 视觉编码器
    - TextEncoder 文本编码器
    - TextDecoder 文本解码器
    - BLIP 完整模型
    - 损失函数
"""

import unittest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blip import (
    BLIPConfig,
    PatchEmbedding,
    MultiHeadAttention,
    MLP,
    TransformerEncoderBlock,
    TransformerDecoderBlock,
    VisionEncoder,
    TextEncoder,
    TextDecoder,
    BLIP,
    itc_loss,
    itm_loss,
    lm_loss,
    create_blip_model,
)


class TestBLIPConfig(unittest.TestCase):
    """测试 BLIP 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = BLIPConfig()
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.patch_size, 16)
        self.assertEqual(config.embed_dim, 256)

    def test_custom_config(self):
        """测试自定义配置"""
        config = BLIPConfig(image_size=384, patch_size=32, embed_dim=512)
        self.assertEqual(config.image_size, 384)
        self.assertEqual(config.patch_size, 32)
        self.assertEqual(config.embed_dim, 512)

    def test_invalid_patch_size(self):
        """测试无效的 patch_size"""
        with self.assertRaises(AssertionError):
            BLIPConfig(image_size=224, patch_size=15)


class TestPatchEmbedding(unittest.TestCase):
    """测试图像分块嵌入"""

    def setUp(self):
        self.config = BLIPConfig(image_size=224, patch_size=16, vision_width=768)
        self.patch_embed = PatchEmbedding(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        output = self.patch_embed(x)

        num_patches = (224 // 16) ** 2  # 196
        expected_shape = (batch_size, num_patches + 1, 768)
        self.assertEqual(output.shape, expected_shape)


class TestMultiHeadAttention(unittest.TestCase):
    """测试多头注意力"""

    def setUp(self):
        self.attn = MultiHeadAttention(d_model=512, num_heads=8)

    def test_self_attention(self):
        """测试自注意力"""
        batch_size, seq_len = 4, 77
        x = torch.randn(batch_size, seq_len, 512)
        output = self.attn(x, x, x)
        self.assertEqual(output.shape, x.shape)

    def test_cross_attention(self):
        """测试交叉注意力"""
        batch_size = 4
        q = torch.randn(batch_size, 20, 512)
        kv = torch.randn(batch_size, 197, 512)
        output = self.attn(q, kv, kv)
        self.assertEqual(output.shape, q.shape)

    def test_causal_mask(self):
        """测试因果掩码"""
        x = torch.randn(2, 10, 512)
        output = self.attn(x, x, x, causal=True)
        self.assertEqual(output.shape, x.shape)


class TestTransformerBlocks(unittest.TestCase):
    """测试 Transformer 块"""

    def test_encoder_block(self):
        """测试编码器块"""
        block = TransformerEncoderBlock(d_model=512, num_heads=8)
        x = torch.randn(4, 77, 512)
        output = block(x)
        self.assertEqual(output.shape, x.shape)

    def test_decoder_block(self):
        """测试解码器块"""
        block = TransformerDecoderBlock(d_model=512, num_heads=8)
        x = torch.randn(4, 20, 512)
        encoder_hidden = torch.randn(4, 197, 512)
        output = block(x, encoder_hidden)
        self.assertEqual(output.shape, x.shape)


class TestVisionEncoder(unittest.TestCase):
    """测试视觉编码器"""

    def setUp(self):
        self.config = BLIPConfig(
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
        num_patches = (224 // 16) ** 2 + 1  # 197
        self.assertEqual(output.shape, (batch_size, num_patches, 384))


class TestTextEncoder(unittest.TestCase):
    """测试文本编码器"""

    def setUp(self):
        self.config = BLIPConfig(
            vocab_size=1000, max_text_length=77,
            text_layers=2, text_width=256, text_heads=4,
            embed_dim=256
        )
        self.encoder = TextEncoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        batch_size, seq_len = 4, 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        output = self.encoder(input_ids)
        self.assertEqual(output.shape, (batch_size, seq_len, 256))

    def test_with_attention_mask(self):
        """测试带注意力掩码"""
        batch_size, seq_len = 4, 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 15:] = 0

        output = self.encoder(input_ids, attention_mask)
        self.assertEqual(output.shape, (batch_size, seq_len, 256))


class TestTextDecoder(unittest.TestCase):
    """测试文本解码器"""

    def setUp(self):
        self.config = BLIPConfig(
            vocab_size=1000, max_text_length=77,
            decoder_layers=2, decoder_width=256, decoder_heads=4,
            vision_width=256
        )
        self.decoder = TextDecoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        batch_size, seq_len = 4, 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        encoder_hidden = torch.randn(batch_size, 197, 256)
        output = self.decoder(input_ids, encoder_hidden)
        self.assertEqual(output.shape, (batch_size, seq_len, 1000))


class TestBLIP(unittest.TestCase):
    """测试完整 BLIP 模型"""

    def setUp(self):
        self.config = BLIPConfig(
            image_size=224, patch_size=16,
            vision_layers=2, vision_width=384, vision_heads=6,
            vocab_size=1000, max_text_length=77,
            text_layers=2, text_width=384, text_heads=6,
            decoder_layers=2, decoder_width=384, decoder_heads=6,
            embed_dim=256
        )
        self.model = BLIP(self.config)

    def test_encode_image(self):
        """测试图像编码"""
        images = torch.randn(4, 3, 224, 224)
        features = self.model.encode_image(images)
        self.assertEqual(features.shape, (4, 197, 384))

    def test_encode_text(self):
        """测试文本编码"""
        input_ids = torch.randint(0, 1000, (4, 20))
        features = self.model.encode_text(input_ids)
        self.assertEqual(features.shape, (4, 20, 384))

    def test_get_image_features(self):
        """测试获取图像特征"""
        images = torch.randn(4, 3, 224, 224)
        features = self.model.get_image_features(images)
        self.assertEqual(features.shape, (4, 256))
        # 检查归一化
        norms = torch.norm(features, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_get_text_features(self):
        """测试获取文本特征"""
        input_ids = torch.randint(0, 1000, (4, 20))
        features = self.model.get_text_features(input_ids)
        self.assertEqual(features.shape, (4, 256))
        # 检查归一化
        norms = torch.norm(features, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_forward_itc(self):
        """测试 ITC 前向传播"""
        images = torch.randn(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (4, 20))

        image_feat, text_feat, logit_scale = self.model.forward_itc(images, input_ids)

        self.assertEqual(image_feat.shape, (4, 256))
        self.assertEqual(text_feat.shape, (4, 256))
        self.assertIsInstance(logit_scale.item(), float)

    def test_forward_itm(self):
        """测试 ITM 前向传播"""
        images = torch.randn(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (4, 20))

        itm_logits = self.model.forward_itm(images, input_ids)
        self.assertEqual(itm_logits.shape, (4, 2))

    def test_forward_lm(self):
        """测试 LM 前向传播"""
        images = torch.randn(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (4, 20))

        logits = self.model.forward_lm(images, input_ids)
        self.assertEqual(logits.shape, (4, 20, 1000))

    def test_generate(self):
        """测试生成"""
        images = torch.randn(2, 3, 224, 224)
        generated = self.model.generate(images, max_length=10)
        self.assertEqual(generated.shape[0], 2)
        self.assertLessEqual(generated.shape[1], 10)


class TestBLIPLoss(unittest.TestCase):
    """测试 BLIP 损失函数"""

    def test_itc_loss(self):
        """测试 ITC 损失"""
        batch_size = 8
        embed_dim = 256

        image_features = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_features = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        logit_scale = torch.tensor(14.0).exp()

        loss = itc_loss(image_features, text_features, logit_scale)

        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)

    def test_itm_loss(self):
        """测试 ITM 损失"""
        batch_size = 8
        itm_logits = torch.randn(batch_size, 2)
        labels = torch.randint(0, 2, (batch_size,))

        loss = itm_loss(itm_logits, labels)

        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)

    def test_lm_loss(self):
        """测试 LM 损失"""
        batch_size, seq_len, vocab_size = 4, 20, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = lm_loss(logits, labels)

        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)


class TestCreateBLIPModel(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_small(self):
        """测试创建小模型"""
        model = create_blip_model("small")
        self.assertIsInstance(model, BLIP)
        self.assertEqual(model.config.embed_dim, 256)

    def test_create_base(self):
        """测试创建基础模型"""
        model = create_blip_model("base")
        self.assertIsInstance(model, BLIP)
        self.assertEqual(model.config.embed_dim, 256)

    def test_create_large(self):
        """测试创建大模型"""
        model = create_blip_model("large")
        self.assertIsInstance(model, BLIP)
        self.assertEqual(model.config.embed_dim, 512)

    def test_invalid_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_blip_model("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
