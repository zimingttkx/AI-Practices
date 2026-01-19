"""
4M 模型单元测试

测试覆盖:
    - FourMConfig 配置验证
    - VectorQuantizer 向量量化
    - VQVAEEncoder/Decoder 编解码器
    - ModalityTokenizer 多模态分词
    - FourMEncoder/Decoder Transformer
    - FourMLoss 损失函数
    - FourM 完整模型
"""

import unittest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fourm import (
    FourMConfig,
    VectorQuantizer,
    VQVAEEncoder,
    VQVAEDecoder,
    ModalityTokenizer,
    MultiHeadAttention,
    FeedForward,
    FourMEncoderLayer,
    FourMDecoderLayer,
    FourMEncoder,
    FourMDecoder,
    FourMLoss,
    FourM,
    create_fourm_model,
)


class TestFourMConfig(unittest.TestCase):
    """测试 FourM 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = FourMConfig()
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.d_model, 768)
        self.assertEqual(config.n_heads, 12)

    def test_custom_config(self):
        """测试自定义配置"""
        config = FourMConfig(image_size=128, d_model=256)
        self.assertEqual(config.image_size, 128)
        self.assertEqual(config.d_model, 256)

    def test_num_patches_calculation(self):
        """测试 patch 数量计算"""
        config = FourMConfig(image_size=224, patch_size=16)
        self.assertEqual(config.num_patches, 196)  # (224/16)^2


class TestVectorQuantizer(unittest.TestCase):
    """测试向量量化器"""

    def setUp(self):
        self.vq = VectorQuantizer(
            codebook_size=512,
            codebook_dim=64,
            commitment_cost=0.25
        )

    def test_output_shape(self):
        """测试输出形状"""
        z = torch.randn(2, 8, 8, 64)
        quantized, indices, loss = self.vq(z)
        self.assertEqual(quantized.shape, z.shape)
        self.assertEqual(indices.shape, (2, 8, 8))

    def test_indices_range(self):
        """测试索引范围"""
        z = torch.randn(2, 8, 8, 64)
        _, indices, _ = self.vq(z)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < 512).all())

    def test_loss_positive(self):
        """测试损失为正"""
        z = torch.randn(2, 8, 8, 64)
        _, _, loss = self.vq(z)
        self.assertGreater(loss.item(), 0)

    def test_get_codebook_entry(self):
        """测试获取 codebook entry"""
        indices = torch.randint(0, 512, (2, 8, 8))
        entries = self.vq.get_codebook_entry(indices)
        self.assertEqual(entries.shape, (2, 8, 8, 64))

    def test_ema_update(self):
        """测试 EMA 更新"""
        self.vq.train()
        z = torch.randn(2, 8, 8, 64)
        old_embedding = self.vq.embedding.weight.data.clone()
        _, _, _ = self.vq(z)
        # EMA 更新后 embedding 应该改变
        self.assertFalse(torch.allclose(old_embedding, self.vq.embedding.weight.data))


class TestVQVAEEncoder(unittest.TestCase):
    """测试 VQ-VAE 编码器"""

    def setUp(self):
        self.encoder = VQVAEEncoder(
            in_channels=3,
            hidden_channels=64,
            codebook_dim=128,
            num_downsamples=4
        )

    def test_output_shape(self):
        """测试输出形状"""
        x = torch.randn(2, 3, 256, 256)
        z = self.encoder(x)
        # 4次下采样: 256 -> 128 -> 64 -> 32 -> 16
        self.assertEqual(z.shape, (2, 16, 16, 128))

    def test_different_input_sizes(self):
        """测试不同输入尺寸"""
        for size in [64, 128, 256]:
            x = torch.randn(1, 3, size, size)
            z = self.encoder(x)
            expected_size = size // 16  # 4次下采样
            self.assertEqual(z.shape[1], expected_size)


class TestVQVAEDecoder(unittest.TestCase):
    """测试 VQ-VAE 解码器"""

    def setUp(self):
        self.decoder = VQVAEDecoder(
            out_channels=3,
            hidden_channels=64,
            codebook_dim=128,
            num_upsamples=4
        )

    def test_output_shape(self):
        """测试输出形状"""
        z = torch.randn(2, 16, 16, 128)
        x = self.decoder(z)
        # 4次上采样: 16 -> 32 -> 64 -> 128 -> 256
        self.assertEqual(x.shape, (2, 3, 256, 256))

    def test_reconstruction_shape(self):
        """测试重建形状匹配"""
        encoder = VQVAEEncoder(3, 64, 128, 4)
        x = torch.randn(2, 3, 256, 256)
        z = encoder(x)
        recon = self.decoder(z)
        self.assertEqual(recon.shape, x.shape)


class TestModalityTokenizer(unittest.TestCase):
    """测试多模态分词器"""

    def setUp(self):
        self.config = FourMConfig(
            image_size=64,
            codebook_size=512,
            codebook_dim=64,
            modalities=["rgb", "depth"]
        )
        self.tokenizer = ModalityTokenizer(self.config)

    def test_tokenize_rgb(self):
        """测试 RGB tokenize"""
        x = torch.randn(2, 3, 64, 64)
        tokens, vq_loss = self.tokenizer.tokenize(x, "rgb")
        self.assertEqual(tokens.dim(), 3)  # [B, h, w]
        self.assertGreater(vq_loss.item(), 0)

    def test_tokenize_depth(self):
        """测试 Depth tokenize"""
        x = torch.randn(2, 1, 64, 64)
        tokens, vq_loss = self.tokenizer.tokenize(x, "depth")
        self.assertEqual(tokens.dim(), 3)  # [B, h, w]

    def test_detokenize(self):
        """测试 detokenize"""
        x = torch.randn(2, 3, 64, 64)
        tokens, _ = self.tokenizer.tokenize(x, "rgb")
        h = w = 4  # 64 / 16 = 4
        tokens_2d = tokens.view(2, h, w)
        recon = self.tokenizer.detokenize(tokens_2d, "rgb")
        self.assertEqual(recon.shape, x.shape)

    def test_encode_decode(self):
        """测试编码解码"""
        x = torch.randn(2, 3, 64, 64)
        quantized, tokens, vq_loss = self.tokenizer.encode(x, "rgb")
        recon = self.tokenizer.decode(quantized, "rgb")
        self.assertEqual(recon.shape, x.shape)

    def test_different_modalities(self):
        """测试不同模态"""
        config = FourMConfig(
            image_size=64,
            modalities=["rgb", "depth", "normal"],
            num_semantic_classes=20
        )
        tokenizer = ModalityTokenizer(config)
        
        # RGB
        rgb = torch.randn(1, 3, 64, 64)
        tokens_rgb, _ = tokenizer.tokenize(rgb, "rgb")
        
        # Depth
        depth = torch.randn(1, 1, 64, 64)
        tokens_depth, _ = tokenizer.tokenize(depth, "depth")
        
        # Normal
        normal = torch.randn(1, 3, 64, 64)
        tokens_normal, _ = tokenizer.tokenize(normal, "normal")
        
        self.assertEqual(tokens_rgb.shape, tokens_depth.shape)
        self.assertEqual(tokens_rgb.shape, tokens_normal.shape)


class TestMultiHeadAttention(unittest.TestCase):
    """测试多头注意力"""

    def test_self_attention(self):
        """测试自注意力"""
        attn = MultiHeadAttention(d_model=256, n_heads=4)
        x = torch.randn(2, 50, 256)
        output = attn(x, x, x)
        self.assertEqual(output.shape, x.shape)

    def test_cross_attention(self):
        """测试交叉注意力"""
        attn = MultiHeadAttention(d_model=256, n_heads=4)
        q = torch.randn(2, 30, 256)
        kv = torch.randn(2, 50, 256)
        output = attn(q, kv, kv)
        self.assertEqual(output.shape, q.shape)

    def test_causal_attention(self):
        """测试因果注意力"""
        attn = MultiHeadAttention(d_model=256, n_heads=4, is_causal=True)
        x = torch.randn(2, 50, 256)
        output = attn(x, x, x)
        self.assertEqual(output.shape, x.shape)


class TestFeedForward(unittest.TestCase):
    """测试前馈网络"""

    def test_output_shape(self):
        """测试输出形状"""
        ff = FeedForward(d_model=256, d_ff=1024)
        x = torch.randn(2, 50, 256)
        output = ff(x)
        self.assertEqual(output.shape, x.shape)


class TestFourMEncoderLayer(unittest.TestCase):
    """测试编码器层"""

    def test_output_shape(self):
        """测试输出形状"""
        config = FourMConfig(d_model=256, n_heads=4, d_ff=1024)
        layer = FourMEncoderLayer(config)
        x = torch.randn(2, 50, 256)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)


class TestFourMDecoderLayer(unittest.TestCase):
    """测试解码器层"""

    def test_output_shape(self):
        """测试输出形状"""
        config = FourMConfig(d_model=256, n_heads=4, d_ff=1024)
        layer = FourMDecoderLayer(config)
        x = torch.randn(2, 30, 256)
        encoder_output = torch.randn(2, 50, 256)
        output = layer(x, encoder_output)
        self.assertEqual(output.shape, x.shape)


class TestFourMEncoder(unittest.TestCase):
    """测试 4M 编码器"""

    def setUp(self):
        self.config = FourMConfig(
            image_size=64,
            patch_size=8,
            codebook_size=512,
            d_model=128,
            n_heads=4,
            n_encoder_layers=2,
            d_ff=512,
            modalities=["rgb", "depth"]
        )
        self.encoder = FourMEncoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        tokens = {
            "rgb": torch.randint(0, 512, (2, 16)),
            "depth": torch.randint(0, 512, (2, 16))
        }
        output = self.encoder(tokens)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 32)  # 16 + 16
        self.assertEqual(output.shape[2], 128)

    def test_with_mask(self):
        """测试带 mask"""
        tokens = {
            "rgb": torch.randint(0, 512, (2, 16)),
            "depth": torch.randint(0, 512, (2, 16))
        }
        mask_dict = {
            "rgb": torch.rand(2, 16) > 0.5,
            "depth": torch.rand(2, 16) > 0.5
        }
        output = self.encoder(tokens, mask_dict)
        self.assertEqual(output.shape[1], 32)


class TestFourMDecoder(unittest.TestCase):
    """测试 4M 解码器"""

    def setUp(self):
        self.config = FourMConfig(
            image_size=64,
            patch_size=8,
            codebook_size=512,
            d_model=128,
            n_heads=4,
            n_decoder_layers=2,
            d_ff=512,
            modalities=["rgb", "depth"]
        )
        self.decoder = FourMDecoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        encoder_output = torch.randn(2, 32, 128)
        target_tokens = torch.randint(0, 512, (2, 16))
        logits = self.decoder(encoder_output, target_tokens, target_modality_idx=0)
        self.assertEqual(logits.shape, (2, 16, 512))

    def test_generate(self):
        """测试生成"""
        encoder_output = torch.randn(2, 32, 128)
        generated = self.decoder.generate(
            encoder_output, target_modality_idx=0, max_len=16, temperature=1.0
        )
        self.assertEqual(generated.shape, (2, 16))

    def test_generate_greedy(self):
        """测试贪婪生成"""
        encoder_output = torch.randn(2, 32, 128)
        generated = self.decoder.generate(
            encoder_output, target_modality_idx=0, max_len=16, temperature=0
        )
        self.assertEqual(generated.shape, (2, 16))


class TestFourMLoss(unittest.TestCase):
    """测试损失函数"""

    def setUp(self):
        self.config = FourMConfig(codebook_size=512)
        self.loss_fn = FourMLoss(self.config)

    def test_loss_computation(self):
        """测试损失计算"""
        logits = torch.randn(2, 16, 512)
        targets = torch.randint(0, 512, (2, 16))
        vq_loss = torch.tensor(0.1)
        
        total_loss, loss_dict = self.loss_fn(logits, targets, vq_loss)
        
        self.assertIn("total", loss_dict)
        self.assertIn("recon", loss_dict)
        self.assertIn("vq", loss_dict)
        self.assertGreater(total_loss.item(), 0)

    def test_loss_with_mask(self):
        """测试带 mask 的损失"""
        logits = torch.randn(2, 16, 512)
        targets = torch.randint(0, 512, (2, 16))
        vq_loss = torch.tensor(0.1)
        mask = torch.rand(2, 16) > 0.5
        
        total_loss, loss_dict = self.loss_fn(logits, targets, vq_loss, mask.float())
        self.assertGreater(total_loss.item(), 0)

    def test_gradient_flow(self):
        """测试梯度流"""
        logits = torch.randn(2, 16, 512, requires_grad=True)
        targets = torch.randint(0, 512, (2, 16))
        vq_loss = torch.tensor(0.1, requires_grad=True)
        
        total_loss, _ = self.loss_fn(logits, targets, vq_loss)
        total_loss.backward()
        
        self.assertIsNotNone(logits.grad)


class TestFourM(unittest.TestCase):
    """测试 4M 完整模型"""

    def setUp(self):
        self.config = FourMConfig(
            image_size=64,
            patch_size=8,
            codebook_size=256,
            codebook_dim=64,
            d_model=128,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=512,
            modalities=["rgb", "depth"]
        )
        self.model = FourM(self.config)

    def test_tokenize(self):
        """测试 tokenize"""
        inputs = {
            "rgb": torch.randn(2, 3, 64, 64),
            "depth": torch.randn(2, 1, 64, 64)
        }
        tokens, vq_loss = self.model.tokenize(inputs)
        
        self.assertIn("rgb", tokens)
        self.assertIn("depth", tokens)
        self.assertGreater(vq_loss.item(), 0)

    def test_forward(self):
        """测试前向传播"""
        inputs = {
            "rgb": torch.randn(2, 3, 64, 64),
            "depth": torch.randn(2, 1, 64, 64)
        }
        loss, loss_dict = self.model(inputs, target_modality="rgb")
        
        self.assertGreater(loss.item(), 0)
        self.assertIn("total", loss_dict)

    def test_encode(self):
        """测试编码"""
        inputs = {
            "rgb": torch.randn(2, 3, 64, 64),
            "depth": torch.randn(2, 1, 64, 64)
        }
        encoder_output = self.model.encode(inputs)
        self.assertEqual(encoder_output.dim(), 3)

    def test_create_random_mask(self):
        """测试随机 mask 创建"""
        tokens = {
            "rgb": torch.randint(0, 256, (2, 16)),
            "depth": torch.randint(0, 256, (2, 16))
        }
        mask_dict = self.model.create_random_mask(tokens, mask_ratio=0.5)
        
        self.assertIn("rgb", mask_dict)
        self.assertIn("depth", mask_dict)
        # 检查 mask 比例大约是 0.5
        mask_ratio = mask_dict["rgb"].float().mean().item()
        self.assertGreater(mask_ratio, 0.3)
        self.assertLess(mask_ratio, 0.7)


class TestCreateFourMModel(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_tiny(self):
        """测试创建 tiny 模型"""
        model = create_fourm_model("tiny")
        self.assertEqual(model.config.d_model, 256)
        self.assertEqual(model.config.image_size, 64)

    def test_create_small(self):
        """测试创建 small 模型"""
        model = create_fourm_model("small")
        self.assertEqual(model.config.d_model, 512)

    def test_create_base(self):
        """测试创建 base 模型"""
        model = create_fourm_model("base")
        self.assertEqual(model.config.d_model, 768)

    def test_invalid_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_fourm_model("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)

