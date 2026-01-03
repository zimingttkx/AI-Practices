"""
VAE 模型单元测试

测试覆盖:
    - VAEConfig 配置验证
    - ResidualBlock 残差块
    - AttentionBlock 注意力块
    - Encoder 编码器
    - Decoder 解码器
    - VAE 完整模型
    - 损失函数
"""

import unittest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vae import (
    VAEConfig,
    ResidualBlock,
    AttentionBlock,
    Encoder,
    Decoder,
    VAE,
    vae_loss,
    create_vae_model,
)


class TestVAEConfig(unittest.TestCase):
    """测试 VAE 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = VAEConfig()
        self.assertEqual(config.image_size, 256)
        self.assertEqual(config.latent_channels, 4)

    def test_custom_config(self):
        """测试自定义配置"""
        config = VAEConfig(image_size=128, latent_channels=8)
        self.assertEqual(config.image_size, 128)
        self.assertEqual(config.latent_channels, 8)

    def test_invalid_image_size(self):
        """测试无效的图像尺寸"""
        with self.assertRaises(AssertionError):
            VAEConfig(image_size=8, encoder_channels=(64, 128, 256, 512))


class TestResidualBlock(unittest.TestCase):
    """测试残差块"""

    def test_same_channels(self):
        """测试相同通道数"""
        block = ResidualBlock(64, 64)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        self.assertEqual(output.shape, x.shape)

    def test_different_channels(self):
        """测试不同通道数"""
        block = ResidualBlock(64, 128)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        self.assertEqual(output.shape, (2, 128, 32, 32))

    def test_downsample(self):
        """测试下采样"""
        block = ResidualBlock(64, 128, downsample=True)
        x = torch.randn(2, 64, 32, 32)
        output = block(x)
        self.assertEqual(output.shape, (2, 128, 16, 16))

    def test_upsample(self):
        """测试上采样"""
        block = ResidualBlock(128, 64, upsample=True)
        x = torch.randn(2, 128, 16, 16)
        output = block(x)
        self.assertEqual(output.shape, (2, 64, 32, 32))


class TestAttentionBlock(unittest.TestCase):
    """测试注意力块"""

    def test_output_shape(self):
        """测试输出形状"""
        block = AttentionBlock(256, num_heads=8)
        x = torch.randn(2, 256, 16, 16)
        output = block(x)
        self.assertEqual(output.shape, x.shape)


class TestEncoder(unittest.TestCase):
    """测试编码器"""

    def setUp(self):
        self.config = VAEConfig(
            image_size=64,
            latent_channels=4,
            encoder_channels=(32, 64, 128, 256)
        )
        self.encoder = Encoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        x = torch.randn(2, 3, 64, 64)
        mu, logvar = self.encoder(x)
        # 64 / 8 = 8 (3次下采样)
        self.assertEqual(mu.shape, (2, 4, 8, 8))
        self.assertEqual(logvar.shape, (2, 4, 8, 8))


class TestDecoder(unittest.TestCase):
    """测试解码器"""

    def setUp(self):
        self.config = VAEConfig(
            image_size=64,
            latent_channels=4,
            decoder_channels=(256, 128, 64, 32)
        )
        self.decoder = Decoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        z = torch.randn(2, 4, 8, 8)
        output = self.decoder(z)
        self.assertEqual(output.shape, (2, 3, 64, 64))


class TestVAE(unittest.TestCase):
    """测试完整 VAE 模型"""

    def setUp(self):
        self.config = VAEConfig(
            image_size=64,
            latent_channels=4,
            encoder_channels=(32, 64, 128, 256),
            decoder_channels=(256, 128, 64, 32)
        )
        self.model = VAE(self.config)

    def test_forward(self):
        """测试前向传播"""
        x = torch.randn(2, 3, 64, 64)
        recon, mu, logvar = self.model(x)
        self.assertEqual(recon.shape, x.shape)
        self.assertEqual(mu.shape, (2, 4, 8, 8))
        self.assertEqual(logvar.shape, (2, 4, 8, 8))

    def test_forward_with_latent(self):
        """测试返回潜在向量"""
        x = torch.randn(2, 3, 64, 64)
        recon, mu, logvar, z = self.model(x, return_latent=True)
        self.assertEqual(z.shape, mu.shape)

    def test_encode(self):
        """测试编码"""
        x = torch.randn(2, 3, 64, 64)
        mu, logvar = self.model.encode(x)
        self.assertEqual(mu.shape, (2, 4, 8, 8))

    def test_decode(self):
        """测试解码"""
        z = torch.randn(2, 4, 8, 8)
        output = self.model.decode(z)
        self.assertEqual(output.shape, (2, 3, 64, 64))

    def test_sample(self):
        """测试采样"""
        samples = self.model.sample(num_samples=4)
        self.assertEqual(samples.shape, (4, 3, 64, 64))

    def test_reconstruct(self):
        """测试重建"""
        x = torch.randn(2, 3, 64, 64)
        recon = self.model.reconstruct(x)
        self.assertEqual(recon.shape, x.shape)

    def test_interpolate(self):
        """测试插值"""
        x1 = torch.randn(1, 3, 64, 64)
        x2 = torch.randn(1, 3, 64, 64)
        interpolated = self.model.interpolate(x1, x2, num_steps=5)
        self.assertEqual(interpolated.shape, (5, 3, 64, 64))

    def test_reparameterize(self):
        """测试重参数化"""
        mu = torch.zeros(2, 4, 8, 8)
        logvar = torch.zeros(2, 4, 8, 8)
        z = self.model.reparameterize(mu, logvar)
        self.assertEqual(z.shape, mu.shape)


class TestVAELoss(unittest.TestCase):
    """测试 VAE 损失函数"""

    def test_loss_computation(self):
        """测试损失计算"""
        recon = torch.randn(4, 3, 64, 64)
        target = torch.randn(4, 3, 64, 64)
        mu = torch.randn(4, 4, 8, 8)
        logvar = torch.randn(4, 4, 8, 8)

        total_loss, recon_loss, kl_loss = vae_loss(recon, target, mu, logvar)

        self.assertIsInstance(total_loss.item(), float)
        self.assertIsInstance(recon_loss.item(), float)
        self.assertIsInstance(kl_loss.item(), float)
        self.assertGreater(total_loss.item(), 0)

    def test_kl_weight(self):
        """测试 KL 权重"""
        recon = torch.randn(4, 3, 64, 64)
        target = torch.randn(4, 3, 64, 64)
        mu = torch.randn(4, 4, 8, 8)
        logvar = torch.randn(4, 4, 8, 8)

        loss1, _, _ = vae_loss(recon, target, mu, logvar, kl_weight=1.0)
        loss2, _, _ = vae_loss(recon, target, mu, logvar, kl_weight=0.1)

        # 不同 KL 权重应产生不同损失
        self.assertNotEqual(loss1.item(), loss2.item())


class TestCreateVAEModel(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_small(self):
        """测试创建小模型"""
        model = create_vae_model("small")
        self.assertIsInstance(model, VAE)
        self.assertEqual(model.config.image_size, 64)

    def test_create_base(self):
        """测试创建基础模型"""
        model = create_vae_model("base")
        self.assertIsInstance(model, VAE)
        self.assertEqual(model.config.image_size, 256)

    def test_invalid_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_vae_model("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
