"""
扩散模型单元测试

测试覆盖:
    - DiffusionConfig 配置验证
    - SinusoidalPositionEmbedding 位置编码
    - TimeEmbedding 时间嵌入
    - ResBlock 残差块
    - AttentionBlock 注意力块
    - UNet 去噪网络
    - NoiseScheduler 噪声调度器
    - DDPM 扩散模型
    - DDIMSampler 采样器
"""

import unittest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffusion import (
    DiffusionConfig,
    SinusoidalPositionEmbedding,
    TimeEmbedding,
    ResBlock,
    AttentionBlock,
    DownBlock,
    UpBlock,
    UNet,
    NoiseScheduler,
    DDPM,
    DDIMSampler,
    create_diffusion_model,
)


class TestDiffusionConfig(unittest.TestCase):
    """测试扩散模型配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = DiffusionConfig()
        self.assertEqual(config.image_size, 64)
        self.assertEqual(config.num_timesteps, 1000)

    def test_custom_config(self):
        """测试自定义配置"""
        config = DiffusionConfig(image_size=128, num_timesteps=500)
        self.assertEqual(config.image_size, 128)
        self.assertEqual(config.num_timesteps, 500)


class TestSinusoidalPositionEmbedding(unittest.TestCase):
    """测试正弦位置编码"""

    def test_output_shape(self):
        """测试输出形状"""
        embed = SinusoidalPositionEmbedding(dim=128)
        timesteps = torch.tensor([0, 100, 500, 999])
        output = embed(timesteps)
        self.assertEqual(output.shape, (4, 128))

    def test_different_timesteps(self):
        """测试不同时间步产生不同嵌入"""
        embed = SinusoidalPositionEmbedding(dim=128)
        t1 = embed(torch.tensor([0]))
        t2 = embed(torch.tensor([500]))
        self.assertFalse(torch.allclose(t1, t2))


class TestTimeEmbedding(unittest.TestCase):
    """测试时间嵌入"""

    def test_output_shape(self):
        """测试输出形状"""
        time_embed = TimeEmbedding(model_channels=128, time_embed_dim=512)
        timesteps = torch.tensor([0, 100, 500, 999])
        output = time_embed(timesteps)
        self.assertEqual(output.shape, (4, 512))


class TestResBlock(unittest.TestCase):
    """测试残差块"""

    def test_same_channels(self):
        """测试相同通道数"""
        block = ResBlock(64, 64, time_embed_dim=256)
        x = torch.randn(2, 64, 32, 32)
        t_emb = torch.randn(2, 256)
        output = block(x, t_emb)
        self.assertEqual(output.shape, x.shape)

    def test_different_channels(self):
        """测试不同通道数"""
        block = ResBlock(64, 128, time_embed_dim=256)
        x = torch.randn(2, 64, 32, 32)
        t_emb = torch.randn(2, 256)
        output = block(x, t_emb)
        self.assertEqual(output.shape, (2, 128, 32, 32))


class TestAttentionBlock(unittest.TestCase):
    """测试注意力块"""

    def test_output_shape(self):
        """测试输出形状"""
        block = AttentionBlock(channels=128, num_heads=4)
        x = torch.randn(2, 128, 16, 16)
        output = block(x)
        self.assertEqual(output.shape, x.shape)


class TestDownBlock(unittest.TestCase):
    """测试下采样块"""

    def test_output_shape(self):
        """测试输出形状"""
        block = DownBlock(64, 128, time_embed_dim=256, num_res_blocks=2, downsample=True)
        x = torch.randn(2, 64, 32, 32)
        t_emb = torch.randn(2, 256)
        output, skips = block(x, t_emb)
        self.assertEqual(output.shape, (2, 128, 16, 16))
        self.assertEqual(len(skips), 2)

    def test_without_downsample(self):
        """测试不下采样"""
        block = DownBlock(64, 128, time_embed_dim=256, num_res_blocks=2, downsample=False)
        x = torch.randn(2, 64, 32, 32)
        t_emb = torch.randn(2, 256)
        output, skips = block(x, t_emb)
        self.assertEqual(output.shape, (2, 128, 32, 32))


class TestUpBlock(unittest.TestCase):
    """测试上采样块"""

    def test_output_shape(self):
        """测试输出形状"""
        block = UpBlock(128, 64, skip_channels=128, time_embed_dim=256, num_res_blocks=2, upsample=True)
        x = torch.randn(2, 128, 16, 16)
        t_emb = torch.randn(2, 256)
        skips = [torch.randn(2, 128, 16, 16), torch.randn(2, 128, 16, 16)]
        output = block(x, t_emb, skips)
        self.assertEqual(output.shape, (2, 64, 32, 32))


class TestUNet(unittest.TestCase):
    """测试 UNet"""

    def setUp(self):
        self.config = DiffusionConfig(
            image_size=32,
            model_channels=64,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(8,),
            num_heads=4
        )
        self.unet = UNet(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        output = self.unet(x, t)
        self.assertEqual(output.shape, x.shape)

    def test_with_class_labels(self):
        """测试带类别标签"""
        config = DiffusionConfig(
            image_size=32,
            model_channels=64,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(8,),
            num_classes=10
        )
        unet = UNet(config)
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        labels = torch.randint(0, 10, (2,))
        output = unet(x, t, labels)
        self.assertEqual(output.shape, x.shape)


class TestNoiseScheduler(unittest.TestCase):
    """测试噪声调度器"""

    def setUp(self):
        self.config = DiffusionConfig(num_timesteps=1000, beta_schedule="linear")
        self.scheduler = NoiseScheduler(self.config)

    def test_beta_schedule(self):
        """测试 beta 调度"""
        self.assertEqual(len(self.scheduler.betas), 1000)
        self.assertTrue(self.scheduler.betas[0] < self.scheduler.betas[-1])

    def test_q_sample(self):
        """测试前向扩散"""
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([0, 999])
        noisy = self.scheduler.q_sample(x, t)
        self.assertEqual(noisy.shape, x.shape)

    def test_cosine_schedule(self):
        """测试余弦调度"""
        config = DiffusionConfig(num_timesteps=1000, beta_schedule="cosine")
        scheduler = NoiseScheduler(config)
        self.assertEqual(len(scheduler.betas), 1000)


class TestDDPM(unittest.TestCase):
    """测试 DDPM 模型"""

    def setUp(self):
        self.config = DiffusionConfig(
            image_size=32,
            model_channels=32,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            num_timesteps=100
        )
        self.model = DDPM(self.config)

    def test_forward(self):
        """测试前向传播"""
        x = torch.randn(2, 3, 32, 32)
        loss = self.model(x)
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)

    def test_training_step(self):
        """测试训练步骤"""
        x = torch.randn(2, 3, 32, 32)
        loss = self.model.training_step(x)
        self.assertIsInstance(loss.item(), float)


class TestDDIMSampler(unittest.TestCase):
    """测试 DDIM 采样器"""

    def setUp(self):
        config = DiffusionConfig(
            image_size=32,
            model_channels=32,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            num_timesteps=100
        )
        self.model = DDPM(config)
        self.sampler = DDIMSampler(self.model, num_inference_steps=10)

    def test_timesteps(self):
        """测试时间步计算"""
        self.assertEqual(len(self.sampler.timesteps), 10)


class TestCreateDiffusionModel(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_small(self):
        """测试创建小模型"""
        model = create_diffusion_model("small")
        self.assertIsInstance(model, DDPM)
        self.assertEqual(model.config.image_size, 32)

    def test_create_base(self):
        """测试创建基础模型"""
        model = create_diffusion_model("base")
        self.assertIsInstance(model, DDPM)
        self.assertEqual(model.config.image_size, 64)

    def test_invalid_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_diffusion_model("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
