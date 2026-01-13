"""
VITS 端到端语音合成模型单元测试

测试覆盖:
    - VITSConfig 配置验证
    - TextEncoder 文本编码器
    - PosteriorEncoder 后验编码器
    - ResidualCouplingBlock 流模型
    - Generator HiFi-GAN解码器
    - VITS 完整模型
"""

import unittest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vits import (
    VITSConfig,
    LayerNorm,
    WaveNetResBlock,
    TextEncoder,
    PosteriorEncoder,
    ResidualCouplingLayer,
    ResidualCouplingBlock,
    StochasticDurationPredictor,
    ResBlock,
    Generator,
    VITS,
    vits_loss,
    kl_divergence,
    create_vits_model,
)


class TestVITSConfig(unittest.TestCase):
    """测试 VITS 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = VITSConfig()
        self.assertEqual(config.hidden_channels, 192)
        self.assertEqual(config.n_flows, 4)

    def test_custom_config(self):
        """测试自定义配置"""
        config = VITSConfig(hidden_channels=128, n_flows=2)
        self.assertEqual(config.hidden_channels, 128)
        self.assertEqual(config.n_flows, 2)


class TestTextEncoder(unittest.TestCase):
    """测试文本编码器"""

    def test_output_shape(self):
        """测试输出形状"""
        config = VITSConfig(hidden_channels=128, n_layers=2)
        encoder = TextEncoder(config)
        text = torch.randint(0, 100, (2, 20))
        lengths = torch.tensor([20, 15])
        x, m_p, logs_p, mask = encoder(text, lengths)
        self.assertEqual(x.shape, (2, 128, 20))
        self.assertEqual(m_p.shape, (2, config.inter_channels, 20))


class TestPosteriorEncoder(unittest.TestCase):
    """测试后验编码器"""

    def test_output_shape(self):
        """测试输出形状"""
        config = VITSConfig(hidden_channels=128, n_layers=2, inter_channels=128)
        encoder = PosteriorEncoder(config)
        spec = torch.randn(2, 80, 50)
        lengths = torch.tensor([50, 40])
        z, m_q, logs_q, mask = encoder(spec, lengths)
        self.assertEqual(z.shape, (2, 128, 50))


class TestResidualCouplingBlock(unittest.TestCase):
    """测试残差耦合块"""

    def test_forward_reverse(self):
        """测试正向和逆向变换"""
        config = VITSConfig(inter_channels=128, n_flows=2, flow_hidden_channels=128)
        flow = ResidualCouplingBlock(config)
        x = torch.randn(2, 128, 50)
        mask = torch.ones(2, 1, 50)
        
        z = flow(x, mask, reverse=False)
        x_recon = flow(z, mask, reverse=True)
        self.assertEqual(z.shape, x.shape)
        self.assertEqual(x_recon.shape, x.shape)


class TestGenerator(unittest.TestCase):
    """测试 HiFi-GAN 生成器"""

    def test_output_shape(self):
        """测试输出形状"""
        config = VITSConfig(inter_channels=128, upsample_initial_channel=256)
        generator = Generator(config)
        z = torch.randn(1, 128, 10)
        audio = generator(z)
        self.assertEqual(audio.dim(), 3)
        self.assertEqual(audio.size(1), 1)


class TestVITS(unittest.TestCase):
    """测试 VITS 完整模型"""

    def test_forward(self):
        """测试前向传播"""
        config = VITSConfig(
            hidden_channels=128, n_layers=2, n_flows=2,
            inter_channels=128, flow_hidden_channels=128,
            upsample_initial_channel=256
        )
        model = VITS(config)
        
        text = torch.randint(0, 100, (2, 15))
        text_lengths = torch.tensor([15, 12])
        spec = torch.randn(2, 80, 50)
        spec_lengths = torch.tensor([50, 40])
        
        outputs = model(text, text_lengths, spec, spec_lengths)
        self.assertIn("audio", outputs)
        self.assertIn("z", outputs)

    def test_create_model(self):
        """测试工厂函数"""
        for size in ["tiny", "base", "large"]:
            model = create_vits_model(size)
            self.assertIsInstance(model, VITS)


class TestVITSLoss(unittest.TestCase):
    """测试损失函数"""

    def test_loss_computation(self):
        """测试损失计算"""
        audio_real = torch.randn(2, 1, 8192)
        audio_fake = torch.randn(2, 1, 8192)
        z_p = torch.randn(2, 128, 50)
        m_p = torch.randn(2, 128, 50)
        logs_p = torch.randn(2, 128, 50)
        m_q = torch.randn(2, 128, 50)
        logs_q = torch.randn(2, 128, 50)
        y_mask = torch.ones(2, 1, 50)
        duration_loss = torch.tensor(0.5)
        
        total_loss, loss_dict = vits_loss(
            audio_real, audio_fake, z_p, m_p, logs_p, m_q, logs_q, y_mask, duration_loss
        )
        self.assertGreater(total_loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
