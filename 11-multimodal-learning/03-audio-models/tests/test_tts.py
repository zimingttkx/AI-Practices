"""
TTS 文本转语音模型单元测试

测试覆盖:
    - TTSConfig 配置验证
    - PositionalEncoding 位置编码
    - ConvBlock 卷积块
    - TextEncoder 文本编码器
    - Prenet/Postnet 预网络/后网络
    - MelDecoder Mel 解码器
    - HiFiGANGenerator 声码器
    - TextToSpeech 完整模型
"""

import unittest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tts import (
    TTSConfig,
    PositionalEncoding,
    ConvBlock,
    TextEncoder,
    Prenet,
    Postnet,
    MelDecoder,
    ResBlock,
    HiFiGANGenerator,
    TextToSpeech,
    tts_loss,
    create_tts_model,
)


class TestTTSConfig(unittest.TestCase):
    """测试 TTS 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = TTSConfig()
        self.assertEqual(config.vocab_size, 256)
        self.assertEqual(config.n_mels, 80)
        self.assertEqual(config.encoder_dim, 256)

    def test_custom_config(self):
        """测试自定义配置"""
        config = TTSConfig(vocab_size=512, encoder_dim=128)
        self.assertEqual(config.vocab_size, 512)
        self.assertEqual(config.encoder_dim, 128)


class TestPositionalEncoding(unittest.TestCase):
    """测试位置编码"""

    def test_output_shape(self):
        """测试输出形状"""
        pe = PositionalEncoding(d_model=256, max_len=1000)
        x = torch.randn(2, 50, 256)
        output = pe(x)
        self.assertEqual(output.shape, x.shape)


class TestConvBlock(unittest.TestCase):
    """测试卷积块"""

    def test_output_shape(self):
        """测试输出形状"""
        conv = ConvBlock(in_channels=256, out_channels=256, kernel_size=5)
        x = torch.randn(2, 256, 50)
        output = conv(x)
        self.assertEqual(output.shape, x.shape)

    def test_different_channels(self):
        """测试不同通道数"""
        conv = ConvBlock(in_channels=128, out_channels=256, kernel_size=5)
        x = torch.randn(2, 128, 50)
        output = conv(x)
        self.assertEqual(output.shape, (2, 256, 50))


class TestTextEncoder(unittest.TestCase):
    """测试文本编码器"""

    def setUp(self):
        self.config = TTSConfig(
            vocab_size=256,
            encoder_dim=128,
            encoder_conv_layers=2,
            encoder_layers=2,
            encoder_heads=4
        )
        self.encoder = TextEncoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        text = torch.randint(0, 256, (2, 30))
        output = self.encoder(text)
        self.assertEqual(output.shape, (2, 30, 128))

    def test_with_mask(self):
        """测试带掩码"""
        text = torch.randint(0, 256, (2, 30))
        mask = torch.ones(2, 30).bool()
        mask[0, 20:] = False
        output = self.encoder(text, mask)
        self.assertEqual(output.shape, (2, 30, 128))


class TestPrenet(unittest.TestCase):
    """测试预网络"""

    def test_output_shape(self):
        """测试输出形状"""
        prenet = Prenet(in_dim=80, hidden_dim=256, out_dim=256)
        x = torch.randn(2, 50, 80)
        output = prenet(x)
        self.assertEqual(output.shape, (2, 50, 256))


class TestPostnet(unittest.TestCase):
    """测试后网络"""

    def setUp(self):
        self.config = TTSConfig(
            n_mels=80,
            postnet_channels=256,
            postnet_kernel=5,
            postnet_layers=3
        )
        self.postnet = Postnet(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        mel = torch.randn(2, 80, 100)
        output = self.postnet(mel)
        self.assertEqual(output.shape, mel.shape)


class TestMelDecoder(unittest.TestCase):
    """测试 Mel 解码器"""

    def setUp(self):
        self.config = TTSConfig(
            n_mels=80,
            decoder_dim=128,
            decoder_heads=4,
            decoder_layers=2,
            prenet_dim=128,
            postnet_channels=256,
            postnet_layers=3
        )
        self.decoder = MelDecoder(self.config)

    def test_training_forward(self):
        """测试训练前向传播"""
        encoder_output = torch.randn(2, 30, 128)
        mel_target = torch.randn(2, 80, 50)
        mel_output, mel_postnet, stop_tokens = self.decoder(
            encoder_output, mel_target
        )
        self.assertEqual(mel_output.shape, mel_target.shape)
        self.assertEqual(mel_postnet.shape, mel_target.shape)
        self.assertEqual(stop_tokens.shape, (2, 50))


class TestResBlock(unittest.TestCase):
    """测试 HiFi-GAN 残差块"""

    def test_output_shape(self):
        """测试输出形状"""
        block = ResBlock(channels=256, kernel_size=3, dilations=(1, 3, 5))
        x = torch.randn(2, 256, 100)
        output = block(x)
        self.assertEqual(output.shape, x.shape)


class TestHiFiGANGenerator(unittest.TestCase):
    """测试 HiFi-GAN 声码器"""

    def setUp(self):
        self.config = TTSConfig(
            n_mels=80,
            vocoder_upsample_rates=(8, 8, 2, 2),
            vocoder_upsample_kernel_sizes=(16, 16, 4, 4),
            vocoder_resblock_kernel_sizes=(3, 7, 11),
            vocoder_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            vocoder_initial_channel=256
        )
        self.vocoder = HiFiGANGenerator(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        mel = torch.randn(2, 80, 10)
        output = self.vocoder(mel)
        # 上采样率: 8 * 8 * 2 * 2 = 256
        expected_len = 10 * 256
        self.assertEqual(output.shape, (2, 1, expected_len))

    def test_output_range(self):
        """测试输出范围 (tanh)"""
        mel = torch.randn(2, 80, 10)
        output = self.vocoder(mel)
        self.assertTrue((output >= -1).all())
        self.assertTrue((output <= 1).all())


class TestTextToSpeech(unittest.TestCase):
    """测试完整 TTS 模型"""

    def setUp(self):
        self.config = TTSConfig(
            vocab_size=256,
            n_mels=80,
            encoder_dim=64,
            encoder_layers=1,
            decoder_dim=64,
            decoder_layers=1,
            vocoder_initial_channel=128
        )
        self.model = TextToSpeech(self.config)

    def test_forward(self):
        """测试前向传播"""
        text = torch.randint(0, 256, (2, 20))
        mel_target = torch.randn(2, 80, 50)
        output = self.model(text, mel_target)
        self.assertIn("mel_output", output)
        self.assertIn("mel_postnet", output)
        self.assertIn("stop_tokens", output)
        self.assertEqual(output["mel_output"].shape, mel_target.shape)


class TestTTSLoss(unittest.TestCase):
    """测试 TTS 损失函数"""

    def test_loss_computation(self):
        """测试损失计算"""
        mel_output = torch.randn(2, 80, 50)
        mel_postnet = torch.randn(2, 80, 50)
        mel_target = torch.randn(2, 80, 50)
        stop_tokens = torch.randn(2, 50)
        stop_target = torch.zeros(2, 50)
        stop_target[:, -5:] = 1.0

        total_loss, loss_dict = tts_loss(
            mel_output, mel_postnet, mel_target,
            stop_tokens, stop_target
        )

        self.assertIsInstance(total_loss.item(), float)
        self.assertIn("mel_loss", loss_dict)
        self.assertIn("mel_postnet_loss", loss_dict)
        self.assertIn("stop_loss", loss_dict)


class TestCreateTTSModel(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_tiny(self):
        """测试创建 tiny 模型"""
        model = create_tts_model("tiny")
        self.assertEqual(model.config.encoder_dim, 128)

    def test_create_base(self):
        """测试创建 base 模型"""
        model = create_tts_model("base")
        self.assertEqual(model.config.encoder_dim, 256)

    def test_invalid_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_tts_model("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
