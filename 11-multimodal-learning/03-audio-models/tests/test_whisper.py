"""
Whisper 语音识别模型单元测试

测试覆盖:
    - WhisperConfig 配置验证
    - SinusoidalPositionalEncoding 位置编码
    - MultiHeadAttention 多头注意力
    - EncoderLayer/DecoderLayer 编解码器层
    - AudioEncoder 音频编码器
    - TextDecoder 文本解码器
    - Whisper 完整模型
"""

import unittest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper import (
    WhisperConfig,
    SinusoidalPositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    EncoderLayer,
    DecoderLayer,
    AudioEncoder,
    TextDecoder,
    Whisper,
    create_whisper_model,
)


class TestWhisperConfig(unittest.TestCase):
    """测试 Whisper 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = WhisperConfig()
        self.assertEqual(config.n_mels, 80)
        self.assertEqual(config.d_model, 512)
        self.assertEqual(config.n_heads, 8)

    def test_custom_config(self):
        """测试自定义配置"""
        config = WhisperConfig(d_model=256, n_heads=4)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.n_heads, 4)


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    """测试正弦位置编码"""

    def test_output_shape(self):
        """测试输出形状"""
        pe = SinusoidalPositionalEncoding(d_model=256)
        x = torch.randn(2, 100, 256)
        output = pe(x)
        self.assertEqual(output.shape, x.shape)

    def test_different_positions(self):
        """测试不同位置有不同编码"""
        pe = SinusoidalPositionalEncoding(d_model=256)
        x = torch.zeros(1, 10, 256)
        output = pe(x)
        # 不同位置的编码应该不同
        self.assertFalse(torch.allclose(output[0, 0], output[0, 5]))


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


class TestEncoderLayer(unittest.TestCase):
    """测试编码器层"""

    def test_output_shape(self):
        """测试输出形状"""
        config = WhisperConfig(d_model=256, n_heads=4, d_ff=1024)
        layer = EncoderLayer(config)
        x = torch.randn(2, 50, 256)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)


class TestDecoderLayer(unittest.TestCase):
    """测试解码器层"""

    def test_output_shape(self):
        """测试输出形状"""
        config = WhisperConfig(d_model=256, n_heads=4, d_ff=1024)
        layer = DecoderLayer(config)
        x = torch.randn(2, 30, 256)
        encoder_output = torch.randn(2, 50, 256)
        output = layer(x, encoder_output)
        self.assertEqual(output.shape, x.shape)


class TestAudioEncoder(unittest.TestCase):
    """测试音频编码器"""

    def setUp(self):
        self.config = WhisperConfig(
            n_mels=80,
            d_model=256,
            n_heads=4,
            n_encoder_layers=2,
            d_ff=1024
        )
        self.encoder = AudioEncoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        mel = torch.randn(2, 80, 100)  # [batch, n_mels, time]
        output = self.encoder(mel)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[2], 256)  # d_model

    def test_downsampling(self):
        """测试下采样"""
        mel = torch.randn(2, 80, 100)
        output = self.encoder(mel)
        # 卷积下采样 stride=2
        self.assertEqual(output.shape[1], 50)


class TestTextDecoder(unittest.TestCase):
    """测试文本解码器"""

    def setUp(self):
        self.config = WhisperConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=4,
            n_decoder_layers=2,
            d_ff=1024
        )
        self.decoder = TextDecoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        tokens = torch.randint(0, 1000, (2, 20))
        encoder_output = torch.randn(2, 50, 256)
        output = self.decoder(tokens, encoder_output)
        self.assertEqual(output.shape, (2, 20, 256))


class TestWhisper(unittest.TestCase):
    """测试 Whisper 完整模型"""

    def setUp(self):
        self.config = WhisperConfig(
            n_mels=80,
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=512
        )
        self.model = Whisper(self.config)

    def test_forward(self):
        """测试前向传播"""
        mel = torch.randn(2, 80, 100)
        tokens = torch.randint(0, 1000, (2, 20))
        logits = self.model(mel, tokens)
        self.assertEqual(logits.shape, (2, 20, 1000))

    def test_encode(self):
        """测试编码"""
        mel = torch.randn(2, 80, 100)
        encoder_output = self.model.encode(mel)
        self.assertEqual(encoder_output.shape[0], 2)
        self.assertEqual(encoder_output.shape[2], 128)

    def test_decode(self):
        """测试解码"""
        tokens = torch.randint(0, 1000, (2, 20))
        encoder_output = torch.randn(2, 50, 128)
        logits = self.model.decode(tokens, encoder_output)
        self.assertEqual(logits.shape, (2, 20, 1000))


class TestCreateWhisperModel(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_tiny(self):
        """测试创建 tiny 模型"""
        model = create_whisper_model("tiny")
        self.assertEqual(model.config.d_model, 384)

    def test_create_base(self):
        """测试创建 base 模型"""
        model = create_whisper_model("base")
        self.assertEqual(model.config.d_model, 512)

    def test_invalid_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_whisper_model("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
