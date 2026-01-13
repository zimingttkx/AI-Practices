"""
Wav2Vec2 自监督语音模型单元测试

测试覆盖:
    - Wav2Vec2Config 配置验证
    - FeatureEncoder 特征编码器
    - TransformerEncoder Transformer编码器
    - GumbelVectorQuantizer 向量量化器
    - Wav2Vec2Model 完整模型
    - Wav2Vec2ForCTC CTC模型
    - Wav2Vec2ForSequenceClassification 分类模型
"""

import unittest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wav2vec2 import (
    Wav2Vec2Config,
    ConvLayerBlock,
    FeatureEncoder,
    FeatureProjection,
    PositionalConvEmbedding,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerEncoderLayer,
    TransformerEncoder,
    GumbelVectorQuantizer,
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    create_wav2vec2_model,
    create_wav2vec2_for_ctc,
)


class TestWav2Vec2Config(unittest.TestCase):
    """测试 Wav2Vec2 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = Wav2Vec2Config()
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.num_hidden_layers, 12)

    def test_custom_config(self):
        """测试自定义配置"""
        config = Wav2Vec2Config(hidden_size=256, num_attention_heads=4)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.num_attention_heads, 4)


class TestFeatureEncoder(unittest.TestCase):
    """测试特征编码器"""

    def test_output_shape(self):
        """测试输出形状"""
        config = Wav2Vec2Config()
        encoder = FeatureEncoder(config)
        audio = torch.randn(2, 16000)  # 1秒音频
        output = encoder(audio)
        self.assertEqual(output.dim(), 3)
        self.assertEqual(output.size(0), 2)
        self.assertEqual(output.size(2), config.conv_dim[-1])

    def test_downsampling(self):
        """测试下采样率"""
        config = Wav2Vec2Config()
        encoder = FeatureEncoder(config)
        audio = torch.randn(1, 32000)
        output = encoder(audio)
        # 下采样率约为320
        self.assertLess(output.size(1), 32000 // 100)


class TestTransformerEncoder(unittest.TestCase):
    """测试 Transformer 编码器"""

    def test_output_shape(self):
        """测试输出形状"""
        config = Wav2Vec2Config(hidden_size=256, num_hidden_layers=4, num_attention_heads=4)
        encoder = TransformerEncoder(config)
        x = torch.randn(2, 50, 256)
        output = encoder(x)
        self.assertEqual(output.shape, x.shape)


class TestGumbelVectorQuantizer(unittest.TestCase):
    """测试 Gumbel 向量量化器"""

    def test_output_shape(self):
        """测试输出形状"""
        config = Wav2Vec2Config()
        quantizer = GumbelVectorQuantizer(config)
        x = torch.randn(2, 50, config.conv_dim[-1])
        quantized, perplexity = quantizer(x)
        self.assertEqual(quantized.size(0), 2)
        self.assertEqual(quantized.size(1), 50)

    def test_perplexity_range(self):
        """测试困惑度范围"""
        config = Wav2Vec2Config()
        quantizer = GumbelVectorQuantizer(config)
        x = torch.randn(2, 50, config.conv_dim[-1])
        _, perplexity = quantizer(x)
        self.assertGreater(perplexity.item(), 0)


class TestWav2Vec2Model(unittest.TestCase):
    """测试 Wav2Vec2 完整模型"""

    def test_forward(self):
        """测试前向传播"""
        config = Wav2Vec2Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4)
        model = Wav2Vec2Model(config)
        audio = torch.randn(2, 16000)
        output = model(audio)
        self.assertIn("last_hidden_state", output)
        self.assertEqual(output["last_hidden_state"].dim(), 3)
        self.assertEqual(output["last_hidden_state"].size(0), 2)
        self.assertEqual(output["last_hidden_state"].size(2), config.hidden_size)

    def test_create_model(self):
        """测试工厂函数"""
        for size in ["tiny", "base", "large"]:
            model = create_wav2vec2_model(size)
            self.assertIsInstance(model, Wav2Vec2Model)


class TestWav2Vec2ForCTC(unittest.TestCase):
    """测试 CTC 模型"""

    def test_forward(self):
        """测试前向传播"""
        vocab_size = 32
        model = create_wav2vec2_for_ctc("tiny", vocab_size)
        audio = torch.randn(2, 16000)
        outputs = model(audio)
        self.assertIn("logits", outputs)
        self.assertEqual(outputs["logits"].dim(), 3)
        self.assertEqual(outputs["logits"].size(0), 2)
        self.assertEqual(outputs["logits"].size(2), vocab_size)


class TestWav2Vec2ForSequenceClassification(unittest.TestCase):
    """测试序列分类模型"""

    def test_forward(self):
        """测试前向传播"""
        config = Wav2Vec2Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4)
        num_classes = 4
        model = Wav2Vec2ForSequenceClassification(config, num_classes)
        audio = torch.randn(2, 16000)
        outputs = model(audio)
        self.assertIn("logits", outputs)
        self.assertEqual(outputs["logits"].shape, (2, num_classes))


if __name__ == "__main__":
    unittest.main()
