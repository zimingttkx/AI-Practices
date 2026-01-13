"""
声音克隆模块单元测试

测试覆盖:
    - SpeakerEncoderConfig 配置验证
    - SpeakerEncoder 说话人编码器
    - GE2ELoss GE2E损失
    - SpeakerAdapter 说话人适配器
    - VoiceCloner 声音克隆器
"""

import unittest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_cloning import (
    SpeakerEncoderConfig,
    VoiceCloningConfig,
    SpeakerEncoder,
    GE2ELoss,
    SpeakerAdapter,
    create_speaker_encoder,
    create_voice_cloning_config,
)


class TestSpeakerEncoderConfig(unittest.TestCase):
    """测试说话人编码器配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = SpeakerEncoderConfig()
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.embedding_size, 256)

    def test_custom_config(self):
        """测试自定义配置"""
        config = SpeakerEncoderConfig(hidden_size=128, embedding_size=128)
        self.assertEqual(config.hidden_size, 128)


class TestSpeakerEncoder(unittest.TestCase):
    """测试说话人编码器"""

    def test_output_shape(self):
        """测试输出形状"""
        config = SpeakerEncoderConfig(hidden_size=128, embedding_size=128)
        encoder = SpeakerEncoder(config)
        mel = torch.randn(2, 80, 200)
        lengths = torch.tensor([200, 150])
        embedding = encoder(mel, lengths)
        self.assertEqual(embedding.shape, (2, 128))

    def test_l2_normalized(self):
        """测试L2归一化"""
        config = SpeakerEncoderConfig()
        encoder = SpeakerEncoder(config)
        mel = torch.randn(2, 80, 200)
        embedding = encoder(mel)
        norms = embedding.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_create_encoder(self):
        """测试工厂函数"""
        for size in ["tiny", "base", "large"]:
            encoder = create_speaker_encoder(size)
            self.assertIsInstance(encoder, SpeakerEncoder)


class TestGE2ELoss(unittest.TestCase):
    """测试 GE2E 损失"""

    def test_loss_computation(self):
        """测试损失计算"""
        ge2e = GE2ELoss()
        speakers = 4
        utterances = 5  # 需要至少2个以上才能计算排除质心
        embedding_size = 256
        
        embeddings = torch.randn(speakers * utterances, embedding_size)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        loss = ge2e(embeddings, speakers, utterances)
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)


class TestSpeakerAdapter(unittest.TestCase):
    """测试说话人适配器"""

    def test_output_shape(self):
        """测试输出形状"""
        config = create_voice_cloning_config(tts_hidden_size=256, speaker_embedding_size=256)
        adapter = SpeakerAdapter(config)
        
        hidden = torch.randn(2, 50, 256)
        speaker_emb = torch.randn(2, 256)
        
        output = adapter(hidden, speaker_emb)
        self.assertEqual(output.shape, hidden.shape)


if __name__ == "__main__":
    unittest.main()
