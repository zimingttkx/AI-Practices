"""
音频特征提取模块单元测试

测试覆盖:
    - AudioConfig 配置验证
    - STFT 短时傅里叶变换
    - MelSpectrogram Mel 频谱图
    - LogMelSpectrogram 对数 Mel 频谱图
    - MFCC 梅尔频率倒谱系数
    - SpecAugment 数据增强
    - AudioFeatureExtractor 统一接口
"""

import unittest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_features import (
    AudioConfig,
    hz_to_mel,
    mel_to_hz,
    create_mel_filterbank,
    create_dct_matrix,
    STFT,
    MelSpectrogram,
    LogMelSpectrogram,
    MFCC,
    SpecAugment,
    AudioFeatureExtractor,
    create_audio_extractor,
)


class TestAudioConfig(unittest.TestCase):
    """测试音频配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = AudioConfig()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.n_mels, 80)
        self.assertEqual(config.n_fft, 400)

    def test_custom_config(self):
        """测试自定义配置"""
        config = AudioConfig(sample_rate=22050, n_mels=128)
        self.assertEqual(config.sample_rate, 22050)
        self.assertEqual(config.n_mels, 128)


class TestMelConversion(unittest.TestCase):
    """测试 Mel 尺度转换"""

    def test_hz_to_mel(self):
        """测试 Hz 到 Mel 转换"""
        freq = torch.tensor([0.0, 1000.0, 8000.0])
        mel = hz_to_mel(freq)
        self.assertEqual(mel.shape, freq.shape)
        self.assertAlmostEqual(mel[0].item(), 0.0, places=2)

    def test_mel_to_hz(self):
        """测试 Mel 到 Hz 转换"""
        mel = torch.tensor([0.0, 1000.0, 2000.0])
        freq = mel_to_hz(mel)
        self.assertEqual(freq.shape, mel.shape)
        self.assertAlmostEqual(freq[0].item(), 0.0, places=2)

    def test_roundtrip(self):
        """测试往返转换"""
        freq = torch.tensor([100.0, 500.0, 1000.0, 4000.0])
        mel = hz_to_mel(freq)
        freq_back = mel_to_hz(mel)
        self.assertTrue(torch.allclose(freq, freq_back, atol=1e-4))


class TestMelFilterbank(unittest.TestCase):
    """测试 Mel 滤波器组"""

    def test_filterbank_shape(self):
        """测试滤波器组形状"""
        filterbank = create_mel_filterbank(
            n_fft=400, n_mels=80, sample_rate=16000
        )
        self.assertEqual(filterbank.shape, (80, 201))

    def test_filterbank_values(self):
        """测试滤波器组值范围"""
        filterbank = create_mel_filterbank(
            n_fft=400, n_mels=80, sample_rate=16000
        )
        self.assertTrue((filterbank >= 0).all())
        self.assertTrue((filterbank <= 1).all())


class TestDCTMatrix(unittest.TestCase):
    """测试 DCT 矩阵"""

    def test_dct_shape(self):
        """测试 DCT 矩阵形状"""
        dct = create_dct_matrix(n_mfcc=13, n_mels=80)
        self.assertEqual(dct.shape, (13, 80))


class TestSTFT(unittest.TestCase):
    """测试短时傅里叶变换"""

    def test_output_shape(self):
        """测试输出形状"""
        stft = STFT(n_fft=400, hop_length=160)
        waveform = torch.randn(2, 16000)
        output = stft(waveform)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 201)  # n_fft // 2 + 1

    def test_single_waveform(self):
        """测试单个波形"""
        stft = STFT(n_fft=400, hop_length=160)
        waveform = torch.randn(16000)
        output = stft(waveform)
        self.assertEqual(output.shape[0], 1)


class TestMelSpectrogram(unittest.TestCase):
    """测试 Mel 频谱图"""

    def setUp(self):
        self.config = AudioConfig(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        self.mel_spec = MelSpectrogram(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        waveform = torch.randn(2, 16000)
        output = self.mel_spec(waveform)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 80)

    def test_normalization(self):
        """测试归一化"""
        waveform = torch.randn(2, 16000)
        output_norm = self.mel_spec(waveform, normalize=True)
        output_no_norm = self.mel_spec(waveform, normalize=False)
        # 归一化后输出应该是有限值
        self.assertTrue(torch.isfinite(output_norm).all())
        # 归一化和非归一化输出应该不同
        self.assertFalse(torch.allclose(output_norm, output_no_norm))


class TestLogMelSpectrogram(unittest.TestCase):
    """测试对数 Mel 频谱图"""

    def setUp(self):
        self.config = AudioConfig()
        self.log_mel = LogMelSpectrogram(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        waveform = torch.randn(2, 16000)
        output = self.log_mel(waveform)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], self.config.n_mels)


class TestMFCC(unittest.TestCase):
    """测试 MFCC"""

    def setUp(self):
        self.config = AudioConfig(n_mfcc=13)
        self.mfcc = MFCC(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        waveform = torch.randn(2, 16000)
        output = self.mfcc(waveform)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 13)

    def test_with_deltas(self):
        """测试带差分"""
        waveform = torch.randn(2, 16000)
        output = self.mfcc(waveform, include_deltas=True)
        self.assertEqual(output.shape[1], 39)  # 13 * 3


class TestSpecAugment(unittest.TestCase):
    """测试 SpecAugment"""

    def setUp(self):
        self.config = AudioConfig(
            freq_mask_param=10,
            time_mask_param=20,
            n_freq_masks=2,
            n_time_masks=2
        )
        self.spec_augment = SpecAugment(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        spec = torch.randn(2, 80, 100)
        output = self.spec_augment(spec, training=True)
        self.assertEqual(output.shape, spec.shape)

    def test_no_augment_eval(self):
        """测试评估模式不增强"""
        spec = torch.randn(2, 80, 100)
        output = self.spec_augment(spec, training=False)
        self.assertTrue(torch.equal(output, spec))


class TestAudioFeatureExtractor(unittest.TestCase):
    """测试音频特征提取器"""

    def setUp(self):
        self.config = AudioConfig()
        self.extractor = AudioFeatureExtractor(self.config)

    def test_extract_mel(self):
        """测试提取 Mel 频谱"""
        waveform = torch.randn(2, 16000)
        output = self.extractor.extract_mel(waveform)
        self.assertEqual(output.shape[1], self.config.n_mels)

    def test_extract_log_mel(self):
        """测试提取对数 Mel 频谱"""
        waveform = torch.randn(2, 16000)
        output = self.extractor.extract_log_mel(waveform)
        self.assertEqual(output.shape[1], self.config.n_mels)

    def test_extract_mfcc(self):
        """测试提取 MFCC"""
        waveform = torch.randn(2, 16000)
        output = self.extractor.extract_mfcc(waveform)
        self.assertEqual(output.shape[1], self.config.n_mfcc)

    def test_forward(self):
        """测试前向传播"""
        waveform = torch.randn(2, 16000)
        output = self.extractor(waveform, feature_type="log_mel")
        self.assertEqual(output.shape[1], self.config.n_mels)


class TestCreateAudioExtractor(unittest.TestCase):
    """测试创建音频提取器"""

    def test_create_whisper(self):
        """测试创建 Whisper 配置"""
        extractor = create_audio_extractor("whisper")
        self.assertEqual(extractor.config.sample_rate, 16000)
        self.assertEqual(extractor.config.n_mels, 80)

    def test_create_asr(self):
        """测试创建 ASR 配置"""
        extractor = create_audio_extractor("asr")
        self.assertEqual(extractor.config.sample_rate, 16000)

    def test_create_tts(self):
        """测试创建 TTS 配置"""
        extractor = create_audio_extractor("tts")
        self.assertEqual(extractor.config.sample_rate, 22050)

    def test_invalid_preset(self):
        """测试无效预设"""
        with self.assertRaises(ValueError):
            create_audio_extractor("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
