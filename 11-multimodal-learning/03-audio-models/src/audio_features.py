"""
音频特征提取模块 (Audio Feature Extraction)

本模块实现音频信号的预处理和特征提取，包括：
- 短时傅里叶变换 (STFT)
- Mel 频谱图
- MFCC (梅尔频率倒谱系数)
- SpecAugment 数据增强

=== 核心概念 ===

1. 短时傅里叶变换 (STFT)
   - 将时域信号转换为时频表示
   - 使用滑动窗口分帧，对每帧做 FFT
   - 参数：n_fft (FFT 点数), hop_length (帧移), win_length (窗长)

2. Mel 频谱图
   - 将线性频率转换为 Mel 尺度
   - Mel 尺度更符合人耳感知特性
   - 公式: mel = 2595 * log10(1 + f/700)

3. MFCC (Mel-Frequency Cepstral Coefficients)
   - 对 Mel 频谱取对数后做 DCT
   - 常用于语音识别的特征表示
   - 通常取前 13-40 个系数

4. SpecAugment
   - 频谱数据增强技术
   - 包括时间遮蔽、频率遮蔽、时间扭曲

=== 参考文献 ===

1. Mel Scale:
   Stevens et al. "A Scale for the Measurement of the Psychological Magnitude Pitch" 1937

2. MFCC:
   Davis & Mermelstein "Comparison of Parametric Representations for Monosyllabic Word Recognition" 1980

3. SpecAugment:
   Park et al. "SpecAugment: A Simple Data Augmentation Method for ASR" 2019
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AudioConfig:
    """音频处理配置"""

    # 采样率
    sample_rate: int = 16000

    # STFT 参数
    n_fft: int = 400          # FFT 点数 (25ms @ 16kHz)
    hop_length: int = 160     # 帧移 (10ms @ 16kHz)
    win_length: int = 400     # 窗长

    # Mel 参数
    n_mels: int = 80          # Mel 滤波器数量
    f_min: float = 0.0        # 最低频率
    f_max: Optional[float] = 8000.0  # 最高频率

    # MFCC 参数
    n_mfcc: int = 13          # MFCC 系数数量

    # SpecAugment 参数
    freq_mask_param: int = 27     # 频率遮蔽最大宽度
    time_mask_param: int = 100    # 时间遮蔽最大宽度
    n_freq_masks: int = 2         # 频率遮蔽次数
    n_time_masks: int = 2         # 时间遮蔽次数


def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    """将频率 (Hz) 转换为 Mel 尺度"""
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """将 Mel 尺度转换为频率 (Hz)"""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None
) -> torch.Tensor:
    """
    创建 Mel 滤波器组

    Args:
        n_fft: FFT 点数
        n_mels: Mel 滤波器数量
        sample_rate: 采样率
        f_min: 最低频率
        f_max: 最高频率

    Returns:
        Mel 滤波器组 [n_mels, n_fft // 2 + 1]
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    # 频率点数
    n_freqs = n_fft // 2 + 1

    # 计算 Mel 尺度的边界
    mel_min = hz_to_mel(torch.tensor(f_min))
    mel_max = hz_to_mel(torch.tensor(f_max))

    # 在 Mel 尺度上均匀分布的点
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # 转换为 FFT bin 索引
    freq_bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    # 创建滤波器组
    filterbank = torch.zeros(n_mels, n_freqs)

    for i in range(n_mels):
        left = freq_bins[i]
        center = freq_bins[i + 1]
        right = freq_bins[i + 2]

        # 上升斜坡
        for j in range(left, center):
            if j < n_freqs:
                filterbank[i, j] = (j - left) / (center - left + 1e-8)

        # 下降斜坡
        for j in range(center, right):
            if j < n_freqs:
                filterbank[i, j] = (right - j) / (right - center + 1e-8)

    return filterbank


def create_dct_matrix(n_mfcc: int, n_mels: int) -> torch.Tensor:
    """
    创建 DCT (离散余弦变换) 矩阵

    Args:
        n_mfcc: MFCC 系数数量
        n_mels: Mel 滤波器数量

    Returns:
        DCT 矩阵 [n_mfcc, n_mels]
    """
    n = torch.arange(n_mels).float()
    k = torch.arange(n_mfcc).float().unsqueeze(1)

    dct = torch.cos(math.pi * k * (2 * n + 1) / (2 * n_mels))

    # 归一化
    dct[0] *= 1.0 / math.sqrt(n_mels)
    dct[1:] *= math.sqrt(2.0 / n_mels)

    return dct


class STFT(nn.Module):
    """短时傅里叶变换"""

    def __init__(
        self,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        window: str = "hann"
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft

        # 创建窗函数
        if window == "hann":
            window_tensor = torch.hann_window(self.win_length)
        elif window == "hamming":
            window_tensor = torch.hamming_window(self.win_length)
        else:
            window_tensor = torch.ones(self.win_length)

        self.register_buffer("window", window_tensor)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        计算 STFT

        Args:
            waveform: 输入波形 [batch_size, num_samples] 或 [num_samples]

        Returns:
            STFT 结果 [batch_size, n_fft // 2 + 1, num_frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # 使用 torch.stft
        stft_out = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect"
        )

        return stft_out


class MelSpectrogram(nn.Module):
    """Mel 频谱图"""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        # STFT
        self.stft = STFT(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length
        )

        # Mel 滤波器组
        mel_filterbank = create_mel_filterbank(
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            f_min=config.f_min,
            f_max=config.f_max
        )
        self.register_buffer("mel_filterbank", mel_filterbank)

    def forward(
        self,
        waveform: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        计算 Mel 频谱图

        Args:
            waveform: 输入波形 [batch_size, num_samples]
            normalize: 是否归一化

        Returns:
            Mel 频谱图 [batch_size, n_mels, num_frames]
        """
        # STFT
        stft_out = self.stft(waveform)

        # 功率谱
        power_spec = torch.abs(stft_out) ** 2

        # 应用 Mel 滤波器
        mel_spec = torch.matmul(self.mel_filterbank, power_spec)

        # 对数压缩
        mel_spec = torch.log(mel_spec + 1e-9)

        # 归一化
        if normalize:
            mel_spec = (mel_spec - mel_spec.mean(dim=-1, keepdim=True)) / (
                mel_spec.std(dim=-1, keepdim=True) + 1e-9
            )

        return mel_spec


class LogMelSpectrogram(nn.Module):
    """对数 Mel 频谱图 (Whisper 风格)"""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.stft = STFT(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length
        )

        mel_filterbank = create_mel_filterbank(
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
            f_min=config.f_min,
            f_max=config.f_max
        )
        self.register_buffer("mel_filterbank", mel_filterbank)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        计算对数 Mel 频谱图

        Args:
            waveform: 输入波形 [batch_size, num_samples]

        Returns:
            对数 Mel 频谱图 [batch_size, n_mels, num_frames]
        """
        stft_out = self.stft(waveform)
        magnitudes = torch.abs(stft_out) ** 2
        mel_spec = torch.matmul(self.mel_filterbank, magnitudes)

        # 对数压缩 (Whisper 风格)
        log_mel = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        return log_mel


class MFCC(nn.Module):
    """梅尔频率倒谱系数 (MFCC)"""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.mel_spectrogram = MelSpectrogram(config)

        dct_matrix = create_dct_matrix(config.n_mfcc, config.n_mels)
        self.register_buffer("dct_matrix", dct_matrix)

    def forward(
        self,
        waveform: torch.Tensor,
        include_deltas: bool = False
    ) -> torch.Tensor:
        """
        计算 MFCC

        Args:
            waveform: 输入波形 [batch_size, num_samples]
            include_deltas: 是否包含一阶和二阶差分

        Returns:
            MFCC [batch_size, n_mfcc (* 3 if include_deltas), num_frames]
        """
        mel_spec = self.mel_spectrogram(waveform, normalize=False)
        mfcc = torch.matmul(self.dct_matrix, mel_spec)

        if include_deltas:
            delta1 = self._compute_deltas(mfcc)
            delta2 = self._compute_deltas(delta1)
            mfcc = torch.cat([mfcc, delta1, delta2], dim=1)

        return mfcc

    def _compute_deltas(self, features: torch.Tensor, width: int = 2) -> torch.Tensor:
        """计算差分特征"""
        padded = F.pad(features, (width, width), mode="replicate")
        deltas = torch.zeros_like(features)

        for t in range(features.shape[-1]):
            for n in range(1, width + 1):
                deltas[..., t] += n * (padded[..., t + width + n] - padded[..., t + width - n])

        norm = 2 * sum(n ** 2 for n in range(1, width + 1))
        return deltas / norm


class SpecAugment(nn.Module):
    """SpecAugment 数据增强"""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        spectrogram: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        应用 SpecAugment

        Args:
            spectrogram: 输入频谱 [batch_size, n_freq, n_time]
            training: 是否在训练模式

        Returns:
            增强后的频谱
        """
        if not training:
            return spectrogram

        spec = spectrogram.clone()

        # 频率遮蔽
        for _ in range(self.config.n_freq_masks):
            spec = self._freq_mask(spec)

        # 时间遮蔽
        for _ in range(self.config.n_time_masks):
            spec = self._time_mask(spec)

        return spec

    def _freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """频率遮蔽"""
        n_freq = spec.shape[1]
        f = torch.randint(0, self.config.freq_mask_param + 1, (1,)).item()
        f = min(f, n_freq)
        f0 = torch.randint(0, n_freq - f + 1, (1,)).item()
        spec[:, f0:f0 + f, :] = 0
        return spec

    def _time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """时间遮蔽"""
        n_time = spec.shape[2]
        t = torch.randint(0, self.config.time_mask_param + 1, (1,)).item()
        t = min(t, n_time)
        t0 = torch.randint(0, n_time - t + 1, (1,)).item()
        spec[:, :, t0:t0 + t] = 0
        return spec


class AudioFeatureExtractor(nn.Module):
    """音频特征提取器 - 统一接口"""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        self.mel_spectrogram = MelSpectrogram(config)
        self.log_mel_spectrogram = LogMelSpectrogram(config)
        self.mfcc = MFCC(config)
        self.spec_augment = SpecAugment(config)

    def extract_mel(
        self,
        waveform: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """提取 Mel 频谱图"""
        return self.mel_spectrogram(waveform, normalize=normalize)

    def extract_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """提取对数 Mel 频谱图 (Whisper 风格)"""
        return self.log_mel_spectrogram(waveform)

    def extract_mfcc(
        self,
        waveform: torch.Tensor,
        include_deltas: bool = False
    ) -> torch.Tensor:
        """提取 MFCC"""
        return self.mfcc(waveform, include_deltas=include_deltas)

    def augment(
        self,
        spectrogram: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """应用 SpecAugment"""
        return self.spec_augment(spectrogram, training=training)

    def forward(
        self,
        waveform: torch.Tensor,
        feature_type: Literal["mel", "log_mel", "mfcc"] = "log_mel",
        augment: bool = False
    ) -> torch.Tensor:
        """
        提取音频特征

        Args:
            waveform: 输入波形 [batch_size, num_samples]
            feature_type: 特征类型
            augment: 是否应用数据增强

        Returns:
            音频特征
        """
        if feature_type == "mel":
            features = self.extract_mel(waveform)
        elif feature_type == "log_mel":
            features = self.extract_log_mel(waveform)
        elif feature_type == "mfcc":
            features = self.extract_mfcc(waveform)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        if augment and self.training:
            features = self.augment(features)

        return features


def create_audio_extractor(preset: str = "whisper") -> AudioFeatureExtractor:
    """
    创建预定义的音频特征提取器

    Args:
        preset: 预设配置 ("whisper", "asr", "tts")

    Returns:
        AudioFeatureExtractor 实例
    """
    presets = {
        "whisper": AudioConfig(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0
        ),
        "asr": AudioConfig(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=80,
            n_mfcc=13,
            f_min=20.0,
            f_max=8000.0
        ),
        "tts": AudioConfig(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0.0,
            f_max=11025.0
        ),
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

    return AudioFeatureExtractor(presets[preset])
