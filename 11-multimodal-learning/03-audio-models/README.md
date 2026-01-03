# 音频模型 (Audio Models)

本模块实现音频处理和语音相关的深度学习模型，包括音频特征提取、语音识别 (ASR) 和文本转语音 (TTS)。

## 目录结构

```
03-audio-models/
├── src/
│   ├── __init__.py
│   ├── audio_features.py    # 音频特征提取 (Mel/MFCC)
│   ├── whisper.py           # Whisper 语音识别
│   └── tts.py               # TTS 文本转语音
├── tests/
│   ├── test_audio_features.py
│   ├── test_whisper.py
│   └── test_tts.py
├── notebooks/
│   ├── 01_AudioFeatures_tutorial.ipynb
│   ├── 02_Whisper_tutorial.ipynb
│   └── 03_TTS_tutorial.ipynb
├── 知识点.md
└── README.md
```

## 核心组件

### 1. 音频特征提取 (audio_features.py)

音频信号的预处理和特征提取：

- **Mel 频谱图**: 将音频转换为 Mel 尺度的频谱表示
- **MFCC**: 梅尔频率倒谱系数，常用于语音识别
- **频谱增强**: SpecAugment 数据增强技术

```python
from audio_features import AudioFeatureExtractor, MelSpectrogram, MFCC

# 创建特征提取器
extractor = AudioFeatureExtractor(
    sample_rate=16000,
    n_mels=80,
    n_fft=400,
    hop_length=160
)

# 提取 Mel 频谱图
mel_spec = extractor.mel_spectrogram(waveform)

# 提取 MFCC
mfcc = extractor.mfcc(waveform, n_mfcc=13)
```

### 2. Whisper 语音识别 (whisper.py)

OpenAI Whisper 风格的语音识别模型：

- **音频编码器**: Transformer 编码器处理 Mel 频谱
- **文本解码器**: 自回归解码器生成文本
- **多任务**: 支持转录、翻译、语言检测

```python
from whisper import WhisperConfig, Whisper, create_whisper_model

# 创建模型
model = create_whisper_model("base")

# 语音识别
transcription = model.transcribe(audio)
```

### 3. TTS 文本转语音 (tts.py)

端到端的文本转语音系统：

- **文本编码器**: 处理输入文本序列
- **声学模型**: 预测 Mel 频谱图
- **声码器**: 将 Mel 频谱转换为波形

```python
from tts import TTSConfig, TextToSpeech, create_tts_model

# 创建模型
model = create_tts_model("base")

# 文本转语音
waveform = model.synthesize("Hello, world!")
```

## 快速开始

```python
import torch
import sys
sys.path.append("src")

# 1. 音频特征提取
from audio_features import AudioFeatureExtractor

extractor = AudioFeatureExtractor(sample_rate=16000)
waveform = torch.randn(1, 16000)  # 1秒音频
mel = extractor.mel_spectrogram(waveform)
print(f"Mel 频谱形状: {mel.shape}")

# 2. 语音识别
from whisper import create_whisper_model

whisper = create_whisper_model("tiny")
# transcription = whisper.transcribe(mel)

# 3. 文本转语音
from tts import create_tts_model

tts = create_tts_model("tiny")
# audio = tts.synthesize("你好世界")
```

## 运行测试

```bash
cd 03-audio-models
python -m unittest discover -s tests -v
```

## 参考资料

### 论文
1. [Whisper] Radford et al. "Robust Speech Recognition via Large-Scale Weak Supervision" (2022)
2. [Tacotron 2] Shen et al. "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" (2018)
3. [HiFi-GAN] Kong et al. "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2020)
4. [SpecAugment] Park et al. "SpecAugment: A Simple Data Augmentation Method for ASR" (2019)

### 代码库
- [OpenAI Whisper](https://github.com/openai/whisper)
- [ESPnet](https://github.com/espnet/espnet)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
