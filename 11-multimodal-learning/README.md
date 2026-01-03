# 11-multimodal-learning 多模态学习

本模块涵盖多模态深度学习的核心技术，包括视觉-语言模型、图像生成和音频处理。

## 目录结构

```
11-multimodal-learning/
├── README.md                      # 本文件
├── 01-vision-language/            # 视觉-语言模型
│   ├── src/                       # 源代码
│   │   ├── clip.py                # CLIP 对比学习
│   │   ├── blip.py                # BLIP 多任务学习
│   │   └── llava.py               # LLaVA 多模态对话
│   ├── tests/                     # 单元测试
│   ├── notebooks/                 # 教程 (5个)
│   ├── 知识点.md                   # 知识点文档 (1845行)
│   └── README.md
│
├── 02-image-generation/           # 图像生成模型
│   ├── src/
│   │   ├── vae.py                 # VAE 变分自编码器
│   │   ├── diffusion.py           # DDPM/DDIM 扩散模型
│   │   ├── stable_diffusion.py    # Stable Diffusion
│   │   └── controlnet.py          # ControlNet 条件控制
│   ├── tests/
│   ├── notebooks/                 # 教程 (4个)
│   ├── 知识点.md                   # 知识点文档 (1628行)
│   └── README.md
│
└── 03-audio-models/               # 音频模型
    ├── src/
    │   ├── audio_features.py      # 音频特征提取
    │   ├── whisper.py             # Whisper 语音识别
    │   └── tts.py                 # 文本转语音
    ├── tests/
    ├── notebooks/                 # 教程 (6个)
    ├── 知识点.md                   # 知识点文档 (1170行)
    └── README.md
```

## 统计

| 模块 | 测试 | 知识点 | Notebooks |
|------|------|--------|-----------|
| 01-vision-language | 76 | 1845 行 | 5 个 |
| 02-image-generation | 89 | 1628 行 | 4 个 |
| 03-audio-models | 52 | 1170 行 | 6 个 |
| **总计** | **217** | **4643 行** | **15 个** |

## 运行测试

```bash
# 运行所有测试
pytest 11-multimodal-learning/ -v

# 运行单个模块测试
pytest 11-multimodal-learning/01-vision-language/ -v
```

## 核心技术

### 视觉-语言模型
- **CLIP**: 对比学习，图文对齐
- **BLIP**: 多任务学习，图像描述/VQA
- **LLaVA**: 大语言模型 + 视觉编码器

### 图像生成
- **VAE**: 变分自编码器，潜空间学习
- **DDPM/DDIM**: 扩散模型，去噪生成
- **Stable Diffusion**: 潜空间扩散
- **ControlNet**: 条件控制生成

### 音频模型
- **音频特征**: Mel频谱、MFCC、SpecAugment
- **Whisper**: 端到端语音识别
- **TTS**: 文本转语音合成
