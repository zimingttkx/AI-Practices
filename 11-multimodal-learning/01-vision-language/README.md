# 01-Vision-Language Models (视觉-语言模型)

> 视觉-语言多模态学习模块，实现图像与文本的联合理解与生成

## 目录结构

```
01-vision-language/
├── README.md                    # 本文件
├── src/
│   ├── __init__.py              # 模块导出
│   ├── clip.py                  # CLIP 对比学习 (~700行)
│   ├── blip.py                  # BLIP 多任务学习 (~750行)
│   └── llava.py                 # LLaVA 多模态对话 (~800行)
├── tests/
│   ├── __init__.py
│   ├── test_clip.py             # CLIP 单元测试 (26个)
│   ├── test_blip.py             # BLIP 单元测试 (28个)
│   └── test_llava.py            # LLaVA 单元测试 (23个)
├── notebooks/
│   └── 01_vision_language_tutorial.ipynb  # 教程
└── 知识点.md                    # 理论知识详解
```

## 核心思想

### 多模态学习的动机

传统的单模态模型只能处理单一类型的数据（如纯文本或纯图像），而人类的认知是多模态的。
视觉-语言模型通过联合学习图像和文本的表示，实现：

1. **跨模态理解**: 理解图像内容并用自然语言描述
2. **跨模态检索**: 用文本搜索图像，或用图像搜索文本
3. **多模态对话**: 基于图像进行自然语言交互

### 三种模型的定位

| 模型 | 核心任务 | 架构特点 | 适用场景 |
|:-----|:---------|:---------|:---------|
| CLIP | 对比学习 | 双塔结构，独立编码 | 零样本分类、图文检索 |
| BLIP | 多任务学习 | 编码器-解码器 | 图像描述、VQA、检索 |
| LLaVA | 多模态对话 | 视觉编码器+LLM | 视觉问答、多轮对话 |

## 快速开始

### CLIP (Contrastive Language-Image Pre-training)

CLIP 通过对比学习将图像和文本映射到共享的嵌入空间。

**数学原理 - InfoNCE 损失**:
```
L = -1/N * Σ log exp(sim(vi, ti)/τ) / Σ exp(sim(vi, tj)/τ)
```

**核心组件**:

| 组件 | 描述 |
|:-----|:-----|
| `CLIPConfig` | 模型配置 |
| `PatchEmbedding` | 图像分块嵌入 + CLS token |
| `VisionEncoder` | 基于 ViT 的图像编码器 |
| `TextEncoder` | Transformer 文本编码器 |
| `CLIP` | 完整模型 (双塔结构) |
| `clip_loss` | 对比损失函数 |

**代码示例**:

```python
from clip import create_clip_model, clip_loss
import torch

# 创建模型 (支持 small, base, large)
model = create_clip_model("base")

# 准备输入
images = torch.randn(4, 3, 224, 224)      # 批次图像
input_ids = torch.randint(0, 49408, (4, 77))  # 文本 token

# 前向传播
image_features, text_features, logit_scale = model(images, input_ids)

# 计算对比损失
loss = clip_loss(image_features, text_features, logit_scale)
print(f"CLIP Loss: {loss.item():.4f}")
```

---

### BLIP (Bootstrapping Language-Image Pre-training)

BLIP 是统一的视觉-语言预训练框架，支持多任务学习。

**支持的任务**:
- **ITC** (Image-Text Contrastive): 图像-文本对比学习
- **ITM** (Image-Text Matching): 图像-文本匹配
- **LM** (Language Modeling): 图像描述生成

**核心组件**:

| 组件 | 描述 |
|:-----|:-----|
| `BLIPConfig` | 模型配置 |
| `VisionEncoder` | ViT 图像编码器 |
| `TextEncoder` | BERT 风格文本编码器 |
| `TextDecoder` | 自回归文本解码器 |
| `BLIP` | 多任务模型 |

**代码示例**:

```python
from blip import create_blip_model, itc_loss
import torch

# 创建模型
model = create_blip_model("base")
images = torch.randn(4, 3, 224, 224)
input_ids = torch.randint(0, 30522, (4, 20))

# ITC: 图像-文本对比
image_feat, text_feat, logit_scale = model.forward_itc(images, input_ids)

# ITM: 图像-文本匹配
itm_logits = model.forward_itm(images, input_ids)

# LM: 图像描述生成
generated = model.generate(images, max_length=30)
```

---

### LLaVA (Large Language and Vision Assistant)

LLaVA 是多模态对话模型，通过视觉投影层将视觉编码器与 LLM 连接。

**架构特点**:
- 使用 CLIP 预训练的视觉编码器
- MLP 投影层连接视觉和语言空间
- LLaMA 风格的语言模型 (RoPE, RMSNorm, SwiGLU)

**核心组件**:

| 组件 | 描述 |
|:-----|:-----|
| `LLaVAConfig` | 模型配置 |
| `VisionEncoder` | CLIP 风格视觉编码器 |
| `VisionProjector` | 视觉特征投影层 |
| `LLaMAModel` | LLaMA 风格语言模型 |
| `LLaVA` | 完整多模态对话模型 |

**代码示例**:

```python
from llava import create_llava_model
import torch

# 创建模型 (支持 tiny, small, base)
model = create_llava_model("tiny")
images = torch.randn(2, 3, 224, 224)
input_ids = torch.randint(0, 32000, (2, 20))

# 前向传播
output = model(input_ids, images)
logits = output["logits"]

# 自回归生成
generated = model.generate(input_ids, images, max_new_tokens=50)
```

---

## 运行测试

```bash
# 运行所有测试
python -m unittest discover -s 11-multimodal-learning/01-vision-language/tests -v

# 单独运行各模型测试
python -m unittest 11-multimodal-learning/01-vision-language/tests/test_clip.py -v
python -m unittest 11-multimodal-learning/01-vision-language/tests/test_blip.py -v
python -m unittest 11-multimodal-learning/01-vision-language/tests/test_llava.py -v
```

**测试统计**: 77个测试，100%通过

## 参考资料

1. [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
2. [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086) - Li et al., 2022
3. [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) - Liu et al., 2023
4. [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., 2021
5. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Touvron et al., 2023
