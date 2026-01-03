# Image Generation Models

图像生成模块，实现从 VAE 到 Stable Diffusion 的完整图像生成技术栈。

## 模块结构

```
02-image-generation/
├── src/
│   ├── vae.py               # 变分自编码器
│   ├── diffusion.py         # 扩散模型基础 (DDPM/DDIM)
│   ├── stable_diffusion.py  # Stable Diffusion
│   └── controlnet.py        # ControlNet 条件控制
├── tests/
│   ├── test_vae.py          # VAE 单元测试
│   ├── test_diffusion.py    # 扩散模型测试
│   ├── test_sd.py           # Stable Diffusion 测试
│   └── test_controlnet.py   # ControlNet 测试
├── notebooks/
│   └── 01_image_generation_tutorial.ipynb  # 教程
├── 知识点.md                    # 理论知识详解
└── README.md
```

---

## VAE (Variational Autoencoder)

变分自编码器是生成模型的基础，学习数据的潜在表示。

### 核心组件

| 组件 | 描述 |
|:-----|:-----|
| `VAEConfig` | 模型配置 |
| `Encoder` | 编码器 (图像 → 潜在分布) |
| `Decoder` | 解码器 (潜在向量 → 图像) |
| `VAE` | 完整模型 |
| `vae_loss` | ELBO 损失函数 |

### 快速开始

```python
from vae import create_vae_model, vae_loss
import torch

model = create_vae_model("base")
images = torch.randn(4, 3, 256, 256)

# 前向传播
recon, mu, logvar = model(images)
loss = vae_loss(recon, images, mu, logvar)

# 采样生成
samples = model.sample(num_samples=4)
```

---

## Diffusion Models (DDPM/DDIM)

扩散模型通过逐步去噪过程生成高质量图像。

### 核心组件

| 组件 | 描述 |
|:-----|:-----|
| `DiffusionConfig` | 扩散配置 |
| `NoiseScheduler` | 噪声调度器 (linear/cosine) |
| `UNet` | 去噪网络 |
| `DDPM` | 去噪扩散概率模型 |
| `DDIMSampler` | DDIM 加速采样 |

### 快速开始

```python
from diffusion import create_diffusion_model
import torch

model = create_diffusion_model("base")
images = torch.randn(4, 3, 64, 64)

# 训练
loss = model.training_step(images)

# 采样
samples = model.sample(batch_size=4, num_steps=50)
```

---

## Stable Diffusion

Stable Diffusion 在潜在空间进行扩散，支持文本条件生成。

### 核心组件

| 组件 | 描述 |
|:-----|:-----|
| `SDConfig` | 模型配置 |
| `TextEncoder` | CLIP 文本编码器 |
| `LatentDiffusion` | 潜在扩散模型 |
| `StableDiffusion` | 完整 SD 模型 |

### 快速开始

```python
from stable_diffusion import create_sd_model
import torch

model = create_sd_model("base")

# 文本到图像
prompt_embeds = torch.randn(1, 77, 768)  # 文本嵌入
images = model.generate(prompt_embeds, num_steps=50)

# 图像到图像
init_image = torch.randn(1, 3, 512, 512)
images = model.img2img(init_image, prompt_embeds, strength=0.75)
```

---

## ControlNet

ControlNet 为 Stable Diffusion 添加精确的条件控制。

### 支持的控制类型

- **Canny Edge**: 边缘检测控制
- **Pose**: 人体姿态控制
- **Depth**: 深度图控制

### 快速开始

```python
from controlnet import create_controlnet
import torch

controlnet = create_controlnet("canny")
control_image = torch.randn(1, 3, 512, 512)  # 边缘图

# 获取控制特征
control_features = controlnet(control_image, timestep=500)
```

---

## 运行测试

```bash
# 运行所有测试
python -m pytest 11-multimodal-learning/02-image-generation/tests -v

# 单独运行
python -m pytest 11-multimodal-learning/02-image-generation/tests/test_vae.py -v
python -m pytest 11-multimodal-learning/02-image-generation/tests/test_diffusion.py -v
```

## 参考论文

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (VAE)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIM)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Stable Diffusion)
- [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) (ControlNet)
