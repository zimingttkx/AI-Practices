<div align="center">

# AI-Practices

### 系统化人工智能学习与研究平台

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

[![Stars](https://img.shields.io/github/stars/zimingttkx/AI-Practices?style=for-the-badge&logo=github&color=yellow)](https://github.com/zimingttkx/AI-Practices/stargazers)
[![Forks](https://img.shields.io/github/forks/zimingttkx/AI-Practices?style=for-the-badge&logo=github&color=blue)](https://github.com/zimingttkx/AI-Practices/network/members)
[![Issues](https://img.shields.io/github/issues/zimingttkx/AI-Practices?style=for-the-badge&logo=github&color=red)](https://github.com/zimingttkx/AI-Practices/issues)

**[English](./README_EN.md)** | **[在线文档](https://zimingttkx.github.io/AI-Practices/)** | **[快速开始](#-快速开始)**

---

**从理论到实战，构建完整的AI知识体系**

*Machine Learning • Deep Learning • Computer Vision • NLP • LLM • Multimodal • Reinforcement Learning*

</div>

---

## 项目亮点

<div align="center">

| **350+ Notebooks** | **13大核心模块** | **1100+ 单元测试** | **生产级代码** | **2枚Kaggle金牌** |
|:------------------:|:----------------:|:-----------------:|:-------------:|:----------------:|
| 可复现实验 | 系统化学习路径 | 质量保证 | 工程实践 | 竞赛验证 |

</div>

### 为什么选择 AI-Practices？

- **渐进式学习** — 从数学推导到框架工程，循序渐进
- **理论+实践** — 不仅知道"怎么做"，更理解"为什么"
- **工程导向** — 从学术研究到工业部署的完整链路
- **竞赛验证** — Kaggle Top 1% 金牌方案，实战检验

---

## 学习路径

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Phase I        │    │  Phase II       │    │  Phase III      │    │  Phase IV       │
│  理论先行       │ -> │  从零实现       │ -> │  框架工程       │ -> │  实战项目       │
│  数学推导与分析 │    │  NumPy手写实现  │    │  PyTorch/TF     │    │  Kaggle竞赛     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 模块总览

| 阶段 | 模块 | 内容 | 文件数 |
|:----:|:-----|:-----|:------:|
| I | **01-机器学习基础** | 线性模型、SVM、决策树、集成学习、降维、聚类 | 75+ |
| II | **02-神经网络** | 反向传播、优化器、正则化、BatchNorm | 42+ |
| II | **03-计算机视觉** | CNN架构演进、迁移学习、目标检测 | 50+ |
| II | **04-序列模型** | RNN/LSTM、Attention、Transformer | 40+ |
| III | **05-高级专题** | 分布式训练、模型压缩、部署优化 | 30+ |
| III | **06-生成模型** | VAE、GAN、Diffusion Models | 35+ |
| III | **07-强化学习** | DQN、PPO、SAC、Actor-Critic | 80+ |
| IV | **09-实战项目** | Kaggle竞赛、游戏AI、股票交易 | 100+ |
| - | **08-理论笔记** | 激活函数、损失函数、架构选型速查 | 16+ |
| V | **10-大语言模型** | Transformer、GPT/LLaMA、LoRA微调、RAG、Agent | 60+ |
| V | **11-多模态学习** | CLIP、Stable Diffusion、Whisper、TTS | 47+ |
| V | **12-部署优化** | 量化剪枝、TensorRT、FastAPI、MLOps | 50+ |
| VI | **13-分布式训练** | DDP、FSDP、ZeRO、张量/流水线并行、混合精度 | 42+ |

<details>
<summary><b>📂 展开完整目录结构</b></summary>

```
AI-Practices/
├── 01-foundations/                 # 机器学习基础
│   ├── 01-training-models/         # 梯度下降、正则化
│   ├── 02-classification/          # 逻辑回归、SVM
│   ├── 03-support-vector-machines/ # 核技巧、软间隔
│   ├── 04-decision-trees/          # CART、剪枝
│   ├── 05-ensemble-learning/       # Bagging、Boosting、Stacking
│   ├── 06-dimensionality-reduction/# PCA、t-SNE、UMAP
│   ├── 07-unsupervised-learning/   # K-Means、DBSCAN、GMM
│   └── 08-end-to-end-project/      # 完整ML流程
│
├── 02-neural-networks/             # 神经网络
│   ├── 01-keras-introduction/      # Sequential、Functional API
│   ├── 02-training-deep-networks/  # BatchNorm、Dropout
│   ├── 03-custom-models-training/  # 自定义层和训练循环
│   └── 04-data-loading-preprocessing/
│
├── 07-reinforcement-learning/      # 强化学习
│   ├── 01-mdp-basics/              # 马尔可夫决策过程
│   ├── 02-temporal-difference/     # TD学习
│   ├── 03-q-learning/              # Q-Learning
│   ├── 04-deep-q-learning/         # DQN、Double DQN
│   ├── 05-policy-gradient/         # REINFORCE、PPO
│   ├── 06-actor-critic/            # A2C、A3C
│   └── 07-advanced-algorithms/     # SAC、TD3
│
├── 09-practical-projects/          # 实战项目
│   ├── 01-ml-basics/               # Titanic、Otto
│   ├── 02-computer-vision/         # MNIST CNN
│   ├── 03-nlp/                     # 情感分析、NER
│   ├── 04-time-series/             # 温度预测
│   ├── 05-kaggle-competitions/     # 金牌方案
│   └── 06-reinforcement-learning/  # 游戏AI、股票交易
│
├── 10-large-language-models/       # 大语言模型
│   ├── 01-llm-fundamentals/        # Transformer架构
│   ├── 02-pretrained-models/       # GPT、LLaMA实现
│   ├── 03-fine-tuning/             # LoRA、QLoRA微调
│   ├── 04-prompt-engineering/      # 提示工程
│   ├── 05-rag/                     # 检索增强生成
│   ├── 06-agents/                  # Agent系统
│   └── 07-alignment/               # RLHF、DPO对齐
│
├── 11-multimodal-learning/         # 多模态学习
│   ├── 01-vision-language/         # CLIP、BLIP、LLaVA
│   ├── 02-image-generation/        # VAE、Diffusion、ControlNet
│   └── 03-audio-models/            # Whisper、TTS
│
├── 12-deployment-optimization/     # 部署优化
│   ├── 01-model-optimization/      # 量化、剪枝、蒸馏
│   ├── 02-inference-engines/       # TensorRT、vLLM
│   ├── 03-serving-systems/         # FastAPI、gRPC
│   └── 04-mlops/                   # 实验追踪、监控
│
└── 13-distributed-training/        # 分布式训练
    ├── 01-data-parallel/           # DDP、FSDP、ZeRO
    ├── 02-model-parallel/          # 张量并行、流水线并行
    ├── 03-mixed-precision/         # AMP、BF16、梯度缩放
    └── 04-large-scale-training/    # DeepSpeed、Megatron
```

</details>

---

## 核心算法覆盖

### 机器学习基础

| 领域 | 算法 | 应用场景 |
|:-----|:-----|:---------|
| **线性模型** | OLS, Ridge, Lasso, ElasticNet | 回归预测、特征选择 |
| **分类算法** | Logistic Regression, SVM, KNN | 二分类、多分类 |
| **树模型** | Decision Tree, Random Forest, GBDT | 结构化数据建模 |
| **集成学习** | Bagging, Boosting, Stacking, XGBoost, LightGBM | 竞赛首选方案 |
| **降维聚类** | PCA, t-SNE, UMAP, K-Means, DBSCAN | 数据可视化、无监督学习 |

### 深度学习

| 领域 | 技术 | 核心概念 |
|:-----|:-----|:---------|
| **优化器** | SGD, Momentum, Adam, AdamW, LAMB | 收敛速度、泛化性能 |
| **正则化** | Dropout, BatchNorm, LayerNorm, Weight Decay | 防止过拟合 |
| **初始化** | Xavier, He, Orthogonal | 梯度稳定性 |
| **学习率** | Step Decay, Cosine Annealing, Warmup | 训练策略 |

### 计算机视觉

**CNN架构演进**:
```
LeNet (1998) → AlexNet (2012) → VGG (2014) → GoogLeNet (2014) → ResNet (2015)
                                                                      ↓
                        ViT (2020) ← EfficientNet (2019) ← DenseNet (2016)
```

| 任务 | 模型/方法 | 说明 |
|:-----|:----------|:-----|
| **图像分类** | ResNet, EfficientNet, ViT | ImageNet SOTA |
| **目标检测** | YOLO, Faster R-CNN, DETR | 实时检测 |
| **语义分割** | U-Net, DeepLab, Mask R-CNN | 像素级分类 |
| **迁移学习** | Fine-tuning, Feature Extraction | 小样本学习 |

### 自然语言处理

**Transformer架构** *(Vaswani et al., 2017)*:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| 任务 | 模型 | 应用 |
|:-----|:-----|:-----|
| **文本分类** | BERT, RoBERTa | 情感分析、意图识别 |
| **序列标注** | BiLSTM-CRF, BERT-NER | 命名实体识别 |
| **文本生成** | GPT, T5 | 摘要、对话 |
| **机器翻译** | Transformer, mBART | 多语言翻译 |

### 生成模型

| 类型 | 模型 | 应用 |
|:-----|:-----|:-----|
| **VAE** | Variational Autoencoder | 图像生成、表征学习 |
| **GAN** | DCGAN, WGAN, StyleGAN | 图像合成、风格迁移 |
| **Diffusion** | DDPM, Stable Diffusion | 高质量图像生成 |
| **Neural Art** | DeepDream, Neural Style Transfer | 艺术创作 |

### 强化学习

| 类别 | 算法 | 特点 |
|:-----|:-----|:-----|
| **值函数方法** | Q-Learning, DQN, Double DQN, Dueling DQN | 经验回放、目标网络 |
| **策略梯度** | REINFORCE, PPO, TRPO, A3C | 直接优化策略 |
| **Actor-Critic** | A2C, SAC, TD3 | 结合值函数与策略 |
| **Model-Based** | Dyna-Q, World Models, MuZero | 环境建模 |

**Bellman最优方程**:

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

### 大语言模型

| 领域 | 技术 | 应用 |
|:-----|:-----|:-----|
| **模型架构** | Transformer、GPT、LLaMA | 文本生成、对话系统 |
| **高效微调** | LoRA、QLoRA、Adapter | 低资源微调 |
| **提示工程** | Few-shot、CoT、ReAct | 任务引导 |
| **检索增强** | RAG、向量数据库 | 知识问答 |
| **Agent系统** | 工具调用、记忆管理 | 自主任务执行 |
| **对齐训练** | RLHF、DPO | 安全对齐 |

### 多模态学习

| 领域 | 模型 | 应用 |
|:-----|:-----|:-----|
| **视觉-语言** | CLIP、BLIP、LLaVA | 图文理解、VQA |
| **图像生成** | VAE、Diffusion、ControlNet | 文生图、图像编辑 |
| **语音识别** | Whisper | 多语言ASR |
| **语音合成** | Tacotron、HiFi-GAN | 文本转语音 |

### 部署优化

| 领域 | 技术 | 说明 |
|:-----|:-----|:-----|
| **模型优化** | 量化、剪枝、蒸馏 | 模型压缩 |
| **推理引擎** | TensorRT、vLLM、Triton | 高性能推理 |
| **服务系统** | FastAPI、gRPC | API服务 |
| **MLOps** | 实验追踪、模型注册、监控 | 生产运维 |

### 分布式训练

| 领域 | 技术 | 说明 |
|:-----|:-----|:-----|
| **数据并行** | DDP、FSDP、ZeRO | 梯度同步、内存优化 |
| **模型并行** | 张量并行、流水线并行、序列并行 | 大模型切分 |
| **混合精度** | AMP、BF16、梯度缩放 | 加速训练、节省显存 |
| **大规模训练** | DeepSpeed、Megatron-LM | 百亿参数训练 |

---

## 技术栈

<div align="center">

| 深度学习 | 数据科学 | 开发工具 |
|:--------:|:--------:|:--------:|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white) |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | ![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Lab_4+-F37626?style=flat-square&logo=jupyter&logoColor=white) |
| ![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=flat-square&logo=keras&logoColor=white) | ![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white) | ![Docker](https://img.shields.io/badge/Docker-24+-2496ED?style=flat-square&logo=docker&logoColor=white) |

</div>

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 创建环境
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# 安装依赖
pip install -r requirements.txt

# 启动Jupyter
jupyter lab
```

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|:-----|:--------:|:--------:|
| CPU | 4核 | 8核+ |
| 内存 | 8 GB | 32 GB |
| GPU | GTX 1060 | RTX 3080+ |
| 存储 | 50 GB | 200 GB SSD |

---

## 竞赛成绩

<div align="center">

| 竞赛 | 排名 | 奖牌 | 年份 |
|:-----|:----:|:----:|:----:|
| **Feedback Prize - ELL** | Top 1% | 金牌 | 2023 |
| **RSNA Abdominal Trauma** | Top 1% | 金牌 | 2023 |
| American Express Default | Top 5% | 银牌 | 2022 |
| RSNA Lumbar Spine | Top 10% | 铜牌 | 2024 |

</div>

---

## 引用

```bibtex
@misc{ai-practices2024,
  author       = {zimingttkx},
  title        = {AI-Practices: 系统化人工智能学习与研究平台},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/zimingttkx/AI-Practices}}
}
```

---

## 许可证

本项目采用 **MIT License** 开源协议 - 详见 [LICENSE](LICENSE)

---

<div align="center">

**如果这个项目对你有帮助，请给一个 Star！**

[![Report Bug](https://img.shields.io/badge/Report-Bug-red?style=for-the-badge)](https://github.com/zimingttkx/AI-Practices/issues)
[![Request Feature](https://img.shields.io/badge/Request-Feature-blue?style=for-the-badge)](https://github.com/zimingttkx/AI-Practices/issues)

</div>
