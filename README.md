<div align="center">

# 🚀 AI-Practices

### 系统化人工智能学习与实践平台

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

[![Stars](https://img.shields.io/github/stars/zimingttkx/AI-Practices?style=for-the-badge&logo=github&color=yellow)](https://github.com/zimingttkx/AI-Practices/stargazers)
[![Forks](https://img.shields.io/github/forks/zimingttkx/AI-Practices?style=for-the-badge&logo=github&color=blue)](https://github.com/zimingttkx/AI-Practices/network/members)

**[English](./README_EN.md)** | **[快速开始](#-快速开始)** | **[项目路线图](./ROADMAP.md)**

---

*从数学原理到工程实践，构建完整的AI知识体系*

</div>

---

## 项目概览

<div align="center">

| **500+ Python文件** | **280+ Notebooks** | **14大核心模块** | **100+ 单元测试** | **2枚Kaggle金牌** |
|:------------------:|:------------------:|:----------------:|:----------------:|:----------------:|
| 生产级代码实现 | 可交互式学习 | 系统化知识体系 | 代码质量保证 | 竞赛实战验证 |

</div>

### 核心特点

- **系统化学习路径** — 从基础数学到前沿技术，14个模块循序渐进
- **理论与实践结合** — 每个概念都有数学推导和代码实现
- **工程化标准** — 遵循工业级代码规范，包含完整测试
- **竞赛级方案** — 包含Kaggle Top 1%金牌解决方案

---

## 模块架构

```
AI-Practices/
│
├── 第一阶段：机器学习基础
│   └── 01-foundations/              # 线性模型、SVM、决策树、集成学习、降维、聚类
│
├── 第二阶段：深度学习核心
│   ├── 02-neural-networks/          # 神经网络基础、优化器、正则化
│   ├── 03-computer-vision/          # CNN架构、迁移学习、目标检测
│   └── 04-sequence-models/          # RNN/LSTM、Attention、Transformer
│
├── 第三阶段：高级专题
│   ├── 05-advanced-topics/          # 函数式API、回调函数、模型优化
│   ├── 06-generative-models/        # VAE、GAN、Diffusion Models
│   └── 07-reinforcement-learning/   # DQN、PPO、SAC、Actor-Critic
│
├── 第四阶段：大模型与多模态
│   ├── 10-large-language-models/    # Transformer、GPT/LLaMA、LoRA、RAG、Agent
│   └── 11-multimodal-learning/      # CLIP、Stable Diffusion、Whisper、TTS
│
├── 第五阶段：工程化部署
│   ├── 12-deployment-optimization/  # 量化剪枝、TensorRT、FastAPI、MLOps
│   └── 13-distributed-training/     # DDP、FSDP、ZeRO、混合精度训练
│
├── 第六阶段：智能体系统
│   └── 14-agents-reasoning/         # 工具调用、推理策略、多智能体、自主Agent
│
├── 理论参考
│   └── 08-theory-notes/             # 激活函数、损失函数、架构速查
│
└── 实战项目
    └── 09-practical-projects/       # Kaggle竞赛、游戏AI、跨模块集成系统
```

---

## 详细模块说明

### 01-foundations | 机器学习基础

| 子模块 | 核心内容 | 关键算法 |
|:-------|:---------|:---------|
| 01-training-models | 模型训练基础 | 线性回归、梯度下降、正则化 |
| 02-classification | 分类算法 | 逻辑回归、MNIST实战 |
| 03-support-vector-machines | 支持向量机 | 核技巧、软间隔、SVM回归 |
| 04-decision-trees | 决策树 | CART、剪枝策略 |
| 05-ensemble-learning | 集成学习 | Bagging、Boosting、XGBoost、Stacking |
| 06-dimensionality-reduction | 降维技术 | PCA、t-SNE、LLE、UMAP |
| 07-unsupervised-learning | 无监督学习 | K-Means、DBSCAN、GMM |
| 08-end-to-end-project | 完整ML项目 | 加州房价预测 |

### 02-neural-networks | 神经网络

| 子模块 | 核心内容 |
|:-------|:---------|
| 01-keras-introduction | Keras入门、Sequential/Functional API |
| 02-training-deep-networks | BatchNorm、Dropout、初始化策略 |
| 03-custom-models-training | 自定义层、训练循环、TensorFlow底层 |
| 04-data-loading-preprocessing | 数据管道、TFRecord、预处理 |

### 03-computer-vision | 计算机视觉

| 子模块 | 核心内容 |
|:-------|:---------|
| 01-cnn-basics | CNN基础、池化层、ResNet实现 |
| 02-classic-architectures | 经典架构演进 |
| 03-transfer-learning | 迁移学习、猫狗分类实战 |
| 04-visualization | 特征可视化、中间层激活 |

### 04-sequence-models | 序列模型

| 子模块 | 核心内容 |
|:-------|:---------|
| 01-rnn-basics | RNN基础、LSTM、时间序列预测 |
| 02-lstm-gru | LSTM/GRU高级用法 |
| 03-text-processing | 词嵌入、One-hot编码 |
| 04-cnn-for-sequences | 一维卷积处理序列 |
| 05-transformer | Self-Attention、Multi-Head、BERT/GPT基础 |

### 05-advanced-topics | 高级专题

| 子模块 | 核心内容 |
|:-------|:---------|
| 01-functional-api | 多输入多输出、残差连接、Inception |
| 02-callbacks-tensorboard | 回调函数、TensorBoard可视化 |
| 03-model-optimization | 量化、剪枝、知识蒸馏、部署 |

### 06-generative-models | 生成模型

| 子模块 | 核心内容 |
|:-------|:---------|
| 01-vae | Vanilla AE、VAE、VQ-VAE |
| 02-gans | GAN、DCGAN、WGAN-GP |
| 03-diffusion | DDPM原理与实现 |
| 04-text-generation | 字符级LSTM文本生成 |
| 05-deepdream | DeepDream艺术生成 |

### 07-reinforcement-learning | 强化学习

| 子模块 | 核心内容 | 测试覆盖 |
|:-------|:---------|:---------|
| 01-mdp-basics | MDP、值迭代、策略迭代 | ✅ |
| 02-temporal-difference | TD学习、SARSA | ✅ |
| 03-q-learning | Q-Learning、探索策略 | ✅ |
| 04-deep-q-learning | DQN、Double DQN、Dueling DQN、Rainbow | ✅ |
| 05-policy-gradient | REINFORCE、基线方法 | ✅ |
| 06-actor-critic | A2C、PPO | ✅ |
| 07-advanced-algorithms | SAC、TD3、DDPG | ✅ |
| 08-reward-optimization | 奖励塑形、好奇心驱动、逆强化学习 | ✅ |

### 08-theory-notes | 理论笔记

快速参考手册，包含：
- 激活函数对比与选择
- 损失函数详解
- 网络架构速查（CNN、RNN、Dense）

### 09-practical-projects | 实战项目

| 子模块 | 项目内容 |
|:-------|:---------|
| 01-ml-basics | Titanic生存预测、Otto分类、SVM文本分类、XGBoost进阶 |
| 02-computer-vision | MNIST CNN分类 |
| 03-nlp | 情感分析LSTM、Transformer文本分类、NER、机器翻译 |
| 04-time-series | 温度预测、股票预测LSTM |
| 05-kaggle-competitions | **4个Kaggle竞赛方案**（含2个金牌） |
| 06-reinforcement-learning | Flappy Bird DQN、Dino Run、股票交易RL |
| 07-integrated-systems | 多模态检索、视觉问答Agent、代码助手（109个测试） |

#### Kaggle竞赛成绩

| 竞赛 | 排名 | 奖牌 |
|:-----|:----:|:----:|
| Feedback Prize - ELL | Top 1% | 🥇 金牌 |
| RSNA Abdominal Trauma | Top 1% | 🥇 金牌 |
| American Express Default | Top 5% | 🥈 银牌 |
| RSNA Lumbar Spine | Top 10% | 🥉 铜牌 |

### 10-large-language-models | 大语言模型

| 子模块 | 核心内容 | 测试覆盖 |
|:-------|:---------|:---------|
| 01-llm-fundamentals | Transformer架构、Tokenizer | ✅ |
| 02-pretrained-models | GPT、LLaMA从零实现 | ✅ |
| 03-fine-tuning | LoRA、QLoRA高效微调 | ✅ |
| 04-prompt-engineering | Few-shot、Chain-of-Thought | ✅ |
| 05-rag | 向量数据库、检索增强生成 | ✅ |
| 06-agents | 工具调用、记忆管理 | ✅ |
| 07-alignment | RLHF、DPO对齐训练 | ✅ |

### 11-multimodal-learning | 多模态学习

| 子模块 | 核心内容 | 测试覆盖 |
|:-------|:---------|:---------|
| 01-vision-language | CLIP、BLIP、LLaVA | ✅ |
| 02-image-generation | VAE、Diffusion、ControlNet | ✅ |
| 03-audio-models | Whisper语音识别、TTS语音合成 | ✅ |

### 12-deployment-optimization | 部署优化

| 子模块 | 核心内容 | 测试覆盖 |
|:-------|:---------|:---------|
| 01-model-optimization | 量化、剪枝、蒸馏、ONNX导出 | ✅ |
| 02-inference-engines | TensorRT、vLLM、ONNX Runtime | ✅ |
| 03-serving-systems | FastAPI、Triton、负载均衡 | ✅ |
| 04-mlops | 实验追踪、模型注册、监控告警 | ✅ |

### 13-distributed-training | 分布式训练

| 子模块 | 核心内容 | 测试覆盖 |
|:-------|:---------|:---------|
| 01-data-parallel | DDP、FSDP、ZeRO | ✅ |
| 02-model-parallel | 张量并行、流水线并行、序列并行 | ✅ |
| 03-mixed-precision | AMP、BF16、梯度缩放 | ✅ |
| 04-large-scale-training | DeepSpeed、Megatron-LM | ✅ |

### 14-agents-reasoning | 智能体与推理

| 子模块 | 核心内容 | 测试覆盖 |
|:-------|:---------|:---------|
| 01-tool-use | Function Calling、工具注册、结构化输出 | ✅ |
| 02-reasoning | CoT、ReAct、ToT、自一致性、反思 | ✅ |
| 03-memory-systems | 短期记忆、长期记忆、向量检索 | ✅ |
| 04-planning | 任务分解、计划生成、动态重规划 | ✅ |
| 05-multi-agent | 辩论式推理、协作式推理、共识达成 | ✅ |
| 06-autonomous-agent | 目标管理、行动执行、自主循环 | ✅ |

---

## 核心算法覆盖

### 机器学习

```
线性模型: OLS, Ridge, Lasso, ElasticNet
分类算法: Logistic Regression, SVM, KNN
树模型: Decision Tree, Random Forest, GBDT
集成学习: Bagging, Boosting, Stacking, XGBoost, LightGBM
降维聚类: PCA, t-SNE, UMAP, K-Means, DBSCAN
```

### 深度学习

```
优化器: SGD, Momentum, Adam, AdamW, LAMB
正则化: Dropout, BatchNorm, LayerNorm, Weight Decay
CNN架构: LeNet → AlexNet → VGG → ResNet → EfficientNet → ViT
序列模型: RNN → LSTM → GRU → Transformer → BERT → GPT
```

### 强化学习

```
值函数方法: Q-Learning, DQN, Double DQN, Dueling DQN, Rainbow
策略梯度: REINFORCE, PPO, TRPO
Actor-Critic: A2C, A3C, SAC, TD3
```

### 生成模型

```
自编码器: AE, VAE, VQ-VAE
对抗网络: GAN, DCGAN, WGAN-GP, StyleGAN
扩散模型: DDPM, Stable Diffusion
```

### 大模型技术

```
架构: Transformer, GPT, LLaMA
微调: LoRA, QLoRA, Adapter
推理: RAG, CoT, ReAct, ToT
对齐: RLHF, DPO
```

---

## 技术栈

<div align="center">

| 深度学习框架 | 数据科学 | 开发工具 |
|:------------:|:--------:|:--------:|
| PyTorch 2.x | NumPy | Python 3.10+ |
| TensorFlow 2.13+ | Pandas | Jupyter Lab |
| Keras 3.x | Scikit-Learn | Docker |

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

## 学习路径建议

```
入门阶段 (1-2个月)
├── 01-foundations          # 机器学习基础
├── 02-neural-networks      # 神经网络入门
└── 08-theory-notes         # 理论参考

进阶阶段 (2-3个月)
├── 03-computer-vision      # 计算机视觉
├── 04-sequence-models      # 序列模型
├── 05-advanced-topics      # 高级专题
└── 06-generative-models    # 生成模型

高级阶段 (2-3个月)
├── 07-reinforcement-learning  # 强化学习
├── 10-large-language-models   # 大语言模型
└── 11-multimodal-learning     # 多模态学习

工程化阶段 (1-2个月)
├── 12-deployment-optimization # 部署优化
├── 13-distributed-training    # 分布式训练
└── 14-agents-reasoning        # 智能体系统

实战阶段 (持续)
└── 09-practical-projects      # 项目实战
```

---

## 引用

```bibtex
@misc{ai-practices2024,
  author       = {zimingttkx},
  title        = {AI-Practices: 系统化人工智能学习与实践平台},
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

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

[![Report Bug](https://img.shields.io/badge/Report-Bug-red?style=for-the-badge)](https://github.com/zimingttkx/AI-Practices/issues)
[![Request Feature](https://img.shields.io/badge/Request-Feature-blue?style=for-the-badge)](https://github.com/zimingttkx/AI-Practices/issues)

</div>
