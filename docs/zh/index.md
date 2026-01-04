---
layout: home

hero:
  name: AI-Practices
  text: 全栈 AI 学习实验室
  tagline: 系统化、工程化的人工智能学习与研究平台
  image:
    src: /logo.svg
    alt: AI-Practices
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/guide/getting-started
    - theme: alt
      text: 课程模块
      link: /zh/modules/
    - theme: alt
      text: GitHub
      link: https://github.com/zimingttkx/AI-Practices

features:
  - icon: 📊
    title: 400+ 可复现实验
    details: 每个算法都有完整的 Jupyter Notebook 实现，含详细注释、数学推导与可视化分析
  - icon: 🧠
    title: 14 大核心模块
    details: 渐进式课程设计，从机器学习基础到智能体推理，覆盖 AI 全技术栈
  - icon: 🏆
    title: Kaggle 金牌方案
    details: 包含 Feedback Prize、RSNA 等顶级竞赛的完整解决方案
  - icon: 🔬
    title: 理论与实践结合
    details: 数学推导 → NumPy 实现 → 框架应用 → 实战项目
  - icon: ⚡
    title: 生产级代码质量
    details: 180k+ 行高质量代码，遵循 PEP8 规范，完整类型注解
  - icon: 🌐
    title: 中英双语文档
    details: 完整的双语文档支持，方便国内外开发者学习交流
---

<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #007AFF 30%, #5856D6);
  --vp-home-hero-image-background-image: linear-gradient(-45deg, #007AFF 50%, #5856D6 50%);
  --vp-home-hero-image-filter: blur(44px);
}

.dark {
  --vp-home-hero-image-background-image: linear-gradient(-45deg, #007AFF 50%, #5856D6 50%);
}
</style>

## 渐进式学习框架

本项目采用 **Progressive Learning Framework** 方法论，构建从理论到实战的完整学习闭环：

| 阶段 | 原则 | 方法 | 产出 | 目标 |
|:----:|:-----|:-----|:-----|:-----|
| **Ⅰ** | 理论先行 | 数学推导 + 算法复杂度分析 | 理论笔记 | 🎯 理解原理 |
| **Ⅱ** | 从零实现 | NumPy 手写核心算法 | 核心代码 | 🔧 掌握细节 |
| **Ⅲ** | 框架精通 | PyTorch / TensorFlow 工程化 | 工程代码 | ⚡ 高效开发 |
| **Ⅳ** | 实战检验 | Kaggle 竞赛 + 工业项目 | 完整方案 | 🏆 实战能力 |

---

## 核心模块一览

### 基础模块

| 模块 | 内容 | 核心技术 |
|:-----|:-----|:---------|
| **01 机器学习基础** | 经典 ML 算法原理与实现 | Linear, SVM, XGBoost, LightGBM |
| **02 神经网络** | 深度学习核心技术 | 反向传播, Adam, Dropout, BatchNorm |

### 核心模块

| 模块 | 内容 | 核心技术 |
|:-----|:-----|:---------|
| **03 计算机视觉** | CNN 架构演进与应用 | ResNet, EfficientNet, ViT |
| **04 序列模型** | 从 RNN 到 Transformer | LSTM, Attention, BERT, GPT |

### 进阶模块

| 模块 | 内容 | 核心技术 |
|:-----|:-----|:---------|
| **05 高级专题** | 工程化与优化 | Optuna, DDP, ONNX, TensorRT |
| **06 生成模型** | 生成式 AI | VAE, GAN, Diffusion |
| **07 强化学习** | 决策与控制 | DQN, PPO, SAC |

---

## 实战项目展示

### 🏆 Kaggle 竞赛成绩

| 竞赛 | 排名 | 奖牌 | 奖金池 | 核心技术 |
|:-----|:----:|:----:|:------:|:---------|
| Feedback Prize - ELL | **Top 1%** | 🥇 | $160k | DeBERTa, Multi-task, Pseudo Label |
| RSNA Abdominal Trauma | **Top 1%** | 🥇 | $140k | EfficientNet, 3D CNN |
| American Express Default | Top 5% | 🥈 | $100k | GBDT Ensemble, Feature Engineering |
| RSNA Lumbar Spine | Top 10% | 🥉 | $50k | 3D Medical Imaging |

### 📊 项目分类

::: details ML 基础项目
| 项目 | 描述 | 技术栈 |
|:-----|:-----|:-------|
| Titanic 生存预测 | 经典二分类，特征工程入门 | XGBoost, Pandas |
| Otto 产品分类 | 多分类问题，集成学习 | LightGBM, Stacking |
| House Prices | 回归问题，数据预处理 | Ridge, Lasso |
:::

::: details 计算机视觉项目
| 项目 | 描述 | 技术栈 |
|:-----|:-----|:-------|
| MNIST 手写数字 | CNN 入门经典 | TensorFlow, Keras |
| CIFAR-10 分类 | 多类别图像分类 | ResNet, Augmentation |
| 图像风格迁移 | 神经风格迁移 | VGG19, PyTorch |
:::

::: details NLP 项目
| 项目 | 描述 | 技术栈 |
|:-----|:-----|:-------|
| LSTM 情感分析 | 电影评论分类 | LSTM, Word2Vec |
| Transformer 文本分类 | 注意力机制分类 | Transformer, PyTorch |
| 命名实体识别 | 序列标注任务 | BiLSTM-CRF, BERT |
:::

::: details 时序预测项目
| 项目 | 描述 | 技术栈 |
|:-----|:-----|:-------|
| 温度预测 | 多变量时序预测 | LSTM, Keras |
| 股票预测 | 金融时序分析 | LSTM, Attention |
:::

---

## 技术栈

### 深度学习框架

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras&logoColor=white)

### 机器学习 & 数据处理

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FDD?style=flat-square&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9ACD32?style=flat-square&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white)

---

## 快速开始

::: code-group

```bash [conda]
# 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 创建 Conda 环境
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter Lab
jupyter lab
```

```bash [pip]
# 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter Lab
jupyter lab
```

:::

---

## 学习路线图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI-Practices 学习路线                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  入门 ──► 01 ML基础 ──► 02 神经网络 ──┬──► 03 CV ──┬──► 05 高级 ──┐     │
│           (4-6周)       (3-4周)       │   (3-4周)  │    (2-3周)   │     │
│                                       │            │              │     │
│                                       └──► 04 NLP ─┘              │     │
│                                            (3-4周)                │     │
│                                                                   │     │
│                                       ┌───────────────────────────┘     │
│                                       │                                 │
│                                       ▼                                 │
│                              ┌────────┴────────┐                        │
│                              │                 │                        │
│                              ▼                 ▼                        │
│                         06 生成模型       07 强化学习                    │
│                           (3-4周)          (3-4周)                      │
│                              │                 │                        │
│                              └────────┬────────┘                        │
│                                       │                                 │
│                                       ▼                                 │
│                              09 实战项目 (持续)                          │
│                              • Kaggle 竞赛                              │
│                              • 工业项目                                 │
│                                                                         │
│  08 理论笔记 ◄─────────────── 随时参考 ──────────────────────────────   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 为什么选择 AI-Practices？

::: tip 🎯 系统化学习
不是零散的教程集合，而是精心设计的完整课程体系，从基础到进阶循序渐进。
:::

::: tip 🔬 理论与实践结合
每个算法都有数学推导、NumPy 实现、框架应用三个层次，真正理解而非死记硬背。
:::

::: tip 🏆 真实竞赛验证
包含多个 Kaggle 金牌方案，学习经过实战检验的工业级解决方案。
:::

::: tip 📚 持续更新
紧跟 AI 领域最新进展，定期更新内容和代码。
:::

---

## 开始你的 AI 之旅

<div class="action-buttons">

[📚 查看课程模块](/zh/modules/) | [🚀 快速开始](/zh/guide/getting-started) | [💻 GitHub](https://github.com/zimingttkx/AI-Practices)

</div>
