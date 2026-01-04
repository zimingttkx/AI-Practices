<div align="center">

# AI-Practices

### A Systematic Approach to AI Research & Engineering

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

[![Stars](https://img.shields.io/github/stars/zimingttkx/AI-Practices?style=for-the-badge&logo=github&color=yellow)](https://github.com/zimingttkx/AI-Practices/stargazers)
[![Forks](https://img.shields.io/github/forks/zimingttkx/AI-Practices?style=for-the-badge&logo=github&color=blue)](https://github.com/zimingttkx/AI-Practices/network/members)
[![Issues](https://img.shields.io/github/issues/zimingttkx/AI-Practices?style=for-the-badge&logo=github&color=red)](https://github.com/zimingttkx/AI-Practices/issues)

**[中文](./README.md)** | **[Documentation](https://zimingttkx.github.io/AI-Practices/)** | **[Quick Start](#-quick-start)**

---

**From Theory to Practice, Build a Complete AI Knowledge System**

*Machine Learning • Deep Learning • Computer Vision • NLP • LLM • Multimodal • Reinforcement Learning*

</div>

---

## Highlights

<div align="center">

| **400+ Notebooks** | **14 Core Modules** | **1200+ Unit Tests** | **Production-Ready** | **2 Kaggle Golds** |
|:------------------:|:-------------------:|:--------------------:|:--------------------:|:------------------:|
| Reproducible Experiments | Systematic Learning Path | Quality Assurance | Engineering Practice | Competition Verified |

</div>

### Why AI-Practices?

- **Progressive Learning** — From math derivation to framework engineering, step by step
- **Theory + Practice** — Not just "how to do", but "why it works"
- **Engineering-Oriented** — Complete pipeline from research to deployment
- **Competition Verified** — Kaggle Top 1% Gold Medal solutions

---

## Learning Path

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Phase I        │    │  Phase II       │    │  Phase III      │    │  Phase IV       │
│  Theory First   │ -> │  From Scratch   │ -> │  Framework      │ -> │  Practice       │
│  Math & Analysis│    │  NumPy Impl.    │    │  PyTorch/TF     │    │  Kaggle & Proj  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Module Overview

| Phase | Module | Content | Files |
|:-----:|:-------|:--------|:-----:|
| I | **01-Foundations** | Linear Models, SVM, Decision Trees, Ensemble, Dimensionality Reduction | 75+ |
| II | **02-Neural Networks** | Backprop, Optimizers, Regularization, BatchNorm | 42+ |
| II | **03-Computer Vision** | CNN Architectures, Transfer Learning, Object Detection | 50+ |
| II | **04-Sequence Models** | RNN/LSTM, Attention, Transformer | 40+ |
| III | **05-Advanced Topics** | Distributed Training, Model Compression, Deployment | 30+ |
| III | **06-Generative Models** | VAE, GAN, Diffusion Models | 35+ |
| III | **07-Reinforcement Learning** | DQN, PPO, SAC, Actor-Critic | 80+ |
| IV | **09-Practical Projects** | Kaggle Competitions, Game AI, Stock Trading | 100+ |
| - | **08-Theory Notes** | Activation Functions, Loss Functions, Architecture Guide | 16+ |
| V | **10-Large Language Models** | Transformer, GPT/LLaMA, LoRA, RAG, Agents | 60+ |
| V | **11-Multimodal Learning** | CLIP, Stable Diffusion, Whisper, TTS | 47+ |
| V | **12-Deployment Optimization** | Quantization, TensorRT, FastAPI, MLOps | 50+ |
| VI | **13-Distributed Training** | DDP, FSDP, ZeRO, Tensor/Pipeline Parallel, Mixed Precision | 42+ |
| VII | **14-Agents & Reasoning** | Tool Use, CoT, ReAct, ToT, Multi-Agent | 30+ |

<details>
<summary><b>📂 Expand Full Directory Structure</b></summary>

```
AI-Practices/
├── 01-foundations/                 # ML Fundamentals
│   ├── 01-training-models/         # Gradient Descent, Regularization
│   ├── 02-classification/          # Logistic Regression, SVM
│   ├── 03-support-vector-machines/ # Kernel Trick, Soft Margin
│   ├── 04-decision-trees/          # CART, Pruning
│   ├── 05-ensemble-learning/       # Bagging, Boosting, Stacking
│   ├── 06-dimensionality-reduction/# PCA, t-SNE, UMAP
│   ├── 07-unsupervised-learning/   # K-Means, DBSCAN, GMM
│   └── 08-end-to-end-project/      # Complete ML Pipeline
│
├── 02-neural-networks/             # Neural Networks
│   ├── 01-keras-introduction/      # Sequential, Functional API
│   ├── 02-training-deep-networks/  # BatchNorm, Dropout
│   ├── 03-custom-models-training/  # Custom Layers and Training Loops
│   └── 04-data-loading-preprocessing/
│
├── 07-reinforcement-learning/      # Reinforcement Learning
│   ├── 01-mdp-basics/              # Markov Decision Process
│   ├── 02-temporal-difference/     # TD Learning
│   ├── 03-q-learning/              # Q-Learning
│   ├── 04-deep-q-learning/         # DQN, Double DQN
│   ├── 05-policy-gradient/         # REINFORCE, PPO
│   ├── 06-actor-critic/            # A2C, A3C
│   └── 07-advanced-algorithms/     # SAC, TD3
│
├── 09-practical-projects/          # Practical Projects
│   ├── 01-ml-basics/               # Titanic, Otto
│   ├── 02-computer-vision/         # MNIST CNN
│   ├── 03-nlp/                     # Sentiment Analysis, NER
│   ├── 04-time-series/             # Temperature Prediction
│   ├── 05-kaggle-competitions/     # Gold Medal Solutions
│   └── 06-reinforcement-learning/  # Game AI, Stock Trading
│
├── 10-large-language-models/       # Large Language Models
│   ├── 01-llm-fundamentals/        # Transformer Architecture
│   ├── 02-pretrained-models/       # GPT, LLaMA Implementation
│   ├── 03-fine-tuning/             # LoRA, QLoRA Fine-tuning
│   ├── 04-prompt-engineering/      # Prompt Engineering
│   ├── 05-rag/                     # Retrieval Augmented Generation
│   ├── 06-agents/                  # Agent Systems
│   └── 07-alignment/               # RLHF, DPO Alignment
│
├── 11-multimodal-learning/         # Multimodal Learning
│   ├── 01-vision-language/         # CLIP, BLIP, LLaVA
│   ├── 02-image-generation/        # VAE, Diffusion, ControlNet
│   └── 03-audio-models/            # Whisper, TTS
│
├── 12-deployment-optimization/     # Deployment Optimization
│   ├── 01-model-optimization/      # Quantization, Pruning, Distillation
│   ├── 02-inference-engines/       # TensorRT, vLLM
│   ├── 03-serving-systems/         # FastAPI, gRPC
│   └── 04-mlops/                   # Experiment Tracking, Monitoring
│
└── 13-distributed-training/        # Distributed Training
    ├── 01-data-parallel/           # DDP, FSDP, ZeRO
    ├── 02-model-parallel/          # Tensor Parallel, Pipeline Parallel
    ├── 03-mixed-precision/         # AMP, BF16, Gradient Scaling
    └── 04-large-scale-training/    # DeepSpeed, Megatron

└── 14-agents-reasoning/            # Agents & Reasoning
    ├── 01-tool-use/                # Tool Calling, Function Calling
    ├── 02-reasoning/               # CoT, ReAct, ToT, Reflection
    ├── 03-memory-systems/          # Short/Long-term Memory
    ├── 04-planning/                # Task Decomposition & Planning
    └── 05-multi-agent/             # Multi-Agent Collaboration
```

</details>

---

## Algorithm Coverage

### Machine Learning Fundamentals

| Domain | Algorithms | Applications |
|:-------|:-----------|:-------------|
| **Linear Models** | OLS, Ridge, Lasso, ElasticNet | Regression, Feature Selection |
| **Classification** | Logistic Regression, SVM, KNN | Binary/Multi-class Classification |
| **Tree Models** | Decision Tree, Random Forest, GBDT | Structured Data Modeling |
| **Ensemble** | Bagging, Boosting, Stacking, XGBoost, LightGBM | Competition Winners |
| **Dim Reduction** | PCA, t-SNE, UMAP, K-Means, DBSCAN | Visualization, Clustering |

### Deep Learning

| Domain | Techniques | Key Concepts |
|:-------|:-----------|:-------------|
| **Optimizers** | SGD, Momentum, Adam, AdamW, LAMB | Convergence, Generalization |
| **Regularization** | Dropout, BatchNorm, LayerNorm, Weight Decay | Prevent Overfitting |
| **Initialization** | Xavier, He, Orthogonal | Gradient Stability |
| **LR Schedule** | Step Decay, Cosine Annealing, Warmup | Training Strategy |

### Computer Vision

**CNN Architecture Evolution**:
```
LeNet (1998) → AlexNet (2012) → VGG (2014) → GoogLeNet (2014) → ResNet (2015)
                                                                      ↓
                        ViT (2020) ← EfficientNet (2019) ← DenseNet (2016)
```

| Task | Models | Description |
|:-----|:-------|:------------|
| **Classification** | ResNet, EfficientNet, ViT | ImageNet SOTA |
| **Detection** | YOLO, Faster R-CNN, DETR | Real-time Detection |
| **Segmentation** | U-Net, DeepLab, Mask R-CNN | Pixel-level Classification |
| **Transfer Learning** | Fine-tuning, Feature Extraction | Few-shot Learning |

### Natural Language Processing

**Transformer Architecture** *(Vaswani et al., 2017)*:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Task | Models | Applications |
|:-----|:-------|:-------------|
| **Text Classification** | BERT, RoBERTa | Sentiment, Intent |
| **Sequence Labeling** | BiLSTM-CRF, BERT-NER | Named Entity Recognition |
| **Text Generation** | GPT, T5 | Summarization, Dialogue |
| **Translation** | Transformer, mBART | Multilingual Translation |

### Generative Models

| Type | Models | Applications |
|:-----|:-------|:-------------|
| **VAE** | Variational Autoencoder | Image Generation, Representation |
| **GAN** | DCGAN, WGAN, StyleGAN | Image Synthesis, Style Transfer |
| **Diffusion** | DDPM, Stable Diffusion | High-quality Generation |
| **Neural Art** | DeepDream, Neural Style Transfer | Artistic Creation |

### Reinforcement Learning

| Category | Algorithms | Key Features |
|:---------|:-----------|:-------------|
| **Value-Based** | Q-Learning, DQN, Double DQN, Dueling DQN | Experience Replay, Target Network |
| **Policy Gradient** | REINFORCE, PPO, TRPO, A3C | Direct Policy Optimization |
| **Actor-Critic** | A2C, SAC, TD3 | Combines Value and Policy |
| **Model-Based** | Dyna-Q, World Models, MuZero | Environment Modeling |

**Bellman Optimality Equation**:

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

### Large Language Models

| Domain | Techniques | Applications |
|:-------|:-----------|:-------------|
| **Architecture** | Transformer, GPT, LLaMA | Text Generation, Dialogue |
| **Efficient Fine-tuning** | LoRA, QLoRA, Adapter | Low-resource Tuning |
| **Prompt Engineering** | Few-shot, CoT, ReAct | Task Guidance |
| **Retrieval Augmented** | RAG, Vector Databases | Knowledge QA |
| **Agent Systems** | Tool Calling, Memory | Autonomous Tasks |
| **Alignment** | RLHF, DPO | Safety Alignment |

### Multimodal Learning

| Domain | Models | Applications |
|:-------|:-------|:-------------|
| **Vision-Language** | CLIP, BLIP, LLaVA | Image-Text Understanding, VQA |
| **Image Generation** | VAE, Diffusion, ControlNet | Text-to-Image, Image Editing |
| **Speech Recognition** | Whisper | Multilingual ASR |
| **Speech Synthesis** | Tacotron, HiFi-GAN | Text-to-Speech |

### Deployment Optimization

| Domain | Techniques | Description |
|:-------|:-----------|:------------|
| **Model Optimization** | Quantization, Pruning, Distillation | Model Compression |
| **Inference Engines** | TensorRT, vLLM, Triton | High-performance Inference |
| **Serving Systems** | FastAPI, gRPC | API Services |
| **MLOps** | Experiment Tracking, Model Registry, Monitoring | Production Operations |

### Distributed Training

| Domain | Techniques | Description |
|:-------|:-----------|:------------|
| **Data Parallel** | DDP, FSDP, ZeRO | Gradient Sync, Memory Optimization |
| **Model Parallel** | Tensor Parallel, Pipeline Parallel, Sequence Parallel | Large Model Partitioning |
| **Mixed Precision** | AMP, BF16, Gradient Scaling | Faster Training, Memory Saving |
| **Large-Scale Training** | DeepSpeed, Megatron-LM | Billion-Parameter Training |

### Agents & Reasoning

| Domain | Techniques | Description |
|:-------|:-----------|:------------|
| **Tool Use** | Function Calling, Tool Registry, Structured Output | LLM Tool Integration |
| **Reasoning** | CoT, ReAct, ToT, Self-Consistency | Enhanced Reasoning |
| **Memory Systems** | Short-term, Long-term, Vector Retrieval | Context Management |
| **Planning** | Task Decomposition, Plan Generation, Re-planning | Complex Task Handling |
| **Multi-Agent** | Debate, Collaboration, Consensus | Multi-Agent Cooperation |

---

## Tech Stack

<div align="center">

| Deep Learning | Data Science | Development |
|:-------------:|:------------:|:-----------:|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white) |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | ![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Lab_4+-F37626?style=flat-square&logo=jupyter&logoColor=white) |
| ![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=flat-square&logo=keras&logoColor=white) | ![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white) | ![Docker](https://img.shields.io/badge/Docker-24+-2496ED?style=flat-square&logo=docker&logoColor=white) |

</div>

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# Create environment
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

### Hardware Requirements

| Component | Minimum | Recommended |
|:----------|:-------:|:-----------:|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 32 GB |
| GPU | GTX 1060 | RTX 3080+ |
| Storage | 50 GB | 200 GB SSD |

---

## Competition Results

<div align="center">

| Competition | Rank | Medal | Year |
|:------------|:----:|:-----:|:----:|
| **Feedback Prize - ELL** | Top 1% | Gold | 2023 |
| **RSNA Abdominal Trauma** | Top 1% | Gold | 2023 |
| American Express Default | Top 5% | Silver | 2022 |
| RSNA Lumbar Spine | Top 10% | Bronze | 2024 |

</div>

---

## Citation

```bibtex
@misc{ai-practices2024,
  author       = {zimingttkx},
  title        = {AI-Practices: A Systematic Approach to AI Research and Engineering},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/zimingttkx/AI-Practices}}
}
```

---

## License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

<div align="center">

**If this project helps you, please give it a Star!**

[![Report Bug](https://img.shields.io/badge/Report-Bug-red?style=for-the-badge)](https://github.com/zimingttkx/AI-Practices/issues)
[![Request Feature](https://img.shields.io/badge/Request-Feature-blue?style=for-the-badge)](https://github.com/zimingttkx/AI-Practices/issues)

</div>
