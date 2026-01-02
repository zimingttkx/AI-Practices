# 07-alignment: AI对齐技术

本模块实现AI对齐的核心技术，包括RLHF（人类反馈强化学习）和DPO（直接偏好优化）。

> **模块状态**: ✅ 已完成 (2026-01-02) | **测试**: ✅ 131个测试全部通过

---

## 目录

1. [核心概念](#核心概念)
2. [理论背景](#理论背景)
3. [模块结构](#模块结构)
4. [快速开始](#快速开始)
5. [组件详解](#组件详解)
6. [训练指南](#训练指南)
7. [最佳实践](#最佳实践)
8. [扩展阅读](#扩展阅读)

---

## 核心概念

### 什么是对齐 (Alignment)？

AI对齐 (AI Alignment) 是指使人工智能系统的行为与人类意图、价值观和期望保持一致的技术和方法。

**核心问题**:
- 模型可能生成有害、偏见或不当内容
- 预训练模型不擅长遵循用户指令
- 模型可能产生幻觉（编造虚假信息）

**对齐技术演进**:
```
预训练 → SFT(监督微调) → RLHF/DPO → Constitutional AI
```

---

## 理论背景

### 核心论文

本模块基于以下开创性研究：

| 论文 | 作者/年份 | 核心贡献 |
|------|-----------|----------|
| **InstructGPT** | Ouyang et al., 2022 | 三阶段RLHF框架 |
| **PPO** | Schulman et al., 2017 | PPO裁剪目标算法 |
| **DPO** | Rafailov et al., 2023 | 直接偏好优化，无需奖励模型 |
| **Constitutional AI** | Bai et al., 2022 | AI反馈代替人类反馈 |

### RLHF vs DPO

| 维度 | RLHF | DPO |
|:-----|:-----|:-----|
| 训练阶段 | 3阶段 | 1阶段 |
| 奖励模型 | 需要 | 不需要 |
| 训练复杂度 | 高 | 低 |
| 超参数数量 | 多 | 少 |
| 稳定性 | 中等 | 高 |

---

## 模块结构

```
07-alignment/
├── src/
│   ├── __init__.py          # 模块导出 (17个组件)
│   ├── reward_model.py      # 奖励模型实现 (~400行)
│   ├── rlhf.py              # RLHF/PPO训练器 (~520行)
│   └── dpo.py               # DPO训练器 (~520行)
│
├── tests/
│   ├── test_reward_model.py # 奖励模型测试
│   ├── test_rlhf.py         # RLHF测试
│   ├── test_dpo.py          # DPO测试
│   └── run_tests.py         # 测试运行器
│
├── notebooks/
│   ├── 01_reward_modeling.ipynb    # 奖励模型详解 (~450行)
│   ├── 02_rlhf_training.ipynb       # PPO训练详解 (~450行)
│   └── 03_dpo_training.ipynb        # DPO训练详解 (~500行)
│
├── knowledge_points.md   # 知识点总结 (完整理论)
└── README.md             # 本文档
```

---

## 快速开始

### 安装依赖

```bash
# 无需额外依赖，纯Python实现
# Python 3.8+ 即可运行
```

### 奖励模型训练

```python
from src import PairwiseRewardModel, PreferenceDataset

# 创建偏好数据集
dataset = PreferenceDataset()
dataset.add(
    prompt="什么是机器学习？",
    chosen="机器学习是人工智能的一个分支...",
    rejected="不知道"
)

# 创建并训练奖励模型
model = PairwiseRewardModel(hidden_size=256)
metrics = model.train_step(dataset.sample(4))

print(f"Loss: {metrics['loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### PPO训练 (RLHF)

```python
from src import PPOTrainer, RLHFConfig

# 配置PPO训练器
config = RLHFConfig(
    learning_rate=1e-5,
    ppo_epochs=4,
    clip_epsilon=0.2,
    kl_coef=0.1
)

trainer = PPOTrainer(config)

# 生成回复并计算奖励
batch = trainer.generate_and_score(["问题1", "问题2"])

# PPO训练步
metrics = trainer.train_step(batch)
print(f"Policy Loss: {metrics['policy_loss']:.4f}")
print(f"KL Divergence: {metrics['kl']:.4f}")
```

### DPO训练

```python
from src import DPOTrainer, DPOBatch, DPOConfig

# 配置DPO训练器
config = DPOConfig(beta=0.1, learning_rate=1e-6)
trainer = DPOTrainer(config)

# 准备偏好数据
batch = DPOBatch(
    prompts=["问题"],
    chosen_responses=["好回答"],
    rejected_responses=["差回答"],
)

# DPO训练步
metrics = trainer.train_step(batch)
print(f"Loss: {metrics['loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

---

## 组件详解

### 奖励模型 (reward_model.py)

| 类 | 说明 |
|:---|:-----|
| `RewardModelConfig` | 奖励模型配置 |
| `RewardModel` | 奖励模型基类 |
| `PairwiseRewardModel` | Bradley-Terry成对比较模型 |
| `PreferenceDataset` | 偏好数据集管理 |

**核心公式**:
```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
L_RM = -E[log σ(r(x, y_w) - r(x, y_l))]
```

### RLHF训练器 (rlhf.py)

| 类 | 说明 |
|:---|:-----|
| `RLHFConfig` | RLHF配置 |
| `PPOTrainer` | PPO训练器 |
| `ValueHead` | 价值函数网络 |
| `PPOTrajectory` | 轨迹数据结构 |

**PPO裁剪目标**:
```
L^CLIP = E[min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t)]
其中 r_t = π_θ / π_old
```

### DPO训练器 (dpo.py)

| 类 | 说明 |
|:---|:-----|
| `DPOConfig` | DPO配置 |
| `DPOTrainer` | DPO训练器 |
| `DPOBatch` | DPO批次数据 |
| `DPOLoss` | DPO损失计算 |

**DPO损失**:
```
L_DPO = -E[log σ(β(log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))]
```

---

## 训练指南

### RLHF完整流程

```python
# 1. 监督微调 (SFT)
sft_model = train_sft(instruction_data)

# 2. 奖励建模
reward_model = PairwiseRewardModel()
for epoch in range(num_epochs):
    batch = preference_dataset.sample(batch_size)
    metrics = reward_model.train_step(batch)

# 3. PPO优化
ppo_trainer = PPOTrainer(
    policy=sft_model,
    reward_model=reward_model
)

for step in range(num_steps):
    # 生成回复并评分
    batch = ppo_trainer.generate_and_score(prompts)
    # PPO优化
    metrics = ppo_trainer.train_step(batch)
```

### DPO直接训练

```python
# DPO更简单，无需单独的奖励模型
dpo_trainer = DPOTrainer(
    policy_model=sft_model,
    ref_model=sft_model,  # 参考模型，固定不动
    beta=0.1
)

for epoch in range(num_epochs):
    batch = preference_dataset.sample(batch_size)
    metrics = dpo_trainer.train_step(batch)
```

---

## 最佳实践

### 数据收集

| 标准 | 说明 |
|:-----|:-----|
| **差异性** | chosen明显优于rejected |
| **完整性** | 回答完整，信息充足 |
| **安全性** | 不包含有害内容 |
| **准确性** | 信息正确，无幻觉 |

### 超参数选择

**RLHF/PPO**:
- learning_rate: 1e-5 ~ 5e-5
- clip_epsilon: 0.2
- kl_coef: 0.1 ~ 0.2
- ppo_epochs: 4

**DPO**:
- beta: 0.1 ~ 0.2
- learning_rate: 1e-6 ~ 5e-7
- batch_size: 4 ~ 16

### 评估指标

| 指标 | 说明 |
|:-----|:-----|
| 奖励准确率 | 奖励模型预测偏好的准确率 |
| KL散度 | 策略与参考模型的差异 |
| 胜率 | 对齐模型vs基线模型的人类偏好 |

---

## 运行测试

```bash
cd 07-alignment
python -m pytest tests/ -v
```

**测试覆盖**: 131个测试用例全部通过

---

## 扩展阅读

### 核心论文

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155) - Training language models to follow instructions with human feedback
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization Algorithms
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- [Constitutional AI](https://arxiv.org/abs/2212.08073) - Constitutional AI: Harmlessness from AI Feedback

### 相关资源

- [OpenAI Alignment](https://openai.com/research/alignment/)
- [Anthropic Constitutional AI](https://www.anthropic.com/index/constitutional-ai)
- [HuggingFace TRL](https://huggingface.co/docs/trl) - Transformer Reinforcement Learning library

---

## 许可证

MIT License

---

**最后更新**: 2026-01-02
