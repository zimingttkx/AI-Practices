# AI对齐技术知识点

## 1. 概述

### 1.1 什么是对齐 (Alignment)？

AI对齐 (AI Alignment) 是指使人工智能系统的行为与人类意图、价值观和期望保持一致的技术和方法。

**核心问题**:
- 模型可能生成有害、偏见或不当内容
- 预训练模型不擅长遵循用户指令  
- 模型可能产生幻觉（编造虚假信息）
- 模型行为可能与人类价值观不一致

### 1.2 对齐技术演进

```
预训练模型
    ↓
SFT (监督微调) - 学习遵循指令
    ↓
RLHF/DPO - 学习人类偏好
    ↓
Constitutional AI - 学习安全原则
```

### 1.3 为什么需要对齐？

| 问题 | 说明 | 后果 |
|------|------|------|
| **有害输出** | 生成仇恨言论、暴力内容 | 用户伤害、品牌风险 |
| **偏见强化** | 学习并放大数据偏见 | 歧视性输出 |
| **指令不遵循** | 无法完成用户指定任务 | 用户体验差 |
| **幻觉** | 编造不存在的"事实" | 信息可信度下降 |
| **价值观偏离** | 与人类伦理不一致 | 社会风险 |

---

## 2. 监督微调 (SFT)

### 2.1 SFT原理

监督微调 (Supervised Fine-Tuning) 是对齐的第一步，使用高质量的指令-回复对来训练模型。

**目标函数**:
```
L_SFT = -Σ log π_θ(y_i | x_i)
```

### 2.2 SFT数据格式

```python
{
    "instruction": "解释什么是机器学习",
    "input": "",  # 可选的额外输入
    "output": "机器学习是..."
}
```

### 2.3 SFT最佳实践

1. **数据质量 > 数量**: 少量高质量数据优于大量低质量数据
2. **指令多样性**: 覆盖不同的任务类型和领域
3. **回答长度**: 避免过短或过长
4. **避免冗余**: 去除高度相似的样本

---

## 3. RLHF (人类反馈强化学习)

### 3.1 RLHF三阶段流程

```
┌─────────────────────────────────────────────────────────────┐
│                     RLHF 完整流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段1: 监督微调 (SFT)                                     │
│  ┌─────────────┐     ┌─────────────┐                       │
│  │ 预训练模型   │  →  │ 高质量指令-对│                       │
│  └─────────────┘     │ 数据集        │                       │
│                      └─────────────┘                       │
│                             ↓                                │
│                      ┌─────────────┐                       │
│                      │  SFT模型     │                       │
│                      └─────────────┘                       │
│                                                             │
│  阶段2: 奖励建模 (RM)                                      │
│  ┌─────────────┐     ┌─────────────┐                       │
│  │ 人类偏好数据 │  →  │ Bradley-Terry│                       │
│  │ (x, y_w, y_l)│     │   奖励模型    │                       │
│  └─────────────┘     └─────────────┘                       │
│                      r(x,y) → P(y_w > y_l | x)                   │
│                                                             │
│  阶段3: 强化学习优化                                        │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│  │ SFT模型     │  →  │   奖励模型   │  →  │  PPO优化    │  │
│  └─────────────┘     └─────────────┘     └─────────────┘  │
│                        ↑                              ↑        │
│                        └────── 生成、评分 ──────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 奖励模型 (Reward Model)

#### Bradley-Terry模型

**核心假设**: 如果 y_w 的奖励高于 y_l，则人类更偏好 y_w。

**偏好概率**:
$$P(y_w > y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是sigmoid函数。

**损失函数**:
$$L_{RM} = -\mathbb{E}[\log \sigma(r(x, y_w) - r(x, y_l))]$$

#### 奖励模型架构

```python
class PairwiseRewardModel(nn.Module):
    """成对比较奖励模型"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.encoder = Encoder()          # 文本编码器
        self scorer = nn.Linear(hidden_size, 1)  # 奖励打分头
        
    def forward(self, prompt: str, response: str) -> float:
        """计算 (prompt, response) 的奖励分数"""
        x = self.encode(prompt + response)
        return self.scaler(x)
```

#### KL散度正则化

防止奖励模型过拟合，保持与初始模型的接近：

$$L_{RM}^{reg} = L_{RM} + \alpha \cdot KL(\pi_{RM} || \pi_{init})$$

### 3.3 PPO算法详解

#### PPO裁剪目标

PPO通过裁剪概率比来限制策略更新幅度：

$$L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$: 概率比
- $\hat{A}_t$: 优势函数估计
- $\epsilon$: 裁剪参数（通常0.2）

#### 直观解释

```
正优势时 (A_t > 0):
  限制策略增长，防止过度优化
  
负优势时 (A_t < 0):
  限制策略下降，保持探索
```

#### 完整PPO目标

$$L^{PPO} = \mathbb{E}_t[L^{CLIP}(\theta) - c_1 L_{VF}(s) + c_2 L_{entropy}(\pi_\theta) - c_3 \cdot KL(\pi_\theta || \pi_{ref})]$$

| 项 | 作用 | 系数 |
|:---|:-----|:-----|
| $L^{CLIP}$ | 裁剪策略目标 | - |
| $L_{VF}$ | 价值函数损失 | $c_1$ |
| $L_{entropy}$ | 熵正则化 | $c_2$ |
| $KL$ | KL散度惩罚 | $c_3$ |

### 3.4 GAE优势估计

#### GAE公式

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

其中 $ \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是TD误差。

#### Lambda的影响

| $\lambda$ | 方差 | 偏差 | 特点 |
|:----------|:-----|:-----|:-----|
| 0 | 高 | 低 | 纯TD误差，单步 |
| 0.95 | 中 | 中 | **推荐值**，平衡方差和偏差 |
| 1.0 | 低 | 高 | 蒙特卡洛估计 |

---

## 4. DPO (直接偏好优化)

### 4.1 DPO核心思想

DPO将RLHF的奖励建模和RL步骤合并为单一监督学习目标。

**关键洞察**: 最优策略的奖励函数满足

$$r^*(x, y) = r_0(x, y) + \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

这意味着奖励函数与策略之间存在直接映射关系！

### 4.2 DPO数学推导

#### Step 1: RL目标

$$\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi}[r(x, y)] - \beta \cdot KL(\pi || \pi_{ref})$$

#### Step 2: 最优策略

$$\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(\frac{1}{\beta} r(x, y))$$

#### Step 3: 重排得到奖励函数

$$\log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} = \frac{1}{\beta}r(x, y) - \log Z(x)$$

$$r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

#### Step 4: 代入Bradley-Terry模型

$$L_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left(\log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)\right]$$

### 4.3 DPO损失变体

| 变体 | 公式 | 特点 |
|:-----|:-----|:-----|
| **Sigmoid** | $-\log \sigma(\beta \cdot \Delta)$ | 标准DPO，平滑 |
| **Hinge** | $\max(0, 1 - \beta \cdot \Delta)$ | 鲁棒，边界清晰 |
| **IPO** | $(\beta \cdot \Delta - 1)^2$ | 防止过拟合 |

其中 $\Delta = \log \frac{\pi(y_w)}{\pi_{ref}(y_w)} - \log \frac{\pi(y_l)}{\pi_{ref}(y_l)}$

### 4.4 DPO vs RLHF 对比

| 维度 | RLHF/PPO | DPO |
|:-----|:----------|:-----|
| **训练阶段** | 3阶段 | 1阶段 |
| **奖励模型** | 需要 | 不需要 |
| **RL训练** | PPO循环 | 不需要 |
| **稳定性** | 较低，易发散 | 较高 |
| **超参数** | 多 (lr, clip, KL, γ, λ...) | 少 (β, lr) |
| **计算成本** | 高 (多次生成评分) | 低 |
| **实现复杂度** | 高 | 低 |
| **训练时间** | 长 | 短 |
| **适用场景** | 复杂任务 | 大多数场景 |

---

## 5. Constitutional AI

### 5.1 核心思想

使用AI反馈 (RLAIF: Reinforcement Learning from AI Feedback) 代替人类反馈。

### 5.2 CAI流程

```
生成 → 批评 → 修订 → 训练
  ↓      ↓      ↓      ↓
  回复  AI批评  AI修订  隐式奖励
```

### 5.3 宪法原则示例

```
宪法原则示例：
- 回复应该是有帮助的、无害的、诚实的
- 避免生成有害、非法或不道德的内容
- 承认不确定性，不编造信息
- 尊重所有用户，不歧视任何人
- 保护用户隐私，不索要敏感信息
```

### 5.4 RLAIF vs RLHF

| 维度 | RLHF | RLAIF |
|:-----|:-----|:-------|
| 反馈来源 | 人类 | AI |
| 成本 | 高 | 低 |
| 可扩展性 | 低 | 高 |
| 偏见风险 | 人类偏见 | 模型偏见 |

---

## 6. 数据收集与处理

### 6.1 偏好数据格式

```python
{
    "prompt": "用户问题",
    "chosen": "人类偏好的回复",
    "rejected": "人类不偏好的回复"
}
```

### 6.2 数据质量标准

| 标准 | 说明 |
|:-----|:-----|
| **差异性** | chosen明显优于rejected |
| **完整性** | 回答完整，信息充足 |
| **安全性** | 不包含有害内容 |
| **准确性** | 信息正确，无幻觉 |
| **多样性** | 覆盖不同领域和任务 |

### 6.3 数据增强技巧

1. **反向采样**: 交换chosen和rejected角色
2. **温度采样**: 生成多样化的回复
3. **难度分层**: 简单/中等/困难样本
4. **对抗生成**: 故意生成差回答

---

## 7. 超参数调优

### 7.1 RLHF/PPO超参数

| 参数 | 推荐值 | 说明 |
|:-----|:-------|:-----|
| learning_rate | 1e-5 ~ 5e-5 | 较小的学习率保证稳定 |
| ppo_epochs | 2-4 | 每批数据重复使用次数 |
| clip_epsilon | 0.2 | 裁剪范围 |
| kl_coef | 0.1 ~ 0.2 | KL散度系数 |
| kl_target | 0.01 ~ 0.02 | 目标KL散度 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE参数 |

### 7.2 DPO超参数

| 参数 | 推荐值 | 说明 |
|:-----|:-------|:-----|
| beta | 0.1 ~ 0.2 | DPO温度参数 |
| learning_rate | 1e-6 ~ 5e-7 | 较小的学习率 |
| batch_size | 4-16 | 根据GPU内存调整 |
| loss_type | sigmoid | sigmoid/hinge/ipo |

### 7.3 Beta调优策略

```
Beta太小 (< 0.01) → 弱正则化，可能过拟合
Beta适中 (0.1-0.2) → 推荐值，平衡性能和稳定性
Beta太大 (> 0.5) → 强正则化，可能欠拟合
```

---

## 8. 评估指标

### 8.1 奖励模型指标

| 指标 | 公式 | 说明 |
|:-----|:-----|:-----|
| **准确率** | 正确预测数 / 总数 | 奖励模型预测偏好 |
| **AUC-ROC** | 曲线下面积 | 排序质量 |
| **Margin** | r(y_w) - r(y_l) | 奖励差距 |

### 8.2 RLHF/DPO指标

| 指标 | 说明 |
|:-----|:-----|
| **KL散度** | 与参考模型的差异 |
| **平均奖励** | 生成回复的平均奖励 |
| **胜率** | 对齐模型 vs 基线模型的人类偏好 |
| **困惑度** | 语言模型质量指标 |

### 8.3 人工评估

```python
评估维度：
1. 有用性 (Helpfulness) - 回复是否解决了问题
2. 无害性 (Harmlessness) - 是否包含有害内容
3. 诚实性 (Honesty) - 是否避免幻觉
4. 遵循度 (Instruction Following) - 是否按指令执行
```

---

## 9. 常见问题与解决方案

### 9.1 RLHF训练不稳定

**症状**: 损失爆炸、KL散度过大、奖励崩溃

**解决方案**:
1. 降低学习率
2. 增大KL系数
3. 使用更小的clip_epsilon
4. 减少PPO轮数

### 9.2 DPO过拟合

**症状**: 验证集性能下降

**解决方案**:
1. 增大beta参数
2. 使用IPO损失代替Sigmoid
3. 增加数据量
4. 添加正则化

### 9.3 奖励模型不准确

**症状**: 奖励模型准确率低

**解决方案**:
1. 检查数据质量
2. 增加数据量
3. 平衡正负样本
4. 调整模型架构

---

## 10. 扩展阅读

### 核心论文

1. **InstructGPT** (Ouyang et al., 2022) - Training language models to follow instructions with human feedback
2. **PPO** (Schulman et al., 2017) - Proximal Policy Optimization Algorithms
3. **DPO** (Rafailov et al., 2023) - Direct Preference Optimization: Your Language Model is Secretly a Reward Model
4. **Constitutional AI** (Bai et al., 2022) - Constitutional AI: Harmlessness from AI Feedback
5. **GAE** (Schulman et al., 2016) - High-Dimensional Continuous Control Using Generalized Advantage Estimation

### 相关资源

- [OpenAI PPO](https://openai.com/research/openai-baselines-ppo/)
- [Anthropic CAI](https://www.anthropic.com/index/constitutional-ai)
- [HuggingFace TRL](https://huggingface.co/docs/trl)
- [Alignment Forum](https://www.alignmentforum.org/)
