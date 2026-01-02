"""
RLHF训练模块 (RLHF Training Module)

============================================================
核心思想 (Core Idea)
============================================================
人类反馈强化学习(RLHF)通过三个阶段训练语言模型：
1. 监督微调(SFT): 在高质量数据上微调预训练模型
2. 奖励建模(RM): 训练奖励模型学习人类偏好
3. 强化学习(RL): 使用PPO优化策略模型

============================================================
数学基础 (Mathematical Foundation)
============================================================
PPO目标函数：
    L^{CLIP}(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

其中：
    - r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t): 概率比
    - A_t: 优势函数估计
    - ε: 裁剪参数

KL惩罚：
    L = L^{CLIP} - β * KL(π_θ || π_ref)

============================================================
算法流程 (Algorithm Flow)
============================================================
1. 采样: 从策略模型生成回复
2. 评分: 使用奖励模型计算奖励
3. 优势估计: 计算GAE优势
4. 策略更新: PPO梯度更新
5. KL控制: 动态调整KL惩罚系数

============================================================
参考文献 (References)
============================================================
[1] Schulman, J., et al. (2017). Proximal Policy Optimization
    Algorithms. arXiv:1707.06347.
[2] Ouyang, L., et al. (2022). Training language models to follow
    instructions with human feedback. NeurIPS 2022.
[3] Ziegler, D., et al. (2019). Fine-Tuning Language Models from
    Human Preferences. arXiv:1909.08593.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


__all__ = [
    "RLHFConfig",
    "RLHFTrainer",
    "PPOTrainer",
    "ValueHead",
    "compute_advantages",
    "compute_rewards",
]


@dataclass
class RLHFConfig:
    """RLHF训练配置。

    参数：
        learning_rate: 学习率
        batch_size: 批次大小
        ppo_epochs: PPO更新轮数
        clip_epsilon: PPO裁剪参数
        gamma: 折扣因子
        gae_lambda: GAE参数
        kl_coef: KL惩罚系数
        kl_target: 目标KL散度
        value_coef: 价值损失系数
        entropy_coef: 熵正则化系数
        max_grad_norm: 梯度裁剪阈值
    """
    learning_rate: float = 1e-5
    batch_size: int = 64
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    kl_coef: float = 0.1
    kl_target: float = 6.0
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate必须为正数，得到 {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size必须为正数，得到 {self.batch_size}")
        if self.ppo_epochs <= 0:
            raise ValueError(f"ppo_epochs必须为正数，得到 {self.ppo_epochs}")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError(f"clip_epsilon必须在(0,1)范围内，得到 {self.clip_epsilon}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma必须在[0,1]范围内，得到 {self.gamma}")


@dataclass
class RLHFBatch:
    """RLHF训练批次。

    参数：
        prompts: 输入提示列表
        responses: 生成的回复列表
        rewards: 奖励分数
        old_logprobs: 旧策略的对数概率
        values: 价值估计
        advantages: 优势估计
        returns: 回报
    """
    prompts: List[str]
    responses: List[str]
    rewards: np.ndarray
    old_logprobs: np.ndarray
    values: np.ndarray
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None


class ValueHead:
    """价值头网络。

    将语言模型的隐藏状态映射到标量价值估计。

    架构：
        Hidden States → Linear → ReLU → Linear → Value

    示例：
        >>> value_head = ValueHead(hidden_size=768)
        >>> values = value_head.forward(hidden_states)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ) -> None:
        """初始化价值头。

        参数：
            hidden_size: 隐藏层维度
            dropout: Dropout概率
        """
        self.hidden_size = hidden_size
        self.dropout = dropout

        # 初始化权重
        self._w1 = np.random.randn(hidden_size, hidden_size) * 0.02
        self._b1 = np.zeros(hidden_size)
        self._w2 = np.random.randn(hidden_size, 1) * 0.02
        self._b2 = np.zeros(1)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """前向传播。

        参数：
            hidden_states: 隐藏状态 [batch_size, hidden_size]

        返回：
            价值估计 [batch_size]
        """
        # 第一层
        h = np.dot(hidden_states, self._w1) + self._b1
        h = np.maximum(h, 0)  # ReLU

        # Dropout（训练时）
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, h.shape)
            h = h * mask / (1 - self.dropout)

        # 第二层
        values = np.dot(h, self._w2) + self._b2

        return values.squeeze(-1)

    def __repr__(self) -> str:
        return f"ValueHead(hidden_size={self.hidden_size})"


def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算GAE优势估计。

    广义优势估计(GAE)：
        A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        δ_t = r_t + γV(s_{t+1}) - V(s_t)

    参数：
        rewards: 奖励序列 [batch_size]
        values: 价值估计 [batch_size]
        gamma: 折扣因子
        gae_lambda: GAE参数

    返回：
        (advantages, returns) 元组
    """
    batch_size = len(rewards)

    # 对于单步奖励（RLHF场景），简化计算
    # δ = r - V
    deltas = rewards - values

    # GAE（单步情况下等于δ）
    advantages = deltas.copy()

    # 回报 = 优势 + 价值
    returns = advantages + values

    # 归一化优势
    adv_mean = np.mean(advantages)
    adv_std = np.std(advantages) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    return advantages, returns


def compute_rewards(
    responses: List[str],
    reward_fn: Callable[[str], float],
    kl_penalties: Optional[np.ndarray] = None,
    kl_coef: float = 0.1,
) -> np.ndarray:
    """计算带KL惩罚的奖励。

    总奖励：
        R = R_model - β * KL(π || π_ref)

    参数：
        responses: 生成的回复列表
        reward_fn: 奖励函数
        kl_penalties: KL散度惩罚
        kl_coef: KL惩罚系数

    返回：
        奖励数组
    """
    # 计算基础奖励
    base_rewards = np.array([reward_fn(r) for r in responses])

    # 添加KL惩罚
    if kl_penalties is not None:
        rewards = base_rewards - kl_coef * kl_penalties
    else:
        rewards = base_rewards

    return rewards


class RLHFTrainer(ABC):
    """RLHF训练器基类。

    所有RLHF训练器必须继承此类。

    训练流程：
        1. 生成回复
        2. 计算奖励
        3. 估计优势
        4. 更新策略

    示例：
        >>> trainer = MyRLHFTrainer(config)
        >>> metrics = trainer.train_step(batch)
    """

    def __init__(self, config: Optional[RLHFConfig] = None) -> None:
        """初始化训练器。

        参数：
            config: 训练配置
        """
        self.config = config or RLHFConfig()
        self._step = 0
        self._kl_coef = self.config.kl_coef

    @abstractmethod
    def train_step(self, batch: RLHFBatch) -> Dict[str, float]:
        """执行一步训练。

        参数：
            batch: 训练批次

        返回：
            训练指标字典
        """
        pass

    def update_kl_coef(self, kl: float) -> None:
        """动态调整KL惩罚系数。

        如果KL > target，增加系数；否则减少。

        参数：
            kl: 当前KL散度
        """
        if kl > self.config.kl_target * 1.5:
            self._kl_coef *= 1.5
        elif kl < self.config.kl_target / 1.5:
            self._kl_coef /= 1.5

    @property
    def kl_coef(self) -> float:
        """当前KL惩罚系数。"""
        return self._kl_coef

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(step={self._step})"


class PPOTrainer(RLHFTrainer):
    """PPO训练器。

    实现Proximal Policy Optimization算法用于RLHF训练。

    PPO核心思想：
        通过裁剪概率比来限制策略更新幅度，保证训练稳定性。

    示例：
        >>> trainer = PPOTrainer(config)
        >>> for batch in dataloader:
        ...     metrics = trainer.train_step(batch)
    """

    def __init__(
        self,
        config: Optional[RLHFConfig] = None,
        value_head: Optional[ValueHead] = None,
        reward_fn: Optional[Callable[[str], float]] = None,
    ) -> None:
        """初始化PPO训练器。

        参数：
            config: 训练配置
            value_head: 价值头网络
            reward_fn: 奖励函数
        """
        super().__init__(config)
        self.value_head = value_head or ValueHead()
        self._reward_fn = reward_fn or self._default_reward_fn
        
        # 训练统计
        self._total_loss = 0.0
        self._policy_loss = 0.0
        self._value_loss = 0.0
        self._entropy = 0.0

    def _default_reward_fn(self, response: str) -> float:
        """默认奖励函数。"""
        reward = 0.0
        length = len(response)
        if 50 <= length <= 500:
            reward += 0.5
        return reward

    def train_step(self, batch: RLHFBatch) -> Dict[str, float]:
        """执行一步PPO训练。

        PPO更新步骤：
            1. 计算优势估计
            2. 多轮PPO更新
            3. 更新KL系数

        参数：
            batch: 训练批次

        返回：
            训练指标字典
        """
        self._step += 1

        # 计算优势和回报
        if batch.advantages is None:
            advantages, returns = compute_advantages(
                batch.rewards,
                batch.values,
                self.config.gamma,
                self.config.gae_lambda,
            )
            batch.advantages = advantages
            batch.returns = returns

        # PPO多轮更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.config.ppo_epochs):
            policy_loss, value_loss, entropy = self._ppo_update(batch)
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy

        # 平均损失
        avg_policy_loss = total_policy_loss / self.config.ppo_epochs
        avg_value_loss = total_value_loss / self.config.ppo_epochs
        avg_entropy = total_entropy / self.config.ppo_epochs

        # 计算KL散度并更新系数
        kl = self._estimate_kl(batch)
        self.update_kl_coef(kl)

        return {
            "step": self._step,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "total_loss": avg_policy_loss + self.config.value_coef * avg_value_loss,
            "kl": kl,
            "kl_coef": self._kl_coef,
            "mean_reward": float(np.mean(batch.rewards)),
            "mean_advantage": float(np.mean(batch.advantages)),
        }

    def _ppo_update(self, batch: RLHFBatch) -> Tuple[float, float, float]:
        """执行单轮PPO更新。

        参数：
            batch: 训练批次

        返回：
            (policy_loss, value_loss, entropy) 元组
        """
        batch_size = len(batch.prompts)

        # 模拟新策略的对数概率
        new_logprobs = batch.old_logprobs + np.random.randn(batch_size) * 0.1

        # 计算概率比
        ratio = np.exp(new_logprobs - batch.old_logprobs)

        # PPO裁剪目标
        surr1 = ratio * batch.advantages
        surr2 = np.clip(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon,
        ) * batch.advantages

        # 策略损失（取最小值）
        policy_loss = -np.mean(np.minimum(surr1, surr2))

        # 价值损失
        hidden_states = np.random.randn(batch_size, self.value_head.hidden_size)
        new_values = self.value_head.forward(hidden_states)
        value_loss = np.mean((new_values - batch.returns) ** 2)

        # 熵正则化（鼓励探索）
        entropy = -np.mean(new_logprobs * np.exp(new_logprobs))

        return float(policy_loss), float(value_loss), float(entropy)

    def _estimate_kl(self, batch: RLHFBatch) -> float:
        """估计KL散度。

        参数：
            batch: 训练批次

        返回：
            KL散度估计
        """
        # 简化的KL估计
        new_logprobs = batch.old_logprobs + np.random.randn(len(batch.prompts)) * 0.05
        kl = np.mean(batch.old_logprobs - new_logprobs)
        return float(np.abs(kl))

    def generate_and_score(
        self,
        prompts: List[str],
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> RLHFBatch:
        """生成回复并计算奖励。

        参数：
            prompts: 输入提示列表
            generate_fn: 生成函数

        返回：
            RLHFBatch对象
        """
        if generate_fn is None:
            generate_fn = lambda p: f"这是对'{p[:20]}...'的回复。"

        # 生成回复
        responses = [generate_fn(p) for p in prompts]

        # 计算奖励
        rewards = np.array([self._reward_fn(r) for r in responses])

        # 模拟对数概率和价值
        batch_size = len(prompts)
        old_logprobs = np.random.randn(batch_size) * 0.5 - 2.0
        hidden_states = np.random.randn(batch_size, self.value_head.hidden_size)
        values = self.value_head.forward(hidden_states)

        return RLHFBatch(
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            old_logprobs=old_logprobs,
            values=values,
        )

    def __repr__(self) -> str:
        return f"PPOTrainer(step={self._step}, kl_coef={self._kl_coef:.4f})"
