"""
奖励模型 (Reward Model)

============================================================
核心思想 (Core Idea)
============================================================
奖励模型是RLHF的核心组件，用于学习人类偏好并为语言模型的输出
打分。通过在人类偏好数据上训练，奖励模型可以预测人类对不同回复
的偏好程度。

============================================================
数学基础 (Mathematical Foundation)
============================================================
Bradley-Terry模型：
    P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

其中：
    - y_w: 人类偏好的回复（winner）
    - y_l: 人类不偏好的回复（loser）
    - r(x, y): 奖励模型对(prompt, response)的打分
    - σ: sigmoid函数

损失函数：
    L = -E[log σ(r(x, y_w) - r(x, y_l))]

============================================================
算法流程 (Algorithm Flow)
============================================================
1. 数据收集: 收集人类偏好对比数据 (x, y_w, y_l)
2. 模型初始化: 从预训练LLM初始化奖励模型
3. 训练: 最小化Bradley-Terry损失
4. 评估: 计算偏好预测准确率

============================================================
参考文献 (References)
============================================================
[1] Ouyang, L., et al. (2022). Training language models to follow
    instructions with human feedback. NeurIPS 2022.
[2] Stiennon, N., et al. (2020). Learning to summarize from human
    feedback. NeurIPS 2020.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


__all__ = [
    "RewardModel",
    "RewardModelConfig",
    "RewardModelOutput",
    "PairwiseRewardModel",
    "PreferenceDataset",
]


@dataclass
class RewardModelConfig:
    """奖励模型配置。

    参数：
        hidden_size: 隐藏层维度
        num_layers: 奖励头层数
        dropout: Dropout概率
        normalize_rewards: 是否归一化奖励
        temperature: 温度参数（用于缩放）
    """
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.1
    normalize_rewards: bool = True
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size必须为正数，得到 {self.hidden_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers必须为正数，得到 {self.num_layers}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout必须在[0,1)范围内，得到 {self.dropout}")
        if self.temperature <= 0:
            raise ValueError(f"temperature必须为正数，得到 {self.temperature}")


@dataclass
class RewardModelOutput:
    """奖励模型输出。

    参数：
        rewards: 奖励分数
        logits: 原始logits（归一化前）
        hidden_states: 隐藏状态（可选）
    """
    rewards: np.ndarray
    logits: np.ndarray
    hidden_states: Optional[np.ndarray] = None

    @property
    def mean_reward(self) -> float:
        """平均奖励。"""
        return float(np.mean(self.rewards))

    @property
    def std_reward(self) -> float:
        """奖励标准差。"""
        return float(np.std(self.rewards))

    def __repr__(self) -> str:
        return f"RewardModelOutput(mean={self.mean_reward:.4f}, std={self.std_reward:.4f})"


@dataclass
class PreferenceExample:
    """偏好数据样本。

    参数：
        prompt: 输入提示
        chosen: 人类偏好的回复
        rejected: 人类不偏好的回复
        chosen_score: 偏好回复的分数（可选）
        rejected_score: 不偏好回复的分数（可选）
    """
    prompt: str
    chosen: str
    rejected: str
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None

    def __repr__(self) -> str:
        return f"PreferenceExample(prompt='{self.prompt[:30]}...')"


class PreferenceDataset:
    """偏好数据集。

    管理人类偏好对比数据。

    示例：
        >>> dataset = PreferenceDataset()
        >>> dataset.add("问题", "好回答", "差回答")
        >>> batch = dataset.sample(batch_size=32)
    """

    def __init__(self, examples: Optional[List[PreferenceExample]] = None) -> None:
        """初始化数据集。

        参数：
            examples: 初始样本列表
        """
        self._examples: List[PreferenceExample] = examples or []

    def add(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        chosen_score: Optional[float] = None,
        rejected_score: Optional[float] = None,
    ) -> None:
        """添加偏好样本。"""
        self._examples.append(PreferenceExample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
        ))

    def sample(self, batch_size: int) -> List[PreferenceExample]:
        """随机采样批次。"""
        if batch_size > len(self._examples):
            batch_size = len(self._examples)
        indices = np.random.choice(len(self._examples), batch_size, replace=False)
        return [self._examples[i] for i in indices]

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> PreferenceExample:
        return self._examples[idx]

    def __iter__(self):
        return iter(self._examples)

    def __repr__(self) -> str:
        return f"PreferenceDataset(size={len(self)})"


class RewardModel(ABC):
    """奖励模型基类。

    所有奖励模型必须继承此类并实现forward方法。

    奖励模型架构：
        LLM Backbone → Hidden States → Reward Head → Scalar Reward

    示例：
        >>> model = MyRewardModel(config)
        >>> output = model.forward(input_ids, attention_mask)
        >>> rewards = output.rewards
    """

    def __init__(self, config: Optional[RewardModelConfig] = None) -> None:
        """初始化奖励模型。

        参数：
            config: 模型配置
        """
        self.config = config or RewardModelConfig()

    @abstractmethod
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> RewardModelOutput:
        """前向传播。

        参数：
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        返回：
            RewardModelOutput对象
        """
        pass

    def compute_reward(self, prompt: str, response: str) -> float:
        """计算单个回复的奖励分数。

        参数：
            prompt: 输入提示
            response: 模型回复

        返回：
            奖励分数
        """
        # 简化实现：基于长度和关键词的启发式奖励
        text = prompt + response
        reward = 0.0

        # 长度奖励（适中长度更好）
        length = len(response)
        if 50 <= length <= 500:
            reward += 0.5
        elif length < 50:
            reward -= 0.3
        else:
            reward -= 0.2

        # 关键词奖励
        positive_keywords = ["谢谢", "帮助", "解决", "明白", "清楚"]
        negative_keywords = ["不知道", "无法", "抱歉", "错误"]

        for kw in positive_keywords:
            if kw in response:
                reward += 0.1

        for kw in negative_keywords:
            if kw in response:
                reward -= 0.1

        return reward

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


class PairwiseRewardModel(RewardModel):
    """成对比较奖励模型。

    基于Bradley-Terry模型学习人类偏好。

    数学原理：
        P(y_w > y_l) = σ((r_w - r_l) / τ)

    其中τ为温度参数，控制偏好的锐度。

    示例：
        >>> model = PairwiseRewardModel(config)
        >>> loss = model.compute_pairwise_loss(chosen_rewards, rejected_rewards)
    """

    def __init__(
        self,
        config: Optional[RewardModelConfig] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        """初始化成对奖励模型。

        参数：
            config: 模型配置
            embed_fn: 文本嵌入函数
        """
        super().__init__(config)
        self._embed_fn = embed_fn or self._default_embed
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._reward_mean: float = 0.0
        self._reward_std: float = 1.0

    def _default_embed(self, text: str) -> np.ndarray:
        """默认嵌入函数（基于字符的简单嵌入）。"""
        # 简单的字符级嵌入
        embed_dim = self.config.hidden_size
        embedding = np.zeros(embed_dim)

        for i, char in enumerate(text[:embed_dim]):
            embedding[i % embed_dim] += ord(char) / 1000.0

        # 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> RewardModelOutput:
        """前向传播计算奖励。"""
        batch_size = input_ids.shape[0]

        # 模拟隐藏状态
        hidden_states = np.random.randn(batch_size, self.config.hidden_size)

        # 计算logits
        if self._weights is None:
            self._weights = np.random.randn(self.config.hidden_size) * 0.01

        logits = np.dot(hidden_states, self._weights) + self._bias

        # 归一化奖励
        if self.config.normalize_rewards:
            rewards = (logits - self._reward_mean) / (self._reward_std + 1e-8)
        else:
            rewards = logits

        return RewardModelOutput(
            rewards=rewards,
            logits=logits,
            hidden_states=hidden_states,
        )

    def compute_pairwise_loss(
        self,
        chosen_rewards: np.ndarray,
        rejected_rewards: np.ndarray,
    ) -> float:
        """计算成对比较损失。

        Bradley-Terry损失：
            L = -log σ((r_w - r_l) / τ)

        参数：
            chosen_rewards: 偏好回复的奖励 [batch_size]
            rejected_rewards: 不偏好回复的奖励 [batch_size]

        返回：
            平均损失
        """
        # 计算奖励差
        reward_diff = (chosen_rewards - rejected_rewards) / self.config.temperature

        # Sigmoid
        probs = 1.0 / (1.0 + np.exp(-reward_diff))

        # 负对数似然
        loss = -np.log(probs + 1e-10)

        return float(np.mean(loss))

    def compute_accuracy(
        self,
        chosen_rewards: np.ndarray,
        rejected_rewards: np.ndarray,
    ) -> float:
        """计算偏好预测准确率。

        参数：
            chosen_rewards: 偏好回复的奖励
            rejected_rewards: 不偏好回复的奖励

        返回：
            准确率 (0-1)
        """
        correct = np.sum(chosen_rewards > rejected_rewards)
        total = len(chosen_rewards)
        return float(correct / total) if total > 0 else 0.0

    def train_step(
        self,
        batch: List[PreferenceExample],
        learning_rate: float = 1e-4,
    ) -> Dict[str, float]:
        """执行一步训练。

        参数：
            batch: 偏好样本批次
            learning_rate: 学习率

        返回：
            训练指标字典
        """
        # 计算嵌入
        chosen_embeds = np.array([
            self._embed_fn(ex.prompt + ex.chosen) for ex in batch
        ])
        rejected_embeds = np.array([
            self._embed_fn(ex.prompt + ex.rejected) for ex in batch
        ])

        # 初始化权重
        if self._weights is None:
            self._weights = np.random.randn(self.config.hidden_size) * 0.01

        # 计算奖励
        chosen_rewards = np.dot(chosen_embeds, self._weights) + self._bias
        rejected_rewards = np.dot(rejected_embeds, self._weights) + self._bias

        # 计算损失
        loss = self.compute_pairwise_loss(chosen_rewards, rejected_rewards)
        accuracy = self.compute_accuracy(chosen_rewards, rejected_rewards)

        # 梯度下降（简化版）
        reward_diff = chosen_rewards - rejected_rewards
        probs = 1.0 / (1.0 + np.exp(-reward_diff / self.config.temperature))

        # 梯度
        grad_scale = (probs - 1.0) / self.config.temperature
        grad_w = np.mean(
            grad_scale[:, np.newaxis] * (chosen_embeds - rejected_embeds),
            axis=0,
        )

        # 更新权重
        self._weights -= learning_rate * grad_w

        return {
            "loss": loss,
            "accuracy": accuracy,
            "mean_chosen_reward": float(np.mean(chosen_rewards)),
            "mean_rejected_reward": float(np.mean(rejected_rewards)),
        }

    def __repr__(self) -> str:
        return f"PairwiseRewardModel(hidden_size={self.config.hidden_size})"
