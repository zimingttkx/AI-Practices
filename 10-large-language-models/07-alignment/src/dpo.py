"""
DPO训练模块 (Direct Preference Optimization)

============================================================
核心思想 (Core Idea)
============================================================
直接偏好优化(DPO)是一种无需显式奖励模型的对齐方法。DPO将RLHF的
奖励建模和强化学习步骤合并为单一的监督学习目标，直接从偏好数据
优化策略模型。

============================================================
数学基础 (Mathematical Foundation)
============================================================
DPO损失函数：
    L_DPO(π_θ; π_ref) = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) 
                                   - log π_θ(y_l|x)/π_ref(y_l|x)))]

简化形式：
    L = -E[log σ(β(r_θ(x, y_w) - r_θ(x, y_l)))]

其中隐式奖励：
    r_θ(x, y) = β log(π_θ(y|x) / π_ref(y|x))

============================================================
算法流程 (Algorithm Flow)
============================================================
1. 准备偏好数据: (x, y_w, y_l) 三元组
2. 计算参考模型对数概率: log π_ref(y|x)
3. 计算当前模型对数概率: log π_θ(y|x)
4. 计算DPO损失并反向传播
5. 更新策略模型参数

============================================================
参考文献 (References)
============================================================
[1] Rafailov, R., et al. (2023). Direct Preference Optimization:
    Your Language Model is Secretly a Reward Model. NeurIPS 2023.
[2] Tunstall, L., et al. (2023). Zephyr: Direct Distillation of
    LM Alignment. arXiv:2310.16944.
[3] Ethayarajh, K., et al. (2024). KTO: Model Alignment as Prospect
    Theoretic Optimization. arXiv:2402.01306.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


__all__ = [
    "DPOConfig",
    "DPOTrainer",
    "DPOLoss",
    "PreferenceDataCollator",
    "compute_dpo_loss",
    "compute_reference_logprobs",
]


@dataclass
class DPOConfig:
    """DPO训练配置。

    参数：
        beta: 温度参数（控制偏好锐度）
        learning_rate: 学习率
        batch_size: 批次大小
        max_length: 最大序列长度
        label_smoothing: 标签平滑系数
        loss_type: 损失类型 ('sigmoid', 'hinge', 'ipo')
        reference_free: 是否使用无参考模式
    """
    beta: float = 0.1
    learning_rate: float = 1e-6
    batch_size: int = 4
    max_length: int = 512
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_free: bool = False

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"beta必须为正数，得到 {self.beta}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate必须为正数，得到 {self.learning_rate}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size必须为正数，得到 {self.batch_size}")
        if not 0 <= self.label_smoothing < 0.5:
            raise ValueError(f"label_smoothing必须在[0,0.5)范围内")
        if self.loss_type not in ("sigmoid", "hinge", "ipo"):
            raise ValueError(f"loss_type必须是sigmoid/hinge/ipo之一")


@dataclass
class DPOBatch:
    """DPO训练批次。

    参数：
        prompts: 输入提示
        chosen_responses: 偏好回复
        rejected_responses: 不偏好回复
        chosen_logprobs: 偏好回复的对数概率
        rejected_logprobs: 不偏好回复的对数概率
        ref_chosen_logprobs: 参考模型的偏好回复对数概率
        ref_rejected_logprobs: 参考模型的不偏好回复对数概率
    """
    prompts: List[str]
    chosen_responses: List[str]
    rejected_responses: List[str]
    chosen_logprobs: Optional[np.ndarray] = None
    rejected_logprobs: Optional[np.ndarray] = None
    ref_chosen_logprobs: Optional[np.ndarray] = None
    ref_rejected_logprobs: Optional[np.ndarray] = None


class PreferenceDataCollator:
    """偏好数据整理器。

    将原始偏好数据转换为模型输入格式。

    示例：
        >>> collator = PreferenceDataCollator(tokenizer)
        >>> batch = collator(examples)
    """

    def __init__(
        self,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> None:
        """初始化整理器。

        参数：
            max_length: 最大序列长度
            padding: 是否填充
            truncation: 是否截断
        """
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __call__(
        self,
        examples: List[Dict[str, str]],
    ) -> DPOBatch:
        """整理批次数据。

        参数：
            examples: 样本列表，每个样本包含prompt/chosen/rejected

        返回：
            DPOBatch对象
        """
        prompts = [ex["prompt"] for ex in examples]
        chosen = [ex["chosen"] for ex in examples]
        rejected = [ex["rejected"] for ex in examples]

        return DPOBatch(
            prompts=prompts,
            chosen_responses=chosen,
            rejected_responses=rejected,
        )


class DPOLoss:
    """DPO损失计算器。

    支持多种损失变体：
        - sigmoid: 标准DPO损失
        - hinge: 铰链损失变体
        - ipo: Identity Preference Optimization

    示例：
        >>> loss_fn = DPOLoss(beta=0.1, loss_type="sigmoid")
        >>> loss = loss_fn(chosen_logps, rejected_logps, ref_chosen, ref_rejected)
    """

    def __init__(
        self,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
    ) -> None:
        """初始化损失计算器。

        参数：
            beta: 温度参数
            loss_type: 损失类型
            label_smoothing: 标签平滑
        """
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        reference_chosen_logps: np.ndarray,
        reference_rejected_logps: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """计算DPO损失。

        参数：
            policy_chosen_logps: 策略模型的偏好回复对数概率
            policy_rejected_logps: 策略模型的不偏好回复对数概率
            reference_chosen_logps: 参考模型的偏好回复对数概率
            reference_rejected_logps: 参考模型的不偏好回复对数概率

        返回：
            (loss, metrics) 元组
        """
        # 计算对数概率比
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # 计算隐式奖励差
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # 根据损失类型计算
        if self.loss_type == "sigmoid":
            loss = self._sigmoid_loss(logits)
        elif self.loss_type == "hinge":
            loss = self._hinge_loss(logits)
        elif self.loss_type == "ipo":
            loss = self._ipo_loss(logits)
        else:
            loss = self._sigmoid_loss(logits)

        # 计算指标
        metrics = {
            "loss": float(loss),
            "chosen_rewards": float(np.mean(self.beta * chosen_logratios)),
            "rejected_rewards": float(np.mean(self.beta * rejected_logratios)),
            "reward_margin": float(np.mean(self.beta * (chosen_logratios - rejected_logratios))),
            "accuracy": float(np.mean(logits > 0)),
        }

        return loss, metrics

    def _sigmoid_loss(self, logits: np.ndarray) -> float:
        """标准DPO Sigmoid损失。"""
        # L = -log(σ(logits))
        # 使用数值稳定的实现
        losses = -np.log(1.0 / (1.0 + np.exp(-logits)) + 1e-10)

        # 标签平滑
        if self.label_smoothing > 0:
            smooth_loss = -np.log(1.0 / (1.0 + np.exp(logits)) + 1e-10)
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * smooth_loss

        return float(np.mean(losses))

    def _hinge_loss(self, logits: np.ndarray) -> float:
        """铰链损失变体。"""
        # L = max(0, 1 - logits)
        losses = np.maximum(0, 1 - logits)
        return float(np.mean(losses))

    def _ipo_loss(self, logits: np.ndarray) -> float:
        """IPO损失（Identity Preference Optimization）。"""
        # L = (logits - 1)^2
        losses = (logits - 1) ** 2
        return float(np.mean(losses))


def compute_dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    reference_chosen_logps: np.ndarray,
    reference_rejected_logps: np.ndarray,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """计算DPO损失（函数式接口）。

    DPO损失：
        L = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

    参数：
        policy_chosen_logps: 策略模型偏好回复对数概率
        policy_rejected_logps: 策略模型不偏好回复对数概率
        reference_chosen_logps: 参考模型偏好回复对数概率
        reference_rejected_logps: 参考模型不偏好回复对数概率
        beta: 温度参数
        label_smoothing: 标签平滑

    返回：
        (loss, metrics) 元组
    """
    loss_fn = DPOLoss(beta=beta, label_smoothing=label_smoothing)
    return loss_fn(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )


def compute_reference_logprobs(
    texts: List[str],
    log_prob_fn: Optional[Callable[[str], float]] = None,
) -> np.ndarray:
    """计算参考模型的对数概率。

    参数：
        texts: 文本列表
        log_prob_fn: 对数概率计算函数

    返回：
        对数概率数组
    """
    if log_prob_fn is None:
        # 默认：基于长度的简单估计
        def log_prob_fn(text: str) -> float:
            # 假设每个token的平均对数概率约为-2
            num_tokens = len(text.split())
            return -2.0 * num_tokens

    return np.array([log_prob_fn(text) for text in texts])


class DPOTrainer:
    """DPO训练器。

    实现Direct Preference Optimization算法。

    DPO优势：
        - 无需训练单独的奖励模型
        - 无需复杂的RL训练循环
        - 训练更稳定，超参数更少

    示例：
        >>> trainer = DPOTrainer(config)
        >>> for batch in dataloader:
        ...     metrics = trainer.train_step(batch)
    """

    def __init__(
        self,
        config: Optional[DPOConfig] = None,
        log_prob_fn: Optional[Callable[[str], float]] = None,
    ) -> None:
        """初始化DPO训练器。

        参数：
            config: 训练配置
            log_prob_fn: 对数概率计算函数
        """
        self.config = config or DPOConfig()
        self._log_prob_fn = log_prob_fn or self._default_log_prob_fn
        self._loss_fn = DPOLoss(
            beta=self.config.beta,
            loss_type=self.config.loss_type,
            label_smoothing=self.config.label_smoothing,
        )
        self._step = 0
        
        # 模拟模型参数
        self._policy_params = np.random.randn(100) * 0.01
        self._ref_params = self._policy_params.copy()

    def _default_log_prob_fn(self, text: str) -> float:
        """默认对数概率函数。"""
        num_tokens = len(text.split())
        return -2.0 * num_tokens + np.random.randn() * 0.1

    def train_step(self, batch: DPOBatch) -> Dict[str, float]:
        """执行一步DPO训练。

        参数：
            batch: 训练批次

        返回：
            训练指标字典
        """
        self._step += 1

        # 计算对数概率
        if batch.chosen_logprobs is None:
            batch.chosen_logprobs = np.array([
                self._compute_policy_logprob(p + c)
                for p, c in zip(batch.prompts, batch.chosen_responses)
            ])
        if batch.rejected_logprobs is None:
            batch.rejected_logprobs = np.array([
                self._compute_policy_logprob(p + r)
                for p, r in zip(batch.prompts, batch.rejected_responses)
            ])
        if batch.ref_chosen_logprobs is None:
            batch.ref_chosen_logprobs = np.array([
                self._compute_ref_logprob(p + c)
                for p, c in zip(batch.prompts, batch.chosen_responses)
            ])
        if batch.ref_rejected_logprobs is None:
            batch.ref_rejected_logprobs = np.array([
                self._compute_ref_logprob(p + r)
                for p, r in zip(batch.prompts, batch.rejected_responses)
            ])

        # 计算损失
        loss, metrics = self._loss_fn(
            batch.chosen_logprobs,
            batch.rejected_logprobs,
            batch.ref_chosen_logprobs,
            batch.ref_rejected_logprobs,
        )

        # 模拟梯度更新
        self._update_policy(loss)

        metrics["step"] = self._step
        return metrics

    def _compute_policy_logprob(self, text: str) -> float:
        """计算策略模型的对数概率。"""
        base_logprob = self._default_log_prob_fn(text)
        param_effect = np.sum(self._policy_params[:10]) * 0.01
        return base_logprob + param_effect

    def _compute_ref_logprob(self, text: str) -> float:
        """计算参考模型的对数概率。"""
        base_logprob = self._default_log_prob_fn(text)
        param_effect = np.sum(self._ref_params[:10]) * 0.01
        return base_logprob + param_effect

    def _update_policy(self, loss: float) -> None:
        """更新策略参数。"""
        # 简化的梯度更新
        grad = np.random.randn(100) * loss * 0.01
        self._policy_params -= self.config.learning_rate * grad

    def compute_implicit_rewards(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> np.ndarray:
        """计算隐式奖励。

        DPO的隐式奖励：
            r(x, y) = β log(π_θ(y|x) / π_ref(y|x))

        参数：
            prompts: 提示列表
            responses: 回复列表

        返回：
            隐式奖励数组
        """
        policy_logprobs = np.array([
            self._compute_policy_logprob(p + r)
            for p, r in zip(prompts, responses)
        ])
        ref_logprobs = np.array([
            self._compute_ref_logprob(p + r)
            for p, r in zip(prompts, responses)
        ])

        return self.config.beta * (policy_logprobs - ref_logprobs)

    def evaluate(self, dataset: List[Dict[str, str]]) -> Dict[str, float]:
        """评估模型性能。

        参数：
            dataset: 评估数据集

        返回：
            评估指标字典
        """
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for i in range(0, len(dataset), self.config.batch_size):
            batch_data = dataset[i:i + self.config.batch_size]
            collator = PreferenceDataCollator()
            batch = collator(batch_data)

            # 计算对数概率
            chosen_logprobs = np.array([
                self._compute_policy_logprob(p + c)
                for p, c in zip(batch.prompts, batch.chosen_responses)
            ])
            rejected_logprobs = np.array([
                self._compute_policy_logprob(p + r)
                for p, r in zip(batch.prompts, batch.rejected_responses)
            ])
            ref_chosen = np.array([
                self._compute_ref_logprob(p + c)
                for p, c in zip(batch.prompts, batch.chosen_responses)
            ])
            ref_rejected = np.array([
                self._compute_ref_logprob(p + r)
                for p, r in zip(batch.prompts, batch.rejected_responses)
            ])

            loss, metrics = self._loss_fn(
                chosen_logprobs, rejected_logprobs,
                ref_chosen, ref_rejected,
            )

            total_loss += loss
            total_accuracy += metrics["accuracy"]
            num_batches += 1

        return {
            "eval_loss": total_loss / max(num_batches, 1),
            "eval_accuracy": total_accuracy / max(num_batches, 1),
        }

    def __repr__(self) -> str:
        return f"DPOTrainer(step={self._step}, beta={self.config.beta})"
