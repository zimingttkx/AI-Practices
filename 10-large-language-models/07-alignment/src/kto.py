"""
KTO (Kahneman-Tversky Optimization)

基于前景理论的模型对齐优化，无需成对偏好数据。

参考文献:
[1] Ethayarajh, K., et al. (2024). KTO: Model Alignment as Prospect Theoretic
    Optimization. https://arxiv.org/abs/2402.01306
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "KTOConfig",
    "KTOLoss",
    "KTOTrainer",
    "KTOBatch",
    "create_kto_trainer",
]


@dataclass
class KTOConfig:
    """KTO 配置"""

    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    learning_rate: float = 5e-7
    reference_free: bool = False

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"beta必须为正数，得到 {self.beta}")
        if self.desirable_weight < 0:
            raise ValueError(f"desirable_weight必须非负，得到 {self.desirable_weight}")
        if self.undesirable_weight < 0:
            raise ValueError(f"undesirable_weight必须非负，得到 {self.undesirable_weight}")


@dataclass
class KTOBatch:
    """KTO 训练批次"""

    prompts: list[str]
    responses: list[str]
    labels: list[int]  # 1=desirable, 0=undesirable

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.responses) or len(self.prompts) != len(
            self.labels
        ):
            raise ValueError("所有列表长度必须相同")
        if not all(label in (0, 1) for label in self.labels):
            raise ValueError("labels必须是0或1")

    def __len__(self) -> int:
        return len(self.prompts)

    def get_desirable(self) -> tuple[list[str], list[str]]:
        """获取期望样本"""
        indices = [i for i, label in enumerate(self.labels) if label == 1]
        return (
            [self.prompts[i] for i in indices],
            [self.responses[i] for i in indices],
        )

    def get_undesirable(self) -> tuple[list[str], list[str]]:
        """获取不期望样本"""
        indices = [i for i, label in enumerate(self.labels) if label == 0]
        return (
            [self.prompts[i] for i in indices],
            [self.responses[i] for i in indices],
        )


class KTOLoss:
    """KTO 损失函数"""

    def __init__(self, config: KTOConfig):
        self.config = config

    def compute_loss(
        self,
        policy_logps: np.ndarray,
        ref_logps: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """计算KTO损失"""
        if len(policy_logps) != len(ref_logps) or len(policy_logps) != len(labels):
            raise ValueError("所有数组长度必须相同")

        kl_div = policy_logps - ref_logps

        desirable_mask = labels == 1
        undesirable_mask = labels == 0

        loss_desirable = 0.0
        loss_undesirable = 0.0

        if desirable_mask.any():
            kl_desirable = kl_div[desirable_mask]
            loss_desirable = -np.mean(
                self._sigmoid(self.config.beta * kl_desirable)
            ) * self.config.desirable_weight

        if undesirable_mask.any():
            kl_undesirable = kl_div[undesirable_mask]
            loss_undesirable = -np.mean(
                self._sigmoid(-self.config.beta * kl_undesirable)
            ) * self.config.undesirable_weight

        total_loss = loss_desirable + loss_undesirable

        return total_loss, {
            "loss_desirable": loss_desirable,
            "loss_undesirable": loss_undesirable,
            "avg_kl": float(np.mean(np.abs(kl_div))),
            "num_desirable": int(desirable_mask.sum()),
            "num_undesirable": int(undesirable_mask.sum()),
        }

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


class KTOTrainer:
    """KTO 训练器"""

    def __init__(self, config: KTOConfig):
        self.config = config
        self.loss_fn = KTOLoss(config)
        self.training_history: list[dict] = []

    def train_step(self, batch: KTOBatch) -> dict[str, float]:
        """训练步骤"""
        policy_logps = self._compute_logprobs(batch.prompts, batch.responses)
        ref_logps = self._compute_reference_logprobs(batch.prompts, batch.responses)
        labels = np.array(batch.labels)

        loss, metrics = self.loss_fn.compute_loss(policy_logps, ref_logps, labels)

        self.training_history.append(
            {
                "loss": loss,
                **metrics,
            }
        )

        return {
            "loss": loss,
            **metrics,
        }

    def compute_kto_loss(self, batch: KTOBatch) -> float:
        """计算KTO损失"""
        policy_logps = self._compute_logprobs(batch.prompts, batch.responses)
        ref_logps = self._compute_reference_logprobs(batch.prompts, batch.responses)
        labels = np.array(batch.labels)

        loss, _ = self.loss_fn.compute_loss(policy_logps, ref_logps, labels)
        return loss

    def evaluate(self, batch: KTOBatch) -> dict[str, float]:
        """评估"""
        policy_logps = self._compute_logprobs(batch.prompts, batch.responses)
        ref_logps = self._compute_reference_logprobs(batch.prompts, batch.responses)
        labels = np.array(batch.labels)

        _, metrics = self.loss_fn.compute_loss(policy_logps, ref_logps, labels)

        desirable_prompts, desirable_responses = batch.get_desirable()
        undesirable_prompts, undesirable_responses = batch.get_undesirable()

        desirable_logps = (
            self._compute_logprobs(desirable_prompts, desirable_responses)
            if desirable_prompts
            else np.array([])
        )
        undesirable_logps = (
            self._compute_logprobs(undesirable_prompts, undesirable_responses)
            if undesirable_prompts
            else np.array([])
        )

        return {
            **metrics,
            "avg_desirable_logp": (
                float(np.mean(desirable_logps)) if len(desirable_logps) > 0 else 0.0
            ),
            "avg_undesirable_logp": (
                float(np.mean(undesirable_logps))
                if len(undesirable_logps) > 0
                else 0.0
            ),
        }

    def _compute_logprobs(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        """计算对数概率"""
        return np.array(
            [-len(r) / 100.0 + np.random.randn() * 0.1 for r in responses]
        )

    def _compute_reference_logprobs(
        self, prompts: list[str], responses: list[str]
    ) -> np.ndarray:
        """计算参考对数概率"""
        return np.array([-len(r) / 100.0 for r in responses])


def create_kto_trainer(
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> KTOTrainer:
    """工厂函数"""
    config = KTOConfig(
        beta=beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
    )
    return KTOTrainer(config)
