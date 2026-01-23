"""
ORPO (Odds Ratio Preference Optimization)

无需参考模型的对齐优化方法。

参考文献:
[1] Hong, J., et al. (2024). ORPO: Monolithic Preference Optimization without
    Reference Model. https://arxiv.org/abs/2403.07691
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = [
    "ORPOConfig",
    "ORPOLoss",
    "ORPOTrainer",
    "PreferenceBatch",
    "create_orpo_trainer",
]


@dataclass
class ORPOConfig:
    """ORPO 配置"""

    lambda_or: float = 0.1
    learning_rate: float = 5e-7
    max_length: int = 512
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda_or < 0:
            raise ValueError(f"lambda_or必须非负，得到 {self.lambda_or}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate必须为正数，得到 {self.learning_rate}")
        if not 0 <= self.label_smoothing < 0.5:
            raise ValueError("label_smoothing必须在[0,0.5)范围内")


@dataclass
class PreferenceBatch:
    """偏好批次"""

    prompts: list[str]
    chosen_responses: list[str]
    rejected_responses: list[str]

    def __post_init__(self) -> None:
        if (
            len(self.prompts) != len(self.chosen_responses)
            or len(self.prompts) != len(self.rejected_responses)
        ):
            raise ValueError("所有列表长度必须相同")

    def __len__(self) -> int:
        return len(self.prompts)


class ORPOLoss:
    """ORPO 损失函数"""

    def __init__(self, config: ORPOConfig):
        self.config = config

    def compute_loss(
        self,
        chosen_logps: np.ndarray,
        rejected_logps: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """计算ORPO损失"""
        if len(chosen_logps) != len(rejected_logps):
            raise ValueError("chosen和rejected长度必须相同")

        nll_loss = -np.mean(chosen_logps)

        log_odds = np.log(self._sigmoid(chosen_logps)) - np.log(
            self._sigmoid(rejected_logps)
        )
        or_loss = -np.mean(np.log(self._sigmoid(log_odds)))

        total_loss = nll_loss + self.config.lambda_or * or_loss

        accuracy = np.mean((chosen_logps > rejected_logps).astype(float))

        return total_loss, {
            "nll_loss": float(nll_loss),
            "or_loss": float(or_loss),
            "accuracy": float(accuracy),
            "avg_chosen_logp": float(np.mean(chosen_logps)),
            "avg_rejected_logp": float(np.mean(rejected_logps)),
        }

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


class ORPOTrainer:
    """ORPO 训练器"""

    def __init__(self, config: ORPOConfig):
        self.config = config
        self.loss_fn = ORPOLoss(config)
        self.training_history: list[dict] = []

    def train_step(self, batch: PreferenceBatch) -> dict[str, float]:
        """训练步骤"""
        chosen_logps = self._compute_logprobs(batch.prompts, batch.chosen_responses)
        rejected_logps = self._compute_logprobs(
            batch.prompts, batch.rejected_responses
        )

        loss, metrics = self.loss_fn.compute_loss(chosen_logps, rejected_logps)

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

    def compute_odds_ratio(
        self, chosen_logps: np.ndarray, rejected_logps: np.ndarray
    ) -> np.ndarray:
        """计算Odds Ratio"""
        if len(chosen_logps) != len(rejected_logps):
            raise ValueError("chosen和rejected长度必须相同")

        chosen_odds = np.exp(chosen_logps) / (1 - np.exp(chosen_logps) + 1e-10)
        rejected_odds = np.exp(rejected_logps) / (1 - np.exp(rejected_logps) + 1e-10)

        return chosen_odds / (rejected_odds + 1e-10)

    def evaluate(self, batch: PreferenceBatch) -> dict[str, float]:
        """评估"""
        chosen_logps = self._compute_logprobs(batch.prompts, batch.chosen_responses)
        rejected_logps = self._compute_logprobs(
            batch.prompts, batch.rejected_responses
        )

        _, metrics = self.loss_fn.compute_loss(chosen_logps, rejected_logps)

        odds_ratios = self.compute_odds_ratio(chosen_logps, rejected_logps)

        return {
            **metrics,
            "avg_odds_ratio": float(np.mean(odds_ratios)),
            "median_odds_ratio": float(np.median(odds_ratios)),
        }

    def _compute_logprobs(self, prompts: list[str], responses: list[str]) -> np.ndarray:
        """计算对数概率"""
        return np.array(
            [-len(r) / 100.0 + np.random.randn() * 0.1 for r in responses]
        )


def create_orpo_trainer(
    lambda_or: float = 0.1,
    learning_rate: float = 5e-7,
) -> ORPOTrainer:
    """工厂函数"""
    config = ORPOConfig(
        lambda_or=lambda_or,
        learning_rate=learning_rate,
    )
    return ORPOTrainer(config)
