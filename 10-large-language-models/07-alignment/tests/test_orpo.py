"""
ORPO 测试
"""

import numpy as np
import pytest
from src.orpo import (
    ORPOConfig,
    ORPOLoss,
    ORPOTrainer,
    PreferenceBatch,
    create_orpo_trainer,
)


class TestORPOConfig:
    """测试配置"""

    def test_valid_config(self):
        config = ORPOConfig()
        assert config.lambda_or == 0.1
        assert config.learning_rate == 5e-7

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lambda_or必须非负"):
            ORPOConfig(lambda_or=-0.1)

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="learning_rate必须为正数"):
            ORPOConfig(learning_rate=0)

    def test_invalid_smoothing(self):
        with pytest.raises(ValueError, match="label_smoothing必须在"):
            ORPOConfig(label_smoothing=0.6)


class TestPreferenceBatch:
    """测试批次"""

    def test_valid_batch(self):
        batch = PreferenceBatch(
            prompts=["q1", "q2"],
            chosen_responses=["good1", "good2"],
            rejected_responses=["bad1", "bad2"],
        )
        assert len(batch) == 2

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="所有列表长度必须相同"):
            PreferenceBatch(
                prompts=["q1"],
                chosen_responses=["good1", "good2"],
                rejected_responses=["bad1"],
            )


class TestORPOLoss:
    """测试损失函数"""

    @pytest.fixture
    def config(self):
        return ORPOConfig()

    @pytest.fixture
    def loss_fn(self, config):
        return ORPOLoss(config)

    def test_compute_loss(self, loss_fn):
        chosen_logps = np.array([-1.0, -1.2, -0.9])
        rejected_logps = np.array([-2.0, -2.5, -1.8])

        loss, metrics = loss_fn.compute_loss(chosen_logps, rejected_logps)
        assert isinstance(loss, float)
        assert "nll_loss" in metrics
        assert "or_loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_mismatched_lengths(self, loss_fn):
        with pytest.raises(ValueError, match="chosen和rejected长度必须相同"):
            loss_fn.compute_loss(np.array([1.0]), np.array([1.0, 2.0]))


class TestORPOTrainer:
    """测试训练器"""

    @pytest.fixture
    def config(self):
        return ORPOConfig()

    @pytest.fixture
    def trainer(self, config):
        return ORPOTrainer(config)

    def test_train_step(self, trainer):
        batch = PreferenceBatch(
            prompts=["q1", "q2"],
            chosen_responses=["good1", "good2"],
            rejected_responses=["bad1", "bad2"],
        )
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert len(trainer.training_history) == 1

    def test_compute_odds_ratio(self, trainer):
        chosen_logps = np.array([-1.0, -1.2])
        rejected_logps = np.array([-2.0, -2.5])
        odds_ratios = trainer.compute_odds_ratio(chosen_logps, rejected_logps)
        assert len(odds_ratios) == 2
        assert all(odds_ratios > 0)

    def test_odds_ratio_mismatched(self, trainer):
        with pytest.raises(ValueError, match="chosen和rejected长度必须相同"):
            trainer.compute_odds_ratio(np.array([1.0]), np.array([1.0, 2.0]))

    def test_evaluate(self, trainer):
        batch = PreferenceBatch(
            prompts=["q1", "q2"],
            chosen_responses=["good1", "good2"],
            rejected_responses=["bad1", "bad2"],
        )
        metrics = trainer.evaluate(batch)
        assert "avg_odds_ratio" in metrics
        assert "median_odds_ratio" in metrics


class TestFactoryFunction:
    """测试工厂函数"""

    def test_create_default(self):
        trainer = create_orpo_trainer()
        assert trainer.config.lambda_or == 0.1

    def test_create_custom(self):
        trainer = create_orpo_trainer(lambda_or=0.2, learning_rate=1e-6)
        assert trainer.config.lambda_or == 0.2
        assert trainer.config.learning_rate == 1e-6
