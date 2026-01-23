"""
KTO 测试
"""

import numpy as np
import pytest
from src.kto import (
    KTOConfig,
    KTOLoss,
    KTOTrainer,
    KTOBatch,
    create_kto_trainer,
)


class TestKTOConfig:
    """测试配置"""

    def test_valid_config(self):
        config = KTOConfig()
        assert config.beta == 0.1
        assert config.desirable_weight == 1.0

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta必须为正数"):
            KTOConfig(beta=-0.1)

    def test_invalid_weights(self):
        with pytest.raises(ValueError, match="desirable_weight必须非负"):
            KTOConfig(desirable_weight=-1.0)


class TestKTOBatch:
    """测试批次"""

    def test_valid_batch(self):
        batch = KTOBatch(
            prompts=["q1", "q2"],
            responses=["a1", "a2"],
            labels=[1, 0],
        )
        assert len(batch) == 2

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="所有列表长度必须相同"):
            KTOBatch(prompts=["q1"], responses=["a1", "a2"], labels=[1])

    def test_invalid_labels(self):
        with pytest.raises(ValueError, match="labels必须是0或1"):
            KTOBatch(prompts=["q1"], responses=["a1"], labels=[2])

    def test_get_desirable(self):
        batch = KTOBatch(
            prompts=["q1", "q2", "q3"],
            responses=["a1", "a2", "a3"],
            labels=[1, 0, 1],
        )
        prompts, responses = batch.get_desirable()
        assert len(prompts) == 2
        assert prompts == ["q1", "q3"]

    def test_get_undesirable(self):
        batch = KTOBatch(
            prompts=["q1", "q2", "q3"],
            responses=["a1", "a2", "a3"],
            labels=[1, 0, 1],
        )
        prompts, responses = batch.get_undesirable()
        assert len(prompts) == 1
        assert prompts == ["q2"]


class TestKTOLoss:
    """测试损失函数"""

    @pytest.fixture
    def config(self):
        return KTOConfig()

    @pytest.fixture
    def loss_fn(self, config):
        return KTOLoss(config)

    def test_compute_loss(self, loss_fn):
        policy_logps = np.array([-1.0, -2.0, -1.5])
        ref_logps = np.array([-1.2, -1.8, -1.6])
        labels = np.array([1, 0, 1])

        loss, metrics = loss_fn.compute_loss(policy_logps, ref_logps, labels)
        assert isinstance(loss, float)
        assert "loss_desirable" in metrics
        assert "loss_undesirable" in metrics
        assert metrics["num_desirable"] == 2
        assert metrics["num_undesirable"] == 1

    def test_mismatched_lengths(self, loss_fn):
        with pytest.raises(ValueError, match="所有数组长度必须相同"):
            loss_fn.compute_loss(
                np.array([1.0]), np.array([1.0, 2.0]), np.array([1])
            )


class TestKTOTrainer:
    """测试训练器"""

    @pytest.fixture
    def config(self):
        return KTOConfig()

    @pytest.fixture
    def trainer(self, config):
        return KTOTrainer(config)

    def test_train_step(self, trainer):
        batch = KTOBatch(
            prompts=["q1", "q2"],
            responses=["a1", "a2"],
            labels=[1, 0],
        )
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert "avg_kl" in metrics
        assert len(trainer.training_history) == 1

    def test_compute_kto_loss(self, trainer):
        batch = KTOBatch(
            prompts=["q1", "q2"],
            responses=["a1", "a2"],
            labels=[1, 0],
        )
        loss = trainer.compute_kto_loss(batch)
        assert isinstance(loss, float)

    def test_evaluate(self, trainer):
        batch = KTOBatch(
            prompts=["q1", "q2", "q3"],
            responses=["a1", "a2", "a3"],
            labels=[1, 0, 1],
        )
        metrics = trainer.evaluate(batch)
        assert "avg_desirable_logp" in metrics
        assert "avg_undesirable_logp" in metrics


class TestFactoryFunction:
    """测试工厂函数"""

    def test_create_default(self):
        trainer = create_kto_trainer()
        assert trainer.config.beta == 0.1

    def test_create_custom(self):
        trainer = create_kto_trainer(
            beta=0.2,
            desirable_weight=1.5,
            undesirable_weight=0.5,
        )
        assert trainer.config.beta == 0.2
        assert trainer.config.desirable_weight == 1.5
