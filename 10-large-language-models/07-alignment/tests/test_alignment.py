"""
07-alignment 模块单元测试

测试覆盖：
    - RewardModel: 奖励模型配置、输出、训练
    - RLHF: PPO训练器、优势估计、奖励计算
    - DPO: DPO损失、训练器、偏好数据
"""

import pytest
import numpy as np
from typing import Dict, List

# 导入被测模块
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from reward_model import (
    RewardModelConfig,
    RewardModelOutput,
    PreferenceExample,
    PreferenceDataset,
    PairwiseRewardModel,
)
from rlhf import (
    RLHFConfig,
    RLHFBatch,
    ValueHead,
    PPOTrainer,
    compute_advantages,
    compute_rewards,
)
from dpo import (
    DPOConfig,
    DPOBatch,
    DPOLoss,
    DPOTrainer,
    PreferenceDataCollator,
    compute_dpo_loss,
    compute_reference_logprobs,
)


# ============================================================
# RewardModel 测试
# ============================================================

class TestRewardModelConfig:
    """RewardModelConfig测试类。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = RewardModelConfig()
        assert config.hidden_size == 768
        assert config.num_layers == 2
        assert config.dropout == 0.1
        assert config.normalize_rewards is True
        assert config.temperature == 1.0

    def test_custom_config(self):
        """测试自定义配置。"""
        config = RewardModelConfig(
            hidden_size=1024,
            num_layers=3,
            dropout=0.2,
            temperature=0.5,
        )
        assert config.hidden_size == 1024
        assert config.num_layers == 3

    def test_invalid_hidden_size(self):
        """测试无效hidden_size。"""
        with pytest.raises(ValueError):
            RewardModelConfig(hidden_size=0)

    def test_invalid_dropout(self):
        """测试无效dropout。"""
        with pytest.raises(ValueError):
            RewardModelConfig(dropout=1.5)

    def test_invalid_temperature(self):
        """测试无效temperature。"""
        with pytest.raises(ValueError):
            RewardModelConfig(temperature=-1)


class TestRewardModelOutput:
    """RewardModelOutput测试类。"""

    def test_output_creation(self):
        """测试输出创建。"""
        rewards = np.array([0.5, 0.3, 0.8])
        logits = np.array([1.0, 0.5, 1.5])
        output = RewardModelOutput(rewards=rewards, logits=logits)
        assert len(output.rewards) == 3
        assert output.hidden_states is None

    def test_mean_reward(self):
        """测试平均奖励。"""
        rewards = np.array([1.0, 2.0, 3.0])
        output = RewardModelOutput(rewards=rewards, logits=rewards)
        assert output.mean_reward == 2.0

    def test_std_reward(self):
        """测试奖励标准差。"""
        rewards = np.array([1.0, 1.0, 1.0])
        output = RewardModelOutput(rewards=rewards, logits=rewards)
        assert output.std_reward == 0.0


class TestPreferenceDataset:
    """PreferenceDataset测试类。"""

    def test_empty_dataset(self):
        """测试空数据集。"""
        dataset = PreferenceDataset()
        assert len(dataset) == 0

    def test_add_example(self):
        """测试添加样本。"""
        dataset = PreferenceDataset()
        dataset.add("问题", "好回答", "差回答")
        assert len(dataset) == 1

    def test_sample(self):
        """测试采样。"""
        dataset = PreferenceDataset()
        for i in range(10):
            dataset.add(f"问题{i}", f"好回答{i}", f"差回答{i}")
        batch = dataset.sample(5)
        assert len(batch) == 5

    def test_getitem(self):
        """测试索引访问。"""
        dataset = PreferenceDataset()
        dataset.add("问题", "好回答", "差回答")
        example = dataset[0]
        assert example.prompt == "问题"
        assert example.chosen == "好回答"


class TestPairwiseRewardModel:
    """PairwiseRewardModel测试类。"""

    def test_model_creation(self):
        """测试模型创建。"""
        model = PairwiseRewardModel()
        assert model.config.hidden_size == 768

    def test_forward(self):
        """测试前向传播。"""
        model = PairwiseRewardModel()
        input_ids = np.random.randint(0, 1000, (4, 32))
        output = model.forward(input_ids)
        assert len(output.rewards) == 4

    def test_pairwise_loss(self):
        """测试成对损失。"""
        model = PairwiseRewardModel()
        chosen = np.array([1.0, 0.8, 0.9])
        rejected = np.array([0.5, 0.3, 0.4])
        loss = model.compute_pairwise_loss(chosen, rejected)
        assert loss >= 0

    def test_accuracy(self):
        """测试准确率计算。"""
        model = PairwiseRewardModel()
        chosen = np.array([1.0, 0.8, 0.9])
        rejected = np.array([0.5, 0.3, 0.4])
        acc = model.compute_accuracy(chosen, rejected)
        assert acc == 1.0  # 所有chosen > rejected

    def test_train_step(self):
        """测试训练步骤。"""
        model = PairwiseRewardModel()
        dataset = PreferenceDataset()
        for i in range(10):
            dataset.add(f"问题{i}", f"好回答{i}", f"差回答{i}")
        batch = dataset.sample(4)
        metrics = model.train_step(batch)
        assert "loss" in metrics
        assert "accuracy" in metrics


# ============================================================
# RLHF 测试
# ============================================================

class TestRLHFConfig:
    """RLHFConfig测试类。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = RLHFConfig()
        assert config.learning_rate == 1e-5
        assert config.batch_size == 64
        assert config.ppo_epochs == 4
        assert config.clip_epsilon == 0.2

    def test_invalid_learning_rate(self):
        """测试无效学习率。"""
        with pytest.raises(ValueError):
            RLHFConfig(learning_rate=-1)

    def test_invalid_clip_epsilon(self):
        """测试无效裁剪参数。"""
        with pytest.raises(ValueError):
            RLHFConfig(clip_epsilon=1.5)


class TestValueHead:
    """ValueHead测试类。"""

    def test_creation(self):
        """测试创建。"""
        vh = ValueHead(hidden_size=256)
        assert vh.hidden_size == 256

    def test_forward(self):
        """测试前向传播。"""
        vh = ValueHead(hidden_size=128, dropout=0.0)
        hidden = np.random.randn(8, 128)
        values = vh.forward(hidden)
        assert values.shape == (8,)


class TestComputeAdvantages:
    """compute_advantages测试类。"""

    def test_basic(self):
        """测试基本优势计算。"""
        rewards = np.array([1.0, 0.5, 0.8])
        values = np.array([0.5, 0.5, 0.5])
        advantages, returns = compute_advantages(rewards, values)
        assert len(advantages) == 3
        assert len(returns) == 3

    def test_normalized(self):
        """测试归一化。"""
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([1.0, 1.0, 1.0])
        advantages, _ = compute_advantages(rewards, values)
        # 归一化后均值接近0
        assert abs(np.mean(advantages)) < 1e-6


class TestComputeRewards:
    """compute_rewards测试类。"""

    def test_basic(self):
        """测试基本奖励计算。"""
        responses = ["回复1", "回复2", "回复3"]
        reward_fn = lambda x: len(x) * 0.1
        rewards = compute_rewards(responses, reward_fn)
        assert len(rewards) == 3

    def test_with_kl_penalty(self):
        """测试带KL惩罚。"""
        responses = ["回复1", "回复2"]
        reward_fn = lambda x: 1.0
        kl = np.array([0.5, 0.3])
        rewards = compute_rewards(responses, reward_fn, kl, kl_coef=0.1)
        assert rewards[0] < 1.0  # 有KL惩罚


class TestPPOTrainer:
    """PPOTrainer测试类。"""

    def test_creation(self):
        """测试创建。"""
        trainer = PPOTrainer()
        assert trainer._step == 0

    def test_generate_and_score(self):
        """测试生成和评分。"""
        trainer = PPOTrainer()
        prompts = ["问题1", "问题2", "问题3"]
        batch = trainer.generate_and_score(prompts)
        assert len(batch.prompts) == 3
        assert len(batch.responses) == 3
        assert len(batch.rewards) == 3

    def test_train_step(self):
        """测试训练步骤。"""
        trainer = PPOTrainer()
        prompts = ["问题1", "问题2", "问题3", "问题4"]
        batch = trainer.generate_and_score(prompts)
        metrics = trainer.train_step(batch)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "kl" in metrics
        assert trainer._step == 1

    def test_kl_coef_update(self):
        """测试KL系数更新。"""
        trainer = PPOTrainer()
        initial_coef = trainer.kl_coef
        trainer.update_kl_coef(100.0)  # 高KL
        assert trainer.kl_coef > initial_coef


# ============================================================
# DPO 测试
# ============================================================

class TestDPOConfig:
    """DPOConfig测试类。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = DPOConfig()
        assert config.beta == 0.1
        assert config.loss_type == "sigmoid"

    def test_invalid_beta(self):
        """测试无效beta。"""
        with pytest.raises(ValueError):
            DPOConfig(beta=-1)

    def test_invalid_loss_type(self):
        """测试无效损失类型。"""
        with pytest.raises(ValueError):
            DPOConfig(loss_type="invalid")


class TestDPOLoss:
    """DPOLoss测试类。"""

    def test_sigmoid_loss(self):
        """测试sigmoid损失。"""
        loss_fn = DPOLoss(beta=0.1, loss_type="sigmoid")
        chosen = np.array([-10.0, -8.0])
        rejected = np.array([-12.0, -10.0])
        ref_chosen = np.array([-10.0, -8.0])
        ref_rejected = np.array([-12.0, -10.0])
        loss, metrics = loss_fn(chosen, rejected, ref_chosen, ref_rejected)
        assert loss >= 0
        assert "accuracy" in metrics

    def test_hinge_loss(self):
        """测试hinge损失。"""
        loss_fn = DPOLoss(beta=0.1, loss_type="hinge")
        chosen = np.array([-10.0])
        rejected = np.array([-12.0])
        loss, _ = loss_fn(chosen, rejected, chosen, rejected)
        assert loss >= 0

    def test_ipo_loss(self):
        """测试IPO损失。"""
        loss_fn = DPOLoss(beta=0.1, loss_type="ipo")
        chosen = np.array([-10.0])
        rejected = np.array([-12.0])
        loss, _ = loss_fn(chosen, rejected, chosen, rejected)
        assert loss >= 0


class TestPreferenceDataCollator:
    """PreferenceDataCollator测试类。"""

    def test_collate(self):
        """测试数据整理。"""
        collator = PreferenceDataCollator()
        examples = [
            {"prompt": "问题1", "chosen": "好", "rejected": "差"},
            {"prompt": "问题2", "chosen": "好2", "rejected": "差2"},
        ]
        batch = collator(examples)
        assert len(batch.prompts) == 2
        assert len(batch.chosen_responses) == 2


class TestComputeDPOLoss:
    """compute_dpo_loss测试类。"""

    def test_basic(self):
        """测试基本DPO损失。"""
        chosen = np.array([-10.0, -8.0])
        rejected = np.array([-12.0, -10.0])
        loss, metrics = compute_dpo_loss(
            chosen, rejected, chosen, rejected, beta=0.1
        )
        assert loss >= 0
        assert "reward_margin" in metrics


class TestComputeReferenceLogprobs:
    """compute_reference_logprobs测试类。"""

    def test_default(self):
        """测试默认对数概率。"""
        texts = ["hello world", "test"]
        logprobs = compute_reference_logprobs(texts)
        assert len(logprobs) == 2
        assert all(lp < 0 for lp in logprobs)

    def test_custom_fn(self):
        """测试自定义函数。"""
        texts = ["a", "b"]
        logprobs = compute_reference_logprobs(texts, lambda x: -1.0)
        assert all(lp == -1.0 for lp in logprobs)


class TestDPOTrainer:
    """DPOTrainer测试类。"""

    def test_creation(self):
        """测试创建。"""
        trainer = DPOTrainer()
        assert trainer._step == 0
        assert trainer.config.beta == 0.1

    def test_train_step(self):
        """测试训练步骤。"""
        trainer = DPOTrainer()
        batch = DPOBatch(
            prompts=["问题1", "问题2"],
            chosen_responses=["好回答1", "好回答2"],
            rejected_responses=["差回答1", "差回答2"],
        )
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert trainer._step == 1

    def test_implicit_rewards(self):
        """测试隐式奖励。"""
        trainer = DPOTrainer()
        prompts = ["问题1", "问题2"]
        responses = ["回复1", "回复2"]
        rewards = trainer.compute_implicit_rewards(prompts, responses)
        assert len(rewards) == 2

    def test_evaluate(self):
        """测试评估。"""
        trainer = DPOTrainer()
        dataset = [
            {"prompt": "问题1", "chosen": "好", "rejected": "差"},
            {"prompt": "问题2", "chosen": "好2", "rejected": "差2"},
        ]
        metrics = trainer.evaluate(dataset)
        assert "eval_loss" in metrics
        assert "eval_accuracy" in metrics


# ============================================================
# 集成测试
# ============================================================

class TestIntegration:
    """集成测试类。"""

    def test_reward_model_with_ppo(self):
        """测试奖励模型与PPO集成。"""
        # 创建奖励模型
        rm = PairwiseRewardModel()
        
        # 创建PPO训练器，使用奖励模型
        def reward_fn(response):
            return rm.compute_reward("", response)
        
        trainer = PPOTrainer(reward_fn=reward_fn)
        batch = trainer.generate_and_score(["问题1", "问题2"])
        metrics = trainer.train_step(batch)
        assert "mean_reward" in metrics

    def test_dpo_vs_rlhf_comparison(self):
        """测试DPO与RLHF对比。"""
        # DPO训练
        dpo_trainer = DPOTrainer()
        dpo_batch = DPOBatch(
            prompts=["问题"],
            chosen_responses=["好回答"],
            rejected_responses=["差回答"],
        )
        dpo_metrics = dpo_trainer.train_step(dpo_batch)
        
        # PPO训练
        ppo_trainer = PPOTrainer()
        ppo_batch = ppo_trainer.generate_and_score(["问题"])
        ppo_metrics = ppo_trainer.train_step(ppo_batch)
        
        # 两者都应该产生有效指标
        assert "loss" in dpo_metrics
        assert "policy_loss" in ppo_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
