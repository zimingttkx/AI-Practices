"""
07-alignment 模块最严格单元测试

测试策略：
    - 边界条件测试
    - 异常输入测试
    - 数值稳定性测试
    - 类型验证测试
    - 状态一致性测试
"""

import pytest
import numpy as np
from typing import Dict, List

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from reward_model import (
    RewardModelConfig,
    RewardModelOutput,
    PreferenceExample,
    PreferenceDataset,
    PairwiseRewardModel,
    RewardModel,
)
from rlhf import (
    RLHFConfig,
    RLHFBatch,
    ValueHead,
    PPOTrainer,
    RLHFTrainer,
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
# RewardModelConfig 严格测试
# ============================================================

class TestStrictRewardModelConfig:
    """RewardModelConfig最严格测试。"""

    def test_hidden_size_zero(self):
        """hidden_size=0应抛出异常。"""
        with pytest.raises(ValueError, match="hidden_size"):
            RewardModelConfig(hidden_size=0)

    def test_hidden_size_negative(self):
        """hidden_size为负数应抛出异常。"""
        with pytest.raises(ValueError, match="hidden_size"):
            RewardModelConfig(hidden_size=-100)

    def test_num_layers_zero(self):
        """num_layers=0应抛出异常。"""
        with pytest.raises(ValueError, match="num_layers"):
            RewardModelConfig(num_layers=0)

    def test_num_layers_negative(self):
        """num_layers为负数应抛出异常。"""
        with pytest.raises(ValueError, match="num_layers"):
            RewardModelConfig(num_layers=-1)

    def test_dropout_negative(self):
        """dropout为负数应抛出异常。"""
        with pytest.raises(ValueError, match="dropout"):
            RewardModelConfig(dropout=-0.1)

    def test_dropout_one(self):
        """dropout=1应抛出异常。"""
        with pytest.raises(ValueError, match="dropout"):
            RewardModelConfig(dropout=1.0)

    def test_dropout_greater_than_one(self):
        """dropout>1应抛出异常。"""
        with pytest.raises(ValueError, match="dropout"):
            RewardModelConfig(dropout=1.5)

    def test_temperature_zero(self):
        """temperature=0应抛出异常。"""
        with pytest.raises(ValueError, match="temperature"):
            RewardModelConfig(temperature=0)

    def test_temperature_negative(self):
        """temperature为负数应抛出异常。"""
        with pytest.raises(ValueError, match="temperature"):
            RewardModelConfig(temperature=-0.5)

    def test_valid_edge_cases(self):
        """测试有效边界值。"""
        # 最小有效值
        config = RewardModelConfig(
            hidden_size=1,
            num_layers=1,
            dropout=0.0,
            temperature=0.001,
        )
        assert config.hidden_size == 1
        assert config.dropout == 0.0

    def test_large_values(self):
        """测试大数值。"""
        config = RewardModelConfig(
            hidden_size=10000,
            num_layers=100,
            temperature=100.0,
        )
        assert config.hidden_size == 10000


# ============================================================
# RewardModelOutput 严格测试
# ============================================================

class TestStrictRewardModelOutput:
    """RewardModelOutput最严格测试。"""

    def test_empty_rewards(self):
        """测试空奖励数组。"""
        rewards = np.array([])
        logits = np.array([])
        output = RewardModelOutput(rewards=rewards, logits=logits)
        assert len(output.rewards) == 0

    def test_single_reward(self):
        """测试单个奖励。"""
        rewards = np.array([0.5])
        output = RewardModelOutput(rewards=rewards, logits=rewards)
        assert output.mean_reward == 0.5
        assert output.std_reward == 0.0

    def test_negative_rewards(self):
        """测试负奖励。"""
        rewards = np.array([-1.0, -2.0, -3.0])
        output = RewardModelOutput(rewards=rewards, logits=rewards)
        assert output.mean_reward == -2.0

    def test_mixed_rewards(self):
        """测试混合正负奖励。"""
        rewards = np.array([-1.0, 0.0, 1.0])
        output = RewardModelOutput(rewards=rewards, logits=rewards)
        assert abs(output.mean_reward) < 1e-10

    def test_large_rewards(self):
        """测试大数值奖励。"""
        rewards = np.array([1e10, 1e10, 1e10])
        output = RewardModelOutput(rewards=rewards, logits=rewards)
        assert output.mean_reward == 1e10

    def test_with_hidden_states(self):
        """测试带隐藏状态。"""
        rewards = np.array([0.5])
        hidden = np.random.randn(1, 768)
        output = RewardModelOutput(
            rewards=rewards, logits=rewards, hidden_states=hidden
        )
        assert output.hidden_states is not None
        assert output.hidden_states.shape == (1, 768)

    def test_repr(self):
        """测试字符串表示。"""
        rewards = np.array([1.0, 2.0])
        output = RewardModelOutput(rewards=rewards, logits=rewards)
        repr_str = repr(output)
        assert "RewardModelOutput" in repr_str
        assert "mean=" in repr_str


# ============================================================
# PreferenceExample 严格测试
# ============================================================

class TestStrictPreferenceExample:
    """PreferenceExample最严格测试。"""

    def test_empty_strings(self):
        """测试空字符串。"""
        ex = PreferenceExample(prompt="", chosen="", rejected="")
        assert ex.prompt == ""

    def test_unicode_content(self):
        """测试Unicode内容。"""
        ex = PreferenceExample(
            prompt="你好世界🌍",
            chosen="好的回答👍",
            rejected="差的回答👎",
        )
        assert "🌍" in ex.prompt

    def test_long_content(self):
        """测试长内容。"""
        long_text = "a" * 10000
        ex = PreferenceExample(
            prompt=long_text,
            chosen=long_text,
            rejected=long_text,
        )
        assert len(ex.prompt) == 10000

    def test_with_scores(self):
        """测试带分数。"""
        ex = PreferenceExample(
            prompt="问题",
            chosen="好",
            rejected="差",
            chosen_score=0.9,
            rejected_score=0.1,
        )
        assert ex.chosen_score == 0.9
        assert ex.rejected_score == 0.1

    def test_repr_truncation(self):
        """测试repr截断。"""
        ex = PreferenceExample(
            prompt="a" * 100,
            chosen="b",
            rejected="c",
        )
        repr_str = repr(ex)
        assert "..." in repr_str


# ============================================================
# PreferenceDataset 严格测试
# ============================================================

class TestStrictPreferenceDataset:
    """PreferenceDataset最严格测试。"""

    def test_sample_larger_than_dataset(self):
        """采样数大于数据集大小。"""
        dataset = PreferenceDataset()
        dataset.add("q", "a", "b")
        batch = dataset.sample(100)
        assert len(batch) == 1

    def test_sample_zero(self):
        """采样0个。"""
        dataset = PreferenceDataset()
        dataset.add("q", "a", "b")
        batch = dataset.sample(0)
        assert len(batch) == 0

    def test_iteration(self):
        """测试迭代。"""
        dataset = PreferenceDataset()
        for i in range(5):
            dataset.add(f"q{i}", f"a{i}", f"b{i}")
        count = 0
        for ex in dataset:
            count += 1
        assert count == 5

    def test_index_out_of_range(self):
        """测试索引越界。"""
        dataset = PreferenceDataset()
        dataset.add("q", "a", "b")
        with pytest.raises(IndexError):
            _ = dataset[10]

    def test_negative_index(self):
        """测试负索引。"""
        dataset = PreferenceDataset()
        dataset.add("q1", "a1", "b1")
        dataset.add("q2", "a2", "b2")
        ex = dataset[-1]
        assert ex.prompt == "q2"

    def test_init_with_examples(self):
        """测试带初始样本初始化。"""
        examples = [
            PreferenceExample("q1", "a1", "b1"),
            PreferenceExample("q2", "a2", "b2"),
        ]
        dataset = PreferenceDataset(examples=examples)
        assert len(dataset) == 2


# ============================================================
# PairwiseRewardModel 严格测试
# ============================================================

class TestStrictPairwiseRewardModel:
    """PairwiseRewardModel最严格测试。"""

    def test_custom_embed_fn(self):
        """测试自定义嵌入函数。"""
        def custom_embed(text):
            return np.ones(768) * len(text)
        model = PairwiseRewardModel(embed_fn=custom_embed)
        assert model._embed_fn == custom_embed

    def test_forward_single_sample(self):
        """测试单样本前向传播。"""
        model = PairwiseRewardModel()
        input_ids = np.random.randint(0, 1000, (1, 10))
        output = model.forward(input_ids)
        assert len(output.rewards) == 1

    def test_forward_large_batch(self):
        """测试大批次前向传播。"""
        model = PairwiseRewardModel()
        input_ids = np.random.randint(0, 1000, (128, 32))
        output = model.forward(input_ids)
        assert len(output.rewards) == 128

    def test_pairwise_loss_equal_rewards(self):
        """测试相等奖励的损失。"""
        model = PairwiseRewardModel()
        rewards = np.array([0.5, 0.5, 0.5])
        loss = model.compute_pairwise_loss(rewards, rewards)
        # 相等奖励时，损失应为-log(0.5) ≈ 0.693
        assert abs(loss - 0.693) < 0.01

    def test_accuracy_all_wrong(self):
        """测试全错准确率。"""
        model = PairwiseRewardModel()
        chosen = np.array([0.1, 0.2, 0.3])
        rejected = np.array([0.5, 0.6, 0.7])
        acc = model.compute_accuracy(chosen, rejected)
        assert acc == 0.0

    def test_accuracy_partial(self):
        """测试部分正确准确率。"""
        model = PairwiseRewardModel()
        chosen = np.array([1.0, 0.2])
        rejected = np.array([0.5, 0.6])
        acc = model.compute_accuracy(chosen, rejected)
        assert acc == 0.5

    def test_train_step_single_sample(self):
        """测试单样本训练。"""
        model = PairwiseRewardModel()
        dataset = PreferenceDataset()
        dataset.add("q", "good", "bad")
        metrics = model.train_step(dataset.sample(1))
        assert "loss" in metrics

    def test_compute_reward_short_response(self):
        """测试短回复奖励。"""
        model = PairwiseRewardModel()
        reward = model.compute_reward("问题", "短")
        assert reward < 0  # 太短应该有惩罚

    def test_compute_reward_long_response(self):
        """测试长回复奖励。"""
        model = PairwiseRewardModel()
        reward = model.compute_reward("问题", "a" * 1000)
        assert reward < 0.5  # 太长应该有惩罚

    def test_compute_reward_optimal_length(self):
        """测试最优长度回复奖励。"""
        model = PairwiseRewardModel()
        reward = model.compute_reward("问题", "a" * 100)
        assert reward >= 0.5


# ============================================================
# RLHFConfig 严格测试
# ============================================================

class TestStrictRLHFConfig:
    """RLHFConfig最严格测试。"""

    def test_learning_rate_zero(self):
        """learning_rate=0应抛出异常。"""
        with pytest.raises(ValueError, match="learning_rate"):
            RLHFConfig(learning_rate=0)

    def test_batch_size_zero(self):
        """batch_size=0应抛出异常。"""
        with pytest.raises(ValueError, match="batch_size"):
            RLHFConfig(batch_size=0)

    def test_ppo_epochs_zero(self):
        """ppo_epochs=0应抛出异常。"""
        with pytest.raises(ValueError, match="ppo_epochs"):
            RLHFConfig(ppo_epochs=0)

    def test_clip_epsilon_zero(self):
        """clip_epsilon=0应抛出异常。"""
        with pytest.raises(ValueError, match="clip_epsilon"):
            RLHFConfig(clip_epsilon=0)

    def test_clip_epsilon_one(self):
        """clip_epsilon=1应抛出异常。"""
        with pytest.raises(ValueError, match="clip_epsilon"):
            RLHFConfig(clip_epsilon=1.0)

    def test_gamma_negative(self):
        """gamma为负数应抛出异常。"""
        with pytest.raises(ValueError, match="gamma"):
            RLHFConfig(gamma=-0.1)

    def test_gamma_greater_than_one(self):
        """gamma>1应抛出异常。"""
        with pytest.raises(ValueError, match="gamma"):
            RLHFConfig(gamma=1.1)

    def test_valid_gamma_boundaries(self):
        """测试gamma边界值。"""
        config0 = RLHFConfig(gamma=0.0)
        config1 = RLHFConfig(gamma=1.0)
        assert config0.gamma == 0.0
        assert config1.gamma == 1.0


# ============================================================
# ValueHead 严格测试
# ============================================================

class TestStrictValueHead:
    """ValueHead最严格测试。"""

    def test_single_sample(self):
        """测试单样本。"""
        vh = ValueHead(hidden_size=64, dropout=0.0)
        hidden = np.random.randn(1, 64)
        values = vh.forward(hidden)
        assert values.shape == (1,)

    def test_large_batch(self):
        """测试大批次。"""
        vh = ValueHead(hidden_size=64, dropout=0.0)
        hidden = np.random.randn(256, 64)
        values = vh.forward(hidden)
        assert values.shape == (256,)

    def test_zero_dropout(self):
        """测试零dropout。"""
        vh = ValueHead(hidden_size=32, dropout=0.0)
        hidden = np.ones((4, 32))
        v1 = vh.forward(hidden)
        v2 = vh.forward(hidden)
        # 无dropout时结果应一致
        np.testing.assert_array_equal(v1, v2)

    def test_repr(self):
        """测试字符串表示。"""
        vh = ValueHead(hidden_size=512)
        assert "512" in repr(vh)


# ============================================================
# compute_advantages 严格测试
# ============================================================

class TestStrictComputeAdvantages:
    """compute_advantages最严格测试。"""

    def test_single_sample(self):
        """测试单样本。"""
        rewards = np.array([1.0])
        values = np.array([0.5])
        adv, ret = compute_advantages(rewards, values)
        assert len(adv) == 1
        assert len(ret) == 1

    def test_zero_rewards(self):
        """测试零奖励。"""
        rewards = np.zeros(5)
        values = np.zeros(5)
        adv, ret = compute_advantages(rewards, values)
        assert len(adv) == 5

    def test_negative_rewards(self):
        """测试负奖励。"""
        rewards = np.array([-1.0, -2.0, -3.0])
        values = np.array([0.0, 0.0, 0.0])
        adv, ret = compute_advantages(rewards, values)
        assert len(adv) == 3

    def test_returns_calculation(self):
        """测试回报计算。"""
        rewards = np.array([1.0, 2.0])
        values = np.array([0.5, 0.5])
        adv, ret = compute_advantages(rewards, values)
        # 在RLHF单步场景中，returns = rewards
        np.testing.assert_array_almost_equal(ret, rewards)


# ============================================================
# compute_rewards 严格测试
# ============================================================

class TestStrictComputeRewards:
    """compute_rewards最严格测试。"""

    def test_empty_responses(self):
        """测试空回复列表。"""
        rewards = compute_rewards([], lambda x: 1.0)
        assert len(rewards) == 0

    def test_constant_reward(self):
        """测试常数奖励。"""
        responses = ["a", "b", "c"]
        rewards = compute_rewards(responses, lambda x: 5.0)
        np.testing.assert_array_equal(rewards, [5.0, 5.0, 5.0])

    def test_kl_penalty_effect(self):
        """测试KL惩罚效果。"""
        responses = ["a"]
        kl = np.array([10.0])
        rewards = compute_rewards(responses, lambda x: 1.0, kl, kl_coef=0.1)
        assert rewards[0] == 0.0  # 1.0 - 0.1 * 10.0 = 0.0

    def test_no_kl_penalty(self):
        """测试无KL惩罚。"""
        responses = ["a"]
        rewards = compute_rewards(responses, lambda x: 1.0, None, kl_coef=0.1)
        assert rewards[0] == 1.0


# ============================================================
# PPOTrainer 严格测试
# ============================================================

class TestStrictPPOTrainer:
    """PPOTrainer最严格测试。"""

    def test_custom_config(self):
        """测试自定义配置。"""
        config = RLHFConfig(learning_rate=1e-4, ppo_epochs=2)
        trainer = PPOTrainer(config=config)
        assert trainer.config.learning_rate == 1e-4

    def test_custom_reward_fn(self):
        """测试自定义奖励函数。"""
        trainer = PPOTrainer(reward_fn=lambda x: len(x) * 0.01)
        batch = trainer.generate_and_score(["short", "longer text"])
        assert batch.rewards[1] > batch.rewards[0]

    def test_multiple_train_steps(self):
        """测试多步训练。"""
        trainer = PPOTrainer()
        for i in range(5):
            batch = trainer.generate_and_score(["q1", "q2"])
            metrics = trainer.train_step(batch)
            assert metrics["step"] == i + 1

    def test_kl_coef_decrease(self):
        """测试KL系数减少。"""
        trainer = PPOTrainer()
        initial = trainer.kl_coef
        trainer.update_kl_coef(0.1)  # 低KL
        assert trainer.kl_coef < initial

    def test_repr(self):
        """测试字符串表示。"""
        trainer = PPOTrainer()
        assert "PPOTrainer" in repr(trainer)


# ============================================================
# DPOConfig 严格测试
# ============================================================

class TestStrictDPOConfig:
    """DPOConfig最严格测试。"""

    def test_beta_zero(self):
        """beta=0应抛出异常。"""
        with pytest.raises(ValueError, match="beta"):
            DPOConfig(beta=0)

    def test_learning_rate_zero(self):
        """learning_rate=0应抛出异常。"""
        with pytest.raises(ValueError, match="learning_rate"):
            DPOConfig(learning_rate=0)

    def test_batch_size_zero(self):
        """batch_size=0应抛出异常。"""
        with pytest.raises(ValueError, match="batch_size"):
            DPOConfig(batch_size=0)

    def test_label_smoothing_negative(self):
        """label_smoothing为负数应抛出异常。"""
        with pytest.raises(ValueError, match="label_smoothing"):
            DPOConfig(label_smoothing=-0.1)

    def test_label_smoothing_half(self):
        """label_smoothing>=0.5应抛出异常。"""
        with pytest.raises(ValueError, match="label_smoothing"):
            DPOConfig(label_smoothing=0.5)

    def test_all_loss_types(self):
        """测试所有损失类型。"""
        for lt in ["sigmoid", "hinge", "ipo"]:
            config = DPOConfig(loss_type=lt)
            assert config.loss_type == lt


# ============================================================
# DPOLoss 严格测试
# ============================================================

class TestStrictDPOLoss:
    """DPOLoss最严格测试。"""

    def test_single_sample(self):
        """测试单样本。"""
        loss_fn = DPOLoss(beta=0.1)
        c = np.array([-5.0])
        r = np.array([-10.0])
        loss, metrics = loss_fn(c, r, c, r)
        assert loss >= 0

    def test_large_batch(self):
        """测试大批次。"""
        loss_fn = DPOLoss(beta=0.1)
        c = np.random.randn(1000)
        r = np.random.randn(1000) - 1
        loss, metrics = loss_fn(c, r, c, r)
        assert "accuracy" in metrics

    def test_label_smoothing(self):
        """测试标签平滑。"""
        loss_fn = DPOLoss(beta=0.1, label_smoothing=0.1)
        c = np.array([-5.0])
        r = np.array([-10.0])
        loss, _ = loss_fn(c, r, c, r)
        assert loss >= 0

    def test_metrics_keys(self):
        """测试指标键。"""
        loss_fn = DPOLoss(beta=0.1)
        c = np.array([-5.0])
        r = np.array([-10.0])
        _, metrics = loss_fn(c, r, c, r)
        assert "loss" in metrics
        assert "chosen_rewards" in metrics
        assert "rejected_rewards" in metrics
        assert "reward_margin" in metrics
        assert "accuracy" in metrics


# ============================================================
# DPOTrainer 严格测试
# ============================================================

class TestStrictDPOTrainer:
    """DPOTrainer最严格测试。"""

    def test_custom_config(self):
        """测试自定义配置。"""
        config = DPOConfig(beta=0.5, loss_type="hinge")
        trainer = DPOTrainer(config=config)
        assert trainer.config.beta == 0.5

    def test_multiple_train_steps(self):
        """测试多步训练。"""
        trainer = DPOTrainer()
        batch = DPOBatch(
            prompts=["q"],
            chosen_responses=["good"],
            rejected_responses=["bad"],
        )
        for i in range(5):
            metrics = trainer.train_step(batch)
            assert metrics["step"] == i + 1

    def test_implicit_rewards_shape(self):
        """测试隐式奖励形状。"""
        trainer = DPOTrainer()
        rewards = trainer.compute_implicit_rewards(
            ["q1", "q2", "q3"],
            ["r1", "r2", "r3"],
        )
        assert rewards.shape == (3,)

    def test_evaluate_empty(self):
        """测试空数据集评估。"""
        trainer = DPOTrainer()
        metrics = trainer.evaluate([])
        assert metrics["eval_loss"] == 0.0

    def test_repr(self):
        """测试字符串表示。"""
        trainer = DPOTrainer()
        assert "DPOTrainer" in repr(trainer)


# ============================================================
# PreferenceDataCollator 严格测试
# ============================================================

class TestStrictPreferenceDataCollator:
    """PreferenceDataCollator最严格测试。"""

    def test_empty_examples(self):
        """测试空样本列表。"""
        collator = PreferenceDataCollator()
        batch = collator([])
        assert len(batch.prompts) == 0

    def test_single_example(self):
        """测试单样本。"""
        collator = PreferenceDataCollator()
        batch = collator([{"prompt": "q", "chosen": "a", "rejected": "b"}])
        assert len(batch.prompts) == 1

    def test_custom_max_length(self):
        """测试自定义最大长度。"""
        collator = PreferenceDataCollator(max_length=128)
        assert collator.max_length == 128


# ============================================================
# 数值稳定性测试
# ============================================================

class TestNumericalStability:
    """数值稳定性测试。"""

    def test_dpo_loss_extreme_values(self):
        """测试DPO损失极端值。"""
        loss_fn = DPOLoss(beta=0.1)
        # 极大差异
        c = np.array([0.0])
        r = np.array([-100.0])
        loss, _ = loss_fn(c, r, c, r)
        assert np.isfinite(loss)

    def test_pairwise_loss_extreme(self):
        """测试成对损失极端值。"""
        model = PairwiseRewardModel()
        chosen = np.array([100.0])
        rejected = np.array([-100.0])
        loss = model.compute_pairwise_loss(chosen, rejected)
        assert np.isfinite(loss)

    def test_advantages_large_values(self):
        """测试大数值优势计算。"""
        rewards = np.array([1e6, 1e6, 1e6])
        values = np.array([1e6, 1e6, 1e6])
        adv, ret = compute_advantages(rewards, values)
        assert all(np.isfinite(adv))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
