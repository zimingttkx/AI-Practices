"""
RLAIF 测试
"""

import pytest
from src.rlaif import (
    RLAIFConfig,
    AIFeedbackGenerator,
    RLAIFTrainer,
    create_rlaif_trainer,
)


class TestRLAIFConfig:
    """测试配置"""

    def test_valid_config(self):
        config = RLAIFConfig()
        assert config.num_samples == 4
        assert config.temperature == 0.7

    def test_invalid_num_samples(self):
        with pytest.raises(ValueError, match="num_samples必须>=2"):
            RLAIFConfig(num_samples=1)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature必须在"):
            RLAIFConfig(temperature=0)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="preference_threshold必须在"):
            RLAIFConfig(preference_threshold=1.5)


class TestAIFeedbackGenerator:
    """测试AI反馈生成器"""

    @pytest.fixture
    def config(self):
        return RLAIFConfig()

    @pytest.fixture
    def generator(self, config):
        return AIFeedbackGenerator(config)

    def test_generate_preferences(self, generator):
        responses = ["回答1", "回答2", "回答3"]
        chosen, rejected = generator.generate_preferences("问题", responses)
        assert 0 <= chosen < len(responses)
        assert 0 <= rejected < len(responses)
        assert chosen != rejected
        assert len(generator.feedback_history) == 1

    def test_generate_preferences_too_few(self, generator):
        with pytest.raises(ValueError, match="至少需要2个响应"):
            generator.generate_preferences("问题", ["回答1"])

    def test_generate_critique(self, generator):
        critique = generator.generate_critique("问题", "回答")
        assert isinstance(critique, str)
        assert len(critique) > 0

    def test_generate_score(self, generator):
        score = generator.generate_score("问题", "回答")
        assert 0 <= score <= 1

    def test_batch_generate_preferences(self, generator):
        prompts = ["q1", "q2"]
        responses_list = [["a1", "a2"], ["b1", "b2"]]
        preferences = generator.batch_generate_preferences(prompts, responses_list)
        assert len(preferences) == 2
        assert all(isinstance(p, tuple) for p in preferences)

    def test_batch_mismatched_lengths(self, generator):
        with pytest.raises(ValueError, match="长度必须相同"):
            generator.batch_generate_preferences(["q1"], [["a1", "a2"], ["b1", "b2"]])


class TestRLAIFTrainer:
    """测试RLAIF训练器"""

    @pytest.fixture
    def config(self):
        return RLAIFConfig()

    @pytest.fixture
    def trainer(self, config):
        return RLAIFTrainer(config)

    def test_collect_ai_preferences(self, trainer):
        prompts = ["q1", "q2"]
        data = trainer.collect_ai_preferences(prompts, num_responses_per_prompt=3)
        assert len(data) == 2
        assert all("prompt" in d for d in data)
        assert all("chosen" in d for d in data)
        assert all("rejected" in d for d in data)

    def test_train_reward_model(self, trainer):
        preference_data = [
            {"prompt": "q1", "chosen": "good", "rejected": "bad"},
            {"prompt": "q2", "chosen": "better", "rejected": "worse"},
        ]
        metrics = trainer.train_reward_model(preference_data)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["num_samples"] == 2

    def test_train_policy(self, trainer):
        prompts = ["q1", "q2"]
        metrics = trainer.train_policy(prompts, num_iterations=5)
        assert "avg_reward" in metrics
        assert metrics["num_iterations"] == 5
        assert len(trainer.training_history) == 10

    def test_train_step(self, trainer):
        prompts = ["q1", "q2"]
        responses = ["a1", "a2"]
        metrics = trainer.train_step(prompts, responses)
        assert "avg_reward" in metrics
        assert "num_samples" in metrics
        assert metrics["num_samples"] == 2

    def test_train_step_mismatched(self, trainer):
        with pytest.raises(ValueError, match="长度必须相同"):
            trainer.train_step(["q1"], ["a1", "a2"])


class TestFactoryFunction:
    """测试工厂函数"""

    def test_create_default(self):
        trainer = create_rlaif_trainer()
        assert trainer.config.critic_model == "gpt-4"
        assert trainer.config.num_samples == 4

    def test_create_custom(self):
        trainer = create_rlaif_trainer(
            critic_model="gpt-3.5-turbo",
            num_samples=8,
            temperature=0.9,
        )
        assert trainer.config.critic_model == "gpt-3.5-turbo"
        assert trainer.config.num_samples == 8
        assert trainer.config.temperature == 0.9
