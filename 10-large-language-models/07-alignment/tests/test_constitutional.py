"""
Constitutional AI 测试
"""

import pytest
from src.constitutional import (
    ConstitutionalPrinciple,
    ConstitutionalConfig,
    ConstitutionalAI,
    SelfCriticTrainer,
    ConstitutionalBatch,
    RevisionBatch,
    create_constitutional_ai,
    DEFAULT_PRINCIPLES,
)


class TestConstitutionalPrinciple:
    """测试宪法原则"""

    def test_valid_principle(self):
        principle = ConstitutionalPrinciple(
            name="test",
            critique_request="critique",
            revision_request="revise",
        )
        assert principle.name == "test"
        assert principle.weight == 1.0

    def test_empty_name(self):
        with pytest.raises(ValueError, match="原则名称不能为空"):
            ConstitutionalPrinciple(name="", critique_request="c", revision_request="r")

    def test_invalid_weight(self):
        with pytest.raises(ValueError, match="权重必须为正数"):
            ConstitutionalPrinciple(
                name="test", critique_request="c", revision_request="r", weight=-1.0
            )


class TestConstitutionalConfig:
    """测试配置"""

    def test_valid_config(self):
        config = ConstitutionalConfig(principles=DEFAULT_PRINCIPLES)
        assert len(config.principles) == 3
        assert config.max_revisions == 3

    def test_empty_principles(self):
        with pytest.raises(ValueError, match="必须至少定义一个原则"):
            ConstitutionalConfig(principles=[])

    def test_invalid_max_revisions(self):
        with pytest.raises(ValueError, match="max_revisions必须为正数"):
            ConstitutionalConfig(principles=DEFAULT_PRINCIPLES, max_revisions=0)


class TestConstitutionalBatch:
    """测试批次"""

    def test_valid_batch(self):
        batch = ConstitutionalBatch(
            prompts=["q1", "q2"],
            responses=["a1", "a2"],
            principles=DEFAULT_PRINCIPLES,
        )
        assert len(batch) == 2

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="长度必须相同"):
            ConstitutionalBatch(
                prompts=["q1"],
                responses=["a1", "a2"],
                principles=DEFAULT_PRINCIPLES,
            )


class TestRevisionBatch:
    """测试修订批次"""

    def test_valid_batch(self):
        batch = RevisionBatch(
            prompts=["q"],
            original_responses=["o"],
            critiques=["c"],
            revised_responses=["r"],
        )
        assert len(batch) == 1

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="所有列表长度必须相同"):
            RevisionBatch(
                prompts=["q1", "q2"],
                original_responses=["o"],
                critiques=["c"],
                revised_responses=["r"],
            )


class TestConstitutionalAI:
    """测试 Constitutional AI"""

    @pytest.fixture
    def config(self):
        return ConstitutionalConfig(principles=DEFAULT_PRINCIPLES)

    @pytest.fixture
    def cai(self, config):
        return ConstitutionalAI(config)

    def test_critique(self, cai):
        critique = cai.critique("问题", "回答")
        assert isinstance(critique, str)
        assert len(cai.critique_history) == 1

    def test_revise(self, cai):
        revised = cai.revise("问题", "回答", "批评")
        assert isinstance(revised, str)
        assert len(cai.revision_history) == 1

    def test_critique_and_revise(self, cai):
        critique, revised = cai.critique_and_revise("问题", "回答")
        assert isinstance(critique, str)
        assert isinstance(revised, str)
        assert len(cai.critique_history) == 1
        assert len(cai.revision_history) == 1

    def test_iterative_revision(self, cai):
        revisions = cai.iterative_revision("问题", "回答")
        assert isinstance(revisions, list)
        assert len(revisions) >= 1

    def test_train_step(self, cai):
        batch = ConstitutionalBatch(
            prompts=["q1", "q2"],
            responses=["a1", "a2"],
            principles=DEFAULT_PRINCIPLES,
        )
        metrics = cai.train_step(batch)
        assert "loss" in metrics
        assert "num_revisions" in metrics
        assert metrics["num_revisions"] == 2


class TestSelfCriticTrainer:
    """测试自我批评训练器"""

    @pytest.fixture
    def config(self):
        return ConstitutionalConfig(principles=DEFAULT_PRINCIPLES)

    @pytest.fixture
    def trainer(self, config):
        return SelfCriticTrainer(config)

    def test_generate_critiques(self, trainer):
        critiques = trainer.generate_critiques(["a1", "a2"], ["q1", "q2"])
        assert len(critiques) == 2

    def test_generate_revisions(self, trainer):
        revisions = trainer.generate_revisions(["a1"], ["c1"], ["q1"])
        assert len(revisions) == 1

    def test_train_on_revisions(self, trainer):
        batch = RevisionBatch(
            prompts=["q"],
            original_responses=["o"],
            critiques=["c"],
            revised_responses=["r"],
        )
        metrics = trainer.train_on_revisions(batch)
        assert "loss" in metrics
        assert "avg_improvement" in metrics
        assert len(trainer.training_history) == 1

    def test_collect_revision_data(self, trainer):
        batch = trainer.collect_revision_data(["q1", "q2"], ["a1", "a2"])
        assert isinstance(batch, RevisionBatch)
        assert len(batch) == 2


class TestFactoryFunction:
    """测试工厂函数"""

    def test_create_default(self):
        cai = create_constitutional_ai()
        assert len(cai.config.principles) == 3

    def test_create_custom(self):
        principles = [
            ConstitutionalPrinciple(name="test", critique_request="c", revision_request="r")
        ]
        cai = create_constitutional_ai(principles=principles, max_revisions=5)
        assert len(cai.config.principles) == 1
        assert cai.config.max_revisions == 5
