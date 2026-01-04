"""
Unit tests for Chain of Thought module.
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

import pytest
from src.chain_of_thought import (
    CoTStrategy,
    CoTStep,
    CoTExample,
    CoTResult,
    CoTPromptBuilder,
    ZeroShotCoT,
    FewShotCoT,
    AutoCoT,
)


class TestCoTStep:
    """Tests for CoTStep dataclass."""

    def test_creation(self):
        step = CoTStep(step_number=1, content="First step", confidence=0.9)
        assert step.step_number == 1
        assert step.content == "First step"
        assert step.confidence == 0.9

    def test_to_dict(self):
        step = CoTStep(step_number=1, content="Test")
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["content"] == "Test"


class TestCoTExample:
    """Tests for CoTExample dataclass."""

    def test_creation(self):
        example = CoTExample(
            question="What is 2+2?",
            reasoning="2+2=4",
            answer="4"
        )
        assert example.question == "What is 2+2?"
        assert example.answer == "4"

    def test_format(self):
        example = CoTExample(
            question="Q",
            reasoning="R",
            answer="A"
        )
        formatted = example.format()
        assert "Q" in formatted
        assert "R" in formatted
        assert "A" in formatted


class TestCoTResult:
    """Tests for CoTResult dataclass."""

    def test_creation(self):
        result = CoTResult(
            question="Test?",
            answer="Yes",
            reasoning="Step 1, Step 2"
        )
        assert result.question == "Test?"
        assert result.answer == "Yes"
        assert result.reasoning == "Step 1, Step 2"

    def test_to_dict(self):
        result = CoTResult(
            question="Q",
            answer="A",
            reasoning="R"
        )
        d = result.to_dict()
        assert d["question"] == "Q"
        assert d["answer"] == "A"


class TestCoTPromptBuilder:
    """Tests for CoTPromptBuilder."""

    def test_build_zero_shot(self):
        builder = CoTPromptBuilder()
        builder.set_question("What is 2+2?")
        prompt = builder.build()
        assert "What is 2+2?" in prompt
        assert "step by step" in prompt.lower()

    def test_build_few_shot(self):
        examples = [
            CoTExample(question="Q1", reasoning="R1", answer="A1"),
        ]
        builder = CoTPromptBuilder()
        builder.add_example(examples[0])
        builder.set_question("New Q")
        prompt = builder.build()
        assert "Q1" in prompt
        assert "New Q" in prompt


class TestZeroShotCoT:
    """Tests for ZeroShotCoT."""

    def test_get_prompt_standard(self):
        cot = ZeroShotCoT(trigger="standard")
        prompt = cot.get_prompt("Test question")
        assert "Test question" in prompt
        assert "step by step" in prompt.lower()

    def test_get_prompt_detailed(self):
        cot = ZeroShotCoT(trigger="detailed")
        prompt = cot.get_prompt("Test")
        assert "step by step" in prompt.lower()

    def test_reason_without_llm(self):
        cot = ZeroShotCoT()
        result = cot.reason("What is 5+5?")
        assert result.question == "What is 5+5?"


class TestFewShotCoT:
    """Tests for FewShotCoT."""

    def test_creation_with_examples(self):
        examples = [
            CoTExample(question="Q1", reasoning="R1", answer="A1"),
            CoTExample(question="Q2", reasoning="R2", answer="A2"),
        ]
        cot = FewShotCoT(examples=examples)
        assert len(cot._examples) == 2

    def test_get_prompt(self):
        examples = [
            CoTExample(question="Q1", reasoning="R1", answer="A1"),
        ]
        cot = FewShotCoT(examples=examples)
        prompt = cot.get_prompt("New question")
        assert "Q1" in prompt
        assert "New question" in prompt

    def test_reason_without_llm(self):
        examples = [
            CoTExample(question="Q1", reasoning="R1", answer="A1"),
        ]
        cot = FewShotCoT(examples=examples)
        result = cot.reason("Test?")
        assert result.question == "Test?"


class TestAutoCoT:
    """Tests for AutoCoT."""

    def test_creation(self):
        questions = ["Q1", "Q2", "Q3"]
        auto = AutoCoT(question_pool=questions, n_demonstrations=2)
        assert len(auto._question_pool) == 3
        assert auto._n_demonstrations == 2


class TestCoTStrategy:
    """Tests for CoTStrategy enum."""

    def test_values(self):
        assert CoTStrategy.ZERO_SHOT.value == "zero_shot"
        assert CoTStrategy.FEW_SHOT.value == "few_shot"
        assert CoTStrategy.AUTO_COT.value == "auto_cot"
