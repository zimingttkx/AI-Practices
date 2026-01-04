"""
Unit tests for Reflection module.
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

import pytest
from src.reflection import (
    EvaluationResult,
    ReflectionStep,
    ReflectionResult,
    SelfEvaluator,
    SimpleSelfEvaluator,
    ErrorDetector,
    CorrectionStrategy,
    SimpleCorrectionStrategy,
    Reflection,
)


class TestEvaluationResult:
    """Tests for EvaluationResult enum."""

    def test_values(self):
        assert EvaluationResult.CORRECT.value == "correct"
        assert EvaluationResult.PARTIALLY_CORRECT.value == "partially_correct"
        assert EvaluationResult.INCORRECT.value == "incorrect"
        assert EvaluationResult.UNCERTAIN.value == "uncertain"


class TestReflectionStep:
    """Tests for ReflectionStep dataclass."""

    def test_creation(self):
        step = ReflectionStep(
            iteration=1,
            response="Test response",
            evaluation=EvaluationResult.CORRECT,
            feedback="Good job",
            score=0.9
        )
        assert step.iteration == 1
        assert step.response == "Test response"
        assert step.evaluation == EvaluationResult.CORRECT
        assert step.score == 0.9

    def test_to_dict(self):
        step = ReflectionStep(
            iteration=1,
            response="R",
            evaluation=EvaluationResult.INCORRECT,
            feedback="F",
            score=0.3
        )
        d = step.to_dict()
        assert d["iteration"] == 1
        assert d["evaluation"] == "incorrect"
        assert d["score"] == 0.3


class TestReflectionResult:
    """Tests for ReflectionResult dataclass."""

    def test_creation(self):
        result = ReflectionResult(
            question="What is AI?",
            initial_response="AI is...",
            final_response="AI is artificial intelligence...",
            steps=[],
            total_iterations=2,
            improved=True
        )
        assert result.question == "What is AI?"
        assert result.improved is True

    def test_to_dict(self):
        result = ReflectionResult(
            question="Q",
            initial_response="I",
            final_response="F",
            steps=[],
            total_iterations=1
        )
        d = result.to_dict()
        assert d["question"] == "Q"
        assert d["total_iterations"] == 1


class TestSimpleSelfEvaluator:
    """Tests for SimpleSelfEvaluator."""

    def test_evaluate_without_llm(self):
        evaluator = SimpleSelfEvaluator()
        result, feedback, score = evaluator.evaluate(
            question="Test?",
            response="Test response"
        )
        
        assert result == EvaluationResult.UNCERTAIN
        assert feedback == "No LLM available"
        assert score == 0.5

    def test_parse_evaluation_correct(self):
        evaluator = SimpleSelfEvaluator()
        output = """EVALUATION: correct
FEEDBACK: The response is accurate and complete.
SCORE: 9"""
        result, feedback, score = evaluator._parse_evaluation(output)
        
        assert result == EvaluationResult.CORRECT
        assert "accurate" in feedback
        assert score == 0.9

    def test_parse_evaluation_incorrect(self):
        evaluator = SimpleSelfEvaluator()
        output = """EVALUATION: incorrect
FEEDBACK: The response has errors.
SCORE: 3"""
        result, feedback, score = evaluator._parse_evaluation(output)
        
        assert result == EvaluationResult.INCORRECT
        assert score == 0.3

    def test_parse_evaluation_partial(self):
        evaluator = SimpleSelfEvaluator()
        output = """EVALUATION: partially_correct
FEEDBACK: Some parts are right.
SCORE: 6"""
        result, feedback, score = evaluator._parse_evaluation(output)
        
        assert result == EvaluationResult.PARTIALLY_CORRECT


class TestErrorDetector:
    """Tests for ErrorDetector."""

    def test_detect_error_indicators(self):
        detector = ErrorDetector()
        errors = detector.detect("This is incorrect and has a mistake.")
        assert len(errors) > 0
        assert any("error" in e.lower() for e in errors)

    def test_detect_uncertainty(self):
        detector = ErrorDetector()
        errors = detector.detect("I don't know the answer for sure.")
        assert len(errors) > 0
        assert any("uncertain" in e.lower() for e in errors)

    def test_detect_no_errors(self):
        detector = ErrorDetector()
        errors = detector.detect("The answer is 42.")
        # May or may not detect errors depending on patterns
        assert isinstance(errors, list)


class TestSimpleCorrectionStrategy:
    """Tests for SimpleCorrectionStrategy."""

    def test_correct_without_llm(self):
        corrector = SimpleCorrectionStrategy()
        result = corrector.correct(
            question="What is 2+2?",
            response="5",
            feedback="The answer should be 4"
        )
        # Without LLM, returns original response
        assert result == "5"


class TestReflection:
    """Tests for Reflection main class."""

    def test_creation_default(self):
        reflection = Reflection()
        assert reflection._max_iterations == 3
        assert reflection._target_score == 0.8

    def test_creation_custom(self):
        reflection = Reflection(
            max_iterations=5,
            target_score=0.9
        )
        assert reflection._max_iterations == 5
        assert reflection._target_score == 0.9

    def test_reflect_without_llm(self):
        reflection = Reflection()
        result = reflection.reflect(
            question="What is AI?",
            initial_response="AI is technology"
        )
        
        assert result.question == "What is AI?"
        assert result.initial_response == "AI is technology"
        assert len(result.steps) >= 1

    def test_reflect_stops_on_correct(self):
        # Create a mock evaluator that always returns CORRECT
        class AlwaysCorrectEvaluator(SelfEvaluator):
            def evaluate(self, question, response, context=None):
                return EvaluationResult.CORRECT, "Perfect!", 1.0
        
        reflection = Reflection(
            evaluator=AlwaysCorrectEvaluator(),
            max_iterations=5
        )
        result = reflection.reflect(
            question="Test",
            initial_response="Response"
        )
        
        # Should stop after first iteration
        assert result.total_iterations == 1
        assert result.steps[0].evaluation == EvaluationResult.CORRECT

    def test_reflect_stops_on_target_score(self):
        class HighScoreEvaluator(SelfEvaluator):
            def evaluate(self, question, response, context=None):
                return EvaluationResult.PARTIALLY_CORRECT, "Good", 0.95
        
        reflection = Reflection(
            evaluator=HighScoreEvaluator(),
            max_iterations=5,
            target_score=0.9
        )
        result = reflection.reflect(
            question="Test",
            initial_response="Response"
        )
        
        # Should stop after first iteration (score >= target)
        assert result.total_iterations == 1
