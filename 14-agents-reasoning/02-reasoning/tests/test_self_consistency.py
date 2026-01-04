"""
Unit tests for Self-Consistency module.
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

import pytest
from src.self_consistency import (
    SampledPath,
    ConsistencyResult,
    VotingStrategy,
    MajorityVoting,
    WeightedVoting,
    SelfConsistency,
)


class TestSampledPath:
    """Tests for SampledPath dataclass."""

    def test_creation(self):
        path = SampledPath(
            reasoning="Step 1, Step 2",
            answer="42",
            confidence=0.9
        )
        assert path.reasoning == "Step 1, Step 2"
        assert path.answer == "42"
        assert path.confidence == 0.9

    def test_default_confidence(self):
        path = SampledPath(reasoning="R", answer="A")
        assert path.confidence == 1.0


class TestConsistencyResult:
    """Tests for ConsistencyResult dataclass."""

    def test_creation(self):
        result = ConsistencyResult(
            question="What is 2+2?",
            final_answer="4",
            confidence=0.8,
            vote_counts={"4": 4, "5": 1},
            paths=[],
            agreement_ratio=0.8
        )
        assert result.question == "What is 2+2?"
        assert result.final_answer == "4"
        assert result.confidence == 0.8

    def test_to_dict(self):
        result = ConsistencyResult(
            question="Q",
            final_answer="A",
            confidence=0.9,
            vote_counts={"A": 5},
            paths=[],
            agreement_ratio=1.0
        )
        d = result.to_dict()
        assert d["question"] == "Q"
        assert d["final_answer"] == "A"
        assert d["confidence"] == 0.9


class TestMajorityVoting:
    """Tests for MajorityVoting strategy."""

    def test_vote_single_winner(self):
        voting = MajorityVoting()
        paths = [
            SampledPath(reasoning="R1", answer="A"),
            SampledPath(reasoning="R2", answer="A"),
            SampledPath(reasoning="R3", answer="B"),
        ]
        winner, confidence = voting.vote(paths)
        assert winner == "A"
        assert confidence == pytest.approx(2/3)

    def test_vote_unanimous(self):
        voting = MajorityVoting()
        paths = [
            SampledPath(reasoning="R1", answer="X"),
            SampledPath(reasoning="R2", answer="X"),
            SampledPath(reasoning="R3", answer="X"),
        ]
        winner, confidence = voting.vote(paths)
        assert winner == "X"
        assert confidence == 1.0

    def test_vote_empty(self):
        voting = MajorityVoting()
        winner, confidence = voting.vote([])
        assert winner == ""
        assert confidence == 0.0

    def test_vote_tie(self):
        voting = MajorityVoting()
        paths = [
            SampledPath(reasoning="R1", answer="A"),
            SampledPath(reasoning="R2", answer="B"),
        ]
        winner, confidence = voting.vote(paths)
        # One of them wins (implementation dependent)
        assert winner in ["A", "B"]
        assert confidence == 0.5


class TestWeightedVoting:
    """Tests for WeightedVoting strategy."""

    def test_vote_weighted(self):
        voting = WeightedVoting()
        paths = [
            SampledPath(reasoning="R1", answer="A", confidence=0.9),
            SampledPath(reasoning="R2", answer="B", confidence=0.1),
        ]
        winner, confidence = voting.vote(paths)
        assert winner == "A"
        assert confidence == pytest.approx(0.9)

    def test_vote_weighted_override(self):
        voting = WeightedVoting()
        paths = [
            SampledPath(reasoning="R1", answer="A", confidence=0.3),
            SampledPath(reasoning="R2", answer="A", confidence=0.3),
            SampledPath(reasoning="R3", answer="B", confidence=0.9),
        ]
        winner, confidence = voting.vote(paths)
        # B has higher total weight (0.9) vs A (0.6)
        assert winner == "B"

    def test_vote_empty(self):
        voting = WeightedVoting()
        winner, confidence = voting.vote([])
        assert winner == ""
        assert confidence == 0.0


class TestSelfConsistency:
    """Tests for SelfConsistency main class."""

    def test_creation_default(self):
        sc = SelfConsistency()
        assert sc._n_samples == 5
        assert isinstance(sc._voting, MajorityVoting)

    def test_creation_custom(self):
        sc = SelfConsistency(
            n_samples=10,
            voting_strategy=WeightedVoting(),
            temperature=0.5
        )
        assert sc._n_samples == 10
        assert isinstance(sc._voting, WeightedVoting)
        assert sc._temperature == 0.5

    def test_reason_without_llm(self):
        sc = SelfConsistency()
        result = sc.reason("What is 2+2?")
        
        assert result.question == "What is 2+2?"
        # Without LLM, returns empty result
        assert result.final_answer == ""

    def test_extract_answer_with_pattern(self):
        sc = SelfConsistency()
        
        # Test "answer is" pattern
        response = "After calculation, the answer is 42."
        answer = sc._extract_answer(response, None)
        assert "42" in answer

    def test_extract_answer_with_custom_extractor(self):
        sc = SelfConsistency()
        
        def custom_extractor(text):
            return text.split()[-1]
        
        response = "The result is FINAL"
        answer = sc._extract_answer(response, custom_extractor)
        assert answer == "FINAL"

    def test_extract_answer_fallback(self):
        sc = SelfConsistency()
        response = "Line 1\nLine 2\nFinal line"
        answer = sc._extract_answer(response, None)
        assert answer == "Final line"
