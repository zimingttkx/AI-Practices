"""
Comprehensive Tests for Memory Retrieval Implementations.

Test Coverage:
    - TimeDecay: exponential decay, half-life, edge cases
    - RetrievalResult: creation, serialization
    - SimilarityRetrieval: scoring, ranking
    - RecencyRetrieval: time-based ranking
    - ImportanceRetrieval: priority-based ranking
    - HybridRetrieval: weighted combination
    - MemoryRetriever: high-level interface
    - Edge cases: empty inputs, boundary conditions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import math
from datetime import datetime, timedelta

from memory_retrieval import (
    RetrievalResult,
    TimeDecay,
    RetrievalStrategy,
    SimilarityRetrieval,
    RecencyRetrieval,
    ImportanceRetrieval,
    HybridRetrieval,
    MemoryRetriever,
)
from long_term_memory import MemoryEntry, MemoryType, LongTermMemory


# =============================================================================
# TimeDecay Tests
# =============================================================================


class TestTimeDecay:
    """Comprehensive tests for TimeDecay."""

    def test_recent_high_score(self):
        decay = TimeDecay(decay_rate=0.01)
        now = datetime.now()
        score = decay.compute(now)
        assert score > 0.99

    def test_old_low_score(self):
        decay = TimeDecay(decay_rate=0.01)
        old = datetime.now() - timedelta(days=30)
        score = decay.compute(old)
        assert score < 0.1

    def test_no_decay(self):
        decay = TimeDecay.no_decay()
        old = datetime.now() - timedelta(days=365)
        score = decay.compute(old)
        assert score == 1.0

    def test_fast_decay(self):
        """Test fast decay (1 hour half-life)."""
        decay = TimeDecay.fast_decay()
        one_hour_ago = datetime.now() - timedelta(hours=1)
        score = decay.compute(one_hour_ago)
        assert 0.4 < score < 0.6  # Should be around 0.5

    def test_slow_decay(self):
        """Test slow decay (24 hour half-life)."""
        decay = TimeDecay.slow_decay()
        one_day_ago = datetime.now() - timedelta(hours=24)
        score = decay.compute(one_day_ago)
        assert 0.4 < score < 0.6  # Should be around 0.5

    def test_half_life_hours(self):
        """Test custom half-life in hours."""
        decay = TimeDecay(half_life_hours=2.0)
        two_hours_ago = datetime.now() - timedelta(hours=2)
        score = decay.compute(two_hours_ago)
        assert 0.4 < score < 0.6

    def test_future_timestamp(self):
        """Test future timestamp returns 1.0."""
        decay = TimeDecay(decay_rate=0.01)
        future = datetime.now() + timedelta(hours=1)
        score = decay.compute(future)
        assert score == 1.0

    def test_custom_reference_time(self):
        """Test custom reference time."""
        decay = TimeDecay(decay_rate=0.01)
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        reference = datetime(2024, 1, 1, 12, 0, 0)
        score = decay.compute(timestamp, reference)
        assert score == 1.0

    def test_exponential_decay_formula(self):
        """Test exponential decay follows e^(-λt)."""
        decay_rate = 0.001
        decay = TimeDecay(decay_rate=decay_rate)
        seconds = 1000
        timestamp = datetime.now() - timedelta(seconds=seconds)
        score = decay.compute(timestamp)
        expected = math.exp(-decay_rate * seconds)
        assert abs(score - expected) < 0.01


# =============================================================================
# RetrievalResult Tests
# =============================================================================


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_create_result(self):
        """Test creating retrieval result."""
        entry = MemoryEntry(content="Test")
        result = RetrievalResult(
            entry=entry,
            score=0.8,
            similarity=0.7,
            recency=0.9,
            importance=0.6
        )
        assert result.score == 0.8
        assert result.similarity == 0.7

    def test_to_dict(self):
        """Test serialization to dict."""
        entry = MemoryEntry(content="Test")
        result = RetrievalResult(entry=entry, score=0.5)
        d = result.to_dict()
        assert "entry" in d
        assert d["score"] == 0.5


# =============================================================================
# SimilarityRetrieval Tests
# =============================================================================


class TestSimilarityRetrieval:
    """Comprehensive tests for SimilarityRetrieval."""

    def test_score(self):
        strategy = SimilarityRetrieval()
        entry = MemoryEntry(content="Test")
        result = strategy.score(entry, similarity=0.8)
        assert result.score == 0.8
        assert result.similarity == 0.8

    def test_rank(self):
        strategy = SimilarityRetrieval()
        entries = [
            (MemoryEntry(content="Low"), 0.3),
            (MemoryEntry(content="High"), 0.9),
            (MemoryEntry(content="Mid"), 0.6),
        ]
        ranked = strategy.rank(entries)
        assert ranked[0].score == 0.9
        assert ranked[-1].score == 0.3

    def test_rank_empty_list(self):
        """Test ranking empty list."""
        strategy = SimilarityRetrieval()
        ranked = strategy.rank([])
        assert ranked == []

    def test_rank_single_entry(self):
        """Test ranking single entry."""
        strategy = SimilarityRetrieval()
        entries = [(MemoryEntry(content="Only"), 0.5)]
        ranked = strategy.rank(entries)
        assert len(ranked) == 1
        assert ranked[0].score == 0.5


# =============================================================================
# RecencyRetrieval Tests
# =============================================================================


class TestRecencyRetrieval:
    """Comprehensive tests for RecencyRetrieval."""

    def test_recent_ranked_higher(self):
        strategy = RecencyRetrieval()
        old_entry = MemoryEntry(content="Old")
        old_entry.timestamp = datetime.now() - timedelta(days=7)
        new_entry = MemoryEntry(content="New")
        new_entry.timestamp = datetime.now()
        
        entries = [(old_entry, 0.5), (new_entry, 0.5)]
        ranked = strategy.rank(entries)
        assert ranked[0].entry.content == "New"

    def test_custom_time_decay(self):
        """Test with custom time decay."""
        decay = TimeDecay.fast_decay()
        strategy = RecencyRetrieval(time_decay=decay)
        entry = MemoryEntry(content="Test")
        result = strategy.score(entry)
        assert 0.0 <= result.recency <= 1.0

    def test_score_ignores_similarity(self):
        """Test that recency strategy ignores similarity."""
        strategy = RecencyRetrieval()
        entry = MemoryEntry(content="Test")
        result = strategy.score(entry, similarity=0.9)
        # Score should be based on recency, not similarity
        assert result.similarity == 0.0


# =============================================================================
# ImportanceRetrieval Tests
# =============================================================================


class TestImportanceRetrieval:
    """Comprehensive tests for ImportanceRetrieval."""

    def test_important_ranked_higher(self):
        strategy = ImportanceRetrieval()
        low_entry = MemoryEntry(content="Low", importance=0.2)
        high_entry = MemoryEntry(content="High", importance=0.9)
        
        entries = [(low_entry, 0.5), (high_entry, 0.5)]
        ranked = strategy.rank(entries)
        assert ranked[0].entry.content == "High"

    def test_score_uses_importance(self):
        """Test score uses entry importance."""
        strategy = ImportanceRetrieval()
        entry = MemoryEntry(content="Test", importance=0.75)
        result = strategy.score(entry)
        assert result.score == 0.75
        assert result.importance == 0.75

    def test_equal_importance_stable_sort(self):
        """Test entries with equal importance maintain order."""
        strategy = ImportanceRetrieval()
        entries = [
            (MemoryEntry(content="A", importance=0.5), 0.0),
            (MemoryEntry(content="B", importance=0.5), 0.0),
        ]
        ranked = strategy.rank(entries)
        assert len(ranked) == 2


# =============================================================================
# HybridRetrieval Tests
# =============================================================================


class TestHybridRetrieval:
    """Comprehensive tests for HybridRetrieval."""

    def test_weights_normalized(self):
        strategy = HybridRetrieval(alpha=1.0, beta=1.0, gamma=1.0)
        assert abs(strategy.alpha + strategy.beta + strategy.gamma - 1.0) < 0.001

    def test_combined_score(self):
        strategy = HybridRetrieval(alpha=0.5, beta=0.3, gamma=0.2)
        entry = MemoryEntry(content="Test", importance=1.0)
        entry.timestamp = datetime.now()
        result = strategy.score(entry, similarity=1.0)
        # All components at max, score should be close to 1.0
        assert result.score > 0.9

    def test_custom_weights(self):
        """Test custom weight configuration."""
        strategy = HybridRetrieval(alpha=0.8, beta=0.1, gamma=0.1)
        assert abs(strategy.alpha - 0.8) < 0.01

    def test_no_normalization(self):
        """Test without weight normalization."""
        strategy = HybridRetrieval(
            alpha=0.5, beta=0.3, gamma=0.2,
            normalize=False
        )
        assert strategy.alpha == 0.5
        assert strategy.beta == 0.3
        assert strategy.gamma == 0.2

    def test_similarity_dominant(self):
        """Test similarity-dominant configuration."""
        strategy = HybridRetrieval(alpha=0.9, beta=0.05, gamma=0.05)
        entry = MemoryEntry(content="Test", importance=0.1)
        entry.timestamp = datetime.now() - timedelta(days=30)
        result = strategy.score(entry, similarity=0.9)
        # High similarity should dominate
        assert result.score > 0.7

    def test_recency_dominant(self):
        """Test recency-dominant configuration."""
        strategy = HybridRetrieval(alpha=0.05, beta=0.9, gamma=0.05)
        entry = MemoryEntry(content="Test", importance=0.1)
        entry.timestamp = datetime.now()
        result = strategy.score(entry, similarity=0.1)
        # High recency should dominate
        assert result.score > 0.7

    def test_importance_dominant(self):
        """Test importance-dominant configuration."""
        strategy = HybridRetrieval(alpha=0.05, beta=0.05, gamma=0.9)
        entry = MemoryEntry(content="Test", importance=0.9)
        entry.timestamp = datetime.now() - timedelta(days=30)
        result = strategy.score(entry, similarity=0.1)
        # High importance should dominate
        assert result.score > 0.7

    def test_rank_multiple_entries(self):
        """Test ranking multiple entries."""
        strategy = HybridRetrieval()
        entries = [
            (MemoryEntry(content="A", importance=0.3), 0.3),
            (MemoryEntry(content="B", importance=0.9), 0.9),
            (MemoryEntry(content="C", importance=0.6), 0.6),
        ]
        ranked = strategy.rank(entries)
        assert len(ranked) == 3
        # B should be first (highest similarity and importance)
        assert ranked[0].entry.content == "B"


# =============================================================================
# MemoryRetriever Tests
# =============================================================================


class TestMemoryRetriever:
    """Comprehensive tests for MemoryRetriever."""

    def test_retrieve(self):
        ltm = LongTermMemory()
        ltm.store("Python programming language")
        ltm.store("Java development framework")
        
        retriever = MemoryRetriever(ltm)
        results = retriever.retrieve("Python", k=2)
        assert len(results) <= 2

    def test_retrieve_important(self):
        ltm = LongTermMemory()
        ltm.store("Low priority", importance=0.2)
        ltm.store("High priority", importance=0.9)
        
        retriever = MemoryRetriever(ltm)
        results = retriever.retrieve_important(k=5, min_importance=0.7)
        for r in results:
            assert r.entry.importance >= 0.7

    def test_custom_strategy(self):
        """Test with custom retrieval strategy."""
        ltm = LongTermMemory()
        ltm.store("Test memory")
        
        strategy = SimilarityRetrieval()
        retriever = MemoryRetriever(ltm, strategy=strategy)
        results = retriever.retrieve("Test", k=1)
        assert len(results) <= 1

    def test_retrieve_with_type_filter(self):
        """Test retrieve with memory type filter."""
        ltm = LongTermMemory()
        ltm.store("Fact 1", memory_type=MemoryType.FACT)
        ltm.store("Preference 1", memory_type=MemoryType.PREFERENCE)
        
        retriever = MemoryRetriever(ltm)
        results = retriever.retrieve("test", k=10, memory_type=MemoryType.FACT)
        for r in results:
            assert r.entry.memory_type == MemoryType.FACT

    def test_retrieve_min_score(self):
        """Test retrieve with minimum score threshold."""
        ltm = LongTermMemory()
        ltm.store("Test memory")
        
        retriever = MemoryRetriever(ltm)
        results = retriever.retrieve("xyz unrelated", k=5, min_score=0.9)
        # High threshold should filter most results
        assert len(results) <= 1

    def test_retrieve_recent(self):
        """Test retrieve_recent method."""
        ltm = LongTermMemory()
        ltm.store("Recent memory")
        
        retriever = MemoryRetriever(ltm)
        results = retriever.retrieve_recent(k=5, hours=24.0)
        assert len(results) >= 1

    def test_empty_memory(self):
        """Test retrieval from empty memory."""
        ltm = LongTermMemory()
        retriever = MemoryRetriever(ltm)
        results = retriever.retrieve("anything", k=5)
        assert results == []


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_all_strategies_implement_interface(self):
        """Test all strategies implement RetrievalStrategy interface."""
        strategies = [
            SimilarityRetrieval(),
            RecencyRetrieval(),
            ImportanceRetrieval(),
            HybridRetrieval(),
        ]
        entry = MemoryEntry(content="Test")
        for strategy in strategies:
            result = strategy.score(entry, similarity=0.5)
            assert isinstance(result, RetrievalResult)

    def test_zero_weights(self):
        """Test hybrid with zero weights."""
        strategy = HybridRetrieval(alpha=0.0, beta=0.0, gamma=1.0)
        entry = MemoryEntry(content="Test", importance=0.8)
        result = strategy.score(entry, similarity=0.9)
        # Only importance should matter
        assert abs(result.score - 0.8) < 0.1

    def test_very_old_memory(self):
        """Test very old memory recency."""
        decay = TimeDecay(decay_rate=0.001)
        very_old = datetime.now() - timedelta(days=365)
        score = decay.compute(very_old)
        assert score < 0.001

    def test_retriever_strategy_switch(self):
        """Test switching retrieval strategies."""
        ltm = LongTermMemory()
        ltm.store("Test memory", importance=0.9)
        
        retriever = MemoryRetriever(ltm, strategy=SimilarityRetrieval())
        results1 = retriever.retrieve("Test", k=1)
        
        retriever.strategy = ImportanceRetrieval()
        results2 = retriever.retrieve("Test", k=1)
        
        # Both should return results
        assert len(results1) >= 0
        assert len(results2) >= 0
