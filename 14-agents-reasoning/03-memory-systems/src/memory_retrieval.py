"""
Memory Retrieval: Intelligent Memory Search Strategies for AI Agents

Core Idea:
    Memory retrieval combines multiple signals (relevance, recency, importance)
    to find the most useful memories for a given context.

Mathematical Foundation:
    Hybrid retrieval score:
    
    $$\\text{score}(m) = \\alpha \\cdot \\text{sim}(q, m) + \\beta \\cdot \\text{recency}(m) + \\gamma \\cdot \\text{importance}(m)$$
    
    Time decay function:
    $$\\text{recency}(m) = e^{-\\lambda (t - t_m)}$$

Strategies:
    | Strategy    | Description                    | Use Case              |
    |-------------|--------------------------------|-----------------------|
    | Similarity  | Semantic similarity only       | Knowledge retrieval   |
    | Recency     | Time-based decay               | Recent context        |
    | Importance  | Priority-based                 | Critical information  |
    | Hybrid      | Weighted combination           | General purpose       |

References:
    - Generative Agents: Interactive Simulacra (Park et al., 2023)
    - MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

try:
    from .long_term_memory import MemoryEntry, MemoryType
except ImportError:
    from long_term_memory import MemoryEntry, MemoryType


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation.
    
    Attributes:
        entry: The retrieved memory entry.
        score: Combined retrieval score.
        similarity: Semantic similarity score.
        recency: Recency score (time decay).
        importance: Importance score.
    """
    entry: MemoryEntry
    score: float
    similarity: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "score": self.score,
            "similarity": self.similarity,
            "recency": self.recency,
            "importance": self.importance,
        }


class TimeDecay:
    """Time decay functions for recency scoring.
    
    Implements exponential decay:
    $$\\text{recency}(t) = e^{-\\lambda t}$$
    """
    
    def __init__(
        self,
        decay_rate: float = 0.01,
        half_life_hours: Optional[float] = None,
    ) -> None:
        if half_life_hours:
            self.decay_rate = math.log(2) / (half_life_hours * 3600)
        else:
            self.decay_rate = decay_rate
    
    def compute(self, timestamp: datetime, reference: Optional[datetime] = None) -> float:
        """Compute recency score with exponential decay."""
        reference = reference or datetime.now()
        delta = (reference - timestamp).total_seconds()
        if delta < 0:
            return 1.0
        return math.exp(-self.decay_rate * delta)
    
    @classmethod
    def no_decay(cls) -> "TimeDecay":
        """Create a decay function that always returns 1.0."""
        return cls(decay_rate=0.0)
    
    @classmethod
    def fast_decay(cls) -> "TimeDecay":
        """Create fast decay (half-life: 1 hour)."""
        return cls(half_life_hours=1.0)
    
    @classmethod
    def slow_decay(cls) -> "TimeDecay":
        """Create slow decay (half-life: 24 hours)."""
        return cls(half_life_hours=24.0)


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""
    
    @abstractmethod
    def score(
        self,
        entry: MemoryEntry,
        query_embedding: Optional[List[float]] = None,
        similarity: float = 0.0,
    ) -> RetrievalResult:
        """Score a memory entry."""
        pass
    
    @abstractmethod
    def rank(
        self,
        entries: List[Tuple[MemoryEntry, float]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """Rank a list of entries."""
        pass


class SimilarityRetrieval(RetrievalStrategy):
    """Retrieval based purely on semantic similarity."""
    
    def score(
        self,
        entry: MemoryEntry,
        query_embedding: Optional[List[float]] = None,
        similarity: float = 0.0,
    ) -> RetrievalResult:
        return RetrievalResult(
            entry=entry,
            score=similarity,
            similarity=similarity,
        )
    
    def rank(
        self,
        entries: List[Tuple[MemoryEntry, float]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        results = [self.score(e, similarity=s) for e, s in entries]
        results.sort(key=lambda r: r.score, reverse=True)
        return results


class RecencyRetrieval(RetrievalStrategy):
    """Retrieval based on time decay."""
    
    def __init__(self, time_decay: Optional[TimeDecay] = None) -> None:
        self.time_decay = time_decay or TimeDecay.slow_decay()
    
    def score(
        self,
        entry: MemoryEntry,
        query_embedding: Optional[List[float]] = None,
        similarity: float = 0.0,
    ) -> RetrievalResult:
        recency = self.time_decay.compute(entry.timestamp)
        return RetrievalResult(
            entry=entry,
            score=recency,
            recency=recency,
        )
    
    def rank(
        self,
        entries: List[Tuple[MemoryEntry, float]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        results = [self.score(e) for e, _ in entries]
        results.sort(key=lambda r: r.score, reverse=True)
        return results


class ImportanceRetrieval(RetrievalStrategy):
    """Retrieval based on importance scores."""
    
    def score(
        self,
        entry: MemoryEntry,
        query_embedding: Optional[List[float]] = None,
        similarity: float = 0.0,
    ) -> RetrievalResult:
        return RetrievalResult(
            entry=entry,
            score=entry.importance,
            importance=entry.importance,
        )
    
    def rank(
        self,
        entries: List[Tuple[MemoryEntry, float]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        results = [self.score(e) for e, _ in entries]
        results.sort(key=lambda r: r.score, reverse=True)
        return results


class HybridRetrieval(RetrievalStrategy):
    """Hybrid retrieval combining similarity, recency, and importance.
    
    Score formula:
    $$score = \\alpha \\cdot similarity + \\beta \\cdot recency + \\gamma \\cdot importance$$
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        time_decay: Optional[TimeDecay] = None,
        normalize: bool = True,
    ) -> None:
        if normalize:
            total = alpha + beta + gamma
            self.alpha = alpha / total
            self.beta = beta / total
            self.gamma = gamma / total
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
        self.time_decay = time_decay or TimeDecay.slow_decay()
    
    def score(
        self,
        entry: MemoryEntry,
        query_embedding: Optional[List[float]] = None,
        similarity: float = 0.0,
    ) -> RetrievalResult:
        recency = self.time_decay.compute(entry.timestamp)
        importance = entry.importance
        combined = (
            self.alpha * similarity +
            self.beta * recency +
            self.gamma * importance
        )
        return RetrievalResult(
            entry=entry,
            score=combined,
            similarity=similarity,
            recency=recency,
            importance=importance,
        )
    
    def rank(
        self,
        entries: List[Tuple[MemoryEntry, float]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        results = [self.score(e, similarity=s) for e, s in entries]
        results.sort(key=lambda r: r.score, reverse=True)
        return results


class MemoryRetriever:
    """High-level memory retrieval interface.
    
    Example:
        >>> from src.long_term_memory import LongTermMemory
        >>> ltm = LongTermMemory()
        >>> retriever = MemoryRetriever(ltm, strategy=HybridRetrieval())
        >>> results = retriever.retrieve("user preferences", k=5)
    """
    
    def __init__(
        self,
        memory: Any,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> None:
        self.memory = memory
        self.strategy = strategy or HybridRetrieval()
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        memory_type: Optional[MemoryType] = None,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve memories using configured strategy."""
        raw_results = self.memory.recall(query, k=k * 2, memory_type=memory_type)
        ranked = self.strategy.rank(raw_results)
        return [r for r in ranked if r.score >= min_score][:k]
    
    def retrieve_recent(self, k: int = 5, hours: float = 24.0) -> List[RetrievalResult]:
        """Retrieve recent memories within time window."""
        cutoff = datetime.now() - timedelta(hours=hours)
        all_entries = self.memory.list_all()
        recent = [(e, 1.0) for e in all_entries if e.timestamp >= cutoff]
        recency_strategy = RecencyRetrieval()
        return recency_strategy.rank(recent)[:k]
    
    def retrieve_important(self, k: int = 5, min_importance: float = 0.7) -> List[RetrievalResult]:
        """Retrieve high-importance memories."""
        entries = self.memory.list_all(min_importance=min_importance)
        importance_strategy = ImportanceRetrieval()
        return importance_strategy.rank([(e, 0.0) for e in entries])[:k]


__all__ = [
    "RetrievalResult",
    "TimeDecay",
    "RetrievalStrategy",
    "SimilarityRetrieval",
    "RecencyRetrieval",
    "ImportanceRetrieval",
    "HybridRetrieval",
    "MemoryRetriever",
]
