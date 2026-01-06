"""
Long-Term Memory: Persistent Knowledge Storage for AI Agents.

Core Idea:
    Long-term memory (LTM) provides persistent storage for information that
    needs to be retained across conversations. It uses vector embeddings for
    semantic similarity search, enabling retrieval of relevant memories based
    on meaning rather than exact keyword matching.

Mathematical Foundation:
    Semantic similarity is computed using cosine distance in embedding space:

    $$\\text{sim}(q, m) = \\frac{\\mathbf{q} \\cdot \\mathbf{m}}{\\|\\mathbf{q}\\| \\|\\mathbf{m}\\|} = \\frac{\\sum_{i=1}^{d} q_i m_i}{\\sqrt{\\sum_{i=1}^{d} q_i^2} \\sqrt{\\sum_{i=1}^{d} m_i^2}}$$

    where:
    - $\\mathbf{q} \\in \\mathbb{R}^d$: Query embedding vector
    - $\\mathbf{m} \\in \\mathbb{R}^d$: Memory embedding vector
    - $d$: Embedding dimension

Problem Statement:
    AI agents need to remember facts, preferences, and past interactions
    across sessions. Traditional databases require exact matches; LTM enables
    semantic retrieval where "user likes dark themes" matches queries about
    "interface preferences" or "UI settings".

Algorithm Comparison:
    | Method          | Complexity | Scalability | Accuracy | Use Case        |
    |-----------------|------------|-------------|----------|-----------------|
    | Brute Force     | O(n·d)     | <10K        | Exact    | Small datasets  |
    | FAISS IVF       | O(√n·d)    | Millions    | ~95%     | Production      |
    | HNSW            | O(log n·d) | Millions    | ~95%     | Low latency     |
    | LSH             | O(1)       | Billions    | ~80%     | Massive scale   |

Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Long-Term Memory System                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  Input Text ──► Embedding ──► Vector Store ──► Retrieval        │
    │       │            │              │               │              │
    │       ▼            ▼              ▼               ▼              │
    │  "User likes"  [0.2, -0.1,   FAISS/Chroma    Top-K Similar     │
    │   dark mode"    0.8, ...]     Index          Memories           │
    └─────────────────────────────────────────────────────────────────┘
    ```

References:
    - RAG: Retrieval-Augmented Generation (Lewis et al., 2020)
    - FAISS: Billion-scale similarity search (Johnson et al., 2019)
    - Sentence-BERT: Sentence Embeddings (Reimers & Gurevych, 2019)

Author: AI-Practices
Version: 2.0.0
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

__all__ = [
    "MemoryType",
    "MemoryEntry",
    "EmbeddingFunction",
    "SimpleEmbedding",
    "VectorStore",
    "InMemoryVectorStore",
    "LongTermMemory",
]


# =============================================================================
# Constants
# =============================================================================

DEFAULT_EMBEDDING_DIM: Final[int] = 128
DEFAULT_IMPORTANCE: Final[float] = 0.5
MIN_IMPORTANCE: Final[float] = 0.0
MAX_IMPORTANCE: Final[float] = 1.0


# =============================================================================
# Memory Type Enumeration
# =============================================================================


class MemoryType(str, Enum):
    """
    Categorical classification of memory entries.

    Core Idea:
        Different memory types enable filtered retrieval and specialized
        handling. Inspired by cognitive science distinctions between
        semantic, episodic, and procedural memory.

    Attributes:
        FACT: Declarative facts ("User's name is Alice")
        EVENT: Temporal events ("User purchased item on Jan 1")
        PREFERENCE: User preferences ("User prefers dark mode")
        KNOWLEDGE: Domain knowledge ("Python is interpreted")
        CONVERSATION: Important dialogue excerpts
        TASK: Pending tasks and reminders
    """

    FACT: Final[str] = "fact"
    EVENT: Final[str] = "event"
    PREFERENCE: Final[str] = "preference"
    KNOWLEDGE: Final[str] = "knowledge"
    CONVERSATION: Final[str] = "conversation"
    TASK: Final[str] = "task"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Memory Entry Data Structure
# =============================================================================


@dataclass
class MemoryEntry:
    """
    Single entry in long-term memory with embedding and metadata.

    Core Idea:
        Each memory encapsulates content, vector representation, and
        metadata enabling semantic retrieval and importance-based ranking.

    Mathematical Foundation:
        Memory relevance combines multiple signals:
        $$R(m, q) = \\alpha \\cdot \\text{sim}(m, q) + \\beta \\cdot I(m) + \\gamma \\cdot \\text{recency}(m)$$

    Attributes:
        content: Text content of the memory.
        memory_type: Categorical classification.
        embedding: Dense vector representation (d-dimensional).
        importance: Priority score in [0, 1].
        timestamp: Creation time (UTC).
        last_accessed: Most recent retrieval time.
        access_count: Total retrieval count.
        metadata: Extensible key-value store.
        source: Origin identifier (conversation, document, etc.).

    Example:
        >>> entry = MemoryEntry(
        ...     content="User prefers dark mode interfaces",
        ...     memory_type=MemoryType.PREFERENCE,
        ...     importance=0.8
        ... )
    """

    content: str
    memory_type: MemoryType = MemoryType.KNOWLEDGE
    embedding: Optional[List[float]] = None
    importance: float = DEFAULT_IMPORTANCE
    timestamp: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    def __post_init__(self) -> None:
        """Validate and normalize fields."""
        if isinstance(self.memory_type, str):
            self.memory_type = MemoryType(self.memory_type)

        if not MIN_IMPORTANCE <= self.importance <= MAX_IMPORTANCE:
            warnings.warn(
                f"Importance {self.importance} outside [{MIN_IMPORTANCE}, {MAX_IMPORTANCE}], clamping.",
                UserWarning,
                stacklevel=2,
            )
            self.importance = max(MIN_IMPORTANCE, min(MAX_IMPORTANCE, self.importance))

    @property
    def id(self) -> str:
        """Generate deterministic unique identifier."""
        hash_input = f"{self.content}:{self.timestamp.isoformat()}"
        return f"mem_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"

    def record_access(self) -> None:
        """Record memory retrieval event."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "embedding": self.embedding,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "metadata": self.metadata,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        last_accessed = data.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)

        return cls(
            content=data["content"],
            memory_type=MemoryType(data.get("memory_type", "knowledge")),
            embedding=data.get("embedding"),
            importance=data.get("importance", DEFAULT_IMPORTANCE),
            timestamp=timestamp or datetime.utcnow(),
            last_accessed=last_accessed,
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
            source=data.get("source", "unknown"),
        )

    def __repr__(self) -> str:
        return (
            f"MemoryEntry(type={self.memory_type.value}, "
            f"content={self.content[:30]!r}..., importance={self.importance:.2f})"
        )


# =============================================================================
# Embedding Protocols and Implementations
# =============================================================================


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for text embedding functions."""

    def embed(self, text: str) -> List[float]: ...
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


class SimpleEmbedding:
    """
    Hash-based embedding for demonstration and testing.
    
    Note: Use sentence-transformers or OpenAI embeddings for production.
    Complexity: O(n·d) where n is word count, d is dimension.
    """

    __slots__ = ("dim",)

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        if dim < 1:
            raise ValueError("Embedding dimension must be positive")
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        """Create normalized hash-based embedding."""
        words = text.lower().split()
        embedding = [0.0] * self.dim
        for word in words:
            h = hash(word)
            for i in range(self.dim):
                embedding[i] += ((h >> i) & 1) * 2 - 1
        norm = math.sqrt(sum(x * x for x in embedding)) or 1.0
        return [x / norm for x in embedding]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


# =============================================================================
# Vector Store Abstract Base Class and Implementation
# =============================================================================


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def add(self, entry: MemoryEntry) -> None: pass

    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5,
               filter_fn: Optional[Callable[[MemoryEntry], bool]] = None) -> List[Tuple[MemoryEntry, float]]: pass

    @abstractmethod
    def delete(self, entry_id: str) -> bool: pass

    @abstractmethod
    def get(self, entry_id: str) -> Optional[MemoryEntry]: pass

    @abstractmethod
    def list_all(self) -> List[MemoryEntry]: pass

    @abstractmethod
    def clear(self) -> None: pass

    @property
    @abstractmethod
    def size(self) -> int: pass


class InMemoryVectorStore(VectorStore):
    """In-memory vector store with brute-force O(n) search."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: Dict[str, MemoryEntry] = {}

    def add(self, entry: MemoryEntry) -> None:
        if entry.embedding is None:
            raise ValueError("Entry must have embedding")
        self._entries[entry.id] = entry

    def search(self, query_embedding: List[float], k: int = 5,
               filter_fn: Optional[Callable[[MemoryEntry], bool]] = None) -> List[Tuple[MemoryEntry, float]]:
        results = []
        for entry in self._entries.values():
            if filter_fn and not filter_fn(entry):
                continue
            if entry.embedding is None:
                continue
            sim = self._cosine_similarity(query_embedding, entry.embedding)
            results.append((entry, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(entry_id)

    def list_all(self) -> List[MemoryEntry]:
        return list(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)


# =============================================================================
# Long-Term Memory Main Interface
# =============================================================================


class LongTermMemory:
    """
    High-level interface for long-term memory operations.

    Core Idea:
        Provides unified API for storing, retrieving, and managing persistent
        memories with automatic embedding generation and importance scoring.

    Example:
        >>> ltm = LongTermMemory()
        >>> ltm.store("User prefers dark mode", memory_type=MemoryType.PREFERENCE)
        >>> results = ltm.recall("interface preferences", k=5)
    """

    __slots__ = ("_store", "_embedding_fn", "_auto_importance")

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_fn: Optional[EmbeddingFunction] = None,
        auto_importance: bool = True,
    ) -> None:
        self._store = vector_store or InMemoryVectorStore()
        self._embedding_fn = embedding_fn or SimpleEmbedding()
        self._auto_importance = auto_importance

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.KNOWLEDGE,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "user",
    ) -> MemoryEntry:
        """Store new memory with automatic embedding."""
        if importance is None and self._auto_importance:
            importance = self._estimate_importance(content)

        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance or DEFAULT_IMPORTANCE,
            metadata=metadata or {},
            source=source,
        )
        entry.embedding = self._embedding_fn.embed(content)
        self._store.add(entry)
        return entry

    def recall(
        self,
        query: str,
        k: int = 5,
        memory_type: Optional[MemoryType] = None,
        min_similarity: float = 0.0,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Recall memories similar to query."""
        query_embedding = self._embedding_fn.embed(query)
        filter_fn = (lambda e: e.memory_type == memory_type) if memory_type else None
        results = self._store.search(query_embedding, k=k * 2, filter_fn=filter_fn)
        filtered = [(e, s) for e, s in results if s >= min_similarity][:k]
        for entry, _ in filtered:
            entry.record_access()
        return filtered

    def forget(self, entry_id: str) -> bool:
        """Remove memory by ID."""
        return self._store.delete(entry_id)

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get specific memory by ID."""
        return self._store.get(entry_id)

    def list_all(
        self,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """List all memories with optional filtering."""
        entries = self._store.list_all()
        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        if min_importance > 0:
            entries = [e for e in entries if e.importance >= min_importance]
        return entries

    def clear(self) -> None:
        """Clear all memories."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Number of stored memories."""
        return self._store.size

    def _estimate_importance(self, content: str) -> float:
        """Heuristic importance estimation."""
        score = 0.5
        keywords = ["important", "remember", "always", "never", "must", "critical"]
        content_lower = content.lower()
        for kw in keywords:
            if kw in content_lower:
                score += 0.1
        if len(content) > 200:
            score += 0.1
        if "?" in content:
            score -= 0.1
        return max(MIN_IMPORTANCE, min(MAX_IMPORTANCE, score))

    def save(self, path: Union[str, Path]) -> None:
        """Save memories to JSON file."""
        path = Path(path)
        entries = [e.to_dict() for e in self._store.list_all()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def load(self, path: Union[str, Path]) -> int:
        """Load memories from JSON file. Returns count loaded."""
        path = Path(path)
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        count = 0
        for data in entries:
            entry = MemoryEntry.from_dict(data)
            if entry.embedding is None:
                entry.embedding = self._embedding_fn.embed(entry.content)
            self._store.add(entry)
            count += 1
        return count
