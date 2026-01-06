"""
Short-Term Memory: Conversation Context Management for AI Agents.

Core Idea:
    Short-term memory (STM) maintains immediate conversation context, enabling
    coherent multi-turn dialogues within LLM token constraints. This module
    implements multiple context management strategies inspired by cognitive
    science and production LLM systems.

Mathematical Foundation:
    Context window optimization can be formulated as a constrained selection:

    $$\\max_{S \\subseteq M} \\sum_{m \\in S} w(m) \\cdot \\text{relevance}(m, q)$$
    $$\\text{s.t.} \\quad \\sum_{m \\in S} \\text{tokens}(m) \\leq T_{\\max}$$

    where:
    - $M$: Complete message history
    - $S$: Selected context subset
    - $w(m)$: Importance weight of message $m$
    - $T_{\\max}$: Maximum token budget

Problem Statement:
    LLMs have fixed context windows (4K-128K tokens). As conversations grow,
    we must intelligently select which messages to retain. Naive truncation
    loses critical context; this module provides principled alternatives.

Algorithm Comparison:
    | Strategy      | Space    | Preserves History | Token Control | Use Case          |
    |---------------|----------|-------------------|---------------|-------------------|
    | Buffer        | O(n)     | Full              | None          | Short sessions    |
    | SlidingWindow | O(k)     | Recent only       | Implicit      | Long dialogues    |
    | Summary       | O(k+s)   | Compressed        | Moderate      | Extended sessions |
    | TokenBased    | O(n)     | Prioritized       | Exact         | API optimization  |

Complexity Analysis:
    - Buffer: O(1) add, O(n) retrieval
    - SlidingWindow: O(1) add (amortized), O(k) retrieval
    - Summary: O(s) add (summarization), O(k) retrieval
    - TokenBased: O(n) add (worst case trim), O(n) retrieval

References:
    - MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)
    - LangChain Memory: https://python.langchain.com/docs/modules/memory/
    - Cognitive Load Theory (Sweller, 1988)

Author: AI-Practices
Version: 2.0.0
"""

from __future__ import annotations

import hashlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

__all__ = [
    "MessageRole",
    "Message",
    "TokenCounter",
    "SimpleTokenCounter",
    "TiktokenCounter",
    "Summarizer",
    "SimpleSummarizer",
    "ShortTermMemory",
    "ConversationBuffer",
    "SlidingWindowMemory",
    "SummaryMemory",
    "TokenBasedMemory",
    "create_conversation_memory",
]


# =============================================================================
# Constants and Type Definitions
# =============================================================================

T = TypeVar("T")
DEFAULT_CHARS_PER_TOKEN: Final[float] = 4.0
DEFAULT_IMPORTANCE: Final[float] = 0.5
MIN_WINDOW_SIZE: Final[int] = 1
MAX_CONTENT_PREVIEW: Final[int] = 50


class MessageRole(str, Enum):
    """
    Role of a message participant in conversation.

    Core Idea:
        Standardized roles following OpenAI Chat Completion API format,
        enabling seamless integration with various LLM providers.

    Attributes:
        SYSTEM: System-level instructions defining agent behavior.
        USER: Human user input messages.
        ASSISTANT: AI assistant generated responses.
        TOOL: Tool/function execution results.
    """

    SYSTEM: Final[str] = "system"
    USER: Final[str] = "user"
    ASSISTANT: Final[str] = "assistant"
    TOOL: Final[str] = "tool"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Message Data Structure
# =============================================================================


@dataclass(slots=True)
class Message:
    """
    Atomic unit of conversation with metadata for tracking and prioritization.

    Core Idea:
        Each message encapsulates content, role, and metadata enabling
        intelligent context management through importance scoring and
        token accounting.

    Mathematical Foundation:
        Message importance can be modeled as:
        $$I(m) = \\alpha \\cdot \\text{recency}(m) + \\beta \\cdot \\text{explicit\\_weight}(m)$$

    Attributes:
        role: Participant role (system/user/assistant/tool).
        content: Text content of the message.
        timestamp: UTC creation timestamp.
        metadata: Extensible key-value store for tool calls, etc.
        token_count: Cached token count (lazy computed if None).
        importance: Priority score in [0, 1] for eviction decisions.

    Example:
        >>> msg = Message(
        ...     role=MessageRole.USER,
        ...     content="Explain quantum entanglement",
        ...     importance=0.8
        ... )
        >>> msg.to_dict()
        {'role': 'user', 'content': 'Explain quantum entanglement'}
    """

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None
    importance: float = DEFAULT_IMPORTANCE

    def __post_init__(self) -> None:
        """Validate and normalize message fields."""
        # Coerce string role to enum
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)

        # Validate importance bounds
        if not 0.0 <= self.importance <= 1.0:
            warnings.warn(
                f"Importance {self.importance} outside [0,1], clamping.",
                UserWarning,
                stacklevel=2,
            )
            self.importance = max(0.0, min(1.0, self.importance))

        # Lazy token estimation if not provided
        if self.token_count is None:
            self.token_count = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        """
        Estimate token count using character heuristic.

        Returns:
            Approximate token count (chars / 4 + 1 for safety margin).
        """
        return int(len(self.content) / DEFAULT_CHARS_PER_TOKEN) + 1

    @property
    def id(self) -> str:
        """
        Generate deterministic unique identifier.

        Uses MD5 hash of (role, content, timestamp) for collision resistance.
        """
        hash_input = f"{self.role.value}:{self.content}:{self.timestamp.isoformat()}"
        return f"msg_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"

    def to_dict(self) -> Dict[str, str]:
        """Convert to OpenAI Chat Completion API format."""
        return {"role": self.role.value, "content": self.content}

    def to_full_dict(self) -> Dict[str, Any]:
        """Serialize complete message with all metadata."""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "token_count": self.token_count,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Deserialize message from dictionary.

        Args:
            data: Dictionary with at minimum 'role' and 'content' keys.

        Returns:
            Reconstructed Message instance.
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=timestamp or datetime.utcnow(),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count"),
            importance=data.get("importance", DEFAULT_IMPORTANCE),
        )

    def __str__(self) -> str:
        """Human-readable preview representation."""
        preview = self.content[:MAX_CONTENT_PREVIEW]
        if len(self.content) > MAX_CONTENT_PREVIEW:
            preview += "..."
        return f"[{self.role.value}] {preview}"

    def __repr__(self) -> str:
        return (
            f"Message(role={self.role!r}, content={self.content[:30]!r}..., "
            f"tokens={self.token_count}, importance={self.importance:.2f})"
        )


# =============================================================================
# Token Counting Protocols and Implementations
# =============================================================================


@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for token counting implementations."""

    def count(self, text: str) -> int:
        """Count tokens in text."""
        ...


class SimpleTokenCounter:
    """
    Heuristic token counter using character-to-token ratio.
    
    Complexity: O(1) - uses string length only.
    """

    __slots__ = ("chars_per_token",)

    def __init__(self, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Estimate token count from character length."""
        return int(len(text) / self.chars_per_token) + 1


class TiktokenCounter:
    """Accurate token counter using OpenAI's tiktoken library."""

    __slots__ = ("_encoding", "_model")

    def __init__(self, model: str = "gpt-4") -> None:
        self._model = model
        self._encoding = None

    def _get_encoding(self) -> Any:
        """Lazy load tiktoken encoding."""
        if self._encoding is None:
            try:
                import tiktoken
                self._encoding = tiktoken.encoding_for_model(self._model)
            except ImportError:
                return None
        return self._encoding

    def count(self, text: str) -> int:
        """Count tokens using tiktoken or fallback to heuristic."""
        encoding = self._get_encoding()
        if encoding is None:
            return int(len(text) / DEFAULT_CHARS_PER_TOKEN) + 1
        return len(encoding.encode(text))


# =============================================================================
# Summarization Protocols and Implementations
# =============================================================================


@runtime_checkable
class Summarizer(Protocol):
    """Protocol for message summarization strategies."""

    def summarize(self, messages: List[Message]) -> str:
        """Compress messages into summary text."""
        ...


class SimpleSummarizer:
    """Basic summarizer using message truncation and concatenation."""

    __slots__ = ("max_chars",)

    def __init__(self, max_chars_per_message: int = 100) -> None:
        if max_chars_per_message < 10:
            raise ValueError("max_chars_per_message must be at least 10")
        self.max_chars = max_chars_per_message

    def summarize(self, messages: List[Message]) -> str:
        """Create truncated summary of message sequence."""
        if not messages:
            return ""
        summaries = []
        for msg in messages:
            preview = msg.content[: self.max_chars]
            if len(msg.content) > self.max_chars:
                preview += "..."
            summaries.append(f"[{msg.role.value}]: {preview}")
        return "Previous conversation summary:\n" + "\n".join(summaries)


# =============================================================================
# Abstract Base Class for Short-Term Memory
# =============================================================================


class ShortTermMemory(ABC):
    """Abstract base class defining short-term memory interface."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Add message to memory."""
        pass

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """Retrieve messages for LLM context."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from memory."""
        pass

    @property
    @abstractmethod
    def token_count(self) -> int:
        """Total token count of current context."""
        pass

    def add_user_message(self, content: str, **kwargs: Any) -> Message:
        """Convenience: add user message."""
        msg = Message(role=MessageRole.USER, content=content, **kwargs)
        self.add(msg)
        return msg

    def add_assistant_message(self, content: str, **kwargs: Any) -> Message:
        """Convenience: add assistant message."""
        msg = Message(role=MessageRole.ASSISTANT, content=content, **kwargs)
        self.add(msg)
        return msg

    def add_system_message(self, content: str, **kwargs: Any) -> Message:
        """Convenience: add system message."""
        msg = Message(role=MessageRole.SYSTEM, content=content, **kwargs)
        self.add(msg)
        return msg

    def to_openai_messages(self) -> List[Dict[str, str]]:
        """Convert to OpenAI Chat Completion API format."""
        return [msg.to_dict() for msg in self.get_messages()]


# =============================================================================
# Concrete Memory Implementations
# =============================================================================


class ConversationBuffer(ShortTermMemory):
    """
    Simple buffer keeping all messages without truncation.

    Core Idea:
        Stores complete conversation history. Suitable for short sessions
        where context window is unlikely to be exceeded.

    Complexity:
        - add: O(1) amortized
        - get_messages: O(n) copy
        - token_count: O(n)
    """

    __slots__ = ("_messages", "_system_message")

    def __init__(self, system_message: Optional[str] = None) -> None:
        self._messages: List[Message] = []
        self._system_message = system_message
        if system_message:
            self.add_system_message(system_message)

    def add(self, message: Message) -> None:
        """Append message to buffer."""
        self._messages.append(message)

    def get_messages(self) -> List[Message]:
        """Return copy of all messages."""
        return self._messages.copy()

    def clear(self) -> None:
        """Clear buffer, optionally preserving system message."""
        self._messages.clear()
        if self._system_message:
            self.add_system_message(self._system_message)

    @property
    def token_count(self) -> int:
        """Sum of all message token counts."""
        return sum(msg.token_count or 0 for msg in self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)


class SlidingWindowMemory(ShortTermMemory):
    """
    Fixed-size window retaining only the k most recent messages.

    Core Idea:
        Implements a FIFO queue with maximum capacity. When capacity is
        exceeded, oldest non-system messages are evicted.

    Mathematical Model:
        Given messages $M = [m_1, ..., m_n]$ and window size $k$:
        $$\\text{context} = [m_{n-k+1}, ..., m_n]$$

    Complexity:
        - add: O(1) amortized (occasional O(k) for slice)
        - get_messages: O(k)
    """

    __slots__ = ("_window_size", "_preserve_system", "_messages", "_system_messages")

    def __init__(
        self,
        window_size: int = 10,
        system_message: Optional[str] = None,
        preserve_system: bool = True,
    ) -> None:
        if window_size < MIN_WINDOW_SIZE:
            raise ValueError(f"window_size must be at least {MIN_WINDOW_SIZE}")
        self._window_size = window_size
        self._preserve_system = preserve_system
        self._messages: List[Message] = []
        self._system_messages: List[Message] = []
        if system_message:
            self.add_system_message(system_message)

    def add(self, message: Message) -> None:
        """Add message, evicting oldest if window exceeded."""
        if message.role == MessageRole.SYSTEM and self._preserve_system:
            self._system_messages.append(message)
        else:
            self._messages.append(message)
            if len(self._messages) > self._window_size:
                self._messages = self._messages[-self._window_size:]

    def get_messages(self) -> List[Message]:
        """Return system messages + windowed messages."""
        return self._system_messages + self._messages

    def clear(self) -> None:
        """Clear non-system messages."""
        self._messages.clear()
        if not self._preserve_system:
            self._system_messages.clear()

    @property
    def token_count(self) -> int:
        return sum(msg.token_count or 0 for msg in self.get_messages())

    @property
    def window_size(self) -> int:
        return self._window_size

    def set_window_size(self, size: int) -> None:
        """Dynamically adjust window size."""
        if size < MIN_WINDOW_SIZE:
            raise ValueError(f"window_size must be at least {MIN_WINDOW_SIZE}")
        self._window_size = size
        if len(self._messages) > size:
            self._messages = self._messages[-size:]


class SummaryMemory(ShortTermMemory):
    """
    Memory that compresses old messages into summaries.

    Core Idea:
        When message count exceeds threshold, oldest messages are summarized
        to preserve context while reducing token usage.

    Complexity:
        - add: O(1) normal, O(s) when summarization triggered
        - get_messages: O(k + summaries)
    """

    __slots__ = (
        "_max_messages", "_summarize_count", "_summarizer",
        "_messages", "_summaries", "_system_message"
    )

    def __init__(
        self,
        max_messages: int = 20,
        summarize_count: int = 10,
        system_message: Optional[str] = None,
        summarizer: Optional[Summarizer] = None,
    ) -> None:
        if summarize_count >= max_messages:
            raise ValueError("summarize_count must be less than max_messages")
        self._max_messages = max_messages
        self._summarize_count = summarize_count
        self._summarizer = summarizer or SimpleSummarizer()
        self._messages: List[Message] = []
        self._summaries: List[str] = []
        self._system_message = system_message
        if system_message:
            self.add_system_message(system_message)

    def add(self, message: Message) -> None:
        """Add message, triggering summarization if threshold exceeded."""
        self._messages.append(message)
        non_system = [m for m in self._messages if m.role != MessageRole.SYSTEM]
        if len(non_system) > self._max_messages:
            self._summarize_oldest()

    def _summarize_oldest(self) -> None:
        """Compress oldest messages into summary."""
        system_msgs = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        other_msgs = [m for m in self._messages if m.role != MessageRole.SYSTEM]
        to_summarize = other_msgs[:self._summarize_count]
        to_keep = other_msgs[self._summarize_count:]
        if to_summarize:
            summary = self._summarizer.summarize(to_summarize)
            self._summaries.append(summary)
        self._messages = system_msgs + to_keep

    def get_messages(self) -> List[Message]:
        """Return messages with summary prepended if exists."""
        result = []
        if self._summaries:
            combined = "\n\n".join(self._summaries)
            result.append(Message(
                role=MessageRole.SYSTEM,
                content=f"[Conversation History Summary]\n{combined}",
                importance=0.3,
            ))
        result.extend(self._messages)
        return result

    def clear(self) -> None:
        self._messages.clear()
        self._summaries.clear()
        if self._system_message:
            self.add_system_message(self._system_message)

    @property
    def token_count(self) -> int:
        return sum(msg.token_count or 0 for msg in self.get_messages())

    @property
    def summary_count(self) -> int:
        return len(self._summaries)


class TokenBasedMemory(ShortTermMemory):
    """
    Memory managing context within strict token budget.

    Core Idea:
        Maintains messages within token limit by evicting lowest-priority
        messages when budget exceeded. Supports both recency and importance
        based eviction policies.

    Mathematical Model:
        Optimization: $\\max \\sum_{m \\in S} I(m)$ s.t. $\\sum_{m \\in S} T(m) \\leq B$
        where $I(m)$ is importance, $T(m)$ is token count, $B$ is budget.

    Complexity:
        - add: O(n) worst case (eviction loop)
        - get_messages: O(n)
    """

    __slots__ = (
        "_max_tokens", "_reserve_tokens", "_effective_limit",
        "_token_counter", "_prioritize_recent", "_messages", "_system_message"
    )

    def __init__(
        self,
        max_tokens: int = 4000,
        system_message: Optional[str] = None,
        token_counter: Optional[TokenCounter] = None,
        reserve_tokens: int = 500,
        prioritize_recent: bool = True,
    ) -> None:
        if max_tokens <= reserve_tokens:
            raise ValueError("max_tokens must be greater than reserve_tokens")
        self._max_tokens = max_tokens
        self._reserve_tokens = reserve_tokens
        self._effective_limit = max_tokens - reserve_tokens
        self._token_counter = token_counter or SimpleTokenCounter()
        self._prioritize_recent = prioritize_recent
        self._messages: List[Message] = []
        self._system_message = system_message
        if system_message:
            self.add_system_message(system_message)

    def add(self, message: Message) -> None:
        """Add message, evicting low-priority messages if budget exceeded."""
        if message.token_count is None:
            message.token_count = self._token_counter.count(message.content)
        self._messages.append(message)
        self._trim_to_budget()

    def _trim_to_budget(self) -> None:
        """Evict messages until within token budget."""
        while self.token_count > self._effective_limit and len(self._messages) > 1:
            removable = [
                (i, m) for i, m in enumerate(self._messages)
                if m.role != MessageRole.SYSTEM
            ]
            if not removable:
                break
            if self._prioritize_recent:
                idx, _ = removable[0]
            else:
                idx, _ = min(removable, key=lambda x: x[1].importance)
            self._messages.pop(idx)

    def get_messages(self) -> List[Message]:
        return self._messages.copy()

    def clear(self) -> None:
        self._messages.clear()
        if self._system_message:
            self.add_system_message(self._system_message)

    @property
    def token_count(self) -> int:
        return sum(msg.token_count or 0 for msg in self._messages)

    @property
    def available_tokens(self) -> int:
        return max(0, self._effective_limit - self.token_count)

    @property
    def max_tokens(self) -> int:
        return self._max_tokens


# =============================================================================
# Factory Function
# =============================================================================


def create_conversation_memory(
    strategy: str = "sliding_window",
    **kwargs: Any,
) -> ShortTermMemory:
    """
    Factory function to create memory instances.

    Args:
        strategy: One of 'buffer', 'sliding_window', 'summary', 'token_based'.
        **kwargs: Strategy-specific configuration.

    Returns:
        Configured ShortTermMemory instance.

    Raises:
        ValueError: If strategy name is unknown.

    Example:
        >>> memory = create_conversation_memory("sliding_window", window_size=10)
        >>> memory = create_conversation_memory("token_based", max_tokens=4000)
    """
    strategies: Dict[str, type] = {
        "buffer": ConversationBuffer,
        "sliding_window": SlidingWindowMemory,
        "summary": SummaryMemory,
        "token_based": TokenBasedMemory,
    }
    if strategy not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown strategy: {strategy}. Available: {available}")
    return strategies[strategy](**kwargs)
