"""Pytest fixtures for memory systems tests."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timedelta

from short_term_memory import (
    Message,
    MessageRole,
    ConversationBuffer,
    SlidingWindowMemory,
    SummaryMemory,
    TokenBasedMemory,
)
from long_term_memory import (
    MemoryEntry,
    MemoryType,
    LongTermMemory,
    SimpleEmbedding,
)
from memory_retrieval import (
    TimeDecay,
    HybridRetrieval,
    MemoryRetriever,
)


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(role=MessageRole.USER, content="Hello, how are you?"),
        Message(role=MessageRole.ASSISTANT, content="I'm doing well, thank you!"),
        Message(role=MessageRole.USER, content="What's the weather like?"),
        Message(role=MessageRole.ASSISTANT, content="I don't have access to weather data."),
    ]


@pytest.fixture
def conversation_buffer():
    """Create a conversation buffer."""
    return ConversationBuffer()


@pytest.fixture
def sliding_window_memory():
    """Create a sliding window memory."""
    return SlidingWindowMemory(window_size=5)


@pytest.fixture
def long_term_memory():
    """Create a long-term memory instance."""
    return LongTermMemory()


@pytest.fixture
def sample_memory_entries():
    """Create sample memory entries."""
    now = datetime.now()
    return [
        MemoryEntry(
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
            importance=0.8,
            timestamp=now - timedelta(hours=1),
        ),
        MemoryEntry(
            content="User's name is Alice",
            memory_type=MemoryType.FACT,
            importance=0.9,
            timestamp=now - timedelta(hours=2),
        ),
        MemoryEntry(
            content="User asked about Python programming",
            memory_type=MemoryType.CONVERSATION,
            importance=0.5,
            timestamp=now - timedelta(minutes=30),
        ),
    ]
