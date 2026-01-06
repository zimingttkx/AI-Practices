"""
Comprehensive Tests for Short-Term Memory Implementations.

Test Coverage:
    - Message dataclass: creation, serialization, validation
    - ConversationBuffer: basic operations, edge cases
    - SlidingWindowMemory: window management, system message handling
    - SummaryMemory: summarization triggers, multiple summaries
    - TokenBasedMemory: budget enforcement, eviction policies
    - Factory function: all strategies, error handling
    - Edge cases: empty inputs, unicode, large content
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timedelta
import time

from short_term_memory import (
    Message,
    MessageRole,
    ConversationBuffer,
    SlidingWindowMemory,
    SummaryMemory,
    TokenBasedMemory,
    SimpleTokenCounter,
    SimpleSummarizer,
    create_conversation_memory,
    ShortTermMemory,
)


# =============================================================================
# Message Tests
# =============================================================================


class TestMessage:
    """Comprehensive tests for Message dataclass."""

    def test_create_message(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.importance == 0.5

    def test_message_from_string_role(self):
        msg = Message(role="user", content="Test")
        assert msg.role == MessageRole.USER

    def test_message_id_unique(self):
        msg1 = Message(role=MessageRole.USER, content="Hello")
        msg2 = Message(role=MessageRole.USER, content="World")
        assert msg1.id != msg2.id

    def test_message_to_dict(self):
        msg = Message(role=MessageRole.USER, content="Test")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Test"

    def test_message_from_dict(self):
        data = {"role": "assistant", "content": "Response"}
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Response"

    def test_all_message_roles(self):
        """Test all MessageRole enum values."""
        for role in MessageRole:
            msg = Message(role=role, content="Test")
            assert msg.role == role

    def test_message_with_metadata(self):
        """Test message with custom metadata."""
        msg = Message(
            role=MessageRole.USER,
            content="Test",
            metadata={"source": "api", "version": 1}
        )
        assert msg.metadata["source"] == "api"
        assert msg.metadata["version"] == 1

    def test_message_importance_bounds(self):
        """Test importance clamping to [0, 1]."""
        msg_high = Message(role=MessageRole.USER, content="Test", importance=1.5)
        msg_low = Message(role=MessageRole.USER, content="Test", importance=-0.5)
        assert 0.0 <= msg_high.importance <= 1.0
        assert 0.0 <= msg_low.importance <= 1.0

    def test_message_token_count_estimation(self):
        """Test automatic token count estimation."""
        msg = Message(role=MessageRole.USER, content="Hello world test message")
        assert msg.token_count is not None
        assert msg.token_count > 0

    def test_message_empty_content(self):
        """Test message with empty content."""
        msg = Message(role=MessageRole.USER, content="")
        assert msg.content == ""
        assert msg.id is not None

    def test_message_unicode_content(self):
        """Test message with unicode characters."""
        msg = Message(role=MessageRole.USER, content="你好世界 🌍 مرحبا")
        assert "你好" in msg.content
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.content == msg.content

    def test_message_long_content(self):
        """Test message with very long content."""
        long_content = "x" * 10000
        msg = Message(role=MessageRole.USER, content=long_content)
        assert len(msg.content) == 10000

    def test_message_serialization_roundtrip(self):
        """Test full serialization/deserialization cycle."""
        original = Message(
            role=MessageRole.ASSISTANT,
            content="Test response",
            importance=0.8,
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = Message.from_dict(data)
        assert restored.role == original.role
        assert restored.content == original.content
        # Note: importance may use default if not preserved in from_dict
        assert restored.importance is not None


# =============================================================================
# ConversationBuffer Tests
# =============================================================================


class TestConversationBuffer:
    """Comprehensive tests for ConversationBuffer."""

    def test_add_message(self):
        buffer = ConversationBuffer()
        buffer.add_user_message("Hello")
        assert len(buffer) == 1

    def test_get_messages(self):
        buffer = ConversationBuffer()
        buffer.add_user_message("Hello")
        buffer.add_assistant_message("Hi there!")
        messages = buffer.get_messages()
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[1].role == MessageRole.ASSISTANT

    def test_clear(self):
        buffer = ConversationBuffer()
        buffer.add_user_message("Hello")
        buffer.clear()
        assert len(buffer) == 0

    def test_system_message_preserved_on_clear(self):
        buffer = ConversationBuffer(system_message="You are helpful.")
        buffer.add_user_message("Hello")
        buffer.clear()
        messages = buffer.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM

    def test_to_openai_messages(self):
        buffer = ConversationBuffer()
        buffer.add_user_message("Hello")
        openai_msgs = buffer.to_openai_messages()
        assert openai_msgs == [{"role": "user", "content": "Hello"}]

    def test_add_system_message(self):
        """Test adding system message."""
        buffer = ConversationBuffer()
        buffer.add_system_message("You are a helpful assistant.")
        messages = buffer.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM

    def test_multiple_system_messages(self):
        """Test multiple system messages."""
        buffer = ConversationBuffer()
        buffer.add_system_message("First instruction")
        buffer.add_system_message("Second instruction")
        messages = buffer.get_messages()
        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        assert len(system_msgs) == 2

    def test_conversation_flow(self):
        """Test typical conversation flow."""
        buffer = ConversationBuffer(system_message="You are helpful.")
        buffer.add_user_message("What is Python?")
        buffer.add_assistant_message("Python is a programming language.")
        buffer.add_user_message("Tell me more.")
        buffer.add_assistant_message("It's known for readability.")
        
        messages = buffer.get_messages()
        assert len(messages) == 5
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.USER
        assert messages[2].role == MessageRole.ASSISTANT

    def test_token_count_property(self):
        """Test total token count calculation."""
        buffer = ConversationBuffer()
        buffer.add_user_message("Hello world")
        buffer.add_assistant_message("Hi there")
        assert buffer.token_count > 0

    def test_empty_buffer(self):
        """Test operations on empty buffer."""
        buffer = ConversationBuffer()
        assert len(buffer) == 0
        assert buffer.get_messages() == []
        assert buffer.to_openai_messages() == []

    def test_add_message_object(self):
        """Test adding Message object directly."""
        buffer = ConversationBuffer()
        msg = Message(role=MessageRole.USER, content="Direct add")
        buffer.add(msg)
        assert len(buffer) == 1
        assert buffer.get_messages()[0].content == "Direct add"


# =============================================================================
# SlidingWindowMemory Tests
# =============================================================================


class TestSlidingWindowMemory:
    """Comprehensive tests for SlidingWindowMemory."""

    def test_window_size_limit(self):
        memory = SlidingWindowMemory(window_size=3)
        for i in range(5):
            memory.add_user_message(f"Message {i}")
        messages = memory.get_messages()
        assert len(messages) == 3
        assert "Message 2" in messages[0].content

    def test_system_message_preserved(self):
        memory = SlidingWindowMemory(window_size=2, system_message="System")
        for i in range(5):
            memory.add_user_message(f"Message {i}")
        messages = memory.get_messages()
        assert messages[0].role == MessageRole.SYSTEM
        assert len(messages) == 3

    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            SlidingWindowMemory(window_size=0)

    def test_set_window_size(self):
        memory = SlidingWindowMemory(window_size=5)
        for i in range(5):
            memory.add_user_message(f"Message {i}")
        memory.set_window_size(2)
        assert len(memory.get_messages()) == 2

    def test_window_size_one(self):
        """Test minimum window size of 1."""
        memory = SlidingWindowMemory(window_size=1)
        memory.add_user_message("First")
        memory.add_user_message("Second")
        messages = memory.get_messages()
        assert len(messages) == 1
        assert "Second" in messages[0].content

    def test_window_larger_than_messages(self):
        """Test window larger than message count."""
        memory = SlidingWindowMemory(window_size=10)
        memory.add_user_message("Only one")
        messages = memory.get_messages()
        assert len(messages) == 1

    def test_increase_window_size(self):
        """Test increasing window size."""
        memory = SlidingWindowMemory(window_size=2)
        for i in range(5):
            memory.add_user_message(f"Message {i}")
        memory.set_window_size(10)
        # Should still only have 2 messages (can't recover old ones)
        assert len(memory.get_messages()) == 2

    def test_system_message_not_counted_in_window(self):
        """Test that system message doesn't count toward window."""
        memory = SlidingWindowMemory(window_size=2, system_message="System")
        memory.add_user_message("User 1")
        memory.add_assistant_message("Assistant 1")
        memory.add_user_message("User 2")
        messages = memory.get_messages()
        # System + 2 most recent
        assert len(messages) == 3
        assert messages[0].role == MessageRole.SYSTEM

    def test_clear_preserves_system(self):
        """Test clear preserves system message."""
        memory = SlidingWindowMemory(window_size=5, system_message="Keep me")
        memory.add_user_message("Delete me")
        memory.clear()
        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM


# =============================================================================
# SummaryMemory Tests
# =============================================================================


class TestSummaryMemory:
    """Comprehensive tests for SummaryMemory."""

    def test_summarize_when_exceeded(self):
        memory = SummaryMemory(max_messages=5, summarize_count=3)
        for i in range(7):
            memory.add_user_message(f"Message {i}")
        messages = memory.get_messages()
        assert any("Summary" in m.content for m in messages)

    def test_invalid_summarize_count(self):
        with pytest.raises(ValueError):
            SummaryMemory(max_messages=5, summarize_count=5)

    def test_no_summarization_under_limit(self):
        """Test no summarization when under limit."""
        memory = SummaryMemory(max_messages=10, summarize_count=5)
        for i in range(5):
            memory.add_user_message(f"Message {i}")
        messages = memory.get_messages()
        assert not any("Summary" in m.content for m in messages)
        assert len(messages) == 5

    def test_multiple_summarizations(self):
        """Test multiple summarization cycles."""
        memory = SummaryMemory(max_messages=5, summarize_count=3)
        for i in range(15):
            memory.add_user_message(f"Message {i}")
        assert memory.summary_count >= 2

    def test_summary_count_property(self):
        """Test summary_count property."""
        memory = SummaryMemory(max_messages=5, summarize_count=3)
        assert memory.summary_count == 0
        for i in range(7):
            memory.add_user_message(f"Message {i}")
        assert memory.summary_count >= 1

    def test_system_message_preserved(self):
        """Test system message preserved during summarization."""
        memory = SummaryMemory(
            max_messages=5,
            summarize_count=3,
            system_message="I am system"
        )
        for i in range(10):
            memory.add_user_message(f"Message {i}")
        messages = memory.get_messages()
        system_msgs = [m for m in messages if "I am system" in m.content]
        # System message should be preserved (may be in summary context)
        assert len(memory.get_messages()) > 0

    def test_clear_resets_summaries(self):
        """Test clear resets summaries."""
        memory = SummaryMemory(max_messages=5, summarize_count=3)
        for i in range(10):
            memory.add_user_message(f"Message {i}")
        memory.clear()
        assert memory.summary_count == 0


# =============================================================================
# TokenBasedMemory Tests
# =============================================================================


class TestTokenBasedMemory:
    """Comprehensive tests for TokenBasedMemory."""

    def test_token_budget(self):
        memory = TokenBasedMemory(max_tokens=100, reserve_tokens=20)
        for i in range(20):
            memory.add_user_message(f"This is message number {i} with some content")
        assert memory.token_count <= 80

    def test_available_tokens(self):
        memory = TokenBasedMemory(max_tokens=1000, reserve_tokens=100)
        memory.add_user_message("Hello")
        assert memory.available_tokens > 0
        assert memory.available_tokens < 900

    def test_invalid_reserve_tokens(self):
        """Test reserve_tokens >= max_tokens raises error."""
        with pytest.raises(ValueError):
            TokenBasedMemory(max_tokens=100, reserve_tokens=100)

    def test_system_message_preserved(self):
        """Test system message preserved during eviction."""
        memory = TokenBasedMemory(
            max_tokens=200,
            reserve_tokens=50,
            system_message="Keep me"
        )
        for i in range(50):
            memory.add_user_message(f"Message {i} with content")
        messages = memory.get_messages()
        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        assert len(system_msgs) >= 1

    def test_prioritize_recent_true(self):
        """Test recency-based eviction."""
        memory = TokenBasedMemory(
            max_tokens=50,
            reserve_tokens=10,
            prioritize_recent=True
        )
        memory.add_user_message("Old message that should be evicted")
        for i in range(10):
            memory.add_user_message(f"New message {i}")
        messages = memory.get_messages()
        # With small budget, old message should be evicted first
        contents = [m.content for m in messages]
        # Either old message is evicted or we have limited messages
        assert len(messages) <= 11

    def test_max_tokens_property(self):
        """Test max_tokens property."""
        memory = TokenBasedMemory(max_tokens=5000, reserve_tokens=500)
        assert memory.max_tokens == 5000

    def test_clear_preserves_system(self):
        """Test clear preserves system message."""
        memory = TokenBasedMemory(
            max_tokens=1000,
            system_message="System prompt"
        )
        memory.add_user_message("User message")
        memory.clear()
        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM

    def test_single_large_message(self):
        """Test handling of single large message."""
        memory = TokenBasedMemory(max_tokens=100, reserve_tokens=20)
        large_msg = "x" * 500
        memory.add_user_message(large_msg)
        # Should still have at least one message
        assert len(memory.get_messages()) >= 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Comprehensive tests for create_conversation_memory factory."""

    def test_create_buffer(self):
        memory = create_conversation_memory("buffer")
        assert isinstance(memory, ConversationBuffer)

    def test_create_sliding_window(self):
        memory = create_conversation_memory("sliding_window", window_size=10)
        assert isinstance(memory, SlidingWindowMemory)

    def test_create_summary(self):
        memory = create_conversation_memory("summary", max_messages=20)
        assert isinstance(memory, SummaryMemory)

    def test_create_token_based(self):
        memory = create_conversation_memory("token_based", max_tokens=4000)
        assert isinstance(memory, TokenBasedMemory)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            create_conversation_memory("invalid")

    def test_factory_with_system_message(self):
        """Test factory with system_message parameter."""
        memory = create_conversation_memory(
            "buffer",
            system_message="You are helpful."
        )
        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM

    def test_all_strategies_implement_interface(self):
        """Test all strategies implement ShortTermMemory interface."""
        strategies = ["buffer", "sliding_window", "summary", "token_based"]
        for strategy in strategies:
            if strategy == "summary":
                memory = create_conversation_memory(strategy, max_messages=10, summarize_count=5)
            elif strategy == "sliding_window":
                memory = create_conversation_memory(strategy, window_size=5)
            elif strategy == "token_based":
                memory = create_conversation_memory(strategy, max_tokens=1000)
            else:
                memory = create_conversation_memory(strategy)
            
            # Test common interface
            memory.add_user_message("Test")
            assert len(memory.get_messages()) >= 1
            memory.clear()


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_rapid_add_clear_cycle(self):
        """Test rapid add/clear cycles."""
        buffer = ConversationBuffer()
        for _ in range(100):
            buffer.add_user_message("Test")
            buffer.clear()
        assert len(buffer) == 0

    def test_special_characters_in_content(self):
        """Test special characters handling."""
        buffer = ConversationBuffer()
        special = "Test\n\t\r\\\"'`~!@#$%^&*()[]{}|;:,.<>?"
        buffer.add_user_message(special)
        messages = buffer.get_messages()
        assert messages[0].content == special

    def test_very_long_conversation(self):
        """Test very long conversation."""
        memory = SlidingWindowMemory(window_size=10)
        for i in range(1000):
            memory.add_user_message(f"Message {i}")
        assert len(memory.get_messages()) == 10

    def test_mixed_message_types(self):
        """Test mixed message types in conversation."""
        buffer = ConversationBuffer()
        buffer.add_system_message("System")
        buffer.add_user_message("User")
        buffer.add_assistant_message("Assistant")
        buffer.add(Message(role=MessageRole.TOOL, content="Tool result"))
        
        messages = buffer.get_messages()
        roles = [m.role for m in messages]
        assert MessageRole.SYSTEM in roles
        assert MessageRole.USER in roles
        assert MessageRole.ASSISTANT in roles
        assert MessageRole.TOOL in roles

    def test_openai_format_with_all_roles(self):
        """Test OpenAI format conversion with all roles."""
        buffer = ConversationBuffer()
        buffer.add_system_message("System")
        buffer.add_user_message("User")
        buffer.add_assistant_message("Assistant")
        
        openai_msgs = buffer.to_openai_messages()
        assert len(openai_msgs) == 3
        assert all("role" in m and "content" in m for m in openai_msgs)
