"""
记忆模块单元测试 (Memory Module Unit Tests)

测试覆盖：
    - Message消息类
    - MessageRole角色枚举
    - BufferMemory缓冲记忆
    - WindowMemory窗口记忆
    - SummaryMemory摘要记忆
    - VectorMemory向量记忆

"""

import pytest
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory import (
    Message,
    MessageRole,
    ConversationMemory,
    BufferMemory,
    WindowMemory,
    SummaryMemory,
    VectorMemory,
)


# ==================== MessageRole Tests ====================

class TestMessageRole:
    """MessageRole测试类。"""

    def test_role_values(self):
        """测试角色值。"""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"

    def test_role_from_string(self):
        """测试从字符串创建角色。"""
        assert MessageRole("system") == MessageRole.SYSTEM
        assert MessageRole("user") == MessageRole.USER
        assert MessageRole("assistant") == MessageRole.ASSISTANT


# ==================== Message Tests ====================

class TestMessage:
    """Message测试类。"""

    def test_create_message(self):
        """测试创建消息。"""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_create_with_string_role(self):
        """测试使用字符串角色创建消息。"""
        msg = Message(role="user", content="Hello")
        assert msg.role == MessageRole.USER

    def test_system_message(self):
        """测试创建系统消息。"""
        msg = Message.system("You are a helpful assistant")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are a helpful assistant"

    def test_user_message(self):
        """测试创建用户消息。"""
        msg = Message.user("What is 2+2?")
        assert msg.role == MessageRole.USER
        assert msg.content == "What is 2+2?"

    def test_assistant_message(self):
        """测试创建助手消息。"""
        msg = Message.assistant("The answer is 4")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "The answer is 4"

    def test_tool_message(self):
        """测试创建工具消息。"""
        msg = Message.tool("Result: 42", tool_call_id="call_123", name="calculator")
        assert msg.role == MessageRole.TOOL
        assert msg.content == "Result: 42"
        assert msg.tool_call_id == "call_123"
        assert msg.name == "calculator"

    def test_to_dict(self):
        """测试转换为字典。"""
        msg = Message.user("Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_to_dict_with_name(self):
        """测试带名称的字典转换。"""
        msg = Message(role=MessageRole.USER, content="Hello", name="Alice")
        d = msg.to_dict()
        assert d["name"] == "Alice"

    def test_from_dict(self):
        """测试从字典创建。"""
        d = {"role": "user", "content": "Hello"}
        msg = Message.from_dict(d)
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_to_openai_format(self):
        """测试转换为OpenAI格式。"""
        msg = Message.user("Hello")
        openai_msg = msg.to_openai_format()
        assert openai_msg["role"] == "user"
        assert openai_msg["content"] == "Hello"

    def test_token_estimate(self):
        """测试token估算。"""
        msg = Message.user("Hello World")
        assert msg.token_estimate > 0

    def test_message_repr(self):
        """测试消息的字符串表示。"""
        msg = Message.user("Hello")
        repr_str = repr(msg)
        assert "Message" in repr_str
        assert "user" in repr_str

    def test_message_repr_long_content(self):
        """测试长内容消息的字符串表示。"""
        long_content = "A" * 100
        msg = Message.user(long_content)
        repr_str = repr(msg)
        assert "..." in repr_str

    def test_message_metadata(self):
        """测试消息元数据。"""
        msg = Message(
            role=MessageRole.USER,
            content="Hello",
            metadata={"key": "value"},
        )
        assert msg.metadata == {"key": "value"}

    def test_message_timestamp(self):
        """测试消息时间戳。"""
        msg = Message.user("Hello")
        assert msg.timestamp is not None


# ==================== BufferMemory Tests ====================

class TestBufferMemory:
    """BufferMemory测试类。"""

    def test_create_empty_memory(self):
        """测试创建空记忆。"""
        memory = BufferMemory()
        assert memory.message_count == 0

    def test_create_with_system_message(self):
        """测试带系统消息创建。"""
        memory = BufferMemory(system_message="You are helpful")
        assert memory.message_count == 1
        messages = memory.get_messages()
        assert messages[0].role == MessageRole.SYSTEM

    def test_add_message(self):
        """测试添加消息。"""
        memory = BufferMemory()
        memory.add_message(Message.user("Hello"))
        assert memory.message_count == 1

    def test_add_user_message(self):
        """测试添加用户消息。"""
        memory = BufferMemory()
        memory.add_user_message("Hello")
        messages = memory.get_messages()
        assert messages[0].role == MessageRole.USER

    def test_add_assistant_message(self):
        """测试添加助手消息。"""
        memory = BufferMemory()
        memory.add_assistant_message("Hi there")
        messages = memory.get_messages()
        assert messages[0].role == MessageRole.ASSISTANT

    def test_add_system_message(self):
        """测试添加系统消息。"""
        memory = BufferMemory()
        memory.add_system_message("Be helpful")
        messages = memory.get_messages()
        assert messages[0].role == MessageRole.SYSTEM

    def test_get_messages(self):
        """测试获取消息。"""
        memory = BufferMemory()
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi")
        messages = memory.get_messages()
        assert len(messages) == 2

    def test_get_messages_as_dicts(self):
        """测试获取字典格式消息。"""
        memory = BufferMemory()
        memory.add_user_message("Hello")
        dicts = memory.get_messages_as_dicts()
        assert isinstance(dicts, list)
        assert dicts[0]["role"] == "user"

    def test_clear_memory(self):
        """测试清空记忆。"""
        memory = BufferMemory()
        memory.add_user_message("Hello")
        memory.clear()
        assert memory.message_count == 0

    def test_clear_preserves_system_message(self):
        """测试清空保留系统消息。"""
        memory = BufferMemory(system_message="Be helpful")
        memory.add_user_message("Hello")
        memory.clear()
        assert memory.message_count == 1
        assert memory.get_messages()[0].role == MessageRole.SYSTEM

    def test_token_count(self):
        """测试token计数。"""
        memory = BufferMemory()
        memory.add_user_message("Hello World")
        assert memory.token_count > 0

    def test_buffer_memory_repr(self):
        """测试缓冲记忆的字符串表示。"""
        memory = BufferMemory()
        memory.add_user_message("Hello")
        repr_str = repr(memory)
        assert "BufferMemory" in repr_str


# ==================== WindowMemory Tests ====================

class TestWindowMemory:
    """WindowMemory测试类。"""

    def test_create_window_memory(self):
        """测试创建窗口记忆。"""
        memory = WindowMemory(k=5)
        assert memory.window_size == 0

    def test_invalid_k(self):
        """测试无效的k值。"""
        with pytest.raises(ValueError, match="k必须为正数"):
            WindowMemory(k=0)
        with pytest.raises(ValueError, match="k必须为正数"):
            WindowMemory(k=-1)

    def test_add_messages_within_window(self):
        """测试在窗口内添加消息。"""
        memory = WindowMemory(k=3)
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi")
        assert memory.window_size == 2

    def test_window_overflow(self):
        """测试窗口溢出。"""
        memory = WindowMemory(k=2)  # 保留2轮 = 4条消息
        for i in range(6):
            memory.add_user_message(f"User {i}")
            memory.add_assistant_message(f"Assistant {i}")
        # 应该只保留最近4条
        assert memory.window_size == 4

    def test_system_message_not_counted(self):
        """测试系统消息不计入窗口。"""
        memory = WindowMemory(k=2, system_message="Be helpful")
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi")
        messages = memory.get_messages()
        # 系统消息 + 2条对话消息
        assert len(messages) == 3
        assert messages[0].role == MessageRole.SYSTEM

    def test_clear_window_memory(self):
        """测试清空窗口记忆。"""
        memory = WindowMemory(k=3, system_message="Be helpful")
        memory.add_user_message("Hello")
        memory.clear()
        assert memory.window_size == 0
        # 系统消息应该保留
        messages = memory.get_messages()
        assert len(messages) == 1

    def test_window_memory_repr(self):
        """测试窗口记忆的字符串表示。"""
        memory = WindowMemory(k=5)
        repr_str = repr(memory)
        assert "WindowMemory" in repr_str
        assert "k=5" in repr_str


# ==================== SummaryMemory Tests ====================

class TestSummaryMemory:
    """SummaryMemory测试类。"""

    def test_create_summary_memory(self):
        """测试创建摘要记忆。"""
        memory = SummaryMemory(max_messages=10)
        assert memory.summary == ""

    def test_invalid_max_messages(self):
        """测试无效的max_messages。"""
        with pytest.raises(ValueError, match="max_messages必须为正数"):
            SummaryMemory(max_messages=0)

    def test_add_messages_no_summary(self):
        """测试添加消息（不触发摘要）。"""
        memory = SummaryMemory(max_messages=10)
        for i in range(5):
            memory.add_user_message(f"Message {i}")
        assert memory.summary == ""

    def test_summary_triggered(self):
        """测试触发摘要生成。"""
        memory = SummaryMemory(max_messages=4)
        for i in range(6):
            memory.add_user_message(f"Message {i}")
        # 超过max_messages应该触发摘要
        assert memory.summary != ""

    def test_custom_summarizer(self):
        """测试自定义摘要器。"""
        def custom_summarizer(messages):
            return f"Summary of {len(messages)} messages"
        
        memory = SummaryMemory(max_messages=4, summarizer=custom_summarizer)
        for i in range(6):
            memory.add_user_message(f"Message {i}")
        assert "Summary of" in memory.summary

    def test_get_messages_with_summary(self):
        """测试获取带摘要的消息。"""
        memory = SummaryMemory(max_messages=4)
        for i in range(6):
            memory.add_user_message(f"Message {i}")
        messages = memory.get_messages()
        # 应该包含摘要消息
        has_summary = any("摘要" in m.content for m in messages if m.role == MessageRole.SYSTEM)
        assert has_summary

    def test_clear_summary_memory(self):
        """测试清空摘要记忆。"""
        memory = SummaryMemory(max_messages=4)
        for i in range(6):
            memory.add_user_message(f"Message {i}")
        memory.clear()
        assert memory.summary == ""

    def test_summary_memory_repr(self):
        """测试摘要记忆的字符串表示。"""
        memory = SummaryMemory(max_messages=10)
        repr_str = repr(memory)
        assert "SummaryMemory" in repr_str


# ==================== VectorMemory Tests ====================

class TestVectorMemory:
    """VectorMemory测试类。"""

    def test_create_vector_memory(self):
        """测试创建向量记忆。"""
        memory = VectorMemory(top_k=5)
        assert memory.total_messages == 0

    def test_invalid_top_k(self):
        """测试无效的top_k。"""
        with pytest.raises(ValueError, match="top_k必须为正数"):
            VectorMemory(top_k=0)

    def test_add_messages(self):
        """测试添加消息。"""
        memory = VectorMemory()
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi")
        assert memory.total_messages == 2

    def test_retrieve_empty(self):
        """测试空记忆检索。"""
        memory = VectorMemory()
        results = memory.retrieve("query")
        assert results == []

    def test_retrieve_messages(self):
        """测试检索消息。"""
        memory = VectorMemory(top_k=2)
        memory.add_user_message("Python programming")
        memory.add_user_message("Machine learning")
        memory.add_user_message("Data science")
        results = memory.retrieve("Python")
        assert len(results) <= 2

    def test_get_context_for_query(self):
        """测试获取查询上下文。"""
        memory = VectorMemory(top_k=2, system_message="Be helpful")
        memory.add_user_message("Python programming")
        memory.add_user_message("Machine learning")
        context = memory.get_context_for_query("Python")
        assert len(context) > 0

    def test_custom_embed_fn(self):
        """测试自定义嵌入函数。"""
        def custom_embed(text):
            return [len(text) / 100.0] * 16
        
        memory = VectorMemory(embed_fn=custom_embed)
        memory.add_user_message("Hello")
        assert memory.total_messages == 1

    def test_clear_vector_memory(self):
        """测试清空向量记忆。"""
        memory = VectorMemory()
        memory.add_user_message("Hello")
        memory.clear()
        assert memory.total_messages == 0

    def test_vector_memory_repr(self):
        """测试向量记忆的字符串表示。"""
        memory = VectorMemory(top_k=5)
        repr_str = repr(memory)
        assert "VectorMemory" in repr_str
        assert "top_k=5" in repr_str


# ==================== ConversationMemory Interface Tests ====================

class TestConversationMemoryInterface:
    """ConversationMemory接口测试。"""

    @pytest.mark.parametrize("memory_class", [
        BufferMemory,
        lambda: WindowMemory(k=5),
        lambda: SummaryMemory(max_messages=10),
        lambda: VectorMemory(top_k=5),
    ])
    def test_memory_has_add_message(self, memory_class):
        """测试记忆有add_message方法。"""
        memory = memory_class() if callable(memory_class) else memory_class
        assert hasattr(memory, 'add_message')

    @pytest.mark.parametrize("memory_class", [
        BufferMemory,
        lambda: WindowMemory(k=5),
        lambda: SummaryMemory(max_messages=10),
        lambda: VectorMemory(top_k=5),
    ])
    def test_memory_has_get_messages(self, memory_class):
        """测试记忆有get_messages方法。"""
        memory = memory_class() if callable(memory_class) else memory_class
        assert hasattr(memory, 'get_messages')

    @pytest.mark.parametrize("memory_class", [
        BufferMemory,
        lambda: WindowMemory(k=5),
        lambda: SummaryMemory(max_messages=10),
        lambda: VectorMemory(top_k=5),
    ])
    def test_memory_has_clear(self, memory_class):
        """测试记忆有clear方法。"""
        memory = memory_class() if callable(memory_class) else memory_class
        assert hasattr(memory, 'clear')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
