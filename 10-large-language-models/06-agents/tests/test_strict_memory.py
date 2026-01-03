"""
记忆模块最严格单元测试 (Strictest Memory Unit Tests)

测试覆盖：
    - 边界条件测试
    - 异常处理测试
    - 类型验证测试
    - 状态一致性测试

"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory import (
    Message, MessageRole,
    BufferMemory, WindowMemory,
    SummaryMemory, VectorMemory,
)


# ==================== MessageRole 严格测试 ====================

class TestMessageRoleStrict:
    """MessageRole最严格测试。"""

    def test_all_role_values(self):
        """测试所有角色值。"""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"

    def test_role_count(self):
        """测试角色数量。"""
        assert len(MessageRole) == 4

    def test_role_from_value(self):
        """测试从值创建角色。"""
        for role in MessageRole:
            assert MessageRole(role.value) == role


# ==================== Message 严格测试 ====================

class TestMessageStrict:
    """Message最严格测试。"""

    def test_empty_content(self):
        """测试空内容。"""
        msg = Message.user("")
        assert msg.content == ""

    def test_very_long_content(self):
        """测试超长内容。"""
        content = "x" * 100000
        msg = Message.user(content)
        assert len(msg.content) == 100000

    def test_unicode_content(self):
        """测试Unicode内容。"""
        msg = Message.user("你好世界🌍🚀")
        assert "你好" in msg.content
        assert "🌍" in msg.content

    def test_newlines_content(self):
        """测试换行内容。"""
        msg = Message.user("line1\nline2\r\nline3")
        assert "\n" in msg.content

    def test_string_role_conversion(self):
        """测试字符串角色转换。"""
        msg = Message(role="user", content="test")
        assert msg.role == MessageRole.USER

    def test_all_role_types(self):
        """测试所有角色类型。"""
        for role in MessageRole:
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_tool_message_complete(self):
        """测试完整工具消息。"""
        msg = Message.tool("result", tool_call_id="id123", name="calc")
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "id123"
        assert msg.name == "calc"

    def test_to_dict_minimal(self):
        """测试最小字典转换。"""
        msg = Message.user("test")
        d = msg.to_dict()
        assert "role" in d
        assert "content" in d
        assert "name" not in d
        assert "tool_call_id" not in d

    def test_to_dict_complete(self):
        """测试完整字典转换。"""
        msg = Message.tool("result", tool_call_id="id", name="tool")
        d = msg.to_dict()
        assert d["name"] == "tool"
        assert d["tool_call_id"] == "id"

    def test_from_dict_minimal(self):
        """测试最小字典创建。"""
        msg = Message.from_dict({"role": "user", "content": "test"})
        assert msg.role == MessageRole.USER
        assert msg.content == "test"

    def test_from_dict_with_metadata(self):
        """测试带元数据字典创建。"""
        msg = Message.from_dict({
            "role": "user",
            "content": "test",
            "metadata": {"key": "value"},
        })
        assert msg.metadata == {"key": "value"}

    def test_token_estimate_empty(self):
        """测试空内容token估算。"""
        msg = Message.user("")
        assert msg.token_estimate == 0

    def test_token_estimate_positive(self):
        """测试正常内容token估算。"""
        msg = Message.user("Hello World")
        assert msg.token_estimate > 0

    def test_repr_short_content(self):
        """测试短内容repr。"""
        msg = Message.user("short")
        repr_str = repr(msg)
        assert "short" in repr_str
        assert "..." not in repr_str

    def test_repr_long_content(self):
        """测试长内容repr。"""
        msg = Message.user("x" * 100)
        repr_str = repr(msg)
        assert "..." in repr_str

    def test_timestamp_exists(self):
        """测试时间戳存在。"""
        msg = Message.user("test")
        assert msg.timestamp is not None

    def test_metadata_default_empty(self):
        """测试元数据默认为空。"""
        msg = Message.user("test")
        assert msg.metadata == {}


# ==================== BufferMemory 严格测试 ====================

class TestBufferMemoryStrict:
    """BufferMemory最严格测试。"""

    def test_empty_memory(self):
        """测试空记忆。"""
        memory = BufferMemory()
        assert memory.message_count == 0
        assert memory.token_count == 0
        assert memory.get_messages() == []

    def test_system_message_init(self):
        """测试系统消息初始化。"""
        memory = BufferMemory(system_message="You are helpful")
        assert memory.message_count == 1
        msgs = memory.get_messages()
        assert msgs[0].role == MessageRole.SYSTEM

    def test_add_all_message_types(self):
        """测试添加所有消息类型。"""
        memory = BufferMemory()
        memory.add_system_message("system")
        memory.add_user_message("user")
        memory.add_assistant_message("assistant")
        assert memory.message_count == 3

    def test_message_order_preserved(self):
        """测试消息顺序保持。"""
        memory = BufferMemory()
        for i in range(10):
            memory.add_user_message(f"msg{i}")
        msgs = memory.get_messages()
        for i, msg in enumerate(msgs):
            assert msg.content == f"msg{i}"

    def test_get_messages_returns_copy(self):
        """测试获取消息返回副本。"""
        memory = BufferMemory()
        memory.add_user_message("test")
        msgs1 = memory.get_messages()
        msgs2 = memory.get_messages()
        assert msgs1 is not msgs2

    def test_get_messages_as_dicts(self):
        """测试获取字典格式消息。"""
        memory = BufferMemory()
        memory.add_user_message("test")
        dicts = memory.get_messages_as_dicts()
        assert isinstance(dicts, list)
        assert isinstance(dicts[0], dict)

    def test_clear_empty_memory(self):
        """测试清空空记忆。"""
        memory = BufferMemory()
        memory.clear()
        assert memory.message_count == 0

    def test_clear_preserves_system(self):
        """测试清空保留系统消息。"""
        memory = BufferMemory(system_message="system")
        memory.add_user_message("user")
        memory.clear()
        assert memory.message_count == 1
        assert memory.get_messages()[0].role == MessageRole.SYSTEM

    def test_large_message_count(self):
        """测试大量消息。"""
        memory = BufferMemory()
        for i in range(1000):
            memory.add_user_message(f"msg{i}")
        assert memory.message_count == 1000

    def test_repr(self):
        """测试repr。"""
        memory = BufferMemory()
        memory.add_user_message("test")
        repr_str = repr(memory)
        assert "BufferMemory" in repr_str


# ==================== WindowMemory 严格测试 ====================

class TestWindowMemoryStrict:
    """WindowMemory最严格测试。"""

    def test_k_equals_one(self):
        """测试k=1。"""
        memory = WindowMemory(k=1)
        memory.add_user_message("u1")
        memory.add_assistant_message("a1")
        memory.add_user_message("u2")
        memory.add_assistant_message("a2")
        assert memory.window_size == 2  # 1轮 = 2条

    def test_invalid_k_zero(self):
        """测试k=0无效。"""
        with pytest.raises(ValueError):
            WindowMemory(k=0)

    def test_invalid_k_negative(self):
        """测试k为负数无效。"""
        with pytest.raises(ValueError):
            WindowMemory(k=-1)

    def test_window_exact_size(self):
        """测试窗口精确大小。"""
        memory = WindowMemory(k=3)  # 6条消息
        for i in range(10):
            memory.add_user_message(f"u{i}")
            memory.add_assistant_message(f"a{i}")
        assert memory.window_size == 6

    def test_system_message_separate(self):
        """测试系统消息独立。"""
        memory = WindowMemory(k=1, system_message="system")
        memory.add_user_message("u1")
        memory.add_assistant_message("a1")
        memory.add_user_message("u2")
        memory.add_assistant_message("a2")
        msgs = memory.get_messages()
        assert msgs[0].role == MessageRole.SYSTEM
        assert len(msgs) == 3  # system + 2条

    def test_add_system_replaces(self):
        """测试添加系统消息替换。"""
        memory = WindowMemory(k=2, system_message="old")
        memory.add_message(Message.system("new"))
        msgs = memory.get_messages()
        system_msgs = [m for m in msgs if m.role == MessageRole.SYSTEM]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "new"

    def test_clear_keeps_system(self):
        """测试清空保留系统消息。"""
        memory = WindowMemory(k=2, system_message="system")
        memory.add_user_message("test")
        memory.clear()
        assert memory.window_size == 0
        msgs = memory.get_messages()
        assert len(msgs) == 1

    def test_repr(self):
        """测试repr。"""
        memory = WindowMemory(k=5)
        repr_str = repr(memory)
        assert "WindowMemory" in repr_str
        assert "k=5" in repr_str


# ==================== SummaryMemory 严格测试 ====================

class TestSummaryMemoryStrict:
    """SummaryMemory最严格测试。"""

    def test_invalid_max_messages_zero(self):
        """测试max_messages=0无效。"""
        with pytest.raises(ValueError):
            SummaryMemory(max_messages=0)

    def test_invalid_max_messages_negative(self):
        """测试max_messages为负数无效。"""
        with pytest.raises(ValueError):
            SummaryMemory(max_messages=-1)

    def test_no_summary_under_limit(self):
        """测试未超限不生成摘要。"""
        memory = SummaryMemory(max_messages=10)
        for i in range(5):
            memory.add_user_message(f"msg{i}")
        assert memory.summary == ""

    def test_summary_triggered_over_limit(self):
        """测试超限触发摘要。"""
        memory = SummaryMemory(max_messages=4)
        for i in range(6):
            memory.add_user_message(f"msg{i}")
        assert memory.summary != ""

    def test_custom_summarizer(self):
        """测试自定义摘要器。"""
        def custom(msgs):
            return f"Count: {len(msgs)}"
        memory = SummaryMemory(max_messages=4, summarizer=custom)
        for i in range(6):
            memory.add_user_message(f"msg{i}")
        assert "Count:" in memory.summary

    def test_system_message_handling(self):
        """测试系统消息处理。"""
        memory = SummaryMemory(max_messages=4, system_message="system")
        for i in range(6):
            memory.add_user_message(f"msg{i}")
        msgs = memory.get_messages()
        assert msgs[0].role == MessageRole.SYSTEM

    def test_clear_resets_summary(self):
        """测试清空重置摘要。"""
        memory = SummaryMemory(max_messages=4)
        for i in range(6):
            memory.add_user_message(f"msg{i}")
        memory.clear()
        assert memory.summary == ""

    def test_repr(self):
        """测试repr。"""
        memory = SummaryMemory(max_messages=10)
        repr_str = repr(memory)
        assert "SummaryMemory" in repr_str


# ==================== VectorMemory 严格测试 ====================

class TestVectorMemoryStrict:
    """VectorMemory最严格测试。"""

    def test_invalid_top_k_zero(self):
        """测试top_k=0无效。"""
        with pytest.raises(ValueError):
            VectorMemory(top_k=0)

    def test_invalid_top_k_negative(self):
        """测试top_k为负数无效。"""
        with pytest.raises(ValueError):
            VectorMemory(top_k=-1)

    def test_empty_retrieve(self):
        """测试空记忆检索。"""
        memory = VectorMemory()
        results = memory.retrieve("query")
        assert results == []

    def test_retrieve_returns_list(self):
        """测试检索返回列表。"""
        memory = VectorMemory(top_k=2)
        memory.add_user_message("test1")
        memory.add_user_message("test2")
        results = memory.retrieve("test")
        assert isinstance(results, list)

    def test_retrieve_respects_top_k(self):
        """测试检索遵守top_k。"""
        memory = VectorMemory(top_k=2)
        for i in range(10):
            memory.add_user_message(f"msg{i}")
        results = memory.retrieve("msg")
        assert len(results) <= 2

    def test_custom_embed_fn(self):
        """测试自定义嵌入函数。"""
        def custom_embed(text):
            return [len(text) / 100.0] * 16
        memory = VectorMemory(embed_fn=custom_embed)
        memory.add_user_message("test")
        assert memory.total_messages == 1

    def test_system_message_handling(self):
        """测试系统消息处理。"""
        memory = VectorMemory(system_message="system")
        memory.add_user_message("test")
        msgs = memory.get_messages()
        assert msgs[0].role == MessageRole.SYSTEM

    def test_get_context_for_query(self):
        """测试获取查询上下文。"""
        memory = VectorMemory(top_k=2, system_message="system")
        memory.add_user_message("Python programming")
        memory.add_user_message("Machine learning")
        context = memory.get_context_for_query("Python")
        assert len(context) > 0

    def test_clear_resets_all(self):
        """测试清空重置所有。"""
        memory = VectorMemory()
        memory.add_user_message("test")
        memory.clear()
        assert memory.total_messages == 0

    def test_repr(self):
        """测试repr。"""
        memory = VectorMemory(top_k=5)
        repr_str = repr(memory)
        assert "VectorMemory" in repr_str
        assert "top_k=5" in repr_str


# ==================== 跨记忆类型测试 ====================

class TestMemoryInterfaceStrict:
    """记忆接口一致性测试。"""

    @pytest.fixture(params=[
        BufferMemory,
        lambda: WindowMemory(k=5),
        lambda: SummaryMemory(max_messages=10),
        lambda: VectorMemory(top_k=5),
    ])
    def memory(self, request):
        """创建记忆实例。"""
        factory = request.param
        return factory() if callable(factory) else factory

    def test_has_add_message(self, memory):
        """测试有add_message方法。"""
        assert hasattr(memory, 'add_message')
        assert callable(memory.add_message)

    def test_has_get_messages(self, memory):
        """测试有get_messages方法。"""
        assert hasattr(memory, 'get_messages')
        assert callable(memory.get_messages)

    def test_has_clear(self, memory):
        """测试有clear方法。"""
        assert hasattr(memory, 'clear')
        assert callable(memory.clear)

    def test_add_and_get(self, memory):
        """测试添加和获取。"""
        memory.add_user_message("test")
        msgs = memory.get_messages()
        assert len(msgs) >= 1

    def test_clear_works(self, memory):
        """测试清空有效。"""
        memory.add_user_message("test")
        memory.clear()
        # 清空后消息数应该减少或为0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
