"""
记忆模块 (Memory Module)

============================================================
核心思想 (Core Idea)
============================================================
对话记忆是Agent保持上下文连贯性的关键组件。通过不同的记忆策略，
Agent可以在有限的上下文窗口内保留最相关的历史信息。

============================================================
数学基础 (Mathematical Foundation)
============================================================
记忆管理可形式化为信息压缩问题：

    M: History → Context

不同策略的信息保留率：
    - BufferMemory: 完整保留，I(Context; History) = H(History)
    - WindowMemory: 截断保留，I ≈ H(History[-k:])
    - SummaryMemory: 压缩保留，I ≈ H(Summary) + H(Recent)
    - VectorMemory: 语义检索，I ≈ H(TopK(sim(query, History)))

Token预算约束：
    |Context| ≤ max_tokens

============================================================
算法流程 (Algorithm Flow)
============================================================
1. 消息添加: memory.add_message(msg)
2. 容量检查: 检查是否超过限制
3. 压缩/截断: 根据策略处理溢出
4. 上下文构建: memory.get_messages()
5. 发送给LLM: llm(context + new_query)

============================================================
参考文献 (References)
============================================================
[1] Park, J.S., et al. (2023). Generative Agents: Interactive Simulacra
    of Human Behavior. arXiv:2304.03442.
[2] Zhong, W., et al. (2024). MemoryBank: Enhancing Large Language Models
    with Long-Term Memory. AAAI 2024.
[3] Wang, L., et al. (2023). Augmenting Language Models with Long-Term
    Memory. arXiv:2306.07174.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib


__all__ = [
    "Message",
    "MessageRole",
    "ConversationMemory",
    "BufferMemory",
    "WindowMemory",
    "SummaryMemory",
    "VectorMemory",
]


class MessageRole(Enum):
    """消息角色。"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """对话消息。

    参数：
        role: 消息角色
        content: 消息内容
        name: 发送者名称（可选）
        tool_call_id: 工具调用ID（工具消息时使用）
        metadata: 额外元数据
        timestamp: 消息时间戳
    """
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        # 允许字符串角色
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI消息格式。"""
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建消息。"""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def system(cls, content: str) -> "Message":
        """创建系统消息。"""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """创建用户消息。"""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """创建助手消息。"""
        return cls(role=MessageRole.ASSISTANT, content=content)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> "Message":
        """创建工具消息。"""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )

    @property
    def token_estimate(self) -> int:
        """估算token数量（粗略估计）。"""
        # 粗略估计：中文约1.5字符/token，英文约4字符/token
        return len(self.content) // 3

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message(role={self.role.value}, content='{preview}')"


class ConversationMemory(ABC):
    """对话记忆基类。

    所有记忆实现必须继承此类。

    记忆管理原则：
        1. 保持上下文连贯性
        2. 控制token使用量
        3. 保留关键信息
    """

    @abstractmethod
    def add_message(self, message: Message) -> None:
        """添加消息到记忆。

        参数：
            message: 消息对象
        """
        pass

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """获取所有记忆中的消息。

        返回：
            消息列表
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空记忆。"""
        pass

    def add_user_message(self, content: str) -> None:
        """添加用户消息。"""
        self.add_message(Message.user(content))

    def add_assistant_message(self, content: str) -> None:
        """添加助手消息。"""
        self.add_message(Message.assistant(content))

    def add_system_message(self, content: str) -> None:
        """添加系统消息。"""
        self.add_message(Message.system(content))

    def get_messages_as_dicts(self) -> List[Dict[str, Any]]:
        """获取消息的字典格式。"""
        return [msg.to_dict() for msg in self.get_messages()]

    @property
    def token_count(self) -> int:
        """估算总token数。"""
        return sum(msg.token_estimate for msg in self.get_messages())

    @property
    def message_count(self) -> int:
        """消息数量。"""
        return len(self.get_messages())


class BufferMemory(ConversationMemory):
    """缓冲记忆。

    保存所有对话历史，适用于短对话。

    注意：
        长对话可能导致token超限。

    示例：
        >>> memory = BufferMemory()
        >>> memory.add_user_message("你好")
        >>> memory.add_assistant_message("你好！有什么可以帮助你的？")
    """

    def __init__(self, system_message: Optional[str] = None) -> None:
        """初始化缓冲记忆。

        参数：
            system_message: 系统提示（可选）
        """
        self._messages: List[Message] = []
        if system_message:
            self._messages.append(Message.system(system_message))

    def add_message(self, message: Message) -> None:
        """添加消息。"""
        self._messages.append(message)

    def get_messages(self) -> List[Message]:
        """获取所有消息。"""
        return self._messages.copy()

    def clear(self) -> None:
        """清空记忆（保留系统消息）。"""
        system_msgs = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        self._messages = system_msgs

    def __repr__(self) -> str:
        return f"BufferMemory(messages={len(self._messages)})"


class WindowMemory(ConversationMemory):
    """窗口记忆。

    只保留最近k轮对话，控制上下文长度。

    数学原理：
        保留最近 2k 条消息（k轮 = k条用户 + k条助手）

    示例：
        >>> memory = WindowMemory(k=5)  # 保留最近5轮
    """

    def __init__(
        self,
        k: int = 5,
        system_message: Optional[str] = None,
    ) -> None:
        """初始化窗口记忆。

        参数：
            k: 保留的对话轮数
            system_message: 系统提示
        """
        if k <= 0:
            raise ValueError(f"k必须为正数，得到 {k}")
        self._k = k
        self._messages: List[Message] = []
        self._system_message: Optional[Message] = None
        if system_message:
            self._system_message = Message.system(system_message)

    def add_message(self, message: Message) -> None:
        """添加消息并维护窗口大小。"""
        # 系统消息单独处理
        if message.role == MessageRole.SYSTEM:
            self._system_message = message
            return
        
        self._messages.append(message)
        # 保留最近2k条消息（k轮对话）
        max_messages = self._k * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def get_messages(self) -> List[Message]:
        """获取消息（包含系统消息）。"""
        result = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._messages)
        return result

    def clear(self) -> None:
        """清空记忆（保留系统消息）。"""
        self._messages = []

    @property
    def window_size(self) -> int:
        """当前窗口大小。"""
        return len(self._messages)

    def __repr__(self) -> str:
        return f"WindowMemory(k={self._k}, current={len(self._messages)})"


class SummaryMemory(ConversationMemory):
    """摘要记忆。

    当对话过长时，自动压缩历史为摘要。

    工作原理：
        1. 保留最近的消息
        2. 将旧消息压缩为摘要
        3. 摘要作为上下文前缀

    示例：
        >>> memory = SummaryMemory(max_messages=10)
    """

    def __init__(
        self,
        max_messages: int = 10,
        summarizer: Optional[Callable[[List[Message]], str]] = None,
        system_message: Optional[str] = None,
    ) -> None:
        """初始化摘要记忆。

        参数：
            max_messages: 保留的最大消息数
            summarizer: 摘要生成函数
            system_message: 系统提示
        """
        if max_messages <= 0:
            raise ValueError(f"max_messages必须为正数，得到 {max_messages}")
        self._max_messages = max_messages
        self._summarizer = summarizer or self._default_summarizer
        self._messages: List[Message] = []
        self._summary: str = ""
        self._system_message: Optional[Message] = None
        if system_message:
            self._system_message = Message.system(system_message)

    def _default_summarizer(self, messages: List[Message]) -> str:
        """默认摘要生成器（简单拼接）。"""
        parts = []
        for msg in messages:
            role = msg.role.value
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            parts.append(f"{role}: {content}")
        return "对话摘要:\n" + "\n".join(parts)

    def add_message(self, message: Message) -> None:
        """添加消息并在需要时生成摘要。"""
        if message.role == MessageRole.SYSTEM:
            self._system_message = message
            return

        self._messages.append(message)

        # 超过限制时生成摘要
        if len(self._messages) > self._max_messages:
            # 取出一半旧消息生成摘要
            half = len(self._messages) // 2
            old_messages = self._messages[:half]
            self._messages = self._messages[half:]
            # 更新摘要
            new_summary = self._summarizer(old_messages)
            if self._summary:
                self._summary = f"{self._summary}\n\n{new_summary}"
            else:
                self._summary = new_summary

    def get_messages(self) -> List[Message]:
        """获取消息（包含摘要）。"""
        result = []
        if self._system_message:
            result.append(self._system_message)
        if self._summary:
            result.append(Message.system(f"[历史摘要]\n{self._summary}"))
        result.extend(self._messages)
        return result

    def clear(self) -> None:
        """清空记忆。"""
        self._messages = []
        self._summary = ""

    @property
    def summary(self) -> str:
        """当前摘要。"""
        return self._summary

    def __repr__(self) -> str:
        return f"SummaryMemory(messages={len(self._messages)}, has_summary={bool(self._summary)})"


class VectorMemory(ConversationMemory):
    """向量记忆。

    使用向量相似度检索相关历史消息。

    工作原理：
        1. 将消息转换为向量
        2. 存储到向量索引
        3. 查询时检索最相关的消息

    示例：
        >>> memory = VectorMemory(embed_fn=my_embed_fn)
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        top_k: int = 5,
        system_message: Optional[str] = None,
    ) -> None:
        """初始化向量记忆。

        参数：
            embed_fn: 嵌入函数
            top_k: 检索的消息数量
            system_message: 系统提示
        """
        if top_k <= 0:
            raise ValueError(f"top_k必须为正数，得到 {top_k}")
        self._embed_fn = embed_fn or self._default_embed
        self._top_k = top_k
        self._messages: List[Message] = []
        self._embeddings: List[List[float]] = []
        self._system_message: Optional[Message] = None
        self._recent_messages: List[Message] = []  # 最近消息
        if system_message:
            self._system_message = Message.system(system_message)

    def _default_embed(self, text: str) -> List[float]:
        """默认嵌入函数（基于哈希的简单实现）。"""
        # 简单的哈希嵌入（仅用于演示）
        hash_bytes = hashlib.md5(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_bytes]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add_message(self, message: Message) -> None:
        """添加消息并生成嵌入。"""
        if message.role == MessageRole.SYSTEM:
            self._system_message = message
            return

        # 添加到消息列表
        self._messages.append(message)
        # 生成嵌入
        embedding = self._embed_fn(message.content)
        self._embeddings.append(embedding)
        # 更新最近消息
        self._recent_messages.append(message)
        if len(self._recent_messages) > 4:
            self._recent_messages = self._recent_messages[-4:]

    def get_messages(self) -> List[Message]:
        """获取消息（最近消息）。"""
        result = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._recent_messages)
        return result

    def retrieve(self, query: str) -> List[Message]:
        """检索与查询相关的消息。

        参数：
            query: 查询文本

        返回：
            相关消息列表
        """
        if not self._messages:
            return []

        # 生成查询嵌入
        query_embedding = self._embed_fn(query)

        # 计算相似度
        similarities = [
            (i, self._cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(self._embeddings)
        ]

        # 排序并取top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in similarities[:self._top_k]]

        return [self._messages[i] for i in top_indices]

    def get_context_for_query(self, query: str) -> List[Message]:
        """获取查询的上下文消息。"""
        result = []
        if self._system_message:
            result.append(self._system_message)
        # 添加检索到的相关消息
        relevant = self.retrieve(query)
        if relevant:
            result.append(Message.system("[相关历史]\n" + "\n".join(
                f"- {m.role.value}: {m.content[:100]}" for m in relevant
            )))
        # 添加最近消息
        result.extend(self._recent_messages)
        return result

    def clear(self) -> None:
        """清空记忆。"""
        self._messages = []
        self._embeddings = []
        self._recent_messages = []

    @property
    def total_messages(self) -> int:
        """总消息数。"""
        return len(self._messages)

    def __repr__(self) -> str:
        return f"VectorMemory(total={len(self._messages)}, top_k={self._top_k})"
