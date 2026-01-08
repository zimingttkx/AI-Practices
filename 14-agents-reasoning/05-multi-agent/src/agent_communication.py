"""
Agent Communication: Message Passing and Communication Protocols.

Core Idea:
    This module implements the communication infrastructure for multi-agent
    systems, including message types, channels, and routing mechanisms.

Mathematical Foundation:
    Communication can be modeled as a graph $G = (V, E)$ where:
    - $V$: Set of agents (nodes)
    - $E$: Communication channels (edges)
    
    Message passing follows: $m: A_i \\to A_j$ with latency $\\tau_{ij}$
    
    Broadcast: $m: A_i \\to \\{A_j | j \\neq i\\}$
    Multicast: $m: A_i \\to S \\subseteq V$

Design Patterns:
    - Observer: Agents subscribe to message channels
    - Mediator: MessageBus coordinates all communication
    - Strategy: Different routing strategies (direct, broadcast, topic-based)

Complexity Analysis:
    - Direct message: O(1)
    - Broadcast: O(n) where n is number of agents
    - Topic subscription: O(k) where k is subscribers

Author: AI-Practices
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import uuid
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
    Awaitable,
)

__all__ = [
    "MessageType",
    "MessagePriority",
    "AgentMessage",
    "MessageHandler",
    "Channel",
    "DirectChannel",
    "BroadcastChannel",
    "TopicChannel",
    "MessageBus",
    "CommunicationProtocol",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_TIMEOUT: Final[float] = 30.0
MAX_MESSAGE_HISTORY: Final[int] = 1000
MESSAGE_ID_PREFIX: Final[str] = "msg"


# =============================================================================
# Enumerations
# =============================================================================


class MessageType(str, Enum):
    """
    Types of messages in the multi-agent system.
    
    Categories:
        - CHAT: Regular conversation messages
        - TASK: Task assignment and status updates
        - SYSTEM: System-level notifications
        - CONTROL: Agent lifecycle control
    """
    # Chat messages
    CHAT = "chat"
    QUERY = "query"
    RESPONSE = "response"
    
    # Task messages
    TASK_ASSIGN = "task_assign"
    TASK_UPDATE = "task_update"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    
    # System messages
    SYSTEM = "system"
    ERROR = "error"
    WARNING = "warning"
    
    # Control messages
    START = "start"
    STOP = "stop"
    RESET = "reset"
    HEARTBEAT = "heartbeat"

    def __str__(self) -> str:
        return self.value


class MessagePriority(int, Enum):
    """Message priority levels for queue ordering."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

    def __lt__(self, other: "MessagePriority") -> bool:
        return self.value < other.value


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AgentMessage:
    """
    Message exchanged between agents.
    
    Attributes:
        content: Message payload (text or structured data)
        sender_id: ID of the sending agent
        sender_name: Name of the sending agent
        msg_type: Type of message
        priority: Message priority
        recipient_id: Target agent ID (None for broadcast)
        topic: Topic for pub/sub routing
        timestamp: Creation timestamp
        metadata: Additional message data
        reply_to: ID of message being replied to
        
    Example:
        >>> msg = AgentMessage(
        ...     content="Please review this code",
        ...     sender_id="agent_001",
        ...     sender_name="Developer",
        ...     msg_type=MessageType.TASK_ASSIGN,
        ...     recipient_id="agent_002"
        ... )
    """
    content: Union[str, Dict[str, Any]]
    sender_id: str
    sender_name: str
    msg_type: MessageType = MessageType.CHAT
    priority: MessagePriority = MessagePriority.NORMAL
    recipient_id: Optional[str] = None
    topic: Optional[str] = None
    id: str = field(default_factory=lambda: f"{MESSAGE_ID_PREFIX}_{uuid.uuid4().hex[:12]}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "msg_type": self.msg_type.value,
            "priority": self.priority.value,
            "recipient_id": self.recipient_id,
            "topic": self.topic,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Deserialize message from dictionary."""
        return cls(
            id=data.get("id", f"{MESSAGE_ID_PREFIX}_{uuid.uuid4().hex[:12]}"),
            content=data["content"],
            sender_id=data["sender_id"],
            sender_name=data["sender_name"],
            msg_type=MessageType(data.get("msg_type", "chat")),
            priority=MessagePriority(data.get("priority", 1)),
            recipient_id=data.get("recipient_id"),
            topic=data.get("topic"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
            reply_to=data.get("reply_to"),
        )

    def create_reply(
        self,
        content: Union[str, Dict[str, Any]],
        sender_id: str,
        sender_name: str,
        **kwargs: Any,
    ) -> "AgentMessage":
        """Create a reply to this message."""
        return AgentMessage(
            content=content,
            sender_id=sender_id,
            sender_name=sender_name,
            msg_type=MessageType.RESPONSE,
            recipient_id=self.sender_id,
            topic=self.topic,
            reply_to=self.id,
            **kwargs,
        )

    def __str__(self) -> str:
        content_preview = str(self.content)[:50]
        if len(str(self.content)) > 50:
            content_preview += "..."
        return f"[{self.sender_name}→{self.recipient_id or 'all'}] {content_preview}"


# =============================================================================
# Message Handler Protocol
# =============================================================================

MessageHandler = Callable[[AgentMessage], Awaitable[Optional[AgentMessage]]]


# =============================================================================
# Channel Abstractions
# =============================================================================


class Channel(ABC):
    """Abstract base class for communication channels."""

    def __init__(self, name: str):
        self.name = name
        self._handlers: List[MessageHandler] = []
        self._message_history: List[AgentMessage] = []

    @abstractmethod
    async def send(self, message: AgentMessage) -> None:
        """Send a message through the channel."""
        pass

    def subscribe(self, handler: MessageHandler) -> None:
        """Subscribe a handler to receive messages."""
        if handler not in self._handlers:
            self._handlers.append(handler)
            logger.debug(f"Handler subscribed to channel {self.name}")

    def unsubscribe(self, handler: MessageHandler) -> None:
        """Unsubscribe a handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def get_history(self, limit: int = 100) -> List[AgentMessage]:
        """Get recent message history."""
        return self._message_history[-limit:]

    def _record_message(self, message: AgentMessage) -> None:
        """Record message in history."""
        self._message_history.append(message)
        if len(self._message_history) > MAX_MESSAGE_HISTORY:
            self._message_history = self._message_history[-MAX_MESSAGE_HISTORY:]


class DirectChannel(Channel):
    """
    Point-to-point communication channel.
    
    Messages are delivered only to the specified recipient.
    """

    def __init__(self, name: str = "direct"):
        super().__init__(name)
        self._agent_handlers: Dict[str, MessageHandler] = {}

    def register_agent(self, agent_id: str, handler: MessageHandler) -> None:
        """Register an agent's message handler."""
        self._agent_handlers[agent_id] = handler
        logger.debug(f"Agent {agent_id} registered on direct channel")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._agent_handlers.pop(agent_id, None)

    async def send(self, message: AgentMessage) -> None:
        """Send message to specific recipient."""
        self._record_message(message)
        
        if message.recipient_id is None:
            logger.warning("DirectChannel requires recipient_id")
            return
        
        handler = self._agent_handlers.get(message.recipient_id)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error delivering message to {message.recipient_id}: {e}")
        else:
            logger.warning(f"No handler for recipient {message.recipient_id}")


class BroadcastChannel(Channel):
    """
    Broadcast communication channel.
    
    Messages are delivered to all subscribed handlers except the sender.
    """

    def __init__(self, name: str = "broadcast"):
        super().__init__(name)
        self._agent_handlers: Dict[str, MessageHandler] = {}

    def register_agent(self, agent_id: str, handler: MessageHandler) -> None:
        """Register an agent's message handler."""
        self._agent_handlers[agent_id] = handler

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._agent_handlers.pop(agent_id, None)

    async def send(self, message: AgentMessage) -> None:
        """Broadcast message to all agents except sender."""
        self._record_message(message)
        
        tasks = []
        for agent_id, handler in self._agent_handlers.items():
            if agent_id != message.sender_id:
                tasks.append(self._deliver(handler, message, agent_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _deliver(
        self, handler: MessageHandler, message: AgentMessage, agent_id: str
    ) -> None:
        """Deliver message to a single handler."""
        try:
            await handler(message)
        except Exception as e:
            logger.error(f"Error broadcasting to {agent_id}: {e}")


class TopicChannel(Channel):
    """
    Topic-based publish/subscribe channel.
    
    Agents subscribe to specific topics and receive only relevant messages.
    """

    def __init__(self, name: str = "topic"):
        super().__init__(name)
        self._topic_subscribers: Dict[str, Dict[str, MessageHandler]] = defaultdict(dict)

    def subscribe_topic(self, topic: str, agent_id: str, handler: MessageHandler) -> None:
        """Subscribe an agent to a specific topic."""
        self._topic_subscribers[topic][agent_id] = handler
        logger.debug(f"Agent {agent_id} subscribed to topic '{topic}'")

    def unsubscribe_topic(self, topic: str, agent_id: str) -> None:
        """Unsubscribe an agent from a topic."""
        if topic in self._topic_subscribers:
            self._topic_subscribers[topic].pop(agent_id, None)

    def get_topics(self) -> List[str]:
        """Get all active topics."""
        return list(self._topic_subscribers.keys())

    async def send(self, message: AgentMessage) -> None:
        """Send message to all subscribers of the message's topic."""
        self._record_message(message)
        
        if message.topic is None:
            logger.warning("TopicChannel requires message.topic")
            return
        
        subscribers = self._topic_subscribers.get(message.topic, {})
        tasks = []
        for agent_id, handler in subscribers.items():
            if agent_id != message.sender_id:
                tasks.append(self._deliver(handler, message, agent_id))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _deliver(
        self, handler: MessageHandler, message: AgentMessage, agent_id: str
    ) -> None:
        """Deliver message to a single handler."""
        try:
            await handler(message)
        except Exception as e:
            logger.error(f"Error delivering to {agent_id} on topic {message.topic}: {e}")


# =============================================================================
# Message Bus (Central Communication Hub)
# =============================================================================


class MessageBus:
    """
    Central message routing hub for multi-agent communication.
    
    Core Idea:
        MessageBus acts as a mediator, routing messages through appropriate
        channels based on message properties (recipient, topic, type).
    
    Architecture:
        ```
        Agent A ──┐                    ┌── Agent B
                  │    ┌──────────┐    │
        Agent C ──┼───>│MessageBus│<───┼── Agent D
                  │    └──────────┘    │
        Agent E ──┘         │          └── Agent F
                            ▼
                    [Direct|Broadcast|Topic]
        ```
    
    Example:
        >>> bus = MessageBus()
        >>> bus.register_agent(agent)
        >>> await bus.send(message)
    """

    def __init__(self) -> None:
        self.direct_channel = DirectChannel()
        self.broadcast_channel = BroadcastChannel()
        self.topic_channel = TopicChannel()
        self._agents: Dict[str, Any] = {}  # agent_id -> agent reference
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._running = False
        self._stats = {"sent": 0, "delivered": 0, "failed": 0}

    def register_agent(
        self,
        agent_id: str,
        agent_name: str,
        handler: MessageHandler,
        topics: Optional[List[str]] = None,
    ) -> None:
        """
        Register an agent with the message bus.
        
        Args:
            agent_id: Unique agent identifier
            agent_name: Agent display name
            handler: Async function to handle incoming messages
            topics: Optional list of topics to subscribe to
        """
        self._agents[agent_id] = {"name": agent_name, "handler": handler}
        self.direct_channel.register_agent(agent_id, handler)
        self.broadcast_channel.register_agent(agent_id, handler)
        
        for topic in (topics or []):
            self.topic_channel.subscribe_topic(topic, agent_id, handler)
        
        logger.info(f"Agent {agent_name} ({agent_id}) registered with MessageBus")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from all channels."""
        self._agents.pop(agent_id, None)
        self.direct_channel.unregister_agent(agent_id)
        self.broadcast_channel.unregister_agent(agent_id)
        for topic in self.topic_channel.get_topics():
            self.topic_channel.unsubscribe_topic(topic, agent_id)

    async def send(self, message: AgentMessage) -> None:
        """
        Route and send a message through appropriate channel.
        
        Routing logic:
        1. If recipient_id specified -> DirectChannel
        2. If topic specified -> TopicChannel
        3. Otherwise -> BroadcastChannel
        """
        self._stats["sent"] += 1
        
        try:
            if message.recipient_id:
                await self.direct_channel.send(message)
            elif message.topic:
                await self.topic_channel.send(message)
            else:
                await self.broadcast_channel.send(message)
            self._stats["delivered"] += 1
        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"Message delivery failed: {e}")
            raise

    async def send_direct(
        self,
        content: str,
        sender_id: str,
        sender_name: str,
        recipient_id: str,
        msg_type: MessageType = MessageType.CHAT,
        **kwargs: Any,
    ) -> AgentMessage:
        """Convenience method for sending direct messages."""
        message = AgentMessage(
            content=content,
            sender_id=sender_id,
            sender_name=sender_name,
            recipient_id=recipient_id,
            msg_type=msg_type,
            **kwargs,
        )
        await self.send(message)
        return message

    async def broadcast(
        self,
        content: str,
        sender_id: str,
        sender_name: str,
        msg_type: MessageType = MessageType.CHAT,
        **kwargs: Any,
    ) -> AgentMessage:
        """Convenience method for broadcasting messages."""
        message = AgentMessage(
            content=content,
            sender_id=sender_id,
            sender_name=sender_name,
            msg_type=msg_type,
            **kwargs,
        )
        await self.send(message)
        return message

    def get_stats(self) -> Dict[str, int]:
        """Get message statistics."""
        return self._stats.copy()

    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent IDs."""
        return list(self._agents.keys())

    def get_history(self, channel: str = "all", limit: int = 100) -> List[AgentMessage]:
        """Get message history from specified channel."""
        if channel == "direct":
            return self.direct_channel.get_history(limit)
        elif channel == "broadcast":
            return self.broadcast_channel.get_history(limit)
        elif channel == "topic":
            return self.topic_channel.get_history(limit)
        else:
            # Combine all histories
            all_msgs = (
                self.direct_channel.get_history(limit) +
                self.broadcast_channel.get_history(limit) +
                self.topic_channel.get_history(limit)
            )
            all_msgs.sort(key=lambda m: m.timestamp)
            return all_msgs[-limit:]


# =============================================================================
# Communication Protocol
# =============================================================================


class CommunicationProtocol:
    """
    High-level communication protocol for agent interactions.
    
    Provides structured communication patterns:
    - Request-Response
    - Publish-Subscribe
    - Round-Robin
    """

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self._pending_responses: Dict[str, asyncio.Future] = {}

    async def request(
        self,
        content: str,
        sender_id: str,
        sender_name: str,
        recipient_id: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Optional[AgentMessage]:
        """
        Send a request and wait for response.
        
        Args:
            content: Request content
            sender_id: Sender agent ID
            sender_name: Sender agent name
            recipient_id: Target agent ID
            timeout: Response timeout in seconds
            
        Returns:
            Response message or None if timeout
        """
        message = AgentMessage(
            content=content,
            sender_id=sender_id,
            sender_name=sender_name,
            recipient_id=recipient_id,
            msg_type=MessageType.QUERY,
        )
        
        future: asyncio.Future = asyncio.Future()
        self._pending_responses[message.id] = future
        
        try:
            await self.bus.send(message)
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Request {message.id} timed out")
            return None
        finally:
            self._pending_responses.pop(message.id, None)

    def handle_response(self, message: AgentMessage) -> bool:
        """
        Handle an incoming response message.
        
        Returns:
            True if response was matched to a pending request
        """
        if message.reply_to and message.reply_to in self._pending_responses:
            future = self._pending_responses[message.reply_to]
            if not future.done():
                future.set_result(message)
            return True
        return False

    async def publish(
        self,
        topic: str,
        content: str,
        sender_id: str,
        sender_name: str,
    ) -> AgentMessage:
        """Publish a message to a topic."""
        message = AgentMessage(
            content=content,
            sender_id=sender_id,
            sender_name=sender_name,
            topic=topic,
            msg_type=MessageType.CHAT,
        )
        await self.bus.send(message)
        return message
