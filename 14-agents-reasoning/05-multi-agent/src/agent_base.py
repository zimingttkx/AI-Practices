"""
Multi-Agent System Base: Core Abstractions for Agent Interactions.

Core Idea:
    This module defines the foundational abstractions for multi-agent systems,
    including the base Agent class, role definitions, and lifecycle management.
    It provides a standardized interface for agents to perceive, reason, and act.

Mathematical Foundation:
    An agent can be modeled as a tuple $A = \\langle S, P, M, \\pi \\rangle$:
    - $S$: Internal state space
    - $P$: Perception function $P: \\text{Env} \\to \\text{Observation}$
    - $M$: Memory module $M: \\text{History} \\to \\text{Context}$
    - $\\pi$: Policy function $\\pi: S \\times O \\to \\text{Action}$

    Agent interaction follows the cycle:
    $$o_t = P(\\text{env}_t) \\to s_t = M(o_t, s_{t-1}) \\to a_t = \\pi(s_t)$$

Design Patterns:
    - Template Method: Defines the `step` workflow (perceive -> think -> act)
    - State Pattern: Explicit state management for lifecycle control
    - Factory Pattern: `create_agent` for flexible instantiation

Complexity Analysis:
    - Agent creation: O(1)
    - Message processing: O(m) where m is memory size
    - State transitions: O(1)

References:
    - Wooldridge & Jennings (1995): Intelligent Agents: Theory and Practice
    - Russell & Norvig: Artificial Intelligence: A Modern Approach
    - AutoGen: https://github.com/microsoft/autogen

Author: AI-Practices
Version: 1.0.0
"""

from __future__ import annotations

import uuid
import logging
import asyncio
from abc import ABC, abstractmethod
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
    Protocol,
    Set,
    TypeVar,
    Union,
    runtime_checkable,
)

__all__ = [
    "AgentRole",
    "AgentState",
    "AgentConfig",
    "BaseAgent",
    "SimpleAgent",
    "LLMInterface",
    "create_agent",
]

# =============================================================================
# Constants and Configuration
# =============================================================================

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS: Final[int] = 10
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_MAX_TOKENS: Final[int] = 1000
AGENT_ID_PREFIX: Final[str] = "agent"


# =============================================================================
# Enumerations
# =============================================================================


class AgentRole(str, Enum):
    """
    Defines the functional role of an agent in the system.
    
    Each role implies different capabilities and behaviors:
    - ASSISTANT: General-purpose helper, responds to queries
    - USER_PROXY: Acts on behalf of human user, can execute code
    - DEBATER: Participates in structured debates
    - CRITIC: Evaluates and critiques other agents' outputs
    - MANAGER: Orchestrates and coordinates other agents
    - RESEARCHER: Gathers and synthesizes information
    - CODER: Specializes in code generation and review
    - PLANNER: Creates and manages execution plans
    """
    ASSISTANT = "assistant"
    USER_PROXY = "user_proxy"
    DEBATER = "debater"
    CRITIC = "critic"
    MANAGER = "manager"
    RESEARCHER = "researcher"
    CODER = "coder"
    PLANNER = "planner"

    def __str__(self) -> str:
        return self.value


class AgentState(str, Enum):
    """
    Lifecycle states of an agent with well-defined transitions.
    
    State Machine:
        IDLE ──receive()──> THINKING ──generate()──> SPEAKING ──send()──> IDLE
          │                    │                        │
          │                    └──error()──> ERROR ─────┘
          │                                    │
          └──terminate()──> TERMINATED <───────┘
    
    States:
        IDLE: Waiting for input, ready to process
        THINKING: Processing input, reasoning
        SPEAKING: Generating response
        EXECUTING: Running tools or actions
        WAITING: Waiting for external response
        ERROR: Encountered a recoverable failure
        TERMINATED: Permanently shut down
    """
    IDLE = "idle"
    THINKING = "thinking"
    SPEAKING = "speaking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"

    def __str__(self) -> str:
        return self.value

    def can_transition_to(self, target: "AgentState") -> bool:
        """Check if transition to target state is valid."""
        valid_transitions: Dict[AgentState, Set[AgentState]] = {
            AgentState.IDLE: {AgentState.THINKING, AgentState.TERMINATED},
            AgentState.THINKING: {AgentState.SPEAKING, AgentState.EXECUTING, 
                                  AgentState.ERROR, AgentState.IDLE},
            AgentState.SPEAKING: {AgentState.IDLE, AgentState.WAITING, AgentState.ERROR},
            AgentState.EXECUTING: {AgentState.THINKING, AgentState.IDLE, AgentState.ERROR},
            AgentState.WAITING: {AgentState.THINKING, AgentState.IDLE, AgentState.ERROR},
            AgentState.ERROR: {AgentState.IDLE, AgentState.TERMINATED},
            AgentState.TERMINATED: set(),
        }
        return target in valid_transitions.get(self, set())


# =============================================================================
# Protocols and Interfaces
# =============================================================================


@runtime_checkable
class LLMInterface(Protocol):
    """Protocol for LLM backends that agents can use."""
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the LLM."""
        ...


class MockLLM:
    """Mock LLM for testing purposes."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ["This is a mock response."]
        self._call_count = 0
        self.call_history: List[List[Dict[str, str]]] = []
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """Return mock response."""
        self.call_history.append(messages)
        response = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return response


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AgentConfig:
    """
    Configuration parameters for an agent.
    
    Attributes:
        name: Human-readable agent name
        role: Functional role determining behavior
        system_prompt: Initial instructions for the agent
        model_name: LLM model identifier
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum response length
        max_iterations: Maximum reasoning iterations
        capabilities: Set of capability tags
        metadata: Additional configuration data
    
    Example:
        >>> config = AgentConfig(
        ...     name="CodeReviewer",
        ...     role=AgentRole.CRITIC,
        ...     system_prompt="You are an expert code reviewer.",
        ...     capabilities={"code_review", "python", "testing"}
        ... )
    """
    name: str
    role: AgentRole = AgentRole.ASSISTANT
    system_prompt: Optional[str] = None
    model_name: str = "gpt-4"
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.name:
            raise ValueError("Agent name cannot be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be in [0.0, 2.0], got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass
class AgentResponse:
    """
    Structured response from an agent.
    
    Attributes:
        content: The main response text
        agent_id: ID of the responding agent
        agent_name: Name of the responding agent
        timestamp: When the response was generated
        metadata: Additional response data (tool calls, reasoning, etc.)
        is_final: Whether this is a final response or intermediate
    """
    content: str
    agent_id: str
    agent_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_final: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize response to dictionary."""
        return {
            "content": self.content,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "is_final": self.is_final,
        }


# =============================================================================
# Base Agent Class
# =============================================================================


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Core Idea:
        BaseAgent defines the fundamental interface and lifecycle for agents.
        Subclasses implement specific reasoning and action strategies.
    
    Template Method Pattern:
        The `step` method defines the skeleton algorithm:
        1. receive() - Accept and store input
        2. think() - Process and reason (abstract)
        3. act() - Generate response (abstract)
    
    Attributes:
        id: Unique identifier (UUID)
        config: Agent configuration
        state: Current lifecycle state
        conversation_history: List of messages
        llm: Language model interface
    
    Example:
        >>> class MyAgent(BaseAgent):
        ...     async def think(self, input_text: str) -> str:
        ...         return f"Thinking about: {input_text}"
        ...     async def act(self, thought: str) -> str:
        ...         return f"Action based on: {thought}"
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: Optional[LLMInterface] = None,
    ) -> None:
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
            llm: Optional LLM interface for generation
        """
        self.id = f"{AGENT_ID_PREFIX}_{uuid.uuid4().hex[:12]}"
        self.config = config
        self._state = AgentState.IDLE
        self.llm = llm or MockLLM()
        self.conversation_history: List[Dict[str, str]] = []
        self._iteration_count = 0
        self._created_at = datetime.utcnow()
        self._last_active = self._created_at
        
        # Initialize with system prompt if provided
        if config.system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": config.system_prompt
            })
        
        logger.info(f"Agent created: {self.name} (id={self.id}, role={self.role})")

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Agent's display name."""
        return self.config.name

    @property
    def role(self) -> AgentRole:
        """Agent's functional role."""
        return self.config.role

    @property
    def state(self) -> AgentState:
        """Current lifecycle state."""
        return self._state

    @property
    def capabilities(self) -> Set[str]:
        """Agent's capability tags."""
        return self.config.capabilities

    @property
    def is_active(self) -> bool:
        """Whether agent can process messages."""
        return self._state not in {AgentState.TERMINATED, AgentState.ERROR}

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def set_state(self, new_state: AgentState) -> None:
        """
        Transition to a new state with validation.
        
        Args:
            new_state: Target state
            
        Raises:
            ValueError: If transition is invalid
        """
        if not self._state.can_transition_to(new_state):
            raise ValueError(
                f"Invalid state transition: {self._state} -> {new_state}"
            )
        old_state = self._state
        self._state = new_state
        self._last_active = datetime.utcnow()
        logger.debug(f"Agent {self.name}: {old_state} -> {new_state}")

    def reset(self) -> None:
        """Reset agent to initial state, clearing history."""
        self.conversation_history.clear()
        if self.config.system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": self.config.system_prompt
            })
        self._state = AgentState.IDLE
        self._iteration_count = 0
        logger.info(f"Agent {self.name} reset")

    def terminate(self) -> None:
        """Permanently terminate the agent."""
        self._state = AgentState.TERMINATED
        logger.info(f"Agent {self.name} terminated")

    # -------------------------------------------------------------------------
    # Message Handling
    # -------------------------------------------------------------------------

    def receive(self, message: str, sender: str = "user") -> None:
        """
        Receive a message and add to conversation history.
        
        Args:
            message: Message content
            sender: Name of the sender
        """
        role = "user" if sender == "user" else "user"
        content = f"[{sender}]: {message}" if sender != "user" else message
        
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        logger.debug(f"Agent {self.name} received message from {sender}")

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get conversation history formatted for LLM."""
        return self.conversation_history.copy()

    # -------------------------------------------------------------------------
    # Abstract Methods (Template Method Pattern)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def think(self, input_text: str) -> str:
        """
        Process input and generate internal reasoning.
        
        Args:
            input_text: The input to reason about
            
        Returns:
            Internal reasoning or processed thought
        """
        pass

    @abstractmethod
    async def act(self, thought: str) -> str:
        """
        Generate action/response based on reasoning.
        
        Args:
            thought: The reasoning from think()
            
        Returns:
            The action or response to take
        """
        pass

    # -------------------------------------------------------------------------
    # Main Interaction Loop
    # -------------------------------------------------------------------------

    async def step(self, input_message: str, sender: str = "user") -> AgentResponse:
        """
        Execute one interaction cycle: receive -> think -> act.
        
        Args:
            input_message: Input from user or another agent
            sender: Name of the message sender
            
        Returns:
            AgentResponse containing the agent's response
            
        Raises:
            RuntimeError: If agent is not in a valid state
        """
        if not self.is_active:
            raise RuntimeError(f"Agent {self.name} is not active (state={self.state})")
        
        try:
            # Receive
            self.receive(input_message, sender)
            self.set_state(AgentState.THINKING)
            
            # Think
            thought = await self.think(input_message)
            self.set_state(AgentState.SPEAKING)
            
            # Act
            response_content = await self.act(thought)
            
            # Record response
            self.conversation_history.append({
                "role": "assistant",
                "content": response_content
            })
            
            self._iteration_count += 1
            self.set_state(AgentState.IDLE)
            
            return AgentResponse(
                content=response_content,
                agent_id=self.id,
                agent_name=self.name,
                metadata={"iteration": self._iteration_count}
            )
            
        except Exception as e:
            logger.error(f"Agent {self.name} error: {e}")
            self._state = AgentState.ERROR
            return AgentResponse(
                content=f"Error: {str(e)}",
                agent_id=self.id,
                agent_name=self.name,
                metadata={"error": str(e)},
                is_final=True
            )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_history(self) -> List[Dict[str, str]]:
        """Get full conversation history."""
        return self.conversation_history.copy()

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, role={self.role}, state={self.state})"

    def __str__(self) -> str:
        return f"Agent[{self.name}]"


# =============================================================================
# Concrete Agent Implementations
# =============================================================================


class SimpleAgent(BaseAgent):
    """
    A simple LLM-based agent that uses direct generation.
    
    Core Idea:
        SimpleAgent provides a straightforward implementation where
        thinking and acting are combined into a single LLM call.
    
    Example:
        >>> config = AgentConfig(name="Assistant", system_prompt="You are helpful.")
        >>> agent = SimpleAgent(config)
        >>> response = await agent.step("Hello!")
        >>> print(response.content)
    """

    async def think(self, input_text: str) -> str:
        """Pass through input for processing."""
        return input_text

    async def act(self, thought: str) -> str:
        """Generate response using LLM."""
        messages = self.get_messages_for_llm()
        response = await self.llm.generate(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response


class ReActAgent(BaseAgent):
    """
    Agent implementing ReAct (Reasoning + Acting) pattern.
    
    Core Idea:
        Alternates between reasoning (thought) and acting (tool use),
        enabling more complex multi-step problem solving.
    
    Pattern:
        Thought -> Action -> Observation -> Thought -> ... -> Final Answer
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: Optional[LLMInterface] = None,
        tools: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__(config, llm)
        self.tools = {tool.__name__: tool for tool in (tools or [])}
        self._reasoning_trace: List[Dict[str, str]] = []

    async def think(self, input_text: str) -> str:
        """Generate reasoning about the input."""
        think_prompt = f"""Given the input, think step by step about how to respond.

Input: {input_text}

Think about:
1. What is being asked?
2. What information or actions are needed?
3. How should I respond?

Thought:"""
        
        messages = self.get_messages_for_llm() + [
            {"role": "user", "content": think_prompt}
        ]
        thought = await self.llm.generate(messages=messages, temperature=0.3)
        self._reasoning_trace.append({"type": "thought", "content": thought})
        return thought

    async def act(self, thought: str) -> str:
        """Generate action based on reasoning."""
        act_prompt = f"""Based on your reasoning, provide a helpful response.

Your reasoning: {thought}

Response:"""
        
        messages = self.get_messages_for_llm() + [
            {"role": "user", "content": act_prompt}
        ]
        response = await self.llm.generate(
            messages=messages,
            temperature=self.config.temperature,
        )
        self._reasoning_trace.append({"type": "action", "content": response})
        return response

    def get_reasoning_trace(self) -> List[Dict[str, str]]:
        """Get the full reasoning trace."""
        return self._reasoning_trace.copy()


# =============================================================================
# Factory Functions
# =============================================================================


def create_agent(
    name: str,
    role: Union[str, AgentRole] = AgentRole.ASSISTANT,
    agent_type: str = "simple",
    system_prompt: Optional[str] = None,
    llm: Optional[LLMInterface] = None,
    **kwargs: Any,
) -> BaseAgent:
    """
    Factory function to create agents.
    
    Args:
        name: Agent name
        role: Agent role (string or AgentRole enum)
        agent_type: Type of agent ("simple", "react")
        system_prompt: System instructions
        llm: LLM interface
        **kwargs: Additional config parameters
        
    Returns:
        Configured agent instance
        
    Example:
        >>> agent = create_agent(
        ...     name="Helper",
        ...     role="assistant",
        ...     system_prompt="You are a helpful assistant."
        ... )
    """
    if isinstance(role, str):
        role = AgentRole(role)
    
    config = AgentConfig(
        name=name,
        role=role,
        system_prompt=system_prompt,
        **kwargs
    )
    
    agent_classes: Dict[str, type] = {
        "simple": SimpleAgent,
        "react": ReActAgent,
    }
    
    if agent_type not in agent_classes:
        available = ", ".join(agent_classes.keys())
        raise ValueError(f"Unknown agent_type: {agent_type}. Available: {available}")
    
    return agent_classes[agent_type](config, llm)
