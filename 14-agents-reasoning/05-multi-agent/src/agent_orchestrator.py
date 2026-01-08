"""
Agent Orchestrator: Coordination and Task Distribution for Multi-Agent Systems.

Core Idea:
    The orchestrator manages agent coordination, task assignment, and execution
    flow in multi-agent systems. It acts as a central coordinator that routes
    tasks to appropriate agents based on capabilities and availability.

Mathematical Foundation:
    Task assignment can be modeled as an optimization problem:
    
    $$\\min_{x_{ij}} \\sum_{i,j} c_{ij} \\cdot x_{ij}$$
    
    Subject to:
    - Each task assigned to exactly one agent: $\\sum_j x_{ij} = 1$
    - Agent capacity constraints: $\\sum_i x_{ij} \\leq C_j$
    - Capability matching: $x_{ij} = 0$ if agent $j$ lacks capability for task $i$

Design Patterns:
    - Mediator: Orchestrator coordinates all agent interactions
    - Strategy: Different assignment strategies (round-robin, capability-based)
    - Observer: Monitors task completion and agent status

Author: AI-Practices
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
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
    Set,
    Tuple,
)

try:
    from .agent_base import BaseAgent, AgentConfig, AgentState, AgentResponse
    from .agent_communication import MessageBus, AgentMessage, MessageType
except ImportError:
    from agent_base import BaseAgent, AgentConfig, AgentState, AgentResponse
    from agent_communication import MessageBus, AgentMessage, MessageType

__all__ = [
    "TaskStatus",
    "TaskAssignment",
    "OrchestratorConfig",
    "AssignmentStrategy",
    "AgentOrchestrator",
    "RoundRobinOrchestrator",
    "CapabilityBasedOrchestrator",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_CONCURRENT: Final[int] = 5
DEFAULT_TASK_TIMEOUT: Final[float] = 60.0


# =============================================================================
# Enumerations
# =============================================================================


class TaskStatus(str, Enum):
    """Status of a task in the orchestration system."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AssignmentStrategy(str, Enum):
    """Strategy for assigning tasks to agents."""
    ROUND_ROBIN = "round_robin"
    CAPABILITY_BASED = "capability_based"
    LOAD_BALANCED = "load_balanced"
    RANDOM = "random"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TaskAssignment:
    """
    Represents a task assigned to an agent.
    
    Attributes:
        task_id: Unique task identifier
        description: Task description/content
        agent_id: Assigned agent ID (None if unassigned)
        status: Current task status
        priority: Task priority (higher = more urgent)
        required_capabilities: Capabilities needed to execute
        result: Task result after completion
        created_at: Creation timestamp
        assigned_at: Assignment timestamp
        completed_at: Completion timestamp
        metadata: Additional task data
    """
    task_id: str
    description: str
    agent_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    required_capabilities: Set[str] = field(default_factory=set)
    result: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def assign(self, agent_id: str) -> None:
        """Assign task to an agent."""
        self.agent_id = agent_id
        self.status = TaskStatus.ASSIGNED
        self.assigned_at = datetime.utcnow()

    def start(self) -> None:
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS

    def complete(self, result: Any) -> None:
        """Mark task as completed with result."""
        self.result = result
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.result = {"error": error}
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "priority": self.priority,
            "required_capabilities": list(self.required_capabilities),
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT
    task_timeout: float = DEFAULT_TASK_TIMEOUT
    strategy: AssignmentStrategy = AssignmentStrategy.CAPABILITY_BASED
    retry_failed: bool = True
    max_retries: int = 3


# =============================================================================
# Base Orchestrator
# =============================================================================


class AgentOrchestrator(ABC):
    """
    Abstract base class for agent orchestration.
    
    Core Idea:
        Manages a pool of agents, assigns tasks based on strategy,
        and monitors execution progress.
    
    Architecture:
        ```
        ┌─────────────────────────────────────────┐
        │            AgentOrchestrator            │
        ├─────────────────────────────────────────┤
        │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
        │  │ Agent A │  │ Agent B │  │ Agent C │  │
        │  └────┬────┘  └────┬────┘  └────┬────┘  │
        │       │            │            │       │
        │       └────────────┼────────────┘       │
        │                    ▼                    │
        │            [Task Queue]                 │
        │                    │                    │
        │            [Assignment]                 │
        │                    │                    │
        │            [Execution]                  │
        └─────────────────────────────────────────┘
        ```
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._agents: Dict[str, BaseAgent] = {}
        self._tasks: Dict[str, TaskAssignment] = {}
        self._task_queue: asyncio.Queue[str] = asyncio.Queue()
        self._running = False
        self._stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
        }
        self.message_bus = MessageBus()

    # -------------------------------------------------------------------------
    # Agent Management
    # -------------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.id] = agent
        
        # Register with message bus
        async def handler(msg: AgentMessage) -> Optional[AgentMessage]:
            response = await agent.step(str(msg.content), msg.sender_name)
            return msg.create_reply(
                content=response.content,
                sender_id=agent.id,
                sender_name=agent.name,
            )
        
        self.message_bus.register_agent(
            agent.id, agent.name, handler,
            topics=list(agent.capabilities)
        )
        logger.info(f"Agent {agent.name} registered with orchestrator")

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the orchestrator."""
        self._agents.pop(agent_id, None)
        self.message_bus.unregister_agent(agent_id)

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def get_available_agents(self) -> List[BaseAgent]:
        """Get all agents that are available for tasks."""
        return [
            agent for agent in self._agents.values()
            if agent.state == AgentState.IDLE
        ]

    def get_agents_with_capability(self, capability: str) -> List[BaseAgent]:
        """Get agents that have a specific capability."""
        return [
            agent for agent in self._agents.values()
            if agent.has_capability(capability)
        ]

    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------

    def create_task(
        self,
        description: str,
        task_id: Optional[str] = None,
        priority: int = 0,
        required_capabilities: Optional[Set[str]] = None,
        **metadata: Any,
    ) -> TaskAssignment:
        """Create a new task."""
        import uuid
        task_id = task_id or f"task_{uuid.uuid4().hex[:12]}"
        
        task = TaskAssignment(
            task_id=task_id,
            description=description,
            priority=priority,
            required_capabilities=required_capabilities or set(),
            metadata=metadata,
        )
        
        self._tasks[task_id] = task
        self._stats["tasks_created"] += 1
        logger.info(f"Task created: {task_id}")
        return task

    def get_task(self, task_id: str) -> Optional[TaskAssignment]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_pending_tasks(self) -> List[TaskAssignment]:
        """Get all pending tasks sorted by priority."""
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        return sorted(pending, key=lambda t: -t.priority)

    @abstractmethod
    def select_agent(self, task: TaskAssignment) -> Optional[BaseAgent]:
        """Select an agent for a task. Implemented by subclasses."""
        pass

    async def assign_task(self, task_id: str) -> bool:
        """Assign a task to an appropriate agent."""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        agent = self.select_agent(task)
        if not agent:
            logger.warning(f"No suitable agent for task {task_id}")
            return False
        
        task.assign(agent.id)
        logger.info(f"Task {task_id} assigned to {agent.name}")
        return True

    async def execute_task(self, task_id: str) -> Optional[AgentResponse]:
        """Execute an assigned task."""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.ASSIGNED:
            return None
        
        agent = self._agents.get(task.agent_id)
        if not agent:
            task.fail("Agent not found")
            return None
        
        task.start()
        
        try:
            response = await asyncio.wait_for(
                agent.step(task.description),
                timeout=self.config.task_timeout
            )
            task.complete(response.content)
            self._stats["tasks_completed"] += 1
            return response
        except asyncio.TimeoutError:
            task.fail("Task timeout")
            self._stats["tasks_failed"] += 1
            return None
        except Exception as e:
            task.fail(str(e))
            self._stats["tasks_failed"] += 1
            return None

    async def submit_and_execute(
        self,
        description: str,
        **kwargs: Any,
    ) -> Optional[AgentResponse]:
        """Create, assign, and execute a task in one call."""
        task = self.create_task(description, **kwargs)
        if await self.assign_task(task.task_id):
            return await self.execute_task(task.task_id)
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self._stats,
            "agents_count": len(self._agents),
            "pending_tasks": len(self.get_pending_tasks()),
        }


# =============================================================================
# Concrete Orchestrator Implementations
# =============================================================================


class RoundRobinOrchestrator(AgentOrchestrator):
    """
    Assigns tasks to agents in round-robin fashion.
    
    Simple and fair distribution, ignores capabilities.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        super().__init__(config)
        self._current_index = 0

    def select_agent(self, task: TaskAssignment) -> Optional[BaseAgent]:
        """Select next available agent in round-robin order."""
        available = self.get_available_agents()
        if not available:
            return None
        
        agent = available[self._current_index % len(available)]
        self._current_index += 1
        return agent


class CapabilityBasedOrchestrator(AgentOrchestrator):
    """
    Assigns tasks based on agent capabilities.
    
    Matches task requirements with agent capabilities for optimal assignment.
    """

    def select_agent(self, task: TaskAssignment) -> Optional[BaseAgent]:
        """Select agent with matching capabilities."""
        available = self.get_available_agents()
        
        if not task.required_capabilities:
            return available[0] if available else None
        
        # Find agents with all required capabilities
        matching = [
            agent for agent in available
            if task.required_capabilities.issubset(agent.capabilities)
        ]
        
        if matching:
            # Return agent with most capabilities (most specialized)
            return max(matching, key=lambda a: len(a.capabilities))
        
        # Fallback: find agent with most matching capabilities
        if available:
            return max(
                available,
                key=lambda a: len(task.required_capabilities & a.capabilities)
            )
        return None


class LoadBalancedOrchestrator(AgentOrchestrator):
    """
    Assigns tasks based on agent workload.
    
    Tracks active tasks per agent and assigns to least loaded.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        super().__init__(config)
        self._agent_load: Dict[str, int] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        super().register_agent(agent)
        self._agent_load[agent.id] = 0

    def select_agent(self, task: TaskAssignment) -> Optional[BaseAgent]:
        """Select agent with lowest current load."""
        available = self.get_available_agents()
        if not available:
            return None
        
        # Filter by capabilities if required
        if task.required_capabilities:
            available = [
                a for a in available
                if task.required_capabilities.issubset(a.capabilities)
            ]
        
        if not available:
            return None
        
        # Select least loaded agent
        return min(available, key=lambda a: self._agent_load.get(a.id, 0))

    async def execute_task(self, task_id: str) -> Optional[AgentResponse]:
        """Execute task with load tracking."""
        task = self._tasks.get(task_id)
        if task and task.agent_id:
            self._agent_load[task.agent_id] = self._agent_load.get(task.agent_id, 0) + 1
        
        try:
            return await super().execute_task(task_id)
        finally:
            if task and task.agent_id:
                self._agent_load[task.agent_id] = max(0, self._agent_load.get(task.agent_id, 1) - 1)
