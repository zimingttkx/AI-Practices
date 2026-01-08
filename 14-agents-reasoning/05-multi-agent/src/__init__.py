"""
Multi-Agent System: Collaborative AI Agent Framework.

This module provides a comprehensive framework for building multi-agent systems,
including agent base classes, communication protocols, orchestration, and
collaboration patterns.

Modules:
    - agent_base: Core agent abstractions and lifecycle management
    - agent_communication: Message passing and communication protocols
    - agent_orchestrator: Agent coordination and task distribution
    - debate_agents: Adversarial debate-based reasoning
    - collaborative_agents: Cooperative multi-agent patterns
"""

from .agent_base import (
    AgentRole,
    AgentState,
    AgentConfig,
    BaseAgent,
    SimpleAgent,
    create_agent,
)

from .agent_communication import (
    MessageType,
    AgentMessage,
    MessageBus,
    DirectChannel,
    BroadcastChannel,
    CommunicationProtocol,
)

from .agent_orchestrator import (
    OrchestratorConfig,
    TaskAssignment,
    AgentOrchestrator,
    RoundRobinOrchestrator,
    CapabilityBasedOrchestrator,
)

from .debate_agents import (
    DebateRole,
    DebateConfig,
    DebaterAgent,
    JudgeAgent,
    DebateArena,
)

from .collaborative_agents import (
    CollaborationMode,
    TeamConfig,
    TeamMember,
    CollaborativeTeam,
    ConsensusBuilder,
)

__all__ = [
    # agent_base
    "AgentRole",
    "AgentState", 
    "AgentConfig",
    "BaseAgent",
    "SimpleAgent",
    "create_agent",
    # agent_communication
    "MessageType",
    "AgentMessage",
    "MessageBus",
    "DirectChannel",
    "BroadcastChannel",
    "CommunicationProtocol",
    # agent_orchestrator
    "OrchestratorConfig",
    "TaskAssignment",
    "AgentOrchestrator",
    "RoundRobinOrchestrator",
    "CapabilityBasedOrchestrator",
    # debate_agents
    "DebateRole",
    "DebateConfig",
    "DebaterAgent",
    "JudgeAgent",
    "DebateArena",
    # collaborative_agents
    "CollaborationMode",
    "TeamConfig",
    "TeamMember",
    "CollaborativeTeam",
    "ConsensusBuilder",
]

__version__ = "1.0.0"
