"""
Autonomous Agent Module: AutoGPT-style Self-Directed Agent System.

This module provides a complete autonomous agent framework including:
- Goal Management: Hierarchical goal decomposition and tracking
- Action Execution: Tool calls, code execution, file operations
- Self-Reflection: Learning from successes and failures
- Agent Loop: OODA-based execution cycle

Example:
    >>> from autonomous_agent import AutonomousAgent, AgentConfig
    >>> 
    >>> agent = AutonomousAgent(AgentConfig(name="ResearchBot"))
    >>> agent.set_objective("Research quantum computing")
    >>> result = agent.run()
    >>> print(agent.get_status())

Author: AI-Practices
Version: 1.0.0
"""

from .goal_manager import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalDecomposer,
    LLMGoalDecomposer,
    RuleBasedDecomposer,
    GoalPriorityQueue,
    CompletionChecker,
    GoalManager,
)

from .action_executor import (
    ActionType,
    ActionStatus,
    ActionResult,
    Action,
    ActionHandler,
    ActionExecutor,
    ToolAction,
    CodeAction,
    FileAction,
    WebAction,
    ThinkAction,
    ActionRegistry,
    create_action,
)

from .self_reflection import (
    ReflectionType,
    Reflection,
    ReflectionEngine,
    SuccessAnalyzer,
    FailureAnalyzer,
    StrategyAdjuster,
    LearningMemory,
    SelfReflector,
)

from .agent_loop import (
    LoopState,
    LoopConfig,
    LoopContext,
    LoopEvent,
    TerminationReason,
    TerminationChecker,
    AgentLoop,
    SimpleAgentLoop,
    IntegratedAgentLoop,
    create_agent_loop,
)

from .autonomous_agent import (
    AgentConfig,
    AgentState,
    AutonomousAgent,
    AgentBuilder,
    create_autonomous_agent,
)

__all__ = [
    # Goal Manager
    "Goal",
    "GoalStatus",
    "GoalPriority",
    "GoalDecomposer",
    "LLMGoalDecomposer",
    "RuleBasedDecomposer",
    "GoalPriorityQueue",
    "CompletionChecker",
    "GoalManager",
    # Action Executor
    "ActionType",
    "ActionStatus",
    "ActionResult",
    "Action",
    "ActionHandler",
    "ActionExecutor",
    "ToolAction",
    "CodeAction",
    "FileAction",
    "WebAction",
    "ThinkAction",
    "ActionRegistry",
    "create_action",
    # Self Reflection
    "ReflectionType",
    "Reflection",
    "ReflectionEngine",
    "SuccessAnalyzer",
    "FailureAnalyzer",
    "StrategyAdjuster",
    "LearningMemory",
    "SelfReflector",
    # Agent Loop
    "LoopState",
    "LoopConfig",
    "LoopContext",
    "LoopEvent",
    "TerminationReason",
    "TerminationChecker",
    "AgentLoop",
    "SimpleAgentLoop",
    "IntegratedAgentLoop",
    "create_agent_loop",
    # Autonomous Agent
    "AgentConfig",
    "AgentState",
    "AutonomousAgent",
    "AgentBuilder",
    "create_autonomous_agent",
]

__version__ = "1.0.0"
