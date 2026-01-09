"""
Autonomous Agent: AutoGPT-style Self-Directed Agent System.

Core Idea:
    This module provides the main AutonomousAgent class that integrates
    goal management, action execution, self-reflection, and the agent loop
    into a cohesive autonomous system.

Mathematical Foundation:
    Agent utility function:
        U(s) = Σ γ^t * R(s_t, a_t) where γ is discount factor
    
    Goal-directed behavior:
        π*(s) = argmax_a Σ P(s'|s,a) * [R(s,a,s') + γV*(s')]

Design Patterns:
    - Facade Pattern: Unified interface to subsystems
    - Mediator Pattern: Coordinates component interactions
    - Builder Pattern: Flexible agent configuration

References:
    - AutoGPT: https://github.com/Significant-Gravitas/AutoGPT
    - BabyAGI: https://github.com/yoheinakajima/babyagi
    - AgentGPT: https://github.com/reworkd/AgentGPT

Author: AI-Practices
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

try:
    from .goal_manager import (
        Goal,
        GoalManager,
        GoalPriority,
        GoalStatus,
        LLMGoalDecomposer,
    )
    from .action_executor import (
        Action,
        ActionExecutor,
        ActionResult,
        ActionType,
        ToolAction,
    )
    from .self_reflection import SelfReflector, Reflection
    from .agent_loop import (
        AgentLoop,
        IntegratedAgentLoop,
        LoopConfig,
        LoopContext,
        TerminationReason,
    )
except ImportError:
    from goal_manager import (
        Goal,
        GoalManager,
        GoalPriority,
        GoalStatus,
        LLMGoalDecomposer,
    )
    from action_executor import (
        Action,
        ActionExecutor,
        ActionResult,
        ActionType,
        ToolAction,
    )
    from self_reflection import SelfReflector, Reflection
    from agent_loop import (
        AgentLoop,
        IntegratedAgentLoop,
        LoopConfig,
        LoopContext,
        TerminationReason,
    )

__all__ = [
    "AgentConfig",
    "AutonomousAgent",
    "create_autonomous_agent",
]

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the autonomous agent."""
    name: str = "AutonomousAgent"
    description: str = "An autonomous AI agent"
    max_iterations: int = 100
    max_time_seconds: float = 3600.0
    enable_reflection: bool = True
    reflection_interval: int = 5
    auto_decompose_goals: bool = True
    verbose: bool = False
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    
    def to_loop_config(self) -> LoopConfig:
        return LoopConfig(
            max_iterations=self.max_iterations,
            max_time_seconds=self.max_time_seconds,
            enable_reflection=self.enable_reflection,
            reflection_interval=self.reflection_interval,
            verbose=self.verbose,
        )


@dataclass
class AgentState:
    """Current state of the autonomous agent."""
    is_running: bool = False
    current_goal: Optional[str] = None
    iteration: int = 0
    start_time: Optional[datetime] = None
    last_action: Optional[str] = None
    last_result: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class AutonomousAgent:
    """
    AutoGPT-style autonomous agent.
    
    Integrates:
    - GoalManager: Hierarchical goal management
    - ActionExecutor: Tool and action execution
    - SelfReflector: Learning from experience
    - AgentLoop: OODA execution cycle
    
    Example:
        >>> agent = AutonomousAgent(name="ResearchBot")
        >>> agent.set_objective("Research quantum computing and write a summary")
        >>> result = agent.run()
        >>> print(result.context.to_dict())
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm_func = llm_func or self._mock_llm
        
        # Initialize components
        self.goal_manager = GoalManager(
            decomposer=LLMGoalDecomposer(llm_func=self.llm_func),
        )
        self.action_executor = ActionExecutor()
        self.reflector = SelfReflector(llm_func=self.llm_func)
        
        # State
        self.state = AgentState()
        self._loop: Optional[AgentLoop] = None
        self._objective: Optional[str] = None
        self._constraints: List[str] = []
        self._tools_registered: List[str] = []
        
        logger.info(f"AutonomousAgent '{self.config.name}' initialized")

    def _mock_llm(self, prompt: str) -> str:
        """Mock LLM for testing."""
        if "decompose" in prompt.lower() or "break down" in prompt.lower():
            return "1. Research the topic\n2. Analyze findings\n3. Write summary\n4. Review output"
        elif "decide" in prompt.lower() or "action" in prompt.lower():
            return "THINK:analyze:thought=Processing the current goal"
        else:
            return "Proceeding with the task systematically."

    # -------------------------------------------------------------------------
    # Configuration Methods
    # -------------------------------------------------------------------------

    def set_objective(self, objective: str, priority: GoalPriority = GoalPriority.HIGH) -> Goal:
        """Set the main objective for the agent."""
        self._objective = objective
        goal = self.goal_manager.add_goal(
            description=objective,
            priority=priority,
            auto_decompose=self.config.auto_decompose_goals,
        )
        logger.info(f"Objective set: {objective}")
        return goal

    def add_goal(
        self,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        success_criteria: Optional[str] = None,
    ) -> Goal:
        """Add an additional goal."""
        return self.goal_manager.add_goal(
            description=description,
            priority=priority,
            success_criteria=success_criteria,
            auto_decompose=self.config.auto_decompose_goals,
        )

    def add_constraint(self, constraint: str) -> None:
        """Add a constraint the agent must follow."""
        self._constraints.append(constraint)
        logger.info(f"Constraint added: {constraint}")

    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a custom tool for the agent to use."""
        tool_handler = None
        for handler in self.action_executor.registry._handlers:
            if isinstance(handler, ToolAction):
                tool_handler = handler
                break
        
        if tool_handler:
            tool_handler.register_tool(name, func, description)
            self._tools_registered.append(name)
            logger.info(f"Tool registered: {name}")

    # -------------------------------------------------------------------------
    # Execution Methods
    # -------------------------------------------------------------------------

    def run(self) -> LoopContext:
        """Run the agent until completion or termination."""
        if not self._objective and self.goal_manager.get_progress()["total"] == 0:
            raise ValueError("No objective or goals set. Call set_objective() first.")
        
        self.state.is_running = True
        self.state.start_time = datetime.utcnow()
        
        self._loop = IntegratedAgentLoop(
            goal_manager=self.goal_manager,
            action_executor=self.action_executor,
            reflector=self.reflector if self.config.enable_reflection else None,
            llm_func=self.llm_func,
            config=self.config.to_loop_config(),
        )
        
        logger.info(f"Agent '{self.config.name}' starting execution")
        
        try:
            context = self._loop.run()
        finally:
            self.state.is_running = False
        
        logger.info(f"Agent completed: {context.termination_reason}")
        return context

    async def run_async(self) -> LoopContext:
        """Run the agent asynchronously."""
        if not self._objective and self.goal_manager.get_progress()["total"] == 0:
            raise ValueError("No objective or goals set.")
        
        self.state.is_running = True
        self.state.start_time = datetime.utcnow()
        
        self._loop = IntegratedAgentLoop(
            goal_manager=self.goal_manager,
            action_executor=self.action_executor,
            reflector=self.reflector if self.config.enable_reflection else None,
            llm_func=self.llm_func,
            config=self.config.to_loop_config(),
        )
        
        try:
            context = await self._loop.run_async()
        finally:
            self.state.is_running = False
        
        return context

    def step(self) -> Optional[Dict[str, Any]]:
        """Execute a single step (for interactive use)."""
        if not self._loop:
            self._loop = IntegratedAgentLoop(
                goal_manager=self.goal_manager,
                action_executor=self.action_executor,
                reflector=self.reflector,
                llm_func=self.llm_func,
                config=self.config.to_loop_config(),
            )
            self._loop.context.start_time = datetime.utcnow()
        
        if self._loop.context.termination_reason:
            return None
        
        self._loop._run_iteration()
        self._loop.context.iteration += 1
        self.state.iteration = self._loop.context.iteration
        
        return {
            "iteration": self._loop.context.iteration,
            "state": self._loop.context.state.value,
            "last_action": self._loop.context.last_action,
            "last_result": self._loop.context.last_result,
        }

    def pause(self) -> None:
        """Pause the agent execution."""
        if self._loop:
            self._loop.pause()
            logger.info("Agent paused")

    def resume(self) -> None:
        """Resume the agent execution."""
        if self._loop:
            self._loop.resume()
            logger.info("Agent resumed")

    def stop(self) -> None:
        """Stop the agent execution."""
        if self._loop:
            self._loop.stop()
            self.state.is_running = False
            logger.info("Agent stopped")

    # -------------------------------------------------------------------------
    # Status and Reporting Methods
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        progress = self.goal_manager.get_progress()
        executor_stats = self.action_executor.get_stats()
        reflection_stats = self.reflector.get_insights_summary()
        
        return {
            "name": self.config.name,
            "is_running": self.state.is_running,
            "objective": self._objective,
            "goals": progress,
            "actions": executor_stats,
            "reflections": reflection_stats,
            "constraints": self._constraints,
            "tools": self._tools_registered,
            "iteration": self.state.iteration,
        }

    def get_progress(self) -> Dict[str, Any]:
        """Get goal progress."""
        return self.goal_manager.get_progress()

    def get_reflections(self, limit: int = 10) -> List[Reflection]:
        """Get recent reflections."""
        return self.reflector.get_reflections(limit)

    def get_action_history(self) -> List[ActionResult]:
        """Get action execution history."""
        return self.action_executor.get_history()

    def get_goals(self) -> List[Goal]:
        """Get all goals."""
        return self.goal_manager.get_all_goals()

    # -------------------------------------------------------------------------
    # Reset and Cleanup
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.goal_manager.reset()
        self.action_executor.clear_history()
        self.reflector.reset()
        self.state = AgentState()
        self._loop = None
        self._objective = None
        self._constraints.clear()
        logger.info(f"Agent '{self.config.name}' reset")

    def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        self.stop()
        self.action_executor.shutdown()
        logger.info(f"Agent '{self.config.name}' shutdown")

    def __repr__(self) -> str:
        status = "running" if self.state.is_running else "idle"
        progress = self.goal_manager.get_progress()
        return f"AutonomousAgent(name='{self.config.name}', status={status}, goals={progress['completed']}/{progress['total']})"


def create_autonomous_agent(
    name: str = "Agent",
    objective: Optional[str] = None,
    llm_func: Optional[Callable[[str], str]] = None,
    tools: Optional[Dict[str, Callable]] = None,
    constraints: Optional[List[str]] = None,
    **config_kwargs: Any,
) -> AutonomousAgent:
    """
    Factory function to create and configure an autonomous agent.
    
    Args:
        name: Agent name
        objective: Main objective (optional, can be set later)
        llm_func: LLM function for generation
        tools: Dictionary of tool_name -> function
        constraints: List of constraints
        **config_kwargs: Additional AgentConfig parameters
    
    Returns:
        Configured AutonomousAgent instance
    
    Example:
        >>> agent = create_autonomous_agent(
        ...     name="ResearchBot",
        ...     objective="Research AI trends",
        ...     tools={"search": search_fn, "summarize": summarize_fn},
        ...     constraints=["Be concise", "Cite sources"],
        ... )
        >>> result = agent.run()
    """
    config = AgentConfig(name=name, **config_kwargs)
    agent = AutonomousAgent(config=config, llm_func=llm_func)
    
    if tools:
        for tool_name, tool_func in tools.items():
            agent.register_tool(tool_name, tool_func)
    
    if constraints:
        for constraint in constraints:
            agent.add_constraint(constraint)
    
    if objective:
        agent.set_objective(objective)
    
    return agent


class AgentBuilder:
    """Builder pattern for creating autonomous agents."""

    def __init__(self, name: str = "Agent") -> None:
        self._name = name
        self._config_kwargs: Dict[str, Any] = {}
        self._llm_func: Optional[Callable[[str], str]] = None
        self._tools: Dict[str, Callable] = {}
        self._constraints: List[str] = []
        self._objective: Optional[str] = None
        self._goals: List[Dict[str, Any]] = []

    def with_config(self, **kwargs: Any) -> "AgentBuilder":
        self._config_kwargs.update(kwargs)
        return self

    def with_llm(self, llm_func: Callable[[str], str]) -> "AgentBuilder":
        self._llm_func = llm_func
        return self

    def with_tool(self, name: str, func: Callable, description: str = "") -> "AgentBuilder":
        self._tools[name] = func
        return self

    def with_constraint(self, constraint: str) -> "AgentBuilder":
        self._constraints.append(constraint)
        return self

    def with_objective(self, objective: str) -> "AgentBuilder":
        self._objective = objective
        return self

    def with_goal(self, description: str, priority: GoalPriority = GoalPriority.MEDIUM) -> "AgentBuilder":
        self._goals.append({"description": description, "priority": priority})
        return self

    def build(self) -> AutonomousAgent:
        agent = create_autonomous_agent(
            name=self._name,
            llm_func=self._llm_func,
            tools=self._tools if self._tools else None,
            constraints=self._constraints if self._constraints else None,
            **self._config_kwargs,
        )
        
        if self._objective:
            agent.set_objective(self._objective)
        
        for goal in self._goals:
            agent.add_goal(**goal)
        
        return agent
