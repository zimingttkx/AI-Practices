"""
Agent Loop: Core Execution Loop for Autonomous Agents.

Core Idea:
    This module implements the main execution loop for autonomous agents,
    following the OODA (Observe-Orient-Decide-Act) cycle pattern.

Mathematical Foundation:
    Agent loop as Markov Decision Process:
        π(a|s) = argmax_a Q(s, a)
    
    Termination condition:
        terminate = goal_achieved ∨ max_iterations ∨ stuck_detection
    
    Stuck detection:
        stuck = (consecutive_failures > threshold) ∨ (no_progress_iterations > limit)

Design Patterns:
    - State Machine: Explicit loop states
    - Template Method: Customizable loop phases
    - Observer: Loop event notifications

References:
    - OODA Loop: Boyd's Decision Cycle
    - AutoGPT Main Loop
    - BabyAGI Task Execution

Author: AI-Practices
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

__all__ = [
    "LoopState",
    "LoopConfig",
    "LoopContext",
    "LoopEvent",
    "TerminationReason",
    "TerminationChecker",
    "AgentLoop",
    "SimpleAgentLoop",
]

logger = logging.getLogger(__name__)


class LoopState(str, Enum):
    """States of the agent loop."""
    IDLE = "idle"
    OBSERVING = "observing"
    ORIENTING = "orienting"
    DECIDING = "deciding"
    ACTING = "acting"
    REFLECTING = "reflecting"
    PAUSED = "paused"
    TERMINATED = "terminated"


class TerminationReason(str, Enum):
    """Reasons for loop termination."""
    GOAL_ACHIEVED = "goal_achieved"
    MAX_ITERATIONS = "max_iterations"
    MAX_TIME = "max_time"
    USER_CANCELLED = "user_cancelled"
    STUCK = "stuck"
    ERROR = "error"
    NO_MORE_GOALS = "no_more_goals"


@dataclass
class LoopConfig:
    """Configuration for the agent loop."""
    max_iterations: int = 100
    max_time_seconds: float = 3600.0
    stuck_threshold: int = 5
    reflection_interval: int = 5
    pause_between_iterations: float = 0.1
    enable_reflection: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be positive")


@dataclass
class LoopContext:
    """Context maintained across loop iterations."""
    iteration: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    state: LoopState = LoopState.IDLE
    termination_reason: Optional[TerminationReason] = None
    consecutive_failures: int = 0
    total_successes: int = 0
    total_failures: int = 0
    current_goal: Optional[str] = None
    last_action: Optional[str] = None
    last_result: Optional[str] = None
    observations: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_success(self) -> None:
        self.total_successes += 1
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.total_failures += 1
        self.consecutive_failures += 1

    def elapsed_time(self) -> float:
        if not self.start_time:
            return 0.0
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "state": self.state.value,
            "elapsed_time": self.elapsed_time(),
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "current_goal": self.current_goal,
            "termination_reason": self.termination_reason.value if self.termination_reason else None,
        }


@dataclass
class LoopEvent:
    """Event emitted during loop execution."""
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@runtime_checkable
class LoopObserver(Protocol):
    """Protocol for loop event observers."""
    def on_event(self, event: LoopEvent) -> None: ...


class TerminationChecker:
    """Checks various termination conditions."""

    def __init__(self, config: LoopConfig) -> None:
        self.config = config

    def should_terminate(self, context: LoopContext) -> Optional[TerminationReason]:
        if context.iteration >= self.config.max_iterations:
            return TerminationReason.MAX_ITERATIONS
        
        if context.elapsed_time() >= self.config.max_time_seconds:
            return TerminationReason.MAX_TIME
        
        if context.consecutive_failures >= self.config.stuck_threshold:
            return TerminationReason.STUCK
        
        return None


class AgentLoop(ABC):
    """Abstract base class for agent execution loops."""

    def __init__(self, config: Optional[LoopConfig] = None) -> None:
        self.config = config or LoopConfig()
        self.context = LoopContext()
        self.termination_checker = TerminationChecker(self.config)
        self._observers: List[LoopObserver] = []
        self._running = False
        self._paused = False

    def add_observer(self, observer: LoopObserver) -> None:
        self._observers.append(observer)

    def _emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        event = LoopEvent(event_type=event_type, data=data or {})
        for observer in self._observers:
            try:
                observer.on_event(event)
            except Exception as e:
                logger.warning(f"Observer error: {e}")

    @abstractmethod
    def observe(self) -> str:
        """Observe the current state/environment."""
        pass

    @abstractmethod
    def orient(self, observation: str) -> str:
        """Orient/analyze the observation."""
        pass

    @abstractmethod
    def decide(self, analysis: str) -> str:
        """Decide on the next action."""
        pass

    @abstractmethod
    def act(self, decision: str) -> tuple[bool, str]:
        """Execute the decided action. Returns (success, result)."""
        pass

    def reflect(self) -> None:
        """Optional reflection phase."""
        pass

    def run(self) -> LoopContext:
        """Run the agent loop synchronously."""
        self._running = True
        self.context = LoopContext()
        self.context.start_time = datetime.utcnow()
        self.context.state = LoopState.IDLE
        
        self._emit_event("loop_started", {"config": self.config.__dict__})
        
        try:
            while self._running:
                if self._paused:
                    self.context.state = LoopState.PAUSED
                    time.sleep(0.1)
                    continue
                
                reason = self.termination_checker.should_terminate(self.context)
                if reason:
                    self.context.termination_reason = reason
                    break
                
                self._run_iteration()
                self.context.iteration += 1
                
                if self.config.pause_between_iterations > 0:
                    time.sleep(self.config.pause_between_iterations)
                    
        except Exception as e:
            logger.error(f"Loop error: {e}")
            self.context.termination_reason = TerminationReason.ERROR
            self.context.metadata["error"] = str(e)
        finally:
            self._running = False
            self.context.end_time = datetime.utcnow()
            self.context.state = LoopState.TERMINATED
            self._emit_event("loop_ended", self.context.to_dict())
        
        return self.context

    def _run_iteration(self) -> None:
        """Execute one iteration of the OODA loop."""
        self._emit_event("iteration_started", {"iteration": self.context.iteration})
        
        self.context.state = LoopState.OBSERVING
        observation = self.observe()
        self.context.observations.append(observation)
        self._emit_event("observed", {"observation": observation[:200]})
        
        self.context.state = LoopState.ORIENTING
        analysis = self.orient(observation)
        self._emit_event("oriented", {"analysis": analysis[:200]})
        
        self.context.state = LoopState.DECIDING
        decision = self.decide(analysis)
        self.context.decisions.append(decision)
        self.context.last_action = decision
        self._emit_event("decided", {"decision": decision[:200]})
        
        self.context.state = LoopState.ACTING
        success, result = self.act(decision)
        self.context.last_result = result
        
        if success:
            self.context.record_success()
            self._emit_event("action_succeeded", {"result": result[:200]})
        else:
            self.context.record_failure()
            self._emit_event("action_failed", {"error": result[:200]})
        
        if self.config.enable_reflection:
            if self.context.iteration % self.config.reflection_interval == 0:
                self.context.state = LoopState.REFLECTING
                self.reflect()
                self._emit_event("reflected", {})
        
        self._emit_event("iteration_ended", {
            "iteration": self.context.iteration,
            "success": success,
        })

    async def run_async(self) -> LoopContext:
        """Run the agent loop asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run)

    def pause(self) -> None:
        self._paused = True
        self._emit_event("loop_paused", {})

    def resume(self) -> None:
        self._paused = False
        self._emit_event("loop_resumed", {})

    def stop(self) -> None:
        self._running = False
        self.context.termination_reason = TerminationReason.USER_CANCELLED
        self._emit_event("loop_stopped", {})

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "paused": self._paused,
            "context": self.context.to_dict(),
        }


class SimpleAgentLoop(AgentLoop):
    """
    Simple implementation of the agent loop with pluggable components.
    
    Example:
        >>> def my_observe():
        ...     return "Current state: ready"
        >>> loop = SimpleAgentLoop(
        ...     observe_fn=my_observe,
        ...     orient_fn=lambda o: f"Analysis: {o}",
        ...     decide_fn=lambda a: "action_1",
        ...     act_fn=lambda d: (True, "Done"),
        ... )
        >>> result = loop.run()
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        observe_fn: Optional[Callable[[], str]] = None,
        orient_fn: Optional[Callable[[str], str]] = None,
        decide_fn: Optional[Callable[[str], str]] = None,
        act_fn: Optional[Callable[[str], tuple[bool, str]]] = None,
        reflect_fn: Optional[Callable[[], None]] = None,
        goal_check_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        super().__init__(config)
        self._observe_fn = observe_fn or (lambda: "No observation")
        self._orient_fn = orient_fn or (lambda o: f"Analyzed: {o}")
        self._decide_fn = decide_fn or (lambda a: "default_action")
        self._act_fn = act_fn or (lambda d: (True, "Action completed"))
        self._reflect_fn = reflect_fn
        self._goal_check_fn = goal_check_fn

    def observe(self) -> str:
        return self._observe_fn()

    def orient(self, observation: str) -> str:
        return self._orient_fn(observation)

    def decide(self, analysis: str) -> str:
        return self._decide_fn(analysis)

    def act(self, decision: str) -> tuple[bool, str]:
        return self._act_fn(decision)

    def reflect(self) -> None:
        if self._reflect_fn:
            self._reflect_fn()

    def _run_iteration(self) -> None:
        if self._goal_check_fn and self._goal_check_fn():
            self.context.termination_reason = TerminationReason.GOAL_ACHIEVED
            self._running = False
            return
        super()._run_iteration()


class IntegratedAgentLoop(AgentLoop):
    """
    Agent loop integrated with GoalManager, ActionExecutor, and SelfReflector.
    """

    def __init__(
        self,
        goal_manager: Any,
        action_executor: Any,
        reflector: Optional[Any] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        config: Optional[LoopConfig] = None,
    ) -> None:
        super().__init__(config)
        self.goal_manager = goal_manager
        self.action_executor = action_executor
        self.reflector = reflector
        self.llm_func = llm_func or self._mock_llm
        self._current_goal = None
        self._action_history: List[str] = []

    def _mock_llm(self, prompt: str) -> str:
        return "Execute the next logical step"

    def observe(self) -> str:
        progress = self.goal_manager.get_progress()
        current = self._current_goal
        
        observation = f"Progress: {progress['completed']}/{progress['total']} goals completed. "
        if current:
            observation += f"Current goal: {current.description}"
        else:
            observation += "No current goal."
        
        if self.context.last_result:
            observation += f" Last result: {self.context.last_result[:100]}"
        
        return observation

    def orient(self, observation: str) -> str:
        prompt = f"""Analyze the current situation and determine the best approach.

Observation: {observation}
Recent actions: {self._action_history[-5:] if self._action_history else 'None'}

What should be the focus? What approach would be most effective?"""

        return self.llm_func(prompt)

    def decide(self, analysis: str) -> str:
        if not self._current_goal:
            self._current_goal = self.goal_manager.get_next_goal()
            if not self._current_goal:
                return "NO_GOAL"
            self.goal_manager.start_goal(self._current_goal.goal_id)
            self.context.current_goal = self._current_goal.description

        prompt = f"""Based on the analysis, decide on the specific action to take.

Goal: {self._current_goal.description}
Analysis: {analysis}

Respond with a specific action in format: ACTION_TYPE:action_name:parameters
Example: TOOL:search:query=python tutorial"""

        decision = self.llm_func(prompt)
        return decision

    def act(self, decision: str) -> tuple[bool, str]:
        if decision == "NO_GOAL":
            self.context.termination_reason = TerminationReason.NO_MORE_GOALS
            self._running = False
            return True, "No more goals to process"

        self._action_history.append(decision)
        
        try:
            from .action_executor import Action, ActionType
            
            parts = decision.split(":", 2)
            if len(parts) >= 2:
                action_type_str = parts[0].upper()
                action_name = parts[1]
                params = {}
                if len(parts) > 2:
                    for param in parts[2].split(","):
                        if "=" in param:
                            k, v = param.split("=", 1)
                            params[k.strip()] = v.strip()
                
                try:
                    action_type = ActionType(action_type_str.lower())
                except ValueError:
                    action_type = ActionType.THINK
                
                action = Action(
                    action_type=action_type,
                    name=action_name,
                    parameters=params,
                )
                result = self.action_executor.execute(action)
                
                if result.is_success:
                    completed, reason = self.goal_manager.complete_goal(
                        self._current_goal.goal_id,
                        str(result.output),
                    )
                    if completed:
                        self._current_goal = None
                    return True, str(result.output)
                else:
                    return False, result.error or "Action failed"
            else:
                return True, f"Processed: {decision}"
                
        except Exception as e:
            return False, str(e)

    def reflect(self) -> None:
        if not self.reflector:
            return
        
        if self._current_goal and self.context.last_result:
            self.reflector.reflect_on_action(
                goal=self._current_goal.description,
                action=self.context.last_action or "unknown",
                result=self.context.last_result,
                success=self.context.consecutive_failures == 0,
                goal_id=self._current_goal.goal_id,
            )


def create_agent_loop(
    loop_type: str = "simple",
    config: Optional[LoopConfig] = None,
    **kwargs: Any,
) -> AgentLoop:
    """Factory function to create agent loops."""
    if loop_type == "simple":
        return SimpleAgentLoop(config=config, **kwargs)
    elif loop_type == "integrated":
        return IntegratedAgentLoop(config=config, **kwargs)
    else:
        raise ValueError(f"Unknown loop type: {loop_type}")
