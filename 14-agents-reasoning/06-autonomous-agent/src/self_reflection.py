"""
Self Reflection: Introspection and Learning for Autonomous Agents.

Core Idea:
    This module provides self-reflection capabilities for autonomous agents,
    enabling them to analyze their performance, learn from mistakes, and
    adjust strategies for improved future performance.

Mathematical Foundation:
    Performance scoring:
        score = α·success_rate + β·efficiency + γ·goal_alignment
    
    Learning rate adjustment:
        lr_new = lr_old * (1 + δ·(target_performance - actual_performance))
    
    Strategy selection (UCB1):
        UCB(s) = avg_reward(s) + c * sqrt(ln(N) / n(s))

Design Patterns:
    - Observer Pattern: Monitor agent actions and outcomes
    - Strategy Pattern: Different reflection strategies
    - Memento Pattern: Store and restore agent states

References:
    - Reflexion: Language Agents with Verbal Reinforcement Learning
    - Self-Refine: Iterative Refinement with Self-Feedback
    - AutoGPT Self-Improvement Mechanisms

Author: AI-Practices
Version: 1.0.0
"""

from __future__ import annotations

import logging
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
    Tuple,
)

__all__ = [
    "ReflectionType",
    "Reflection",
    "ReflectionEngine",
    "SuccessAnalyzer",
    "FailureAnalyzer",
    "StrategyAdjuster",
    "LearningMemory",
    "SelfReflector",
]

logger = logging.getLogger(__name__)


class ReflectionType(str, Enum):
    """Types of reflection."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    STRATEGY = "strategy"


@dataclass
class Reflection:
    """A single reflection entry."""
    reflection_type: ReflectionType
    content: str
    action_id: Optional[str] = None
    goal_id: Optional[str] = None
    insights: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflection_type": self.reflection_type.value,
            "content": self.content,
            "action_id": self.action_id,
            "goal_id": self.goal_id,
            "insights": self.insights,
            "lessons_learned": self.lessons_learned,
            "suggested_improvements": self.suggested_improvements,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class ReflectionEngine(ABC):
    """Abstract base class for reflection engines."""

    @abstractmethod
    def reflect(
        self,
        context: Dict[str, Any],
        outcome: str,
        success: bool,
    ) -> Reflection:
        """Generate a reflection based on context and outcome."""
        pass


class SuccessAnalyzer(ReflectionEngine):
    """Analyzes successful outcomes to extract patterns."""

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.llm_func = llm_func or self._mock_llm

    def _mock_llm(self, prompt: str) -> str:
        return "The approach was effective. Key factors: clear goal definition, appropriate tool selection."

    def reflect(
        self,
        context: Dict[str, Any],
        outcome: str,
        success: bool = True,
    ) -> Reflection:
        goal = context.get("goal", "Unknown goal")
        actions = context.get("actions", [])
        
        prompt = f"""Analyze this successful outcome and extract key insights.

Goal: {goal}
Actions taken: {actions}
Outcome: {outcome}

What made this approach successful? What patterns can be reused?"""

        analysis = self.llm_func(prompt)
        
        return Reflection(
            reflection_type=ReflectionType.SUCCESS,
            content=analysis,
            goal_id=context.get("goal_id"),
            insights=self._extract_insights(analysis),
            lessons_learned=["Approach was effective for this type of task"],
            confidence=0.8,
            metadata={"actions_count": len(actions)},
        )

    def _extract_insights(self, analysis: str) -> List[str]:
        lines = analysis.split(".")
        return [line.strip() for line in lines if line.strip()][:3]


class FailureAnalyzer(ReflectionEngine):
    """Analyzes failures to identify root causes and improvements."""

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.llm_func = llm_func or self._mock_llm
        self.failure_patterns: Dict[str, int] = {}

    def _mock_llm(self, prompt: str) -> str:
        return "Failure likely due to: insufficient information, wrong tool selection. Suggestion: gather more context first."

    def reflect(
        self,
        context: Dict[str, Any],
        outcome: str,
        success: bool = False,
    ) -> Reflection:
        goal = context.get("goal", "Unknown goal")
        actions = context.get("actions", [])
        error = context.get("error", "Unknown error")
        
        prompt = f"""Analyze this failure and identify root causes.

Goal: {goal}
Actions taken: {actions}
Error: {error}
Outcome: {outcome}

What went wrong? How can this be prevented in the future?"""

        analysis = self.llm_func(prompt)
        
        root_cause = self._identify_root_cause(error, actions)
        self._record_pattern(root_cause)
        
        return Reflection(
            reflection_type=ReflectionType.FAILURE,
            content=analysis,
            goal_id=context.get("goal_id"),
            insights=[f"Root cause: {root_cause}"],
            lessons_learned=self._generate_lessons(root_cause),
            suggested_improvements=self._suggest_improvements(root_cause),
            confidence=0.6,
            metadata={"error": error, "root_cause": root_cause},
        )

    def _identify_root_cause(self, error: str, actions: List[str]) -> str:
        error_lower = error.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "not found" in error_lower:
            return "resource_not_found"
        elif "permission" in error_lower:
            return "permission_denied"
        elif "syntax" in error_lower:
            return "syntax_error"
        elif len(actions) == 0:
            return "no_action_taken"
        else:
            return "unknown"

    def _record_pattern(self, root_cause: str) -> None:
        self.failure_patterns[root_cause] = self.failure_patterns.get(root_cause, 0) + 1

    def _generate_lessons(self, root_cause: str) -> List[str]:
        lessons_map = {
            "timeout": ["Set appropriate timeouts", "Break down long-running tasks"],
            "resource_not_found": ["Verify resource existence before access", "Handle missing resources gracefully"],
            "permission_denied": ["Check permissions before operations", "Request necessary access"],
            "syntax_error": ["Validate syntax before execution", "Use linting tools"],
            "no_action_taken": ["Ensure action selection logic is working", "Add fallback actions"],
        }
        return lessons_map.get(root_cause, ["Investigate the specific error"])

    def _suggest_improvements(self, root_cause: str) -> List[str]:
        improvements_map = {
            "timeout": ["Increase timeout", "Optimize operation", "Add progress tracking"],
            "resource_not_found": ["Add existence check", "Implement search fallback"],
            "permission_denied": ["Add permission check", "Request elevated access"],
            "syntax_error": ["Add validation step", "Use safer parsing"],
            "no_action_taken": ["Review decision logic", "Add default action"],
        }
        return improvements_map.get(root_cause, ["Review and debug the failure"])

    def get_common_failures(self) -> List[Tuple[str, int]]:
        return sorted(self.failure_patterns.items(), key=lambda x: x[1], reverse=True)


class StrategyAdjuster:
    """
    Adjusts agent strategies based on reflection outcomes.
    
    Uses UCB1 (Upper Confidence Bound) for strategy selection:
        UCB(s) = avg_reward(s) + c * sqrt(ln(N) / n(s))
    """

    def __init__(self, exploration_constant: float = 1.41) -> None:
        self.exploration_constant = exploration_constant
        self.strategies: Dict[str, Dict[str, Any]] = {}
        self.total_selections = 0

    def register_strategy(self, name: str, description: str = "") -> None:
        if name not in self.strategies:
            self.strategies[name] = {
                "description": description,
                "selections": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
            }

    def record_outcome(self, strategy_name: str, reward: float) -> None:
        if strategy_name not in self.strategies:
            self.register_strategy(strategy_name)
        
        s = self.strategies[strategy_name]
        s["selections"] += 1
        s["total_reward"] += reward
        s["avg_reward"] = s["total_reward"] / s["selections"]
        self.total_selections += 1

    def select_strategy(self) -> Optional[str]:
        import math
        
        if not self.strategies:
            return None
        
        for name, s in self.strategies.items():
            if s["selections"] == 0:
                return name
        
        best_strategy = None
        best_ucb = float("-inf")
        
        for name, s in self.strategies.items():
            ucb = s["avg_reward"] + self.exploration_constant * math.sqrt(
                math.log(self.total_selections) / s["selections"]
            )
            if ucb > best_ucb:
                best_ucb = ucb
                best_strategy = name
        
        return best_strategy

    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        return {name: dict(s) for name, s in self.strategies.items()}

    def suggest_adjustment(self, current_strategy: str, performance: float) -> Optional[str]:
        if performance < 0.3:
            alternatives = [
                name for name in self.strategies
                if name != current_strategy and self.strategies[name]["avg_reward"] > performance
            ]
            if alternatives:
                return max(alternatives, key=lambda n: self.strategies[n]["avg_reward"])
        return None


@dataclass
class LearningEntry:
    """A single learning entry."""
    situation: str
    action: str
    outcome: str
    reward: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class LearningMemory:
    """
    Stores and retrieves learning experiences for the agent.
    
    Implements experience replay for reinforcement learning style updates.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self.max_entries = max_entries
        self._entries: List[LearningEntry] = []
        self._situation_index: Dict[str, List[int]] = {}

    def add_entry(
        self,
        situation: str,
        action: str,
        outcome: str,
        reward: float,
    ) -> None:
        entry = LearningEntry(
            situation=situation,
            action=action,
            outcome=outcome,
            reward=reward,
        )
        
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
            self._rebuild_index()
        
        idx = len(self._entries)
        self._entries.append(entry)
        
        situation_key = self._normalize_situation(situation)
        if situation_key not in self._situation_index:
            self._situation_index[situation_key] = []
        self._situation_index[situation_key].append(idx)

    def _normalize_situation(self, situation: str) -> str:
        words = situation.lower().split()[:5]
        return " ".join(sorted(words))

    def _rebuild_index(self) -> None:
        self._situation_index.clear()
        for idx, entry in enumerate(self._entries):
            key = self._normalize_situation(entry.situation)
            if key not in self._situation_index:
                self._situation_index[key] = []
            self._situation_index[key].append(idx)

    def find_similar(self, situation: str, top_k: int = 5) -> List[LearningEntry]:
        key = self._normalize_situation(situation)
        
        if key in self._situation_index:
            indices = self._situation_index[key][-top_k:]
            return [self._entries[i] for i in indices]
        
        return self._entries[-top_k:] if self._entries else []

    def get_best_action(self, situation: str) -> Optional[Tuple[str, float]]:
        similar = self.find_similar(situation)
        if not similar:
            return None
        
        action_rewards: Dict[str, List[float]] = {}
        for entry in similar:
            if entry.action not in action_rewards:
                action_rewards[entry.action] = []
            action_rewards[entry.action].append(entry.reward)
        
        best_action = max(
            action_rewards.keys(),
            key=lambda a: sum(action_rewards[a]) / len(action_rewards[a])
        )
        avg_reward = sum(action_rewards[best_action]) / len(action_rewards[best_action])
        
        return best_action, avg_reward

    def get_stats(self) -> Dict[str, Any]:
        if not self._entries:
            return {"total": 0, "avg_reward": 0.0}
        
        rewards = [e.reward for e in self._entries]
        return {
            "total": len(self._entries),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "unique_situations": len(self._situation_index),
        }

    def clear(self) -> None:
        self._entries.clear()
        self._situation_index.clear()


class SelfReflector:
    """
    Main self-reflection system for autonomous agents.
    
    Coordinates reflection engines, strategy adjustment, and learning memory.
    
    Example:
        >>> reflector = SelfReflector()
        >>> reflection = reflector.reflect_on_action(
        ...     goal="Write a Python script",
        ...     action="execute_code",
        ...     result="Script executed successfully",
        ...     success=True
        ... )
        >>> print(reflection.insights)
    """

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.success_analyzer = SuccessAnalyzer(llm_func)
        self.failure_analyzer = FailureAnalyzer(llm_func)
        self.strategy_adjuster = StrategyAdjuster()
        self.learning_memory = LearningMemory()
        self._reflections: List[Reflection] = []

    def reflect_on_action(
        self,
        goal: str,
        action: str,
        result: str,
        success: bool,
        goal_id: Optional[str] = None,
        action_id: Optional[str] = None,
    ) -> Reflection:
        context = {
            "goal": goal,
            "goal_id": goal_id,
            "actions": [action],
            "error": result if not success else None,
        }
        
        if success:
            reflection = self.success_analyzer.reflect(context, result, success)
            reward = 1.0
        else:
            reflection = self.failure_analyzer.reflect(context, result, success)
            reward = -0.5
        
        reflection.action_id = action_id
        self._reflections.append(reflection)
        
        self.learning_memory.add_entry(
            situation=goal,
            action=action,
            outcome=result,
            reward=reward,
        )
        
        return reflection

    def reflect_on_goal(
        self,
        goal: str,
        actions: List[str],
        final_result: str,
        success: bool,
        goal_id: Optional[str] = None,
    ) -> Reflection:
        context = {
            "goal": goal,
            "goal_id": goal_id,
            "actions": actions,
            "error": final_result if not success else None,
        }
        
        if success:
            reflection = self.success_analyzer.reflect(context, final_result, success)
        else:
            reflection = self.failure_analyzer.reflect(context, final_result, success)
        
        self._reflections.append(reflection)
        return reflection

    def get_advice(self, situation: str) -> Optional[str]:
        best = self.learning_memory.get_best_action(situation)
        if best:
            action, reward = best
            if reward > 0:
                return f"Based on past experience, consider: {action}"
        return None

    def suggest_strategy(self) -> Optional[str]:
        return self.strategy_adjuster.select_strategy()

    def record_strategy_outcome(self, strategy: str, success: bool) -> None:
        reward = 1.0 if success else 0.0
        self.strategy_adjuster.record_outcome(strategy, reward)

    def get_reflections(self, limit: int = 10) -> List[Reflection]:
        return self._reflections[-limit:]

    def get_insights_summary(self) -> Dict[str, Any]:
        if not self._reflections:
            return {"total_reflections": 0}
        
        successes = sum(1 for r in self._reflections if r.reflection_type == ReflectionType.SUCCESS)
        failures = sum(1 for r in self._reflections if r.reflection_type == ReflectionType.FAILURE)
        
        all_insights = []
        for r in self._reflections[-20:]:
            all_insights.extend(r.insights)
        
        return {
            "total_reflections": len(self._reflections),
            "successes": successes,
            "failures": failures,
            "success_rate": successes / len(self._reflections) if self._reflections else 0,
            "recent_insights": all_insights[-10:],
            "common_failures": self.failure_analyzer.get_common_failures()[:5],
            "learning_stats": self.learning_memory.get_stats(),
        }

    def reset(self) -> None:
        self._reflections.clear()
        self.learning_memory.clear()
        self.failure_analyzer.failure_patterns.clear()

    def __repr__(self) -> str:
        stats = self.get_insights_summary()
        return f"SelfReflector(reflections={stats['total_reflections']}, success_rate={stats.get('success_rate', 0):.1%})"
