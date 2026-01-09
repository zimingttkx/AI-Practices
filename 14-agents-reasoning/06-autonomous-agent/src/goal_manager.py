"""
Goal Manager: Hierarchical Goal Management for Autonomous Agents.

Core Idea:
    This module provides goal decomposition, prioritization, and completion tracking
    for autonomous agents. Goals are organized hierarchically and managed through
    a priority-based execution system.

Mathematical Foundation:
    Goal priority scoring:
        priority = α·urgency + β·importance + γ·(1/dependency_depth)
    
    Goal completion:
        completed(G) = ∀g ∈ subgoals(G): completed(g)
    
    Hierarchical decomposition:
        G → {g₁, g₂, ..., gₙ} where Σ effort(gᵢ) ≈ effort(G)

Design Patterns:
    - Composite Pattern: Goals contain sub-goals
    - Strategy Pattern: Different decomposition strategies
    - Observer Pattern: Goal state change notifications

References:
    - HTN Planning: Hierarchical Task Network
    - AutoGPT: https://github.com/Significant-Gravitas/AutoGPT
    - BabyAGI: https://github.com/yoheinakajima/babyagi

Author: AI-Practices
Version: 1.0.0
"""

from __future__ import annotations

import uuid
import heapq
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
    Set,
    Tuple,
)

__all__ = [
    "GoalStatus",
    "GoalPriority",
    "Goal",
    "GoalDecomposer",
    "LLMGoalDecomposer",
    "RuleBasedDecomposer",
    "GoalPriorityQueue",
    "CompletionChecker",
    "GoalManager",
]


class GoalStatus(str, Enum):
    """Goal lifecycle status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        return self in {GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.CANCELLED}


class GoalPriority(int, Enum):
    """Goal priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Goal:
    """
    Represents a goal in the autonomous agent system.
    
    Attributes:
        goal_id: Unique identifier
        description: Human-readable goal description
        priority: Priority level
        status: Current status
        parent_id: Parent goal ID (for sub-goals)
        sub_goals: List of child goal IDs
        dependencies: Goals that must complete first
        success_criteria: Conditions for completion
        metadata: Additional goal data
        created_at: Creation timestamp
        completed_at: Completion timestamp
        max_attempts: Maximum retry attempts
        attempt_count: Current attempt count
    """
    description: str
    goal_id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    parent_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    success_criteria: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    max_attempts: int = 3
    attempt_count: int = 0

    def __lt__(self, other: "Goal") -> bool:
        return self.priority.value < other.priority.value

    def is_leaf(self) -> bool:
        return len(self.sub_goals) == 0

    def is_actionable(self) -> bool:
        return self.status == GoalStatus.PENDING and self.is_leaf()

    def mark_in_progress(self) -> None:
        self.status = GoalStatus.IN_PROGRESS
        self.attempt_count += 1

    def mark_completed(self) -> None:
        self.status = GoalStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def mark_failed(self, can_retry: bool = True) -> None:
        if can_retry and self.attempt_count < self.max_attempts:
            self.status = GoalStatus.PENDING
        else:
            self.status = GoalStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority.name,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "sub_goals": self.sub_goals,
            "dependencies": self.dependencies,
            "success_criteria": self.success_criteria,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempt_count": self.attempt_count,
        }


class GoalDecomposer(ABC):
    """Abstract base class for goal decomposition strategies."""

    @abstractmethod
    def decompose(self, goal: Goal) -> List[Goal]:
        """
        Decompose a goal into sub-goals.
        
        Args:
            goal: The goal to decompose
            
        Returns:
            List of sub-goals
        """
        pass


class LLMGoalDecomposer(GoalDecomposer):
    """
    LLM-based goal decomposition.
    
    Uses a language model to intelligently break down complex goals
    into actionable sub-goals.
    """

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
        max_sub_goals: int = 5,
    ) -> None:
        self.llm_func = llm_func or self._mock_llm
        self.max_sub_goals = max_sub_goals

    def _mock_llm(self, prompt: str) -> str:
        return "1. Research the topic\n2. Create an outline\n3. Write the content\n4. Review and refine"

    def decompose(self, goal: Goal) -> List[Goal]:
        prompt = f"""Break down the following goal into {self.max_sub_goals} or fewer actionable sub-goals.
Each sub-goal should be specific, measurable, and achievable.

Goal: {goal.description}

List the sub-goals (one per line, numbered):"""

        response = self.llm_func(prompt)
        sub_goals = self._parse_response(response, goal)
        return sub_goals

    def _parse_response(self, response: str, parent: Goal) -> List[Goal]:
        sub_goals = []
        lines = response.strip().split("\n")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            for prefix in ["- ", "* ", f"{i+1}. ", f"{i+1}) "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            
            if line:
                sub_goal = Goal(
                    description=line,
                    priority=parent.priority,
                    parent_id=parent.goal_id,
                )
                sub_goals.append(sub_goal)
        
        return sub_goals[:self.max_sub_goals]


class RuleBasedDecomposer(GoalDecomposer):
    """
    Rule-based goal decomposition using predefined patterns.
    """

    def __init__(self) -> None:
        self.patterns: Dict[str, List[str]] = {
            "write": ["Research topic", "Create outline", "Write draft", "Review and edit"],
            "build": ["Design architecture", "Implement core", "Add features", "Test and debug"],
            "learn": ["Find resources", "Study fundamentals", "Practice exercises", "Apply knowledge"],
            "analyze": ["Gather data", "Process data", "Identify patterns", "Draw conclusions"],
        }

    def decompose(self, goal: Goal) -> List[Goal]:
        description_lower = goal.description.lower()
        
        for keyword, steps in self.patterns.items():
            if keyword in description_lower:
                return [
                    Goal(
                        description=f"{step}: {goal.description}",
                        priority=goal.priority,
                        parent_id=goal.goal_id,
                    )
                    for step in steps
                ]
        
        return [
            Goal(
                description=f"Step 1: Plan approach for '{goal.description}'",
                priority=goal.priority,
                parent_id=goal.goal_id,
            ),
            Goal(
                description=f"Step 2: Execute '{goal.description}'",
                priority=goal.priority,
                parent_id=goal.goal_id,
            ),
            Goal(
                description=f"Step 3: Verify completion of '{goal.description}'",
                priority=goal.priority,
                parent_id=goal.goal_id,
            ),
        ]


class GoalPriorityQueue:
    """
    Priority queue for goals with dependency awareness.
    
    Goals are ordered by:
    1. Priority level (CRITICAL > HIGH > MEDIUM > LOW > BACKGROUND)
    2. Creation time (older goals first within same priority)
    """

    def __init__(self) -> None:
        self._heap: List[Tuple[int, float, Goal]] = []
        self._goal_set: Set[str] = set()

    def push(self, goal: Goal) -> None:
        if goal.goal_id in self._goal_set:
            return
        entry = (goal.priority.value, goal.created_at.timestamp(), goal)
        heapq.heappush(self._heap, entry)
        self._goal_set.add(goal.goal_id)

    def pop(self) -> Optional[Goal]:
        while self._heap:
            _, _, goal = heapq.heappop(self._heap)
            if goal.goal_id in self._goal_set:
                self._goal_set.discard(goal.goal_id)
                return goal
        return None

    def peek(self) -> Optional[Goal]:
        while self._heap:
            if self._heap[0][2].goal_id in self._goal_set:
                return self._heap[0][2]
            heapq.heappop(self._heap)
        return None

    def remove(self, goal_id: str) -> bool:
        if goal_id in self._goal_set:
            self._goal_set.discard(goal_id)
            return True
        return False

    def __len__(self) -> int:
        return len(self._goal_set)

    def __bool__(self) -> bool:
        return bool(self._goal_set)

    def to_list(self) -> List[Goal]:
        return [g for _, _, g in sorted(self._heap) if g.goal_id in self._goal_set]


class CompletionChecker:
    """
    Checks goal completion based on various criteria.
    
    Supports:
    - Keyword-based completion detection
    - LLM-based semantic completion checking
    - Custom completion functions
    """

    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.llm_func = llm_func
        self.completion_keywords = [
            "done", "completed", "finished", "success", "achieved",
            "accomplished", "fulfilled", "resolved"
        ]

    def check_completion(
        self,
        goal: Goal,
        result: str,
        use_llm: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if a goal is completed based on the result.
        
        Args:
            goal: The goal to check
            result: The result/output from executing the goal
            use_llm: Whether to use LLM for semantic checking
            
        Returns:
            Tuple of (is_completed, reason)
        """
        if goal.success_criteria:
            return self._check_criteria(goal, result, use_llm)
        return self._check_keywords(result)

    def _check_keywords(self, result: str) -> Tuple[bool, str]:
        result_lower = result.lower()
        for keyword in self.completion_keywords:
            if keyword in result_lower:
                return True, f"Completion keyword '{keyword}' found"
        return False, "No completion indicators found"

    def _check_criteria(
        self,
        goal: Goal,
        result: str,
        use_llm: bool,
    ) -> Tuple[bool, str]:
        if use_llm and self.llm_func:
            prompt = f"""Determine if the goal has been completed.

Goal: {goal.description}
Success Criteria: {goal.success_criteria}
Result: {result}

Answer with 'YES' or 'NO' followed by a brief explanation."""
            
            response = self.llm_func(prompt).strip().upper()
            is_completed = response.startswith("YES")
            return is_completed, response
        
        criteria_lower = goal.success_criteria.lower() if goal.success_criteria else ""
        result_lower = result.lower()
        
        criteria_words = set(criteria_lower.split())
        result_words = set(result_lower.split())
        overlap = len(criteria_words & result_words) / max(len(criteria_words), 1)
        
        if overlap > 0.5:
            return True, f"Criteria overlap: {overlap:.0%}"
        return False, f"Insufficient criteria match: {overlap:.0%}"


class GoalManager:
    """
    Central manager for goal lifecycle in autonomous agents.
    
    Responsibilities:
    - Goal creation and storage
    - Goal decomposition
    - Priority queue management
    - Dependency tracking
    - Completion checking
    - State persistence
    
    Example:
        >>> manager = GoalManager()
        >>> goal = manager.add_goal("Build a web scraper")
        >>> manager.decompose_goal(goal.goal_id)
        >>> next_goal = manager.get_next_goal()
        >>> manager.complete_goal(next_goal.goal_id, "Scraper built successfully")
    """

    def __init__(
        self,
        decomposer: Optional[GoalDecomposer] = None,
        completion_checker: Optional[CompletionChecker] = None,
    ) -> None:
        self.decomposer = decomposer or LLMGoalDecomposer()
        self.completion_checker = completion_checker or CompletionChecker()
        self._goals: Dict[str, Goal] = {}
        self._queue = GoalPriorityQueue()
        self._root_goals: List[str] = []

    def add_goal(
        self,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        success_criteria: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        auto_decompose: bool = False,
    ) -> Goal:
        """Add a new goal to the manager."""
        goal = Goal(
            description=description,
            priority=priority,
            success_criteria=success_criteria,
            dependencies=dependencies or [],
        )
        self._goals[goal.goal_id] = goal
        self._root_goals.append(goal.goal_id)
        
        if auto_decompose:
            self.decompose_goal(goal.goal_id)
        else:
            self._queue.push(goal)
        
        return goal

    def decompose_goal(self, goal_id: str) -> List[Goal]:
        """Decompose a goal into sub-goals."""
        goal = self._goals.get(goal_id)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")
        
        sub_goals = self.decomposer.decompose(goal)
        
        for i, sub_goal in enumerate(sub_goals):
            if i > 0:
                sub_goal.dependencies.append(sub_goals[i-1].goal_id)
            self._goals[sub_goal.goal_id] = sub_goal
            goal.sub_goals.append(sub_goal.goal_id)
        
        self._queue.remove(goal_id)
        
        if sub_goals:
            self._queue.push(sub_goals[0])
        
        return sub_goals

    def get_next_goal(self) -> Optional[Goal]:
        """Get the next actionable goal."""
        while self._queue:
            goal = self._queue.peek()
            if not goal:
                break
            
            if self._dependencies_met(goal):
                return self._queue.pop()
            
            goal.status = GoalStatus.BLOCKED
            self._queue.pop()
        
        return None

    def _dependencies_met(self, goal: Goal) -> bool:
        for dep_id in goal.dependencies:
            dep = self._goals.get(dep_id)
            if dep and dep.status != GoalStatus.COMPLETED:
                return False
        return True

    def start_goal(self, goal_id: str) -> Goal:
        """Mark a goal as in progress."""
        goal = self._goals.get(goal_id)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")
        goal.mark_in_progress()
        return goal

    def complete_goal(
        self,
        goal_id: str,
        result: str,
        use_llm_check: bool = False,
    ) -> Tuple[bool, str]:
        """Attempt to complete a goal."""
        goal = self._goals.get(goal_id)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")
        
        is_completed, reason = self.completion_checker.check_completion(
            goal, result, use_llm_check
        )
        
        if is_completed:
            goal.mark_completed()
            self._propagate_completion(goal)
            self._unblock_dependents(goal_id)
        else:
            goal.mark_failed(can_retry=True)
            if goal.status == GoalStatus.PENDING:
                self._queue.push(goal)
        
        return is_completed, reason

    def _propagate_completion(self, goal: Goal) -> None:
        """Propagate completion to parent goals."""
        if not goal.parent_id:
            return
        
        parent = self._goals.get(goal.parent_id)
        if not parent:
            return
        
        all_completed = all(
            self._goals[sg_id].status == GoalStatus.COMPLETED
            for sg_id in parent.sub_goals
            if sg_id in self._goals
        )
        
        if all_completed:
            parent.mark_completed()
            self._propagate_completion(parent)

    def _unblock_dependents(self, completed_goal_id: str) -> None:
        """Unblock goals that depend on the completed goal."""
        for goal in self._goals.values():
            if completed_goal_id in goal.dependencies:
                if goal.status == GoalStatus.BLOCKED:
                    if self._dependencies_met(goal):
                        goal.status = GoalStatus.PENDING
                        self._queue.push(goal)

    def fail_goal(self, goal_id: str, reason: str = "") -> None:
        """Mark a goal as failed."""
        goal = self._goals.get(goal_id)
        if goal:
            goal.mark_failed(can_retry=False)
            goal.metadata["failure_reason"] = reason

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def get_all_goals(self) -> List[Goal]:
        """Get all goals."""
        return list(self._goals.values())

    def get_pending_goals(self) -> List[Goal]:
        """Get all pending goals."""
        return [g for g in self._goals.values() if g.status == GoalStatus.PENDING]

    def get_progress(self) -> Dict[str, Any]:
        """Get overall progress statistics."""
        total = len(self._goals)
        if total == 0:
            return {"total": 0, "completed": 0, "progress": 0.0}
        
        completed = sum(1 for g in self._goals.values() if g.status == GoalStatus.COMPLETED)
        failed = sum(1 for g in self._goals.values() if g.status == GoalStatus.FAILED)
        in_progress = sum(1 for g in self._goals.values() if g.status == GoalStatus.IN_PROGRESS)
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": total - completed - failed - in_progress,
            "progress": completed / total,
        }

    def reset(self) -> None:
        """Reset the goal manager."""
        self._goals.clear()
        self._queue = GoalPriorityQueue()
        self._root_goals.clear()

    def __repr__(self) -> str:
        progress = self.get_progress()
        return f"GoalManager(goals={progress['total']}, completed={progress['completed']})"
