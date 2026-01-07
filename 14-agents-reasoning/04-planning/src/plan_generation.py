"""
Plan Generation: Creating Executable Plans from Goals.

Core Idea:
    Plan generation transforms high-level goals into structured, executable
    plans consisting of ordered tasks with dependencies. This module implements
    multiple planning strategies inspired by classical AI planning and modern
    LLM-based approaches.

Mathematical Foundation:
    Planning can be formalized as a search problem:

    $$P = (S, A, T, s_0, G, C)$$

    where:
    - $S$: State space (all possible world states)
    - $A$: Action space (available operations)
    - $T: S \\times A \\rightarrow S$: Transition function
    - $s_0 \\in S$: Initial state
    - $G \\subseteq S$: Goal states
    - $C: A \\rightarrow \\mathbb{R}^+$: Cost function

    Objective: Find action sequence $\\pi = [a_1, ..., a_n]$ such that:
    $$T(...T(T(s_0, a_1), a_2)..., a_n) \\in G$$
    $$\\text{minimize } \\sum_{i=1}^{n} C(a_i)$$

Problem Statement:
    Given a goal description, generate a plan that:
    1. Achieves the goal when executed
    2. Respects constraints and dependencies
    3. Is efficient and practical

Planning Strategies:
    | Strategy      | Approach              | Best For                    |
    |---------------|-----------------------|-----------------------------|
    | Forward       | Start -> Goal         | Clear initial state         |
    | Backward      | Goal -> Start         | Clear goal conditions       |
    | Hierarchical  | Abstract -> Concrete  | Complex multi-level tasks   |
    | Constraint    | Satisfy constraints   | Resource-limited scenarios  |

References:
    - STRIPS: Fikes & Nilsson (1971)
    - HTN Planning: Erol, Hendler, Nau (1994)
    - LLM Planning: Huang et al. (2022) "Language Models as Zero-Shot Planners"
"""

from __future__ import annotations

import json
import re
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
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
import uuid

try:
    from .task_decomposition import (
    Task,
    TaskStatus,
    TaskPriority,
    TaskType,
    DependencyAnalyzer,
    LLMInterface,
    )
except ImportError:
    from task_decomposition import (
        Task,
        TaskStatus,
        TaskPriority,
        TaskType,
        DependencyAnalyzer,
        LLMInterface,
    )

__all__ = [
    "Plan",
    "PlanStatus",
    "PlanningStrategy",
    "Constraint",
    "ConstraintType",
    "PlanGenerator",
    "ForwardPlanner",
    "BackwardPlanner",
    "HierarchicalPlanner",
    "PlanValidator",
    "ValidationResult",
    "create_planner",
]


# =============================================================================
# Constants and Enums
# =============================================================================

DEFAULT_MAX_TASKS: Final[int] = 20
DEFAULT_MAX_RETRIES: Final[int] = 3


class PlanStatus(str, Enum):
    """Status of a plan in its lifecycle."""
    DRAFT: Final[str] = "draft"
    VALIDATED: Final[str] = "validated"
    IN_PROGRESS: Final[str] = "in_progress"
    COMPLETED: Final[str] = "completed"
    FAILED: Final[str] = "failed"
    CANCELLED: Final[str] = "cancelled"


class PlanningStrategy(str, Enum):
    """Available planning strategies."""
    FORWARD: Final[str] = "forward"
    BACKWARD: Final[str] = "backward"
    HIERARCHICAL: Final[str] = "hierarchical"


class ConstraintType(str, Enum):
    """Types of constraints on plans."""
    TIME: Final[str] = "time"
    RESOURCE: Final[str] = "resource"
    DEPENDENCY: Final[str] = "dependency"
    ORDERING: Final[str] = "ordering"
    EXCLUSION: Final[str] = "exclusion"


# =============================================================================
# Constraint Data Structure
# =============================================================================


@dataclass
class Constraint:
    """Represents a constraint on plan execution.

    Attributes:
        name: Constraint identifier.
        constraint_type: Type of constraint.
        description: Human-readable description.
        parameters: Constraint-specific parameters.
        is_hard: If True, must be satisfied; if False, preference only.
    """
    name: str
    constraint_type: ConstraintType
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_hard: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.constraint_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "is_hard": self.is_hard,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        return cls(
            name=data["name"],
            constraint_type=ConstraintType(data["type"]),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            is_hard=data.get("is_hard", True),
        )


# =============================================================================
# Plan Data Structure
# =============================================================================


@dataclass
class Plan:
    """Executable plan consisting of ordered tasks.

    Core Idea:
        A Plan is a directed acyclic graph (DAG) of tasks with a defined
        goal. It supports execution tracking, progress monitoring, and
        dynamic modification.

    Mathematical Model:
        Plan as DAG: G = (V, E) where V = tasks, E = dependencies
        Valid execution: topological ordering of G
        Progress: |completed tasks| / |total tasks|

    Attributes:
        goal: High-level objective of the plan.
        tasks: List of tasks in the plan.
        constraints: Constraints on plan execution.
        status: Current plan status.
        metadata: Additional plan information.
    """
    goal: str
    tasks: List[Task] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = PlanStatus(self.status)

    def add_task(self, task: Task) -> Task:
        """Add a task to the plan."""
        self.tasks.append(task)
        return task

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the plan."""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                self.tasks.pop(i)
                # Remove from dependencies
                for other in self.tasks:
                    if task_id in other.dependencies:
                        other.dependencies.remove(task_id)
                return True
        return False

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the plan."""
        self.constraints.append(constraint)

    @property
    def progress(self) -> float:
        """Calculate plan completion progress (0.0 to 1.0)."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    @property
    def has_failed_tasks(self) -> bool:
        """Check if any task has failed."""
        return any(t.status == TaskStatus.FAILED for t in self.tasks)

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks ready for execution (dependencies satisfied)."""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        ready = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                deps_satisfied = all(d in completed_ids for d in task.dependencies)
                if deps_satisfied:
                    ready.append(task)
        return ready

    def get_execution_order(self) -> List[Task]:
        """Get tasks in topological order."""
        in_degree: Dict[str, int] = {t.id: 0 for t in self.tasks}
        
        for task in self.tasks:
            for dep_id in task.dependencies:
                if dep_id in in_degree:
                    in_degree[task.id] += 1

        queue = [t for t in self.tasks if in_degree[t.id] == 0]
        result: List[Task] = []

        while queue:
            queue.sort(key=lambda t: t.priority.to_numeric(), reverse=True)
            task = queue.pop(0)
            result.append(task)

            for other in self.tasks:
                if task.id in other.dependencies:
                    in_degree[other.id] -= 1
                    if in_degree[other.id] == 0:
                        queue.append(other)

        if len(result) != len(self.tasks):
            raise ValueError("Circular dependency detected")

        return result

    def start(self) -> None:
        """Mark plan as started."""
        self.status = PlanStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark plan as completed."""
        self.status = PlanStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def fail(self) -> None:
        """Mark plan as failed."""
        self.status = PlanStatus.FAILED
        self.completed_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plan to dictionary."""
        return {
            "id": self.id,
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
            "constraints": [c.to_dict() for c in self.constraints],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        """Deserialize plan from dictionary."""
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        constraints = [Constraint.from_dict(c) for c in data.get("constraints", [])]
        
        for dt_field in ["created_at", "started_at", "completed_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        return cls(
            id=data.get("id", f"plan_{uuid.uuid4().hex[:8]}"),
            goal=data["goal"],
            tasks=tasks,
            constraints=constraints,
            status=PlanStatus(data.get("status", "draft")),
            created_at=data.get("created_at", datetime.utcnow()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        return f"Plan({self.id}): {self.goal} [{self.status.value}] {self.progress:.0%}"

    def __repr__(self) -> str:
        return f"Plan(id={self.id!r}, goal={self.goal!r}, tasks={len(self.tasks)})"


# =============================================================================
# Plan Generator Abstract Base
# =============================================================================


class PlanGenerator(ABC):
    """Abstract base class for plan generators.

    Core Idea:
        Different planning strategies suit different problem types.
        This abstract class defines the common interface.
    """

    @abstractmethod
    def generate(
        self,
        goal: str,
        context: str = "",
        constraints: Optional[List[Constraint]] = None,
    ) -> Plan:
        """Generate a plan to achieve the goal."""
        pass


class ForwardPlanner(PlanGenerator):
    """Forward planning: Start from initial state, work toward goal.

    Core Idea:
        Generates a plan by progressively adding actions that move
        from the current state toward the goal state.

    Algorithm:
        1. Analyze the goal
        2. Identify required actions
        3. Order actions by dependencies
        4. Create plan with tasks
    """

    PLAN_PROMPT: Final[str] = '''You are an expert planner. Create a step-by-step plan to achieve the following goal.

Goal: {goal}
Context: {context}
Constraints: {constraints}

Requirements:
1. Break down the goal into 3-10 specific, actionable tasks
2. Order tasks logically (dependencies first)
3. Each task should be concrete and achievable
4. Consider any constraints provided

Format your response as a numbered list:
1. [Task Name]: [Description] (Priority: high/medium/low)
2. [Task Name]: [Description] (Priority: high/medium/low)
...

Plan:'''

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_tasks: int = DEFAULT_MAX_TASKS,
    ) -> None:
        self._llm = llm
        self._max_tasks = max_tasks
        self._analyzer = DependencyAnalyzer(llm)

    def generate(
        self,
        goal: str,
        context: str = "",
        constraints: Optional[List[Constraint]] = None,
    ) -> Plan:
        """Generate a forward plan."""
        plan = Plan(goal=goal)
        
        if constraints:
            for c in constraints:
                plan.add_constraint(c)

        if self._llm is None:
            tasks = self._generate_placeholder_tasks(goal)
        else:
            constraints_str = self._format_constraints(constraints)
            prompt = self.PLAN_PROMPT.format(
                goal=goal,
                context=context or "No additional context.",
                constraints=constraints_str,
            )
            response = self._llm.generate(prompt)
            tasks = self._parse_tasks(response)

        # Add sequential dependencies
        for i, task in enumerate(tasks):
            if i > 0:
                task.add_dependency(tasks[i - 1].id)
            plan.add_task(task)

        plan.status = PlanStatus.DRAFT
        return plan

    def _generate_placeholder_tasks(self, goal: str) -> List[Task]:
        """Generate placeholder tasks when no LLM available."""
        return [
            Task(
                name="Analyze Requirements",
                description=f"Understand what is needed for: {goal}",
                task_type=TaskType.ANALYSIS,
                priority=TaskPriority.HIGH,
            ),
            Task(
                name="Design Solution",
                description=f"Design approach for: {goal}",
                task_type=TaskType.ANALYSIS,
                priority=TaskPriority.HIGH,
            ),
            Task(
                name="Implement Solution",
                description=f"Execute the plan for: {goal}",
                task_type=TaskType.IMPLEMENTATION,
                priority=TaskPriority.MEDIUM,
            ),
            Task(
                name="Test and Verify",
                description=f"Verify goal achievement: {goal}",
                task_type=TaskType.TESTING,
                priority=TaskPriority.MEDIUM,
            ),
        ]

    def _format_constraints(self, constraints: Optional[List[Constraint]]) -> str:
        """Format constraints for prompt."""
        if not constraints:
            return "None"
        return "\n".join(f"- {c.name}: {c.description}" for c in constraints)

    def _parse_tasks(self, response: str) -> List[Task]:
        """Parse tasks from LLM response."""
        tasks = []
        pattern = r'(\d+)\.\s*\[?([^\]:\n]+)\]?:\s*(.+?)(?:\(Priority:\s*(\w+)\))?(?=\d+\.|$)'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for _, name, description, priority in matches:
            priority_enum = self._parse_priority(priority)
            tasks.append(Task(
                name=name.strip(),
                description=description.strip(),
                priority=priority_enum,
            ))

        if not tasks:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            for line in lines:
                if line and line[0].isdigit():
                    content = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if content:
                        tasks.append(Task(
                            name=content[:50],
                            description=content,
                        ))

        return tasks[:self._max_tasks]

    def _parse_priority(self, priority_str: str) -> TaskPriority:
        """Parse priority string to enum."""
        if not priority_str:
            return TaskPriority.MEDIUM
        priority_str = priority_str.lower().strip()
        mapping = {
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
        }
        return mapping.get(priority_str, TaskPriority.MEDIUM)


class BackwardPlanner(PlanGenerator):
    """Backward planning: Start from goal, work backward to initial state.

    Core Idea:
        Generates a plan by identifying what conditions must be true
        for the goal, then finding actions that achieve those conditions.
    """

    BACKWARD_PROMPT: Final[str] = '''You are an expert planner using backward chaining. Start from the goal and work backward.

Goal: {goal}
Context: {context}

Think backward:
1. What must be true for the goal to be achieved?
2. What actions create those conditions?
3. What preconditions do those actions need?
4. Continue until you reach actions that can be done now.

List the tasks in EXECUTION order (not backward order):
1. [Task Name]: [Description]
2. [Task Name]: [Description]
...

Plan:'''

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_tasks: int = DEFAULT_MAX_TASKS,
    ) -> None:
        self._llm = llm
        self._max_tasks = max_tasks

    def generate(
        self,
        goal: str,
        context: str = "",
        constraints: Optional[List[Constraint]] = None,
    ) -> Plan:
        """Generate a backward-chained plan."""
        plan = Plan(goal=goal)
        
        if constraints:
            for c in constraints:
                plan.add_constraint(c)

        if self._llm is None:
            tasks = self._generate_placeholder_tasks(goal)
        else:
            prompt = self.BACKWARD_PROMPT.format(
                goal=goal,
                context=context or "No additional context.",
            )
            response = self._llm.generate(prompt)
            tasks = self._parse_tasks(response)

        for i, task in enumerate(tasks):
            if i > 0:
                task.add_dependency(tasks[i - 1].id)
            plan.add_task(task)

        return plan

    def _generate_placeholder_tasks(self, goal: str) -> List[Task]:
        """Generate placeholder tasks."""
        return [
            Task(name="Prerequisites", description=f"Ensure prerequisites for: {goal}"),
            Task(name="Core Action", description=f"Main action for: {goal}"),
            Task(name="Verification", description=f"Verify: {goal}"),
        ]

    def _parse_tasks(self, response: str) -> List[Task]:
        """Parse tasks from response."""
        tasks = []
        pattern = r'(\d+)\.\s*\[?([^\]:\n]+)\]?:\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for _, name, description in matches:
            tasks.append(Task(
                name=name.strip(),
                description=description.strip(),
            ))

        return tasks[:self._max_tasks]


class HierarchicalPlanner(PlanGenerator):
    """Hierarchical planning: Decompose into abstract levels then refine.

    Core Idea:
        Creates plans at multiple abstraction levels, starting with
        high-level phases and refining into concrete tasks.

    Algorithm:
        1. Generate high-level phases (2-4)
        2. For each phase, generate detailed tasks
        3. Establish dependencies within and across phases
    """

    PHASE_PROMPT: Final[str] = '''Create a high-level plan with 2-4 major phases for this goal.

Goal: {goal}
Context: {context}

Each phase should be a significant milestone. Format:
Phase 1: [Name] - [Description]
Phase 2: [Name] - [Description]
...

Phases:'''

    DETAIL_PROMPT: Final[str] = '''Break down this phase into specific tasks.

Phase: {phase_name}
Description: {phase_description}
Overall Goal: {goal}

Create 2-4 concrete tasks for this phase:
1. [Task Name]: [Description]
2. [Task Name]: [Description]
...

Tasks:'''

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_phases: int = 4,
        max_tasks_per_phase: int = 5,
    ) -> None:
        self._llm = llm
        self._max_phases = max_phases
        self._max_tasks_per_phase = max_tasks_per_phase

    def generate(
        self,
        goal: str,
        context: str = "",
        constraints: Optional[List[Constraint]] = None,
    ) -> Plan:
        """Generate a hierarchical plan."""
        plan = Plan(goal=goal)
        
        if constraints:
            for c in constraints:
                plan.add_constraint(c)

        phases = self._generate_phases(goal, context)
        
        prev_phase_last_task: Optional[Task] = None
        for phase in phases:
            phase_tasks = self._detail_phase(phase, goal)
            
            # First task of phase depends on last task of previous phase
            if prev_phase_last_task and phase_tasks:
                phase_tasks[0].add_dependency(prev_phase_last_task.id)
            
            # Sequential dependencies within phase
            for i, task in enumerate(phase_tasks):
                if i > 0:
                    task.add_dependency(phase_tasks[i - 1].id)
                task.metadata["phase"] = phase.name
                plan.add_task(task)
            
            if phase_tasks:
                prev_phase_last_task = phase_tasks[-1]

        return plan

    def _generate_phases(self, goal: str, context: str) -> List[Task]:
        """Generate high-level phases."""
        if self._llm is None:
            return [
                Task(name="Planning", description="Plan the approach"),
                Task(name="Implementation", description="Execute the plan"),
                Task(name="Verification", description="Verify results"),
            ]

        prompt = self.PHASE_PROMPT.format(goal=goal, context=context)
        response = self._llm.generate(prompt)
        return self._parse_phases(response)

    def _parse_phases(self, response: str) -> List[Task]:
        """Parse phases from response."""
        phases = []
        pattern = r'Phase\s*\d+:\s*\[?([^\]:\n-]+)\]?\s*[-:]\s*(.+?)(?=Phase|$)'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for name, desc in matches:
            phases.append(Task(name=name.strip(), description=desc.strip()))

        return phases[:self._max_phases] or [
            Task(name="Main Phase", description=response.strip())
        ]

    def _detail_phase(self, phase: Task, goal: str) -> List[Task]:
        """Generate detailed tasks for a phase."""
        if self._llm is None:
            return [
                Task(name=f"{phase.name} - Step 1", description="First step"),
                Task(name=f"{phase.name} - Step 2", description="Second step"),
            ]

        prompt = self.DETAIL_PROMPT.format(
            phase_name=phase.name,
            phase_description=phase.description,
            goal=goal,
        )
        response = self._llm.generate(prompt)
        return self._parse_tasks(response)[:self._max_tasks_per_phase]

    def _parse_tasks(self, response: str) -> List[Task]:
        """Parse tasks from response."""
        tasks = []
        pattern = r'(\d+)\.\s*\[?([^\]:\n]+)\]?:\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for _, name, desc in matches:
            tasks.append(Task(name=name.strip(), description=desc.strip()))

        return tasks


# =============================================================================
# Plan Validation
# =============================================================================


@dataclass
class ValidationResult:
    """Result of plan validation."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 1.0  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "score": self.score,
        }


class PlanValidator:
    """Validates plans for correctness and feasibility.

    Checks:
    1. Structural validity (no cycles, valid dependencies)
    2. Completeness (all required tasks present)
    3. Constraint satisfaction
    4. Feasibility (tasks are actionable)
    """

    def __init__(self, llm: Optional[LLMInterface] = None) -> None:
        self._llm = llm
        self._analyzer = DependencyAnalyzer(llm)

    def validate(self, plan: Plan) -> ValidationResult:
        """Validate a plan."""
        issues: List[str] = []
        warnings: List[str] = []

        # Check for empty plan
        if not plan.tasks:
            issues.append("Plan has no tasks")
            return ValidationResult(is_valid=False, issues=issues)

        # Check for cycles
        cycles = self._analyzer.detect_cycles(plan.tasks)
        if cycles:
            issues.append(f"Circular dependencies detected: {cycles}")

        # Check for invalid dependencies
        valid_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            for dep_id in task.dependencies:
                if dep_id not in valid_ids:
                    issues.append(f"Task {task.id} has invalid dependency {dep_id}")

        # Check for empty task names
        for task in plan.tasks:
            if not task.name or not task.name.strip():
                issues.append(f"Task {task.id} has empty name")

        # Check for unreachable tasks
        reachable = self._find_reachable_tasks(plan)
        for task in plan.tasks:
            if task.id not in reachable:
                warnings.append(f"Task {task.id} may be unreachable")

        # Check constraints
        constraint_issues = self._check_constraints(plan)
        issues.extend(constraint_issues)

        # Calculate score
        total_checks = len(plan.tasks) + len(plan.constraints) + 3
        failed_checks = len(issues)
        score = max(0.0, 1.0 - (failed_checks / total_checks))

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            score=score,
        )

    def _find_reachable_tasks(self, plan: Plan) -> Set[str]:
        """Find all tasks reachable from entry points."""
        entry_points = [t for t in plan.tasks if not t.dependencies]
        reachable: Set[str] = set()
        
        def dfs(task_id: str) -> None:
            if task_id in reachable:
                return
            reachable.add(task_id)
            for task in plan.tasks:
                if task_id in task.dependencies:
                    dfs(task.id)
        
        for entry in entry_points:
            dfs(entry.id)
        
        return reachable

    def _check_constraints(self, plan: Plan) -> List[str]:
        """Check if plan satisfies constraints."""
        issues = []
        for constraint in plan.constraints:
            if constraint.is_hard:
                if constraint.constraint_type == ConstraintType.TIME:
                    max_time = constraint.parameters.get("max_duration")
                    if max_time:
                        total_est = sum(
                            t.estimated_duration or 0 for t in plan.tasks
                        )
                        if total_est > max_time:
                            issues.append(
                                f"Constraint '{constraint.name}' violated: "
                                f"estimated {total_est}s > max {max_time}s"
                            )
        return issues


# =============================================================================
# Factory Functions
# =============================================================================


def create_planner(
    strategy: Union[str, PlanningStrategy] = PlanningStrategy.HIERARCHICAL,
    llm: Optional[LLMInterface] = None,
    **kwargs: Any,
) -> PlanGenerator:
    """Factory function to create a plan generator.

    Args:
        strategy: Planning strategy to use.
        llm: LLM interface for AI-powered planning.
        **kwargs: Strategy-specific parameters.

    Returns:
        Configured PlanGenerator instance.

    Example:
        >>> planner = create_planner("forward")
        >>> plan = planner.generate("Build a web app")
    """
    if isinstance(strategy, str):
        strategy = PlanningStrategy(strategy)

    planners: Dict[PlanningStrategy, type] = {
        PlanningStrategy.FORWARD: ForwardPlanner,
        PlanningStrategy.BACKWARD: BackwardPlanner,
        PlanningStrategy.HIERARCHICAL: HierarchicalPlanner,
    }

    if strategy not in planners:
        available = ", ".join(s.value for s in PlanningStrategy)
        raise ValueError(f"Unknown strategy: {strategy}. Available: {available}")

    return planners[strategy](llm=llm, **kwargs)


def create_plan(
    goal: str,
    tasks: Optional[List[Task]] = None,
    constraints: Optional[List[Constraint]] = None,
    **kwargs: Any,
) -> Plan:
    """Factory function to create a plan.

    Args:
        goal: Plan goal.
        tasks: Initial tasks.
        constraints: Plan constraints.
        **kwargs: Additional plan attributes.

    Returns:
        Configured Plan instance.
    """
    plan = Plan(goal=goal, **kwargs)
    if tasks:
        for task in tasks:
            plan.add_task(task)
    if constraints:
        for constraint in constraints:
            plan.add_constraint(constraint)
    return plan


def format_plan(plan: Plan, verbose: bool = False) -> str:
    """Format a plan for display.

    Args:
        plan: Plan to format.
        verbose: Include detailed information.

    Returns:
        Formatted string representation.
    """
    lines = [
        f"Plan: {plan.goal}",
        f"Status: {plan.status.value} | Progress: {plan.progress:.0%}",
        f"Tasks ({len(plan.tasks)}):",
    ]

    try:
        ordered_tasks = plan.get_execution_order()
    except ValueError:
        ordered_tasks = plan.tasks

    for i, task in enumerate(ordered_tasks, 1):
        status_icon = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.IN_PROGRESS: "[~]",
            TaskStatus.COMPLETED: "[x]",
            TaskStatus.FAILED: "[!]",
            TaskStatus.BLOCKED: "[#]",
            TaskStatus.CANCELLED: "[-]",
        }.get(task.status, "[?]")
        
        deps_str = ""
        if task.dependencies and verbose:
            deps_str = f" (deps: {', '.join(task.dependencies[:2])})"
        
        lines.append(f"  {i}. {status_icon} {task.name}{deps_str}")
        
        if verbose and task.description:
            lines.append(f"      {task.description[:60]}...")

    if plan.constraints:
        lines.append(f"Constraints ({len(plan.constraints)}):")
        for c in plan.constraints:
            lines.append(f"  - {c.name}: {c.description}")

    return "\n".join(lines)
