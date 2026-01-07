"""
Task Decomposition: Breaking Complex Tasks into Manageable Subtasks.

Core Idea:
    Task decomposition transforms complex, high-level goals into hierarchical
    structures of smaller, actionable subtasks. This enables systematic problem
    solving by dividing and conquering complex objectives.

Mathematical Foundation:
    Task decomposition can be modeled as a tree transformation:

    $$T_{complex} \\rightarrow T_{tree} = (V, E)$$

    where:
    - $V = \\{t_1, t_2, ..., t_n\\}$ is the set of task nodes
    - $E \\subseteq V \\times V$ represents parent-child relationships
    - Root node $t_1$ is the original complex task
    - Leaf nodes are atomic, executable tasks

    Decomposition satisfies:
    $$\\forall t \\in V: \\text{is\\_leaf}(t) \\Rightarrow \\text{is\\_atomic}(t)$$
    $$\\text{complete}(t_{root}) \\Leftrightarrow \\forall t \\in \\text{leaves}(T): \\text{complete}(t)$$

Problem Statement:
    Complex tasks often cannot be executed directly because:
    1. They require multiple distinct actions
    2. They have implicit dependencies between steps
    3. They exceed the capability of a single operation

    Task decomposition addresses these by:
    - Breaking tasks into smaller, manageable units
    - Identifying dependencies between subtasks
    - Creating a structured execution plan

Algorithm Comparison:
    | Strategy      | Approach           | Best For                    |
    |---------------|--------------------|-----------------------------|
    | Hierarchical  | Top-down recursive | Complex multi-step tasks    |
    | Sequential    | Linear breakdown   | Ordered procedures          |
    | Parallel      | Independent splits | Parallelizable work         |
    | Hybrid        | Mixed strategies   | Real-world complex tasks    |

References:
    - HTN Planning: Hierarchical Task Network Planning
    - STRIPS: Stanford Research Institute Problem Solver
    - Sacerdoti (1974): Planning in a Hierarchy of Abstraction Spaces
"""

from __future__ import annotations

import hashlib
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterator,
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

__all__ = [
    "TaskStatus",
    "TaskPriority",
    "TaskType",
    "Task",
    "DecompositionStrategy",
    "HierarchicalDecomposer",
    "SequentialDecomposer",
    "DependencyAnalyzer",
    "TaskDecomposer",
    "create_task",
    "create_decomposer",
]


# =============================================================================
# Constants and Type Definitions
# =============================================================================

T = TypeVar("T")
DEFAULT_MAX_DEPTH: Final[int] = 5
DEFAULT_MAX_SUBTASKS: Final[int] = 7
MIN_ATOMIC_LENGTH: Final[int] = 50
MAX_TASK_NAME_LENGTH: Final[int] = 200


class TaskStatus(str, Enum):
    """Status of a task in its lifecycle.

    State Transition Diagram:
        PENDING -> IN_PROGRESS -> COMPLETED
                              \\-> FAILED -> PENDING (retry)
        PENDING -> BLOCKED -> PENDING (unblocked)
        Any -> CANCELLED
    """
    PENDING: Final[str] = "pending"
    IN_PROGRESS: Final[str] = "in_progress"
    COMPLETED: Final[str] = "completed"
    FAILED: Final[str] = "failed"
    BLOCKED: Final[str] = "blocked"
    CANCELLED: Final[str] = "cancelled"

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)

    def is_active(self) -> bool:
        """Check if task is actively being worked on."""
        return self == TaskStatus.IN_PROGRESS


class TaskPriority(str, Enum):
    """Priority level for task scheduling."""
    LOW: Final[str] = "low"
    MEDIUM: Final[str] = "medium"
    HIGH: Final[str] = "high"
    CRITICAL: Final[str] = "critical"

    def __lt__(self, other: "TaskPriority") -> bool:
        order = [TaskPriority.LOW, TaskPriority.MEDIUM, 
                 TaskPriority.HIGH, TaskPriority.CRITICAL]
        return order.index(self) < order.index(other)

    def to_numeric(self) -> int:
        """Convert to numeric value for sorting."""
        return {
            TaskPriority.LOW: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.HIGH: 3,
            TaskPriority.CRITICAL: 4,
        }[self]


class TaskType(str, Enum):
    """Type classification for tasks."""
    GENERIC: Final[str] = "generic"
    ANALYSIS: Final[str] = "analysis"
    IMPLEMENTATION: Final[str] = "implementation"
    TESTING: Final[str] = "testing"
    DOCUMENTATION: Final[str] = "documentation"
    RESEARCH: Final[str] = "research"
    COMMUNICATION: Final[str] = "communication"
    DECISION: Final[str] = "decision"


# =============================================================================
# Task Data Structure
# =============================================================================


@dataclass
class Task:
    """Atomic unit of work in a planning system.

    Core Idea:
        A Task represents a unit of work that can be decomposed, scheduled,
        and executed. Tasks form a tree structure where complex tasks contain
        subtasks, and leaf tasks are atomic operations.

    Mathematical Model:
        Task completion follows:
        $$\\text{complete}(t) = \\begin{cases}
            \\text{execute}(t) & \\text{if } t \\text{ is leaf} \\\\
            \\bigwedge_{c \\in \\text{children}(t)} \\text{complete}(c) & \\text{otherwise}
        \\end{cases}$$

    Attributes:
        name: Short descriptive name for the task.
        description: Detailed description of what needs to be done.
        id: Unique identifier (auto-generated if not provided).
        status: Current execution status.
        priority: Scheduling priority.
        task_type: Classification of the task.
        dependencies: IDs of tasks that must complete before this one.
        subtasks: Child tasks (empty for leaf/atomic tasks).
        parent_id: ID of parent task (None for root tasks).
        estimated_duration: Estimated time to complete (seconds).
        actual_duration: Actual time taken (seconds).
        result: Output/result after completion.
        error: Error message if failed.
        metadata: Extensible key-value store.

    Example:
        >>> task = Task(
        ...     name="Build REST API",
        ...     description="Create a RESTful API with CRUD operations",
        ...     priority=TaskPriority.HIGH,
        ... )
        >>> subtask = Task(name="Design endpoints", description="Define API routes")
        >>> task.add_subtask(subtask)
        >>> print(task.depth)
        1
    """
    name: str
    description: str = ""
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    task_type: TaskType = TaskType.GENERIC
    dependencies: List[str] = field(default_factory=list)
    subtasks: List["Task"] = field(default_factory=list)
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize task fields."""
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)
        if isinstance(self.priority, str):
            self.priority = TaskPriority(self.priority)
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)
        if len(self.name) > MAX_TASK_NAME_LENGTH:
            self.name = self.name[:MAX_TASK_NAME_LENGTH]
            warnings.warn(f"Task name truncated to {MAX_TASK_NAME_LENGTH} chars")

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf task (no subtasks)."""
        return len(self.subtasks) == 0

    @property
    def is_root(self) -> bool:
        """Check if this is a root task (no parent)."""
        return self.parent_id is None

    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (no pending dependencies)."""
        return (
            self.status == TaskStatus.PENDING
            and len(self.dependencies) == 0
        )

    @property
    def is_blocked(self) -> bool:
        """Check if task is blocked by dependencies."""
        return (
            self.status == TaskStatus.PENDING
            and len(self.dependencies) > 0
        )

    @property
    def depth(self) -> int:
        """Calculate the depth of the subtask tree."""
        if self.is_leaf:
            return 0
        return 1 + max(child.depth for child in self.subtasks)

    @property
    def total_subtasks(self) -> int:
        """Count total number of subtasks recursively."""
        count = len(self.subtasks)
        for subtask in self.subtasks:
            count += subtask.total_subtasks
        return count

    @property
    def completion_ratio(self) -> float:
        """Calculate completion ratio for this task and subtasks."""
        if self.is_leaf:
            return 1.0 if self.status == TaskStatus.COMPLETED else 0.0
        if not self.subtasks:
            return 0.0
        completed = sum(s.completion_ratio for s in self.subtasks)
        return completed / len(self.subtasks)

    def add_subtask(self, subtask: "Task") -> "Task":
        """Add a subtask to this task."""
        subtask.parent_id = self.id
        self.subtasks.append(subtask)
        return subtask

    def add_dependency(self, task_id: str) -> None:
        """Add a dependency on another task."""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)

    def remove_dependency(self, task_id: str) -> bool:
        """Remove a dependency. Returns True if removed."""
        if task_id in self.dependencies:
            self.dependencies.remove(task_id)
            return True
        return False

    def get_all_subtasks(self) -> List["Task"]:
        """Get all subtasks recursively (flattened)."""
        result = []
        for subtask in self.subtasks:
            result.append(subtask)
            result.extend(subtask.get_all_subtasks())
        return result

    def get_leaf_tasks(self) -> List["Task"]:
        """Get all leaf tasks (atomic tasks)."""
        if self.is_leaf:
            return [self]
        leaves = []
        for subtask in self.subtasks:
            leaves.extend(subtask.get_leaf_tasks())
        return leaves

    def find_subtask(self, task_id: str) -> Optional["Task"]:
        """Find a subtask by ID recursively."""
        for subtask in self.subtasks:
            if subtask.id == task_id:
                return subtask
            found = subtask.find_subtask(task_id)
            if found:
                return found
        return None

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def complete(self, result: Any = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.actual_duration = delta.total_seconds()

    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()

    def reset(self) -> None:
        """Reset task to pending state."""
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.actual_duration = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "task_type": self.task_type.value,
            "dependencies": self.dependencies.copy(),
            "subtasks": [s.to_dict() for s in self.subtasks],
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserialize task from dictionary."""
        subtasks_data = data.pop("subtasks", [])
        
        # Handle datetime fields
        for dt_field in ["created_at", "started_at", "completed_at"]:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])
        
        task = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        
        for subtask_data in subtasks_data:
            subtask = cls.from_dict(subtask_data)
            task.subtasks.append(subtask)
        
        return task

    def __str__(self) -> str:
        status_icon = {
            TaskStatus.PENDING: "○",
            TaskStatus.IN_PROGRESS: "◐",
            TaskStatus.COMPLETED: "●",
            TaskStatus.FAILED: "✗",
            TaskStatus.BLOCKED: "◌",
            TaskStatus.CANCELLED: "⊘",
        }.get(self.status, "?")
        return f"{status_icon} [{self.priority.value[0].upper()}] {self.name}"

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id!r}, name={self.name!r}, "
            f"status={self.status.value}, subtasks={len(self.subtasks)})"
        )


# =============================================================================
# LLM Interface Protocol
# =============================================================================


@runtime_checkable
class LLMInterface(Protocol):
    """Protocol for LLM interaction in decomposition."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


# =============================================================================
# Decomposition Strategies
# =============================================================================


class DecompositionStrategy(ABC):
    """Abstract base class for task decomposition strategies.

    Core Idea:
        Different decomposition strategies suit different task types.
        This abstract class defines the interface for all strategies.
    """

    @abstractmethod
    def decompose(
        self,
        task: Task,
        context: str = "",
        **kwargs: Any,
    ) -> List[Task]:
        """Decompose a task into subtasks.

        Args:
            task: The task to decompose.
            context: Additional context for decomposition.
            **kwargs: Strategy-specific parameters.

        Returns:
            List of subtasks.
        """
        pass

    @abstractmethod
    def is_atomic(self, task: Task) -> bool:
        """Check if a task is atomic (cannot be further decomposed).

        Args:
            task: The task to check.

        Returns:
            True if task is atomic.
        """
        pass


class HierarchicalDecomposer(DecompositionStrategy):
    """Hierarchical task decomposition using LLM.

    Core Idea:
        Recursively decomposes complex tasks into subtasks using LLM,
        creating a tree structure. Stops when tasks become atomic or
        maximum depth is reached.

    Algorithm:
        1. Check if task is atomic -> return as-is
        2. Generate subtasks using LLM prompt
        3. Parse LLM response into Task objects
        4. Recursively decompose each subtask
        5. Return decomposed task tree

    Complexity:
        - Time: O(b^d) where b=branching factor, d=depth
        - Space: O(b^d) for the task tree
    """

    DECOMPOSE_PROMPT: Final[str] = """You are a task decomposition expert. Break down the following task into smaller, actionable subtasks.

Task: {task_name}
Description: {task_description}
Context: {context}

Requirements:
1. Create 2-5 subtasks that together accomplish the main task
2. Each subtask should be specific and actionable
3. Subtasks should be as independent as possible
4. Order subtasks logically (dependencies should come first)

Format your response as a numbered list:
1. [Subtask Name]: [Brief description of what needs to be done]
2. [Subtask Name]: [Brief description of what needs to be done]
...

Subtasks:"""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_subtasks: int = DEFAULT_MAX_SUBTASKS,
        atomic_checker: Optional[Callable[[Task], bool]] = None,
    ) -> None:
        """Initialize hierarchical decomposer.

        Args:
            llm: LLM interface for generating decompositions.
            max_depth: Maximum decomposition depth.
            max_subtasks: Maximum subtasks per decomposition.
            atomic_checker: Custom function to check if task is atomic.
        """
        self._llm = llm
        self._max_depth = max_depth
        self._max_subtasks = max_subtasks
        self._atomic_checker = atomic_checker

    def is_atomic(self, task: Task) -> bool:
        """Check if task is atomic."""
        if self._atomic_checker:
            return self._atomic_checker(task)
        # Default: task is atomic if description is short
        return len(task.description) < MIN_ATOMIC_LENGTH

    def decompose(
        self,
        task: Task,
        context: str = "",
        **kwargs: Any,
    ) -> List[Task]:
        """Decompose task into subtasks."""
        if self.is_atomic(task):
            return []

        if self._llm is None:
            return self._generate_placeholder_subtasks(task)

        prompt = self.DECOMPOSE_PROMPT.format(
            task_name=task.name,
            task_description=task.description,
            context=context or "No additional context provided.",
        )

        response = self._llm.generate(prompt)
        subtasks = self._parse_subtasks(response, task.id)

        return subtasks[:self._max_subtasks]

    def decompose_recursive(
        self,
        task: Task,
        context: str = "",
        current_depth: int = 0,
    ) -> Task:
        """Recursively decompose task to specified depth.

        Args:
            task: Task to decompose.
            context: Additional context.
            current_depth: Current recursion depth.

        Returns:
            Task with populated subtasks tree.
        """
        if current_depth >= self._max_depth or self.is_atomic(task):
            return task

        subtasks = self.decompose(task, context)

        for subtask in subtasks:
            decomposed = self.decompose_recursive(
                subtask, context, current_depth + 1
            )
            task.add_subtask(decomposed)

        return task

    def _generate_placeholder_subtasks(self, task: Task) -> List[Task]:
        """Generate placeholder subtasks when no LLM available."""
        return [
            Task(
                name=f"{task.name} - Analysis",
                description=f"Analyze requirements for: {task.description}",
                task_type=TaskType.ANALYSIS,
            ),
            Task(
                name=f"{task.name} - Implementation",
                description=f"Implement: {task.description}",
                task_type=TaskType.IMPLEMENTATION,
            ),
            Task(
                name=f"{task.name} - Verification",
                description=f"Verify completion of: {task.description}",
                task_type=TaskType.TESTING,
            ),
        ]

    def _parse_subtasks(self, response: str, parent_id: str) -> List[Task]:
        """Parse LLM response into subtask list."""
        subtasks = []

        # Pattern: "1. [Name]: [Description]" or "1. Name: Description"
        pattern = r'(\d+)\.\s*\[?([^\]:\n]+)\]?:\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for _, name, description in matches:
            name = name.strip()
            description = description.strip()
            if name and description:
                subtasks.append(Task(
                    name=name,
                    description=description,
                    parent_id=parent_id,
                ))

        # Fallback: try simpler pattern
        if not subtasks:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit():
                    # Remove leading number and punctuation
                    content = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if ':' in content:
                        name, desc = content.split(':', 1)
                        subtasks.append(Task(
                            name=name.strip(),
                            description=desc.strip(),
                            parent_id=parent_id,
                        ))
                    elif content:
                        subtasks.append(Task(
                            name=content[:50],
                            description=content,
                            parent_id=parent_id,
                        ))

        return subtasks


class SequentialDecomposer(DecompositionStrategy):
    """Sequential task decomposition for ordered procedures.

    Core Idea:
        Decomposes tasks into a linear sequence of steps, where each step
        depends on the previous one. Best for procedural tasks with clear
        ordering requirements.

    Example:
        "Deploy application" -> [Build, Test, Package, Deploy, Verify]
    """

    SEQUENTIAL_PROMPT: Final[str] = """Break down this task into sequential steps that must be performed in order.

Task: {task_name}
Description: {task_description}
Context: {context}

List the steps in the exact order they should be executed:
1. [Step Name]: [What to do]
2. [Step Name]: [What to do]
...

Steps:"""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_steps: int = 10,
    ) -> None:
        self._llm = llm
        self._max_steps = max_steps

    def is_atomic(self, task: Task) -> bool:
        """Sequential tasks are atomic if description is short."""
        return len(task.description) < MIN_ATOMIC_LENGTH

    def decompose(
        self,
        task: Task,
        context: str = "",
        **kwargs: Any,
    ) -> List[Task]:
        """Decompose into sequential steps with dependencies."""
        if self.is_atomic(task):
            return []

        if self._llm is None:
            steps = self._generate_placeholder_steps(task)
        else:
            prompt = self.SEQUENTIAL_PROMPT.format(
                task_name=task.name,
                task_description=task.description,
                context=context or "No additional context.",
            )
            response = self._llm.generate(prompt)
            steps = self._parse_steps(response, task.id)

        # Add sequential dependencies
        for i in range(1, len(steps)):
            steps[i].add_dependency(steps[i - 1].id)

        return steps[:self._max_steps]

    def _generate_placeholder_steps(self, task: Task) -> List[Task]:
        """Generate placeholder sequential steps."""
        return [
            Task(name="Prepare", description=f"Prepare for: {task.name}"),
            Task(name="Execute", description=f"Execute: {task.name}"),
            Task(name="Finalize", description=f"Finalize: {task.name}"),
        ]

    def _parse_steps(self, response: str, parent_id: str) -> List[Task]:
        """Parse sequential steps from response."""
        steps = []
        pattern = r'(\d+)\.\s*\[?([^\]:\n]+)\]?:\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, response, re.DOTALL)

        for _, name, description in matches:
            steps.append(Task(
                name=name.strip(),
                description=description.strip(),
                parent_id=parent_id,
            ))

        if not steps:
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            for line in lines:
                if line and line[0].isdigit():
                    content = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if content:
                        steps.append(Task(
                            name=content[:50],
                            description=content,
                            parent_id=parent_id,
                        ))

        return steps

# =============================================================================
# Dependency Analysis
# =============================================================================


@dataclass
class DependencyEdge:
    """Represents a dependency relationship between tasks."""
    from_task_id: str
    to_task_id: str
    dependency_type: str = "requires"  # requires, blocks, suggests
    strength: float = 1.0  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_task_id,
            "to": self.to_task_id,
            "type": self.dependency_type,
            "strength": self.strength,
        }


class DependencyAnalyzer:
    """Analyzes and manages dependencies between tasks.

    Core Idea:
        Identifies explicit and implicit dependencies between tasks,
        constructs a dependency graph (DAG), and provides utilities
        for topological sorting and cycle detection.

    Mathematical Model:
        Dependency graph G = (V, E) where:
        - V = set of tasks
        - E = {(u, v) | task v depends on task u}

        Valid plan requires: G is a DAG (no cycles)
        Execution order: topological_sort(G)
    """

    DEPENDENCY_PROMPT: Final[str] = '''Analyze dependencies between these tasks.

Tasks:
{tasks}

For each task, identify which other tasks it depends on (must complete first).
Consider:
1. Data dependencies (needs output from another task)
2. Resource dependencies (needs same resource)
3. Logical ordering (must happen after)

Format (one per line, only if dependency exists):
[task_id] depends on [task_id]: [reason]

Dependencies:'''

    def __init__(self, llm: Optional[LLMInterface] = None) -> None:
        self._llm = llm

    def analyze(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """Analyze dependencies between tasks."""
        if len(tasks) <= 1:
            return {t.id: [] for t in tasks}

        if self._llm is None:
            return self._infer_sequential_dependencies(tasks)

        tasks_str = "\n".join(
            f"- {t.id}: {t.name} - {t.description[:100]}"
            for t in tasks
        )
        prompt = self.DEPENDENCY_PROMPT.format(tasks=tasks_str)
        response = self._llm.generate(prompt)

        return self._parse_dependencies(response, tasks)

    def _infer_sequential_dependencies(
        self, tasks: List[Task]
    ) -> Dict[str, List[str]]:
        """Infer sequential dependencies based on task order."""
        deps: Dict[str, List[str]] = {}
        for i, task in enumerate(tasks):
            deps[task.id] = [tasks[i - 1].id] if i > 0 else []
        return deps

    def _parse_dependencies(
        self,
        response: str,
        tasks: List[Task],
    ) -> Dict[str, List[str]]:
        """Parse dependency relationships from LLM response."""
        deps: Dict[str, List[str]] = {t.id: [] for t in tasks}
        valid_ids = {t.id for t in tasks}

        pattern = r'(\w+)\s+depends on\s+(\w+)'
        for match in re.finditer(pattern, response, re.IGNORECASE):
            task_id, dep_id = match.groups()
            if task_id in valid_ids and dep_id in valid_ids:
                if dep_id not in deps[task_id]:
                    deps[task_id].append(dep_id)

        return deps

    def apply_dependencies(
        self,
        tasks: List[Task],
        dependencies: Dict[str, List[str]],
    ) -> None:
        """Apply dependency relationships to tasks."""
        for task in tasks:
            task.dependencies = dependencies.get(task.id, [])

    def detect_cycles(self, tasks: List[Task]) -> List[List[str]]:
        """Detect cycles in task dependencies."""
        task_map = {t.id: t for t in tasks}
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(task_id: str, path: List[str]) -> None:
            visited.add(task_id)
            rec_stack.add(task_id)
            path.append(task_id)

            task = task_map.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        dfs(dep_id, path.copy())
                    elif dep_id in rec_stack:
                        cycle_start = path.index(dep_id)
                        cycles.append(path[cycle_start:] + [dep_id])

            rec_stack.remove(task_id)

        for task in tasks:
            if task.id not in visited:
                dfs(task.id, [])

        return cycles

    def topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks in dependency order (Kahn's algorithm)."""
        task_map = {t.id: t for t in tasks}
        in_degree: Dict[str, int] = {t.id: 0 for t in tasks}

        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in in_degree:
                    in_degree[task.id] += 1

        queue = [t for t in tasks if in_degree[t.id] == 0]
        result: List[Task] = []

        while queue:
            queue.sort(key=lambda t: t.priority.to_numeric(), reverse=True)
            task = queue.pop(0)
            result.append(task)

            for other in tasks:
                if task.id in other.dependencies:
                    in_degree[other.id] -= 1
                    if in_degree[other.id] == 0:
                        queue.append(other)

        if len(result) != len(tasks):
            raise ValueError("Circular dependency detected in task graph")

        return result

    def get_ready_tasks(
        self,
        tasks: List[Task],
        completed_ids: Set[str],
    ) -> List[Task]:
        """Get tasks that are ready to execute."""
        ready = []
        for task in tasks:
            if task.status == TaskStatus.PENDING:
                deps_satisfied = all(
                    d in completed_ids for d in task.dependencies
                )
                if deps_satisfied:
                    ready.append(task)
        return ready


# =============================================================================
# Main Task Decomposer
# =============================================================================


class TaskDecomposer:
    """Main interface for task decomposition.

    Core Idea:
        Provides a unified interface for decomposing tasks using various
        strategies, with automatic dependency analysis and validation.

    Example:
        >>> decomposer = TaskDecomposer()
        >>> task = Task(
        ...     name="Build web application",
        ...     description="Create a full-stack web app with React and FastAPI"
        ... )
        >>> subtasks = decomposer.decompose(task)
    """

    def __init__(
        self,
        strategy: Optional[DecompositionStrategy] = None,
        dependency_analyzer: Optional[DependencyAnalyzer] = None,
        llm: Optional[LLMInterface] = None,
    ) -> None:
        self._llm = llm
        self._strategy = strategy or HierarchicalDecomposer(llm)
        self._analyzer = dependency_analyzer or DependencyAnalyzer(llm)

    @property
    def strategy(self) -> DecompositionStrategy:
        """Get current decomposition strategy."""
        return self._strategy

    def set_strategy(self, strategy: DecompositionStrategy) -> None:
        """Set decomposition strategy."""
        self._strategy = strategy

    def decompose(
        self,
        task: Task,
        context: str = "",
        analyze_dependencies: bool = True,
        recursive: bool = False,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> List[Task]:
        """Decompose a task into subtasks."""
        if recursive and isinstance(self._strategy, HierarchicalDecomposer):
            self._strategy.decompose_recursive(task, context, 0)
            return task.subtasks

        subtasks = self._strategy.decompose(task, context)

        if analyze_dependencies and len(subtasks) > 1:
            deps = self._analyzer.analyze(subtasks)
            self._analyzer.apply_dependencies(subtasks, deps)

        return subtasks

    def decompose_and_attach(
        self,
        task: Task,
        context: str = "",
        analyze_dependencies: bool = True,
    ) -> Task:
        """Decompose task and attach subtasks to it."""
        subtasks = self.decompose(task, context, analyze_dependencies)
        for subtask in subtasks:
            task.add_subtask(subtask)
        return task

    def validate_decomposition(self, task: Task) -> Tuple[bool, List[str]]:
        """Validate a task decomposition."""
        issues: List[str] = []

        if not task.subtasks:
            issues.append("Task has no subtasks")
            return False, issues

        all_tasks = task.get_all_subtasks()
        cycles = self._analyzer.detect_cycles(all_tasks)
        if cycles:
            issues.append(f"Circular dependencies detected: {cycles}")

        all_ids = {t.id for t in all_tasks}
        for subtask in all_tasks:
            for dep_id in subtask.dependencies:
                if dep_id not in all_ids:
                    issues.append(
                        f"Task {subtask.id} has invalid dependency {dep_id}"
                    )

        for subtask in all_tasks:
            if not subtask.name:
                issues.append(f"Task {subtask.id} has empty name")

        return len(issues) == 0, issues


# =============================================================================
# Factory Functions
# =============================================================================


def create_task(
    name: str,
    description: str = "",
    priority: Union[str, TaskPriority] = TaskPriority.MEDIUM,
    task_type: Union[str, TaskType] = TaskType.GENERIC,
    **kwargs: Any,
) -> Task:
    """Factory function to create a task."""
    if isinstance(priority, str):
        priority = TaskPriority(priority)
    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    return Task(
        name=name,
        description=description,
        priority=priority,
        task_type=task_type,
        **kwargs,
    )


def create_decomposer(
    strategy: str = "hierarchical",
    llm: Optional[LLMInterface] = None,
    **kwargs: Any,
) -> TaskDecomposer:
    """Factory function to create a task decomposer."""
    strategies: Dict[str, type] = {
        "hierarchical": HierarchicalDecomposer,
        "sequential": SequentialDecomposer,
    }

    if strategy not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown strategy: {strategy}. Available: {available}")

    strategy_instance = strategies[strategy](llm=llm, **kwargs)
    return TaskDecomposer(strategy=strategy_instance, llm=llm)


def format_task_tree(task: Task, indent: int = 0) -> str:
    """Format a task tree for display."""
    prefix = "  " * indent
    lines = [f"{prefix}{task}"]

    for subtask in task.subtasks:
        lines.append(format_task_tree(subtask, indent + 1))

    return "\n".join(lines)
