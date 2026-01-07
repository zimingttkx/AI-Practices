"""
Plan Execution: Running Plans with Monitoring and Control.

Core Idea:
    Plan execution manages the runtime lifecycle of plans, handling task
    scheduling, execution, monitoring, and error recovery. It bridges the
    gap between abstract plans and concrete actions.

Mathematical Foundation:
    Execution can be modeled as a state machine:

    $$M = (Q, \\Sigma, \\delta, q_0, F)$$

    where:
    - $Q = \\{pending, running, completed, failed, blocked\\}$: States
    - $\\Sigma = \\{start, complete, fail, retry, cancel\\}$: Events
    - $\\delta: Q \\times \\Sigma \\rightarrow Q$: Transition function
    - $q_0 = pending$: Initial state
    - $F = \\{completed, cancelled\\}$: Final states

    State transitions:
    - $\\delta(pending, start) = running$
    - $\\delta(running, complete) = completed$
    - $\\delta(running, fail) = failed$
    - $\\delta(failed, retry) = pending$

Problem Statement:
    Given a validated plan, execute it by:
    1. Scheduling tasks based on dependencies
    2. Executing tasks (possibly in parallel)
    3. Handling failures and retries
    4. Monitoring progress and reporting status

References:
    - Workflow Engines: Apache Airflow, Prefect
    - State Machines: Harel Statecharts
    - Execution Patterns: Saga Pattern, Compensating Transactions
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
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
)

try:
    from .task_decomposition import Task, TaskStatus, TaskPriority, LLMInterface
    from .plan_generation import Plan, PlanStatus
except ImportError:
    from task_decomposition import Task, TaskStatus, TaskPriority, LLMInterface
    from plan_generation import Plan, PlanStatus

__all__ = [
    "ExecutionResult",
    "ExecutionStatus",
    "ExecutionContext",
    "ExecutionPolicy",
    "TaskExecutor",
    "SimpleTaskExecutor",
    "LLMTaskExecutor",
    "PlanExecutor",
    "ExecutionMonitor",
    "ExecutionCallback",
]


# =============================================================================
# Constants and Enums
# =============================================================================

DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[float] = 1.0
DEFAULT_TIMEOUT: Final[float] = 300.0
DEFAULT_MAX_WORKERS: Final[int] = 4


class ExecutionStatus(str, Enum):
    """Status of an execution operation."""
    SUCCESS: Final[str] = "success"
    FAILURE: Final[str] = "failure"
    TIMEOUT: Final[str] = "timeout"
    CANCELLED: Final[str] = "cancelled"
    SKIPPED: Final[str] = "skipped"


# =============================================================================
# Execution Result
# =============================================================================


@dataclass
class ExecutionResult:
    """Result of executing a single task.

    Attributes:
        task_id: ID of the executed task.
        status: Execution status.
        result: Output/return value if successful.
        error: Error message if failed.
        duration: Execution time in seconds.
        retries: Number of retry attempts.
        metadata: Additional execution information.
    """
    task_id: str
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    retries: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.status in (ExecutionStatus.FAILURE, ExecutionStatus.TIMEOUT)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "duration": self.duration,
            "retries": self.retries,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


# =============================================================================
# Execution Context
# =============================================================================


@dataclass
class ExecutionContext:
    """Shared context for task execution.

    Core Idea:
        Provides a shared state container that tasks can read from and
        write to, enabling data flow between dependent tasks.

    Attributes:
        variables: Shared variables accessible to all tasks.
        results: Execution results indexed by task ID.
        plan: The plan being executed.
        metadata: Additional context information.
    """
    variables: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, ExecutionResult] = field(default_factory=dict)
    plan: Optional[Plan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable from context."""
        return self.variables.get(name, default)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in context."""
        self.variables[name] = value

    def get_result(self, task_id: str) -> Optional[ExecutionResult]:
        """Get execution result for a task."""
        return self.results.get(task_id)

    def get_task_output(self, task_id: str) -> Any:
        """Get the output of a completed task."""
        result = self.results.get(task_id)
        return result.result if result and result.is_success else None

    def add_result(self, result: ExecutionResult) -> None:
        """Add an execution result."""
        self.results[result.task_id] = result

    def get_completed_task_ids(self) -> Set[str]:
        """Get IDs of successfully completed tasks."""
        return {
            task_id for task_id, result in self.results.items()
            if result.is_success
        }


# =============================================================================
# Execution Policy
# =============================================================================


@dataclass
class ExecutionPolicy:
    """Configuration for plan execution behavior.

    Attributes:
        max_retries: Maximum retry attempts per task.
        retry_delay: Delay between retries in seconds.
        timeout: Maximum execution time per task.
        parallel: Enable parallel task execution.
        max_workers: Maximum parallel workers.
        stop_on_failure: Stop plan on first task failure.
        continue_on_skip: Continue if task is skipped.
    """
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    timeout: Optional[float] = DEFAULT_TIMEOUT
    parallel: bool = False
    max_workers: int = DEFAULT_MAX_WORKERS
    stop_on_failure: bool = True
    continue_on_skip: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
            "stop_on_failure": self.stop_on_failure,
            "continue_on_skip": self.continue_on_skip,
        }


# =============================================================================
# Execution Callbacks
# =============================================================================


class ExecutionCallback(Protocol):
    """Protocol for execution event callbacks."""

    def on_task_start(self, task: Task, context: ExecutionContext) -> None:
        """Called when a task starts execution."""
        ...

    def on_task_complete(self, task: Task, result: ExecutionResult, context: ExecutionContext) -> None:
        """Called when a task completes."""
        ...

    def on_task_error(self, task: Task, error: Exception, context: ExecutionContext) -> None:
        """Called when a task encounters an error."""
        ...

    def on_plan_start(self, plan: Plan, context: ExecutionContext) -> None:
        """Called when plan execution starts."""
        ...

    def on_plan_complete(self, plan: Plan, context: ExecutionContext) -> None:
        """Called when plan execution completes."""
        ...


class DefaultCallback:
    """Default no-op callback implementation."""

    def on_task_start(self, task: Task, context: ExecutionContext) -> None:
        pass

    def on_task_complete(self, task: Task, result: ExecutionResult, context: ExecutionContext) -> None:
        pass

    def on_task_error(self, task: Task, error: Exception, context: ExecutionContext) -> None:
        pass

    def on_plan_start(self, plan: Plan, context: ExecutionContext) -> None:
        pass

    def on_plan_complete(self, plan: Plan, context: ExecutionContext) -> None:
        pass


# =============================================================================
# Task Executors
# =============================================================================


class TaskExecutor(ABC):
    """Abstract base class for task executors."""

    @abstractmethod
    def execute(self, task: Task, context: ExecutionContext) -> Any:
        """Execute a task and return the result."""
        pass


class SimpleTaskExecutor(TaskExecutor):
    """Simple task executor using registered handlers.

    Core Idea:
        Maps task types to handler functions, allowing flexible
        task execution based on task classification.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[[Task, ExecutionContext], Any]] = {}
        self._default_handler: Optional[Callable[[Task, ExecutionContext], Any]] = None

    def register(
        self,
        task_type: str,
        handler: Callable[[Task, ExecutionContext], Any],
    ) -> "SimpleTaskExecutor":
        """Register a handler for a task type."""
        self._handlers[task_type] = handler
        return self

    def set_default_handler(
        self,
        handler: Callable[[Task, ExecutionContext], Any],
    ) -> "SimpleTaskExecutor":
        """Set the default handler for unregistered task types."""
        self._default_handler = handler
        return self

    def execute(self, task: Task, context: ExecutionContext) -> Any:
        """Execute a task using the appropriate handler."""
        task_type = task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)
        handler = self._handlers.get(task_type, self._default_handler)

        if handler is None:
            return f"Task '{task.name}' completed (no handler)"

        return handler(task, context)


class LLMTaskExecutor(TaskExecutor):
    """Task executor using LLM for intelligent task completion.

    Core Idea:
        Uses an LLM to interpret and execute tasks, generating
        appropriate responses based on task descriptions.
    """

    EXECUTE_PROMPT: Final[str] = '''Execute the following task and provide the result.

Task: {task_name}
Description: {task_description}
Context Variables: {context_vars}
Previous Results: {prev_results}

Provide a clear, actionable result for this task.

Result:'''

    def __init__(self, llm: Optional[LLMInterface] = None) -> None:
        self._llm = llm

    def execute(self, task: Task, context: ExecutionContext) -> Any:
        """Execute task using LLM."""
        if self._llm is None:
            return f"Task '{task.name}' simulated completion"

        context_vars = {k: str(v)[:100] for k, v in context.variables.items()}
        prev_results = {
            tid: r.result for tid, r in context.results.items()
            if r.is_success and r.result
        }

        prompt = self.EXECUTE_PROMPT.format(
            task_name=task.name,
            task_description=task.description,
            context_vars=context_vars or "None",
            prev_results=prev_results or "None",
        )

        return self._llm.generate(prompt)


# =============================================================================
# Execution Monitor
# =============================================================================


class ExecutionMonitor:
    """Monitors plan execution progress and statistics.

    Tracks:
    - Task completion rates
    - Execution times
    - Error rates
    - Resource usage
    """

    def __init__(self) -> None:
        self._start_time: Optional[datetime] = None
        self._task_times: Dict[str, float] = {}
        self._error_count: int = 0
        self._success_count: int = 0

    def start(self) -> None:
        """Start monitoring."""
        self._start_time = datetime.utcnow()

    def record_task(self, task_id: str, duration: float, success: bool) -> None:
        """Record task execution."""
        self._task_times[task_id] = duration
        if success:
            self._success_count += 1
        else:
            self._error_count += 1

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.utcnow() - self._start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Task success rate (0.0 to 1.0)."""
        total = self._success_count + self._error_count
        return self._success_count / total if total > 0 else 0.0

    @property
    def average_task_time(self) -> float:
        """Average task execution time."""
        if not self._task_times:
            return 0.0
        return sum(self._task_times.values()) / len(self._task_times)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "elapsed_time": self.elapsed_time,
            "tasks_completed": self._success_count,
            "tasks_failed": self._error_count,
            "success_rate": self.success_rate,
            "average_task_time": self.average_task_time,
        }


# =============================================================================
# Plan Executor
# =============================================================================


class PlanExecutor:
    """Main executor for running plans.

    Core Idea:
        Orchestrates the execution of a plan by scheduling tasks based on
        dependencies, executing them with the configured executor, and
        handling errors according to the execution policy.

    Algorithm:
        1. Initialize execution context
        2. While there are pending tasks:
           a. Find ready tasks (dependencies satisfied)
           b. Execute ready tasks (sequential or parallel)
           c. Update task states and context
           d. Handle failures according to policy
        3. Return final execution results

    Example:
        >>> executor = PlanExecutor()
        >>> context = executor.execute(plan)
        >>> print(f"Success rate: {context.get_completed_task_ids()}")
    """

    def __init__(
        self,
        task_executor: Optional[TaskExecutor] = None,
        policy: Optional[ExecutionPolicy] = None,
        callback: Optional[ExecutionCallback] = None,
    ) -> None:
        self._task_executor = task_executor or SimpleTaskExecutor()
        self._policy = policy or ExecutionPolicy()
        self._callback = callback or DefaultCallback()
        self._monitor = ExecutionMonitor()

    @property
    def policy(self) -> ExecutionPolicy:
        return self._policy

    @property
    def monitor(self) -> ExecutionMonitor:
        return self._monitor

    def execute(
        self,
        plan: Plan,
        context: Optional[ExecutionContext] = None,
    ) -> ExecutionContext:
        """Execute a plan."""
        context = context or ExecutionContext()
        context.plan = plan
        
        self._monitor.start()
        self._callback.on_plan_start(plan, context)
        plan.start()

        try:
            if self._policy.parallel:
                self._execute_parallel(plan, context)
            else:
                self._execute_sequential(plan, context)
        except Exception as e:
            plan.fail()
            raise

        if plan.is_complete:
            plan.complete()
        elif plan.has_failed_tasks:
            plan.fail()

        self._callback.on_plan_complete(plan, context)
        return context

    def _execute_sequential(self, plan: Plan, context: ExecutionContext) -> None:
        """Execute tasks sequentially."""
        while True:
            ready_tasks = self._get_ready_tasks(plan, context)
            if not ready_tasks:
                break

            for task in ready_tasks:
                result = self._execute_task(task, context)
                context.add_result(result)

                if result.is_failure and self._policy.stop_on_failure:
                    return

    def _execute_parallel(self, plan: Plan, context: ExecutionContext) -> None:
        """Execute tasks in parallel where possible."""
        with ThreadPoolExecutor(max_workers=self._policy.max_workers) as executor:
            while True:
                ready_tasks = self._get_ready_tasks(plan, context)
                if not ready_tasks:
                    break

                futures: Dict[Future, Task] = {}
                for task in ready_tasks:
                    future = executor.submit(self._execute_task, task, context)
                    futures[future] = task

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        context.add_result(result)
                        
                        if result.is_failure and self._policy.stop_on_failure:
                            executor.shutdown(wait=False)
                            return
                    except Exception as e:
                        result = ExecutionResult(
                            task_id=task.id,
                            status=ExecutionStatus.FAILURE,
                            error=str(e),
                        )
                        context.add_result(result)
                        task.fail(str(e))

    def _get_ready_tasks(self, plan: Plan, context: ExecutionContext) -> List[Task]:
        """Get tasks ready for execution."""
        completed_ids = context.get_completed_task_ids()
        ready = []
        for task in plan.tasks:
            if task.status == TaskStatus.PENDING:
                deps_satisfied = all(d in completed_ids for d in task.dependencies)
                if deps_satisfied:
                    ready.append(task)
        return ready

    def _execute_task(self, task: Task, context: ExecutionContext) -> ExecutionResult:
        """Execute a single task with retry logic."""
        self._callback.on_task_start(task, context)
        task.start()

        retries = 0
        last_error: Optional[str] = None
        start_time = datetime.utcnow()

        while retries <= self._policy.max_retries:
            try:
                result_value = self._task_executor.execute(task, context)
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                task.complete(result_value)
                result = ExecutionResult(
                    task_id=task.id,
                    status=ExecutionStatus.SUCCESS,
                    result=result_value,
                    duration=duration,
                    retries=retries,
                    started_at=start_time,
                    completed_at=datetime.utcnow(),
                )
                
                self._monitor.record_task(task.id, duration, True)
                self._callback.on_task_complete(task, result, context)
                return result

            except Exception as e:
                last_error = str(e)
                self._callback.on_task_error(task, e, context)
                retries += 1
                
                if retries <= self._policy.max_retries:
                    time.sleep(self._policy.retry_delay)

        duration = (datetime.utcnow() - start_time).total_seconds()
        task.fail(last_error or "Unknown error")
        
        result = ExecutionResult(
            task_id=task.id,
            status=ExecutionStatus.FAILURE,
            error=last_error,
            duration=duration,
            retries=retries - 1,
            started_at=start_time,
            completed_at=datetime.utcnow(),
        )
        
        self._monitor.record_task(task.id, duration, False)
        return result


# =============================================================================
# Factory Functions
# =============================================================================


def create_executor(
    executor_type: str = "simple",
    llm: Optional[LLMInterface] = None,
    policy: Optional[ExecutionPolicy] = None,
    **kwargs: Any,
) -> PlanExecutor:
    """Factory function to create a plan executor.

    Args:
        executor_type: Type of task executor ("simple" or "llm").
        llm: LLM interface for LLM executor.
        policy: Execution policy.
        **kwargs: Additional executor parameters.

    Returns:
        Configured PlanExecutor instance.
    """
    if executor_type == "llm":
        task_executor = LLMTaskExecutor(llm=llm)
    else:
        task_executor = SimpleTaskExecutor()

    return PlanExecutor(
        task_executor=task_executor,
        policy=policy or ExecutionPolicy(**kwargs),
    )


def execute_plan(
    plan: Plan,
    executor: Optional[PlanExecutor] = None,
    context: Optional[ExecutionContext] = None,
    **policy_kwargs: Any,
) -> ExecutionContext:
    """Convenience function to execute a plan.

    Args:
        plan: Plan to execute.
        executor: Optional executor (creates default if not provided).
        context: Optional execution context.
        **policy_kwargs: Policy configuration.

    Returns:
        Execution context with results.
    """
    if executor is None:
        policy = ExecutionPolicy(**policy_kwargs) if policy_kwargs else None
        executor = PlanExecutor(policy=policy)

    return executor.execute(plan, context)
