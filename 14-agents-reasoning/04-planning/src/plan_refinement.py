"""
Plan Refinement: Dynamic Replanning and Optimization.

Core Idea:
    Plan refinement enables adaptive planning by modifying plans during
    execution based on feedback, failures, or changing conditions. This
    module implements strategies for error recovery, plan optimization,
    and dynamic replanning.

Mathematical Foundation:
    Replanning can be modeled as an optimization problem:

    $$\\pi^* = \\arg\\min_{\\pi \\in \\Pi} C(\\pi) + \\lambda \\cdot D(\\pi, \\pi_{old})$$

    where:
    - $\\pi$: New plan
    - $\\Pi$: Space of valid plans
    - $C(\\pi)$: Cost of plan
    - $D(\\pi, \\pi_{old})$: Distance from original plan
    - $\\lambda$: Stability preference weight

Problem Statement:
    During execution, plans may need modification due to:
    1. Task failures requiring alternative approaches
    2. New information changing requirements
    3. Resource constraints becoming apparent
    4. Opportunities for optimization

References:
    - Replanning: Koenig & Likhachev (2002) "D* Lite"
    - Plan Repair: Fox et al. (2006) "Plan Stability"
    - Adaptive Planning: Brenner & Nebel (2009)
"""

from __future__ import annotations

import copy
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
    from .task_decomposition import Task, TaskStatus, TaskPriority, LLMInterface
    from .plan_generation import Plan, PlanStatus, PlanValidator, ValidationResult
    from .plan_execution import ExecutionContext, ExecutionResult, ExecutionStatus
except ImportError:
    from task_decomposition import Task, TaskStatus, TaskPriority, LLMInterface
    from plan_generation import Plan, PlanStatus, PlanValidator, ValidationResult
    from plan_execution import ExecutionContext, ExecutionResult, ExecutionStatus

__all__ = [
    "RefinementStrategy",
    "RefinementTrigger",
    "RefinementResult",
    "PlanRefinement",
    "AdaptiveReplanner",
    "FailureRecovery",
    "PlanOptimizer",
]


# =============================================================================
# Constants and Enums
# =============================================================================

DEFAULT_MAX_REFINEMENTS: Final[int] = 5


class RefinementTrigger(str, Enum):
    """Triggers for plan refinement."""
    TASK_FAILURE: Final[str] = "task_failure"
    TIMEOUT: Final[str] = "timeout"
    CONSTRAINT_VIOLATION: Final[str] = "constraint_violation"
    NEW_INFORMATION: Final[str] = "new_information"
    OPTIMIZATION: Final[str] = "optimization"
    MANUAL: Final[str] = "manual"


class RefinementStrategy(str, Enum):
    """Available refinement strategies."""
    RETRY: Final[str] = "retry"
    SKIP: Final[str] = "skip"
    REPLACE: Final[str] = "replace"
    DECOMPOSE: Final[str] = "decompose"
    REORDER: Final[str] = "reorder"
    ABORT: Final[str] = "abort"


# =============================================================================
# Refinement Result
# =============================================================================


@dataclass
class RefinementResult:
    """Result of a plan refinement operation.

    Attributes:
        success: Whether refinement was successful.
        strategy_used: Strategy that was applied.
        trigger: What triggered the refinement.
        changes_made: Description of changes.
        original_plan: Plan before refinement.
        refined_plan: Plan after refinement.
        metadata: Additional information.
    """
    success: bool
    strategy_used: RefinementStrategy
    trigger: RefinementTrigger
    changes_made: List[str] = field(default_factory=list)
    original_plan: Optional[Plan] = None
    refined_plan: Optional[Plan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.value,
            "trigger": self.trigger.value,
            "changes_made": self.changes_made,
            "metadata": self.metadata,
        }


# =============================================================================
# Failure Recovery
# =============================================================================


class FailureRecovery:
    """Handles task failure recovery strategies.

    Core Idea:
        When a task fails, this class determines the best recovery
        strategy based on the failure type and task characteristics.
    """

    RECOVERY_PROMPT: Final[str] = '''A task has failed during plan execution. Suggest a recovery strategy.

Failed Task: {task_name}
Description: {task_description}
Error: {error}
Attempt: {attempt} of {max_attempts}

Available strategies:
1. RETRY - Try the same task again
2. SKIP - Skip this task and continue
3. REPLACE - Replace with an alternative task
4. DECOMPOSE - Break into smaller subtasks
5. ABORT - Stop plan execution

Recommend a strategy and explain why:'''

    def __init__(self, llm: Optional[LLMInterface] = None) -> None:
        self._llm = llm

    def suggest_recovery(
        self,
        task: Task,
        error: str,
        attempt: int = 1,
        max_attempts: int = 3,
    ) -> RefinementStrategy:
        """Suggest a recovery strategy for a failed task."""
        if self._llm is None:
            return self._default_recovery(task, attempt, max_attempts)

        prompt = self.RECOVERY_PROMPT.format(
            task_name=task.name,
            task_description=task.description,
            error=error,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        response = self._llm.generate(prompt).lower()

        if "retry" in response:
            return RefinementStrategy.RETRY
        elif "skip" in response:
            return RefinementStrategy.SKIP
        elif "replace" in response:
            return RefinementStrategy.REPLACE
        elif "decompose" in response:
            return RefinementStrategy.DECOMPOSE
        elif "abort" in response:
            return RefinementStrategy.ABORT
        else:
            return self._default_recovery(task, attempt, max_attempts)

    def _default_recovery(
        self,
        task: Task,
        attempt: int,
        max_attempts: int,
    ) -> RefinementStrategy:
        """Default recovery logic without LLM."""
        if attempt < max_attempts:
            return RefinementStrategy.RETRY
        elif task.priority == TaskPriority.CRITICAL:
            return RefinementStrategy.ABORT
        else:
            return RefinementStrategy.SKIP

    def create_alternative_task(
        self,
        failed_task: Task,
        error: str,
    ) -> Task:
        """Create an alternative task to replace a failed one."""
        return Task(
            name=f"Alternative: {failed_task.name}",
            description=f"Alternative approach for: {failed_task.description}. Original failed with: {error}",
            priority=failed_task.priority,
            task_type=failed_task.task_type,
            dependencies=failed_task.dependencies.copy(),
        )


# =============================================================================
# Plan Optimizer
# =============================================================================


class PlanOptimizer:
    """Optimizes plans for efficiency and effectiveness.

    Optimization strategies:
    1. Remove redundant tasks
    2. Parallelize independent tasks
    3. Reorder for efficiency
    4. Merge similar tasks
    """

    def __init__(self, llm: Optional[LLMInterface] = None) -> None:
        self._llm = llm

    def optimize(self, plan: Plan) -> RefinementResult:
        """Optimize a plan."""
        original = copy.deepcopy(plan)
        changes: List[str] = []

        # Remove redundant tasks
        removed = self._remove_redundant_tasks(plan)
        if removed:
            changes.append(f"Removed {len(removed)} redundant tasks")

        # Identify parallelizable tasks
        parallel_groups = self._find_parallel_groups(plan)
        if parallel_groups:
            changes.append(f"Identified {len(parallel_groups)} parallel groups")
            plan.metadata["parallel_groups"] = parallel_groups

        # Reorder by priority
        reordered = self._optimize_order(plan)
        if reordered:
            changes.append("Reordered tasks by priority")

        return RefinementResult(
            success=len(changes) > 0,
            strategy_used=RefinementStrategy.REORDER,
            trigger=RefinementTrigger.OPTIMIZATION,
            changes_made=changes,
            original_plan=original,
            refined_plan=plan,
        )

    def _remove_redundant_tasks(self, plan: Plan) -> List[str]:
        """Remove duplicate or redundant tasks."""
        removed = []
        seen_names: Set[str] = set()
        tasks_to_remove = []

        for task in plan.tasks:
            normalized = task.name.lower().strip()
            if normalized in seen_names:
                tasks_to_remove.append(task.id)
                removed.append(task.id)
            else:
                seen_names.add(normalized)

        for task_id in tasks_to_remove:
            plan.remove_task(task_id)

        return removed

    def _find_parallel_groups(self, plan: Plan) -> List[List[str]]:
        """Find groups of tasks that can run in parallel."""
        groups: List[List[str]] = []
        task_deps = {t.id: set(t.dependencies) for t in plan.tasks}

        # Group tasks by their dependency set
        dep_groups: Dict[tuple, List[str]] = {}
        for task in plan.tasks:
            dep_key = tuple(sorted(task.dependencies))
            if dep_key not in dep_groups:
                dep_groups[dep_key] = []
            dep_groups[dep_key].append(task.id)

        # Groups with multiple tasks can run in parallel
        for dep_key, task_ids in dep_groups.items():
            if len(task_ids) > 1:
                groups.append(task_ids)

        return groups

    def _optimize_order(self, plan: Plan) -> bool:
        """Optimize task order based on priority."""
        original_order = [t.id for t in plan.tasks]
        
        # Sort by priority while respecting dependencies
        plan.tasks.sort(key=lambda t: t.priority.to_numeric(), reverse=True)
        
        new_order = [t.id for t in plan.tasks]
        return original_order != new_order


# =============================================================================
# Adaptive Replanner
# =============================================================================


class AdaptiveReplanner:
    """Dynamically replans based on execution feedback.

    Core Idea:
        Monitors execution and triggers replanning when conditions
        warrant plan modification.
    """

    REPLAN_PROMPT: Final[str] = '''The current plan needs adjustment based on execution feedback.

Original Goal: {goal}
Current Progress: {progress}
Issue: {issue}
Completed Tasks: {completed}
Remaining Tasks: {remaining}

Suggest modifications to the plan to address the issue and achieve the goal.
Format as a list of actions:
1. [Action]: [Description]
...

Suggested modifications:'''

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_refinements: int = DEFAULT_MAX_REFINEMENTS,
    ) -> None:
        self._llm = llm
        self._max_refinements = max_refinements
        self._refinement_count = 0

    def should_replan(
        self,
        plan: Plan,
        context: ExecutionContext,
        trigger: RefinementTrigger,
    ) -> bool:
        """Determine if replanning is needed."""
        if self._refinement_count >= self._max_refinements:
            return False

        if trigger == RefinementTrigger.TASK_FAILURE:
            failed_count = sum(
                1 for r in context.results.values()
                if r.status == ExecutionStatus.FAILURE
            )
            return failed_count > 0

        if trigger == RefinementTrigger.CONSTRAINT_VIOLATION:
            return True

        return False

    def replan(
        self,
        plan: Plan,
        context: ExecutionContext,
        trigger: RefinementTrigger,
        issue: str = "",
    ) -> RefinementResult:
        """Generate a refined plan."""
        self._refinement_count += 1
        original = copy.deepcopy(plan)
        changes: List[str] = []

        completed = [t.name for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        remaining = [t.name for t in plan.tasks if t.status != TaskStatus.COMPLETED]

        if self._llm is None:
            # Default replanning: reset failed tasks
            for task in plan.tasks:
                if task.status == TaskStatus.FAILED:
                    task.reset()
                    changes.append(f"Reset failed task: {task.name}")
        else:
            prompt = self.REPLAN_PROMPT.format(
                goal=plan.goal,
                progress=f"{plan.progress:.0%}",
                issue=issue or trigger.value,
                completed=completed or "None",
                remaining=remaining or "None",
            )
            response = self._llm.generate(prompt)
            changes.append(f"LLM suggested: {response[:100]}...")

        return RefinementResult(
            success=len(changes) > 0,
            strategy_used=RefinementStrategy.REPLACE,
            trigger=trigger,
            changes_made=changes,
            original_plan=original,
            refined_plan=plan,
            metadata={"refinement_count": self._refinement_count},
        )


# =============================================================================
# Main Plan Refinement Class
# =============================================================================


class PlanRefinement:
    """Main interface for plan refinement operations.

    Combines failure recovery, optimization, and replanning into
    a unified interface.

    Example:
        >>> refinement = PlanRefinement()
        >>> result = refinement.handle_failure(plan, task, "Connection timeout")
        >>> if result.success:
        ...     print(f"Applied: {result.strategy_used}")
    """

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        max_refinements: int = DEFAULT_MAX_REFINEMENTS,
    ) -> None:
        self._llm = llm
        self._recovery = FailureRecovery(llm)
        self._optimizer = PlanOptimizer(llm)
        self._replanner = AdaptiveReplanner(llm, max_refinements)
        self._validator = PlanValidator(llm)

    def handle_failure(
        self,
        plan: Plan,
        failed_task: Task,
        error: str,
        context: Optional[ExecutionContext] = None,
        attempt: int = 1,
    ) -> RefinementResult:
        """Handle a task failure."""
        strategy = self._recovery.suggest_recovery(failed_task, error, attempt)
        changes: List[str] = []

        if strategy == RefinementStrategy.RETRY:
            failed_task.reset()
            changes.append(f"Reset task '{failed_task.name}' for retry")

        elif strategy == RefinementStrategy.SKIP:
            failed_task.status = TaskStatus.CANCELLED
            self._update_dependents(plan, failed_task.id)
            changes.append(f"Skipped task '{failed_task.name}'")

        elif strategy == RefinementStrategy.REPLACE:
            alt_task = self._recovery.create_alternative_task(failed_task, error)
            idx = next(i for i, t in enumerate(plan.tasks) if t.id == failed_task.id)
            plan.tasks[idx] = alt_task
            changes.append(f"Replaced '{failed_task.name}' with alternative")

        elif strategy == RefinementStrategy.ABORT:
            plan.status = PlanStatus.FAILED
            changes.append("Aborted plan execution")

        return RefinementResult(
            success=strategy != RefinementStrategy.ABORT,
            strategy_used=strategy,
            trigger=RefinementTrigger.TASK_FAILURE,
            changes_made=changes,
            refined_plan=plan,
        )

    def _update_dependents(self, plan: Plan, skipped_task_id: str) -> None:
        """Update tasks that depend on a skipped task."""
        for task in plan.tasks:
            if skipped_task_id in task.dependencies:
                task.dependencies.remove(skipped_task_id)

    def optimize(self, plan: Plan) -> RefinementResult:
        """Optimize a plan."""
        return self._optimizer.optimize(plan)

    def replan(
        self,
        plan: Plan,
        context: ExecutionContext,
        trigger: RefinementTrigger,
        issue: str = "",
    ) -> RefinementResult:
        """Trigger replanning."""
        if self._replanner.should_replan(plan, context, trigger):
            return self._replanner.replan(plan, context, trigger, issue)
        return RefinementResult(
            success=False,
            strategy_used=RefinementStrategy.SKIP,
            trigger=trigger,
            changes_made=["Replanning not needed"],
        )

    def validate_and_refine(self, plan: Plan) -> Tuple[bool, RefinementResult]:
        """Validate plan and refine if needed."""
        validation = self._validator.validate(plan)
        
        if validation.is_valid:
            return True, RefinementResult(
                success=True,
                strategy_used=RefinementStrategy.SKIP,
                trigger=RefinementTrigger.MANUAL,
                changes_made=["Plan is valid, no refinement needed"],
            )

        # Try to fix issues
        changes = []
        for issue in validation.issues:
            if "Circular dependency" in issue:
                changes.append("Cannot auto-fix circular dependencies")
            elif "invalid dependency" in issue:
                self._fix_invalid_dependencies(plan)
                changes.append("Removed invalid dependencies")

        return len(validation.issues) == 0, RefinementResult(
            success=len(changes) > 0,
            strategy_used=RefinementStrategy.REPLACE,
            trigger=RefinementTrigger.CONSTRAINT_VIOLATION,
            changes_made=changes,
            refined_plan=plan,
        )

    def _fix_invalid_dependencies(self, plan: Plan) -> None:
        """Remove invalid dependencies from tasks."""
        valid_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            task.dependencies = [d for d in task.dependencies if d in valid_ids]


# =============================================================================
# Factory Functions
# =============================================================================


def create_refinement(
    llm: Optional[LLMInterface] = None,
    max_refinements: int = DEFAULT_MAX_REFINEMENTS,
) -> PlanRefinement:
    """Factory function to create a plan refinement handler."""
    return PlanRefinement(llm=llm, max_refinements=max_refinements)
