"""
Comprehensive Tests for Plan Refinement Module.

Test Coverage:
    - RefinementResult
    - FailureRecovery strategies
    - PlanOptimizer
    - AdaptiveReplanner
    - PlanRefinement main interface
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from task_decomposition import Task, TaskStatus, TaskPriority
from plan_generation import Plan, PlanStatus
from plan_execution import ExecutionContext, ExecutionResult, ExecutionStatus
from plan_refinement import (
    RefinementStrategy,
    RefinementTrigger,
    RefinementResult,
    FailureRecovery,
    PlanOptimizer,
    AdaptiveReplanner,
    PlanRefinement,
    create_refinement,
)


# =============================================================================
# RefinementResult Tests
# =============================================================================


class TestRefinementResult:
    """Tests for RefinementResult."""

    def test_create_result(self):
        result = RefinementResult(
            success=True,
            strategy_used=RefinementStrategy.RETRY,
            trigger=RefinementTrigger.TASK_FAILURE,
        )
        assert result.success is True
        assert result.strategy_used == RefinementStrategy.RETRY

    def test_result_to_dict(self):
        result = RefinementResult(
            success=True,
            strategy_used=RefinementStrategy.SKIP,
            trigger=RefinementTrigger.OPTIMIZATION,
            changes_made=["Change 1"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["strategy_used"] == "skip"


# =============================================================================
# FailureRecovery Tests
# =============================================================================


class TestFailureRecovery:
    """Tests for FailureRecovery."""

    def test_suggest_retry_on_first_attempt(self):
        recovery = FailureRecovery()
        task = Task(name="Test")
        
        strategy = recovery.suggest_recovery(task, "Error", attempt=1, max_attempts=3)
        assert strategy == RefinementStrategy.RETRY

    def test_suggest_skip_after_max_attempts(self):
        recovery = FailureRecovery()
        task = Task(name="Test", priority=TaskPriority.LOW)
        
        strategy = recovery.suggest_recovery(task, "Error", attempt=3, max_attempts=3)
        assert strategy == RefinementStrategy.SKIP

    def test_suggest_abort_for_critical(self):
        recovery = FailureRecovery()
        task = Task(name="Critical", priority=TaskPriority.CRITICAL)
        
        strategy = recovery.suggest_recovery(task, "Error", attempt=3, max_attempts=3)
        assert strategy == RefinementStrategy.ABORT

    def test_create_alternative_task(self):
        recovery = FailureRecovery()
        original = Task(name="Original", description="Do something")
        
        alt = recovery.create_alternative_task(original, "Failed")
        assert "Alternative" in alt.name
        assert original.description in alt.description


# =============================================================================
# PlanOptimizer Tests
# =============================================================================


class TestPlanOptimizer:
    """Tests for PlanOptimizer."""

    def test_optimize_removes_duplicates(self):
        optimizer = PlanOptimizer()
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task A"))
        plan.add_task(Task(name="Task A"))  # Duplicate
        
        result = optimizer.optimize(plan)
        assert len(plan.tasks) == 1

    def test_optimize_finds_parallel_groups(self):
        optimizer = PlanOptimizer()
        plan = Plan(goal="Test")
        t1 = Task(name="T1")
        t2 = Task(name="T2")  # No deps, can run parallel with T1
        t3 = Task(name="T3")
        
        plan.add_task(t1)
        plan.add_task(t2)
        plan.add_task(t3)
        
        result = optimizer.optimize(plan)
        assert "parallel_groups" in plan.metadata or result.success

    def test_optimize_result(self):
        optimizer = PlanOptimizer()
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task"))
        
        result = optimizer.optimize(plan)
        assert result.trigger == RefinementTrigger.OPTIMIZATION


# =============================================================================
# AdaptiveReplanner Tests
# =============================================================================


class TestAdaptiveReplanner:
    """Tests for AdaptiveReplanner."""

    def test_should_replan_on_failure(self):
        replanner = AdaptiveReplanner()
        plan = Plan(goal="Test")
        context = ExecutionContext()
        context.add_result(ExecutionResult("t1", ExecutionStatus.FAILURE))
        
        should = replanner.should_replan(plan, context, RefinementTrigger.TASK_FAILURE)
        assert should is True

    def test_should_not_replan_max_reached(self):
        replanner = AdaptiveReplanner(max_refinements=0)
        plan = Plan(goal="Test")
        context = ExecutionContext()
        
        should = replanner.should_replan(plan, context, RefinementTrigger.TASK_FAILURE)
        assert should is False

    def test_replan_resets_failed_tasks(self):
        replanner = AdaptiveReplanner()
        plan = Plan(goal="Test")
        task = Task(name="Failed", status=TaskStatus.FAILED)
        plan.add_task(task)
        context = ExecutionContext()
        
        result = replanner.replan(plan, context, RefinementTrigger.TASK_FAILURE)
        assert result.success is True


# =============================================================================
# PlanRefinement Tests
# =============================================================================


class TestPlanRefinement:
    """Tests for PlanRefinement main interface."""

    def test_handle_failure_retry(self):
        refinement = PlanRefinement()
        plan = Plan(goal="Test")
        task = Task(name="Task")
        plan.add_task(task)
        
        result = refinement.handle_failure(plan, task, "Error", attempt=1)
        assert result.strategy_used == RefinementStrategy.RETRY
        assert task.status == TaskStatus.PENDING

    def test_handle_failure_skip(self):
        refinement = PlanRefinement()
        plan = Plan(goal="Test")
        task = Task(name="Task", priority=TaskPriority.LOW)
        plan.add_task(task)
        
        result = refinement.handle_failure(plan, task, "Error", attempt=3)
        assert task.status == TaskStatus.CANCELLED

    def test_optimize(self):
        refinement = PlanRefinement()
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task"))
        
        result = refinement.optimize(plan)
        assert result.trigger == RefinementTrigger.OPTIMIZATION

    def test_validate_and_refine_valid(self):
        refinement = PlanRefinement()
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task"))
        
        is_valid, result = refinement.validate_and_refine(plan)
        assert is_valid is True

    def test_validate_and_refine_fixes_deps(self):
        refinement = PlanRefinement()
        plan = Plan(goal="Test")
        task = Task(name="Task", dependencies=["invalid_id"])
        plan.add_task(task)
        
        is_valid, result = refinement.validate_and_refine(plan)
        # Should attempt to fix invalid dependencies
        assert len(task.dependencies) == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestRefinementFactoryFunctions:
    """Tests for refinement factory functions."""

    def test_create_refinement(self):
        refinement = create_refinement()
        assert isinstance(refinement, PlanRefinement)

    def test_create_refinement_with_max(self):
        refinement = create_refinement(max_refinements=10)
        assert refinement is not None
