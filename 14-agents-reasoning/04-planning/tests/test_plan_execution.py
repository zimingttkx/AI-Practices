"""
Comprehensive Tests for Plan Execution Module.

Test Coverage:
    - ExecutionResult and ExecutionContext
    - ExecutionPolicy configuration
    - SimpleTaskExecutor and LLMTaskExecutor
    - PlanExecutor: sequential and parallel execution
    - ExecutionMonitor statistics
    - Factory functions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import time

from task_decomposition import Task, TaskStatus, TaskPriority, TaskType
from plan_generation import Plan, PlanStatus
from plan_execution import (
    ExecutionResult,
    ExecutionStatus,
    ExecutionContext,
    ExecutionPolicy,
    SimpleTaskExecutor,
    LLMTaskExecutor,
    PlanExecutor,
    ExecutionMonitor,
    create_executor,
    execute_plan,
)


# =============================================================================
# ExecutionResult Tests
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_create_success_result(self):
        result = ExecutionResult(
            task_id="task_1",
            status=ExecutionStatus.SUCCESS,
            result="Done"
        )
        assert result.is_success is True
        assert result.is_failure is False

    def test_create_failure_result(self):
        result = ExecutionResult(
            task_id="task_1",
            status=ExecutionStatus.FAILURE,
            error="Error occurred"
        )
        assert result.is_success is False
        assert result.is_failure is True

    def test_result_to_dict(self):
        result = ExecutionResult(
            task_id="task_1",
            status=ExecutionStatus.SUCCESS,
            duration=1.5
        )
        d = result.to_dict()
        assert d["task_id"] == "task_1"
        assert d["status"] == "success"
        assert d["duration"] == 1.5


# =============================================================================
# ExecutionContext Tests
# =============================================================================


class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_create_context(self):
        context = ExecutionContext()
        assert len(context.variables) == 0
        assert len(context.results) == 0

    def test_set_get_variable(self):
        context = ExecutionContext()
        context.set_variable("key", "value")
        assert context.get_variable("key") == "value"

    def test_get_variable_default(self):
        context = ExecutionContext()
        assert context.get_variable("missing", "default") == "default"

    def test_add_get_result(self):
        context = ExecutionContext()
        result = ExecutionResult(task_id="t1", status=ExecutionStatus.SUCCESS)
        context.add_result(result)
        
        assert context.get_result("t1") is not None

    def test_get_task_output(self):
        context = ExecutionContext()
        result = ExecutionResult(
            task_id="t1",
            status=ExecutionStatus.SUCCESS,
            result="output"
        )
        context.add_result(result)
        
        assert context.get_task_output("t1") == "output"

    def test_get_completed_task_ids(self):
        context = ExecutionContext()
        context.add_result(ExecutionResult("t1", ExecutionStatus.SUCCESS))
        context.add_result(ExecutionResult("t2", ExecutionStatus.FAILURE))
        
        completed = context.get_completed_task_ids()
        assert "t1" in completed
        assert "t2" not in completed


# =============================================================================
# ExecutionPolicy Tests
# =============================================================================


class TestExecutionPolicy:
    """Tests for ExecutionPolicy."""

    def test_default_policy(self):
        policy = ExecutionPolicy()
        assert policy.max_retries == 3
        assert policy.parallel is False
        assert policy.stop_on_failure is True

    def test_custom_policy(self):
        policy = ExecutionPolicy(
            max_retries=5,
            parallel=True,
            max_workers=8
        )
        assert policy.max_retries == 5
        assert policy.parallel is True
        assert policy.max_workers == 8

    def test_policy_to_dict(self):
        policy = ExecutionPolicy()
        d = policy.to_dict()
        assert "max_retries" in d
        assert "parallel" in d


# =============================================================================
# TaskExecutor Tests
# =============================================================================


class TestSimpleTaskExecutor:
    """Tests for SimpleTaskExecutor."""

    def test_execute_without_handler(self):
        executor = SimpleTaskExecutor()
        task = Task(name="Test")
        context = ExecutionContext()
        
        result = executor.execute(task, context)
        assert "completed" in result.lower()

    def test_register_handler(self):
        executor = SimpleTaskExecutor()
        
        def handler(task, ctx):
            return f"Handled: {task.name}"
        
        executor.register("generic", handler)
        task = Task(name="Test", task_type=TaskType.GENERIC)
        context = ExecutionContext()
        
        result = executor.execute(task, context)
        assert "Handled" in result

    def test_default_handler(self):
        executor = SimpleTaskExecutor()
        
        def default(task, ctx):
            return "Default"
        
        executor.set_default_handler(default)
        task = Task(name="Test")
        context = ExecutionContext()
        
        result = executor.execute(task, context)
        assert result == "Default"


class TestLLMTaskExecutor:
    """Tests for LLMTaskExecutor."""

    def test_execute_without_llm(self):
        executor = LLMTaskExecutor()
        task = Task(name="Test")
        context = ExecutionContext()
        
        result = executor.execute(task, context)
        assert "simulated" in result.lower()


# =============================================================================
# ExecutionMonitor Tests
# =============================================================================


class TestExecutionMonitor:
    """Tests for ExecutionMonitor."""

    def test_monitor_start(self):
        monitor = ExecutionMonitor()
        monitor.start()
        assert monitor.elapsed_time >= 0

    def test_record_task(self):
        monitor = ExecutionMonitor()
        monitor.record_task("t1", 1.0, True)
        monitor.record_task("t2", 2.0, False)
        
        assert monitor.success_rate == 0.5

    def test_average_task_time(self):
        monitor = ExecutionMonitor()
        monitor.record_task("t1", 1.0, True)
        monitor.record_task("t2", 3.0, True)
        
        assert monitor.average_task_time == 2.0

    def test_get_stats(self):
        monitor = ExecutionMonitor()
        monitor.start()
        monitor.record_task("t1", 1.0, True)
        
        stats = monitor.get_stats()
        assert "elapsed_time" in stats
        assert "success_rate" in stats


# =============================================================================
# PlanExecutor Tests
# =============================================================================


class TestPlanExecutor:
    """Tests for PlanExecutor."""

    def test_execute_simple_plan(self):
        executor = PlanExecutor()
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task 1"))
        
        context = executor.execute(plan)
        assert len(context.results) == 1

    def test_execute_with_dependencies(self):
        executor = PlanExecutor()
        plan = Plan(goal="Test")
        t1 = Task(name="T1")
        t2 = Task(name="T2", dependencies=[t1.id])
        plan.add_task(t1)
        plan.add_task(t2)
        
        context = executor.execute(plan)
        assert len(context.results) == 2

    def test_execute_respects_order(self):
        executor = PlanExecutor()
        plan = Plan(goal="Test")
        t1 = Task(name="T1")
        t2 = Task(name="T2", dependencies=[t1.id])
        plan.add_task(t2)
        plan.add_task(t1)
        
        context = executor.execute(plan)
        
        # T1 should complete before T2
        r1 = context.get_result(t1.id)
        r2 = context.get_result(t2.id)
        assert r1.completed_at <= r2.completed_at

    def test_plan_status_updated(self):
        executor = PlanExecutor()
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task"))
        
        executor.execute(plan)
        assert plan.status == PlanStatus.COMPLETED

    def test_monitor_stats(self):
        executor = PlanExecutor()
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task"))
        
        executor.execute(plan)
        stats = executor.monitor.get_stats()
        assert stats["tasks_completed"] == 1


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestExecutionFactoryFunctions:
    """Tests for execution factory functions."""

    def test_create_executor_simple(self):
        executor = create_executor("simple")
        assert isinstance(executor, PlanExecutor)

    def test_create_executor_llm(self):
        executor = create_executor("llm")
        assert isinstance(executor, PlanExecutor)

    def test_create_executor_with_policy(self):
        policy = ExecutionPolicy(max_retries=5)
        executor = create_executor(policy=policy)
        assert executor.policy.max_retries == 5

    def test_execute_plan_function(self):
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task"))
        
        context = execute_plan(plan)
        assert len(context.results) == 1
