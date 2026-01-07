"""
Comprehensive Tests for Plan Generation Module.

Test Coverage:
    - Plan dataclass: creation, task management, progress tracking
    - Constraint handling
    - ForwardPlanner, BackwardPlanner, HierarchicalPlanner
    - PlanValidator
    - Factory functions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime

from task_decomposition import Task, TaskStatus, TaskPriority
from plan_generation import (
    Plan,
    PlanStatus,
    PlanningStrategy,
    Constraint,
    ConstraintType,
    ForwardPlanner,
    BackwardPlanner,
    HierarchicalPlanner,
    PlanValidator,
    ValidationResult,
    create_planner,
    create_plan,
    format_plan,
)


# =============================================================================
# Plan Tests
# =============================================================================


class TestPlan:
    """Tests for Plan dataclass."""

    def test_create_plan(self):
        plan = Plan(goal="Build a website")
        assert plan.goal == "Build a website"
        assert plan.status == PlanStatus.DRAFT
        assert len(plan.tasks) == 0

    def test_plan_id_generated(self):
        plan = Plan(goal="Test")
        assert plan.id is not None
        assert plan.id.startswith("plan_")

    def test_add_task(self):
        plan = Plan(goal="Test")
        task = Task(name="Task 1")
        plan.add_task(task)
        
        assert len(plan.tasks) == 1
        assert plan.tasks[0].name == "Task 1"

    def test_remove_task(self):
        plan = Plan(goal="Test")
        task = Task(name="Task 1")
        plan.add_task(task)
        
        result = plan.remove_task(task.id)
        assert result is True
        assert len(plan.tasks) == 0

    def test_get_task(self):
        plan = Plan(goal="Test")
        task = Task(name="Task 1")
        plan.add_task(task)
        
        found = plan.get_task(task.id)
        assert found is not None
        assert found.name == "Task 1"

    def test_add_constraint(self):
        plan = Plan(goal="Test")
        constraint = Constraint(
            name="Time Limit",
            constraint_type=ConstraintType.TIME,
            description="Must complete in 1 hour"
        )
        plan.add_constraint(constraint)
        
        assert len(plan.constraints) == 1

    def test_progress_empty(self):
        plan = Plan(goal="Test")
        assert plan.progress == 0.0

    def test_progress_partial(self):
        plan = Plan(goal="Test")
        plan.add_task(Task(name="T1", status=TaskStatus.COMPLETED))
        plan.add_task(Task(name="T2", status=TaskStatus.PENDING))
        
        assert plan.progress == 0.5

    def test_progress_complete(self):
        plan = Plan(goal="Test")
        plan.add_task(Task(name="T1", status=TaskStatus.COMPLETED))
        plan.add_task(Task(name="T2", status=TaskStatus.COMPLETED))
        
        assert plan.progress == 1.0

    def test_is_complete(self):
        plan = Plan(goal="Test")
        plan.add_task(Task(name="T1", status=TaskStatus.COMPLETED))
        
        assert plan.is_complete is True

    def test_has_failed_tasks(self):
        plan = Plan(goal="Test")
        plan.add_task(Task(name="T1", status=TaskStatus.FAILED))
        
        assert plan.has_failed_tasks is True

    def test_get_ready_tasks(self):
        plan = Plan(goal="Test")
        t1 = Task(name="T1", status=TaskStatus.COMPLETED)
        t2 = Task(name="T2", dependencies=[t1.id])
        t3 = Task(name="T3")
        
        plan.add_task(t1)
        plan.add_task(t2)
        plan.add_task(t3)
        
        ready = plan.get_ready_tasks()
        assert len(ready) == 2

    def test_get_execution_order(self):
        plan = Plan(goal="Test")
        t1 = Task(name="T1")
        t2 = Task(name="T2", dependencies=[t1.id])
        t3 = Task(name="T3", dependencies=[t2.id])
        
        plan.add_task(t3)
        plan.add_task(t1)
        plan.add_task(t2)
        
        order = plan.get_execution_order()
        assert order[0].id == t1.id
        assert order[-1].id == t3.id

    def test_execution_order_cycle_raises(self):
        plan = Plan(goal="Test")
        t1 = Task(name="T1")
        t2 = Task(name="T2")
        t1.dependencies = [t2.id]
        t2.dependencies = [t1.id]
        
        plan.add_task(t1)
        plan.add_task(t2)
        
        with pytest.raises(ValueError):
            plan.get_execution_order()

    def test_plan_start(self):
        plan = Plan(goal="Test")
        plan.start()
        
        assert plan.status == PlanStatus.IN_PROGRESS
        assert plan.started_at is not None

    def test_plan_complete(self):
        plan = Plan(goal="Test")
        plan.start()
        plan.complete()
        
        assert plan.status == PlanStatus.COMPLETED
        assert plan.completed_at is not None

    def test_plan_fail(self):
        plan = Plan(goal="Test")
        plan.fail()
        
        assert plan.status == PlanStatus.FAILED

    def test_plan_to_dict(self):
        plan = Plan(goal="Test Goal")
        plan.add_task(Task(name="Task 1"))
        
        d = plan.to_dict()
        assert d["goal"] == "Test Goal"
        assert len(d["tasks"]) == 1

    def test_plan_from_dict(self):
        data = {
            "goal": "Test Goal",
            "tasks": [{"name": "Task 1", "description": "Desc", "status": "pending", "priority": "medium"}],
            "constraints": [],
            "status": "draft",
        }
        plan = Plan.from_dict(data)
        
        assert plan.goal == "Test Goal"
        assert len(plan.tasks) == 1


# =============================================================================
# Constraint Tests
# =============================================================================


class TestConstraint:
    """Tests for Constraint dataclass."""

    def test_create_constraint(self):
        c = Constraint(
            name="Time",
            constraint_type=ConstraintType.TIME,
            description="1 hour limit"
        )
        assert c.name == "Time"
        assert c.is_hard is True

    def test_constraint_to_dict(self):
        c = Constraint(name="Test", constraint_type=ConstraintType.RESOURCE)
        d = c.to_dict()
        assert d["name"] == "Test"
        assert d["type"] == "resource"

    def test_constraint_from_dict(self):
        data = {"name": "Test", "type": "time", "is_hard": False}
        c = Constraint.from_dict(data)
        assert c.constraint_type == ConstraintType.TIME
        assert c.is_hard is False


# =============================================================================
# Planner Tests
# =============================================================================


class TestForwardPlanner:
    """Tests for ForwardPlanner."""

    def test_generate_without_llm(self):
        planner = ForwardPlanner()
        plan = planner.generate("Build a website")
        
        assert plan.goal == "Build a website"
        assert len(plan.tasks) > 0

    def test_generate_with_constraints(self):
        planner = ForwardPlanner()
        constraints = [
            Constraint(name="Time", constraint_type=ConstraintType.TIME)
        ]
        plan = planner.generate("Test", constraints=constraints)
        
        assert len(plan.constraints) == 1

    def test_tasks_have_dependencies(self):
        planner = ForwardPlanner()
        plan = planner.generate("Build app")
        
        # Second task should depend on first
        if len(plan.tasks) > 1:
            assert len(plan.tasks[1].dependencies) > 0


class TestBackwardPlanner:
    """Tests for BackwardPlanner."""

    def test_generate_without_llm(self):
        planner = BackwardPlanner()
        plan = planner.generate("Achieve goal")
        
        assert plan.goal == "Achieve goal"
        assert len(plan.tasks) > 0


class TestHierarchicalPlanner:
    """Tests for HierarchicalPlanner."""

    def test_generate_without_llm(self):
        planner = HierarchicalPlanner()
        plan = planner.generate("Complete project")
        
        assert plan.goal == "Complete project"
        assert len(plan.tasks) > 0

    def test_tasks_have_phase_metadata(self):
        planner = HierarchicalPlanner()
        plan = planner.generate("Build system")
        
        # Tasks should have phase metadata
        phases = set()
        for task in plan.tasks:
            if "phase" in task.metadata:
                phases.add(task.metadata["phase"])


# =============================================================================
# PlanValidator Tests
# =============================================================================


class TestPlanValidator:
    """Tests for PlanValidator."""

    def test_validate_empty_plan(self):
        validator = PlanValidator()
        plan = Plan(goal="Empty")
        
        result = validator.validate(plan)
        assert result.is_valid is False
        assert "no tasks" in result.issues[0].lower()

    def test_validate_valid_plan(self):
        validator = PlanValidator()
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task 1"))
        
        result = validator.validate(plan)
        assert result.is_valid is True

    def test_validate_invalid_dependency(self):
        validator = PlanValidator()
        plan = Plan(goal="Test")
        task = Task(name="Task", dependencies=["nonexistent"])
        plan.add_task(task)
        
        result = validator.validate(plan)
        assert result.is_valid is False

    def test_validation_score(self):
        validator = PlanValidator()
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task 1"))
        
        result = validator.validate(plan)
        assert 0.0 <= result.score <= 1.0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestPlanFactoryFunctions:
    """Tests for plan factory functions."""

    def test_create_planner_forward(self):
        planner = create_planner("forward")
        assert isinstance(planner, ForwardPlanner)

    def test_create_planner_backward(self):
        planner = create_planner("backward")
        assert isinstance(planner, BackwardPlanner)

    def test_create_planner_hierarchical(self):
        planner = create_planner("hierarchical")
        assert isinstance(planner, HierarchicalPlanner)

    def test_create_planner_enum(self):
        planner = create_planner(PlanningStrategy.FORWARD)
        assert isinstance(planner, ForwardPlanner)

    def test_create_plan(self):
        tasks = [Task(name="T1"), Task(name="T2")]
        plan = create_plan("Goal", tasks=tasks)
        
        assert plan.goal == "Goal"
        assert len(plan.tasks) == 2

    def test_format_plan(self):
        plan = Plan(goal="Test Goal")
        plan.add_task(Task(name="Task 1"))
        
        output = format_plan(plan)
        assert "Test Goal" in output
        assert "Task 1" in output

    def test_format_plan_verbose(self):
        plan = Plan(goal="Test")
        plan.add_task(Task(name="Task", description="Description"))
        
        output = format_plan(plan, verbose=True)
        assert "Description" in output
