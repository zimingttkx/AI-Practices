"""
Comprehensive Tests for Task Decomposition Module.

Test Coverage:
    - Task dataclass: creation, serialization, state management
    - TaskStatus and TaskPriority enums
    - HierarchicalDecomposer: decomposition, recursion
    - SequentialDecomposer: sequential steps, dependencies
    - DependencyAnalyzer: analysis, cycle detection, topological sort
    - TaskDecomposer: main interface, validation
    - Factory functions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime

from task_decomposition import (
    Task,
    TaskStatus,
    TaskPriority,
    TaskType,
    HierarchicalDecomposer,
    SequentialDecomposer,
    DependencyAnalyzer,
    TaskDecomposer,
    create_task,
    create_decomposer,
    format_task_tree,
)


# =============================================================================
# Task Tests
# =============================================================================


class TestTask:
    """Tests for Task dataclass."""

    def test_create_task(self):
        task = Task(name="Test Task", description="A test task")
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM

    def test_task_id_generated(self):
        task = Task(name="Test")
        assert task.id is not None
        assert task.id.startswith("task_")

    def test_task_status_from_string(self):
        task = Task(name="Test", status="completed")
        assert task.status == TaskStatus.COMPLETED

    def test_task_priority_from_string(self):
        task = Task(name="Test", priority="high")
        assert task.priority == TaskPriority.HIGH

    def test_task_is_leaf(self):
        task = Task(name="Leaf Task")
        assert task.is_leaf is True
        
        task.add_subtask(Task(name="Child"))
        assert task.is_leaf is False

    def test_task_is_root(self):
        task = Task(name="Root")
        assert task.is_root is True
        
        child = Task(name="Child")
        task.add_subtask(child)
        assert child.is_root is False

    def test_task_add_subtask(self):
        parent = Task(name="Parent")
        child = Task(name="Child")
        parent.add_subtask(child)
        
        assert len(parent.subtasks) == 1
        assert child.parent_id == parent.id

    def test_task_add_dependency(self):
        task = Task(name="Task")
        task.add_dependency("dep_1")
        task.add_dependency("dep_2")
        
        assert "dep_1" in task.dependencies
        assert "dep_2" in task.dependencies

    def test_task_remove_dependency(self):
        task = Task(name="Task", dependencies=["dep_1", "dep_2"])
        result = task.remove_dependency("dep_1")
        
        assert result is True
        assert "dep_1" not in task.dependencies

    def test_task_is_ready(self):
        task = Task(name="Task")
        assert task.is_ready is True
        
        task.add_dependency("dep_1")
        assert task.is_ready is False

    def test_task_depth(self):
        root = Task(name="Root")
        child = Task(name="Child")
        grandchild = Task(name="Grandchild")
        
        root.add_subtask(child)
        child.add_subtask(grandchild)
        
        assert root.depth == 2
        assert child.depth == 1
        assert grandchild.depth == 0

    def test_task_total_subtasks(self):
        root = Task(name="Root")
        child1 = Task(name="Child1")
        child2 = Task(name="Child2")
        grandchild = Task(name="Grandchild")
        
        root.add_subtask(child1)
        root.add_subtask(child2)
        child1.add_subtask(grandchild)
        
        assert root.total_subtasks == 3

    def test_task_completion_ratio(self):
        root = Task(name="Root")
        child1 = Task(name="Child1", status=TaskStatus.COMPLETED)
        child2 = Task(name="Child2", status=TaskStatus.PENDING)
        
        root.add_subtask(child1)
        root.add_subtask(child2)
        
        assert root.completion_ratio == 0.5

    def test_task_get_leaf_tasks(self):
        root = Task(name="Root")
        child = Task(name="Child")
        leaf1 = Task(name="Leaf1")
        leaf2 = Task(name="Leaf2")
        
        root.add_subtask(child)
        child.add_subtask(leaf1)
        child.add_subtask(leaf2)
        
        leaves = root.get_leaf_tasks()
        assert len(leaves) == 2

    def test_task_find_subtask(self):
        root = Task(name="Root")
        child = Task(name="Child")
        grandchild = Task(name="Grandchild")
        
        root.add_subtask(child)
        child.add_subtask(grandchild)
        
        found = root.find_subtask(grandchild.id)
        assert found is not None
        assert found.name == "Grandchild"

    def test_task_start(self):
        task = Task(name="Task")
        task.start()
        
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None

    def test_task_complete(self):
        task = Task(name="Task")
        task.start()
        task.complete(result="Done")
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Done"
        assert task.completed_at is not None

    def test_task_fail(self):
        task = Task(name="Task")
        task.start()
        task.fail("Error occurred")
        
        assert task.status == TaskStatus.FAILED
        assert task.error == "Error occurred"

    def test_task_reset(self):
        task = Task(name="Task")
        task.start()
        task.complete("Done")
        task.reset()
        
        assert task.status == TaskStatus.PENDING
        assert task.result is None

    def test_task_to_dict(self):
        task = Task(name="Test", description="Desc")
        d = task.to_dict()
        
        assert d["name"] == "Test"
        assert d["description"] == "Desc"
        assert "id" in d

    def test_task_from_dict(self):
        data = {
            "id": "task_123",
            "name": "Test",
            "description": "Desc",
            "status": "pending",
            "priority": "high",
        }
        task = Task.from_dict(data)
        
        assert task.name == "Test"
        assert task.priority == TaskPriority.HIGH

    def test_task_serialization_roundtrip(self):
        original = Task(
            name="Test",
            description="Description",
            priority=TaskPriority.HIGH,
        )
        original.add_subtask(Task(name="Child"))
        
        data = original.to_dict()
        restored = Task.from_dict(data)
        
        assert restored.name == original.name
        assert len(restored.subtasks) == 1


# =============================================================================
# TaskStatus Tests
# =============================================================================


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses(self):
        statuses = list(TaskStatus)
        assert len(statuses) == 6

    def test_is_terminal(self):
        assert TaskStatus.COMPLETED.is_terminal() is True
        assert TaskStatus.CANCELLED.is_terminal() is True
        assert TaskStatus.PENDING.is_terminal() is False

    def test_is_active(self):
        assert TaskStatus.IN_PROGRESS.is_active() is True
        assert TaskStatus.PENDING.is_active() is False


# =============================================================================
# TaskPriority Tests
# =============================================================================


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_priority_comparison(self):
        assert TaskPriority.LOW < TaskPriority.MEDIUM
        assert TaskPriority.MEDIUM < TaskPriority.HIGH
        assert TaskPriority.HIGH < TaskPriority.CRITICAL

    def test_to_numeric(self):
        assert TaskPriority.LOW.to_numeric() == 1
        assert TaskPriority.CRITICAL.to_numeric() == 4


# =============================================================================
# HierarchicalDecomposer Tests
# =============================================================================


class TestHierarchicalDecomposer:
    """Tests for HierarchicalDecomposer."""

    def test_decompose_without_llm(self):
        decomposer = HierarchicalDecomposer()
        task = Task(
            name="Build App",
            description="Build a complete web application with frontend and backend"
        )
        subtasks = decomposer.decompose(task)
        
        assert len(subtasks) == 3
        assert any("Analysis" in t.name for t in subtasks)

    def test_is_atomic_short_description(self):
        decomposer = HierarchicalDecomposer()
        task = Task(name="Short", description="Do it")
        
        assert decomposer.is_atomic(task) is True

    def test_is_atomic_long_description(self):
        decomposer = HierarchicalDecomposer()
        task = Task(
            name="Complex",
            description="This is a very long description that exceeds the minimum atomic length threshold"
        )
        
        assert decomposer.is_atomic(task) is False

    def test_custom_atomic_checker(self):
        def always_atomic(task):
            return True
        
        decomposer = HierarchicalDecomposer(atomic_checker=always_atomic)
        task = Task(name="Test", description="Long description here")
        
        assert decomposer.is_atomic(task) is True

    def test_decompose_recursive(self):
        decomposer = HierarchicalDecomposer(max_depth=2)
        task = Task(
            name="Project",
            description="Complete software project with multiple phases and deliverables"
        )
        
        result = decomposer.decompose_recursive(task)
        assert len(result.subtasks) > 0


# =============================================================================
# SequentialDecomposer Tests
# =============================================================================


class TestSequentialDecomposer:
    """Tests for SequentialDecomposer."""

    def test_decompose_creates_dependencies(self):
        decomposer = SequentialDecomposer()
        task = Task(
            name="Deploy",
            description="Deploy application to production environment with all steps"
        )
        
        steps = decomposer.decompose(task)
        
        assert len(steps) >= 2
        # Second step should depend on first
        if len(steps) > 1:
            assert steps[0].id in steps[1].dependencies

    def test_atomic_task_returns_empty(self):
        decomposer = SequentialDecomposer()
        task = Task(name="Simple", description="Do it")
        
        steps = decomposer.decompose(task)
        assert len(steps) == 0


# =============================================================================
# DependencyAnalyzer Tests
# =============================================================================


class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer."""

    def test_analyze_single_task(self):
        analyzer = DependencyAnalyzer()
        tasks = [Task(name="Single")]
        
        deps = analyzer.analyze(tasks)
        assert deps[tasks[0].id] == []

    def test_analyze_multiple_tasks(self):
        analyzer = DependencyAnalyzer()
        tasks = [
            Task(name="Task1"),
            Task(name="Task2"),
            Task(name="Task3"),
        ]
        
        deps = analyzer.analyze(tasks)
        assert len(deps) == 3

    def test_apply_dependencies(self):
        analyzer = DependencyAnalyzer()
        tasks = [Task(name="T1"), Task(name="T2")]
        deps = {tasks[0].id: [], tasks[1].id: [tasks[0].id]}
        
        analyzer.apply_dependencies(tasks, deps)
        assert tasks[1].dependencies == [tasks[0].id]

    def test_detect_cycles_no_cycle(self):
        analyzer = DependencyAnalyzer()
        t1 = Task(name="T1")
        t2 = Task(name="T2", dependencies=[t1.id])
        
        cycles = analyzer.detect_cycles([t1, t2])
        assert len(cycles) == 0

    def test_topological_sort(self):
        analyzer = DependencyAnalyzer()
        t1 = Task(name="T1")
        t2 = Task(name="T2", dependencies=[t1.id])
        t3 = Task(name="T3", dependencies=[t2.id])
        
        sorted_tasks = analyzer.topological_sort([t3, t1, t2])
        assert sorted_tasks[0].id == t1.id
        assert sorted_tasks[-1].id == t3.id

    def test_topological_sort_cycle_raises(self):
        analyzer = DependencyAnalyzer()
        t1 = Task(name="T1")
        t2 = Task(name="T2")
        t1.dependencies = [t2.id]
        t2.dependencies = [t1.id]
        
        with pytest.raises(ValueError, match="Circular"):
            analyzer.topological_sort([t1, t2])

    def test_get_ready_tasks(self):
        analyzer = DependencyAnalyzer()
        t1 = Task(name="T1", status=TaskStatus.COMPLETED)
        t2 = Task(name="T2", dependencies=[t1.id])
        t3 = Task(name="T3")
        
        ready = analyzer.get_ready_tasks([t1, t2, t3], {t1.id})
        assert len(ready) == 2


# =============================================================================
# TaskDecomposer Tests
# =============================================================================


class TestTaskDecomposer:
    """Tests for TaskDecomposer main interface."""

    def test_decompose_basic(self):
        decomposer = TaskDecomposer()
        task = Task(
            name="Build Feature",
            description="Implement a new feature with multiple components and testing"
        )
        
        subtasks = decomposer.decompose(task)
        assert len(subtasks) > 0

    def test_decompose_and_attach(self):
        decomposer = TaskDecomposer()
        task = Task(
            name="Project",
            description="Complete project with design, implementation, and testing phases"
        )
        
        result = decomposer.decompose_and_attach(task)
        assert len(result.subtasks) > 0

    def test_validate_decomposition_valid(self):
        decomposer = TaskDecomposer()
        task = Task(name="Root")
        task.add_subtask(Task(name="Child1"))
        task.add_subtask(Task(name="Child2"))
        
        is_valid, issues = decomposer.validate_decomposition(task)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_decomposition_no_subtasks(self):
        decomposer = TaskDecomposer()
        task = Task(name="Empty")
        
        is_valid, issues = decomposer.validate_decomposition(task)
        assert is_valid is False

    def test_set_strategy(self):
        decomposer = TaskDecomposer()
        new_strategy = SequentialDecomposer()
        decomposer.set_strategy(new_strategy)
        
        assert decomposer.strategy == new_strategy


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_task(self):
        task = create_task("Test", "Description", priority="high")
        assert task.name == "Test"
        assert task.priority == TaskPriority.HIGH

    def test_create_task_with_type(self):
        task = create_task("Test", task_type="implementation")
        assert task.task_type == TaskType.IMPLEMENTATION

    def test_create_decomposer_hierarchical(self):
        decomposer = create_decomposer("hierarchical")
        assert isinstance(decomposer.strategy, HierarchicalDecomposer)

    def test_create_decomposer_sequential(self):
        decomposer = create_decomposer("sequential")
        assert isinstance(decomposer.strategy, SequentialDecomposer)

    def test_create_decomposer_invalid(self):
        with pytest.raises(ValueError):
            create_decomposer("invalid")

    def test_format_task_tree(self):
        root = Task(name="Root")
        child = Task(name="Child")
        root.add_subtask(child)
        
        output = format_task_tree(root)
        assert "Root" in output
        assert "Child" in output
