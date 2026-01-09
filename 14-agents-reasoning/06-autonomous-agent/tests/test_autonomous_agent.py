"""
Unit tests for the 06-autonomous-agent module.

Tests cover:
- Goal Manager: Goal creation, decomposition, priority queue
- Action Executor: Tool, code, file actions
- Self Reflection: Success/failure analysis, learning memory
- Agent Loop: OODA cycle, termination conditions
- Autonomous Agent: Integration tests

Author: AI-Practices
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goal_manager import (
    Goal,
    GoalStatus,
    GoalPriority,
    GoalManager,
    GoalPriorityQueue,
    LLMGoalDecomposer,
    RuleBasedDecomposer,
    CompletionChecker,
)
from action_executor import (
    Action,
    ActionType,
    ActionStatus,
    ActionResult,
    ActionExecutor,
    ToolAction,
    CodeAction,
    ThinkAction,
    create_action,
)
from self_reflection import (
    Reflection,
    ReflectionType,
    SelfReflector,
    SuccessAnalyzer,
    FailureAnalyzer,
    StrategyAdjuster,
    LearningMemory,
)
from agent_loop import (
    LoopState,
    LoopConfig,
    LoopContext,
    TerminationReason,
    TerminationChecker,
    SimpleAgentLoop,
)


# =============================================================================
# Goal Manager Tests
# =============================================================================

class TestGoal:
    """Tests for Goal dataclass."""

    def test_goal_creation(self):
        goal = Goal(description="Test goal")
        assert goal.description == "Test goal"
        assert goal.status == GoalStatus.PENDING
        assert goal.priority == GoalPriority.MEDIUM
        assert goal.goal_id.startswith("goal_")

    def test_goal_is_leaf(self):
        goal = Goal(description="Leaf goal")
        assert goal.is_leaf() is True
        goal.sub_goals.append("sub_1")
        assert goal.is_leaf() is False

    def test_goal_mark_completed(self):
        goal = Goal(description="Test")
        goal.mark_completed()
        assert goal.status == GoalStatus.COMPLETED
        assert goal.completed_at is not None

    def test_goal_mark_failed_with_retry(self):
        goal = Goal(description="Test", max_attempts=3)
        goal.attempt_count = 1
        goal.mark_failed(can_retry=True)
        assert goal.status == GoalStatus.PENDING

    def test_goal_mark_failed_no_retry(self):
        goal = Goal(description="Test", max_attempts=3)
        goal.attempt_count = 3
        goal.mark_failed(can_retry=True)
        assert goal.status == GoalStatus.FAILED

    def test_goal_to_dict(self):
        goal = Goal(description="Test")
        d = goal.to_dict()
        assert d["description"] == "Test"
        assert "goal_id" in d
        assert d["status"] == "pending"


class TestGoalPriorityQueue:
    """Tests for GoalPriorityQueue."""

    def test_push_and_pop(self):
        queue = GoalPriorityQueue()
        goal = Goal(description="Test")
        queue.push(goal)
        assert len(queue) == 1
        popped = queue.pop()
        assert popped.goal_id == goal.goal_id
        assert len(queue) == 0

    def test_priority_ordering(self):
        queue = GoalPriorityQueue()
        low = Goal(description="Low", priority=GoalPriority.LOW)
        high = Goal(description="High", priority=GoalPriority.HIGH)
        queue.push(low)
        queue.push(high)
        first = queue.pop()
        assert first.priority == GoalPriority.HIGH

    def test_peek(self):
        queue = GoalPriorityQueue()
        goal = Goal(description="Test")
        queue.push(goal)
        peeked = queue.peek()
        assert peeked.goal_id == goal.goal_id
        assert len(queue) == 1

    def test_remove(self):
        queue = GoalPriorityQueue()
        goal = Goal(description="Test")
        queue.push(goal)
        removed = queue.remove(goal.goal_id)
        assert removed is True
        assert len(queue) == 0


class TestGoalDecomposer:
    """Tests for goal decomposition."""

    def test_llm_decomposer(self):
        decomposer = LLMGoalDecomposer()
        goal = Goal(description="Build a website")
        sub_goals = decomposer.decompose(goal)
        assert len(sub_goals) > 0
        assert all(g.parent_id == goal.goal_id for g in sub_goals)

    def test_rule_based_decomposer_write(self):
        decomposer = RuleBasedDecomposer()
        goal = Goal(description="Write a report")
        sub_goals = decomposer.decompose(goal)
        assert len(sub_goals) == 4
        assert "Research" in sub_goals[0].description

    def test_rule_based_decomposer_build(self):
        decomposer = RuleBasedDecomposer()
        goal = Goal(description="Build an app")
        sub_goals = decomposer.decompose(goal)
        assert len(sub_goals) == 4
        assert "Design" in sub_goals[0].description


class TestCompletionChecker:
    """Tests for CompletionChecker."""

    def test_keyword_completion(self):
        checker = CompletionChecker()
        goal = Goal(description="Test")
        is_done, reason = checker.check_completion(goal, "Task completed successfully")
        assert is_done is True
        assert "completed" in reason.lower()

    def test_no_completion(self):
        checker = CompletionChecker()
        goal = Goal(description="Test")
        is_done, reason = checker.check_completion(goal, "Still working on it")
        assert is_done is False


class TestGoalManager:
    """Tests for GoalManager."""

    def test_add_goal(self):
        manager = GoalManager()
        goal = manager.add_goal("Test goal")
        assert goal.description == "Test goal"
        assert manager.get_goal(goal.goal_id) is not None

    def test_decompose_goal(self):
        manager = GoalManager()
        goal = manager.add_goal("Build something")
        sub_goals = manager.decompose_goal(goal.goal_id)
        assert len(sub_goals) > 0
        assert len(goal.sub_goals) > 0

    def test_get_next_goal(self):
        manager = GoalManager()
        manager.add_goal("Goal 1")
        next_goal = manager.get_next_goal()
        assert next_goal is not None
        assert next_goal.description == "Goal 1"

    def test_complete_goal(self):
        manager = GoalManager()
        goal = manager.add_goal("Test")
        manager.start_goal(goal.goal_id)
        is_done, _ = manager.complete_goal(goal.goal_id, "Done successfully")
        assert is_done is True
        assert goal.status == GoalStatus.COMPLETED

    def test_get_progress(self):
        manager = GoalManager()
        manager.add_goal("Goal 1")
        manager.add_goal("Goal 2")
        progress = manager.get_progress()
        assert progress["total"] == 2
        assert progress["completed"] == 0

    def test_reset(self):
        manager = GoalManager()
        manager.add_goal("Test")
        manager.reset()
        assert manager.get_progress()["total"] == 0


# =============================================================================
# Action Executor Tests
# =============================================================================

class TestAction:
    """Tests for Action dataclass."""

    def test_action_creation(self):
        action = Action(ActionType.TOOL, "test_tool", {"param": "value"})
        assert action.action_type == ActionType.TOOL
        assert action.name == "test_tool"
        assert action.parameters["param"] == "value"

    def test_create_action_factory(self):
        action = create_action("tool", "my_tool", key="value")
        assert action.action_type == ActionType.TOOL
        assert action.parameters["key"] == "value"


class TestToolAction:
    """Tests for ToolAction handler."""

    def test_register_and_execute(self):
        handler = ToolAction()
        handler.register_tool("add", lambda a, b: a + b)
        action = Action(ActionType.TOOL, "add", {"a": 2, "b": 3})
        result = handler.execute(action)
        assert result.is_success
        assert result.output == 5

    def test_tool_not_found(self):
        handler = ToolAction()
        action = Action(ActionType.TOOL, "unknown", {})
        result = handler.execute(action)
        assert result.status == ActionStatus.FAILED
        assert "not found" in result.error.lower()


class TestCodeAction:
    """Tests for CodeAction handler."""

    def test_execute_simple_code(self):
        handler = CodeAction()
        action = Action(ActionType.CODE, "exec", {"code": "result = 2 + 2"})
        result = handler.execute(action)
        assert result.is_success
        assert result.output == 4

    def test_syntax_error(self):
        handler = CodeAction()
        action = Action(ActionType.CODE, "exec", {"code": "def broken("})
        result = handler.execute(action)
        assert result.status == ActionStatus.FAILED
        assert "syntax" in result.error.lower()


class TestThinkAction:
    """Tests for ThinkAction handler."""

    def test_think_action(self):
        handler = ThinkAction()
        action = Action(ActionType.THINK, "think", {"thought": "Analyzing..."})
        result = handler.execute(action)
        assert result.is_success
        assert result.output == "Analyzing..."


class TestActionExecutor:
    """Tests for ActionExecutor."""

    def test_execute_tool(self):
        executor = ActionExecutor()
        for h in executor.registry._handlers:
            if isinstance(h, ToolAction):
                h.register_tool("multiply", lambda x, y: x * y)
        action = Action(ActionType.TOOL, "multiply", {"x": 3, "y": 4})
        result = executor.execute(action)
        assert result.is_success
        assert result.output == 12

    def test_execute_think(self):
        executor = ActionExecutor()
        action = Action(ActionType.THINK, "ponder", {"thought": "Hmm..."})
        result = executor.execute(action)
        assert result.is_success

    def test_get_stats(self):
        executor = ActionExecutor()
        action = Action(ActionType.THINK, "test", {"thought": "test"})
        executor.execute(action)
        stats = executor.get_stats()
        assert stats["total"] == 1
        assert stats["success"] == 1

    def test_clear_history(self):
        executor = ActionExecutor()
        action = Action(ActionType.THINK, "test", {"thought": "test"})
        executor.execute(action)
        executor.clear_history()
        assert len(executor.get_history()) == 0


# =============================================================================
# Self Reflection Tests
# =============================================================================

class TestReflection:
    """Tests for Reflection dataclass."""

    def test_reflection_creation(self):
        reflection = Reflection(
            reflection_type=ReflectionType.SUCCESS,
            content="Task completed well",
        )
        assert reflection.reflection_type == ReflectionType.SUCCESS
        assert reflection.confidence == 0.5

    def test_reflection_to_dict(self):
        reflection = Reflection(
            reflection_type=ReflectionType.FAILURE,
            content="Task failed",
            insights=["Need more data"],
        )
        d = reflection.to_dict()
        assert d["reflection_type"] == "failure"
        assert len(d["insights"]) == 1


class TestSuccessAnalyzer:
    """Tests for SuccessAnalyzer."""

    def test_analyze_success(self):
        analyzer = SuccessAnalyzer()
        context = {"goal": "Write code", "actions": ["code"]}
        reflection = analyzer.reflect(context, "Code written successfully")
        assert reflection.reflection_type == ReflectionType.SUCCESS
        assert reflection.confidence > 0.5


class TestFailureAnalyzer:
    """Tests for FailureAnalyzer."""

    def test_analyze_failure(self):
        analyzer = FailureAnalyzer()
        context = {"goal": "Read file", "actions": ["read"], "error": "File not found"}
        reflection = analyzer.reflect(context, "Failed to read")
        assert reflection.reflection_type == ReflectionType.FAILURE
        assert len(reflection.suggested_improvements) > 0

    def test_identify_timeout(self):
        analyzer = FailureAnalyzer()
        context = {"goal": "Long task", "actions": [], "error": "Operation timeout"}
        reflection = analyzer.reflect(context, "Timed out")
        assert "timeout" in reflection.metadata.get("root_cause", "")


class TestStrategyAdjuster:
    """Tests for StrategyAdjuster."""

    def test_register_strategy(self):
        adjuster = StrategyAdjuster()
        adjuster.register_strategy("fast", "Quick approach")
        assert "fast" in adjuster.strategies

    def test_record_outcome(self):
        adjuster = StrategyAdjuster()
        adjuster.record_outcome("strategy_a", 1.0)
        adjuster.record_outcome("strategy_a", 0.5)
        stats = adjuster.get_strategy_stats()
        assert stats["strategy_a"]["selections"] == 2
        assert stats["strategy_a"]["avg_reward"] == 0.75

    def test_select_strategy_ucb(self):
        adjuster = StrategyAdjuster()
        adjuster.register_strategy("a")
        adjuster.register_strategy("b")
        selected = adjuster.select_strategy()
        assert selected in ["a", "b"]


class TestLearningMemory:
    """Tests for LearningMemory."""

    def test_add_entry(self):
        memory = LearningMemory()
        memory.add_entry("situation", "action", "outcome", 1.0)
        assert memory.get_stats()["total"] == 1

    def test_find_similar(self):
        memory = LearningMemory()
        memory.add_entry("write python code", "execute", "success", 1.0)
        memory.add_entry("write python script", "run", "done", 0.8)
        similar = memory.find_similar("write python function")
        assert len(similar) > 0

    def test_get_best_action(self):
        memory = LearningMemory()
        memory.add_entry("task", "action_a", "ok", 0.5)
        memory.add_entry("task", "action_b", "great", 1.0)
        best = memory.get_best_action("task")
        assert best is not None
        assert best[0] == "action_b"

    def test_max_entries_limit(self):
        memory = LearningMemory(max_entries=5)
        for i in range(10):
            memory.add_entry(f"sit_{i}", f"act_{i}", "out", 0.5)
        assert memory.get_stats()["total"] == 5


class TestSelfReflector:
    """Tests for SelfReflector."""

    def test_reflect_on_action_success(self):
        reflector = SelfReflector()
        reflection = reflector.reflect_on_action(
            goal="Test goal",
            action="test_action",
            result="Completed successfully",
            success=True,
        )
        assert reflection.reflection_type == ReflectionType.SUCCESS

    def test_reflect_on_action_failure(self):
        reflector = SelfReflector()
        reflection = reflector.reflect_on_action(
            goal="Test goal",
            action="test_action",
            result="Error occurred",
            success=False,
        )
        assert reflection.reflection_type == ReflectionType.FAILURE

    def test_get_insights_summary(self):
        reflector = SelfReflector()
        reflector.reflect_on_action("g1", "a1", "ok", True)
        reflector.reflect_on_action("g2", "a2", "fail", False)
        summary = reflector.get_insights_summary()
        assert summary["total_reflections"] == 2
        assert summary["successes"] == 1
        assert summary["failures"] == 1


# =============================================================================
# Agent Loop Tests
# =============================================================================

class TestLoopConfig:
    """Tests for LoopConfig."""

    def test_default_config(self):
        config = LoopConfig()
        assert config.max_iterations == 100
        assert config.enable_reflection is True

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            LoopConfig(max_iterations=0)


class TestLoopContext:
    """Tests for LoopContext."""

    def test_record_success(self):
        ctx = LoopContext()
        ctx.consecutive_failures = 3
        ctx.record_success()
        assert ctx.total_successes == 1
        assert ctx.consecutive_failures == 0

    def test_record_failure(self):
        ctx = LoopContext()
        ctx.record_failure()
        ctx.record_failure()
        assert ctx.total_failures == 2
        assert ctx.consecutive_failures == 2


class TestTerminationChecker:
    """Tests for TerminationChecker."""

    def test_max_iterations(self):
        config = LoopConfig(max_iterations=10)
        checker = TerminationChecker(config)
        ctx = LoopContext(iteration=10)
        reason = checker.should_terminate(ctx)
        assert reason == TerminationReason.MAX_ITERATIONS

    def test_stuck_detection(self):
        config = LoopConfig(stuck_threshold=3)
        checker = TerminationChecker(config)
        ctx = LoopContext(consecutive_failures=3)
        ctx.start_time = datetime.utcnow()
        reason = checker.should_terminate(ctx)
        assert reason == TerminationReason.STUCK


class TestSimpleAgentLoop:
    """Tests for SimpleAgentLoop."""

    def test_basic_loop(self):
        iterations = [0]
        def act_fn(d):
            iterations[0] += 1
            return (True, "done")
        
        config = LoopConfig(max_iterations=3, pause_between_iterations=0)
        loop = SimpleAgentLoop(
            config=config,
            act_fn=act_fn,
        )
        ctx = loop.run()
        assert ctx.iteration == 3
        assert ctx.termination_reason == TerminationReason.MAX_ITERATIONS

    def test_goal_achieved(self):
        achieved = [False]
        def goal_check():
            return achieved[0]
        def act_fn(d):
            achieved[0] = True
            return (True, "done")
        
        config = LoopConfig(max_iterations=10, pause_between_iterations=0)
        loop = SimpleAgentLoop(
            config=config,
            act_fn=act_fn,
            goal_check_fn=goal_check,
        )
        ctx = loop.run()
        assert ctx.termination_reason == TerminationReason.GOAL_ACHIEVED


# =============================================================================
# Integration Tests
# =============================================================================

class TestAutonomousAgentIntegration:
    """Integration tests for AutonomousAgent."""

    def test_agent_creation(self):
        from autonomous_agent import AutonomousAgent, AgentConfig
        config = AgentConfig(name="TestBot")
        agent = AutonomousAgent(config=config)
        assert agent.config.name == "TestBot"

    def test_set_objective(self):
        from autonomous_agent import AutonomousAgent
        agent = AutonomousAgent()
        goal = agent.set_objective("Test objective")
        assert goal.description == "Test objective"

    def test_get_status(self):
        from autonomous_agent import AutonomousAgent
        agent = AutonomousAgent()
        agent.set_objective("Test")
        status = agent.get_status()
        assert "goals" in status
        assert status["goals"]["total"] > 0

    def test_reset(self):
        from autonomous_agent import AutonomousAgent
        agent = AutonomousAgent()
        agent.set_objective("Test")
        agent.reset()
        assert agent.get_progress()["total"] == 0

    def test_agent_builder(self):
        from autonomous_agent import AgentBuilder, GoalPriority
        agent = (
            AgentBuilder("BuilderBot")
            .with_config(max_iterations=5)
            .with_constraint("Be helpful")
            .with_objective("Test task")
            .build()
        )
        assert agent.config.name == "BuilderBot"
        assert agent.config.max_iterations == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
