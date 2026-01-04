"""
Unit tests for ReAct module.
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

import pytest
from src.react import (
    Thought,
    Action,
    Observation,
    ReActStep,
    ReActTrace,
    SimpleTool,
    ReActPromptBuilder,
    ReActParser,
    ReActAgent,
)


class TestThought:
    """Tests for Thought dataclass."""

    def test_creation(self):
        thought = Thought(content="I need to search", step_number=1)
        assert thought.content == "I need to search"
        assert thought.step_number == 1

    def test_to_dict(self):
        thought = Thought(content="Test", step_number=1)
        d = thought.to_dict()
        assert d["content"] == "Test"
        assert d["step"] == 1


class TestAction:
    """Tests for Action dataclass."""

    def test_creation(self):
        action = Action(name="search", input={"query": "test"}, step_number=1)
        assert action.name == "search"
        assert action.input == {"query": "test"}

    def test_to_dict(self):
        action = Action(name="calc", input="2+2", step_number=1)
        d = action.to_dict()
        assert d["name"] == "calc"


class TestObservation:
    """Tests for Observation dataclass."""

    def test_creation(self):
        obs = Observation(content="Result: 4", step_number=1, success=True)
        assert obs.content == "Result: 4"
        assert obs.success is True

    def test_creation_with_error(self):
        obs = Observation(content="Error", step_number=1, success=False)
        assert obs.success is False


class TestReActStep:
    """Tests for ReActStep dataclass."""

    def test_creation(self):
        thought = Thought(content="Think", step_number=1)
        action = Action(name="act", input="input", step_number=1)
        obs = Observation(content="Result", step_number=1)
        
        step = ReActStep(
            thought=thought,
            action=action,
            observation=obs,
            step_number=1
        )
        assert step.step_number == 1
        assert step.thought.content == "Think"

    def test_to_dict(self):
        step = ReActStep(
            thought=Thought(content="T", step_number=1),
            action=Action(name="A", input="I", step_number=1),
            observation=Observation(content="O", step_number=1),
            step_number=1
        )
        d = step.to_dict()
        assert "thought" in d
        assert "action" in d
        assert "observation" in d


class TestReActTrace:
    """Tests for ReActTrace dataclass."""

    def test_creation(self):
        trace = ReActTrace(question="What is 2+2?")
        assert trace.question == "What is 2+2?"
        assert len(trace.steps) == 0

    def test_add_step(self):
        trace = ReActTrace(question="Q")
        step = ReActStep(
            thought=Thought(content="T", step_number=1),
            action=Action(name="A", input="I", step_number=1),
            observation=Observation(content="O", step_number=1),
            step_number=1
        )
        trace.add_step(step)
        assert len(trace.steps) == 1


class TestSimpleTool:
    """Tests for SimpleTool."""

    def test_creation(self):
        tool = SimpleTool(
            name="calculator",
            description="Calculate math",
            func=lambda x: str(eval(x))
        )
        assert tool.name == "calculator"

    def test_execute(self):
        tool = SimpleTool(
            name="calc",
            description="Calc",
            func=lambda x: str(int(x) * 2)
        )
        result = tool.execute("5")
        assert result == "10"

    def test_execute_with_dict(self):
        tool = SimpleTool(
            name="greet",
            description="Greet",
            func=lambda name: f"Hello, {name}!"
        )
        result = tool.execute("World")
        assert result == "Hello, World!"


class TestReActPromptBuilder:
    """Tests for ReActPromptBuilder."""

    def test_build_with_tools(self):
        tools = [
            SimpleTool(name="search", description="Search web", func=lambda x: x),
            SimpleTool(name="calc", description="Calculate", func=lambda x: x),
        ]
        builder = ReActPromptBuilder(tools=tools)
        prompt = builder.build("What is 2+2?")
        assert "search" in prompt
        assert "calc" in prompt
        assert "What is 2+2?" in prompt

    def test_build_without_tools(self):
        builder = ReActPromptBuilder(tools=[])
        prompt = builder.build("What is 2+2?")
        assert "What is 2+2?" in prompt


class TestReActParser:
    """Tests for ReActParser."""

    def test_parse_thought(self):
        parser = ReActParser()
        text = "Thought: I need to search for information"
        thought = parser.parse_thought(text)
        assert thought is not None
        assert "search" in thought

    def test_parse_action(self):
        parser = ReActParser()
        text = """Action: search
Action Input: python tutorial"""
        result = parser.parse_action(text)
        assert result is not None
        action_name, action_input = result
        assert action_name == "search"
        assert action_input == "python tutorial"

    def test_parse_final_answer(self):
        parser = ReActParser()
        text = "Final Answer: The answer is 42"
        answer = parser.parse_final_answer(text)
        assert answer == "The answer is 42"

    def test_is_final_answer(self):
        parser = ReActParser()
        assert parser.is_final_answer("Final Answer: Yes")
        assert not parser.is_final_answer("Thought: I need to think")


class TestReActAgent:
    """Tests for ReActAgent."""

    def test_creation(self):
        tools = [
            SimpleTool(name="test", description="Test", func=lambda x: x)
        ]
        agent = ReActAgent(tools=tools, max_iterations=5)
        assert len(agent._tools) == 1
        assert agent._max_iterations == 5

    def test_run_without_llm(self):
        tools = [
            SimpleTool(name="test", description="Test", func=lambda x: x)
        ]
        agent = ReActAgent(tools=tools)
        result = agent.run("Test question")
        assert result.question == "Test question"
