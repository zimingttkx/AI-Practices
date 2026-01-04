"""
ReAct: Synergizing Reasoning and Acting in Language Models

Core Idea:
    ReAct (Reasoning + Acting) interleaves reasoning traces with task-specific
    actions, allowing LLMs to dynamically reason about and interact with
    external environments. This creates a synergy where reasoning helps guide
    actions, and action outcomes inform further reasoning.

Mathematical Foundation:
    ReAct models the agent as a policy over a state space:

    $$\pi(a_t | s_t, h_t) = P(a_t | o_1, a_1, ..., o_{t-1}, a_{t-1}, o_t, r_t)$$

    where:
    - $o_t$ is the observation at time $t$
    - $a_t$ is the action at time $t$
    - $r_t$ is the reasoning trace at time $t$
    - $h_t = (o_1, a_1, r_1, ..., o_t, r_t)$ is the history

    The key insight is that explicit reasoning $r_t$ provides:
    1. Interpretable decision-making process
    2. Working memory for multi-step tasks
    3. Error recovery through self-reflection

Problem Statement:
    Pure reasoning (CoT) lacks grounding in real-world information.
    Pure acting (tool use) lacks strategic planning and error handling.

    ReAct combines both:
    - Thought: Internal reasoning about the current situation
    - Action: External interaction with tools/environment
    - Observation: Feedback from the environment

Algorithm:
    ```
    while not done:
        thought = generate_thought(history)
        action = decide_action(thought, available_tools)
        observation = execute_action(action)
        history.append(thought, action, observation)
        if is_final_answer(thought):
            return extract_answer(thought)
    ```

Comparison with Other Approaches:
    | Approach    | Reasoning | Acting | Grounding | Interpretability |
    |-------------|-----------|--------|-----------|------------------|
    | Standard    | No        | No     | No        | Low              |
    | CoT         | Yes       | No     | No        | High             |
    | Tool Use    | No        | Yes    | Yes       | Low              |
    | ReAct       | Yes       | Yes    | Yes       | High             |

References:
    - Yao et al. (2022): "ReAct: Synergizing Reasoning and Acting in LLMs"
    - https://arxiv.org/abs/2210.03629
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)


class StepType(str, Enum):
    """Types of steps in a ReAct trace.

    Attributes:
        THOUGHT: Internal reasoning step.
        ACTION: External action/tool invocation.
        OBSERVATION: Result from action execution.
        FINAL: Final answer step.
    """
    THOUGHT: Final[str] = "thought"
    ACTION: Final[str] = "action"
    OBSERVATION: Final[str] = "observation"
    FINAL: Final[str] = "final"


@dataclass(frozen=True, slots=True)
class Thought:
    """A reasoning step in the ReAct loop.

    Core Idea:
        Represents the agent's internal reasoning about the current state,
        what information is needed, and what action to take next.

    Attributes:
        content: The reasoning text.
        step_number: Position in the ReAct trace.

    Example:
        >>> thought = Thought(
        ...     content="I need to find the population of Tokyo to answer this question.",
        ...     step_number=1
        ... )
    """
    content: str
    step_number: int = 1

    def __str__(self) -> str:
        return f"Thought {self.step_number}: {self.content}"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "thought", "content": self.content, "step": self.step_number}


@dataclass(frozen=True, slots=True)
class Action:
    """An action step in the ReAct loop.

    Core Idea:
        Represents a tool invocation or external action the agent decides
        to take based on its reasoning.

    Attributes:
        name: The action/tool name.
        input: The input to the action (can be string or dict).
        step_number: Position in the ReAct trace.

    Example:
        >>> action = Action(
        ...     name="search",
        ...     input="Tokyo population 2024",
        ...     step_number=1
        ... )
    """
    name: str
    input: Union[str, Dict[str, Any]]
    step_number: int = 1

    def __str__(self) -> str:
        return f"Action {self.step_number}: {self.name}[{self.input}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "action",
            "name": self.name,
            "input": self.input,
            "step": self.step_number,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    """An observation step in the ReAct loop.

    Core Idea:
        Represents the result/feedback from executing an action,
        which informs the next reasoning step.

    Attributes:
        content: The observation text (action result).
        step_number: Position in the ReAct trace.
        success: Whether the action succeeded.

    Example:
        >>> obs = Observation(
        ...     content="Tokyo has a population of approximately 14 million.",
        ...     step_number=1,
        ...     success=True
        ... )
    """
    content: str
    step_number: int = 1
    success: bool = True

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"Observation {self.step_number} {status}: {self.content}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "observation",
            "content": self.content,
            "step": self.step_number,
            "success": self.success,
        }


@dataclass
class ReActStep:
    """A complete Thought-Action-Observation triple.

    Core Idea:
        Groups related thought, action, and observation into a single
        logical step for easier manipulation and display.

    Attributes:
        thought: The reasoning for this step.
        action: The action taken (None if final answer).
        observation: The result of the action (None if final answer).
        step_number: Position in the ReAct trace.
    """
    thought: Thought
    action: Optional[Action] = None
    observation: Optional[Observation] = None
    step_number: int = 1

    @property
    def is_final(self) -> bool:
        """Check if this is a final answer step (no action)."""
        return self.action is None

    def format(self, include_observation: bool = True) -> str:
        """Format the step for display or prompt construction."""
        parts = [str(self.thought)]
        if self.action:
            parts.append(str(self.action))
        if include_observation and self.observation:
            parts.append(str(self.observation))
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "step_number": self.step_number,
            "thought": self.thought.to_dict(),
        }
        if self.action:
            result["action"] = self.action.to_dict()
        if self.observation:
            result["observation"] = self.observation.to_dict()
        return result


@dataclass
class ReActTrace:
    """Complete trace of a ReAct reasoning process.

    Attributes:
        question: The original question/task.
        steps: List of ReActStep objects.
        final_answer: The final answer (if reached).
        success: Whether the task was completed successfully.
        metadata: Additional information (tokens, latency, etc.).
    """
    question: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        """Number of reasoning steps."""
        return len(self.steps)

    @property
    def num_actions(self) -> int:
        """Number of actions taken."""
        return sum(1 for s in self.steps if s.action is not None)

    def add_step(self, step: ReActStep) -> None:
        """Add a step to the trace."""
        self.steps.append(step)

    def get_history_text(self) -> str:
        """Get formatted history for prompt construction."""
        return "\n\n".join(s.format() for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "question": self.question,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "success": self.success,
            "num_steps": self.num_steps,
            "num_actions": self.num_actions,
            "metadata": self.metadata,
        }


class ToolInterface(Protocol):
    """Protocol for tools that can be used in ReAct."""

    @property
    def name(self) -> str:
        """Tool name."""
        ...

    @property
    def description(self) -> str:
        """Tool description for the LLM."""
        ...

    def execute(self, input: Union[str, Dict[str, Any]]) -> str:
        """Execute the tool with given input."""
        ...


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class SimpleTool:
    """Simple tool implementation for ReAct.

    Example:
        >>> def search(query: str) -> str:
        ...     return f"Results for: {query}"
        >>> tool = SimpleTool("search", "Search the web", search)
    """
    name: str
    description: str
    func: Callable[[Any], str]

    def execute(self, input: Union[str, Dict[str, Any]]) -> str:
        """Execute the tool."""
        if isinstance(input, dict):
            return self.func(**input)
        return self.func(input)


class ReActPromptBuilder:
    """Builder for constructing ReAct prompts.

    Core Idea:
        Provides templates and methods for building ReAct-style prompts
        with tool descriptions, examples, and history.
    """

    SYSTEM_TEMPLATE: Final[str] = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
{history}"""

    THOUGHT_ACTION_TEMPLATE: Final[str] = """Thought: {thought}
Action: {action}
Action Input: {action_input}"""

    def __init__(self, tools: Optional[List[ToolInterface]] = None) -> None:
        """Initialize the prompt builder.

        Args:
            tools: List of available tools.
        """
        self._tools = tools or []

    def add_tool(self, tool: ToolInterface) -> "ReActPromptBuilder":
        """Add a tool to the builder."""
        self._tools.append(tool)
        return self

    def set_tools(self, tools: List[ToolInterface]) -> "ReActPromptBuilder":
        """Set the tools list."""
        self._tools = tools
        return self

    def _format_tools(self) -> str:
        """Format tool descriptions for the prompt."""
        return "\n".join(
            f"{tool.name}: {tool.description}"
            for tool in self._tools
        )

    def _get_tool_names(self) -> str:
        """Get comma-separated tool names."""
        return ", ".join(tool.name for tool in self._tools)

    def build(self, question: str, history: str = "") -> str:
        """Build the complete ReAct prompt.

        Args:
            question: The question to answer.
            history: Previous thought-action-observation history.

        Returns:
            Complete prompt string.
        """
        history_section = f"\n{history}" if history else ""
        return self.SYSTEM_TEMPLATE.format(
            tools=self._format_tools(),
            tool_names=self._get_tool_names(),
            question=question,
            history=history_section,
        )

    def build_continuation(self, observation: str) -> str:
        """Build a continuation prompt after an observation.

        Args:
            observation: The observation from the last action.

        Returns:
            Continuation prompt.
        """
        return f"Observation: {observation}\nThought:"


class ReActParser:
    """Parser for extracting ReAct components from LLM output.

    Core Idea:
        Parses LLM-generated text to extract thoughts, actions, and final answers
        using regex patterns that match the ReAct format.
    """

    # Regex patterns for parsing ReAct output
    THOUGHT_PATTERN: Final[re.Pattern] = re.compile(
        r"Thought\s*\d*\s*:?\s*(.+?)(?=Action|Final Answer|$)",
        re.IGNORECASE | re.DOTALL
    )
    ACTION_PATTERN: Final[re.Pattern] = re.compile(
        r"Action\s*\d*\s*:?\s*(.+?)(?=Action Input|$)",
        re.IGNORECASE | re.DOTALL
    )
    ACTION_INPUT_PATTERN: Final[re.Pattern] = re.compile(
        r"Action Input\s*\d*\s*:?\s*(.+?)(?=Observation|Thought|$)",
        re.IGNORECASE | re.DOTALL
    )
    FINAL_ANSWER_PATTERN: Final[re.Pattern] = re.compile(
        r"Final Answer\s*:?\s*(.+?)$",
        re.IGNORECASE | re.DOTALL
    )

    def parse_thought(self, text: str) -> Optional[str]:
        """Extract thought from text."""
        match = self.THOUGHT_PATTERN.search(text)
        return match.group(1).strip() if match else None

    def parse_action(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract action name and input from text.

        Returns:
            Tuple of (action_name, action_input) or None.
        """
        action_match = self.ACTION_PATTERN.search(text)
        input_match = self.ACTION_INPUT_PATTERN.search(text)

        if action_match and input_match:
            return (
                action_match.group(1).strip(),
                input_match.group(1).strip(),
            )
        return None

    def parse_final_answer(self, text: str) -> Optional[str]:
        """Extract final answer from text."""
        match = self.FINAL_ANSWER_PATTERN.search(text)
        return match.group(1).strip() if match else None

    def is_final_answer(self, text: str) -> bool:
        """Check if text contains a final answer."""
        return "final answer" in text.lower()

    def parse_step(self, text: str, step_number: int = 1) -> Optional[ReActStep]:
        """Parse a complete ReAct step from text.

        Args:
            text: LLM output text.
            step_number: Current step number.

        Returns:
            ReActStep if parsing successful, None otherwise.
        """
        thought_text = self.parse_thought(text)
        if not thought_text:
            return None

        thought = Thought(content=thought_text, step_number=step_number)

        # Check for final answer
        final_answer = self.parse_final_answer(text)
        if final_answer:
            return ReActStep(thought=thought, step_number=step_number)

        # Parse action
        action_result = self.parse_action(text)
        if action_result:
            action_name, action_input = action_result
            action = Action(
                name=action_name,
                input=action_input,
                step_number=step_number,
            )
            return ReActStep(thought=thought, action=action, step_number=step_number)

        return ReActStep(thought=thought, step_number=step_number)


class ReActAgent:
    """ReAct Agent: Reasoning and Acting in an interleaved manner.

    Core Idea:
        Implements the ReAct loop that alternates between reasoning (Thought),
        acting (Action), and observing (Observation) until a final answer
        is reached or max iterations exceeded.

    Algorithm:
        1. Generate thought about current state
        2. Decide on action based on thought
        3. Execute action and get observation
        4. Repeat until final answer or max iterations

    Attributes:
        tools: Dictionary of available tools.
        llm: LLM interface for generation.
        max_iterations: Maximum number of reasoning steps.
        prompt_builder: Builder for constructing prompts.
        parser: Parser for extracting ReAct components.

    Example:
        >>> agent = ReActAgent(tools=[search_tool, calc_tool])
        >>> result = agent.run("What is the population of Tokyo?")
        >>> print(result.final_answer)
    """

    def __init__(
        self,
        tools: Optional[List[ToolInterface]] = None,
        llm: Optional[LLMInterface] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> None:
        """Initialize ReAct agent.

        Args:
            tools: List of available tools.
            llm: LLM interface for generation.
            max_iterations: Maximum reasoning iterations.
            verbose: Whether to print intermediate steps.
        """
        self._tools: Dict[str, ToolInterface] = {
            tool.name: tool for tool in (tools or [])
        }
        self._llm = llm
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._prompt_builder = ReActPromptBuilder(tools)
        self._parser = ReActParser()

    def add_tool(self, tool: ToolInterface) -> None:
        """Add a tool to the agent."""
        self._tools[tool.name] = tool
        self._prompt_builder.add_tool(tool)

    def run(self, question: str, **kwargs: Any) -> ReActTrace:
        """Run the ReAct loop to answer a question.

        Args:
            question: The question to answer.
            **kwargs: Additional arguments for LLM generation.

        Returns:
            ReActTrace containing the complete reasoning trace.
        """
        trace = ReActTrace(question=question)

        if self._llm is None:
            # Return prompt without execution
            prompt = self._prompt_builder.build(question)
            trace.metadata["prompt_only"] = True
            trace.metadata["prompt"] = prompt
            return trace

        history = ""
        for i in range(1, self._max_iterations + 1):
            # Build prompt with current history
            prompt = self._prompt_builder.build(question, history)

            # Generate next step
            output = self._llm.generate(prompt, **kwargs)

            if self._verbose:
                print(f"\n--- Step {i} ---")
                print(output)

            # Parse the output
            step = self._parser.parse_step(output, step_number=i)
            if step is None:
                trace.metadata["error"] = "Failed to parse LLM output"
                break

            # Check for final answer
            final_answer = self._parser.parse_final_answer(output)
            if final_answer:
                trace.add_step(step)
                trace.final_answer = final_answer
                trace.success = True
                break

            # Execute action if present
            if step.action:
                observation = self._execute_action(step.action)
                step = ReActStep(
                    thought=step.thought,
                    action=step.action,
                    observation=observation,
                    step_number=i,
                )

                # Update history
                history += f"\n{step.format()}"

            trace.add_step(step)

            # Check if no action was taken (stuck)
            if step.action is None and not final_answer:
                trace.metadata["error"] = "No action taken and no final answer"
                break

        if not trace.success and trace.num_steps >= self._max_iterations:
            trace.metadata["error"] = "Max iterations reached"

        return trace

    def _execute_action(self, action: Action) -> Observation:
        """Execute an action and return observation.

        Args:
            action: The action to execute.

        Returns:
            Observation with the result.
        """
        tool = self._tools.get(action.name)
        if tool is None:
            return Observation(
                content=f"Error: Unknown tool '{action.name}'. Available tools: {list(self._tools.keys())}",
                step_number=action.step_number,
                success=False,
            )

        try:
            result = tool.execute(action.input)
            return Observation(
                content=result,
                step_number=action.step_number,
                success=True,
            )
        except Exception as e:
            return Observation(
                content=f"Error executing {action.name}: {str(e)}",
                step_number=action.step_number,
                success=False,
            )

    def get_prompt(self, question: str, history: str = "") -> str:
        """Get the prompt without executing.

        Args:
            question: The question to answer.
            history: Previous history.

        Returns:
            Complete prompt string.
        """
        return self._prompt_builder.build(question, history)

    @property
    def tools(self) -> Dict[str, ToolInterface]:
        """Get available tools."""
        return self._tools.copy()


# Utility functions

def create_react_agent(
    tools: List[Tuple[str, str, Callable]],
    llm: Optional[LLMInterface] = None,
    max_iterations: int = 10,
) -> ReActAgent:
    """Factory function to create a ReAct agent with simple tools.

    Args:
        tools: List of (name, description, function) tuples.
        llm: LLM interface.
        max_iterations: Maximum iterations.

    Returns:
        Configured ReActAgent.

    Example:
        >>> agent = create_react_agent([
        ...     ("search", "Search the web", lambda q: f"Results for {q}"),
        ...     ("calc", "Calculate math", lambda e: str(eval(e))),
        ... ])
    """
    tool_objects = [
        SimpleTool(name=name, description=desc, func=func)
        for name, desc, func in tools
    ]
    return ReActAgent(tools=tool_objects, llm=llm, max_iterations=max_iterations)


def format_react_trace(trace: ReActTrace) -> str:
    """Format a ReAct trace for display.

    Args:
        trace: The trace to format.

    Returns:
        Formatted string representation.
    """
    lines = [f"Question: {trace.question}", ""]

    for step in trace.steps:
        lines.append(step.format())
        lines.append("")

    if trace.final_answer:
        lines.append(f"Final Answer: {trace.final_answer}")

    if not trace.success:
        lines.append(f"\n[Failed: {trace.metadata.get('error', 'Unknown error')}]")

    return "\n".join(lines)


__all__ = [
    "StepType",
    "Thought",
    "Action",
    "Observation",
    "ReActStep",
    "ReActTrace",
    "ToolInterface",
    "SimpleTool",
    "ReActPromptBuilder",
    "ReActParser",
    "ReActAgent",
    "create_react_agent",
    "format_react_trace",
]
