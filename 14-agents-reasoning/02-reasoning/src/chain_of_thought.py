"""
Chain of Thought (CoT): Step-by-Step Reasoning for LLMs

Core Idea:
    Chain of Thought prompting enables LLMs to decompose complex problems into
    intermediate reasoning steps, dramatically improving performance on tasks
    requiring multi-step reasoning such as arithmetic, commonsense reasoning,
    and symbolic manipulation.

Mathematical Foundation:
    Traditional prompting models the output as:
    $$P(answer | question) = P(a | q)$$

    CoT introduces intermediate reasoning steps $r_1, r_2, ..., r_n$:
    $$P(answer | question) = \sum_{r} P(a | r, q) \cdot P(r | q)$$

    where the reasoning chain $r$ serves as a "cognitive scaffold" that guides
    the model toward the correct answer through explicit intermediate steps.

    The key insight is that by generating $r$ explicitly, the model can:
    1. Break down complex computations into simpler sub-problems
    2. Maintain working memory through the generated text
    3. Apply learned reasoning patterns to novel problems

Problem Statement:
    Standard prompting fails on complex reasoning tasks because:
    1. LLMs struggle with multi-step computations in a single forward pass
    2. Implicit reasoning provides no interpretability
    3. Errors compound without intermediate verification

    CoT addresses these by making reasoning explicit and decomposed.

Algorithm Variants:
    | Variant          | Description                              | Use Case              |
    |------------------|------------------------------------------|-----------------------|
    | Zero-shot CoT    | "Let's think step by step"               | Quick, no examples    |
    | Few-shot CoT     | Demonstrations with reasoning chains     | Best accuracy         |
    | Auto-CoT         | Automatic demonstration generation       | Scalable, diverse     |
    | Program-of-Thought| Generate code for computation           | Math-heavy tasks      |

Complexity Analysis:
    - Token overhead: O(k * r) where k = examples, r = avg reasoning length
    - Latency increase: ~2-3x due to longer generation
    - Accuracy improvement: 10-40% on reasoning benchmarks

References:
    - Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
    - Kojima et al. (2022): "Large Language Models are Zero-Shot Reasoners"
    - Zhang et al. (2022): "Automatic Chain of Thought Prompting in LLMs"
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


class CoTStrategy(str, Enum):
    """Chain of Thought strategy variants.

    Attributes:
        ZERO_SHOT: Add "Let's think step by step" without examples.
        FEW_SHOT: Provide demonstrations with explicit reasoning chains.
        AUTO_COT: Automatically generate diverse demonstrations.
        PROGRAM_OF_THOUGHT: Generate executable code for computation.
    """
    ZERO_SHOT: Final[str] = "zero_shot"
    FEW_SHOT: Final[str] = "few_shot"
    AUTO_COT: Final[str] = "auto_cot"
    PROGRAM_OF_THOUGHT: Final[str] = "program_of_thought"


@dataclass(frozen=True, slots=True)
class CoTStep:
    """A single step in a chain of thought reasoning process.

    Core Idea:
        Represents one atomic reasoning step with its content and optional
        metadata. Steps can be chained together to form complete reasoning traces.

    Attributes:
        content: The textual content of this reasoning step.
        step_number: Position in the reasoning chain (1-indexed).
        step_type: Category of reasoning (e.g., "decomposition", "calculation").
        confidence: Optional confidence score for this step (0.0 to 1.0).

    Example:
        >>> step = CoTStep(
        ...     content="First, I need to find the total number of apples.",
        ...     step_number=1,
        ...     step_type="decomposition"
        ... )
    """
    content: str
    step_number: int = 1
    step_type: str = "reasoning"
    confidence: Optional[float] = None

    def __str__(self) -> str:
        """Format step for display."""
        return f"Step {self.step_number}: {self.content}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "step_number": self.step_number,
            "step_type": self.step_type,
            "confidence": self.confidence,
        }


@dataclass
class CoTExample:
    """A demonstration example for few-shot CoT prompting.

    Core Idea:
        Encapsulates a complete question-reasoning-answer triple that serves
        as a demonstration for the LLM. High-quality examples are crucial
        for effective few-shot CoT.

    Attributes:
        question: The input question or problem.
        reasoning: The step-by-step reasoning chain.
        answer: The final answer derived from the reasoning.
        steps: Optional parsed list of individual reasoning steps.

    Example:
        >>> example = CoTExample(
        ...     question="If John has 5 apples and gives 2 to Mary, how many does he have?",
        ...     reasoning="John starts with 5 apples. He gives 2 to Mary. 5 - 2 = 3.",
        ...     answer="3 apples"
        ... )
    """
    question: str
    reasoning: str
    answer: str
    steps: List[CoTStep] = field(default_factory=list)

    def format(self, template: str = "Q: {question}\n\nA: Let's think step by step.\n{reasoning}\n\nTherefore, the answer is {answer}.") -> str:
        """Format example using template.

        Args:
            template: Format string with {question}, {reasoning}, {answer} placeholders.

        Returns:
            Formatted example string.
        """
        return template.format(
            question=self.question,
            reasoning=self.reasoning,
            answer=self.answer,
        )

    def parse_steps(self) -> List[CoTStep]:
        """Parse reasoning text into individual steps.

        Attempts to identify step boundaries using common patterns:
        - Numbered steps: "1.", "2.", etc.
        - Step markers: "First,", "Then,", "Finally,"
        - Sentence boundaries

        Returns:
            List of CoTStep objects.
        """
        if self.steps:
            return self.steps

        # Try numbered pattern first
        numbered_pattern = r'(\d+)[.):]\s*(.+?)(?=\d+[.):]\s*|$)'
        matches = re.findall(numbered_pattern, self.reasoning, re.DOTALL)

        if matches:
            self.steps = [
                CoTStep(content=content.strip(), step_number=int(num))
                for num, content in matches
            ]
            return self.steps

        # Try marker-based splitting
        markers = ["First,", "Second,", "Third,", "Then,", "Next,", "Finally,", "Therefore,"]
        pattern = '|'.join(re.escape(m) for m in markers)
        parts = re.split(f'({pattern})', self.reasoning)

        if len(parts) > 1:
            steps = []
            step_num = 1
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    content = parts[i] + parts[i + 1]
                    steps.append(CoTStep(content=content.strip(), step_number=step_num))
                    step_num += 1
            if steps:
                self.steps = steps
                return self.steps

        # Fallback: split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', self.reasoning)
        self.steps = [
            CoTStep(content=s.strip(), step_number=i + 1)
            for i, s in enumerate(sentences) if s.strip()
        ]
        return self.steps


@dataclass
class CoTResult:
    """Result of a Chain of Thought reasoning process.

    Attributes:
        question: Original input question.
        reasoning: Complete reasoning chain text.
        answer: Extracted final answer.
        steps: Parsed reasoning steps.
        raw_output: Raw LLM output before parsing.
        metadata: Additional information (tokens, latency, etc.).
    """
    question: str
    reasoning: str
    answer: str
    steps: List[CoTStep] = field(default_factory=list)
    raw_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        """Number of reasoning steps."""
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "question": self.question,
            "reasoning": self.reasoning,
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "num_steps": self.num_steps,
            "metadata": self.metadata,
        }


class LLMInterface(Protocol):
    """Protocol for LLM interaction.

    Any LLM client implementing this protocol can be used with CoT classes.
    """
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


class CoTPromptBuilder:
    """Builder for constructing Chain of Thought prompts.

    Core Idea:
        Provides a fluent interface for constructing CoT prompts with
        configurable templates, examples, and formatting options.

    Design Pattern:
        Builder Pattern - allows step-by-step construction of complex prompts.

    Example:
        >>> builder = CoTPromptBuilder()
        >>> prompt = (builder
        ...     .set_instruction("Solve the following math problem.")
        ...     .add_example(example1)
        ...     .add_example(example2)
        ...     .set_question("What is 15 + 27?")
        ...     .build())
    """

    # Default templates for different CoT strategies
    ZERO_SHOT_TEMPLATE: Final[str] = """{instruction}

Q: {question}

A: Let's think step by step."""

    FEW_SHOT_TEMPLATE: Final[str] = """{instruction}

{examples}

Q: {question}

A: Let's think step by step."""

    EXAMPLE_TEMPLATE: Final[str] = """Q: {question}

A: Let's think step by step.
{reasoning}

Therefore, the answer is {answer}."""

    def __init__(self) -> None:
        """Initialize the prompt builder."""
        self._instruction: str = "Solve the following problem step by step."
        self._examples: List[CoTExample] = []
        self._question: str = ""
        self._template: str = self.FEW_SHOT_TEMPLATE
        self._example_template: str = self.EXAMPLE_TEMPLATE
        self._custom_trigger: str = "Let's think step by step."

    def set_instruction(self, instruction: str) -> "CoTPromptBuilder":
        """Set the task instruction."""
        self._instruction = instruction
        return self

    def add_example(self, example: CoTExample) -> "CoTPromptBuilder":
        """Add a demonstration example."""
        self._examples.append(example)
        return self

    def add_examples(self, examples: List[CoTExample]) -> "CoTPromptBuilder":
        """Add multiple demonstration examples."""
        self._examples.extend(examples)
        return self

    def set_question(self, question: str) -> "CoTPromptBuilder":
        """Set the question to answer."""
        self._question = question
        return self

    def set_template(self, template: str) -> "CoTPromptBuilder":
        """Set custom prompt template."""
        self._template = template
        return self

    def set_example_template(self, template: str) -> "CoTPromptBuilder":
        """Set custom example template."""
        self._example_template = template
        return self

    def set_trigger(self, trigger: str) -> "CoTPromptBuilder":
        """Set custom CoT trigger phrase."""
        self._custom_trigger = trigger
        return self

    def clear_examples(self) -> "CoTPromptBuilder":
        """Remove all examples."""
        self._examples.clear()
        return self

    def build(self) -> str:
        """Build the final prompt string.

        Returns:
            Complete prompt ready for LLM input.

        Raises:
            ValueError: If question is not set.
        """
        if not self._question:
            raise ValueError("Question must be set before building prompt")

        # Format examples
        examples_text = "\n\n".join(
            ex.format(self._example_template) for ex in self._examples
        )

        # Choose template based on whether we have examples
        if self._examples:
            template = self._template
        else:
            template = self.ZERO_SHOT_TEMPLATE

        return template.format(
            instruction=self._instruction,
            examples=examples_text,
            question=self._question,
        )

    def build_zero_shot(self) -> str:
        """Build a zero-shot CoT prompt (no examples)."""
        if not self._question:
            raise ValueError("Question must be set before building prompt")

        return self.ZERO_SHOT_TEMPLATE.format(
            instruction=self._instruction,
            question=self._question,
        )


class ChainOfThought(ABC):
    """Abstract base class for Chain of Thought implementations.

    Core Idea:
        Defines the interface for all CoT variants. Subclasses implement
        specific prompting strategies while sharing common parsing and
        result handling logic.

    Template Method Pattern:
        - `reason()` is the template method defining the algorithm skeleton
        - `_build_prompt()` is the hook method overridden by subclasses
    """

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        answer_pattern: str = r"(?:answer is|answer:|therefore,?\s*(?:the answer is)?)\s*(.+?)(?:\.|$)",
    ) -> None:
        """Initialize CoT reasoner.

        Args:
            llm: LLM interface for generation. If None, prompts are returned without execution.
            answer_pattern: Regex pattern to extract final answer from reasoning.
        """
        self._llm = llm
        self._answer_pattern = re.compile(answer_pattern, re.IGNORECASE)

    @abstractmethod
    def _build_prompt(self, question: str) -> str:
        """Build the prompt for the given question.

        Args:
            question: The question to answer.

        Returns:
            Complete prompt string.
        """
        pass

    def reason(self, question: str, **kwargs: Any) -> CoTResult:
        """Execute chain of thought reasoning.

        Args:
            question: The question to answer.
            **kwargs: Additional arguments passed to LLM.

        Returns:
            CoTResult containing reasoning chain and answer.
        """
        prompt = self._build_prompt(question)

        if self._llm is None:
            # Return prompt without execution
            return CoTResult(
                question=question,
                reasoning="",
                answer="",
                raw_output=prompt,
                metadata={"prompt_only": True},
            )

        # Generate response
        raw_output = self._llm.generate(prompt, **kwargs)

        # Parse reasoning and answer
        reasoning, answer = self._parse_output(raw_output)

        # Parse steps
        example = CoTExample(question=question, reasoning=reasoning, answer=answer)
        steps = example.parse_steps()

        return CoTResult(
            question=question,
            reasoning=reasoning,
            answer=answer,
            steps=steps,
            raw_output=raw_output,
        )

    def _parse_output(self, output: str) -> Tuple[str, str]:
        """Parse LLM output to extract reasoning and answer.

        Args:
            output: Raw LLM output.

        Returns:
            Tuple of (reasoning, answer).
        """
        # Try to extract answer using pattern
        match = self._answer_pattern.search(output)
        if match:
            answer = match.group(1).strip()
            # Reasoning is everything before the answer pattern
            reasoning = output[:match.start()].strip()
        else:
            # No clear answer pattern found
            answer = ""
            reasoning = output.strip()

        return reasoning, answer

    def get_prompt(self, question: str) -> str:
        """Get the prompt without executing.

        Args:
            question: The question to answer.

        Returns:
            Complete prompt string.
        """
        return self._build_prompt(question)


class ZeroShotCoT(ChainOfThought):
    """Zero-shot Chain of Thought: "Let's think step by step"

    Core Idea:
        The simplest CoT variant that adds a trigger phrase to elicit
        step-by-step reasoning without any demonstrations. Surprisingly
        effective despite its simplicity.

    Key Finding (Kojima et al., 2022):
        Simply adding "Let's think step by step" improves accuracy on
        reasoning tasks by 10-40% compared to standard prompting.

    Attributes:
        trigger: The phrase that triggers step-by-step reasoning.
        instruction: Task-specific instruction prepended to prompt.

    Example:
        >>> cot = ZeroShotCoT()
        >>> result = cot.reason("What is 15% of 80?")
        >>> print(result.reasoning)
        "To find 15% of 80, I need to multiply 80 by 0.15..."
    """

    # Common trigger phrases ranked by effectiveness
    TRIGGERS: Final[Dict[str, str]] = {
        "standard": "Let's think step by step.",
        "detailed": "Let's work through this step by step, showing all calculations.",
        "careful": "Let's think about this carefully, step by step.",
        "logical": "Let's approach this logically, one step at a time.",
        "breakdown": "Let's break this down into steps.",
    }

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        trigger: str = "standard",
        instruction: str = "",
    ) -> None:
        """Initialize Zero-shot CoT.

        Args:
            llm: LLM interface for generation.
            trigger: Trigger phrase key or custom trigger string.
            instruction: Optional task instruction.
        """
        super().__init__(llm)
        self._trigger = self.TRIGGERS.get(trigger, trigger)
        self._instruction = instruction

    def _build_prompt(self, question: str) -> str:
        """Build zero-shot CoT prompt."""
        parts = []
        if self._instruction:
            parts.append(self._instruction)
        parts.append(f"Q: {question}")
        parts.append(f"\nA: {self._trigger}")
        return "\n\n".join(parts)


class FewShotCoT(ChainOfThought):
    """Few-shot Chain of Thought with demonstration examples.

    Core Idea:
        Provides explicit demonstrations of reasoning chains before the
        target question. The LLM learns to mimic the reasoning pattern
        from the examples.

    Best Practices:
        1. Use 3-8 diverse examples covering different reasoning patterns
        2. Ensure examples are similar in complexity to target questions
        3. Include examples with different answer types
        4. Order examples from simple to complex

    Example:
        >>> examples = [
        ...     CoTExample(
        ...         question="What is 2 + 3?",
        ...         reasoning="2 + 3 = 5",
        ...         answer="5"
        ...     ),
        ... ]
        >>> cot = FewShotCoT(examples=examples)
        >>> result = cot.reason("What is 7 + 8?")
    """

    def __init__(
        self,
        examples: List[CoTExample],
        llm: Optional[LLMInterface] = None,
        instruction: str = "Solve the following problem step by step.",
    ) -> None:
        """Initialize Few-shot CoT.

        Args:
            examples: List of demonstration examples.
            llm: LLM interface for generation.
            instruction: Task instruction.
        """
        super().__init__(llm)
        self._examples = examples
        self._instruction = instruction
        self._builder = CoTPromptBuilder()

    def _build_prompt(self, question: str) -> str:
        """Build few-shot CoT prompt with examples."""
        self._builder.clear_examples()
        self._builder.set_instruction(self._instruction)
        self._builder.add_examples(self._examples)
        self._builder.set_question(question)
        return self._builder.build()

    def add_example(self, example: CoTExample) -> None:
        """Add a demonstration example."""
        self._examples.append(example)

    def remove_example(self, index: int) -> Optional[CoTExample]:
        """Remove example by index."""
        if 0 <= index < len(self._examples):
            return self._examples.pop(index)
        return None

    @property
    def examples(self) -> List[CoTExample]:
        """Get current examples."""
        return self._examples.copy()


class AutoCoT(ChainOfThought):
    """Automatic Chain of Thought: Diverse demonstration generation.

    Core Idea:
        Automatically generates diverse demonstrations by clustering
        questions and selecting representative examples. Reduces manual
        effort while maintaining or improving accuracy.

    Algorithm (Zhang et al., 2022):
        1. Cluster questions by semantic similarity
        2. Select one representative question per cluster
        3. Generate reasoning chains using Zero-shot CoT
        4. Filter low-quality chains (too short, errors, etc.)
        5. Use filtered chains as demonstrations

    Diversity Heuristics:
        - Question length variation
        - Different reasoning patterns
        - Various answer types
        - Multiple domains (if applicable)

    Example:
        >>> auto_cot = AutoCoT(question_pool=questions)
        >>> auto_cot.generate_demonstrations(n_clusters=5)
        >>> result = auto_cot.reason("New question here")
    """

    def __init__(
        self,
        question_pool: Optional[List[str]] = None,
        llm: Optional[LLMInterface] = None,
        n_demonstrations: int = 5,
        min_steps: int = 2,
        max_steps: int = 10,
    ) -> None:
        """Initialize Auto-CoT.

        Args:
            question_pool: Pool of questions for demonstration generation.
            llm: LLM interface for generation.
            n_demonstrations: Number of demonstrations to generate.
            min_steps: Minimum reasoning steps for valid demonstration.
            max_steps: Maximum reasoning steps for valid demonstration.
        """
        super().__init__(llm)
        self._question_pool = question_pool or []
        self._n_demonstrations = n_demonstrations
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._demonstrations: List[CoTExample] = []
        self._zero_shot = ZeroShotCoT(llm)

    def _build_prompt(self, question: str) -> str:
        """Build prompt using generated demonstrations."""
        if not self._demonstrations:
            # Fall back to zero-shot if no demonstrations
            return self._zero_shot._build_prompt(question)

        builder = CoTPromptBuilder()
        builder.set_instruction("Solve the following problem step by step.")
        builder.add_examples(self._demonstrations)
        builder.set_question(question)
        return builder.build()

    def generate_demonstrations(
        self,
        questions: Optional[List[str]] = None,
        diversity_method: str = "length",
    ) -> List[CoTExample]:
        """Generate diverse demonstrations from question pool.

        Args:
            questions: Questions to use (defaults to question_pool).
            diversity_method: Method for selecting diverse questions.
                - "length": Select questions of varying lengths
                - "random": Random selection
                - "first": First n questions

        Returns:
            List of generated demonstration examples.
        """
        questions = questions or self._question_pool
        if not questions:
            return []

        # Select diverse questions
        selected = self._select_diverse(questions, diversity_method)

        # Generate reasoning for each
        demonstrations = []
        for q in selected:
            result = self._zero_shot.reason(q)
            if self._is_valid_demonstration(result):
                demonstrations.append(CoTExample(
                    question=q,
                    reasoning=result.reasoning,
                    answer=result.answer,
                    steps=result.steps,
                ))

        self._demonstrations = demonstrations[:self._n_demonstrations]
        return self._demonstrations

    def _select_diverse(self, questions: List[str], method: str) -> List[str]:
        """Select diverse questions using specified method."""
        if method == "random":
            import random
            return random.sample(questions, min(len(questions), self._n_demonstrations * 2))
        elif method == "length":
            # Sort by length and select evenly spaced
            sorted_q = sorted(questions, key=len)
            step = max(1, len(sorted_q) // (self._n_demonstrations * 2))
            return sorted_q[::step][:self._n_demonstrations * 2]
        else:  # "first"
            return questions[:self._n_demonstrations * 2]

    def _is_valid_demonstration(self, result: CoTResult) -> bool:
        """Check if a reasoning result is valid for use as demonstration."""
        if not result.reasoning or not result.answer:
            return False
        if result.num_steps < self._min_steps:
            return False
        if result.num_steps > self._max_steps:
            return False
        # Check for error indicators
        error_patterns = ["i don't know", "cannot", "error", "invalid"]
        lower_reasoning = result.reasoning.lower()
        if any(p in lower_reasoning for p in error_patterns):
            return False
        return True

    @property
    def demonstrations(self) -> List[CoTExample]:
        """Get current demonstrations."""
        return self._demonstrations.copy()


# Utility functions

def extract_answer(text: str, patterns: Optional[List[str]] = None) -> str:
    """Extract final answer from reasoning text.

    Args:
        text: Text containing reasoning and answer.
        patterns: Custom regex patterns to try.

    Returns:
        Extracted answer or empty string if not found.
    """
    default_patterns = [
        r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
        r"(?:therefore,?)\s*(.+?)(?:\.|$)",
        r"(?:so,?)\s*(.+?)(?:\.|$)",
        r"=\s*(\d+(?:\.\d+)?)",
    ]
    patterns = patterns or default_patterns

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def count_reasoning_steps(text: str) -> int:
    """Count the number of reasoning steps in text.

    Args:
        text: Reasoning text.

    Returns:
        Estimated number of steps.
    """
    # Count numbered steps
    numbered = len(re.findall(r'\d+[.):]', text))
    if numbered > 0:
        return numbered

    # Count step markers
    markers = ["first", "second", "third", "then", "next", "finally", "therefore"]
    marker_count = sum(1 for m in markers if m in text.lower())
    if marker_count > 0:
        return marker_count

    # Fall back to sentence count
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return len([s for s in sentences if s.strip()])


__all__ = [
    "CoTStrategy",
    "CoTStep",
    "CoTExample",
    "CoTResult",
    "CoTPromptBuilder",
    "ChainOfThought",
    "ZeroShotCoT",
    "FewShotCoT",
    "AutoCoT",
    "extract_answer",
    "count_reasoning_steps",
]
