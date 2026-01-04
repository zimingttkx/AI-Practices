"""
Reflection: Self-Evaluation and Iterative Improvement

Core Idea:
    Reflection enables LLMs to evaluate their own outputs, detect errors,
    and iteratively improve responses. This mimics human metacognition -
    the ability to think about one's own thinking.

Mathematical Foundation:
    Reflection can be modeled as an iterative refinement process:
    $$r_{t+1} = f(r_t, e(r_t, g))$$

    where:
    - $r_t$ is the response at iteration $t$
    - $e(r_t, g)$ is the evaluation function comparing response to goal $g$
    - $f$ is the refinement function

References:
    - Shinn et al. (2023): "Reflexion: Language Agents with Verbal Reinforcement"
    - https://arxiv.org/abs/2303.11366
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Protocol


class EvaluationResult(str, Enum):
    """Result of self-evaluation."""
    CORRECT: Final[str] = "correct"
    PARTIALLY_CORRECT: Final[str] = "partially_correct"
    INCORRECT: Final[str] = "incorrect"
    UNCERTAIN: Final[str] = "uncertain"


@dataclass
class ReflectionStep:
    """A single reflection iteration."""
    iteration: int
    response: str
    evaluation: EvaluationResult
    feedback: str
    score: float = 0.0
    improvements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "response": self.response,
            "evaluation": self.evaluation.value,
            "feedback": self.feedback,
            "score": self.score,
            "improvements": self.improvements,
        }


@dataclass
class ReflectionResult:
    """Result of reflection process."""
    question: str
    initial_response: str
    final_response: str
    steps: List[ReflectionStep]
    total_iterations: int
    improved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "initial_response": self.initial_response,
            "final_response": self.final_response,
            "total_iterations": self.total_iterations,
            "improved": self.improved,
            "steps": [s.to_dict() for s in self.steps],
        }


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""
    def generate(self, prompt: str, **kwargs: Any) -> str: ...


class SelfEvaluator(ABC):
    """Abstract base class for self-evaluation."""

    @abstractmethod
    def evaluate(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> tuple[EvaluationResult, str, float]:
        """Evaluate a response.
        
        Returns:
            Tuple of (result, feedback, score).
        """
        pass


class SimpleSelfEvaluator(SelfEvaluator):
    """Simple LLM-based self-evaluator."""

    EVAL_PROMPT: Final[str] = """Question: {question}

Response: {response}

Evaluate this response:
1. Is it correct and complete?
2. Are there any errors or missing information?
3. Rate it from 1-10.

Format: EVALUATION: [correct/partially_correct/incorrect]
FEEDBACK: [your feedback]
SCORE: [1-10]"""

    def __init__(self, llm: Optional[LLMInterface] = None) -> None:
        self._llm = llm

    def evaluate(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> tuple[EvaluationResult, str, float]:
        if self._llm is None:
            return EvaluationResult.UNCERTAIN, "No LLM available", 0.5

        prompt = self.EVAL_PROMPT.format(question=question, response=response)
        output = self._llm.generate(prompt)
        return self._parse_evaluation(output)

    def _parse_evaluation(
        self, output: str
    ) -> tuple[EvaluationResult, str, float]:
        import re
        
        # Parse evaluation
        eval_match = re.search(r"EVALUATION:\s*(\w+)", output, re.IGNORECASE)
        eval_str = eval_match.group(1).lower() if eval_match else "uncertain"
        
        eval_map = {
            "correct": EvaluationResult.CORRECT,
            "partially_correct": EvaluationResult.PARTIALLY_CORRECT,
            "incorrect": EvaluationResult.INCORRECT,
        }
        result = eval_map.get(eval_str, EvaluationResult.UNCERTAIN)
        
        # Parse feedback
        feedback_match = re.search(r"FEEDBACK:\s*(.+?)(?=SCORE:|$)", output, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else ""
        
        # Parse score
        score_match = re.search(r"SCORE:\s*(\d+)", output)
        score = int(score_match.group(1)) / 10.0 if score_match else 0.5
        
        return result, feedback, score


class ErrorDetector:
    """Detects common errors in responses."""

    ERROR_PATTERNS: Final[List[tuple[str, str]]] = [
        (r"\b(incorrect|wrong|error|mistake)\b", "Contains error indicators"),
        (r"\b(i don't know|unsure|uncertain)\b", "Expresses uncertainty"),
        (r"\b(however|but|although)\b.*\b(not|no)\b", "Contains contradictions"),
    ]

    def detect(self, response: str) -> List[str]:
        """Detect potential errors in response."""
        import re
        errors = []
        for pattern, description in self.ERROR_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                errors.append(description)
        return errors


class CorrectionStrategy(ABC):
    """Abstract base class for correction strategies."""

    @abstractmethod
    def correct(
        self,
        question: str,
        response: str,
        feedback: str,
    ) -> str:
        """Generate corrected response."""
        pass


class SimpleCorrectionStrategy(CorrectionStrategy):
    """Simple LLM-based correction."""

    CORRECT_PROMPT: Final[str] = """Question: {question}

Previous response: {response}

Feedback: {feedback}

Please provide an improved response that addresses the feedback."""

    def __init__(self, llm: Optional[LLMInterface] = None) -> None:
        self._llm = llm

    def correct(
        self,
        question: str,
        response: str,
        feedback: str,
    ) -> str:
        if self._llm is None:
            return response
        
        prompt = self.CORRECT_PROMPT.format(
            question=question,
            response=response,
            feedback=feedback,
        )
        return self._llm.generate(prompt)


class Reflection:
    """Reflection: Self-evaluation and iterative improvement."""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        evaluator: Optional[SelfEvaluator] = None,
        corrector: Optional[CorrectionStrategy] = None,
        max_iterations: int = 3,
        target_score: float = 0.8,
    ) -> None:
        self._llm = llm
        self._evaluator = evaluator or SimpleSelfEvaluator(llm)
        self._corrector = corrector or SimpleCorrectionStrategy(llm)
        self._max_iterations = max_iterations
        self._target_score = target_score

    def reflect(
        self,
        question: str,
        initial_response: str,
    ) -> ReflectionResult:
        """Run reflection loop."""
        steps: List[ReflectionStep] = []
        current_response = initial_response
        
        for i in range(self._max_iterations):
            # Evaluate
            result, feedback, score = self._evaluator.evaluate(
                question, current_response
            )
            
            step = ReflectionStep(
                iteration=i + 1,
                response=current_response,
                evaluation=result,
                feedback=feedback,
                score=score,
            )
            steps.append(step)
            
            # Check if good enough
            if result == EvaluationResult.CORRECT or score >= self._target_score:
                break
            
            # Correct
            current_response = self._corrector.correct(
                question, current_response, feedback
            )

        return ReflectionResult(
            question=question,
            initial_response=initial_response,
            final_response=current_response,
            steps=steps,
            total_iterations=len(steps),
            improved=current_response != initial_response,
        )


__all__ = [
    "EvaluationResult",
    "ReflectionStep",
    "ReflectionResult",
    "SelfEvaluator",
    "SimpleSelfEvaluator",
    "ErrorDetector",
    "CorrectionStrategy",
    "SimpleCorrectionStrategy",
    "Reflection",
]
