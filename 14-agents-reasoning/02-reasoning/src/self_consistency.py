"""
Self-Consistency: Improving Reasoning via Multiple Sampling

Core Idea:
    Self-Consistency improves LLM reasoning by sampling multiple reasoning
    paths and selecting the most consistent answer through voting. This
    leverages the intuition that correct reasoning paths are more likely
    to converge on the same answer.

Mathematical Foundation:
    Given a question q, we sample n reasoning paths:
    $$\{(r_1, a_1), (r_2, a_2), ..., (r_n, a_n)\}$$

    The final answer is selected by majority voting:
    $$a^* = \arg\max_a \sum_{i=1}^{n} \mathbb{1}[a_i = a]$$

    For weighted voting with confidence scores $c_i$:
    $$a^* = \arg\max_a \sum_{i=1}^{n} c_i \cdot \mathbb{1}[a_i = a]$$

References:
    - Wang et al. (2022): "Self-Consistency Improves Chain of Thought Reasoning"
    - https://arxiv.org/abs/2203.11171
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Final, List, Optional, Protocol, Tuple


@dataclass
class SampledPath:
    """A single sampled reasoning path."""
    reasoning: str
    answer: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyResult:
    """Result of self-consistency voting."""
    question: str
    final_answer: str
    confidence: float
    vote_counts: Dict[str, int]
    paths: List[SampledPath]
    agreement_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "vote_counts": self.vote_counts,
            "num_paths": len(self.paths),
            "agreement_ratio": self.agreement_ratio,
        }


class VotingStrategy(ABC):
    """Abstract base class for voting strategies."""

    @abstractmethod
    def vote(self, paths: List[SampledPath]) -> Tuple[str, float]:
        """Select answer from paths.
        
        Returns:
            Tuple of (selected_answer, confidence).
        """
        pass


class MajorityVoting(VotingStrategy):
    """Simple majority voting - most common answer wins."""

    def vote(self, paths: List[SampledPath]) -> Tuple[str, float]:
        if not paths:
            return "", 0.0
        
        answers = [p.answer for p in paths]
        counter = Counter(answers)
        winner, count = counter.most_common(1)[0]
        confidence = count / len(paths)
        return winner, confidence


class WeightedVoting(VotingStrategy):
    """Weighted voting using confidence scores."""

    def vote(self, paths: List[SampledPath]) -> Tuple[str, float]:
        if not paths:
            return "", 0.0
        
        scores: Dict[str, float] = {}
        for path in paths:
            scores[path.answer] = scores.get(path.answer, 0) + path.confidence
        
        winner = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[winner] / total if total > 0 else 0.0
        return winner, confidence


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""
    def generate(self, prompt: str, **kwargs: Any) -> str: ...


class SelfConsistency:
    """Self-Consistency reasoning with multiple sampling."""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        n_samples: int = 5,
        voting_strategy: Optional[VotingStrategy] = None,
        temperature: float = 0.7,
    ) -> None:
        self._llm = llm
        self._n_samples = n_samples
        self._voting = voting_strategy or MajorityVoting()
        self._temperature = temperature

    def reason(
        self,
        question: str,
        prompt_template: str = "Q: {question}\nA: Let's think step by step.",
        answer_extractor: Optional[Callable[[str], str]] = None,
    ) -> ConsistencyResult:
        """Run self-consistency reasoning."""
        paths: List[SampledPath] = []
        
        if self._llm is None:
            # Return placeholder for testing
            return ConsistencyResult(
                question=question,
                final_answer="",
                confidence=0.0,
                vote_counts={},
                paths=[],
            )

        prompt = prompt_template.format(question=question)
        
        # Sample multiple paths
        for _ in range(self._n_samples):
            response = self._llm.generate(prompt, temperature=self._temperature)
            answer = self._extract_answer(response, answer_extractor)
            paths.append(SampledPath(reasoning=response, answer=answer))

        # Vote
        final_answer, confidence = self._voting.vote(paths)
        vote_counts = Counter(p.answer for p in paths)
        agreement = vote_counts[final_answer] / len(paths) if paths else 0.0

        return ConsistencyResult(
            question=question,
            final_answer=final_answer,
            confidence=confidence,
            vote_counts=dict(vote_counts),
            paths=paths,
            agreement_ratio=agreement,
        )

    def _extract_answer(
        self,
        response: str,
        extractor: Optional[Callable[[str], str]] = None,
    ) -> str:
        """Extract answer from response."""
        if extractor:
            return extractor(response)
        
        import re
        patterns = [
            r"(?:answer is|answer:)\s*(.+?)(?:\.|$)",
            r"(?:therefore,?)\s*(.+?)(?:\.|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return response.strip().split('\n')[-1]


__all__ = [
    "SampledPath",
    "ConsistencyResult",
    "VotingStrategy",
    "MajorityVoting",
    "WeightedVoting",
    "SelfConsistency",
]
