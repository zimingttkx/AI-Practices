"""
Debate Agents: Adversarial Multi-Agent Reasoning System.

Core Idea:
    Debate agents implement adversarial reasoning where multiple agents argue
    different positions on a topic. A judge agent evaluates arguments and
    determines the winner or synthesizes a final conclusion.

Mathematical Foundation:
    Debate can be modeled as a zero-sum game:
    
    $$V(s) = \\max_a \\min_b Q(s, a, b)$$
    
    where agents A and B alternate moves, and the judge evaluates:
    $$J(\\text{debate}) = \\arg\\max_{p \\in \\{A, B\\}} \\text{score}(\\text{arguments}_p)$$

Key Patterns:
    - Adversarial: Agents argue opposing positions
    - Socratic: Question-answer dialogue to expose flaws
    - Collaborative Debate: Agents refine ideas through critique

References:
    - Irving et al. (2018): AI Safety via Debate
    - Du et al. (2023): Improving Factuality through Multi-Agent Debate

Author: AI-Practices
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

try:
    from .agent_base import BaseAgent, AgentConfig, AgentRole, AgentResponse, MockLLM, LLMInterface
except ImportError:
    from agent_base import BaseAgent, AgentConfig, AgentRole, AgentResponse, MockLLM, LLMInterface

__all__ = [
    "DebateRole",
    "DebateConfig",
    "DebateRound",
    "DebateResult",
    "DebaterAgent",
    "JudgeAgent",
    "DebateArena",
]

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_ROUNDS: Final[int] = 3
DEFAULT_ARGUMENT_MAX_TOKENS: Final[int] = 500


# =============================================================================
# Enumerations
# =============================================================================


class DebateRole(str, Enum):
    """Role in a debate."""
    PROPONENT = "proponent"  # Argues FOR the proposition
    OPPONENT = "opponent"    # Argues AGAINST the proposition
    JUDGE = "judge"          # Evaluates arguments


class DebatePhase(str, Enum):
    """Phase of the debate."""
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    CLOSING = "closing"
    JUDGMENT = "judgment"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DebateConfig:
    """Configuration for a debate session."""
    topic: str
    max_rounds: int = DEFAULT_MAX_ROUNDS
    allow_rebuttals: bool = True
    require_evidence: bool = False
    time_limit_per_turn: float = 60.0
    judge_criteria: List[str] = field(default_factory=lambda: [
        "logical_coherence",
        "evidence_quality", 
        "persuasiveness",
        "addressing_counterarguments"
    ])


@dataclass
class Argument:
    """A single argument in the debate."""
    agent_id: str
    agent_name: str
    role: DebateRole
    content: str
    phase: DebatePhase
    round_num: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "role": self.role.value,
            "content": self.content,
            "phase": self.phase.value,
            "round_num": self.round_num,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DebateRound:
    """A single round of debate."""
    round_num: int
    proponent_argument: Optional[Argument] = None
    opponent_argument: Optional[Argument] = None
    
    @property
    def is_complete(self) -> bool:
        return self.proponent_argument is not None and self.opponent_argument is not None


@dataclass
class JudgmentScore:
    """Scoring for a debate participant."""
    agent_id: str
    scores: Dict[str, float]  # criterion -> score
    total: float
    feedback: str


@dataclass
class DebateResult:
    """Final result of a debate."""
    topic: str
    winner: Optional[str]  # agent_id or None for tie
    winner_role: Optional[DebateRole]
    proponent_score: JudgmentScore
    opponent_score: JudgmentScore
    rounds: List[DebateRound]
    judgment: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "winner": self.winner,
            "winner_role": self.winner_role.value if self.winner_role else None,
            "proponent_total": self.proponent_score.total,
            "opponent_total": self.opponent_score.total,
            "judgment": self.judgment,
            "rounds_count": len(self.rounds),
        }


# =============================================================================
# Debater Agent
# =============================================================================


class DebaterAgent(BaseAgent):
    """
    Agent that participates in debates.
    
    Can argue either for (proponent) or against (opponent) a proposition.
    """

    def __init__(
        self,
        config: AgentConfig,
        debate_role: DebateRole,
        llm: Optional[LLMInterface] = None,
    ):
        super().__init__(config, llm)
        self.debate_role = debate_role
        self._debate_history: List[Argument] = []

    async def think(self, input_text: str) -> str:
        """Analyze the debate context and plan argument."""
        return input_text

    async def act(self, thought: str) -> str:
        """Generate argument based on analysis."""
        messages = self.get_messages_for_llm()
        response = await self.llm.generate(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response

    async def generate_opening(self, topic: str) -> Argument:
        """Generate opening argument."""
        stance = "FOR" if self.debate_role == DebateRole.PROPONENT else "AGAINST"
        prompt = f"""You are arguing {stance} the following proposition:

Topic: {topic}

Generate a compelling opening argument. Include:
1. Clear thesis statement
2. 2-3 main supporting points
3. Brief evidence or reasoning for each point

Opening Argument:"""

        self.receive(prompt)
        response = await self.act(prompt)
        
        argument = Argument(
            agent_id=self.id,
            agent_name=self.name,
            role=self.debate_role,
            content=response,
            phase=DebatePhase.OPENING,
            round_num=0,
        )
        self._debate_history.append(argument)
        return argument

    async def generate_rebuttal(
        self,
        topic: str,
        opponent_argument: Argument,
        round_num: int,
    ) -> Argument:
        """Generate rebuttal to opponent's argument."""
        stance = "FOR" if self.debate_role == DebateRole.PROPONENT else "AGAINST"
        prompt = f"""You are arguing {stance}: {topic}

Your opponent argued:
{opponent_argument.content}

Generate a rebuttal that:
1. Addresses their main points
2. Points out logical flaws or weak evidence
3. Reinforces your position with new arguments

Rebuttal:"""

        self.receive(prompt)
        response = await self.act(prompt)
        
        argument = Argument(
            agent_id=self.id,
            agent_name=self.name,
            role=self.debate_role,
            content=response,
            phase=DebatePhase.REBUTTAL,
            round_num=round_num,
        )
        self._debate_history.append(argument)
        return argument

    async def generate_closing(self, topic: str) -> Argument:
        """Generate closing statement."""
        history_summary = "\n".join([
            f"[Round {a.round_num}] {a.content[:200]}..."
            for a in self._debate_history[-4:]
        ])
        
        prompt = f"""Topic: {topic}

Your debate history:
{history_summary}

Generate a powerful closing statement that:
1. Summarizes your strongest arguments
2. Addresses key counterarguments raised
3. Ends with a compelling conclusion

Closing Statement:"""

        self.receive(prompt)
        response = await self.act(prompt)
        
        argument = Argument(
            agent_id=self.id,
            agent_name=self.name,
            role=self.debate_role,
            content=response,
            phase=DebatePhase.CLOSING,
            round_num=len(self._debate_history),
        )
        self._debate_history.append(argument)
        return argument


# =============================================================================
# Judge Agent
# =============================================================================


class JudgeAgent(BaseAgent):
    """
    Agent that judges debates and determines winners.
    
    Evaluates arguments based on configurable criteria and provides
    detailed scoring and feedback.
    """

    def __init__(
        self,
        config: AgentConfig,
        criteria: Optional[List[str]] = None,
        llm: Optional[LLMInterface] = None,
    ):
        super().__init__(config, llm)
        self.criteria = criteria or [
            "logical_coherence",
            "evidence_quality",
            "persuasiveness",
            "addressing_counterarguments"
        ]

    async def think(self, input_text: str) -> str:
        return input_text

    async def act(self, thought: str) -> str:
        messages = self.get_messages_for_llm()
        return await self.llm.generate(
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent judging
            max_tokens=self.config.max_tokens,
        )

    async def evaluate_debate(
        self,
        topic: str,
        rounds: List[DebateRound],
        proponent_name: str,
        opponent_name: str,
    ) -> Tuple[JudgmentScore, JudgmentScore, str]:
        """Evaluate a complete debate and return scores."""
        
        transcript = self._format_transcript(rounds, proponent_name, opponent_name)
        criteria_list = "\n".join([f"- {c}" for c in self.criteria])
        
        prompt = f"""You are an impartial debate judge. Evaluate the following debate.

Topic: {topic}

Debate Transcript:
{transcript}

Evaluation Criteria:
{criteria_list}

For each participant, provide scores (0-10) for each criterion, total score, and feedback.
Then declare a winner or tie with justification."""

        self.receive(prompt)
        response = await self.act(prompt)
        
        proponent_score = self._parse_scores(response, "PROPONENT", proponent_name)
        opponent_score = self._parse_scores(response, "OPPONENT", opponent_name)
        judgment = response
        
        return proponent_score, opponent_score, judgment

    def _format_transcript(self, rounds: List[DebateRound], pro_name: str, opp_name: str) -> str:
        lines = []
        for r in rounds:
            lines.append(f"=== Round {r.round_num} ===")
            if r.proponent_argument:
                lines.append(f"[{pro_name} - FOR]: {r.proponent_argument.content}")
            if r.opponent_argument:
                lines.append(f"[{opp_name} - AGAINST]: {r.opponent_argument.content}")
        return "\n".join(lines)

    def _parse_scores(self, response: str, role: str, agent_name: str) -> JudgmentScore:
        scores = {c: 5.0 for c in self.criteria}
        return JudgmentScore(agent_id=agent_name, scores=scores, total=sum(scores.values()), feedback=f"Evaluation for {role}")


# =============================================================================
# Debate Arena
# =============================================================================


class DebateArena:
    """
    Orchestrates debates between agents.
    
    Manages the debate flow, turn-taking, and final judgment.
    
    Example:
        >>> arena = DebateArena(proponent, opponent, judge)
        >>> result = await arena.run_debate("AI will benefit humanity")
    """

    def __init__(
        self,
        proponent: DebaterAgent,
        opponent: DebaterAgent,
        judge: JudgeAgent,
        config: Optional[DebateConfig] = None,
    ):
        self.proponent = proponent
        self.opponent = opponent
        self.judge = judge
        self.config = config or DebateConfig(topic="")
        self._rounds: List[DebateRound] = []
        self._all_arguments: List[Argument] = []

    async def run_debate(self, topic: Optional[str] = None) -> DebateResult:
        """Run a complete debate and return results."""
        topic = topic or self.config.topic
        if not topic:
            raise ValueError("Debate topic is required")
        
        self._rounds = []
        self._all_arguments = []
        
        logger.info(f"Starting debate: {topic}")
        
        # Opening statements
        pro_open = await self.proponent.generate_opening(topic)
        opp_open = await self.opponent.generate_opening(topic)
        self._rounds.append(DebateRound(0, pro_open, opp_open))
        self._all_arguments.extend([pro_open, opp_open])
        
        # Rebuttal rounds
        for round_num in range(1, self.config.max_rounds + 1):
            last_opp = self._rounds[-1].opponent_argument
            last_pro = self._rounds[-1].proponent_argument
            pro_reb = await self.proponent.generate_rebuttal(topic, last_opp, round_num)
            opp_reb = await self.opponent.generate_rebuttal(topic, last_pro, round_num)
            self._rounds.append(DebateRound(round_num, pro_reb, opp_reb))
            self._all_arguments.extend([pro_reb, opp_reb])
        
        # Closing statements
        pro_close = await self.proponent.generate_closing(topic)
        opp_close = await self.opponent.generate_closing(topic)
        self._rounds.append(DebateRound(len(self._rounds), pro_close, opp_close))
        self._all_arguments.extend([pro_close, opp_close])
        
        # Judgment
        pro_score, opp_score, judgment = await self.judge.evaluate_debate(
            topic, self._rounds, self.proponent.name, self.opponent.name
        )
        
        winner = self.proponent.id if pro_score.total > opp_score.total else (
            self.opponent.id if opp_score.total > pro_score.total else None
        )
        winner_role = DebateRole.PROPONENT if winner == self.proponent.id else (
            DebateRole.OPPONENT if winner == self.opponent.id else None
        )
        
        return DebateResult(
            topic=topic, winner=winner, winner_role=winner_role,
            proponent_score=pro_score, opponent_score=opp_score,
            rounds=self._rounds, judgment=judgment
        )

    def get_transcript(self) -> str:
        """Get formatted debate transcript."""
        lines = [f"Debate Topic: {self.config.topic}", "=" * 50]
        for arg in self._all_arguments:
            role_str = "FOR" if arg.role == DebateRole.PROPONENT else "AGAINST"
            lines.append(f"\n[{arg.agent_name} - {role_str}] ({arg.phase.value})")
            lines.append(arg.content)
        return "\n".join(lines)
