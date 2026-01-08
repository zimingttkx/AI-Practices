"""
Collaborative Agents: Cooperative Multi-Agent Patterns.

Core Idea:
    Collaborative agents work together to solve problems through cooperation,
    consensus building, and task division. Unlike debate agents, they aim
    for synergy rather than competition.

Mathematical Foundation:
    Collaboration can be modeled as a cooperative game:
    
    $$V(S) = \\sum_{i \\in S} v_i + \\text{synergy}(S)$$
    
    where synergy captures the additional value from cooperation.
    
    Consensus: $c^* = \\arg\\min_c \\sum_i d(c, c_i)$

Key Patterns:
    - Division of Labor: Agents specialize in subtasks
    - Consensus Building: Agents converge on shared conclusions
    - Iterative Refinement: Agents improve each other's outputs
    - Voting: Democratic decision making

References:
    - AutoGen: https://github.com/microsoft/autogen
    - CAMEL: Communicative Agents for Mind Exploration

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
from typing import Any, Dict, Final, List, Optional, Set, Callable

try:
    from .agent_base import BaseAgent, AgentConfig, AgentRole, AgentResponse, LLMInterface
except ImportError:
    from agent_base import BaseAgent, AgentConfig, AgentRole, AgentResponse, LLMInterface

__all__ = [
    "CollaborationMode",
    "TeamConfig",
    "TeamMember",
    "Contribution",
    "ConsensusResult",
    "CollaborativeTeam",
    "ConsensusBuilder",
    "VotingSystem",
]

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS: Final[int] = 5
DEFAULT_CONSENSUS_THRESHOLD: Final[float] = 0.8


class CollaborationMode(str, Enum):
    """Mode of collaboration between agents."""
    SEQUENTIAL = "sequential"      # Agents work one after another
    PARALLEL = "parallel"          # Agents work simultaneously
    ROUND_ROBIN = "round_robin"    # Agents take turns
    HIERARCHICAL = "hierarchical"  # Manager delegates to workers


@dataclass
class TeamConfig:
    """Configuration for a collaborative team."""
    name: str
    mode: CollaborationMode = CollaborationMode.SEQUENTIAL
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD
    allow_voting: bool = True


@dataclass
class TeamMember:
    """A member of a collaborative team."""
    agent: BaseAgent
    role_description: str
    specialties: Set[str] = field(default_factory=set)
    weight: float = 1.0  # Voting weight


@dataclass
class Contribution:
    """A contribution from a team member."""
    agent_id: str
    agent_name: str
    content: str
    iteration: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Result of consensus building."""
    final_answer: str
    agreement_score: float
    iterations: int
    contributions: List[Contribution]
    dissenting_views: List[str] = field(default_factory=list)


class CollaborativeTeam:
    """
    A team of agents working together on tasks.
    
    Example:
        >>> team = CollaborativeTeam(config)
        >>> team.add_member(researcher, "Research specialist")
        >>> team.add_member(writer, "Content writer")
        >>> result = await team.collaborate("Write a report on AI")
    """

    def __init__(self, config: TeamConfig):
        self.config = config
        self._members: List[TeamMember] = []
        self._contributions: List[Contribution] = []

    def add_member(self, agent: BaseAgent, role_description: str, **kwargs) -> None:
        """Add an agent to the team."""
        member = TeamMember(agent=agent, role_description=role_description, **kwargs)
        self._members.append(member)
        logger.info(f"Added {agent.name} to team {self.config.name}")

    def remove_member(self, agent_id: str) -> bool:
        """Remove an agent from the team."""
        for i, m in enumerate(self._members):
            if m.agent.id == agent_id:
                self._members.pop(i)
                return True
        return False

    async def collaborate(self, task: str) -> str:
        """Execute collaborative task based on configured mode."""
        self._contributions = []
        
        if self.config.mode == CollaborationMode.SEQUENTIAL:
            return await self._sequential_collaboration(task)
        elif self.config.mode == CollaborationMode.PARALLEL:
            return await self._parallel_collaboration(task)
        elif self.config.mode == CollaborationMode.ROUND_ROBIN:
            return await self._round_robin_collaboration(task)
        else:
            return await self._sequential_collaboration(task)

    async def _sequential_collaboration(self, task: str) -> str:
        """Agents work sequentially, each building on previous work."""
        current_output = task
        
        for iteration in range(self.config.max_iterations):
            for member in self._members:
                prompt = f"""Task: {task}

Previous work:
{current_output}

Your role: {member.role_description}

Please contribute your expertise to improve or extend this work."""

                response = await member.agent.step(prompt)
                current_output = response.content
                
                self._contributions.append(Contribution(
                    agent_id=member.agent.id,
                    agent_name=member.agent.name,
                    content=response.content,
                    iteration=iteration,
                ))
        
        return current_output

    async def _parallel_collaboration(self, task: str) -> str:
        """Agents work in parallel, results are synthesized."""
        tasks = []
        for member in self._members:
            prompt = f"""Task: {task}

Your role: {member.role_description}

Provide your contribution based on your expertise."""
            tasks.append(member.agent.step(prompt))
        
        responses = await asyncio.gather(*tasks)
        
        for i, (member, response) in enumerate(zip(self._members, responses)):
            self._contributions.append(Contribution(
                agent_id=member.agent.id,
                agent_name=member.agent.name,
                content=response.content,
                iteration=0,
            ))
        
        # Synthesize results
        all_contributions = "\n\n".join([
            f"[{m.agent.name}]: {r.content}"
            for m, r in zip(self._members, responses)
        ])
        
        if self._members:
            synthesizer = self._members[0].agent
            synthesis_prompt = f"""Synthesize these contributions into a coherent response:

{all_contributions}

Provide a unified answer that incorporates the best from each contribution."""
            
            final = await synthesizer.step(synthesis_prompt)
            return final.content
        
        return all_contributions

    async def _round_robin_collaboration(self, task: str) -> str:
        """Agents take turns refining the solution."""
        current = task
        member_idx = 0
        
        for iteration in range(self.config.max_iterations * len(self._members)):
            member = self._members[member_idx % len(self._members)]
            
            prompt = f"""Current solution:
{current}

Your role: {member.role_description}

Review and improve this solution. Focus on your area of expertise."""

            response = await member.agent.step(prompt)
            current = response.content
            
            self._contributions.append(Contribution(
                agent_id=member.agent.id,
                agent_name=member.agent.name,
                content=response.content,
                iteration=iteration,
            ))
            
            member_idx += 1
        
        return current

    def get_contributions(self) -> List[Contribution]:
        return self._contributions.copy()


# =============================================================================
# Consensus Builder
# =============================================================================


class ConsensusBuilder:
    """
    Builds consensus among multiple agents through iterative refinement.
    
    Process:
    1. Each agent provides initial opinion
    2. Agents review others' opinions
    3. Agents revise their positions
    4. Repeat until consensus or max iterations
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        self.agents = agents
        self.threshold = threshold
        self.max_iterations = max_iterations
        self._history: List[Dict[str, str]] = []

    async def build_consensus(self, question: str) -> ConsensusResult:
        """Build consensus on a question."""
        opinions: Dict[str, str] = {}
        
        # Initial opinions
        for agent in self.agents:
            prompt = f"Question: {question}\n\nProvide your opinion:"
            response = await agent.step(prompt)
            opinions[agent.id] = response.content
        
        contributions = [
            Contribution(a.id, a.name, opinions[a.id], 0)
            for a in self.agents
        ]
        
        # Iterative refinement
        for iteration in range(1, self.max_iterations + 1):
            all_opinions = "\n".join([
                f"[{a.name}]: {opinions[a.id]}" for a in self.agents
            ])
            
            new_opinions = {}
            for agent in self.agents:
                prompt = f"""Question: {question}

Current opinions from all participants:
{all_opinions}

Consider others' views and provide your revised opinion. 
If you agree with the emerging consensus, state so clearly."""

                response = await agent.step(prompt)
                new_opinions[agent.id] = response.content
                contributions.append(
                    Contribution(agent.id, agent.name, response.content, iteration)
                )
            
            opinions = new_opinions
            
            # Check for consensus
            agreement = self._calculate_agreement(list(opinions.values()))
            if agreement >= self.threshold:
                break
        
        # Generate final answer
        final = await self._synthesize_consensus(question, opinions)
        
        return ConsensusResult(
            final_answer=final,
            agreement_score=agreement,
            iterations=iteration,
            contributions=contributions,
        )

    def _calculate_agreement(self, opinions: List[str]) -> float:
        """Calculate agreement score (simplified)."""
        if len(opinions) <= 1:
            return 1.0
        # Simplified: check for common keywords
        return 0.7  # Placeholder

    async def _synthesize_consensus(self, question: str, opinions: Dict[str, str]) -> str:
        """Synthesize final consensus answer."""
        if self.agents:
            all_opinions = "\n".join(opinions.values())
            prompt = f"Synthesize this consensus:\n{all_opinions}"
            response = await self.agents[0].step(prompt)
            return response.content
        return list(opinions.values())[0] if opinions else ""


# =============================================================================
# Voting System
# =============================================================================


class VotingSystem:
    """
    Democratic voting system for multi-agent decisions.
    
    Supports:
    - Simple majority
    - Weighted voting
    - Ranked choice
    """

    def __init__(self, agents: List[BaseAgent], weights: Optional[Dict[str, float]] = None):
        self.agents = agents
        self.weights = weights or {a.id: 1.0 for a in agents}

    async def vote(self, question: str, options: List[str]) -> Dict[str, Any]:
        """Conduct a vote among agents."""
        votes: Dict[str, str] = {}
        
        options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        
        for agent in self.agents:
            prompt = f"""Question: {question}

Options:
{options_str}

Vote by responding with ONLY the option number (1, 2, etc.)."""

            response = await agent.step(prompt)
            # Parse vote
            try:
                vote_num = int(response.content.strip().split()[0]) - 1
                if 0 <= vote_num < len(options):
                    votes[agent.id] = options[vote_num]
            except (ValueError, IndexError):
                votes[agent.id] = options[0]  # Default
        
        # Tally votes
        tally: Dict[str, float] = {opt: 0.0 for opt in options}
        for agent_id, vote in votes.items():
            tally[vote] += self.weights.get(agent_id, 1.0)
        
        winner = max(tally, key=tally.get)
        total = sum(tally.values())
        
        return {
            "winner": winner,
            "votes": votes,
            "tally": tally,
            "margin": tally[winner] / total if total > 0 else 0,
        }
