"""
02-reasoning: LLM Reasoning Strategies Module

This module implements various reasoning strategies for Large Language Models:
- Chain of Thought (CoT): Step-by-step reasoning
- ReAct: Reasoning + Acting framework
- Tree of Thoughts (ToT): Exploration-based reasoning
- Self-Consistency: Multiple reasoning paths with voting
- Reflection: Self-evaluation and correction

References:
    - Chain of Thought: https://arxiv.org/abs/2201.11903
    - ReAct: https://arxiv.org/abs/2210.03629
    - Tree of Thoughts: https://arxiv.org/abs/2305.10601
    - Self-Consistency: https://arxiv.org/abs/2203.11171
    - Reflexion: https://arxiv.org/abs/2303.11366
"""

from .chain_of_thought import (
    CoTStrategy,
    CoTStep,
    CoTExample,
    CoTResult,
    CoTPromptBuilder,
    ZeroShotCoT,
    FewShotCoT,
    AutoCoT,
)
from .react import (
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
from .tree_of_thoughts import (
    NodeStatus,
    ThoughtNode,
    ThoughtGenerator,
    ThoughtEvaluator,
    SimpleThoughtGenerator,
    SimpleThoughtEvaluator,
    SearchStrategy,
    BFSSearch,
    DFSSearch,
    BeamSearch,
    ToTResult,
    TreeOfThoughts,
)
from .self_consistency import (
    SampledPath,
    ConsistencyResult,
    VotingStrategy,
    MajorityVoting,
    WeightedVoting,
    SelfConsistency,
)
from .reflection import (
    EvaluationResult,
    ReflectionStep,
    ReflectionResult,
    SelfEvaluator,
    SimpleSelfEvaluator,
    ErrorDetector,
    CorrectionStrategy,
    SimpleCorrectionStrategy,
    Reflection,
)

__all__ = [
    # Chain of Thought
    "CoTStrategy", "CoTStep", "CoTExample", "CoTResult",
    "CoTPromptBuilder", "ZeroShotCoT", "FewShotCoT", "AutoCoT",
    # ReAct
    "Thought", "Action", "Observation", "ReActStep", "ReActTrace",
    "SimpleTool", "ReActPromptBuilder", "ReActParser", "ReActAgent",
    # Tree of Thoughts
    "NodeStatus", "ThoughtNode", "ThoughtGenerator", "ThoughtEvaluator",
    "SimpleThoughtGenerator", "SimpleThoughtEvaluator", "SearchStrategy",
    "BFSSearch", "DFSSearch", "BeamSearch", "ToTResult", "TreeOfThoughts",
    # Self-Consistency
    "SampledPath", "ConsistencyResult", "VotingStrategy",
    "MajorityVoting", "WeightedVoting", "SelfConsistency",
    # Reflection
    "EvaluationResult", "ReflectionStep", "ReflectionResult",
    "SelfEvaluator", "SimpleSelfEvaluator", "ErrorDetector",
    "CorrectionStrategy", "SimpleCorrectionStrategy", "Reflection",
]
