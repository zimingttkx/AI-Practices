"""
Tree of Thoughts (ToT): Deliberate Problem Solving with LLMs

Core Idea:
    Tree of Thoughts extends Chain of Thought by exploring multiple reasoning
    paths simultaneously, using tree search algorithms (BFS/DFS) to find the
    best solution. This enables deliberate planning and backtracking when
    reasoning paths lead to dead ends.

Mathematical Foundation:
    ToT models reasoning as a search problem over a tree:

    $$T = (V, E)$$

    where:
    - $V$ is the set of thought nodes (partial solutions)
    - $E$ is the set of edges (thought transitions)

    Each node $v$ has a value function:
    $$V(v) = f(s_v, g)$$

    where $s_v$ is the state at node $v$ and $g$ is the goal.

    The search objective is:
    $$v^* = \arg\max_{v \in \text{leaves}(T)} V(v)$$

Problem Statement:
    Chain of Thought generates a single reasoning path, which may:
    1. Get stuck in local optima
    2. Cannot backtrack from wrong decisions
    3. Miss better alternative solutions

    ToT addresses these by:
    - Generating multiple thoughts at each step
    - Evaluating thought quality
    - Using search algorithms to explore the tree
    - Backtracking when needed

Algorithm Comparison:
    | Strategy | Exploration | Memory  | Best For                    |
    |----------|-------------|---------|------------------------------|
    | BFS      | Breadth     | O(b^d)  | Shallow solutions            |
    | DFS      | Depth       | O(d)    | Deep solutions               |
    | Beam     | Best-k      | O(k*d)  | Balance exploration/memory   |
    | A*       | Heuristic   | O(b^d)  | Known heuristics             |

References:
    - Yao et al. (2023): "Tree of Thoughts: Deliberate Problem Solving with LLMs"
    - https://arxiv.org/abs/2305.10601
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Final,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)
import uuid


class NodeStatus(str, Enum):
    """Status of a thought node in the tree."""
    PENDING: Final[str] = "pending"      # Not yet expanded
    EXPANDED: Final[str] = "expanded"    # Children generated
    EVALUATED: Final[str] = "evaluated"  # Value computed
    PRUNED: Final[str] = "pruned"        # Discarded
    SOLUTION: Final[str] = "solution"    # Valid solution found


@dataclass
class ThoughtNode:
    """A node in the Tree of Thoughts.

    Core Idea:
        Represents a single thought/reasoning step in the search tree.
        Contains the thought content, evaluation score, and tree structure.

    Attributes:
        thought: The reasoning content at this node.
        value: Evaluation score (higher is better).
        depth: Distance from root node.
        parent: Parent node reference.
        children: List of child nodes.
        status: Current node status.
        metadata: Additional information.

    Example:
        >>> root = ThoughtNode(thought="Start solving the puzzle")
        >>> child = ThoughtNode(thought="Try moving piece A", parent=root)
        >>> root.add_child(child)
    """
    thought: str
    value: float = 0.0
    depth: int = 0
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def add_child(self, child: "ThoughtNode") -> None:
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def get_path(self) -> List["ThoughtNode"]:
        """Get path from root to this node."""
        path = []
        node: Optional[ThoughtNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_path_thoughts(self) -> List[str]:
        """Get thoughts along the path from root."""
        return [node.thought for node in self.get_path()]

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None

    def __lt__(self, other: "ThoughtNode") -> bool:
        """Compare by value (for heap operations)."""
        return self.value > other.value  # Higher value = higher priority

    def __repr__(self) -> str:
        return f"ThoughtNode({self.id}, depth={self.depth}, value={self.value:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "thought": self.thought,
            "value": self.value,
            "depth": self.depth,
            "status": self.status.value,
            "num_children": len(self.children),
            "metadata": self.metadata,
        }


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


class ThoughtGenerator(ABC):
    """Abstract base class for generating candidate thoughts."""

    @abstractmethod
    def generate(
        self,
        node: ThoughtNode,
        problem: str,
        n_candidates: int = 3,
    ) -> List[str]:
        """Generate candidate thoughts from current node.

        Args:
            node: Current node to expand.
            problem: Original problem statement.
            n_candidates: Number of candidates to generate.

        Returns:
            List of candidate thought strings.
        """
        pass


class ThoughtEvaluator(ABC):
    """Abstract base class for evaluating thought quality."""

    @abstractmethod
    def evaluate(
        self,
        node: ThoughtNode,
        problem: str,
    ) -> float:
        """Evaluate the quality of a thought node.

        Args:
            node: Node to evaluate.
            problem: Original problem statement.

        Returns:
            Score between 0.0 and 1.0 (higher is better).
        """
        pass

    @abstractmethod
    def is_solution(
        self,
        node: ThoughtNode,
        problem: str,
    ) -> bool:
        """Check if node represents a valid solution.

        Args:
            node: Node to check.
            problem: Original problem statement.

        Returns:
            True if node is a valid solution.
        """
        pass


class SimpleThoughtGenerator(ThoughtGenerator):
    """Simple thought generator using LLM prompting.

    Generates candidate thoughts by prompting the LLM with the current
    reasoning path and asking for next steps.
    """

    PROMPT_TEMPLATE: Final[str] = """Problem: {problem}

Current reasoning path:
{path}

Generate {n} different possible next steps for solving this problem.
Each step should be a distinct approach or continuation.

Format your response as:
1. [First possible next step]
2. [Second possible next step]
...

Next steps:"""

    def __init__(self, llm: Optional[LLMInterface] = None) -> None:
        self._llm = llm

    def generate(
        self,
        node: ThoughtNode,
        problem: str,
        n_candidates: int = 3,
    ) -> List[str]:
        """Generate candidate thoughts."""
        if self._llm is None:
            # Return placeholder candidates for testing
            return [f"Candidate thought {i+1} from node {node.id}" for i in range(n_candidates)]

        # Build path string
        path = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(node.get_path_thoughts()))

        prompt = self.PROMPT_TEMPLATE.format(
            problem=problem,
            path=path,
            n=n_candidates,
        )

        response = self._llm.generate(prompt)
        return self._parse_candidates(response, n_candidates)

    def _parse_candidates(self, response: str, n: int) -> List[str]:
        """Parse numbered candidates from response."""
        import re
        pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, response, re.DOTALL)
        candidates = [m.strip() for m in matches if m.strip()]
        return candidates[:n] if candidates else [response.strip()]


class SimpleThoughtEvaluator(ThoughtEvaluator):
    """Simple thought evaluator using LLM scoring.

    Evaluates thoughts by asking the LLM to rate them on a scale.
    """

    EVAL_PROMPT: Final[str] = """Problem: {problem}

Reasoning path:
{path}

Evaluate this reasoning path on a scale of 1-10:
- Is it making progress toward the solution?
- Is the logic sound?
- Are there any errors or dead ends?

Respond with just a number from 1 to 10."""

    SOLUTION_PROMPT: Final[str] = """Problem: {problem}

Proposed solution:
{path}

Does this completely solve the problem? Answer YES or NO."""

    def __init__(self, llm: Optional[LLMInterface] = None) -> None:
        self._llm = llm

    def evaluate(self, node: ThoughtNode, problem: str) -> float:
        """Evaluate thought quality."""
        if self._llm is None:
            # Heuristic: prefer shorter paths with content
            base_score = 0.5
            if node.thought:
                base_score += 0.2
            if node.depth < 5:
                base_score += 0.1
            return min(base_score, 1.0)

        path = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(node.get_path_thoughts()))
        prompt = self.EVAL_PROMPT.format(problem=problem, path=path)

        response = self._llm.generate(prompt)
        try:
            score = float(response.strip().split()[0])
            return score / 10.0
        except (ValueError, IndexError):
            return 0.5

    def is_solution(self, node: ThoughtNode, problem: str) -> bool:
        """Check if node is a solution."""
        if self._llm is None:
            # Heuristic: consider solution if depth >= 3 and has "answer" or "solution"
            thought_lower = node.thought.lower()
            return node.depth >= 3 and ("answer" in thought_lower or "solution" in thought_lower)

        path = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(node.get_path_thoughts()))
        prompt = self.SOLUTION_PROMPT.format(problem=problem, path=path)

        response = self._llm.generate(prompt)
        return "yes" in response.lower()


class SearchStrategy(ABC):
    """Abstract base class for tree search strategies."""

    @abstractmethod
    def search(
        self,
        root: ThoughtNode,
        problem: str,
        generator: ThoughtGenerator,
        evaluator: ThoughtEvaluator,
        max_iterations: int = 100,
        n_candidates: int = 3,
    ) -> Optional[ThoughtNode]:
        """Search for a solution starting from root.

        Args:
            root: Root node of the search tree.
            problem: Problem statement.
            generator: Thought generator.
            evaluator: Thought evaluator.
            max_iterations: Maximum search iterations.
            n_candidates: Candidates per expansion.

        Returns:
            Solution node if found, None otherwise.
        """
        pass


class BFSSearch(SearchStrategy):
    """Breadth-First Search strategy.

    Explores all nodes at current depth before moving deeper.
    Good for finding shortest solution paths.
    """

    def search(
        self,
        root: ThoughtNode,
        problem: str,
        generator: ThoughtGenerator,
        evaluator: ThoughtEvaluator,
        max_iterations: int = 100,
        n_candidates: int = 3,
    ) -> Optional[ThoughtNode]:
        queue: Deque[ThoughtNode] = deque([root])
        iterations = 0

        while queue and iterations < max_iterations:
            node = queue.popleft()
            iterations += 1

            # Evaluate current node
            node.value = evaluator.evaluate(node, problem)
            node.status = NodeStatus.EVALUATED

            # Check if solution
            if evaluator.is_solution(node, problem):
                node.status = NodeStatus.SOLUTION
                return node

            # Generate and add children
            candidates = generator.generate(node, problem, n_candidates)
            for thought in candidates:
                child = ThoughtNode(thought=thought)
                node.add_child(child)
                queue.append(child)

            node.status = NodeStatus.EXPANDED

        return None


class DFSSearch(SearchStrategy):
    """Depth-First Search strategy.

    Explores as deep as possible before backtracking.
    Memory efficient, good for deep solutions.
    """

    def __init__(self, max_depth: int = 10) -> None:
        self._max_depth = max_depth

    def search(
        self,
        root: ThoughtNode,
        problem: str,
        generator: ThoughtGenerator,
        evaluator: ThoughtEvaluator,
        max_iterations: int = 100,
        n_candidates: int = 3,
    ) -> Optional[ThoughtNode]:
        stack: List[ThoughtNode] = [root]
        iterations = 0

        while stack and iterations < max_iterations:
            node = stack.pop()
            iterations += 1

            # Skip if too deep
            if node.depth > self._max_depth:
                node.status = NodeStatus.PRUNED
                continue

            # Evaluate
            node.value = evaluator.evaluate(node, problem)
            node.status = NodeStatus.EVALUATED

            # Check if solution
            if evaluator.is_solution(node, problem):
                node.status = NodeStatus.SOLUTION
                return node

            # Generate children (add in reverse for correct order)
            candidates = generator.generate(node, problem, n_candidates)
            for thought in reversed(candidates):
                child = ThoughtNode(thought=thought)
                node.add_child(child)
                stack.append(child)

            node.status = NodeStatus.EXPANDED

        return None


class BeamSearch(SearchStrategy):
    """Beam Search strategy.

    Keeps only top-k nodes at each level.
    Balances exploration and memory usage.
    """

    def __init__(self, beam_width: int = 3) -> None:
        self._beam_width = beam_width

    def search(
        self,
        root: ThoughtNode,
        problem: str,
        generator: ThoughtGenerator,
        evaluator: ThoughtEvaluator,
        max_iterations: int = 100,
        n_candidates: int = 3,
    ) -> Optional[ThoughtNode]:
        current_level: List[ThoughtNode] = [root]
        iterations = 0

        while current_level and iterations < max_iterations:
            # Evaluate all nodes in current level
            for node in current_level:
                node.value = evaluator.evaluate(node, problem)
                node.status = NodeStatus.EVALUATED
                iterations += 1

                if evaluator.is_solution(node, problem):
                    node.status = NodeStatus.SOLUTION
                    return node

            # Generate next level
            next_level: List[ThoughtNode] = []
            for node in current_level:
                candidates = generator.generate(node, problem, n_candidates)
                for thought in candidates:
                    child = ThoughtNode(thought=thought)
                    node.add_child(child)
                    next_level.append(child)
                node.status = NodeStatus.EXPANDED

            # Keep only top-k by parent value (heuristic)
            if len(next_level) > self._beam_width:
                # Evaluate children quickly
                for child in next_level:
                    child.value = evaluator.evaluate(child, problem)
                next_level.sort(key=lambda n: n.value, reverse=True)
                next_level = next_level[:self._beam_width]

            current_level = next_level

        return None


@dataclass
class ToTResult:
    """Result of Tree of Thoughts search."""
    problem: str
    solution_node: Optional[ThoughtNode] = None
    solution_path: List[str] = field(default_factory=list)
    total_nodes: int = 0
    iterations: int = 0
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "solution_path": self.solution_path,
            "total_nodes": self.total_nodes,
            "iterations": self.iterations,
            "success": self.success,
            "metadata": self.metadata,
        }


class TreeOfThoughts:
    """Tree of Thoughts: Deliberate problem solving with search.

    Example:
        >>> tot = TreeOfThoughts()
        >>> result = tot.solve("What is 24 using 1,2,3,4?")
        >>> print(result.solution_path)
    """

    def __init__(
        self,
        generator: Optional[ThoughtGenerator] = None,
        evaluator: Optional[ThoughtEvaluator] = None,
        search_strategy: Optional[SearchStrategy] = None,
        llm: Optional[LLMInterface] = None,
    ) -> None:
        self._llm = llm
        self._generator = generator or SimpleThoughtGenerator(llm)
        self._evaluator = evaluator or SimpleThoughtEvaluator(llm)
        self._search_strategy = search_strategy or BFSSearch()

    def solve(
        self,
        problem: str,
        initial_thought: str = "Let me analyze this problem.",
        max_iterations: int = 100,
        n_candidates: int = 3,
    ) -> ToTResult:
        """Solve a problem using Tree of Thoughts.

        Args:
            problem: Problem statement.
            initial_thought: Starting thought for root node.
            max_iterations: Maximum search iterations.
            n_candidates: Candidates per node expansion.

        Returns:
            ToTResult with solution if found.
        """
        root = ThoughtNode(thought=initial_thought)
        
        solution = self._search_strategy.search(
            root=root,
            problem=problem,
            generator=self._generator,
            evaluator=self._evaluator,
            max_iterations=max_iterations,
            n_candidates=n_candidates,
        )

        result = ToTResult(problem=problem)
        result.total_nodes = self._count_nodes(root)
        
        if solution:
            result.solution_node = solution
            result.solution_path = solution.get_path_thoughts()
            result.success = True

        return result

    def _count_nodes(self, root: ThoughtNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in root.children:
            count += self._count_nodes(child)
        return count


__all__ = [
    "NodeStatus",
    "ThoughtNode",
    "ThoughtGenerator",
    "ThoughtEvaluator",
    "SimpleThoughtGenerator",
    "SimpleThoughtEvaluator",
    "SearchStrategy",
    "BFSSearch",
    "DFSSearch",
    "BeamSearch",
    "ToTResult",
    "TreeOfThoughts",
]
