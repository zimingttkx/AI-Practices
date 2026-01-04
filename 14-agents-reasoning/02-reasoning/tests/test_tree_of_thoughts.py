"""
Unit tests for Tree of Thoughts module.
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

import pytest
from src.tree_of_thoughts import (
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


class TestNodeStatus:
    """Tests for NodeStatus enum."""

    def test_values(self):
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.EXPANDED.value == "expanded"
        assert NodeStatus.EVALUATED.value == "evaluated"
        assert NodeStatus.PRUNED.value == "pruned"
        assert NodeStatus.SOLUTION.value == "solution"


class TestThoughtNode:
    """Tests for ThoughtNode dataclass."""

    def test_creation(self):
        node = ThoughtNode(thought="Test thought")
        assert node.thought == "Test thought"
        assert node.value == 0.0
        assert node.depth == 0
        assert node.status == NodeStatus.PENDING

    def test_add_child(self):
        parent = ThoughtNode(thought="Parent")
        child = ThoughtNode(thought="Child")
        parent.add_child(child)
        
        assert len(parent.children) == 1
        assert child.parent == parent
        assert child.depth == 1

    def test_get_path(self):
        root = ThoughtNode(thought="Root")
        child = ThoughtNode(thought="Child")
        grandchild = ThoughtNode(thought="Grandchild")
        
        root.add_child(child)
        child.add_child(grandchild)
        
        path = grandchild.get_path()
        assert len(path) == 3
        assert path[0].thought == "Root"
        assert path[2].thought == "Grandchild"

    def test_get_path_thoughts(self):
        root = ThoughtNode(thought="Step 1")
        child = ThoughtNode(thought="Step 2")
        root.add_child(child)
        
        thoughts = child.get_path_thoughts()
        assert thoughts == ["Step 1", "Step 2"]

    def test_is_leaf(self):
        node = ThoughtNode(thought="Leaf")
        assert node.is_leaf is True
        
        child = ThoughtNode(thought="Child")
        node.add_child(child)
        assert node.is_leaf is False

    def test_is_root(self):
        root = ThoughtNode(thought="Root")
        assert root.is_root is True
        
        child = ThoughtNode(thought="Child")
        root.add_child(child)
        assert child.is_root is False

    def test_comparison(self):
        node1 = ThoughtNode(thought="High", value=0.9)
        node2 = ThoughtNode(thought="Low", value=0.1)
        # Higher value should be "less than" for heap (max-heap behavior)
        assert node1 < node2

    def test_to_dict(self):
        node = ThoughtNode(thought="Test", value=0.5)
        d = node.to_dict()
        assert d["thought"] == "Test"
        assert d["value"] == 0.5


class TestSimpleThoughtGenerator:
    """Tests for SimpleThoughtGenerator."""

    def test_generate_without_llm(self):
        generator = SimpleThoughtGenerator()
        node = ThoughtNode(thought="Start")
        candidates = generator.generate(node, "Test problem", n_candidates=3)
        
        assert len(candidates) == 3
        assert all("Candidate thought" in c for c in candidates)

    def test_parse_candidates(self):
        generator = SimpleThoughtGenerator()
        response = """1. First option
2. Second option
3. Third option"""
        candidates = generator._parse_candidates(response, 3)
        assert len(candidates) == 3


class TestSimpleThoughtEvaluator:
    """Tests for SimpleThoughtEvaluator."""

    def test_evaluate_without_llm(self):
        evaluator = SimpleThoughtEvaluator()
        node = ThoughtNode(thought="Test thought")
        score = evaluator.evaluate(node, "Problem")
        
        assert 0.0 <= score <= 1.0

    def test_is_solution_without_llm(self):
        evaluator = SimpleThoughtEvaluator()
        
        # Not a solution (depth < 3)
        node1 = ThoughtNode(thought="Test")
        assert evaluator.is_solution(node1, "Problem") is False
        
        # Potential solution (depth >= 3 and contains "answer")
        root = ThoughtNode(thought="Start")
        child1 = ThoughtNode(thought="Step 1")
        child2 = ThoughtNode(thought="Step 2")
        child3 = ThoughtNode(thought="The answer is 42")
        
        root.add_child(child1)
        child1.add_child(child2)
        child2.add_child(child3)
        
        assert evaluator.is_solution(child3, "Problem") is True


class TestBFSSearch:
    """Tests for BFSSearch."""

    def test_creation(self):
        bfs = BFSSearch()
        assert bfs is not None

    def test_search_basic(self):
        bfs = BFSSearch()
        generator = SimpleThoughtGenerator()
        evaluator = SimpleThoughtEvaluator()
        root = ThoughtNode(thought="Start")
        
        # With limited iterations, may not find solution
        result = bfs.search(
            root=root,
            problem="Test",
            generator=generator,
            evaluator=evaluator,
            max_iterations=5,
            n_candidates=2
        )
        # Result may be None with limited iterations
        assert root.status in [NodeStatus.EXPANDED, NodeStatus.EVALUATED]


class TestDFSSearch:
    """Tests for DFSSearch."""

    def test_creation(self):
        dfs = DFSSearch(max_depth=5)
        assert dfs._max_depth == 5

    def test_search_basic(self):
        dfs = DFSSearch(max_depth=3)
        generator = SimpleThoughtGenerator()
        evaluator = SimpleThoughtEvaluator()
        root = ThoughtNode(thought="Start")
        
        result = dfs.search(
            root=root,
            problem="Test",
            generator=generator,
            evaluator=evaluator,
            max_iterations=5,
            n_candidates=2
        )
        assert root.status in [NodeStatus.EXPANDED, NodeStatus.EVALUATED]


class TestBeamSearch:
    """Tests for BeamSearch."""

    def test_creation(self):
        beam = BeamSearch(beam_width=5)
        assert beam._beam_width == 5

    def test_search_basic(self):
        beam = BeamSearch(beam_width=2)
        generator = SimpleThoughtGenerator()
        evaluator = SimpleThoughtEvaluator()
        root = ThoughtNode(thought="Start")
        
        result = beam.search(
            root=root,
            problem="Test",
            generator=generator,
            evaluator=evaluator,
            max_iterations=5,
            n_candidates=2
        )
        assert root.status in [NodeStatus.EXPANDED, NodeStatus.EVALUATED]


class TestToTResult:
    """Tests for ToTResult dataclass."""

    def test_creation(self):
        result = ToTResult(problem="Test problem")
        assert result.problem == "Test problem"
        assert result.success is False
        assert result.solution_path == []

    def test_to_dict(self):
        result = ToTResult(
            problem="Test",
            solution_path=["Step 1", "Step 2"],
            success=True
        )
        d = result.to_dict()
        assert d["problem"] == "Test"
        assert d["success"] is True


class TestTreeOfThoughts:
    """Tests for TreeOfThoughts main class."""

    def test_creation_default(self):
        tot = TreeOfThoughts()
        assert tot._generator is not None
        assert tot._evaluator is not None
        assert tot._search_strategy is not None

    def test_creation_with_strategy(self):
        tot = TreeOfThoughts(search_strategy=DFSSearch(max_depth=5))
        assert isinstance(tot._search_strategy, DFSSearch)

    def test_solve_basic(self):
        tot = TreeOfThoughts()
        result = tot.solve(
            problem="Test problem",
            max_iterations=5,
            n_candidates=2
        )
        
        assert result.problem == "Test problem"
        assert result.total_nodes >= 1

    def test_count_nodes(self):
        tot = TreeOfThoughts()
        root = ThoughtNode(thought="Root")
        child1 = ThoughtNode(thought="Child1")
        child2 = ThoughtNode(thought="Child2")
        root.add_child(child1)
        root.add_child(child2)
        
        count = tot._count_nodes(root)
        assert count == 3
