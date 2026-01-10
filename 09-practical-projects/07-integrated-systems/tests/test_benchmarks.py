"""
性能基准测试单元测试。
"""

import time
import numpy as np
import pytest

from src.rag_benchmark import RAGBenchmark, RAGMetrics, BenchmarkResult
from src.agent_benchmark import AgentBenchmark, AgentMetrics, TaskResult
from src.multimodal_benchmark import MultimodalBenchmark, MultimodalMetrics


# =============================================================================
# RAGBenchmark Tests
# =============================================================================

class TestRAGMetrics:
    """RAGMetrics测试。"""
    
    def test_create_metrics(self):
        metrics = RAGMetrics(retrieval_latency_ms=10.0, recall_at_k=0.8)
        assert metrics.retrieval_latency_ms == 10.0
        assert metrics.recall_at_k == 0.8
    
    def test_to_dict(self):
        metrics = RAGMetrics(throughput_qps=100.0)
        d = metrics.to_dict()
        assert "throughput_qps" in d
        assert d["throughput_qps"] == 100.0


class TestRAGBenchmark:
    """RAGBenchmark测试。"""
    
    @pytest.fixture
    def benchmark(self):
        return RAGBenchmark(warmup_runs=1)
    
    def test_latency_test(self, benchmark):
        def mock_retrieve(query):
            time.sleep(0.001)
            return ["doc1", "doc2"]
        
        queries = ["q1", "q2", "q3"]
        result = benchmark.run_latency_test(mock_retrieve, queries, num_runs=2)
        
        assert result.name == "latency_test"
        assert result.metrics.retrieval_latency_ms > 0
        assert result.num_queries == 6
    
    def test_throughput_test(self, benchmark):
        def mock_retrieve(query):
            return ["doc"]
        
        result = benchmark.run_throughput_test(mock_retrieve, ["q1"], duration_seconds=0.1)
        assert result.metrics.throughput_qps > 0
    
    def test_accuracy_test(self, benchmark):
        def mock_retrieve(query):
            return ["doc1", "doc2", "doc3"]
        
        queries = ["q1", "q2"]
        ground_truth = [["doc1", "doc2"], ["doc2", "doc3"]]
        
        result = benchmark.run_accuracy_test(mock_retrieve, queries, ground_truth, k=3)
        assert result.metrics.recall_at_k > 0
        assert result.metrics.mrr > 0
    
    def test_get_results(self, benchmark):
        def mock_retrieve(query):
            return []
        
        benchmark.run_latency_test(mock_retrieve, ["q"], num_runs=1)
        results = benchmark.get_results()
        assert len(results) == 1
    
    def test_clear_results(self, benchmark):
        def mock_retrieve(query):
            return []
        
        benchmark.run_latency_test(mock_retrieve, ["q"], num_runs=1)
        benchmark.clear_results()
        assert len(benchmark.get_results()) == 0
    
    def test_summary(self, benchmark):
        def mock_retrieve(query):
            return []
        
        benchmark.run_latency_test(mock_retrieve, ["q"], num_runs=1)
        summary = benchmark.summary()
        assert "num_tests" in summary


# =============================================================================
# AgentBenchmark Tests
# =============================================================================

class TestAgentMetrics:
    """AgentMetrics测试。"""
    
    def test_create_metrics(self):
        metrics = AgentMetrics(avg_steps=5.0, success_rate=0.9)
        assert metrics.avg_steps == 5.0
        assert metrics.success_rate == 0.9
    
    def test_to_dict(self):
        metrics = AgentMetrics(avg_latency_ms=100.0)
        d = metrics.to_dict()
        assert d["avg_latency_ms"] == 100.0


class TestTaskResult:
    """TaskResult测试。"""
    
    def test_create_result(self):
        result = TaskResult(
            task_id="t1",
            success=True,
            steps=3,
            latency_ms=50.0,
        )
        assert result.success
        assert result.steps == 3


class TestAgentBenchmark:
    """AgentBenchmark测试。"""
    
    @pytest.fixture
    def benchmark(self):
        return AgentBenchmark()
    
    def test_task_completion_test(self, benchmark):
        def mock_run(task):
            return {"success": True, "steps": 3, "tool_calls": 2}
        
        tasks = ["task1", "task2"]
        result = benchmark.run_task_completion_test(mock_run, tasks)
        
        assert result.metrics.success_rate == 1.0
        assert result.metrics.avg_steps == 3.0
        assert len(result.task_results) == 2
    
    def test_task_with_failure(self, benchmark):
        def mock_run(task):
            if "fail" in task:
                raise ValueError("Task failed")
            return {"success": True, "steps": 1}
        
        tasks = ["ok", "fail"]
        result = benchmark.run_task_completion_test(mock_run, tasks)
        assert result.metrics.success_rate == 0.5
    
    def test_step_efficiency_test(self, benchmark):
        def mock_run(task):
            return {"success": True, "steps": 5}
        
        tasks = ["t1", "t2"]
        optimal = [3, 4]
        result = benchmark.run_step_efficiency_test(mock_run, tasks, optimal)
        assert result.num_tasks == 2
    
    def test_summary(self, benchmark):
        def mock_run(task):
            return {"success": True, "steps": 1}
        
        benchmark.run_task_completion_test(mock_run, ["t"])
        summary = benchmark.summary()
        assert "avg_success_rate" in summary


# =============================================================================
# MultimodalBenchmark Tests
# =============================================================================

class TestMultimodalMetrics:
    """MultimodalMetrics测试。"""
    
    def test_create_metrics(self):
        metrics = MultimodalMetrics(image_encode_ms=5.0, text_encode_ms=2.0)
        assert metrics.image_encode_ms == 5.0
    
    def test_to_dict(self):
        metrics = MultimodalMetrics(cross_modal_retrieval_ms=10.0)
        d = metrics.to_dict()
        assert d["cross_modal_retrieval_ms"] == 10.0


class TestMultimodalBenchmark:
    """MultimodalBenchmark测试。"""
    
    @pytest.fixture
    def benchmark(self):
        return MultimodalBenchmark(warmup_runs=1)
    
    def test_encoding_test(self, benchmark):
        def img_encode(img):
            return np.random.rand(128)
        
        def txt_encode(txt):
            return np.random.rand(128)
        
        images = [np.random.rand(32, 32, 3) for _ in range(3)]
        texts = ["text1", "text2", "text3"]
        
        result = benchmark.run_encoding_test(img_encode, txt_encode, images, texts)
        assert result.metrics.image_encode_ms > 0
        assert result.metrics.text_encode_ms > 0
    
    def test_cross_modal_test(self, benchmark):
        def search(images, query):
            return [0, 1, 2]
        
        images = [np.random.rand(32, 32, 3) for _ in range(5)]
        queries = ["q1", "q2"]
        
        result = benchmark.run_cross_modal_test(search, images, queries)
        assert result.metrics.cross_modal_retrieval_ms > 0
    
    def test_similarity_test(self, benchmark):
        def similarity(img, txt):
            return 0.8
        
        pairs = [(np.random.rand(32, 32, 3), "text") for _ in range(3)]
        result = benchmark.run_similarity_test(similarity, pairs)
        assert np.isclose(result.metrics.image_text_similarity, 0.8)
    
    def test_get_results(self, benchmark):
        def sim(img, txt):
            return 0.5
        
        benchmark.run_similarity_test(sim, [(np.zeros((2, 2)), "t")])
        assert len(benchmark.get_results()) == 1
    
    def test_clear_results(self, benchmark):
        def sim(img, txt):
            return 0.5
        
        benchmark.run_similarity_test(sim, [(np.zeros((2, 2)), "t")])
        benchmark.clear_results()
        assert len(benchmark.get_results()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
