"""
RAG性能基准测试。

测试检索增强生成系统的延迟、吞吐量和准确率。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class RAGMetrics:
    """RAG性能指标。"""
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "throughput_qps": self.throughput_qps,
            "recall@k": self.recall_at_k,
            "precision@k": self.precision_at_k,
            "mrr": self.mrr,
        }


@dataclass
class BenchmarkResult:
    """基准测试结果。"""
    name: str
    metrics: RAGMetrics
    num_queries: int
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)


class RAGBenchmark:
    """RAG性能基准测试。
    
    示例:
        >>> benchmark = RAGBenchmark()
        >>> result = benchmark.run_latency_test(retriever, queries)
    """
    
    def __init__(self, warmup_runs: int = 3) -> None:
        self.warmup_runs = warmup_runs
        self._results: List[BenchmarkResult] = []
    
    def run_latency_test(
        self,
        retrieve_func: Callable[[str], Any],
        queries: List[str],
        num_runs: int = 10,
    ) -> BenchmarkResult:
        """测试检索延迟。"""
        # 预热
        for q in queries[:self.warmup_runs]:
            retrieve_func(q)
        
        latencies = []
        start_time = time.time()
        
        for _ in range(num_runs):
            for query in queries:
                t0 = time.time()
                retrieve_func(query)
                latencies.append((time.time() - t0) * 1000)
        
        duration = time.time() - start_time
        total_queries = num_runs * len(queries)
        
        metrics = RAGMetrics(
            retrieval_latency_ms=np.mean(latencies),
            total_latency_ms=np.mean(latencies),
            throughput_qps=total_queries / duration if duration > 0 else 0,
        )
        
        result = BenchmarkResult(
            name="latency_test",
            metrics=metrics,
            num_queries=total_queries,
            duration_seconds=duration,
            details={
                "p50_ms": np.percentile(latencies, 50),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "min_ms": np.min(latencies),
                "max_ms": np.max(latencies),
            },
        )
        self._results.append(result)
        return result
    
    def run_throughput_test(
        self,
        retrieve_func: Callable[[str], Any],
        queries: List[str],
        duration_seconds: float = 10.0,
    ) -> BenchmarkResult:
        """测试吞吐量。"""
        query_count = 0
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            for query in queries:
                retrieve_func(query)
                query_count += 1
                if time.time() - start_time >= duration_seconds:
                    break
        
        actual_duration = time.time() - start_time
        qps = query_count / actual_duration if actual_duration > 0 else 0
        
        metrics = RAGMetrics(throughput_qps=qps)
        
        result = BenchmarkResult(
            name="throughput_test",
            metrics=metrics,
            num_queries=query_count,
            duration_seconds=actual_duration,
        )
        self._results.append(result)
        return result
    
    def run_accuracy_test(
        self,
        retrieve_func: Callable[[str], List[str]],
        queries: List[str],
        ground_truth: List[List[str]],
        k: int = 10,
    ) -> BenchmarkResult:
        """测试检索准确率。"""
        recalls = []
        precisions = []
        mrrs = []
        
        for query, relevant_ids in zip(queries, ground_truth):
            retrieved = retrieve_func(query)[:k]
            
            # Recall@K
            hits = len(set(retrieved) & set(relevant_ids))
            recall = hits / len(relevant_ids) if relevant_ids else 0
            recalls.append(recall)
            
            # Precision@K
            precision = hits / len(retrieved) if retrieved else 0
            precisions.append(precision)
            
            # MRR
            rr = 0.0
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant_ids:
                    rr = 1.0 / (i + 1)
                    break
            mrrs.append(rr)
        
        metrics = RAGMetrics(
            recall_at_k=np.mean(recalls),
            precision_at_k=np.mean(precisions),
            mrr=np.mean(mrrs),
        )
        
        result = BenchmarkResult(
            name="accuracy_test",
            metrics=metrics,
            num_queries=len(queries),
            duration_seconds=0,
            details={"k": k},
        )
        self._results.append(result)
        return result
    
    def get_results(self) -> List[BenchmarkResult]:
        return self._results.copy()
    
    def clear_results(self) -> None:
        self._results.clear()
    
    def summary(self) -> Dict[str, Any]:
        """生成测试摘要。"""
        if not self._results:
            return {}
        
        return {
            "num_tests": len(self._results),
            "tests": [r.name for r in self._results],
            "avg_latency_ms": np.mean([
                r.metrics.total_latency_ms for r in self._results
                if r.metrics.total_latency_ms > 0
            ]) if any(r.metrics.total_latency_ms > 0 for r in self._results) else 0,
        }
