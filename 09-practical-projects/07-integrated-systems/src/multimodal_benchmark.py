"""
多模态性能基准测试。

测试多模态系统的编码、检索和生成性能。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class MultimodalMetrics:
    """多模态性能指标。"""
    image_encode_ms: float = 0.0
    text_encode_ms: float = 0.0
    cross_modal_retrieval_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    image_text_similarity: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "image_encode_ms": self.image_encode_ms,
            "text_encode_ms": self.text_encode_ms,
            "cross_modal_retrieval_ms": self.cross_modal_retrieval_ms,
            "fusion_latency_ms": self.fusion_latency_ms,
            "image_text_similarity": self.image_text_similarity,
        }


@dataclass
class BenchmarkResult:
    """基准测试结果。"""
    name: str
    metrics: MultimodalMetrics
    num_samples: int
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)


class MultimodalBenchmark:
    """多模态性能基准测试。
    
    示例:
        >>> benchmark = MultimodalBenchmark()
        >>> result = benchmark.run_encoding_test(encoder, images, texts)
    """
    
    def __init__(self, warmup_runs: int = 3) -> None:
        self.warmup_runs = warmup_runs
        self._results: List[BenchmarkResult] = []
    
    def run_encoding_test(
        self,
        image_encode_func: Callable[[np.ndarray], np.ndarray],
        text_encode_func: Callable[[str], np.ndarray],
        images: List[np.ndarray],
        texts: List[str],
    ) -> BenchmarkResult:
        """测试编码性能。"""
        # 预热
        for img in images[:self.warmup_runs]:
            image_encode_func(img)
        for txt in texts[:self.warmup_runs]:
            text_encode_func(txt)
        
        # 图像编码
        image_latencies = []
        for img in images:
            t0 = time.time()
            image_encode_func(img)
            image_latencies.append((time.time() - t0) * 1000)
        
        # 文本编码
        text_latencies = []
        for txt in texts:
            t0 = time.time()
            text_encode_func(txt)
            text_latencies.append((time.time() - t0) * 1000)
        
        metrics = MultimodalMetrics(
            image_encode_ms=np.mean(image_latencies),
            text_encode_ms=np.mean(text_latencies),
        )
        
        result = BenchmarkResult(
            name="encoding_test",
            metrics=metrics,
            num_samples=len(images) + len(texts),
            duration_seconds=sum(image_latencies + text_latencies) / 1000,
            details={
                "image_p95_ms": np.percentile(image_latencies, 95),
                "text_p95_ms": np.percentile(text_latencies, 95),
            },
        )
        self._results.append(result)
        return result
    
    def run_cross_modal_test(
        self,
        search_func: Callable[[Any, str], List[Any]],
        images: List[np.ndarray],
        queries: List[str],
    ) -> BenchmarkResult:
        """测试跨模态检索。"""
        latencies = []
        start_time = time.time()
        
        for query in queries:
            t0 = time.time()
            search_func(images, query)
            latencies.append((time.time() - t0) * 1000)
        
        duration = time.time() - start_time
        
        metrics = MultimodalMetrics(
            cross_modal_retrieval_ms=np.mean(latencies),
        )
        
        result = BenchmarkResult(
            name="cross_modal_test",
            metrics=metrics,
            num_samples=len(queries),
            duration_seconds=duration,
            details={
                "p50_ms": np.percentile(latencies, 50),
                "p95_ms": np.percentile(latencies, 95),
            },
        )
        self._results.append(result)
        return result
    
    def run_similarity_test(
        self,
        similarity_func: Callable[[np.ndarray, str], float],
        image_text_pairs: List[tuple],
    ) -> BenchmarkResult:
        """测试图文相似度。"""
        similarities = []
        latencies = []
        
        for img, txt in image_text_pairs:
            t0 = time.time()
            sim = similarity_func(img, txt)
            latencies.append((time.time() - t0) * 1000)
            similarities.append(sim)
        
        metrics = MultimodalMetrics(
            image_text_similarity=np.mean(similarities),
            fusion_latency_ms=np.mean(latencies),
        )
        
        result = BenchmarkResult(
            name="similarity_test",
            metrics=metrics,
            num_samples=len(image_text_pairs),
            duration_seconds=sum(latencies) / 1000,
            details={
                "sim_std": np.std(similarities),
                "sim_min": np.min(similarities),
                "sim_max": np.max(similarities),
            },
        )
        self._results.append(result)
        return result
    
    def get_results(self) -> List[BenchmarkResult]:
        return self._results.copy()
    
    def clear_results(self) -> None:
        self._results.clear()
    
    def summary(self) -> Dict[str, Any]:
        if not self._results:
            return {}
        
        return {
            "num_tests": len(self._results),
            "tests": [r.name for r in self._results],
        }
