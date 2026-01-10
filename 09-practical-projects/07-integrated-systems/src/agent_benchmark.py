"""
Agent性能基准测试。

测试智能体系统的执行效率和任务完成率。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class AgentMetrics:
    """Agent性能指标。"""
    avg_steps: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 0.0
    avg_tool_calls: float = 0.0
    timeout_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "avg_steps": self.avg_steps,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "avg_tool_calls": self.avg_tool_calls,
            "timeout_rate": self.timeout_rate,
        }


@dataclass
class TaskResult:
    """任务执行结果。"""
    task_id: str
    success: bool
    steps: int
    latency_ms: float
    tool_calls: int = 0
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """基准测试结果。"""
    name: str
    metrics: AgentMetrics
    num_tasks: int
    duration_seconds: float
    task_results: List[TaskResult] = field(default_factory=list)


class AgentBenchmark:
    """Agent性能基准测试。
    
    示例:
        >>> benchmark = AgentBenchmark()
        >>> result = benchmark.run_task_completion_test(agent, tasks)
    """
    
    def __init__(self, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._results: List[BenchmarkResult] = []
    
    def run_task_completion_test(
        self,
        run_func: Callable[[str], Dict[str, Any]],
        tasks: List[str],
    ) -> BenchmarkResult:
        """测试任务完成率。"""
        task_results = []
        start_time = time.time()
        
        for i, task in enumerate(tasks):
            t0 = time.time()
            try:
                result = run_func(task)
                latency = (time.time() - t0) * 1000
                
                task_results.append(TaskResult(
                    task_id=f"task_{i}",
                    success=result.get("success", True),
                    steps=result.get("steps", 1),
                    latency_ms=latency,
                    tool_calls=result.get("tool_calls", 0),
                ))
            except Exception as e:
                task_results.append(TaskResult(
                    task_id=f"task_{i}",
                    success=False,
                    steps=0,
                    latency_ms=(time.time() - t0) * 1000,
                    error=str(e),
                ))
        
        duration = time.time() - start_time
        
        # 计算指标
        successful = [r for r in task_results if r.success]
        metrics = AgentMetrics(
            avg_steps=np.mean([r.steps for r in successful]) if successful else 0,
            avg_latency_ms=np.mean([r.latency_ms for r in task_results]),
            success_rate=len(successful) / len(task_results) if task_results else 0,
            avg_tool_calls=np.mean([r.tool_calls for r in successful]) if successful else 0,
        )
        
        result = BenchmarkResult(
            name="task_completion_test",
            metrics=metrics,
            num_tasks=len(tasks),
            duration_seconds=duration,
            task_results=task_results,
        )
        self._results.append(result)
        return result
    
    def run_step_efficiency_test(
        self,
        run_func: Callable[[str], Dict[str, Any]],
        tasks: List[str],
        optimal_steps: List[int],
    ) -> BenchmarkResult:
        """测试步骤效率。"""
        task_results = []
        efficiencies = []
        start_time = time.time()
        
        for i, (task, opt_steps) in enumerate(zip(tasks, optimal_steps)):
            t0 = time.time()
            result = run_func(task)
            latency = (time.time() - t0) * 1000
            
            actual_steps = result.get("steps", 1)
            efficiency = opt_steps / actual_steps if actual_steps > 0 else 0
            efficiencies.append(efficiency)
            
            task_results.append(TaskResult(
                task_id=f"task_{i}",
                success=result.get("success", True),
                steps=actual_steps,
                latency_ms=latency,
            ))
        
        duration = time.time() - start_time
        
        metrics = AgentMetrics(
            avg_steps=np.mean([r.steps for r in task_results]),
            avg_latency_ms=np.mean([r.latency_ms for r in task_results]),
            success_rate=np.mean(efficiencies),
        )
        
        result = BenchmarkResult(
            name="step_efficiency_test",
            metrics=metrics,
            num_tasks=len(tasks),
            duration_seconds=duration,
            task_results=task_results,
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
            "avg_success_rate": np.mean([
                r.metrics.success_rate for r in self._results
            ]),
            "avg_latency_ms": np.mean([
                r.metrics.avg_latency_ms for r in self._results
            ]),
        }
