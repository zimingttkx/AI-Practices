"""
Continuous Batching: 连续批处理实现

============================================================
核心思想 (Core Idea)
============================================================
传统静态批处理在所有序列完成后才处理下一批，导致 GPU 利用率低。
连续批处理 (Continuous Batching / Iteration-Level Scheduling) 
在每次迭代后动态调整批次，已完成的序列立即移出，新请求立即加入。

============================================================
关键创新 (Key Innovations)
============================================================
1. 迭代级调度: 每个 decode step 后重新调度
2. Prefill-Decode 分离: 区分首次处理和增量生成
3. 动态批次: 批次大小随请求动态变化
4. 内存感知: 根据 KV Cache 内存动态调整

============================================================
性能提升 (Performance Improvement)
============================================================
- 吞吐量提升: 2-3x (相比静态批处理)
- GPU 利用率: >90% (相比 50-60%)
- 延迟优化: 短序列不必等待长序列

============================================================
参考文献 (References)
============================================================
[1] Yu, G., et al. (2022). ORCA: A Distributed Serving System for 
    Transformer-Based Generative Models. OSDI 2022.
[2] vLLM: https://github.com/vllm-project/vllm
[3] TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
"""

from __future__ import annotations

import time
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np

__all__ = [
    "RequestStatus",
    "Request",
    "RequestQueue",
    "SchedulerConfig",
    "BatchScheduler",
    "ContinuousBatcher",
    "SchedulingPolicy",
    "create_continuous_batcher",
]


# =============================================================================
# 配置和枚举
# =============================================================================

class RequestStatus(Enum):
    """请求状态枚举"""
    PENDING = auto()      # 等待处理
    RUNNING = auto()      # 正在处理
    PREEMPTED = auto()    # 被抢占
    COMPLETED = auto()    # 已完成
    FAILED = auto()       # 失败
    CANCELLED = auto()    # 已取消


class SchedulingPolicy(Enum):
    """调度策略枚举"""
    FCFS = "fcfs"         # 先来先服务
    SJF = "sjf"           # 最短作业优先
    PRIORITY = "priority"  # 优先级调度


@dataclass
class SchedulerConfig:
    """调度器配置。
    
    Args:
        max_batch_size: 最大批次大小
        max_num_sequences: 最大并发序列数
        max_tokens_per_batch: 每批次最大 token 数
        max_sequence_length: 最大序列长度
        block_size: KV Cache 块大小
        num_blocks: 总块数
        scheduling_policy: 调度策略
        preemption_mode: 抢占模式 (swap/recompute)
        enable_chunked_prefill: 是否启用分块 prefill
        max_prefill_tokens: 单次 prefill 最大 token 数
    """
    max_batch_size: int = 256
    max_num_sequences: int = 256
    max_tokens_per_batch: int = 8192
    max_sequence_length: int = 4096
    block_size: int = 16
    num_blocks: int = 1024
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.FCFS
    preemption_mode: str = "swap"
    enable_chunked_prefill: bool = False
    max_prefill_tokens: int = 512
    
    def __post_init__(self):
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be positive")
        if isinstance(self.scheduling_policy, str):
            self.scheduling_policy = SchedulingPolicy(self.scheduling_policy)


# =============================================================================
# Request 数据结构
# =============================================================================

@dataclass
class SamplingParams:
    """采样参数。"""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    min_tokens: int = 0
    stop_sequences: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None


@dataclass
class Request:
    """推理请求。
    
    Attributes:
        request_id: 请求唯一标识
        prompt: 输入提示文本
        prompt_token_ids: 输入 token IDs
        sampling_params: 采样参数
        arrival_time: 到达时间
        status: 请求状态
        priority: 优先级 (数值越小优先级越高)
        output_token_ids: 生成的 token IDs
        num_computed_tokens: 已计算的 token 数
        sequence_id: 关联的序列 ID (用于 KV Cache)
    """
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING
    priority: int = 0
    output_token_ids: List[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    sequence_id: Optional[int] = None
    
    # 统计信息
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    prefill_time: Optional[float] = None
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"req_{id(self)}_{time.time()}"
    
    @property
    def prompt_length(self) -> int:
        """获取 prompt 长度"""
        return len(self.prompt_token_ids)
    
    @property
    def output_length(self) -> int:
        """获取已生成的输出长度"""
        return len(self.output_token_ids)
    
    @property
    def total_length(self) -> int:
        """获取总长度 (prompt + output)"""
        return self.prompt_length + self.output_length
    
    @property
    def remaining_tokens(self) -> int:
        """获取剩余可生成的 token 数"""
        return self.sampling_params.max_tokens - self.output_length
    
    @property
    def is_finished(self) -> bool:
        """检查是否已完成"""
        return self.status in (
            RequestStatus.COMPLETED,
            RequestStatus.FAILED,
            RequestStatus.CANCELLED
        )
    
    @property
    def is_prefill_done(self) -> bool:
        """检查 prefill 是否完成"""
        return self.num_computed_tokens >= self.prompt_length
    
    def get_latency(self) -> Optional[float]:
        """获取请求延迟 (秒)"""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def get_time_to_first_token(self) -> Optional[float]:
        """获取首 token 延迟 (秒)"""
        if self.start_time is None or self.prefill_time is None:
            return None
        return self.prefill_time - self.start_time
    
    def mark_running(self) -> None:
        """标记为运行中"""
        self.status = RequestStatus.RUNNING
        if self.start_time is None:
            self.start_time = time.time()
    
    def mark_completed(self) -> None:
        """标记为已完成"""
        self.status = RequestStatus.COMPLETED
        self.end_time = time.time()
    
    def mark_failed(self, reason: str = "") -> None:
        """标记为失败"""
        self.status = RequestStatus.FAILED
        self.end_time = time.time()
    
    def add_token(self, token_id: int) -> None:
        """添加生成的 token"""
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1
    
    def should_stop(self, token_id: int) -> bool:
        """检查是否应该停止生成"""
        # 检查最大长度
        if self.output_length >= self.sampling_params.max_tokens:
            return True
        # 检查停止 token
        if token_id in self.sampling_params.stop_token_ids:
            return True
        return False
    
    def __lt__(self, other: "Request") -> bool:
        """用于优先级队列比较"""
        return (self.priority, self.arrival_time) < (other.priority, other.arrival_time)


# =============================================================================
# Request Queue 实现
# =============================================================================

class RequestQueue:
    """请求队列：管理待处理请求。
    
    支持多种调度策略：FCFS、SJF、优先级调度。
    
    Attributes:
        policy: 调度策略
        _queue: 内部队列
        _request_map: 请求 ID 到请求的映射
    """
    
    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.FCFS):
        self.policy = policy
        self._queue: List[Request] = []
        self._request_map: Dict[str, Request] = {}
    
    def __len__(self) -> int:
        return len(self._queue)
    
    def __bool__(self) -> bool:
        return len(self._queue) > 0
    
    def add(self, request: Request) -> None:
        """添加请求到队列。"""
        if request.request_id in self._request_map:
            raise ValueError(f"Request {request.request_id} already exists")
        
        self._request_map[request.request_id] = request
        
        if self.policy == SchedulingPolicy.FCFS:
            self._queue.append(request)
        elif self.policy == SchedulingPolicy.SJF:
            # 按 prompt 长度排序 (最短优先)
            heapq.heappush(self._queue, (request.prompt_length, request.arrival_time, request))
        elif self.policy == SchedulingPolicy.PRIORITY:
            heapq.heappush(self._queue, request)
    
    def pop(self) -> Optional[Request]:
        """弹出下一个请求。"""
        if not self._queue:
            return None
        
        if self.policy == SchedulingPolicy.FCFS:
            request = self._queue.pop(0)
        elif self.policy == SchedulingPolicy.SJF:
            _, _, request = heapq.heappop(self._queue)
        elif self.policy == SchedulingPolicy.PRIORITY:
            request = heapq.heappop(self._queue)
        else:
            request = self._queue.pop(0)
        
        del self._request_map[request.request_id]
        return request
    
    def peek(self) -> Optional[Request]:
        """查看下一个请求但不移除。"""
        if not self._queue:
            return None
        
        if self.policy == SchedulingPolicy.FCFS:
            return self._queue[0]
        elif self.policy == SchedulingPolicy.SJF:
            return self._queue[0][2]
        elif self.policy == SchedulingPolicy.PRIORITY:
            return self._queue[0]
        return self._queue[0]
    
    def remove(self, request_id: str) -> Optional[Request]:
        """移除指定请求。"""
        if request_id not in self._request_map:
            return None
        
        request = self._request_map.pop(request_id)
        
        if self.policy == SchedulingPolicy.FCFS:
            self._queue.remove(request)
        elif self.policy == SchedulingPolicy.SJF:
            self._queue = [(l, t, r) for l, t, r in self._queue if r.request_id != request_id]
            heapq.heapify(self._queue)
        elif self.policy == SchedulingPolicy.PRIORITY:
            self._queue = [r for r in self._queue if r.request_id != request_id]
            heapq.heapify(self._queue)
        
        return request
    
    def get(self, request_id: str) -> Optional[Request]:
        """获取指定请求。"""
        return self._request_map.get(request_id)
    
    def clear(self) -> None:
        """清空队列。"""
        self._queue.clear()
        self._request_map.clear()
    
    def get_all(self) -> List[Request]:
        """获取所有请求。"""
        if self.policy == SchedulingPolicy.SJF:
            return [r for _, _, r in self._queue]
        return list(self._queue)


# =============================================================================
# Batch Scheduler 实现
# =============================================================================

@dataclass
class SchedulerOutput:
    """调度器输出。"""
    prefill_requests: List[Request]  # 需要 prefill 的请求
    decode_requests: List[Request]   # 需要 decode 的请求
    preempted_requests: List[Request]  # 被抢占的请求
    num_batched_tokens: int  # 批次中的总 token 数
    
    @property
    def is_empty(self) -> bool:
        return not self.prefill_requests and not self.decode_requests


class BatchScheduler:
    """批次调度器：决定每次迭代处理哪些请求。
    
    实现迭代级调度，支持：
    1. Prefill-Decode 分离
    2. 内存感知调度
    3. 抢占机制
    
    Attributes:
        config: 调度器配置
        waiting_queue: 等待队列
        running_requests: 正在运行的请求
        preempted_requests: 被抢占的请求
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.waiting_queue = RequestQueue(config.scheduling_policy)
        self.running_requests: Dict[str, Request] = {}
        self.preempted_requests: List[Request] = []
        
        # 内存管理
        self._allocated_blocks: Dict[str, int] = {}  # request_id -> num_blocks
        self._total_allocated_blocks: int = 0
    
    def add_request(self, request: Request) -> None:
        """添加新请求到等待队列。"""
        self.waiting_queue.add(request)
    
    def abort_request(self, request_id: str) -> Optional[Request]:
        """中止请求。"""
        # 从等待队列移除
        request = self.waiting_queue.remove(request_id)
        if request:
            request.status = RequestStatus.CANCELLED
            return request
        
        # 从运行中移除
        if request_id in self.running_requests:
            request = self.running_requests.pop(request_id)
            request.status = RequestStatus.CANCELLED
            self._free_blocks(request_id)
            return request
        
        return None
    
    def schedule(self) -> SchedulerOutput:
        """执行一次调度，返回本次迭代要处理的请求。"""
        prefill_requests = []
        decode_requests = []
        preempted = []
        num_batched_tokens = 0
        
        # 1. 处理被抢占的请求 (优先恢复)
        while self.preempted_requests:
            request = self.preempted_requests[0]
            if self._can_schedule(request, is_prefill=False):
                self.preempted_requests.pop(0)
                request.status = RequestStatus.RUNNING
                self.running_requests[request.request_id] = request
                decode_requests.append(request)
                num_batched_tokens += 1
            else:
                break
        
        # 2. 处理正在运行的请求 (decode)
        for request in list(self.running_requests.values()):
            if request.is_prefill_done:
                decode_requests.append(request)
                num_batched_tokens += 1
        
        # 3. 调度新请求 (prefill)
        while self.waiting_queue:
            request = self.waiting_queue.peek()
            if request is None:
                break
            
            # 检查是否可以调度
            tokens_needed = request.prompt_length
            if self.config.enable_chunked_prefill:
                tokens_needed = min(tokens_needed, self.config.max_prefill_tokens)
            
            if not self._can_schedule(request, is_prefill=True, tokens=tokens_needed):
                # 尝试抢占
                if self._should_preempt(request):
                    preempted_req = self._preempt_one()
                    if preempted_req:
                        preempted.append(preempted_req)
                        continue
                break
            
            # 调度请求
            self.waiting_queue.pop()
            request.mark_running()
            self.running_requests[request.request_id] = request
            self._allocate_blocks(request)
            prefill_requests.append(request)
            num_batched_tokens += tokens_needed
            
            # 检查批次限制
            if len(prefill_requests) + len(decode_requests) >= self.config.max_batch_size:
                break
            if num_batched_tokens >= self.config.max_tokens_per_batch:
                break
        
        return SchedulerOutput(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            preempted_requests=preempted,
            num_batched_tokens=num_batched_tokens
        )
    
    def _can_schedule(
        self,
        request: Request,
        is_prefill: bool,
        tokens: Optional[int] = None
    ) -> bool:
        """检查是否可以调度请求。"""
        # 检查序列数限制
        if len(self.running_requests) >= self.config.max_num_sequences:
            return False
        
        # 检查内存
        if is_prefill:
            tokens = tokens or request.prompt_length
            blocks_needed = (tokens + self.config.block_size - 1) // self.config.block_size
            if self._total_allocated_blocks + blocks_needed > self.config.num_blocks:
                return False
        
        return True
    
    def _should_preempt(self, waiting_request: Request) -> bool:
        """判断是否应该抢占。"""
        if not self.running_requests:
            return False
        # 基于优先级的抢占策略
        if waiting_request.priority < min(r.priority for r in self.running_requests.values()):
            return True
        return False
    
    def _preempt_one(self) -> Optional[Request]:
        """抢占一个运行中的请求。"""
        if not self.running_requests:
            return None
        
        # 选择优先级最低的请求
        victim = max(self.running_requests.values(), key=lambda r: (r.priority, -r.output_length))
        
        del self.running_requests[victim.request_id]
        victim.status = RequestStatus.PREEMPTED
        self._free_blocks(victim.request_id)
        self.preempted_requests.append(victim)
        
        return victim
    
    def _allocate_blocks(self, request: Request) -> None:
        """为请求分配内存块。"""
        tokens = request.prompt_length + request.sampling_params.max_tokens
        blocks = (tokens + self.config.block_size - 1) // self.config.block_size
        self._allocated_blocks[request.request_id] = blocks
        self._total_allocated_blocks += blocks
    
    def _free_blocks(self, request_id: str) -> None:
        """释放请求的内存块。"""
        if request_id in self._allocated_blocks:
            blocks = self._allocated_blocks.pop(request_id)
            self._total_allocated_blocks -= blocks
    
    def finish_request(self, request_id: str) -> Optional[Request]:
        """完成请求。"""
        if request_id not in self.running_requests:
            return None
        
        request = self.running_requests.pop(request_id)
        request.mark_completed()
        self._free_blocks(request_id)
        return request
    
    def get_num_waiting(self) -> int:
        """获取等待中的请求数。"""
        return len(self.waiting_queue)
    
    def get_num_running(self) -> int:
        """获取运行中的请求数。"""
        return len(self.running_requests)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息。"""
        return {
            "num_waiting": len(self.waiting_queue),
            "num_running": len(self.running_requests),
            "num_preempted": len(self.preempted_requests),
            "allocated_blocks": self._total_allocated_blocks,
            "free_blocks": self.config.num_blocks - self._total_allocated_blocks,
            "memory_utilization": self._total_allocated_blocks / self.config.num_blocks,
        }


# =============================================================================
# Continuous Batcher 实现
# =============================================================================

@dataclass
class BatcherOutput:
    """批处理器输出。"""
    request_id: str
    prompt: str
    generated_text: str
    output_token_ids: List[int]
    finish_reason: str  # "length", "stop", "error"
    latency_ms: float
    time_to_first_token_ms: Optional[float]
    num_prompt_tokens: int
    num_generated_tokens: int


class ContinuousBatcher:
    """连续批处理器：实现迭代级调度的推理引擎。
    
    核心功能：
    1. 接收请求并加入队列
    2. 每次迭代动态调度批次
    3. 执行 prefill 和 decode
    4. 返回完成的请求
    
    Attributes:
        config: 调度器配置
        scheduler: 批次调度器
        model_runner: 模型执行器 (可选)
        completed_requests: 已完成的请求
    """
    
    def __init__(
        self,
        config: SchedulerConfig,
        model_runner: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.config = config
        self.scheduler = BatchScheduler(config)
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.completed_requests: List[BatcherOutput] = []
        
        # 统计信息
        self._total_requests = 0
        self._total_tokens_generated = 0
        self._total_prefill_tokens = 0
        self._start_time = time.time()
    
    def add_request(
        self,
        prompt: str,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """添加新请求。
        
        Args:
            prompt: 输入提示
            prompt_token_ids: 输入 token IDs (如果不提供则使用 tokenizer)
            sampling_params: 采样参数
            request_id: 请求 ID
            priority: 优先级
            
        Returns:
            请求 ID
        """
        if prompt_token_ids is None:
            if self.tokenizer is not None:
                prompt_token_ids = self.tokenizer.encode(prompt)
            else:
                # 使用简单的空格分词作为后备方案
                prompt_token_ids = list(range(len(prompt.split())))
        
        if request_id is None:
            request_id = f"req_{self._total_requests}_{time.time()}"
        
        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params or SamplingParams(),
            priority=priority
        )
        
        self.scheduler.add_request(request)
        self._total_requests += 1
        
        return request_id
    
    def abort_request(self, request_id: str) -> bool:
        """中止请求。"""
        request = self.scheduler.abort_request(request_id)
        return request is not None
    
    def step(self) -> List[BatcherOutput]:
        """执行一次迭代。
        
        Returns:
            本次迭代完成的请求列表
        """
        # 调度
        schedule_output = self.scheduler.schedule()
        
        if schedule_output.is_empty:
            return []
        
        completed = []
        
        # 执行 prefill
        for request in schedule_output.prefill_requests:
            self._execute_prefill(request)
        
        # 执行 decode
        for request in schedule_output.decode_requests:
            token_id = self._execute_decode(request)
            
            # 检查是否完成
            if request.should_stop(token_id):
                finished = self.scheduler.finish_request(request.request_id)
                if finished:
                    output = self._create_output(finished, "stop" if token_id in request.sampling_params.stop_token_ids else "length")
                    completed.append(output)
        
        self.completed_requests.extend(completed)
        return completed
    
    def _execute_prefill(self, request: Request) -> None:
        """执行 prefill 阶段。"""
        # Prefill 阶段处理
        request.num_computed_tokens = request.prompt_length
        request.prefill_time = time.time()
        self._total_prefill_tokens += request.prompt_length
    
    def _execute_decode(self, request: Request) -> int:
        """执行 decode 阶段，返回生成的 token ID。"""
        # Decode 阶段处理
        if self.model_runner is not None:
            token_id = self.model_runner.generate_token(request)
        else:
            # 使用确定性哈希生成 token（实际应用中由模型生成）
            token_id = hash(request.request_id + str(request.output_length)) % 32000
        
        request.add_token(token_id)
        self._total_tokens_generated += 1
        
        return token_id
    
    def _create_output(self, request: Request, finish_reason: str) -> BatcherOutput:
        """创建输出对象。"""
        latency = request.get_latency() or 0
        ttft = request.get_time_to_first_token()
        
        # 解码输出
        if self.tokenizer is not None:
            generated_text = self.tokenizer.decode(request.output_token_ids)
        else:
            generated_text = f"[Generated {len(request.output_token_ids)} tokens]"
        
        return BatcherOutput(
            request_id=request.request_id,
            prompt=request.prompt,
            generated_text=generated_text,
            output_token_ids=request.output_token_ids,
            finish_reason=finish_reason,
            latency_ms=latency * 1000,
            time_to_first_token_ms=ttft * 1000 if ttft else None,
            num_prompt_tokens=request.prompt_length,
            num_generated_tokens=request.output_length
        )
    
    def run_until_complete(self, max_iterations: int = 10000) -> List[BatcherOutput]:
        """运行直到所有请求完成。
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            所有完成的请求
        """
        all_completed = []
        
        for _ in range(max_iterations):
            completed = self.step()
            all_completed.extend(completed)
            
            # 检查是否全部完成
            if (self.scheduler.get_num_waiting() == 0 and 
                self.scheduler.get_num_running() == 0):
                break
        
        return all_completed
    
    def has_pending_requests(self) -> bool:
        """检查是否有待处理的请求。"""
        return (self.scheduler.get_num_waiting() > 0 or 
                self.scheduler.get_num_running() > 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        elapsed = time.time() - self._start_time
        scheduler_stats = self.scheduler.get_stats()
        
        return {
            **scheduler_stats,
            "total_requests": self._total_requests,
            "completed_requests": len(self.completed_requests),
            "total_tokens_generated": self._total_tokens_generated,
            "total_prefill_tokens": self._total_prefill_tokens,
            "elapsed_time_s": elapsed,
            "tokens_per_second": self._total_tokens_generated / elapsed if elapsed > 0 else 0,
            "requests_per_second": len(self.completed_requests) / elapsed if elapsed > 0 else 0,
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_continuous_batcher(
    max_batch_size: int = 256,
    max_num_sequences: int = 256,
    max_tokens_per_batch: int = 8192,
    block_size: int = 16,
    num_blocks: int = 1024,
    scheduling_policy: str = "fcfs",
    **kwargs
) -> ContinuousBatcher:
    """创建连续批处理器的工厂函数。
    
    Args:
        max_batch_size: 最大批次大小
        max_num_sequences: 最大并发序列数
        max_tokens_per_batch: 每批次最大 token 数
        block_size: KV Cache 块大小
        num_blocks: 总块数
        scheduling_policy: 调度策略 (fcfs/sjf/priority)
        **kwargs: 其他配置参数
        
    Returns:
        ContinuousBatcher 实例
        
    Example:
        >>> batcher = create_continuous_batcher(max_batch_size=32)
        >>> batcher.add_request("Hello, world!")
        >>> outputs = batcher.run_until_complete()
    """
    config = SchedulerConfig(
        max_batch_size=max_batch_size,
        max_num_sequences=max_num_sequences,
        max_tokens_per_batch=max_tokens_per_batch,
        block_size=block_size,
        num_blocks=num_blocks,
        scheduling_policy=SchedulingPolicy(scheduling_policy),
        **kwargs
    )
    return ContinuousBatcher(config)


def create_request(
    prompt: str,
    prompt_token_ids: Optional[List[int]] = None,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_sequences: Optional[List[str]] = None,
    request_id: Optional[str] = None,
    priority: int = 0
) -> Request:
    """创建请求的便捷函数。
    
    Args:
        prompt: 输入提示
        prompt_token_ids: 输入 token IDs
        max_tokens: 最大生成长度
        temperature: 温度
        top_p: Top-p 采样
        stop_sequences: 停止序列
        request_id: 请求 ID
        priority: 优先级
        
    Returns:
        Request 实例
    """
    if prompt_token_ids is None:
        prompt_token_ids = list(range(len(prompt.split())))
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_sequences or []
    )
    
    return Request(
        request_id=request_id or f"req_{time.time()}",
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        priority=priority
    )
