"""
Continuous Batching 模块单元测试

测试覆盖:
- RequestStatus 枚举
- SchedulingPolicy 枚举
- SchedulerConfig 配置
- SamplingParams 采样参数
- Request 请求类
- RequestQueue 请求队列
- BatchScheduler 批调度器
- ContinuousBatcher 连续批处理器
"""

import pytest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.continuous_batch import (
    RequestStatus,
    SchedulingPolicy,
    SchedulerConfig,
    SamplingParams,
    Request,
    RequestQueue,
    SchedulerOutput,
    BatchScheduler,
    ContinuousBatcher,
    create_continuous_batcher,
    create_request,
)


class TestRequestStatus:
    """RequestStatus 枚举测试"""
    
    def test_status_values(self):
        """测试状态值存在"""
        assert RequestStatus.PENDING is not None
        assert RequestStatus.RUNNING is not None
        assert RequestStatus.COMPLETED is not None
        assert RequestStatus.FAILED is not None
        assert RequestStatus.CANCELLED is not None


class TestSchedulingPolicy:
    """SchedulingPolicy 枚举测试"""
    
    def test_policy_values(self):
        """测试策略值存在"""
        assert SchedulingPolicy.FCFS is not None
        assert SchedulingPolicy.SJF is not None
        assert SchedulingPolicy.PRIORITY is not None


class TestSchedulerConfig:
    """SchedulerConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SchedulerConfig()
        assert config.max_batch_size > 0
        assert config.max_num_sequences > 0
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SchedulerConfig(
            max_batch_size=16,
            max_num_sequences=64,
            max_tokens_per_batch=4096,
            scheduling_policy=SchedulingPolicy.SJF
        )
        assert config.max_batch_size == 16
        assert config.max_num_sequences == 64
        assert config.max_tokens_per_batch == 4096
        assert config.scheduling_policy == SchedulingPolicy.SJF


class TestSamplingParams:
    """SamplingParams 测试"""
    
    def test_default_params(self):
        """测试默认参数"""
        params = SamplingParams()
        assert params.temperature >= 0
        assert params.max_tokens > 0
    
    def test_custom_params(self):
        """测试自定义参数"""
        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=256
        )
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 256


class TestRequest:
    """Request 测试"""
    
    def test_request_creation(self):
        """测试请求创建"""
        request = Request(
            request_id="req-001",
            prompt="Hello, world!",
            prompt_token_ids=[1, 2, 3, 4, 5]
        )
        assert request.request_id == "req-001"
        assert request.prompt == "Hello, world!"
        assert request.prompt_token_ids == [1, 2, 3, 4, 5]
        assert request.status == RequestStatus.PENDING
    
    def test_request_with_sampling_params(self):
        """测试带采样参数的请求"""
        params = SamplingParams(temperature=0.5, max_tokens=100)
        request = Request(
            request_id="req-002",
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3],
            sampling_params=params
        )
        assert request.sampling_params.temperature == 0.5
        assert request.sampling_params.max_tokens == 100
    
    def test_request_prompt_length(self):
        """测试 prompt 长度"""
        request = Request(
            request_id="req-003",
            prompt="Test",
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        assert request.prompt_length == 10
    
    def test_request_output_length(self):
        """测试输出长度"""
        request = Request(
            request_id="req-004",
            prompt="Test",
            prompt_token_ids=[1, 2, 3]
        )
        assert request.output_length == 0
        
        request.output_token_ids = [10, 11, 12, 13, 14]
        assert request.output_length == 5
    
    def test_request_is_finished(self):
        """测试请求是否完成"""
        request = Request(
            request_id="req-005",
            prompt="Test",
            prompt_token_ids=[1, 2, 3]
        )
        assert not request.is_finished
        
        request.status = RequestStatus.COMPLETED
        assert request.is_finished
        
        request.status = RequestStatus.FAILED
        assert request.is_finished
    
    def test_request_add_token(self):
        """测试添加 token"""
        request = Request(
            request_id="req-006",
            prompt="Test",
            prompt_token_ids=[1, 2, 3]
        )
        request.add_token(100)
        request.add_token(101)
        
        assert request.output_token_ids == [100, 101]
        assert request.output_length == 2


class TestRequestQueue:
    """RequestQueue 测试"""
    
    def test_queue_creation(self):
        """测试队列创建"""
        queue = RequestQueue()
        assert len(queue) == 0
        assert not queue
    
    def test_add_request(self):
        """测试添加请求"""
        queue = RequestQueue()
        request = Request(
            request_id="req-001",
            prompt="Test",
            prompt_token_ids=[1, 2, 3]
        )
        queue.add(request)
        
        assert len(queue) == 1
        assert queue
    
    def test_get_request(self):
        """测试获取请求"""
        queue = RequestQueue()
        request = Request(
            request_id="req-001",
            prompt="Test",
            prompt_token_ids=[1, 2, 3]
        )
        queue.add(request)
        
        retrieved = queue.get("req-001")
        assert retrieved is request
    
    def test_get_nonexistent_request(self):
        """测试获取不存在的请求"""
        queue = RequestQueue()
        retrieved = queue.get("nonexistent")
        assert retrieved is None
    
    def test_remove_request(self):
        """测试移除请求"""
        queue = RequestQueue()
        request = Request(
            request_id="req-001",
            prompt="Test",
            prompt_token_ids=[1, 2, 3]
        )
        queue.add(request)
        queue.remove("req-001")
        
        assert len(queue) == 0
        assert queue.get("req-001") is None
    
    def test_pop_request_fcfs(self):
        """测试 FCFS 策略弹出请求"""
        queue = RequestQueue(policy=SchedulingPolicy.FCFS)
        
        req1 = Request(request_id="req-001", prompt="First", prompt_token_ids=[1])
        req2 = Request(request_id="req-002", prompt="Second", prompt_token_ids=[1, 2])
        req3 = Request(request_id="req-003", prompt="Third", prompt_token_ids=[1, 2, 3])
        
        queue.add(req1)
        queue.add(req2)
        queue.add(req3)
        
        # FCFS: 先进先出
        popped = queue.pop()
        assert popped.request_id == "req-001"
    
    def test_pop_request_sjf(self):
        """测试 SJF 策略弹出请求"""
        queue = RequestQueue(policy=SchedulingPolicy.SJF)
        
        req1 = Request(request_id="req-001", prompt="Long", prompt_token_ids=[1, 2, 3, 4, 5])
        req2 = Request(request_id="req-002", prompt="Short", prompt_token_ids=[1])
        req3 = Request(request_id="req-003", prompt="Medium", prompt_token_ids=[1, 2, 3])
        
        queue.add(req1)
        queue.add(req2)
        queue.add(req3)
        
        # SJF: 最短作业优先
        popped = queue.pop()
        assert popped.request_id == "req-002"


class TestBatchScheduler:
    """BatchScheduler 测试"""
    
    def test_scheduler_creation(self):
        """测试调度器创建"""
        config = SchedulerConfig(max_batch_size=8, max_num_sequences=32)
        scheduler = BatchScheduler(config)
        assert scheduler is not None
    
    def test_add_request(self):
        """测试添加请求"""
        config = SchedulerConfig(max_batch_size=8, max_num_sequences=32)
        scheduler = BatchScheduler(config)
        
        request = Request(
            request_id="req-001",
            prompt="Test",
            prompt_token_ids=[1, 2, 3]
        )
        scheduler.add_request(request)
        
        assert scheduler.get_num_waiting() == 1
    
    def test_schedule_single_request(self):
        """测试调度单个请求"""
        config = SchedulerConfig(max_batch_size=8, max_num_sequences=32)
        scheduler = BatchScheduler(config)
        
        request = Request(
            request_id="req-001",
            prompt="Test",
            prompt_token_ids=[1, 2, 3]
        )
        scheduler.add_request(request)
        
        output = scheduler.schedule()
        assert output is not None
        # prefill_requests 或 decode_requests 中应该有请求
        total_scheduled = len(output.prefill_requests) + len(output.decode_requests)
        assert total_scheduled == 1
    
    def test_schedule_multiple_requests(self):
        """测试调度多个请求"""
        config = SchedulerConfig(max_batch_size=8, max_num_sequences=32)
        scheduler = BatchScheduler(config)
        
        for i in range(5):
            request = Request(
                request_id=f"req-{i:03d}",
                prompt=f"Test {i}",
                prompt_token_ids=list(range(i + 1))
            )
            scheduler.add_request(request)
        
        output = scheduler.schedule()
        total_scheduled = len(output.prefill_requests) + len(output.decode_requests)
        assert total_scheduled == 5
    
    def test_schedule_respects_max_batch_size(self):
        """测试调度遵守最大批大小"""
        config = SchedulerConfig(max_batch_size=3, max_num_sequences=32)
        scheduler = BatchScheduler(config)
        
        for i in range(10):
            request = Request(
                request_id=f"req-{i:03d}",
                prompt=f"Test {i}",
                prompt_token_ids=[1, 2, 3]
            )
            scheduler.add_request(request)
        
        output = scheduler.schedule()
        total_scheduled = len(output.prefill_requests) + len(output.decode_requests)
        assert total_scheduled <= 3
    
    def test_abort_request(self):
        """测试中止请求"""
        config = SchedulerConfig(max_batch_size=8, max_num_sequences=32)
        scheduler = BatchScheduler(config)
        
        request = Request(
            request_id="req-001",
            prompt="Test",
            prompt_token_ids=[1, 2, 3]
        )
        scheduler.add_request(request)
        scheduler.abort_request("req-001")
        
        assert scheduler.get_num_waiting() == 0


class TestContinuousBatcher:
    """ContinuousBatcher 测试"""
    
    def test_batcher_creation(self):
        """测试批处理器创建"""
        config = SchedulerConfig(max_batch_size=8, max_num_sequences=32)
        batcher = ContinuousBatcher(config)
        assert batcher is not None
    
    def test_add_request(self):
        """测试添加请求"""
        config = SchedulerConfig(max_batch_size=8, max_num_sequences=32)
        batcher = ContinuousBatcher(config)
        
        # add_request 接受 prompt 字符串
        request_id = batcher.add_request(
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3]
        )
        assert request_id is not None
    
    def test_has_pending_requests(self):
        """测试是否有待处理请求"""
        config = SchedulerConfig(max_batch_size=8, max_num_sequences=32)
        batcher = ContinuousBatcher(config)
        
        assert not batcher.has_pending_requests()
        
        batcher.add_request(
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3]
        )
        
        assert batcher.has_pending_requests()


class TestCreateContinuousBatcher:
    """create_continuous_batcher 工厂函数测试"""
    
    def test_create_default(self):
        """测试默认创建"""
        batcher = create_continuous_batcher()
        assert batcher is not None
        assert isinstance(batcher, ContinuousBatcher)


class TestCreateRequest:
    """create_request 工厂函数测试"""
    
    def test_create_request(self):
        """测试创建请求"""
        request = create_request(
            prompt="Hello, world!",
            prompt_token_ids=[1, 2, 3, 4, 5]
        )
        assert request is not None
        assert request.prompt == "Hello, world!"
        assert request.prompt_token_ids == [1, 2, 3, 4, 5]
    
    def test_create_request_with_params(self):
        """测试带参数创建请求"""
        request = create_request(
            prompt="Test",
            prompt_token_ids=[1, 2, 3],
            temperature=0.7,
            max_tokens=100
        )
        assert request.sampling_params.temperature == 0.7
        assert request.sampling_params.max_tokens == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
