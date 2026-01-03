"""
FastAPI 服务模块测试
"""

import pytest
import sys
import asyncio
import time
import os
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi_server import (
    FASTAPI_AVAILABLE,
    MetricsCollector,
    InferenceCache,
    DynamicBatcher,
)


# ==================== MetricsCollector 测试 ====================

class TestMetricsCollector:
    """MetricsCollector 测试"""

    @pytest.fixture
    def collector(self):
        return MetricsCollector(max_history=100)

    @pytest.mark.asyncio
    async def test_record_request_success(self, collector):
        """测试记录成功请求"""
        await collector.record_request(10.0, success=True)
        await collector.record_request(20.0, success=True)

        assert collector.total_requests == 2
        assert collector.successful_requests == 2
        assert collector.failed_requests == 0
        assert len(collector.latencies) == 2

    @pytest.mark.asyncio
    async def test_record_request_failure(self, collector):
        """测试记录失败请求"""
        await collector.record_request(10.0, success=True)
        await collector.record_request(20.0, success=False)

        assert collector.total_requests == 2
        assert collector.successful_requests == 1
        assert collector.failed_requests == 1

    def test_get_metrics_empty(self, collector):
        """测试空指标"""
        metrics = collector.get_metrics()

        assert metrics["total_requests"] == 0
        assert metrics["avg_latency_ms"] == 0
        assert metrics["p50_latency_ms"] == 0
        assert metrics["p99_latency_ms"] == 0

    @pytest.mark.asyncio
    async def test_get_metrics_with_data(self, collector):
        """测试有数据的指标"""
        for i in range(100):
            await collector.record_request(float(i), success=True)

        metrics = collector.get_metrics()

        assert metrics["total_requests"] == 100
        assert metrics["successful_requests"] == 100
        assert 45 < metrics["avg_latency_ms"] < 55  # 约 49.5
        assert 45 < metrics["p50_latency_ms"] < 55
        assert metrics["p99_latency_ms"] > 90

    @pytest.mark.asyncio
    async def test_max_history_limit(self, collector):
        """测试最大历史记录限制"""
        for i in range(200):
            await collector.record_request(float(i), success=True)

        assert len(collector.latencies) == 100
        assert collector.total_requests == 200


# ==================== InferenceCache 测试 ====================

class TestInferenceCache:
    """InferenceCache 测试"""

    @pytest.fixture
    def cache(self):
        return InferenceCache(max_size=10, ttl=1.0)

    def test_set_and_get(self, cache):
        """测试设置和获取缓存"""
        data = [1.0, 2.0, 3.0]
        result = [0.5, 0.3, 0.2]

        cache.set(data, result)
        cached = cache.get(data)

        assert cached == result

    def test_get_nonexistent(self, cache):
        """测试获取不存在的缓存"""
        result = cache.get([1.0, 2.0])
        assert result is None

    def test_ttl_expiry(self, cache):
        """测试 TTL 过期"""
        data = [1.0, 2.0, 3.0]
        result = [0.5, 0.3, 0.2]

        cache.set(data, result)
        assert cache.get(data) == result

        # 等待过期
        time.sleep(1.1)
        assert cache.get(data) is None

    def test_lru_eviction(self, cache):
        """测试 LRU 淘汰"""
        # 填满缓存
        for i in range(10):
            cache.set([float(i)], [float(i)])

        # 访问第一个元素
        cache.get([0.0])

        # 添加新元素，应该淘汰最久未访问的
        cache.set([10.0], [10.0])

        # 第一个元素应该还在（刚访问过）
        assert cache.get([0.0]) is not None

    def test_clear(self, cache):
        """测试清空缓存"""
        cache.set([1.0], [1.0])
        cache.set([2.0], [2.0])

        cache.clear()

        assert cache.get([1.0]) is None
        assert cache.get([2.0]) is None
        assert len(cache.cache) == 0

    def test_hash_consistency(self, cache):
        """测试哈希一致性"""
        data1 = [1.0, 2.0, 3.0]
        data2 = [1.0, 2.0, 3.0]

        cache.set(data1, [0.5])
        result = cache.get(data2)

        assert result == [0.5]


# ==================== DynamicBatcher 测试 ====================

class TestDynamicBatcher:
    """DynamicBatcher 测试"""

    @pytest.fixture
    def batcher(self):
        def predict_fn(batch_data):
            return [sum(d) for d in batch_data]

        return DynamicBatcher(
            predict_fn=predict_fn,
            max_batch_size=4,
            max_wait_time=0.05
        )

    @pytest.mark.asyncio
    async def test_single_request(self, batcher):
        """测试单个请求"""
        result = await batcher.add_request([1.0, 2.0, 3.0])
        assert result == 6.0

    @pytest.mark.asyncio
    async def test_batch_processing(self, batcher):
        """测试批处理"""
        # 并发发送多个请求
        tasks = [
            batcher.add_request([1.0, 2.0]),
            batcher.add_request([3.0, 4.0]),
            batcher.add_request([5.0, 6.0]),
        ]

        results = await asyncio.gather(*tasks)

        assert results[0] == 3.0
        assert results[1] == 7.0
        assert results[2] == 11.0

    @pytest.mark.asyncio
    async def test_max_batch_size_trigger(self, batcher):
        """测试达到最大批次大小触发处理"""
        # 发送 4 个请求（等于 max_batch_size）
        tasks = [
            batcher.add_request([float(i)])
            for i in range(4)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        for i, result in enumerate(results):
            assert result == float(i)


# ==================== ModelServer 测试 (需要 FastAPI) ====================

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestModelServer:
    """ModelServer 测试"""

    @pytest.fixture
    def server(self):
        from fastapi_server import ModelServer

        def predict_fn(data):
            return [sum(data)]

        return ModelServer(
            predict_fn=predict_fn,
            name="test-server",
            enable_batching=False,
            enable_cache=False
        )

    def test_server_creation(self, server):
        """测试服务器创建"""
        assert server.name == "test-server"
        assert server.model_loaded is True
        assert server.app is not None

    def test_server_with_batching(self):
        """测试带批处理的服务器"""
        from fastapi_server import ModelServer

        def predict_fn(data):
            return [sum(data)]

        server = ModelServer(
            predict_fn=predict_fn,
            enable_batching=True,
            max_batch_size=8
        )

        assert server.enable_batching is True
        assert server.batcher is not None

    def test_server_with_cache(self):
        """测试带缓存的服务器"""
        from fastapi_server import ModelServer

        def predict_fn(data):
            return [sum(data)]

        server = ModelServer(
            predict_fn=predict_fn,
            enable_cache=True,
            cache_size=100
        )

        assert server.enable_cache is True
        assert server.cache is not None


# ==================== 便捷函数测试 ====================

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_create_model_server(self):
        """测试 create_model_server"""
        from fastapi_server import create_model_server

        def predict_fn(data):
            return [1.0]

        server = create_model_server(predict_fn, name="test")
        assert server is not None
        assert server.name == "test"

    def test_create_app(self):
        """测试 create_app"""
        from fastapi_server import create_app

        def predict_fn(data):
            return [1.0]

        app = create_app(predict_fn, name="test-app")
        assert app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
