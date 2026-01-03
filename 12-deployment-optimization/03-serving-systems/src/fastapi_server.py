"""
FastAPI 模型服务模块

提供基于 FastAPI 的模型推理服务封装。

主要功能:
1. ModelServer: 模型服务器封装
2. DynamicBatcher: 动态批处理
3. 健康检查和监控
4. 请求/响应模型
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import asyncio
import time
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor

# 检查 FastAPI 是否可用
try:
    from fastapi import FastAPI, HTTPException, Response, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    BaseModel = object
    # Mock Field for when pydantic is not available
    def Field(*args, **kwargs):
        return None

# 检查 uvicorn 是否可用
try:
    import uvicorn
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False
    uvicorn = None

import numpy as np


# ==================== 请求/响应模型 ====================

class PredictRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """预测请求模型"""
    data: List[float] = Field(..., description="输入数据")
    batch_size: Optional[int] = Field(1, description="批次大小")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [1.0, 2.0, 3.0, 4.0],
                "batch_size": 1
            }
        }


class ImagePredictRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """图像预测请求模型"""
    image_data: List[List[List[float]]] = Field(..., description="图像数据 [H, W, C]")

    class Config:
        json_schema_extra = {
            "example": {
                "image_data": [[[0.5, 0.5, 0.5]]]
            }
        }


class PredictResponse(BaseModel if FASTAPI_AVAILABLE else object):
    """预测响应模型"""
    prediction: List[float] = Field(..., description="预测结果")
    confidence: Optional[float] = Field(None, description="置信度")
    latency_ms: float = Field(..., description="推理延迟(毫秒)")


class HealthResponse(BaseModel if FASTAPI_AVAILABLE else object):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    model_loaded: bool = Field(..., description="模型是否加载")
    uptime_seconds: float = Field(..., description="运行时间(秒)")


class MetricsResponse(BaseModel if FASTAPI_AVAILABLE else object):
    """指标响应"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float


# ==================== 批处理请求 ====================

@dataclass
class BatchRequest:
    """批处理请求"""
    data: Any
    future: asyncio.Future
    timestamp: float


class DynamicBatcher:
    """动态批处理器"""

    def __init__(
        self,
        predict_fn: Callable,
        max_batch_size: int = 32,
        max_wait_time: float = 0.01  # 10ms
    ):
        """
        初始化动态批处理器

        Args:
            predict_fn: 批量预测函数
            max_batch_size: 最大批次大小
            max_wait_time: 最大等待时间(秒)
        """
        self.predict_fn = predict_fn
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue: List[BatchRequest] = []
        self.lock = asyncio.Lock()
        self._processing = False
        self._task = None

    async def add_request(self, data: Any) -> Any:
        """
        添加请求到批处理队列

        Args:
            data: 输入数据

        Returns:
            预测结果
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request = BatchRequest(
            data=data,
            future=future,
            timestamp=time.time()
        )

        async with self.lock:
            self.queue.append(request)

            # 检查是否需要处理
            if len(self.queue) >= self.max_batch_size:
                asyncio.create_task(self._process_batch())
            elif not self._processing:
                asyncio.create_task(self._wait_and_process())

        return await future

    async def _wait_and_process(self):
        """等待并处理批次"""
        self._processing = True
        await asyncio.sleep(self.max_wait_time)
        await self._process_batch()
        self._processing = False

    async def _process_batch(self):
        """处理当前批次"""
        async with self.lock:
            if not self.queue:
                return

            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]

        # 批量推理
        try:
            batch_data = [r.data for r in batch]
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.predict_fn,
                batch_data
            )

            # 返回结果
            for request, result in zip(batch, results):
                if not request.future.done():
                    request.future.set_result(result)
        except Exception as e:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)


# ==================== 指标收集 ====================

class MetricsCollector:
    """指标收集器"""

    def __init__(self, max_history: int = 1000):
        """
        初始化指标收集器

        Args:
            max_history: 最大历史记录数
        """
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies: List[float] = []
        self.max_history = max_history
        self._lock = asyncio.Lock()

    async def record_request(self, latency_ms: float, success: bool = True):
        """
        记录请求

        Args:
            latency_ms: 延迟(毫秒)
            success: 是否成功
        """
        async with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            self.latencies.append(latency_ms)
            if len(self.latencies) > self.max_history:
                self.latencies = self.latencies[-self.max_history:]

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        if not self.latencies:
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p99_latency_ms": 0,
            }

        latencies = np.array(self.latencies)
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
        }


# ==================== 缓存 ====================

class InferenceCache:
    """推理结果缓存"""

    def __init__(self, max_size: int = 1000, ttl: float = 300):
        """
        初始化缓存

        Args:
            max_size: 最大缓存大小
            ttl: 缓存过期时间(秒)
        """
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times: Dict[str, float] = {}

    def _hash_input(self, data: Any) -> str:
        """计算输入哈希"""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get(self, data: Any) -> Optional[Any]:
        """获取缓存结果"""
        key = self._hash_input(data)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                self.access_times[key] = time.time()
                return entry['result']
            else:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        return None

    def set(self, data: Any, result: Any):
        """设置缓存"""
        if len(self.cache) >= self.max_size:
            self._evict()

        key = self._hash_input(data)
        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
        self.access_times[key] = time.time()

    def _evict(self):
        """LRU 淘汰"""
        if not self.access_times:
            return
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()


# ==================== 模型服务器 ====================

class ModelServer:
    """模型服务器"""

    def __init__(
        self,
        predict_fn: Callable,
        name: str = "model-server",
        enable_batching: bool = False,
        max_batch_size: int = 32,
        max_wait_time: float = 0.01,
        enable_cache: bool = False,
        cache_size: int = 1000,
        cache_ttl: float = 300
    ):
        """
        初始化模型服务器

        Args:
            predict_fn: 预测函数
            name: 服务名称
            enable_batching: 是否启用批处理
            max_batch_size: 最大批次大小
            max_wait_time: 最大等待时间
            enable_cache: 是否启用缓存
            cache_size: 缓存大小
            cache_ttl: 缓存过期时间
        """
        if not FASTAPI_AVAILABLE:
            raise RuntimeError(
                "FastAPI not available. "
                "Install with: pip install fastapi uvicorn"
            )

        self.predict_fn = predict_fn
        self.name = name
        self.start_time = time.time()
        self.model_loaded = True

        # 创建 FastAPI 应用
        self.app = FastAPI(
            title=name,
            description="Model Inference Server",
            version="1.0.0"
        )

        # 添加 CORS 中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 指标收集
        self.metrics = MetricsCollector()

        # 批处理
        self.enable_batching = enable_batching
        if enable_batching:
            self.batcher = DynamicBatcher(
                predict_fn=self._batch_predict,
                max_batch_size=max_batch_size,
                max_wait_time=max_wait_time
            )

        # 缓存
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache = InferenceCache(max_size=cache_size, ttl=cache_ttl)

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 注册路由
        self._register_routes()

    def _batch_predict(self, batch_data: List[Any]) -> List[Any]:
        """批量预测"""
        return [self.predict_fn(data) for data in batch_data]

    def _register_routes(self):
        """注册路由"""

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(
                status="healthy",
                model_loaded=self.model_loaded,
                uptime_seconds=time.time() - self.start_time
            )

        @self.app.get("/ready")
        async def ready():
            if self.model_loaded:
                return {"status": "ready"}
            return Response(status_code=503)

        @self.app.get("/metrics", response_model=MetricsResponse)
        async def metrics():
            return self.metrics.get_metrics()

        @self.app.post("/predict", response_model=PredictResponse)
        async def predict(request: PredictRequest):
            start_time = time.time()
            success = True

            try:
                data = request.data

                # 检查缓存
                if self.enable_cache:
                    cached = self.cache.get(data)
                    if cached is not None:
                        latency = (time.time() - start_time) * 1000
                        return PredictResponse(
                            prediction=cached,
                            latency_ms=latency
                        )

                # 执行预测
                if self.enable_batching:
                    result = await self.batcher.add_request(data)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor,
                        self.predict_fn,
                        data
                    )

                # 更新缓存
                if self.enable_cache:
                    self.cache.set(data, result)

                latency = (time.time() - start_time) * 1000

                # 处理结果
                if isinstance(result, np.ndarray):
                    result = result.tolist()
                if not isinstance(result, list):
                    result = [result]

                return PredictResponse(
                    prediction=result,
                    latency_ms=latency
                )

            except Exception as e:
                success = False
                raise HTTPException(status_code=500, detail=str(e))

            finally:
                latency = (time.time() - start_time) * 1000
                await self.metrics.record_request(latency, success)

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行服务器"""
        if not UVICORN_AVAILABLE:
            raise RuntimeError("uvicorn not available")
        uvicorn.run(self.app, host=host, port=port)


# ==================== 便捷函数 ====================

def create_model_server(
    predict_fn: Callable,
    name: str = "model-server",
    **kwargs
) -> ModelServer:
    """
    创建模型服务器

    Args:
        predict_fn: 预测函数
        name: 服务名称
        **kwargs: 其他配置

    Returns:
        模型服务器实例
    """
    return ModelServer(predict_fn=predict_fn, name=name, **kwargs)


def create_app(
    predict_fn: Callable,
    name: str = "model-server",
    **kwargs
) -> "FastAPI":
    """
    创建 FastAPI 应用

    Args:
        predict_fn: 预测函数
        name: 服务名称
        **kwargs: 其他配置

    Returns:
        FastAPI 应用实例
    """
    server = ModelServer(predict_fn=predict_fn, name=name, **kwargs)
    return server.app
