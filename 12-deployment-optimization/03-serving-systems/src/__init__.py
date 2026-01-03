"""
模型服务系统模块

提供模型服务化部署的完整解决方案。

主要组件:
1. FastAPI 服务: 快速构建 REST API
2. Triton 客户端: 高性能推理服务器客户端
3. 负载均衡: 多实例部署和流量分发
"""

# FastAPI 服务
from .fastapi_server import (
    # 请求/响应模型
    PredictRequest,
    PredictResponse,
    HealthResponse,
    MetricsResponse,
    ImagePredictRequest,
    # 核心类
    ModelServer,
    DynamicBatcher,
    MetricsCollector,
    InferenceCache,
    # 便捷函数
    create_model_server,
    create_app,
    # 可用性标志
    FASTAPI_AVAILABLE,
    UVICORN_AVAILABLE,
)

# Triton 客户端
from .triton_client import (
    # 数据类
    ModelInput,
    ModelOutput,
    ModelMetadata,
    InferenceResult,
    TritonDataType,
    # 客户端类
    TritonHTTPClient,
    TritonGRPCClient,
    TritonClient,
    # 配置生成
    ModelConfigGenerator,
    # 便捷函数
    create_triton_client,
    quick_infer,
    numpy_to_triton_dtype,
    # 可用性标志
    TRITON_AVAILABLE,
    TRITON_HTTP_AVAILABLE,
    TRITON_GRPC_AVAILABLE,
)

# 负载均衡
from .load_balancer import (
    # 枚举
    LoadBalanceStrategy,
    CircuitBreakerState,
    # 数据类
    Server,
    # 策略类
    BalancingStrategy,
    RoundRobinStrategy,
    WeightedRoundRobinStrategy,
    LeastConnectionsStrategy,
    IPHashStrategy,
    RandomStrategy,
    ResponseTimeStrategy,
    # 核心类
    LoadBalancer,
    HealthChecker,
    ConnectionPool,
    RateLimiter,
    CircuitBreaker,
    # 便捷函数
    create_load_balancer,
    quick_request,
    # 可用性标志
    HTTPX_AVAILABLE,
)

__all__ = [
    # FastAPI
    "PredictRequest",
    "PredictResponse",
    "HealthResponse",
    "MetricsResponse",
    "ImagePredictRequest",
    "ModelServer",
    "DynamicBatcher",
    "MetricsCollector",
    "InferenceCache",
    "create_model_server",
    "create_app",
    "FASTAPI_AVAILABLE",
    "UVICORN_AVAILABLE",
    # Triton
    "ModelInput",
    "ModelOutput",
    "ModelMetadata",
    "InferenceResult",
    "TritonDataType",
    "TritonHTTPClient",
    "TritonGRPCClient",
    "TritonClient",
    "ModelConfigGenerator",
    "create_triton_client",
    "quick_infer",
    "numpy_to_triton_dtype",
    "TRITON_AVAILABLE",
    "TRITON_HTTP_AVAILABLE",
    "TRITON_GRPC_AVAILABLE",
    # 负载均衡
    "LoadBalanceStrategy",
    "CircuitBreakerState",
    "Server",
    "BalancingStrategy",
    "RoundRobinStrategy",
    "WeightedRoundRobinStrategy",
    "LeastConnectionsStrategy",
    "IPHashStrategy",
    "RandomStrategy",
    "ResponseTimeStrategy",
    "LoadBalancer",
    "HealthChecker",
    "ConnectionPool",
    "RateLimiter",
    "CircuitBreaker",
    "create_load_balancer",
    "quick_request",
    "HTTPX_AVAILABLE",
]

__version__ = "1.0.0"
