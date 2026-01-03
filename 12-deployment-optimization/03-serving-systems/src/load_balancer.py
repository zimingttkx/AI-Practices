"""
负载均衡模块

提供多种负载均衡策略和服务发现功能。

主要功能:
1. LoadBalancer: 负载均衡器
2. 多种负载均衡策略
3. 健康检查
4. 连接池管理
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
import asyncio
import time
import random
import hashlib
from abc import ABC, abstractmethod

import numpy as np

# 检查 httpx 是否可用
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


# ==================== 负载均衡策略 ====================

class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    RESPONSE_TIME = "response_time"


# ==================== 服务器节点 ====================

@dataclass
class Server:
    """服务器节点"""
    url: str
    weight: int = 1
    healthy: bool = True
    connections: int = 0
    last_check: float = 0
    response_times: List[float] = field(default_factory=list)
    failure_count: int = 0
    success_count: int = 0

    @property
    def avg_response_time(self) -> float:
        """平均响应时间"""
        if not self.response_times:
            return float('inf')
        # 只取最近 10 次
        recent = self.response_times[-10:]
        return sum(recent) / len(recent)

    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    def record_response(self, response_time: float, success: bool = True):
        """记录响应"""
        self.response_times.append(response_time)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]

        if success:
            self.success_count += 1
            self.failure_count = max(0, self.failure_count - 1)
        else:
            self.failure_count += 1

    def reset_stats(self):
        """重置统计"""
        self.response_times = []
        self.failure_count = 0
        self.success_count = 0


# ==================== 负载均衡策略实现 ====================

class BalancingStrategy(ABC):
    """负载均衡策略基类"""

    @abstractmethod
    def select(
        self,
        servers: List[Server],
        client_ip: Optional[str] = None
    ) -> Server:
        """选择服务器"""
        pass


class RoundRobinStrategy(BalancingStrategy):
    """轮询策略"""

    def __init__(self):
        self.current_index = 0
        self._lock = asyncio.Lock()

    def select(
        self,
        servers: List[Server],
        client_ip: Optional[str] = None
    ) -> Server:
        if not servers:
            raise RuntimeError("No servers available")

        server = servers[self.current_index % len(servers)]
        self.current_index += 1
        return server


class WeightedRoundRobinStrategy(BalancingStrategy):
    """加权轮询策略"""

    def __init__(self):
        self.current_index = 0
        self._lock = asyncio.Lock()

    def select(
        self,
        servers: List[Server],
        client_ip: Optional[str] = None
    ) -> Server:
        if not servers:
            raise RuntimeError("No servers available")

        total_weight = sum(s.weight for s in servers)
        point = self.current_index % total_weight
        self.current_index += 1

        current = 0
        for server in servers:
            current += server.weight
            if point < current:
                return server

        return servers[-1]


class LeastConnectionsStrategy(BalancingStrategy):
    """最少连接策略"""

    def select(
        self,
        servers: List[Server],
        client_ip: Optional[str] = None
    ) -> Server:
        if not servers:
            raise RuntimeError("No servers available")

        return min(servers, key=lambda s: s.connections)


class IPHashStrategy(BalancingStrategy):
    """IP 哈希策略"""

    def select(
        self,
        servers: List[Server],
        client_ip: Optional[str] = None
    ) -> Server:
        if not servers:
            raise RuntimeError("No servers available")

        if not client_ip:
            return random.choice(servers)

        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return servers[hash_value % len(servers)]


class RandomStrategy(BalancingStrategy):
    """随机策略"""

    def select(
        self,
        servers: List[Server],
        client_ip: Optional[str] = None
    ) -> Server:
        if not servers:
            raise RuntimeError("No servers available")

        return random.choice(servers)


class ResponseTimeStrategy(BalancingStrategy):
    """响应时间策略 (选择响应最快的服务器)"""

    def select(
        self,
        servers: List[Server],
        client_ip: Optional[str] = None
    ) -> Server:
        if not servers:
            raise RuntimeError("No servers available")

        return min(servers, key=lambda s: s.avg_response_time)


# 策略工厂
STRATEGY_MAP = {
    LoadBalanceStrategy.ROUND_ROBIN: RoundRobinStrategy,
    LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinStrategy,
    LoadBalanceStrategy.LEAST_CONNECTIONS: LeastConnectionsStrategy,
    LoadBalanceStrategy.IP_HASH: IPHashStrategy,
    LoadBalanceStrategy.RANDOM: RandomStrategy,
    LoadBalanceStrategy.RESPONSE_TIME: ResponseTimeStrategy,
}


# ==================== 健康检查器 ====================

class HealthChecker:
    """健康检查器"""

    def __init__(
        self,
        check_interval: float = 10.0,
        timeout: float = 5.0,
        healthy_threshold: int = 2,
        unhealthy_threshold: int = 3,
        health_path: str = "/health"
    ):
        """
        初始化健康检查器

        Args:
            check_interval: 检查间隔 (秒)
            timeout: 超时时间 (秒)
            healthy_threshold: 健康阈值 (连续成功次数)
            unhealthy_threshold: 不健康阈值 (连续失败次数)
            health_path: 健康检查路径
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self.healthy_threshold = healthy_threshold
        self.unhealthy_threshold = unhealthy_threshold
        self.health_path = health_path
        self._running = False
        self._task = None
        self._consecutive_success: Dict[str, int] = {}
        self._consecutive_failure: Dict[str, int] = {}

    async def check_server(self, server: Server) -> bool:
        """检查单个服务器"""
        if not HTTPX_AVAILABLE:
            return True

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{server.url}{self.health_path}",
                    timeout=self.timeout
                )
                return response.status_code == 200
        except Exception:
            return False

    async def check_all(self, servers: List[Server]):
        """检查所有服务器"""
        for server in servers:
            healthy = await self.check_server(server)
            server.last_check = time.time()

            if healthy:
                self._consecutive_success[server.url] = \
                    self._consecutive_success.get(server.url, 0) + 1
                self._consecutive_failure[server.url] = 0

                if self._consecutive_success[server.url] >= self.healthy_threshold:
                    server.healthy = True
            else:
                self._consecutive_failure[server.url] = \
                    self._consecutive_failure.get(server.url, 0) + 1
                self._consecutive_success[server.url] = 0

                if self._consecutive_failure[server.url] >= self.unhealthy_threshold:
                    server.healthy = False

    async def start(self, servers: List[Server]):
        """启动健康检查循环"""
        self._running = True
        while self._running:
            await self.check_all(servers)
            await asyncio.sleep(self.check_interval)

    def stop(self):
        """停止健康检查"""
        self._running = False


# ==================== 连接池 ====================

class ConnectionPool:
    """HTTP 连接池"""

    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        keepalive_expiry: float = 5.0,
        timeout: float = 30.0
    ):
        """
        初始化连接池

        Args:
            max_connections: 最大连接数
            max_keepalive_connections: 最大保持连接数
            keepalive_expiry: 保持连接过期时间
            timeout: 超时时间
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx not available. Install with: pip install httpx"
            )

        self.limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry
        )
        self.timeout = httpx.Timeout(timeout)
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        """获取客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                limits=self.limits,
                timeout=self.timeout
            )
        return self._client

    async def close(self):
        """关闭连接池"""
        if self._client:
            await self._client.aclose()
            self._client = None


# ==================== 限流器 ====================

class RateLimiter:
    """令牌桶限流器"""

    def __init__(self, rate: float, capacity: int):
        """
        初始化限流器

        Args:
            rate: 每秒生成的令牌数
            capacity: 桶容量
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """尝试获取令牌"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait_for_token(self, timeout: float = 10.0) -> bool:
        """等待获取令牌"""
        start = time.time()
        while time.time() - start < timeout:
            if await self.acquire():
                return True
            await asyncio.sleep(0.01)
        return False


# ==================== 熔断器 ====================

class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"  # 正常
    OPEN = "open"  # 熔断
    HALF_OPEN = "half_open"  # 半开


class CircuitBreaker:
    """熔断器"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3
    ):
        """
        初始化熔断器

        Args:
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时时间
            half_open_requests: 半开状态允许的请求数
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_count = 0
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        """检查是否可以执行"""
        async with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True

            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_count = 0
                    return True
                return False

            # HALF_OPEN
            if self.half_open_count < self.half_open_requests:
                self.half_open_count += 1
                return True
            return False

    async def record_success(self):
        """记录成功"""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_count += 1
                if self.half_open_count >= self.half_open_requests:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0

    async def record_failure(self):
        """记录失败"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN


# ==================== 负载均衡器 ====================

class LoadBalancer:
    """负载均衡器"""

    def __init__(
        self,
        servers: List[str],
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        weights: Optional[List[int]] = None,
        health_check_interval: float = 10.0,
        enable_circuit_breaker: bool = True,
        rate_limit: Optional[float] = None,
        rate_capacity: int = 100
    ):
        """
        初始化负载均衡器

        Args:
            servers: 服务器地址列表
            strategy: 负载均衡策略
            weights: 服务器权重列表
            health_check_interval: 健康检查间隔
            enable_circuit_breaker: 是否启用熔断器
            rate_limit: 限流速率 (每秒请求数)
            rate_capacity: 限流桶容量
        """
        # 初始化服务器
        self.servers = []
        for i, url in enumerate(servers):
            weight = weights[i] if weights and i < len(weights) else 1
            self.servers.append(Server(url=url, weight=weight))

        # 初始化策略
        self.strategy_type = strategy
        strategy_class = STRATEGY_MAP.get(strategy, RoundRobinStrategy)
        self.strategy = strategy_class()

        # 健康检查
        self.health_checker = HealthChecker(
            check_interval=health_check_interval
        )
        self._health_check_task = None

        # 连接池
        self.pool = ConnectionPool() if HTTPX_AVAILABLE else None

        # 熔断器
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        if enable_circuit_breaker:
            for server in self.servers:
                self.circuit_breakers[server.url] = CircuitBreaker()

        # 限流器
        self.rate_limiter = None
        if rate_limit:
            self.rate_limiter = RateLimiter(rate_limit, rate_capacity)

    def _get_healthy_servers(self) -> List[Server]:
        """获取健康的服务器"""
        healthy = [s for s in self.servers if s.healthy]
        if not healthy:
            # 如果没有健康服务器，返回所有服务器
            return self.servers
        return healthy

    async def _select_server(self, client_ip: Optional[str] = None) -> Server:
        """选择服务器"""
        healthy = self._get_healthy_servers()

        # 过滤熔断的服务器
        if self.enable_circuit_breaker:
            available = []
            for server in healthy:
                breaker = self.circuit_breakers.get(server.url)
                if breaker and await breaker.can_execute():
                    available.append(server)
            if available:
                healthy = available

        return self.strategy.select(healthy, client_ip)

    async def request(
        self,
        path: str,
        method: str = "POST",
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
        client_ip: Optional[str] = None,
        timeout: float = 30.0,
        retries: int = 3
    ) -> dict:
        """
        发送请求

        Args:
            path: 请求路径
            method: HTTP 方法
            data: 请求数据
            headers: 请求头
            client_ip: 客户端 IP
            timeout: 超时时间
            retries: 重试次数

        Returns:
            响应数据
        """
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not available")

        # 限流检查
        if self.rate_limiter:
            if not await self.rate_limiter.wait_for_token():
                raise RuntimeError("Rate limit exceeded")

        last_error = None
        for attempt in range(retries):
            server = await self._select_server(client_ip)
            server.connections += 1

            start_time = time.time()
            try:
                client = await self.pool.get_client()
                url = f"{server.url}{path}"

                response = await client.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                    timeout=timeout
                )
                response.raise_for_status()

                response_time = time.time() - start_time
                server.record_response(response_time, success=True)

                if self.enable_circuit_breaker:
                    breaker = self.circuit_breakers.get(server.url)
                    if breaker:
                        await breaker.record_success()

                return response.json()

            except Exception as e:
                response_time = time.time() - start_time
                server.record_response(response_time, success=False)

                if self.enable_circuit_breaker:
                    breaker = self.circuit_breakers.get(server.url)
                    if breaker:
                        await breaker.record_failure()

                last_error = e

            finally:
                server.connections -= 1

        raise last_error or RuntimeError("All retries failed")

    async def start_health_check(self):
        """启动健康检查"""
        self._health_check_task = asyncio.create_task(
            self.health_checker.start(self.servers)
        )

    async def stop_health_check(self):
        """停止健康检查"""
        self.health_checker.stop()
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    async def close(self):
        """关闭负载均衡器"""
        await self.stop_health_check()
        if self.pool:
            await self.pool.close()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "strategy": self.strategy_type.value,
            "servers": [
                {
                    "url": s.url,
                    "healthy": s.healthy,
                    "connections": s.connections,
                    "avg_response_time": s.avg_response_time,
                    "success_rate": s.success_rate,
                    "weight": s.weight
                }
                for s in self.servers
            ]
        }

    async def __aenter__(self):
        await self.start_health_check()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ==================== 便捷函数 ====================

def create_load_balancer(
    servers: List[str],
    strategy: str = "round_robin",
    **kwargs
) -> LoadBalancer:
    """
    创建负载均衡器

    Args:
        servers: 服务器地址列表
        strategy: 策略名称
        **kwargs: 其他参数

    Returns:
        负载均衡器实例
    """
    strategy_enum = LoadBalanceStrategy(strategy)
    return LoadBalancer(servers=servers, strategy=strategy_enum, **kwargs)


async def quick_request(
    servers: List[str],
    path: str,
    data: Optional[dict] = None,
    method: str = "POST"
) -> dict:
    """
    快速请求

    Args:
        servers: 服务器列表
        path: 请求路径
        data: 请求数据
        method: HTTP 方法

    Returns:
        响应数据
    """
    balancer = LoadBalancer(servers=servers)
    try:
        return await balancer.request(path=path, method=method, data=data)
    finally:
        await balancer.close()
