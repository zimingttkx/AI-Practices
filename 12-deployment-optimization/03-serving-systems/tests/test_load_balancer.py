"""
负载均衡模块测试
"""

import pytest
import sys
import asyncio
import time
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from load_balancer import (
    HTTPX_AVAILABLE,
    LoadBalanceStrategy,
    CircuitBreakerState,
    Server,
    RoundRobinStrategy,
    WeightedRoundRobinStrategy,
    LeastConnectionsStrategy,
    IPHashStrategy,
    RandomStrategy,
    ResponseTimeStrategy,
    RateLimiter,
    CircuitBreaker,
    HealthChecker,
)


# ==================== Server 测试 ====================

class TestServer:
    """Server 测试"""

    def test_server_creation(self):
        """测试服务器创建"""
        server = Server(url="http://localhost:8000", weight=2)
        assert server.url == "http://localhost:8000"
        assert server.weight == 2
        assert server.healthy is True
        assert server.connections == 0

    def test_avg_response_time_empty(self):
        """测试空响应时间"""
        server = Server(url="http://localhost:8000")
        assert server.avg_response_time == float('inf')

    def test_avg_response_time_with_data(self):
        """测试有数据的响应时间"""
        server = Server(url="http://localhost:8000")
        for i in range(10):
            server.record_response(float(i), success=True)

        # 平均值应该是 4.5
        assert 4.0 < server.avg_response_time < 5.0

    def test_success_rate(self):
        """测试成功率"""
        server = Server(url="http://localhost:8000")

        for _ in range(8):
            server.record_response(1.0, success=True)
        for _ in range(2):
            server.record_response(1.0, success=False)

        assert server.success_rate == 0.8

    def test_reset_stats(self):
        """测试重置统计"""
        server = Server(url="http://localhost:8000")
        server.record_response(1.0, success=True)
        server.record_response(1.0, success=False)

        server.reset_stats()

        assert len(server.response_times) == 0
        assert server.success_count == 0
        assert server.failure_count == 0


# ==================== 负载均衡策略测试 ====================

class TestRoundRobinStrategy:
    """轮询策略测试"""

    @pytest.fixture
    def strategy(self):
        return RoundRobinStrategy()

    @pytest.fixture
    def servers(self):
        return [
            Server(url="http://server1:8000"),
            Server(url="http://server2:8000"),
            Server(url="http://server3:8000"),
        ]

    def test_round_robin_order(self, strategy, servers):
        """测试轮询顺序"""
        results = []
        for _ in range(6):
            server = strategy.select(servers)
            results.append(server.url)

        # 应该按顺序循环
        assert results == [
            "http://server1:8000",
            "http://server2:8000",
            "http://server3:8000",
            "http://server1:8000",
            "http://server2:8000",
            "http://server3:8000",
        ]

    def test_empty_servers(self, strategy):
        """测试空服务器列表"""
        with pytest.raises(RuntimeError):
            strategy.select([])


class TestWeightedRoundRobinStrategy:
    """加权轮询策略测试"""

    @pytest.fixture
    def strategy(self):
        return WeightedRoundRobinStrategy()

    @pytest.fixture
    def servers(self):
        return [
            Server(url="http://server1:8000", weight=3),
            Server(url="http://server2:8000", weight=2),
            Server(url="http://server3:8000", weight=1),
        ]

    def test_weighted_distribution(self, strategy, servers):
        """测试加权分布"""
        counts = {"http://server1:8000": 0, "http://server2:8000": 0, "http://server3:8000": 0}

        for _ in range(60):
            server = strategy.select(servers)
            counts[server.url] += 1

        # 权重比例应该大致为 3:2:1
        assert counts["http://server1:8000"] > counts["http://server2:8000"]
        assert counts["http://server2:8000"] > counts["http://server3:8000"]


class TestLeastConnectionsStrategy:
    """最少连接策略测试"""

    @pytest.fixture
    def strategy(self):
        return LeastConnectionsStrategy()

    def test_select_least_connections(self, strategy):
        """测试选择最少连接"""
        servers = [
            Server(url="http://server1:8000"),
            Server(url="http://server2:8000"),
            Server(url="http://server3:8000"),
        ]
        servers[0].connections = 5
        servers[1].connections = 2
        servers[2].connections = 8

        selected = strategy.select(servers)
        assert selected.url == "http://server2:8000"


class TestIPHashStrategy:
    """IP 哈希策略测试"""

    @pytest.fixture
    def strategy(self):
        return IPHashStrategy()

    @pytest.fixture
    def servers(self):
        return [
            Server(url="http://server1:8000"),
            Server(url="http://server2:8000"),
            Server(url="http://server3:8000"),
        ]

    def test_consistent_hashing(self, strategy, servers):
        """测试一致性哈希"""
        client_ip = "192.168.1.100"

        # 相同 IP 应该总是选择相同服务器
        results = set()
        for _ in range(10):
            server = strategy.select(servers, client_ip)
            results.add(server.url)

        assert len(results) == 1

    def test_different_ips(self, strategy, servers):
        """测试不同 IP"""
        ips = ["192.168.1.1", "192.168.1.2", "10.0.0.1", "172.16.0.1"]
        results = set()

        for ip in ips:
            server = strategy.select(servers, ip)
            results.add(server.url)

        # 不同 IP 可能选择不同服务器
        assert len(results) >= 1

    def test_no_client_ip(self, strategy, servers):
        """测试无客户端 IP"""
        # 应该随机选择
        server = strategy.select(servers, None)
        assert server in servers


class TestRandomStrategy:
    """随机策略测试"""

    @pytest.fixture
    def strategy(self):
        return RandomStrategy()

    @pytest.fixture
    def servers(self):
        return [
            Server(url="http://server1:8000"),
            Server(url="http://server2:8000"),
            Server(url="http://server3:8000"),
        ]

    def test_random_selection(self, strategy, servers):
        """测试随机选择"""
        results = set()
        for _ in range(100):
            server = strategy.select(servers)
            results.add(server.url)

        # 应该选择到多个服务器
        assert len(results) > 1


class TestResponseTimeStrategy:
    """响应时间策略测试"""

    @pytest.fixture
    def strategy(self):
        return ResponseTimeStrategy()

    def test_select_fastest(self, strategy):
        """测试选择最快服务器"""
        servers = [
            Server(url="http://server1:8000"),
            Server(url="http://server2:8000"),
            Server(url="http://server3:8000"),
        ]

        # 设置响应时间
        for _ in range(5):
            servers[0].record_response(100.0, True)
            servers[1].record_response(50.0, True)
            servers[2].record_response(200.0, True)

        selected = strategy.select(servers)
        assert selected.url == "http://server2:8000"


# ==================== RateLimiter 测试 ====================

class TestRateLimiter:
    """RateLimiter 测试"""

    @pytest.fixture
    def limiter(self):
        return RateLimiter(rate=10.0, capacity=5)

    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter):
        """测试成功获取令牌"""
        result = await limiter.acquire()
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_exhausted(self, limiter):
        """测试令牌耗尽"""
        # 消耗所有令牌
        for _ in range(5):
            await limiter.acquire()

        # 下一次应该失败
        result = await limiter.acquire()
        assert result is False

    @pytest.mark.asyncio
    async def test_token_refill(self, limiter):
        """测试令牌补充"""
        # 消耗所有令牌
        for _ in range(5):
            await limiter.acquire()

        # 等待令牌补充
        await asyncio.sleep(0.2)

        # 应该能获取令牌
        result = await limiter.acquire()
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_token(self, limiter):
        """测试等待令牌"""
        # 消耗所有令牌
        for _ in range(5):
            await limiter.acquire()

        # 等待获取令牌
        result = await limiter.wait_for_token(timeout=1.0)
        assert result is True


# ==================== CircuitBreaker 测试 ====================

class TestCircuitBreaker:
    """CircuitBreaker 测试"""

    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.5,
            half_open_requests=2
        )

    @pytest.mark.asyncio
    async def test_initial_state(self, breaker):
        """测试初始状态"""
        assert breaker.state == CircuitBreakerState.CLOSED
        assert await breaker.can_execute() is True

    @pytest.mark.asyncio
    async def test_open_after_failures(self, breaker):
        """测试失败后打开"""
        for _ in range(3):
            await breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN
        assert await breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, breaker):
        """测试超时后半开"""
        # 触发熔断
        for _ in range(3):
            await breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN

        # 等待恢复超时
        await asyncio.sleep(0.6)

        # 应该转为半开
        assert await breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_close_after_success(self, breaker):
        """测试成功后关闭"""
        # 触发熔断
        for _ in range(3):
            await breaker.record_failure()

        # 等待恢复
        await asyncio.sleep(0.6)
        await breaker.can_execute()

        # 记录成功
        for _ in range(2):
            await breaker.record_success()

        assert breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_reopen_on_failure_in_half_open(self, breaker):
        """测试半开状态失败后重新打开"""
        # 触发熔断
        for _ in range(3):
            await breaker.record_failure()

        # 等待恢复
        await asyncio.sleep(0.6)
        await breaker.can_execute()

        # 在半开状态记录失败
        await breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN


# ==================== HealthChecker 测试 ====================

class TestHealthChecker:
    """HealthChecker 测试"""

    @pytest.fixture
    def checker(self):
        return HealthChecker(
            check_interval=1.0,
            timeout=1.0,
            healthy_threshold=2,
            unhealthy_threshold=2
        )

    @pytest.fixture
    def servers(self):
        return [
            Server(url="http://localhost:8000"),
            Server(url="http://localhost:8001"),
        ]

    def test_checker_creation(self, checker):
        """测试检查器创建"""
        assert checker.check_interval == 1.0
        assert checker.timeout == 1.0
        assert checker.healthy_threshold == 2
        assert checker.unhealthy_threshold == 2


# ==================== LoadBalancer 测试 ====================

class TestLoadBalancer:
    """LoadBalancer 测试"""

    def test_balancer_creation(self):
        """测试负载均衡器创建"""
        from load_balancer import LoadBalancer

        balancer = LoadBalancer(
            servers=["http://server1:8000", "http://server2:8000"],
            strategy=LoadBalanceStrategy.ROUND_ROBIN
        )

        assert len(balancer.servers) == 2
        assert balancer.strategy_type == LoadBalanceStrategy.ROUND_ROBIN

    def test_balancer_with_weights(self):
        """测试带权重的负载均衡器"""
        from load_balancer import LoadBalancer

        balancer = LoadBalancer(
            servers=["http://server1:8000", "http://server2:8000"],
            strategy=LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN,
            weights=[3, 1]
        )

        assert balancer.servers[0].weight == 3
        assert balancer.servers[1].weight == 1

    def test_get_healthy_servers(self):
        """测试获取健康服务器"""
        from load_balancer import LoadBalancer

        balancer = LoadBalancer(
            servers=["http://server1:8000", "http://server2:8000", "http://server3:8000"]
        )

        balancer.servers[1].healthy = False

        healthy = balancer._get_healthy_servers()
        assert len(healthy) == 2

    def test_get_stats(self):
        """测试获取统计信息"""
        from load_balancer import LoadBalancer

        balancer = LoadBalancer(
            servers=["http://server1:8000", "http://server2:8000"]
        )

        stats = balancer.get_stats()

        assert "strategy" in stats
        assert "servers" in stats
        assert len(stats["servers"]) == 2


# ==================== 便捷函数测试 ====================

class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_create_load_balancer(self):
        """测试 create_load_balancer"""
        from load_balancer import create_load_balancer

        balancer = create_load_balancer(
            servers=["http://server1:8000", "http://server2:8000"],
            strategy="round_robin"
        )

        assert balancer is not None
        assert balancer.strategy_type == LoadBalanceStrategy.ROUND_ROBIN

    def test_create_load_balancer_least_connections(self):
        """测试创建最少连接策略"""
        from load_balancer import create_load_balancer

        balancer = create_load_balancer(
            servers=["http://server1:8000"],
            strategy="least_connections"
        )

        assert balancer.strategy_type == LoadBalanceStrategy.LEAST_CONNECTIONS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
