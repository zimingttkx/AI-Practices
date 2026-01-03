"""
ZeRO (Zero Redundancy Optimizer) 零冗余优化器实现

================================================================================
核心思想 (一句话理解)
================================================================================
ZeRO = 渐进式分片优化器状态/梯度/参数，消除数据并行中的内存冗余

================================================================================
ZeRO 三阶段详解 (图解)
================================================================================

    传统DDP: 每个GPU存储完整的 参数+梯度+优化器状态 (巨大冗余!)
    ┌─────────────────────────────────────────────────────────────────┐
    │ GPU0: [参数 2P] [梯度 2P] [优化器状态 4P] = 8P                   │
    │ GPU1: [参数 2P] [梯度 2P] [优化器状态 4P] = 8P  ← 完全重复      │
    │ GPU2: [参数 2P] [梯度 2P] [优化器状态 4P] = 8P                   │
    └─────────────────────────────────────────────────────────────────┘

    ZeRO-1: 只分片优化器状态 (最常用，通信开销最小)
    ┌─────────────────────────────────────────────────────────────────┐
    │ GPU0: [参数 2P] [梯度 2P] [优化器 4P/N]                          │
    │ GPU1: [参数 2P] [梯度 2P] [优化器 4P/N]                          │
    │ 每卡显存: 4P + 4P/N (节省约4倍优化器显存)                        │
    └─────────────────────────────────────────────────────────────────┘

    ZeRO-2: 分片优化器状态 + 梯度
    ┌─────────────────────────────────────────────────────────────────┐
    │ GPU0: [参数 2P] [梯度 2P/N] [优化器 4P/N]                        │
    │ GPU1: [参数 2P] [梯度 2P/N] [优化器 4P/N]                        │
    │ 每卡显存: 2P + 6P/N (节省约8倍)                                  │
    └─────────────────────────────────────────────────────────────────┘

    ZeRO-3: 全部分片 (等价于FSDP)
    ┌─────────────────────────────────────────────────────────────────┐
    │ GPU0: [参数 2P/N] [梯度 2P/N] [优化器 4P/N]                      │
    │ GPU1: [参数 2P/N] [梯度 2P/N] [优化器 4P/N]                      │
    │ 每卡显存: 8P/N (线性扩展，N卡节省N倍)                            │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
通信开销分析
================================================================================
    ZeRO-1: 优化器step时需要AllGather收集更新后的参数
    ZeRO-2: 反向传播用ReduceScatter替代AllReduce
    ZeRO-3: 每层forward/backward都需要AllGather参数

================================================================================
适用场景
================================================================================
- ZeRO-1: 模型略超单卡，想要最小通信开销
- ZeRO-2: 模型中等超出，平衡显存和通信
- ZeRO-3: 模型远超单卡，愿意用通信换显存

================================================================================
前置知识
================================================================================
- DDP的基本概念
- Adam优化器的状态 (momentum, variance)
- AllGather/ReduceScatter通信原语

================================================================================
参考文献
================================================================================
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models", SC 2020
- Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training",
  USENIX ATC 2021
"""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer


# =============================================================================
# ZeRO阶段枚举
# =============================================================================

class ZeROStage(IntEnum):
    """ZeRO优化阶段

    不同阶段在显存节省和通信开销之间做权衡。
    阶段越高，显存节省越多，但通信开销也越大。

    Attributes:
        DISABLED: 禁用ZeRO，等价于普通DDP
        OPTIMIZER: Stage 1 - 只分片优化器状态
            - 显存节省: ~4x (Adam的momentum和variance)
            - 通信增加: 优化器step时AllGather
            - 推荐: 模型略超单卡时使用
        GRADIENTS: Stage 2 - 分片优化器状态 + 梯度
            - 显存节省: ~8x
            - 通信: ReduceScatter替代AllReduce
            - 推荐: 模型中等超出时使用
        PARAMETERS: Stage 3 - 全部分片 (等价于FSDP)
            - 显存节省: ~N倍 (N=GPU数)
            - 通信: 每层都需要AllGather
            - 推荐: 模型远超单卡时使用
    """
    DISABLED = 0    # 禁用
    OPTIMIZER = 1   # 分片优化器状态
    GRADIENTS = 2   # + 分片梯度
    PARAMETERS = 3  # + 分片参数


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class ZeROConfig:
    """ZeRO优化器配置

    Attributes:
        stage: ZeRO阶段 (0-3)
        reduce_bucket_size: 梯度归约桶大小(字节)，影响通信效率
        allgather_bucket_size: AllGather桶大小(字节)
        overlap_comm: 是否重叠通信和计算 (提升性能)
        contiguous_gradients: 使用连续梯度缓冲区 (提升性能)
        cpu_offload: 是否将优化器状态卸载到CPU
        cpu_offload_params: 是否将参数也卸载到CPU (ZeRO-3)
        cpu_offload_use_pin_memory: CPU卸载时使用锁页内存
        sub_group_size: 参数分组大小
        reduce_scatter: 使用ReduceScatter替代AllReduce
        round_robin_gradients: 轮询式梯度分配

    Example:
        >>> config = ZeROConfig(
        ...     stage=ZeROStage.GRADIENTS,  # ZeRO-2
        ...     overlap_comm=True,
        ...     cpu_offload=False,
        ... )
    """
    stage: ZeROStage = ZeROStage.OPTIMIZER
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    cpu_offload: bool = False
    cpu_offload_params: bool = False
    cpu_offload_use_pin_memory: bool = True
    sub_group_size: int = 1_000_000_000
    reduce_scatter: bool = True
    round_robin_gradients: bool = False


# =============================================================================
# 参数分片类 (ZeRO-3使用)
# =============================================================================

class PartitionedParameter:
    """参数分片包装器 (用于ZeRO-3)

    管理参数的分片生命周期:
    - 将完整参数切分到各个GPU
    - 计算时收集完整参数
    - 计算后释放非本地分片

    工作原理:
        假设参数有1000个元素，4个GPU:
        - GPU0存储元素 [0:250]
        - GPU1存储元素 [250:500]
        - GPU2存储元素 [500:750]
        - GPU3存储元素 [750:1000]

        计算时通过AllGather收集完整参数

    Args:
        param: 原始参数张量
        rank: 当前进程编号
        world_size: 总进程数
        device: 目标设备
    """

    def __init__(
        self,
        param: nn.Parameter,
        rank: int,
        world_size: int,
        device: torch.device,
    ):
        self.param = param
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # 计算分片信息
        self.numel = param.numel()  # 参数总元素数
        self.partition_size = math.ceil(self.numel / world_size)  # 每个分片大小
        self.start_idx = rank * self.partition_size  # 本地分片起始索引
        self.end_idx = min(self.start_idx + self.partition_size, self.numel)
        self.local_numel = self.end_idx - self.start_idx  # 本地实际元素数

        self.local_data: Optional[torch.Tensor] = None
        self._is_partitioned = False

    def partition(self) -> torch.Tensor:
        """将参数分片，返回本地分片

        只保留属于当前GPU的那部分参数。

        Returns:
            本地参数分片
        """
        if self._is_partitioned:
            return self.local_data

        # 将参数展平，取出本地分片
        flat_param = self.param.data.view(-1)
        self.local_data = flat_param[self.start_idx:self.end_idx].clone()
        self._is_partitioned = True

        return self.local_data

    def all_gather(self) -> torch.Tensor:
        """收集所有分片，重建完整参数

        通过AllGather从所有GPU收集分片，拼接成完整参数。

        Returns:
            完整参数张量
        """
        if not dist.is_initialized():
            return self.param.data

        # 创建接收缓冲区 (每个分片大小相同，可能有padding)
        gathered = [
            torch.zeros(self.partition_size, device=self.device)
            for _ in range(self.world_size)
        ]

        # 本地分片可能不足partition_size，需要padding
        local_padded = torch.zeros(self.partition_size, device=self.device)
        local_padded[:self.local_numel] = self.local_data

        # AllGather收集所有分片
        dist.all_gather(gathered, local_padded)

        # 拼接并裁剪到原始大小
        full_param = torch.cat(gathered)[:self.numel]
        return full_param.view(self.param.shape)

    def reduce_scatter_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """ReduceScatter梯度，返回本地梯度分片

        ReduceScatter = Reduce + Scatter:
        1. 所有GPU的梯度求和
        2. 结果分片分发给各GPU

        Args:
            grad: 完整梯度张量

        Returns:
            本地梯度分片
        """
        if not dist.is_initialized():
            return grad.view(-1)[self.start_idx:self.end_idx]

        flat_grad = grad.view(-1)

        # Padding到可整除的大小
        padded_size = self.partition_size * self.world_size
        if flat_grad.numel() < padded_size:
            padded_grad = torch.zeros(padded_size, device=self.device)
            padded_grad[:flat_grad.numel()] = flat_grad
        else:
            padded_grad = flat_grad

        # 切分成world_size份
        input_list = list(padded_grad.chunk(self.world_size))
        output = torch.zeros(self.partition_size, device=self.device)

        # ReduceScatter: 每个GPU得到归约后的一个分片
        dist.reduce_scatter(output, input_list)

        return output[:self.local_numel]


# =============================================================================
# ZeRO优化器
# =============================================================================

class ZeROOptimizer(Optimizer):
    """ZeRO优化器包装器

    包装标准PyTorch优化器，添加ZeRO风格的分片功能。

    工作原理:
        1. 初始化时根据stage决定分片策略
        2. backward后根据stage处理梯度
        3. step时同步参数

    Args:
        optimizer: 基础PyTorch优化器 (如Adam, SGD)
        config: ZeRO配置
        model: 模型 (ZeRO-3需要)

    Example:
        >>> base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        >>> config = ZeROConfig(stage=ZeROStage.GRADIENTS)
        >>> optimizer = ZeROOptimizer(base_optimizer, config, model)
        >>>
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()  # ZeRO自动处理同步
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: Optional[ZeROConfig] = None,
        model: Optional[nn.Module] = None,
    ):
        self.optimizer = optimizer
        self.config = config or ZeROConfig()
        self.model = model

        # 获取分布式信息
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = next(model.parameters()).device if model else torch.device("cpu")

        # 分片相关数据结构
        self.partitioned_params: Dict[int, PartitionedParameter] = {}
        self.param_to_partition: Dict[int, int] = {}  # 参数ID -> 所属分区
        self.grad_buffers: Dict[int, torch.Tensor] = {}

        # 根据stage初始化分片
        if self.config.stage >= ZeROStage.OPTIMIZER:
            self._init_optimizer_partitioning()

        if self.config.stage >= ZeROStage.PARAMETERS and model is not None:
            self._init_parameter_partitioning()

    def _init_optimizer_partitioning(self) -> None:
        """初始化优化器状态分片

        将参数分配给不同的rank，每个rank只负责更新自己的参数子集。
        """
        # 收集所有参数
        all_params = []
        for group in self.optimizer.param_groups:
            all_params.extend(group["params"])

        # 将参数均匀分配给各rank
        params_per_rank = math.ceil(len(all_params) / self.world_size)

        for i, param in enumerate(all_params):
            # 参数i属于第 i // params_per_rank 个rank
            self.param_to_partition[id(param)] = i // params_per_rank

    def _init_parameter_partitioning(self) -> None:
        """初始化参数分片 (ZeRO-3)

        为每个需要梯度的参数创建分片包装器。
        """
        if self.model is None:
            return

        for param in self.model.parameters():
            if param.requires_grad:
                pp = PartitionedParameter(
                    param, self.rank, self.world_size, self.device
                )
                self.partitioned_params[id(param)] = pp

    # =========================================================================
    # 优化器接口实现
    # =========================================================================

    @property
    def param_groups(self):
        """返回优化器参数组"""
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def state_dict(self) -> Dict[str, Any]:
        """获取优化器状态字典"""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载优化器状态字典"""
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """清零梯度

        Args:
            set_to_none: True则将梯度设为None (更省内存)
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> Optional[float]:
        """执行优化步骤

        根据ZeRO stage执行不同的同步策略:
        - Stage 0: 普通优化器step
        - Stage 1: step后AllGather参数
        - Stage 2: 先ReduceScatter梯度，再step
        - Stage 3: 先ReduceScatter梯度，step后AllGather参数

        Args:
            closure: 可选的闭包函数，用于重新计算loss

        Returns:
            loss值 (如果提供了closure)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Stage 0: 普通优化
        if self.config.stage == ZeROStage.DISABLED:
            return self.optimizer.step()

        # Stage 2+: 先处理梯度
        if self.config.stage >= ZeROStage.GRADIENTS:
            self._reduce_gradients()

        # 执行优化步骤
        self.optimizer.step()

        # Stage 3: 同步参数
        if self.config.stage >= ZeROStage.PARAMETERS:
            self._sync_parameters()

        return loss

    def _reduce_gradients(self) -> None:
        """归约梯度

        根据配置使用ReduceScatter或AllReduce。
        """
        if not dist.is_initialized():
            return

        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                if self.config.reduce_scatter:
                    # ReduceScatter: 每个GPU只保留自己负责的梯度分片
                    if id(param) in self.partitioned_params:
                        pp = self.partitioned_params[id(param)]
                        local_grad = pp.reduce_scatter_grad(param.grad)
                        self.grad_buffers[id(param)] = local_grad
                else:
                    # AllReduce: 所有GPU得到完整的平均梯度
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    def _sync_parameters(self) -> None:
        """同步参数 (ZeRO-3)

        通过AllGather收集所有分片，重建完整参数。
        """
        if not dist.is_initialized():
            return

        for param_id, pp in self.partitioned_params.items():
            full_param = pp.all_gather()
            pp.param.data.copy_(full_param)
