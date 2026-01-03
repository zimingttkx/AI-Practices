"""
ZeRO (Zero Redundancy Optimizer) 实现

ZeRO 是 DeepSpeed 提出的优化器状态分片技术，通过分片优化器状态、
梯度和参数来减少内存冗余。

ZeRO 阶段:
    - ZeRO-1: 优化器状态分片
    - ZeRO-2: 优化器状态 + 梯度分片
    - ZeRO-3: 优化器状态 + 梯度 + 参数分片
"""

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer


class ZeROStage(IntEnum):
    """ZeRO 阶段枚举"""
    DISABLED = 0      # 不启用 ZeRO
    OPTIMIZER = 1     # ZeRO-1: 优化器状态分片
    GRADIENTS = 2     # ZeRO-2: + 梯度分片
    PARAMETERS = 3    # ZeRO-3: + 参数分片


@dataclass
class ZeROConfig:
    """ZeRO 配置类
    
    Attributes:
        stage: ZeRO 阶段
        reduce_bucket_size: 梯度归约桶大小
        allgather_bucket_size: AllGather 桶大小
        overlap_comm: 是否重叠通信和计算
        contiguous_gradients: 是否使用连续梯度
        cpu_offload: 是否卸载到 CPU
        cpu_offload_params: 是否卸载参数到 CPU
        cpu_offload_use_pin_memory: 是否使用固定内存
        sub_group_size: 子组大小（用于分片）
        reduce_scatter: 是否使用 ReduceScatter
        round_robin_gradients: 是否轮询梯度
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


class PartitionedParameter:
    """分片参数包装器
    
    用于 ZeRO-3 的参数分片管理。
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
        self.numel = param.numel()
        self.partition_size = math.ceil(self.numel / world_size)
        self.start_idx = rank * self.partition_size
        self.end_idx = min(self.start_idx + self.partition_size, self.numel)
        self.local_numel = self.end_idx - self.start_idx
        
        # 存储本地分片
        self.local_data: Optional[torch.Tensor] = None
        self._is_partitioned = False
    
    def partition(self) -> torch.Tensor:
        """将参数分片，返回本地分片"""
        if self._is_partitioned:
            return self.local_data
        
        flat_param = self.param.data.view(-1)
        self.local_data = flat_param[self.start_idx:self.end_idx].clone()
        self._is_partitioned = True
        
        return self.local_data
    
    def all_gather(self) -> torch.Tensor:
        """收集所有分片，重建完整参数"""
        if not dist.is_initialized():
            return self.param.data
        
        # 准备接收缓冲区
        gathered = [
            torch.zeros(self.partition_size, device=self.device)
            for _ in range(self.world_size)
        ]
        
        # 填充本地数据
        local_padded = torch.zeros(self.partition_size, device=self.device)
        local_padded[:self.local_numel] = self.local_data
        
        # AllGather
        dist.all_gather(gathered, local_padded)
        
        # 重建完整参数
        full_param = torch.cat(gathered)[:self.numel]
        return full_param.view(self.param.shape)
    
    def reduce_scatter_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """ReduceScatter 梯度，返回本地梯度分片"""
        if not dist.is_initialized():
            return grad.view(-1)[self.start_idx:self.end_idx]
        
        flat_grad = grad.view(-1)
        
        # 填充到可整除大小
        padded_size = self.partition_size * self.world_size
        if flat_grad.numel() < padded_size:
            padded_grad = torch.zeros(padded_size, device=self.device)
            padded_grad[:flat_grad.numel()] = flat_grad
        else:
            padded_grad = flat_grad
        
        # 分割为多个分片
        input_list = list(padded_grad.chunk(self.world_size))
        output = torch.zeros(self.partition_size, device=self.device)
        
        # ReduceScatter
        dist.reduce_scatter(output, input_list)
        
        return output[:self.local_numel]


class ZeROOptimizer(Optimizer):
    """ZeRO 优化器包装器
    
    将标准 PyTorch 优化器包装为支持 ZeRO 分片的优化器。
    
    Args:
        optimizer: 基础优化器
        config: ZeRO 配置
        model: 模型（用于参数分片）
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
        
        # 分布式信息
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = next(model.parameters()).device if model else torch.device("cpu")
        
        # 参数分片管理
        self.partitioned_params: Dict[int, PartitionedParameter] = {}
        self.param_to_partition: Dict[int, int] = {}
        
        # 梯度缓冲区
        self.grad_buffers: Dict[int, torch.Tensor] = {}
        
        # 初始化分片
        if self.config.stage >= ZeROStage.OPTIMIZER:
            self._init_optimizer_partitioning()
        
        if self.config.stage >= ZeROStage.PARAMETERS and model is not None:
            self._init_parameter_partitioning()
    
    def _init_optimizer_partitioning(self) -> None:
        """初始化优化器状态分片"""
        # 将参数分配到不同的 rank
        all_params = []
        for group in self.optimizer.param_groups:
            all_params.extend(group["params"])
        
        # 按 rank 分配参数
        params_per_rank = math.ceil(len(all_params) / self.world_size)
        start_idx = self.rank * params_per_rank
        end_idx = min(start_idx + params_per_rank, len(all_params))
        
        for i, param in enumerate(all_params):
            self.param_to_partition[id(param)] = i // params_per_rank
    
    def _init_parameter_partitioning(self) -> None:
        """初始化参数分片 (ZeRO-3)"""
        if self.model is None:
            return
        
        for param in self.model.parameters():
            if param.requires_grad:
                pp = PartitionedParameter(
                    param, self.rank, self.world_size, self.device
                )
                self.partitioned_params[id(param)] = pp
    
    @property
    def param_groups(self):
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
        """清零梯度"""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None) -> Optional[float]:
        """执行优化步骤"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if self.config.stage == ZeROStage.DISABLED:
            return self.optimizer.step()
        
        # ZeRO-1/2: 同步梯度
        if self.config.stage >= ZeROStage.GRADIENTS:
            self._reduce_gradients()
        
        # 执行优化步骤
        self.optimizer.step()
        
        # ZeRO-3: 同步参数
        if self.config.stage >= ZeROStage.PARAMETERS:
            self._sync_parameters()
        
        return loss
    
    def _reduce_gradients(self) -> None:
        """归约梯度"""
        if not dist.is_initialized():
            return
        
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                if self.config.reduce_scatter:
                    # 使用 ReduceScatter
                    if id(param) in self.partitioned_params:
                        pp = self.partitioned_params[id(param)]
                        local_grad = pp.reduce_scatter_grad(param.grad)
                        self.grad_buffers[id(param)] = local_grad
                else:
                    # 使用 AllReduce
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    
    def _sync_parameters(self) -> None:
        """同步参数 (ZeRO-3)"""
        if not dist.is_initialized():
            return
        
        for param_id, pp in self.partitioned_params.items():
            full_param = pp.all_gather()
            # 更新原始参数
            pp.param.data.copy_(full_param)
