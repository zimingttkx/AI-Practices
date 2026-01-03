"""
ZeRO (Zero Redundancy Optimizer) Implementation

Core Idea:
    ZeRO eliminates memory redundancy in data-parallel training by partitioning
    optimizer states, gradients, and parameters across GPUs instead of replicating
    them on each device.

Mathematical Theory:
    For a model with P parameters and Adam optimizer (2P states), memory per GPU:
    
    .. math::
        M_{DDP} = 2P + 2P + 4P = 8P  \\quad \\text{(FP16 params/grads + FP32 optimizer)}
    
    ZeRO stages reduce this progressively:
    
    .. math::
        M_{ZeRO-1} = 2P + 2P + \\frac{4P}{N} = 4P + \\frac{4P}{N}
        
        M_{ZeRO-2} = 2P + \\frac{2P}{N} + \\frac{4P}{N} = 2P + \\frac{6P}{N}
        
        M_{ZeRO-3} = \\frac{2P}{N} + \\frac{2P}{N} + \\frac{4P}{N} = \\frac{8P}{N}

Problem Statement:
    Large models require more memory than available on a single GPU. Traditional
    data parallelism replicates all states, wasting memory. ZeRO partitions states
    to achieve model parallelism memory efficiency with data parallelism simplicity.

Comparison:
    - ZeRO-1 vs DDP: 4x memory reduction for optimizer states
    - ZeRO-2 vs ZeRO-1: Additional gradient memory savings
    - ZeRO-3 vs FSDP: Equivalent functionality, DeepSpeed implementation

Complexity:
    - ZeRO-1: O(P) AllGather for optimizer step
    - ZeRO-2: O(P) ReduceScatter for gradients
    - ZeRO-3: O(P) AllGather per layer forward/backward

References:
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


class ZeROStage(IntEnum):
    """ZeRO optimization stages.
    
    Attributes:
        DISABLED: No ZeRO optimization.
        OPTIMIZER: Stage 1 - Partition optimizer states.
        GRADIENTS: Stage 2 - Partition optimizer states and gradients.
        PARAMETERS: Stage 3 - Partition everything including parameters.
    """
    DISABLED = 0
    OPTIMIZER = 1
    GRADIENTS = 2
    PARAMETERS = 3


@dataclass
class ZeROConfig:
    """Configuration for ZeRO optimizer.
    
    Attributes:
        stage: ZeRO optimization stage (0-3).
        reduce_bucket_size: Bucket size for gradient reduction (bytes).
        allgather_bucket_size: Bucket size for AllGather operations (bytes).
        overlap_comm: Overlap communication with computation.
        contiguous_gradients: Use contiguous gradient buffer.
        cpu_offload: Offload optimizer states to CPU.
        cpu_offload_params: Offload parameters to CPU (ZeRO-3).
        cpu_offload_use_pin_memory: Use pinned memory for CPU offload.
        sub_group_size: Parameter sub-group size for partitioning.
        reduce_scatter: Use ReduceScatter instead of AllReduce.
        round_robin_gradients: Round-robin gradient partitioning.
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
    """Parameter partitioning wrapper for ZeRO-3.
    
    Manages the lifecycle of partitioned parameters, including splitting
    parameters across ranks and gathering them when needed for computation.
    
    Args:
        param: Original parameter tensor.
        rank: Current process rank.
        world_size: Total number of processes.
        device: Target device for tensors.
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
        
        self.numel = param.numel()
        self.partition_size = math.ceil(self.numel / world_size)
        self.start_idx = rank * self.partition_size
        self.end_idx = min(self.start_idx + self.partition_size, self.numel)
        self.local_numel = self.end_idx - self.start_idx
        
        self.local_data: Optional[torch.Tensor] = None
        self._is_partitioned = False
    
    def partition(self) -> torch.Tensor:
        """Partition parameter and return local shard."""
        if self._is_partitioned:
            return self.local_data
        
        flat_param = self.param.data.view(-1)
        self.local_data = flat_param[self.start_idx:self.end_idx].clone()
        self._is_partitioned = True
        
        return self.local_data
    
    def all_gather(self) -> torch.Tensor:
        """Gather all shards to reconstruct full parameter."""
        if not dist.is_initialized():
            return self.param.data
        
        gathered = [
            torch.zeros(self.partition_size, device=self.device)
            for _ in range(self.world_size)
        ]
        
        local_padded = torch.zeros(self.partition_size, device=self.device)
        local_padded[:self.local_numel] = self.local_data
        
        dist.all_gather(gathered, local_padded)
        
        full_param = torch.cat(gathered)[:self.numel]
        return full_param.view(self.param.shape)
    
    def reduce_scatter_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """ReduceScatter gradient and return local shard."""
        if not dist.is_initialized():
            return grad.view(-1)[self.start_idx:self.end_idx]
        
        flat_grad = grad.view(-1)
        
        padded_size = self.partition_size * self.world_size
        if flat_grad.numel() < padded_size:
            padded_grad = torch.zeros(padded_size, device=self.device)
            padded_grad[:flat_grad.numel()] = flat_grad
        else:
            padded_grad = flat_grad
        
        input_list = list(padded_grad.chunk(self.world_size))
        output = torch.zeros(self.partition_size, device=self.device)
        
        dist.reduce_scatter(output, input_list)
        
        return output[:self.local_numel]


class ZeROOptimizer(Optimizer):
    """ZeRO optimizer wrapper for memory-efficient distributed training.
    
    Wraps a standard PyTorch optimizer to enable ZeRO-style partitioning of
    optimizer states, gradients, and parameters across distributed processes.
    
    Mathematical Background:
        ZeRO partitions the optimizer state O, gradients G, and parameters P:
        - Stage 1: O is partitioned, each rank holds O/N
        - Stage 2: O and G are partitioned
        - Stage 3: O, G, and P are all partitioned
        
        Communication overhead increases with stage but memory savings are
        proportionally larger.
    
    Args:
        optimizer: Base PyTorch optimizer to wrap.
        config: ZeRO configuration.
        model: Model for parameter partitioning (required for ZeRO-3).
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
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = next(model.parameters()).device if model else torch.device("cpu")
        
        self.partitioned_params: Dict[int, PartitionedParameter] = {}
        self.param_to_partition: Dict[int, int] = {}
        self.grad_buffers: Dict[int, torch.Tensor] = {}
        
        if self.config.stage >= ZeROStage.OPTIMIZER:
            self._init_optimizer_partitioning()
        
        if self.config.stage >= ZeROStage.PARAMETERS and model is not None:
            self._init_parameter_partitioning()
    
    def _init_optimizer_partitioning(self) -> None:
        """Initialize optimizer state partitioning across ranks."""
        all_params = []
        for group in self.optimizer.param_groups:
            all_params.extend(group["params"])
        
        params_per_rank = math.ceil(len(all_params) / self.world_size)
        
        for i, param in enumerate(all_params):
            self.param_to_partition[id(param)] = i // params_per_rank
    
    def _init_parameter_partitioning(self) -> None:
        """Initialize parameter partitioning for ZeRO-3."""
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
        """Return optimizer parameter groups."""
        return self.optimizer.param_groups
    
    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value
    
    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dictionary."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dictionary."""
        self.optimizer.load_state_dict(state_dict)
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients of all parameters."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None) -> Optional[float]:
        """Perform optimization step with ZeRO synchronization."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if self.config.stage == ZeROStage.DISABLED:
            return self.optimizer.step()
        
        if self.config.stage >= ZeROStage.GRADIENTS:
            self._reduce_gradients()
        
        self.optimizer.step()
        
        if self.config.stage >= ZeROStage.PARAMETERS:
            self._sync_parameters()
        
        return loss
    
    def _reduce_gradients(self) -> None:
        """Reduce gradients across processes."""
        if not dist.is_initialized():
            return
        
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                if self.config.reduce_scatter:
                    if id(param) in self.partitioned_params:
                        pp = self.partitioned_params[id(param)]
                        local_grad = pp.reduce_scatter_grad(param.grad)
                        self.grad_buffers[id(param)] = local_grad
                else:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    
    def _sync_parameters(self) -> None:
        """Synchronize parameters across processes (ZeRO-3)."""
        if not dist.is_initialized():
            return
        
        for param_id, pp in self.partitioned_params.items():
            full_param = pp.all_gather()
            pp.param.data.copy_(full_param)
