"""
Megatron-Core 集成

Megatron-LM 是 NVIDIA 开发的大规模 Transformer 训练框架，
提供高效的模型并行和数据并行实现。

核心功能:
    - 张量并行 (Tensor Parallelism)
    - 流水线并行 (Pipeline Parallelism)
    - 序列并行 (Sequence Parallelism)
    - 分布式优化器
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist


@dataclass
class MegatronConfig:
    """Megatron 配置
    
    Attributes:
        tensor_model_parallel_size: 张量并行大小
        pipeline_model_parallel_size: 流水线并行大小
        data_parallel_size: 数据并行大小
        sequence_parallel: 是否启用序列并行
        virtual_pipeline_model_parallel_size: 虚拟流水线大小
        context_parallel_size: 上下文并行大小
    """
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    data_parallel_size: int = 1
    sequence_parallel: bool = False
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1


class MegatronParallelState:
    """Megatron 并行状态管理
    
    管理各种并行组和排名信息。
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._tensor_model_parallel_group = None
        self._pipeline_model_parallel_group = None
        self._data_parallel_group = None
        self._model_parallel_group = None
        
        self._tensor_model_parallel_rank = 0
        self._pipeline_model_parallel_rank = 0
        self._data_parallel_rank = 0
        
        self._tensor_model_parallel_world_size = 1
        self._pipeline_model_parallel_world_size = 1
        self._data_parallel_world_size = 1
        
        self._initialized = False
    
    def initialize(self, config: MegatronConfig) -> None:
        """初始化并行状态"""
        if not dist.is_initialized():
            self._initialized = True
            return
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        tp_size = config.tensor_model_parallel_size
        pp_size = config.pipeline_model_parallel_size
        dp_size = world_size // (tp_size * pp_size)
        
        assert world_size == tp_size * pp_size * dp_size
        
        # 创建张量并行组
        self._create_tensor_parallel_groups(rank, world_size, tp_size, pp_size, dp_size)
        
        # 创建流水线并行组
        self._create_pipeline_parallel_groups(rank, world_size, tp_size, pp_size, dp_size)
        
        # 创建数据并行组
        self._create_data_parallel_groups(rank, world_size, tp_size, pp_size, dp_size)
        
        self._tensor_model_parallel_world_size = tp_size
        self._pipeline_model_parallel_world_size = pp_size
        self._data_parallel_world_size = dp_size
        
        self._initialized = True
    
    def _create_tensor_parallel_groups(
        self, rank: int, world_size: int, tp_size: int, pp_size: int, dp_size: int
    ) -> None:
        """创建张量并行组"""
        num_tp_groups = world_size // tp_size
        
        for i in range(num_tp_groups):
            ranks = list(range(i * tp_size, (i + 1) * tp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                self._tensor_model_parallel_group = group
                self._tensor_model_parallel_rank = ranks.index(rank)
    
    def _create_pipeline_parallel_groups(
        self, rank: int, world_size: int, tp_size: int, pp_size: int, dp_size: int
    ) -> None:
        """创建流水线并行组"""
        num_pp_groups = world_size // pp_size
        
        for i in range(dp_size):
            for j in range(tp_size):
                ranks = [i * tp_size * pp_size + j + k * tp_size for k in range(pp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self._pipeline_model_parallel_group = group
                    self._pipeline_model_parallel_rank = ranks.index(rank)
    
    def _create_data_parallel_groups(
        self, rank: int, world_size: int, tp_size: int, pp_size: int, dp_size: int
    ) -> None:
        """创建数据并行组"""
        for i in range(pp_size):
            for j in range(tp_size):
                ranks = [i * tp_size + j + k * tp_size * pp_size for k in range(dp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self._data_parallel_group = group
                    self._data_parallel_rank = ranks.index(rank)
    
    @property
    def tensor_model_parallel_group(self):
        return self._tensor_model_parallel_group
    
    @property
    def pipeline_model_parallel_group(self):
        return self._pipeline_model_parallel_group
    
    @property
    def data_parallel_group(self):
        return self._data_parallel_group
    
    @property
    def tensor_model_parallel_rank(self) -> int:
        return self._tensor_model_parallel_rank
    
    @property
    def pipeline_model_parallel_rank(self) -> int:
        return self._pipeline_model_parallel_rank
    
    @property
    def data_parallel_rank(self) -> int:
        return self._data_parallel_rank
    
    @property
    def tensor_model_parallel_world_size(self) -> int:
        return self._tensor_model_parallel_world_size
    
    @property
    def pipeline_model_parallel_world_size(self) -> int:
        return self._pipeline_model_parallel_world_size
    
    @property
    def data_parallel_world_size(self) -> int:
        return self._data_parallel_world_size
    
    def is_pipeline_first_stage(self) -> bool:
        return self._pipeline_model_parallel_rank == 0
    
    def is_pipeline_last_stage(self) -> bool:
        return self._pipeline_model_parallel_rank == self._pipeline_model_parallel_world_size - 1


def initialize_megatron(config: MegatronConfig) -> MegatronParallelState:
    """初始化 Megatron 并行环境"""
    state = MegatronParallelState()
    state.initialize(config)
    return state


def get_model_parallel_group():
    """获取模型并行组"""
    state = MegatronParallelState()
    return state.tensor_model_parallel_group


def get_data_parallel_group():
    """获取数据并行组"""
    state = MegatronParallelState()
    return state.data_parallel_group


def get_pipeline_parallel_group():
    """获取流水线并行组"""
    state = MegatronParallelState()
    return state.pipeline_model_parallel_group
