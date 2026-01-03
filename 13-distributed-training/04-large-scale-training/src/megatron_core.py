"""
Megatron-Core Integration

Core Idea:
    Megatron-LM is NVIDIA's framework for training large Transformer models,
    providing efficient tensor, pipeline, and sequence parallelism with
    optimized communication patterns.

Mathematical Theory:
    3D parallelism combines data (D), tensor (T), and pipeline (P) parallelism:
    
    .. math::
        N_{\\text{total}} = D \\times T \\times P
    
    Each GPU is identified by a unique (d, t, p) coordinate in the 3D mesh.
    Communication groups are formed along each dimension.

Problem Statement:
    Training trillion-parameter models requires distributing computation across
    thousands of GPUs. Megatron provides the infrastructure for managing
    complex parallel topologies and communication patterns.

Comparison:
    - vs PyTorch DDP: Adds tensor and pipeline parallelism
    - vs DeepSpeed: More focus on model parallelism, less on memory optimization
    - vs FairScale: Native NVIDIA optimization, better Tensor Core utilization

Complexity:
    - Communication: O(B*H/T) for tensor parallel, O(B*H) for pipeline
    - Memory: O(P/T) per GPU for model parameters
    - Initialization: O(N) for creating process groups

References:
    - Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language
      Models Using Model Parallelism", arXiv 2019
    - Narayanan et al., "Efficient Large-Scale Language Model Training on GPU
      Clusters Using Megatron-LM", SC 2021
"""

from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist


@dataclass
class MegatronConfig:
    """Configuration for Megatron parallelism.
    
    Attributes:
        tensor_model_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        data_parallel_size: Data parallelism degree.
        sequence_parallel: Enable sequence parallelism.
        virtual_pipeline_model_parallel_size: Virtual pipeline stages.
        context_parallel_size: Context parallelism degree.
    """
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    data_parallel_size: int = 1
    sequence_parallel: bool = False
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1


class MegatronParallelState:
    """Singleton managing Megatron parallel state and process groups.
    
    Manages tensor, pipeline, and data parallel groups along with
    rank information for each parallelism dimension.
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
        """Initialize parallel state with given configuration."""
        if not dist.is_initialized():
            self._initialized = True
            return
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        tp_size = config.tensor_model_parallel_size
        pp_size = config.pipeline_model_parallel_size
        dp_size = world_size // (tp_size * pp_size)
        
        assert world_size == tp_size * pp_size * dp_size
        
        self._create_tensor_parallel_groups(rank, world_size, tp_size, pp_size, dp_size)
        self._create_pipeline_parallel_groups(rank, world_size, tp_size, pp_size, dp_size)
        self._create_data_parallel_groups(rank, world_size, tp_size, pp_size, dp_size)
        
        self._tensor_model_parallel_world_size = tp_size
        self._pipeline_model_parallel_world_size = pp_size
        self._data_parallel_world_size = dp_size
        
        self._initialized = True
    
    def _create_tensor_parallel_groups(
        self, rank: int, world_size: int, tp_size: int, pp_size: int, dp_size: int
    ) -> None:
        """Create tensor model parallel groups."""
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
        """Create pipeline model parallel groups."""
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
        """Create data parallel groups."""
        for i in range(pp_size):
            for j in range(tp_size):
                ranks = [i * tp_size + j + k * tp_size * pp_size for k in range(dp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self._data_parallel_group = group
                    self._data_parallel_rank = ranks.index(rank)
    
    @property
    def tensor_model_parallel_group(self):
        """Get tensor model parallel process group."""
        return self._tensor_model_parallel_group
    
    @property
    def pipeline_model_parallel_group(self):
        """Get pipeline model parallel process group."""
        return self._pipeline_model_parallel_group
    
    @property
    def data_parallel_group(self):
        """Get data parallel process group."""
        return self._data_parallel_group
    
    @property
    def tensor_model_parallel_rank(self) -> int:
        """Get rank within tensor model parallel group."""
        return self._tensor_model_parallel_rank
    
    @property
    def pipeline_model_parallel_rank(self) -> int:
        """Get rank within pipeline model parallel group."""
        return self._pipeline_model_parallel_rank
    
    @property
    def data_parallel_rank(self) -> int:
        """Get rank within data parallel group."""
        return self._data_parallel_rank
    
    @property
    def tensor_model_parallel_world_size(self) -> int:
        """Get tensor model parallel world size."""
        return self._tensor_model_parallel_world_size
    
    @property
    def pipeline_model_parallel_world_size(self) -> int:
        """Get pipeline model parallel world size."""
        return self._pipeline_model_parallel_world_size
    
    @property
    def data_parallel_world_size(self) -> int:
        """Get data parallel world size."""
        return self._data_parallel_world_size
    
    def is_pipeline_first_stage(self) -> bool:
        """Check if current rank is first pipeline stage."""
        return self._pipeline_model_parallel_rank == 0
    
    def is_pipeline_last_stage(self) -> bool:
        """Check if current rank is last pipeline stage."""
        return self._pipeline_model_parallel_rank == self._pipeline_model_parallel_world_size - 1


def initialize_megatron(config: MegatronConfig) -> MegatronParallelState:
    """Initialize Megatron parallel environment."""
    state = MegatronParallelState()
    state.initialize(config)
    return state


def get_model_parallel_group():
    """Get tensor model parallel process group."""
    state = MegatronParallelState()
    return state.tensor_model_parallel_group


def get_data_parallel_group():
    """Get data parallel process group."""
    state = MegatronParallelState()
    return state.data_parallel_group


def get_pipeline_parallel_group():
    """Get pipeline model parallel process group."""
    state = MegatronParallelState()
    return state.pipeline_model_parallel_group
