"""
模型并行模块

包含张量并行、流水线并行和序列并行实现。
"""

from .tensor_parallel import (
    TensorParallelConfig,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    tensor_parallel_split,
    tensor_parallel_gather,
)

from .pipeline_parallel import (
    PipelineConfig,
    PipelineStage,
    PipelineScheduler,
    GPipeScheduler,
    PipeDreamScheduler,
)

from .sequence_parallel import (
    SequenceParallelConfig,
    SequenceParallelAttention,
    scatter_to_sequence_parallel,
    gather_from_sequence_parallel,
)

__all__ = [
    # Tensor Parallel
    "TensorParallelConfig",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "tensor_parallel_split",
    "tensor_parallel_gather",
    # Pipeline Parallel
    "PipelineConfig",
    "PipelineStage",
    "PipelineScheduler",
    "GPipeScheduler",
    "PipeDreamScheduler",
    # Sequence Parallel
    "SequenceParallelConfig",
    "SequenceParallelAttention",
    "scatter_to_sequence_parallel",
    "gather_from_sequence_parallel",
]
