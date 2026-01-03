"""
大规模训练模块

包含 DeepSpeed 配置、Megatron-Core 集成和分布式检查点工具。
"""

from .deepspeed_config import (
    DeepSpeedConfig,
    create_deepspeed_config,
    get_zero_config,
    get_optimizer_config,
)

from .megatron_core import (
    MegatronConfig,
    MegatronParallelState,
    initialize_megatron,
    get_model_parallel_group,
)

from .checkpoint_utils import (
    CheckpointConfig,
    DistributedCheckpointer,
    save_distributed_checkpoint,
    load_distributed_checkpoint,
)

__all__ = [
    # DeepSpeed
    "DeepSpeedConfig",
    "create_deepspeed_config",
    "get_zero_config",
    "get_optimizer_config",
    # Megatron
    "MegatronConfig",
    "MegatronParallelState",
    "initialize_megatron",
    "get_model_parallel_group",
    # Checkpoint
    "CheckpointConfig",
    "DistributedCheckpointer",
    "save_distributed_checkpoint",
    "load_distributed_checkpoint",
]
