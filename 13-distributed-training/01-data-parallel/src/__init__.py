"""
数据并行模块

包含 DDP、FSDP、ZeRO 等分布式数据并行实现。
"""

from .ddp import (
    DDPConfig,
    DDPTrainer,
    setup_ddp,
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_main_process,
)

from .fsdp import (
    FSDPConfig,
    FSDPTrainer,
    get_fsdp_wrap_policy,
    ShardingStrategy,
)

from .zero import (
    ZeROConfig,
    ZeROOptimizer,
    ZeROStage,
)

__all__ = [
    # DDP
    "DDPConfig",
    "DDPTrainer",
    "setup_ddp",
    "cleanup_ddp",
    "get_rank",
    "get_world_size",
    "is_main_process",
    # FSDP
    "FSDPConfig",
    "FSDPTrainer",
    "get_fsdp_wrap_policy",
    "ShardingStrategy",
    # ZeRO
    "ZeROConfig",
    "ZeROOptimizer",
    "ZeROStage",
]
