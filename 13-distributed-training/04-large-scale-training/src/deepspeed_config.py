"""
DeepSpeed Configuration Generator DeepSpeed配置生成器

================================================================================
核心思想 (一句话理解)
================================================================================
DeepSpeed = ZeRO内存优化 + 混合精度 + CPU卸载 + 梯度累积 = 用更少显存训练更大模型

================================================================================
什么是DeepSpeed？
================================================================================

    DeepSpeed是微软开发的深度学习优化库，核心功能:
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. ZeRO优化器: 分片消除内存冗余                                  │
    │  2. 混合精度: FP16/BF16加速                                      │
    │  3. CPU Offload: 将状态卸载到CPU内存                             │
    │  4. 梯度累积: 模拟大batch训练                                    │
    │  5. 激活检查点: 用计算换内存                                      │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
ZeRO优化阶段详解
================================================================================

    传统数据并行的内存占用 (以1B参数模型为例):
    ┌─────────────────────────────────────────────────────────────────┐
    │  每个GPU都存储:                                                  │
    │  - 模型参数 (FP16): 2GB                                         │
    │  - 梯度 (FP16): 2GB                                             │
    │  - 优化器状态 (FP32):                                            │
    │    - Adam动量: 4GB                                               │
    │    - Adam方差: 4GB                                               │
    │    - FP32参数副本: 4GB                                           │
    │  总计: 16GB/GPU (N个GPU总共N×16GB，大量冗余!)                    │
    └─────────────────────────────────────────────────────────────────┘

    ZeRO各阶段的内存优化:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Stage 0 (禁用): 与DDP相同，每GPU 16GB                          │
    │                                                                 │
    │  Stage 1 (分片优化器状态):                                       │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  每GPU: 参数(2GB) + 梯度(2GB) + 优化器状态(12GB/N)       │   │
    │  │  8个GPU: 4 + 12/8 = 5.5GB/GPU                           │   │
    │  │  通信量: 与DDP相同                                       │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  Stage 2 (分片优化器状态 + 梯度):                                │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  每GPU: 参数(2GB) + 梯度(2GB/N) + 优化器状态(12GB/N)     │   │
    │  │  8个GPU: 2 + 14/8 = 3.75GB/GPU                          │   │
    │  │  通信量: 与DDP相同 (推荐!)                                │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  Stage 3 (全分片):                                               │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  每GPU: 参数(2GB/N) + 梯度(2GB/N) + 优化器状态(12GB/N)   │   │
    │  │  8个GPU: 16/8 = 2GB/GPU (线性扩展!)                      │   │
    │  │  通信量: 增加 (需要AllGather参数)                         │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
CPU Offload (ZeRO-Offload)
================================================================================
    当GPU显存仍然不足时，可以将状态卸载到CPU:
    - offload_optimizer: 优化器状态存储在CPU内存
    - offload_param: 参数也存储在CPU内存 (Stage 3)

    代价: CPU-GPU数据传输会降低训练速度

================================================================================
前置知识
================================================================================
- 数据并行的基本概念
- Adam优化器的内存占用
- 混合精度训练

================================================================================
参考文献
================================================================================
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion
  Parameter Models", SC 2020
- DeepSpeed documentation: https://www.deepspeed.ai/
"""

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional


# =============================================================================
# ZeRO阶段枚举
# =============================================================================

class ZeROStage(IntEnum):
    """ZeRO优化阶段

    DISABLED (0): 禁用ZeRO，与普通DDP相同
    OPTIMIZER_STATES (1): 只分片优化器状态
    GRADIENTS (2): 分片优化器状态 + 梯度 (推荐)
    PARAMETERS (3): 全分片 (优化器状态 + 梯度 + 参数)
    """
    DISABLED = 0
    OPTIMIZER_STATES = 1
    GRADIENTS = 2
    PARAMETERS = 3


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class DeepSpeedConfig:
    """DeepSpeed训练配置

    Attributes:
        train_batch_size: 全局batch大小 (所有GPU的总和)
        train_micro_batch_size_per_gpu: 每个GPU的微批次大小
        gradient_accumulation_steps: 梯度累积步数
        gradient_clipping: 梯度裁剪的最大范数
        zero_stage: ZeRO优化阶段 (0-3)
        fp16_enabled: 是否启用FP16混合精度
        bf16_enabled: 是否启用BF16混合精度
        offload_optimizer: 是否将优化器状态卸载到CPU
        offload_param: 是否将参数卸载到CPU (仅Stage 3)
        optimizer_type: 优化器类型 (AdamW, Adam等)
        learning_rate: 学习率
        weight_decay: 权重衰减系数
        warmup_steps: 学习率预热步数
        activation_checkpointing: 是否启用激活检查点

    batch大小关系:
        train_batch_size = train_micro_batch_size_per_gpu
                         × gradient_accumulation_steps
                         × world_size

    Example:
        >>> config = DeepSpeedConfig(
        ...     train_batch_size=256,
        ...     train_micro_batch_size_per_gpu=4,
        ...     gradient_accumulation_steps=8,
        ...     zero_stage=2,
        ... )
    """
    train_batch_size: int = 32
    train_micro_batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    zero_stage: int = 2
    fp16_enabled: bool = True
    bf16_enabled: bool = False
    offload_optimizer: bool = False
    offload_param: bool = False
    optimizer_type: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    activation_checkpointing: bool = False


# =============================================================================
# 配置生成函数
# =============================================================================

def get_zero_config(
    stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    reduce_bucket_size: int = 500_000_000,
    allgather_bucket_size: int = 500_000_000,
) -> Dict[str, Any]:
    """生成ZeRO优化配置

    Args:
        stage: ZeRO阶段 (0-3)
        offload_optimizer: 是否卸载优化器状态到CPU
        offload_param: 是否卸载参数到CPU
        reduce_bucket_size: 梯度归约的桶大小 (字节)
        allgather_bucket_size: AllGather操作的桶大小 (字节)

    Returns:
        ZeRO配置字典

    桶大小说明:
        - 较大的桶: 更高的通信效率，但更大的内存占用
        - 较小的桶: 更低的内存占用，但通信效率降低
        - 默认500MB是个好的平衡点
    """
    config = {
        "stage": stage,
        "reduce_bucket_size": reduce_bucket_size,
        "allgather_bucket_size": allgather_bucket_size,
        "allgather_partitions": True,      # 分区AllGather
        "reduce_scatter": True,            # 使用ReduceScatter
        "contiguous_gradients": True,      # 连续梯度内存
        "overlap_comm": True,              # 通信与计算重叠
    }

    # CPU卸载配置
    if offload_optimizer:
        config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,  # 使用锁页内存加速传输
        }

    if offload_param:
        config["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    # Stage 3特有配置
    if stage == 3:
        config["stage3_prefetch_bucket_size"] = reduce_bucket_size
        config["stage3_param_persistence_threshold"] = 100_000
        config["stage3_max_live_parameters"] = 1_000_000_000
        config["stage3_max_reuse_distance"] = 1_000_000_000

    return config


def get_optimizer_config(
    optimizer_type: str = "AdamW",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """生成优化器配置

    Args:
        optimizer_type: 优化器类型 (AdamW, Adam, SGD等)
        learning_rate: 学习率
        weight_decay: 权重衰减
        betas: Adam的beta参数
        eps: 数值稳定性的小常数

    Returns:
        优化器配置字典
    """
    return {
        "type": optimizer_type,
        "params": {
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "betas": list(betas),
            "eps": eps,
        }
    }


def get_scheduler_config(
    scheduler_type: str = "WarmupDecayLR",
    warmup_steps: int = 1000,
    total_steps: int = 100000,
) -> Dict[str, Any]:
    """生成学习率调度器配置

    Args:
        scheduler_type: 调度器类型
        warmup_steps: 预热步数
        total_steps: 总训练步数

    Returns:
        调度器配置字典
    """
    return {
        "type": scheduler_type,
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",  # 自动使用优化器的学习率
            "warmup_num_steps": warmup_steps,
            "total_num_steps": total_steps,
        }
    }


def get_fp16_config(
    enabled: bool = True,
    loss_scale: float = 0,
    initial_scale_power: int = 16,
    loss_scale_window: int = 1000,
    hysteresis: int = 2,
    min_loss_scale: float = 1,
) -> Dict[str, Any]:
    """生成FP16混合精度配置

    Args:
        enabled: 是否启用FP16
        loss_scale: 静态损失缩放 (0表示动态缩放)
        initial_scale_power: 初始缩放因子的指数 (2^16 = 65536)
        loss_scale_window: 动态缩放的窗口大小
        hysteresis: 缩放因子调整的滞后步数
        min_loss_scale: 最小缩放因子

    Returns:
        FP16配置字典
    """
    return {
        "enabled": enabled,
        "loss_scale": loss_scale,
        "initial_scale_power": initial_scale_power,
        "loss_scale_window": loss_scale_window,
        "hysteresis": hysteresis,
        "min_loss_scale": min_loss_scale,
    }


def get_bf16_config(enabled: bool = True) -> Dict[str, Any]:
    """生成BF16混合精度配置

    BF16不需要损失缩放，配置更简单。

    Args:
        enabled: 是否启用BF16

    Returns:
        BF16配置字典
    """
    return {"enabled": enabled}


def get_activation_checkpointing_config(
    partition_activations: bool = True,
    contiguous_memory_optimization: bool = True,
    cpu_checkpointing: bool = False,
) -> Dict[str, Any]:
    """生成激活检查点配置

    激活检查点通过重新计算来节省内存。

    Args:
        partition_activations: 是否分区激活值
        contiguous_memory_optimization: 是否优化内存连续性
        cpu_checkpointing: 是否将激活值检查点到CPU

    Returns:
        激活检查点配置字典
    """
    return {
        "partition_activations": partition_activations,
        "contiguous_memory_optimization": contiguous_memory_optimization,
        "cpu_checkpointing": cpu_checkpointing,
    }


def create_deepspeed_config(config: DeepSpeedConfig) -> Dict[str, Any]:
    """创建完整的DeepSpeed配置字典

    将DeepSpeedConfig对象转换为DeepSpeed可用的JSON配置。

    Args:
        config: DeepSpeed配置对象

    Returns:
        完整的DeepSpeed JSON配置字典

    Example:
        >>> config = DeepSpeedConfig(zero_stage=2, fp16_enabled=True)
        >>> ds_config = create_deepspeed_config(config)
        >>> # 可以直接传给deepspeed.initialize()
    """
    ds_config = {
        "train_batch_size": config.train_batch_size,
        "train_micro_batch_size_per_gpu": config.train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.gradient_clipping,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    # ZeRO配置
    ds_config["zero_optimization"] = get_zero_config(
        stage=config.zero_stage,
        offload_optimizer=config.offload_optimizer,
        offload_param=config.offload_param,
    )

    # 优化器配置
    ds_config["optimizer"] = get_optimizer_config(
        optimizer_type=config.optimizer_type,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 学习率调度器配置
    ds_config["scheduler"] = get_scheduler_config(
        warmup_steps=config.warmup_steps,
    )

    # 混合精度配置 (BF16和FP16互斥)
    if config.bf16_enabled:
        ds_config["bf16"] = get_bf16_config(enabled=True)
        ds_config["fp16"] = {"enabled": False}
    elif config.fp16_enabled:
        ds_config["fp16"] = get_fp16_config(enabled=True)
        ds_config["bf16"] = {"enabled": False}

    # 激活检查点配置
    if config.activation_checkpointing:
        ds_config["activation_checkpointing"] = get_activation_checkpointing_config()

    return ds_config


# =============================================================================
# 配置文件I/O
# =============================================================================

def save_deepspeed_config(config: Dict[str, Any], path: str) -> None:
    """保存DeepSpeed配置到JSON文件

    Args:
        config: DeepSpeed配置字典
        path: 保存路径

    Example:
        >>> ds_config = create_deepspeed_config(DeepSpeedConfig())
        >>> save_deepspeed_config(ds_config, "ds_config.json")
    """
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_deepspeed_config(path: str) -> Dict[str, Any]:
    """从JSON文件加载DeepSpeed配置

    Args:
        path: 配置文件路径

    Returns:
        DeepSpeed配置字典

    Example:
        >>> ds_config = load_deepspeed_config("ds_config.json")
    """
    with open(path, "r") as f:
        return json.load(f)
