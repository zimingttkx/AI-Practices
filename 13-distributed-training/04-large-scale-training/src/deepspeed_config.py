"""
DeepSpeed Configuration Generator

Core Idea:
    DeepSpeed is Microsoft's deep learning optimization library providing ZeRO
    optimizer, mixed precision, gradient accumulation, and activation checkpointing
    for efficient large-scale model training.

Mathematical Theory:
    ZeRO (Zero Redundancy Optimizer) partitions optimizer states, gradients,
    and parameters across data parallel ranks:
    
    .. math::
        \\text{Memory per GPU} = \\frac{\\Psi + \\nabla\\Psi + O_s}{N_d}
    
    where Psi is parameters, nabla Psi is gradients, O_s is optimizer states,
    and N_d is data parallel degree.

Problem Statement:
    Training large models requires memory optimization beyond standard data
    parallelism. DeepSpeed's ZeRO eliminates memory redundancy while maintaining
    data parallel efficiency.

Comparison:
    - ZeRO-1: Partition optimizer states only
    - ZeRO-2: Partition optimizer states + gradients
    - ZeRO-3: Partition all (states + gradients + parameters)
    - ZeRO-Offload: CPU/NVMe offloading for memory extension

Complexity:
    - Communication: O(Psi) per step (same as DDP for ZeRO-1/2)
    - Memory: O(Psi/N) for ZeRO-3 vs O(Psi) for DDP
    - Compute: Minimal overhead from partitioning

References:
    - Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion
      Parameter Models", SC 2020
    - DeepSpeed documentation: https://www.deepspeed.ai/
"""

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional


class ZeROStage(IntEnum):
    """ZeRO optimization stages."""
    DISABLED = 0
    OPTIMIZER_STATES = 1
    GRADIENTS = 2
    PARAMETERS = 3


@dataclass
class DeepSpeedConfig:
    """Configuration for DeepSpeed training.
    
    Attributes:
        train_batch_size: Global batch size across all GPUs.
        train_micro_batch_size_per_gpu: Micro-batch size per GPU.
        gradient_accumulation_steps: Steps to accumulate before update.
        gradient_clipping: Maximum gradient norm.
        zero_stage: ZeRO optimization stage (0-3).
        fp16_enabled: Enable FP16 mixed precision.
        bf16_enabled: Enable BF16 mixed precision.
        offload_optimizer: Offload optimizer states to CPU.
        offload_param: Offload parameters to CPU (ZeRO-3).
        optimizer_type: Optimizer type (AdamW, Adam, etc.).
        learning_rate: Learning rate.
        weight_decay: Weight decay coefficient.
        warmup_steps: Learning rate warmup steps.
        activation_checkpointing: Enable activation checkpointing.
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


def get_zero_config(
    stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    reduce_bucket_size: int = 500_000_000,
    allgather_bucket_size: int = 500_000_000,
) -> Dict[str, Any]:
    """Generate ZeRO optimization configuration.
    
    Args:
        stage: ZeRO stage (0-3).
        offload_optimizer: Offload optimizer states to CPU.
        offload_param: Offload parameters to CPU.
        reduce_bucket_size: Bucket size for gradient reduction.
        allgather_bucket_size: Bucket size for AllGather operations.
        
    Returns:
        ZeRO configuration dictionary.
    """
    config = {
        "stage": stage,
        "reduce_bucket_size": reduce_bucket_size,
        "allgather_bucket_size": allgather_bucket_size,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "contiguous_gradients": True,
        "overlap_comm": True,
    }
    
    if offload_optimizer:
        config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    
    if offload_param:
        config["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    
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
    """Generate optimizer configuration."""
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
    """Generate learning rate scheduler configuration."""
    return {
        "type": scheduler_type,
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": "auto",
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
    """Generate FP16 mixed precision configuration."""
    return {
        "enabled": enabled,
        "loss_scale": loss_scale,
        "initial_scale_power": initial_scale_power,
        "loss_scale_window": loss_scale_window,
        "hysteresis": hysteresis,
        "min_loss_scale": min_loss_scale,
    }


def get_bf16_config(enabled: bool = True) -> Dict[str, Any]:
    """Generate BF16 mixed precision configuration."""
    return {"enabled": enabled}


def get_activation_checkpointing_config(
    partition_activations: bool = True,
    contiguous_memory_optimization: bool = True,
    cpu_checkpointing: bool = False,
) -> Dict[str, Any]:
    """Generate activation checkpointing configuration."""
    return {
        "partition_activations": partition_activations,
        "contiguous_memory_optimization": contiguous_memory_optimization,
        "cpu_checkpointing": cpu_checkpointing,
    }


def create_deepspeed_config(config: DeepSpeedConfig) -> Dict[str, Any]:
    """Create complete DeepSpeed configuration dictionary.
    
    Args:
        config: DeepSpeed configuration object.
        
    Returns:
        Complete DeepSpeed JSON configuration dictionary.
    """
    ds_config = {
        "train_batch_size": config.train_batch_size,
        "train_micro_batch_size_per_gpu": config.train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.gradient_clipping,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    
    ds_config["zero_optimization"] = get_zero_config(
        stage=config.zero_stage,
        offload_optimizer=config.offload_optimizer,
        offload_param=config.offload_param,
    )
    
    ds_config["optimizer"] = get_optimizer_config(
        optimizer_type=config.optimizer_type,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    ds_config["scheduler"] = get_scheduler_config(
        warmup_steps=config.warmup_steps,
    )
    
    if config.bf16_enabled:
        ds_config["bf16"] = get_bf16_config(enabled=True)
        ds_config["fp16"] = {"enabled": False}
    elif config.fp16_enabled:
        ds_config["fp16"] = get_fp16_config(enabled=True)
        ds_config["bf16"] = {"enabled": False}
    
    if config.activation_checkpointing:
        ds_config["activation_checkpointing"] = get_activation_checkpointing_config()
    
    return ds_config


def save_deepspeed_config(config: Dict[str, Any], path: str) -> None:
    """Save DeepSpeed configuration to JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_deepspeed_config(path: str) -> Dict[str, Any]:
    """Load DeepSpeed configuration from JSON file."""
    with open(path, "r") as f:
        return json.load(f)
