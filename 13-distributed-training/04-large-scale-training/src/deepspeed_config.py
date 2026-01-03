"""
DeepSpeed 配置生成器

DeepSpeed 是微软开发的深度学习优化库，提供 ZeRO、混合精度、
梯度累积等功能，用于大规模模型训练。

核心功能:
    - ZeRO 优化器 (Stage 1/2/3)
    - 混合精度训练
    - 梯度累积和裁剪
    - 激活检查点
"""

import json
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union


class ZeROStage(IntEnum):
    """ZeRO 阶段"""
    DISABLED = 0
    OPTIMIZER_STATES = 1
    GRADIENTS = 2
    PARAMETERS = 3


@dataclass
class DeepSpeedConfig:
    """DeepSpeed 配置
    
    Attributes:
        train_batch_size: 全局批次大小
        train_micro_batch_size_per_gpu: 每 GPU 微批次大小
        gradient_accumulation_steps: 梯度累积步数
        gradient_clipping: 梯度裁剪值
        zero_stage: ZeRO 阶段
        fp16_enabled: 是否启用 FP16
        bf16_enabled: 是否启用 BF16
        offload_optimizer: 是否卸载优化器到 CPU
        offload_param: 是否卸载参数到 CPU
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
    """生成 ZeRO 配置
    
    Args:
        stage: ZeRO 阶段 (0-3)
        offload_optimizer: 卸载优化器到 CPU
        offload_param: 卸载参数到 CPU
        reduce_bucket_size: 归约桶大小
        allgather_bucket_size: AllGather 桶大小
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
    """生成优化器配置"""
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
    """生成学习率调度器配置"""
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
    """生成 FP16 配置"""
    return {
        "enabled": enabled,
        "loss_scale": loss_scale,
        "initial_scale_power": initial_scale_power,
        "loss_scale_window": loss_scale_window,
        "hysteresis": hysteresis,
        "min_loss_scale": min_loss_scale,
    }


def get_bf16_config(enabled: bool = True) -> Dict[str, Any]:
    """生成 BF16 配置"""
    return {"enabled": enabled}


def get_activation_checkpointing_config(
    partition_activations: bool = True,
    contiguous_memory_optimization: bool = True,
    cpu_checkpointing: bool = False,
) -> Dict[str, Any]:
    """生成激活检查点配置"""
    return {
        "partition_activations": partition_activations,
        "contiguous_memory_optimization": contiguous_memory_optimization,
        "cpu_checkpointing": cpu_checkpointing,
    }


def create_deepspeed_config(config: DeepSpeedConfig) -> Dict[str, Any]:
    """从 DeepSpeedConfig 创建完整配置字典
    
    Args:
        config: DeepSpeed 配置对象
        
    Returns:
        DeepSpeed JSON 配置字典
    """
    ds_config = {
        "train_batch_size": config.train_batch_size,
        "train_micro_batch_size_per_gpu": config.train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.gradient_clipping,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    
    # ZeRO 配置
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
    
    # 学习率调度器
    ds_config["scheduler"] = get_scheduler_config(
        warmup_steps=config.warmup_steps,
    )
    
    # 混合精度配置
    if config.bf16_enabled:
        ds_config["bf16"] = get_bf16_config(enabled=True)
        ds_config["fp16"] = {"enabled": False}
    elif config.fp16_enabled:
        ds_config["fp16"] = get_fp16_config(enabled=True)
        ds_config["bf16"] = {"enabled": False}
    
    # 激活检查点
    if config.activation_checkpointing:
        ds_config["activation_checkpointing"] = get_activation_checkpointing_config()
    
    return ds_config


def save_deepspeed_config(config: Dict[str, Any], path: str) -> None:
    """保存 DeepSpeed 配置到 JSON 文件"""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_deepspeed_config(path: str) -> Dict[str, Any]:
    """从 JSON 文件加载 DeepSpeed 配置"""
    with open(path, "r") as f:
        return json.load(f)
