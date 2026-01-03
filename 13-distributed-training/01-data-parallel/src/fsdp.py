"""
Fully Sharded Data Parallel (FSDP) 实现

FSDP 是 PyTorch 的全分片数据并行实现，通过将模型参数、梯度和优化器状态
分片到多个 GPU 上，大幅减少每个 GPU 的内存占用。

核心概念:
    - 参数分片: 每个 GPU 只存储部分参数
    - 按需聚合: 前向/反向传播时临时聚合完整参数
    - 梯度分片: 反向传播后立即分片梯度
    - 优化器状态分片: 每个 GPU 只维护部分优化器状态
"""

import functools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy as TorchShardingStrategy,
        CPUOffload,
        StateDictType,
        FullStateDictConfig,
        ShardedStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
        enable_wrap,
        wrap,
    )
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


class ShardingStrategy(Enum):
    """分片策略枚举"""
    FULL_SHARD = auto()      # 完全分片 (ZeRO-3)
    SHARD_GRAD_OP = auto()   # 梯度和优化器状态分片 (ZeRO-2)
    NO_SHARD = auto()        # 不分片 (DDP)
    HYBRID_SHARD = auto()    # 混合分片


@dataclass
class FSDPConfig:
    """FSDP 配置类
    
    Attributes:
        sharding_strategy: 分片策略
        cpu_offload: 是否将参数卸载到 CPU
        backward_prefetch: 反向传播预取策略
        mixed_precision: 混合精度配置
        auto_wrap_policy: 自动包装策略
        min_num_params: 自动包装的最小参数数量
        transformer_layer_cls: Transformer 层类（用于自动包装）
        use_orig_params: 是否使用原始参数
        sync_module_states: 是否同步模块状态
        forward_prefetch: 是否启用前向预取
        limit_all_gathers: 是否限制 AllGather 操作
        activation_checkpointing: 是否启用激活检查点
    """
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    cpu_offload: bool = False
    backward_prefetch: str = "backward_pre"  # "backward_pre", "backward_post", None
    mixed_precision: bool = False
    mixed_precision_dtype: torch.dtype = torch.float16
    auto_wrap_policy: str = "size_based"  # "size_based", "transformer", "none"
    min_num_params: int = 100_000
    transformer_layer_cls: Optional[List[Type[nn.Module]]] = None
    use_orig_params: bool = True
    sync_module_states: bool = True
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    activation_checkpointing: bool = False
    activation_checkpointing_layers: Optional[List[Type[nn.Module]]] = None


def _get_sharding_strategy(strategy: ShardingStrategy):
    """转换分片策略为 PyTorch FSDP 策略"""
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP not available in this PyTorch version")
    
    mapping = {
        ShardingStrategy.FULL_SHARD: TorchShardingStrategy.FULL_SHARD,
        ShardingStrategy.SHARD_GRAD_OP: TorchShardingStrategy.SHARD_GRAD_OP,
        ShardingStrategy.NO_SHARD: TorchShardingStrategy.NO_SHARD,
        ShardingStrategy.HYBRID_SHARD: TorchShardingStrategy.HYBRID_SHARD,
    }
    return mapping[strategy]


def _get_backward_prefetch(prefetch: Optional[str]):
    """转换反向预取策略"""
    if not FSDP_AVAILABLE or prefetch is None:
        return None
    
    mapping = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    return mapping.get(prefetch)


def get_fsdp_wrap_policy(
    config: FSDPConfig,
) -> Optional[Callable]:
    """获取 FSDP 自动包装策略
    
    Args:
        config: FSDP 配置
        
    Returns:
        包装策略函数
    """
    if not FSDP_AVAILABLE:
        return None
    
    if config.auto_wrap_policy == "none":
        return None
    
    if config.auto_wrap_policy == "size_based":
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=config.min_num_params,
        )
    
    if config.auto_wrap_policy == "transformer":
        if config.transformer_layer_cls is None:
            raise ValueError(
                "transformer_layer_cls must be specified for transformer wrap policy"
            )
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(config.transformer_layer_cls),
        )
    
    raise ValueError(f"Unknown auto_wrap_policy: {config.auto_wrap_policy}")


def get_fsdp_mixed_precision(config: FSDPConfig) -> Optional[Any]:
    """获取 FSDP 混合精度配置"""
    if not FSDP_AVAILABLE or not config.mixed_precision:
        return None
    
    return MixedPrecision(
        param_dtype=config.mixed_precision_dtype,
        reduce_dtype=config.mixed_precision_dtype,
        buffer_dtype=config.mixed_precision_dtype,
    )


class FSDPTrainer:
    """FSDP 训练器
    
    封装了 FSDP 训练的常用操作。
    
    Args:
        model: PyTorch 模型
        config: FSDP 配置
        device: 训练设备
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[FSDPConfig] = None,
        device: Optional[torch.device] = None,
    ):
        if not FSDP_AVAILABLE:
            raise RuntimeError(
                "FSDP not available. Please upgrade to PyTorch 1.12+"
            )
        
        self.config = config or FSDPConfig()
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{self.local_rank}")
            else:
                device = torch.device("cpu")
        self.device = device
        
        self.model = model
        self.fsdp_model = None
        self._is_wrapped = False
    
    def wrap_model(self) -> FSDP:
        """将模型包装为 FSDP 模型"""
        if self._is_wrapped:
            return self.fsdp_model
        
        # 应用激活检查点
        if self.config.activation_checkpointing:
            self._apply_activation_checkpointing()
        
        # 获取配置
        sharding_strategy = _get_sharding_strategy(self.config.sharding_strategy)
        backward_prefetch = _get_backward_prefetch(self.config.backward_prefetch)
        mixed_precision = get_fsdp_mixed_precision(self.config)
        auto_wrap_policy = get_fsdp_wrap_policy(self.config)
        
        # CPU 卸载配置
        cpu_offload = None
        if self.config.cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        
        # 包装模型
        self.fsdp_model = FSDP(
            self.model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=backward_prefetch,
            mixed_precision=mixed_precision,
            device_id=self.device,
            sync_module_states=self.config.sync_module_states,
            forward_prefetch=self.config.forward_prefetch,
            limit_all_gathers=self.config.limit_all_gathers,
            use_orig_params=self.config.use_orig_params,
        )
        
        self._is_wrapped = True
        return self.fsdp_model
    
    def _apply_activation_checkpointing(self) -> None:
        """应用激活检查点"""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )
        
        check_fn = None
        if self.config.activation_checkpointing_layers:
            layer_classes = tuple(self.config.activation_checkpointing_layers)
            check_fn = lambda submodule: isinstance(submodule, layer_classes)
        
        apply_activation_checkpointing(
            self.model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
    
    def get_model(self) -> nn.Module:
        """获取模型"""
        if self._is_wrapped:
            return self.fsdp_model
        return self.model
    
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        **kwargs,
    ) -> DataLoader:
        """创建分布式数据加载器"""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        full_state_dict: bool = True,
        **kwargs,
    ) -> None:
        """保存 FSDP 检查点
        
        Args:
            path: 保存路径
            optimizer: 优化器
            epoch: 当前轮次
            full_state_dict: 是否保存完整状态字典
        """
        if full_state_dict:
            # 保存完整状态字典（仅主进程）
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.FULL_STATE_DICT,
                save_policy,
            ):
                state_dict = self.fsdp_model.state_dict()
                
                if self.rank == 0:
                    checkpoint = {
                        "model_state_dict": state_dict,
                        "epoch": epoch,
                        **kwargs,
                    }
                    if optimizer is not None:
                        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                    torch.save(checkpoint, path)
        else:
            # 保存分片状态字典（每个进程保存自己的分片）
            save_policy = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.SHARDED_STATE_DICT,
                save_policy,
            ):
                state_dict = self.fsdp_model.state_dict()
                checkpoint = {
                    "model_state_dict": state_dict,
                    "epoch": epoch,
                    **kwargs,
                }
                shard_path = f"{path}.{self.rank}"
                torch.save(checkpoint, shard_path)
    
    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        full_state_dict: bool = True,
    ) -> Dict[str, Any]:
        """加载 FSDP 检查点"""
        if full_state_dict:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.FULL_STATE_DICT,
                save_policy,
            ):
                if self.rank == 0:
                    checkpoint = torch.load(path, map_location="cpu")
                else:
                    checkpoint = {}
                
                # 广播检查点
                if dist.is_initialized():
                    checkpoint_list = [checkpoint]
                    dist.broadcast_object_list(checkpoint_list, src=0)
                    checkpoint = checkpoint_list[0]
                
                self.fsdp_model.load_state_dict(checkpoint["model_state_dict"])
                
                if optimizer is not None and "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                
                return checkpoint
        else:
            shard_path = f"{path}.{self.rank}"
            checkpoint = torch.load(shard_path, map_location="cpu")
            
            save_policy = ShardedStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.SHARDED_STATE_DICT,
                save_policy,
            ):
                self.fsdp_model.load_state_dict(checkpoint["model_state_dict"])
            
            return checkpoint


# 需要导入 os
import os
