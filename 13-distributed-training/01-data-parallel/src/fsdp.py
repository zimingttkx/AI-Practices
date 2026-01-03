"""
Fully Sharded Data Parallel (FSDP) Implementation

Core Idea:
    FSDP shards model parameters, gradients, and optimizer states across GPUs,
    gathering them on-demand during forward/backward passes. This enables
    training models that exceed single-GPU memory capacity.

Mathematical Theory:
    For a model with P parameters across N GPUs, memory per GPU is reduced from:
    
    .. math::
        M_{DDP} = P + P + kP = (2+k)P  \\quad \\text{(params + grads + optimizer)}
    
    to:
    
    .. math::
        M_{FSDP} = \\frac{P}{N} + \\frac{P}{N} + \\frac{kP}{N} = \\frac{(2+k)P}{N}
    
    where k is the optimizer state multiplier (k=2 for Adam: momentum + variance).

Problem Statement:
    Large models (billions of parameters) cannot fit on a single GPU even with
    gradient checkpointing. FSDP solves this by distributing memory footprint
    while maintaining data parallelism semantics.

Comparison:
    - vs DDP: FSDP shards everything, DDP replicates everything
    - vs ZeRO-3: Equivalent memory efficiency, FSDP is PyTorch-native
    - vs Pipeline: FSDP is data-parallel, Pipeline is model-parallel

Complexity:
    - Communication: O(P) per AllGather + O(P) per ReduceScatter per layer
    - Memory: O(P/N) per GPU (sharded state)
    - Computation: Same as DDP, O(B/N) per GPU

References:
    - Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data
      Parallel", VLDB 2023
    - Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion
      Parameter Models", SC 2020
"""

import functools
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type

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
    )
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


class ShardingStrategy(Enum):
    """FSDP sharding strategy options.
    
    Attributes:
        FULL_SHARD: Shard parameters, gradients, and optimizer states (ZeRO-3).
        SHARD_GRAD_OP: Shard gradients and optimizer states only (ZeRO-2).
        NO_SHARD: No sharding, equivalent to DDP.
        HYBRID_SHARD: Shard within nodes, replicate across nodes.
    """
    FULL_SHARD = auto()
    SHARD_GRAD_OP = auto()
    NO_SHARD = auto()
    HYBRID_SHARD = auto()


@dataclass
class FSDPConfig:
    """Configuration for Fully Sharded Data Parallel training.
    
    Attributes:
        sharding_strategy: How to shard model state across GPUs.
        cpu_offload: Offload parameters to CPU when not in use.
        backward_prefetch: Prefetch strategy for backward pass.
        mixed_precision: Enable mixed precision training.
        mixed_precision_dtype: Data type for mixed precision (float16/bfloat16).
        auto_wrap_policy: Policy for automatic FSDP wrapping.
        min_num_params: Minimum parameters for size-based wrapping.
        transformer_layer_cls: Layer classes for transformer wrapping.
        use_orig_params: Use original parameter references.
        sync_module_states: Synchronize module states on initialization.
        forward_prefetch: Enable forward pass prefetching.
        limit_all_gathers: Limit concurrent AllGather operations.
        activation_checkpointing: Enable gradient checkpointing.
        activation_checkpointing_layers: Layers to apply checkpointing.
    """
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    cpu_offload: bool = False
    backward_prefetch: str = "backward_pre"
    mixed_precision: bool = False
    mixed_precision_dtype: torch.dtype = torch.float16
    auto_wrap_policy: str = "size_based"
    min_num_params: int = 100_000
    transformer_layer_cls: Optional[List[Type[nn.Module]]] = None
    use_orig_params: bool = True
    sync_module_states: bool = True
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    activation_checkpointing: bool = False
    activation_checkpointing_layers: Optional[List[Type[nn.Module]]] = None


def _get_sharding_strategy(strategy: ShardingStrategy):
    """Convert ShardingStrategy enum to PyTorch FSDP strategy."""
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
    """Convert backward prefetch string to PyTorch enum."""
    if not FSDP_AVAILABLE or prefetch is None:
        return None
    
    mapping = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    return mapping.get(prefetch)


def get_fsdp_wrap_policy(config: FSDPConfig) -> Optional[Callable]:
    """Create FSDP auto-wrap policy based on configuration.
    
    Args:
        config: FSDP configuration.
        
    Returns:
        Wrap policy function or None.
        
    Raises:
        ValueError: If transformer policy specified without layer classes.
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
    """Create FSDP mixed precision configuration."""
    if not FSDP_AVAILABLE or not config.mixed_precision:
        return None
    
    return MixedPrecision(
        param_dtype=config.mixed_precision_dtype,
        reduce_dtype=config.mixed_precision_dtype,
        buffer_dtype=config.mixed_precision_dtype,
    )


class FSDPTrainer:
    """High-level trainer for Fully Sharded Data Parallel training.
    
    This class provides a simplified interface for FSDP training, handling
    model wrapping, activation checkpointing, and distributed checkpointing.
    
    Mathematical Background:
        FSDP reduces memory by a factor of N (number of GPUs) compared to DDP.
        The trade-off is increased communication: each layer requires AllGather
        before forward/backward and ReduceScatter after backward.
    
    Args:
        model: PyTorch model to be sharded.
        config: FSDP configuration options.
        device: Target device (auto-detected if None).
        
    Raises:
        RuntimeError: If FSDP is not available (PyTorch < 1.12).
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
        """Wrap the model with FSDP for distributed training.
        
        Returns:
            FSDP-wrapped model with configured sharding strategy.
        """
        if self._is_wrapped:
            return self.fsdp_model
        
        if self.config.activation_checkpointing:
            self._apply_activation_checkpointing()
        
        sharding_strategy = _get_sharding_strategy(self.config.sharding_strategy)
        backward_prefetch = _get_backward_prefetch(self.config.backward_prefetch)
        mixed_precision = get_fsdp_mixed_precision(self.config)
        auto_wrap_policy = get_fsdp_wrap_policy(self.config)
        
        cpu_offload = None
        if self.config.cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        
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
        """Apply gradient checkpointing to reduce memory usage."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
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
        """Return the FSDP-wrapped model if available, otherwise the raw model."""
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
        """Create a distributed DataLoader with automatic data sharding."""
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
        """Save FSDP checkpoint with configurable state dict type.
        
        Args:
            path: File path for checkpoint.
            optimizer: Optimizer state to save.
            epoch: Current training epoch.
            full_state_dict: If True, gather full state on rank 0; else save shards.
            **kwargs: Additional state to include.
        """
        if full_state_dict:
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
        """Load FSDP checkpoint with proper state dict handling.
        
        Args:
            path: Checkpoint file path.
            optimizer: Optimizer to restore state.
            full_state_dict: If True, load from full checkpoint; else load shards.
            
        Returns:
            Loaded checkpoint dictionary.
        """
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
