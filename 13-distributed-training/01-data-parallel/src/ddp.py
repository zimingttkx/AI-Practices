"""
Distributed Data Parallel (DDP) Implementation

Core Idea:
    DDP replicates the model on each GPU and synchronizes gradients via AllReduce
    after backward pass, enabling data-parallel training across multiple devices.

Mathematical Theory:
    Given N workers, each processes a mini-batch B_i. The gradient update is:
    
    .. math::
        g = \\frac{1}{N} \\sum_{i=1}^{N} \\nabla L(B_i, \\theta)
    
    where L is the loss function and theta represents model parameters.
    AllReduce computes this average efficiently using ring-based communication.

Problem Statement:
    Single-GPU training is limited by memory and compute. DDP addresses this by
    distributing data across GPUs while maintaining synchronous gradient updates,
    achieving near-linear scaling with minimal communication overhead.

Comparison:
    - vs DataParallel: DDP uses multi-process (no GIL), DP uses multi-thread
    - vs FSDP: DDP replicates full model, FSDP shards parameters
    - vs Pipeline: DDP splits data, Pipeline splits model layers

Complexity:
    - Communication: O(P) per AllReduce, where P is parameter count
    - Memory: O(P) per GPU (full model replica)
    - Computation: O(B/N) per GPU, where B is total batch size

References:
    - Li et al., "PyTorch Distributed: Experiences on Accelerating Data Parallel
      Training", VLDB 2020
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


@dataclass
class DDPConfig:
    """Configuration for Distributed Data Parallel training.
    
    Attributes:
        backend: Communication backend. "nccl" for GPU, "gloo" for CPU.
        init_method: Process group initialization method.
        world_size: Total number of processes (-1 for auto-detection).
        rank: Global rank of current process (-1 for auto-detection).
        local_rank: Local GPU index on current node.
        find_unused_parameters: Enable gradient computation for unused params.
        broadcast_buffers: Synchronize module buffers across replicas.
        gradient_as_bucket_view: Memory optimization for gradient storage.
        static_graph: Enable static graph optimization for fixed architectures.
        bucket_cap_mb: Maximum bucket size for gradient bucketing (MB).
    """
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = -1
    rank: int = -1
    local_rank: int = -1
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    bucket_cap_mb: int = 25


def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    init_method: str = "env://",
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """Initialize the distributed process group for DDP training.
    
    This function sets up the communication infrastructure required for
    gradient synchronization across multiple processes/GPUs.
    
    Args:
        rank: Unique identifier for this process (0 to world_size-1).
        world_size: Total number of processes participating in training.
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU).
        init_method: URL specifying how to initialize the process group.
        master_addr: IP address of the rank-0 process.
        master_port: Port number for inter-process communication.
    
    Raises:
        RuntimeError: If process group initialization fails.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Destroy the distributed process group and release resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Return the global rank of the current process (0 if not distributed)."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Return the total number of processes (1 if not distributed)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def get_local_rank() -> int:
    """Return the local GPU index on the current node."""
    return int(os.environ.get("LOCAL_RANK", 0))


def synchronize() -> None:
    """Block until all processes reach this barrier."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_tensor(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    async_op: bool = False,
) -> Union[torch.Tensor, tuple]:
    """Perform AllReduce operation on a tensor across all processes.
    
    AllReduce aggregates tensors from all processes using the specified
    reduction operation (SUM, AVG, MAX, MIN) and distributes the result
    back to all processes.
    
    Args:
        tensor: Input tensor to reduce.
        op: Reduction operation (SUM, AVG, MAX, MIN, PRODUCT).
        async_op: If True, return immediately with a handle for later wait.
        
    Returns:
        Reduced tensor, or tuple of (tensor, handle) if async_op=True.
    """
    if not dist.is_initialized():
        return tensor
    
    handle = dist.all_reduce(tensor, op=op, async_op=async_op)
    if async_op:
        return tensor, handle
    return tensor


def all_gather_tensor(
    tensor: torch.Tensor,
    world_size: Optional[int] = None,
) -> List[torch.Tensor]:
    """Gather tensors from all processes into a list.
    
    Args:
        tensor: Local tensor to gather.
        world_size: Total number of processes (auto-detected if None).
        
    Returns:
        List of tensors from all processes, ordered by rank.
    """
    if not dist.is_initialized():
        return [tensor]
    
    if world_size is None:
        world_size = get_world_size()
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast_tensor(
    tensor: torch.Tensor,
    src: int = 0,
) -> torch.Tensor:
    """Broadcast tensor from source process to all other processes.
    
    Args:
        tensor: Tensor to broadcast (only src process value is used).
        src: Source process rank.
        
    Returns:
        Broadcasted tensor (same value on all processes).
    """
    if not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


def reduce_dict(
    input_dict: Dict[str, torch.Tensor],
    average: bool = True,
) -> Dict[str, torch.Tensor]:
    """Reduce all tensors in a dictionary across processes.
    
    Args:
        input_dict: Dictionary mapping names to tensors.
        average: If True, compute mean; otherwise compute sum.
        
    Returns:
        Dictionary with reduced tensors.
    """
    if not dist.is_initialized():
        return input_dict
    
    world_size = get_world_size()
    names = []
    values = []
    
    for k, v in sorted(input_dict.items()):
        names.append(k)
        values.append(v)
    
    values = torch.stack(values, dim=0)
    dist.all_reduce(values)
    
    if average:
        values /= world_size
    
    return {k: v for k, v in zip(names, values)}


class DDPTrainer:
    """High-level trainer for Distributed Data Parallel training.
    
    This class encapsulates common DDP operations including model wrapping,
    distributed data loading, checkpointing, and logging.
    
    Mathematical Background:
        DDP achieves data parallelism by partitioning the global batch B into
        N local batches, where each GPU processes B/N samples. Gradients are
        synchronized via ring-AllReduce with O(2P(N-1)/N) communication cost,
        where P is the parameter count.
    
    Args:
        model: PyTorch model to be distributed.
        config: DDP configuration options.
        device: Target device (auto-detected if None).
    
    Example:
        >>> trainer = DDPTrainer(model, DDPConfig())
        >>> trainer.wrap_model()
        >>> dataloader = trainer.create_dataloader(dataset, batch_size=32)
        >>> for batch in dataloader:
        ...     loss = trainer.get_model()(batch)
        ...     loss.backward()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DDPConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or DDPConfig()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()
        
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{self.local_rank}")
            else:
                device = torch.device("cpu")
        self.device = device
        
        self.model = model.to(self.device)
        self.ddp_model = None
        self._is_wrapped = False
    
    def wrap_model(self) -> DDP:
        """Wrap the model with DistributedDataParallel.
        
        Returns:
            DDP-wrapped model ready for distributed training.
        """
        if self._is_wrapped:
            return self.ddp_model
        
        self.ddp_model = DDP(
            self.model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph,
            bucket_cap_mb=self.config.bucket_cap_mb,
        )
        self._is_wrapped = True
        return self.ddp_model
    
    def get_model(self) -> nn.Module:
        """Return the DDP-wrapped model if available, otherwise the raw model."""
        if self._is_wrapped:
            return self.ddp_model
        return self.model
    
    def get_raw_model(self) -> nn.Module:
        """Return the underlying model without DDP wrapper."""
        if self._is_wrapped:
            return self.ddp_model.module
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
        """Create a distributed DataLoader with automatic data sharding.
        
        The DistributedSampler ensures each process receives a unique subset
        of the data, enabling efficient parallel data loading.
        
        Args:
            dataset: Source dataset.
            batch_size: Batch size per GPU (global batch = batch_size * world_size).
            shuffle: Whether to shuffle data each epoch.
            num_workers: Number of data loading worker processes.
            pin_memory: Pin memory for faster GPU transfer.
            drop_last: Drop incomplete final batch.
            **kwargs: Additional DataLoader arguments.
            
        Returns:
            Configured DataLoader with distributed sampling.
        """
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
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        **kwargs,
    ) -> None:
        """Save training checkpoint (main process only).
        
        Args:
            path: File path for checkpoint.
            optimizer: Optimizer state to save.
            scheduler: Learning rate scheduler state to save.
            epoch: Current training epoch.
            **kwargs: Additional state to include.
        """
        if not is_main_process():
            return
        
        checkpoint = {
            "model_state_dict": self.get_raw_model().state_dict(),
            "epoch": epoch,
            **kwargs,
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load training checkpoint with proper device mapping.
        
        Args:
            path: Checkpoint file path.
            optimizer: Optimizer to restore state.
            scheduler: Scheduler to restore state.
            strict: Require exact parameter match.
            
        Returns:
            Loaded checkpoint dictionary.
        """
        map_location = {"cuda:0": f"cuda:{self.local_rank}"}
        checkpoint = torch.load(path, map_location=map_location)
        
        self.get_raw_model().load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict,
        )
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint
    
    def log(self, message: str, *args, **kwargs) -> None:
        """Print message (main process only to avoid duplicate output)."""
        if is_main_process():
            print(message, *args, **kwargs)
