"""
Distributed Checkpoint Utilities

Core Idea:
    Distributed checkpointing saves model state across multiple files, one per
    rank, enabling efficient parallel I/O and supporting models too large for
    single-file checkpoints.

Mathematical Theory:
    For a model with P parameters across N ranks, each rank saves P/N parameters.
    Parallel I/O achieves throughput:
    
    .. math::
        T_{\\text{total}} = N \\times T_{\\text{single}}
    
    where T_single is single-rank I/O throughput.

Problem Statement:
    Large models cannot be saved to single files due to memory constraints.
    Distributed checkpointing enables saving/loading model state in parallel
    while maintaining consistency across ranks.

Comparison:
    - Single-file: Simple but memory-limited, sequential I/O
    - Sharded: Parallel I/O, scalable, requires coordination
    - Async: Non-blocking, overlaps with compute, complex state management

Complexity:
    - I/O: O(P/N) per rank for parallel save/load
    - Memory: O(P/N) per rank during checkpointing
    - Coordination: O(1) barrier synchronization

References:
    - PyTorch Distributed Checkpoint documentation
    - Megatron-LM checkpoint utilities
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class CheckpointConfig:
    """Configuration for distributed checkpointing.
    
    Attributes:
        save_dir: Directory for checkpoint storage.
        save_interval: Steps between checkpoints.
        keep_last_n: Number of recent checkpoints to retain.
        async_save: Enable asynchronous saving.
        use_distributed: Enable distributed (sharded) checkpointing.
        compression: Compression method (None, "gzip", "lz4").
    """
    save_dir: str = "./checkpoints"
    save_interval: int = 1000
    keep_last_n: int = 3
    async_save: bool = False
    use_distributed: bool = True
    compression: Optional[str] = None


class DistributedCheckpointer:
    """Manager for distributed checkpoint save and load operations."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._saved_checkpoints: List[str] = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        **kwargs,
    ) -> str:
        """Save distributed checkpoint.
        
        Args:
            model: Model to checkpoint.
            optimizer: Optional optimizer state.
            scheduler: Optional scheduler state.
            step: Current training step.
            epoch: Current epoch.
            
        Returns:
            Path to checkpoint directory.
        """
        checkpoint_dir = self.save_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.use_distributed:
            checkpoint_path = checkpoint_dir / f"rank_{self._rank}.pt"
        else:
            checkpoint_path = checkpoint_dir / "model.pt"
        
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": self._get_model_state_dict(model),
            **kwargs,
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if self._rank == 0:
            self._save_metadata(checkpoint_dir, step, epoch)
        
        if dist.is_initialized():
            dist.barrier()
        
        self._cleanup_old_checkpoints(str(checkpoint_dir))
        
        return str(checkpoint_dir)
    
    def _get_model_state_dict(self, model: nn.Module) -> Dict[str, Any]:
        """Extract model state dict, handling DDP wrapper."""
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()
    
    def _save_metadata(self, checkpoint_dir: Path, step: int, epoch: int) -> None:
        """Save checkpoint metadata (rank 0 only)."""
        import json
        metadata = {
            "step": step,
            "epoch": epoch,
            "world_size": self._world_size,
        }
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
    
    def _cleanup_old_checkpoints(self, new_checkpoint: str) -> None:
        """Remove old checkpoints beyond retention limit."""
        self._saved_checkpoints.append(new_checkpoint)
        
        while len(self._saved_checkpoints) > self.config.keep_last_n:
            old_checkpoint = self._saved_checkpoints.pop(0)
            if self._rank == 0 and os.path.exists(old_checkpoint):
                shutil.rmtree(old_checkpoint)
    
    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load distributed checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if self.config.use_distributed:
            file_path = checkpoint_dir / f"rank_{self._rank}.pt"
        else:
            file_path = checkpoint_dir / "model.pt"
        
        checkpoint = torch.load(file_path, map_location="cpu")
        
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        checkpoints = list(self.save_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        return str(checkpoints[-1])


def save_distributed_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs,
) -> None:
    """Save distributed checkpoint (simplified interface)."""
    config = CheckpointConfig(save_dir=os.path.dirname(path))
    checkpointer = DistributedCheckpointer(config)
    checkpointer.save(model, optimizer, **kwargs)


def load_distributed_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Load distributed checkpoint (simplified interface)."""
    config = CheckpointConfig(save_dir=os.path.dirname(path))
    checkpointer = DistributedCheckpointer(config)
    return checkpointer.load(path, model, optimizer, **kwargs)
