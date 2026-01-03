"""
分布式检查点工具

提供大规模分布式训练的检查点保存和加载功能。

核心功能:
    - 分片检查点保存/加载
    - 异步检查点
    - 检查点压缩
    - 自动恢复
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class CheckpointConfig:
    """检查点配置
    
    Attributes:
        save_dir: 保存目录
        save_interval: 保存间隔（步数）
        keep_last_n: 保留最近 N 个检查点
        async_save: 是否异步保存
        use_distributed: 是否使用分布式保存
        compression: 压缩方式 (None, "gzip", "lz4")
    """
    save_dir: str = "./checkpoints"
    save_interval: int = 1000
    keep_last_n: int = 3
    async_save: bool = False
    use_distributed: bool = True
    compression: Optional[str] = None


class DistributedCheckpointer:
    """分布式检查点管理器"""
    
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
        """保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            step: 当前步数
            epoch: 当前轮次
            
        Returns:
            检查点路径
        """
        checkpoint_dir = self.save_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.use_distributed:
            checkpoint_path = checkpoint_dir / f"rank_{self._rank}.pt"
        else:
            checkpoint_path = checkpoint_dir / "model.pt"
        
        # 构建检查点
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
        
        # 保存
        torch.save(checkpoint, checkpoint_path)
        
        # 保存元数据（仅主进程）
        if self._rank == 0:
            self._save_metadata(checkpoint_dir, step, epoch)
        
        # 同步
        if dist.is_initialized():
            dist.barrier()
        
        # 清理旧检查点
        self._cleanup_old_checkpoints(str(checkpoint_dir))
        
        return str(checkpoint_dir)
    
    def _get_model_state_dict(self, model: nn.Module) -> Dict[str, Any]:
        """获取模型状态字典"""
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()
    
    def _save_metadata(self, checkpoint_dir: Path, step: int, epoch: int) -> None:
        """保存元数据"""
        import json
        metadata = {
            "step": step,
            "epoch": epoch,
            "world_size": self._world_size,
        }
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
    
    def _cleanup_old_checkpoints(self, new_checkpoint: str) -> None:
        """清理旧检查点"""
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
        """加载检查点"""
        checkpoint_dir = Path(checkpoint_path)
        
        if self.config.use_distributed:
            file_path = checkpoint_dir / f"rank_{self._rank}.pt"
        else:
            file_path = checkpoint_dir / "model.pt"
        
        checkpoint = torch.load(file_path, map_location="cpu")
        
        # 加载模型
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        # 加载优化器
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 加载调度器
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新检查点路径"""
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
    """保存分布式检查点（简化接口）"""
    config = CheckpointConfig(save_dir=os.path.dirname(path))
    checkpointer = DistributedCheckpointer(config)
    checkpointer.save(model, optimizer, **kwargs)


def load_distributed_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs,
) -> Dict[str, Any]:
    """加载分布式检查点（简化接口）"""
    config = CheckpointConfig(save_dir=os.path.dirname(path))
    checkpointer = DistributedCheckpointer(config)
    return checkpointer.load(path, model, optimizer, **kwargs)
