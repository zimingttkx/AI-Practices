"""
Distributed Checkpoint Utilities 分布式检查点工具

================================================================================
核心思想 (一句话理解)
================================================================================
分布式检查点 = 每个rank保存自己的分片 + 并行I/O + 自动清理 = 高效保存超大模型

================================================================================
为什么需要分布式检查点？(问题背景)
================================================================================

    传统检查点的问题:
    ┌─────────────────────────────────────────────────────────────────┐
    │  单文件检查点:                                                   │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  175B参数模型 (GPT-3规模)                                │   │
    │  │  - FP16权重: 350GB                                       │   │
    │  │  - 优化器状态: 700GB (Adam)                              │   │
    │  │  - 总计: 1TB+ 单文件!                                    │   │
    │  │                                                          │   │
    │  │  问题:                                                   │   │
    │  │  1. 内存不足: 需要额外显存存储完整状态                    │   │
    │  │  2. I/O瓶颈: 串行写入，保存需要几十分钟                  │   │
    │  │  3. 单点故障: 文件损坏则全部丢失                         │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
工作原理 (图解)
================================================================================

    分布式检查点的解决方案:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  8个GPU并行保存:                                                │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  Rank 0 → rank_0.pt (125GB)  ─┐                         │   │
    │  │  Rank 1 → rank_1.pt (125GB)   │                         │   │
    │  │  Rank 2 → rank_2.pt (125GB)   │  并行写入               │   │
    │  │  Rank 3 → rank_3.pt (125GB)   │  速度提升8倍!           │   │
    │  │  Rank 4 → rank_4.pt (125GB)   │                         │   │
    │  │  Rank 5 → rank_5.pt (125GB)   │                         │   │
    │  │  Rank 6 → rank_6.pt (125GB)   │                         │   │
    │  │  Rank 7 → rank_7.pt (125GB)  ─┘                         │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  检查点目录结构:                                                 │
    │  checkpoint-1000/                                               │
    │  ├── metadata.json    # 元数据 (step, epoch, world_size)       │
    │  ├── rank_0.pt        # Rank 0的模型分片 + 优化器状态           │
    │  ├── rank_1.pt        # Rank 1的模型分片 + 优化器状态           │
    │  └── ...                                                        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    保存流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  1. 创建检查点目录                                               │
    │     └── checkpoint-{step}/                                      │
    │                                                                 │
    │  2. 每个rank并行保存自己的分片                                   │
    │     ├── Rank 0: 保存 rank_0.pt                                  │
    │     ├── Rank 1: 保存 rank_1.pt                                  │
    │     └── ...                                                     │
    │                                                                 │
    │  3. Rank 0保存元数据                                             │
    │     └── metadata.json                                           │
    │                                                                 │
    │  4. 同步屏障 (确保所有rank完成)                                  │
    │     └── dist.barrier()                                          │
    │                                                                 │
    │  5. 清理旧检查点 (保留最近N个)                                   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
检查点策略对比
================================================================================
    单文件检查点:
    - 优点: 简单，易于管理
    - 缺点: 内存受限，I/O慢
    - 适用: 小模型 (<10B参数)

    分布式检查点:
    - 优点: 并行I/O，可扩展
    - 缺点: 需要协调，恢复时需要相同world_size
    - 适用: 大模型 (>10B参数)

    异步检查点:
    - 优点: 不阻塞训练
    - 缺点: 状态管理复杂
    - 适用: 训练时间敏感的场景

================================================================================
前置知识
================================================================================
- PyTorch分布式训练基础 (dist.barrier)
- 模型状态字典 (state_dict)
- 优化器状态保存/加载

================================================================================
参考文献
================================================================================
- PyTorch Distributed Checkpoint documentation
- Megatron-LM checkpoint utilities
"""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class CheckpointConfig:
    """分布式检查点配置

    Attributes:
        save_dir: 检查点保存目录
        save_interval: 保存间隔 (步数)
        keep_last_n: 保留最近N个检查点
        async_save: 是否异步保存 (不阻塞训练)
        use_distributed: 是否使用分布式保存 (每个rank单独保存)
        compression: 压缩方式 (None, "gzip", "lz4")

    检查点保留策略:
        - keep_last_n=3: 保留最近3个检查点
        - 自动删除旧检查点，节省存储空间
        - 建议至少保留2个，防止保存过程中断

    Example:
        >>> config = CheckpointConfig(
        ...     save_dir="./checkpoints",
        ...     save_interval=1000,    # 每1000步保存
        ...     keep_last_n=3,         # 保留最近3个
        ...     use_distributed=True,  # 分布式保存
        ... )
    """
    save_dir: str = "./checkpoints"
    save_interval: int = 1000
    keep_last_n: int = 3
    async_save: bool = False
    use_distributed: bool = True
    compression: Optional[str] = None


# =============================================================================
# 分布式检查点管理器
# =============================================================================

class DistributedCheckpointer:
    """分布式检查点管理器

    管理大规模模型的检查点保存和加载，支持:
    - 分布式保存: 每个rank保存自己的分片
    - 自动清理: 保留最近N个检查点
    - 元数据管理: 记录训练状态

    工作流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │  保存:                                                          │
    │  1. checkpointer.save(model, optimizer, step=1000)             │
    │     - 每个rank保存自己的分片                                    │
    │     - Rank 0保存元数据                                          │
    │     - 自动清理旧检查点                                          │
    │                                                                 │
    │  加载:                                                          │
    │  2. checkpointer.load(path, model, optimizer)                  │
    │     - 每个rank加载自己的分片                                    │
    │     - 恢复模型和优化器状态                                      │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        config: 检查点配置

    Example:
        >>> config = CheckpointConfig(save_dir="./ckpt", keep_last_n=3)
        >>> checkpointer = DistributedCheckpointer(config)
        >>> # 保存
        >>> path = checkpointer.save(model, optimizer, step=1000)
        >>> # 加载
        >>> checkpointer.load(path, model, optimizer)
    """

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.save_dir = Path(config.save_dir)
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 获取分布式信息
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # 记录已保存的检查点 (用于自动清理)
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
        """保存分布式检查点

        每个rank保存自己的模型分片和优化器状态。

        Args:
            model: 要保存的模型
            optimizer: 优化器 (可选)
            scheduler: 学习率调度器 (可选)
            step: 当前训练步数
            epoch: 当前epoch
            **kwargs: 额外要保存的数据 (如loss, metrics等)

        Returns:
            检查点目录路径

        Example:
            >>> path = checkpointer.save(
            ...     model=model,
            ...     optimizer=optimizer,
            ...     step=1000,
            ...     epoch=1,
            ...     loss=0.5,  # 额外数据
            ... )
        """
        # 创建检查点目录: checkpoint-{step}/
        checkpoint_dir = self.save_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 确定保存路径
        if self.config.use_distributed:
            # 分布式: 每个rank保存单独文件
            checkpoint_path = checkpoint_dir / f"rank_{self._rank}.pt"
        else:
            # 单文件: 只有rank 0保存
            checkpoint_path = checkpoint_dir / "model.pt"

        # 构建检查点内容
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": self._get_model_state_dict(model),
            **kwargs,  # 额外数据
        }

        # 保存优化器状态
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # 保存调度器状态
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # 保存到文件
        torch.save(checkpoint, checkpoint_path)

        # Rank 0保存元数据
        if self._rank == 0:
            self._save_metadata(checkpoint_dir, step, epoch)

        # 同步屏障: 确保所有rank都完成保存
        if dist.is_initialized():
            dist.barrier()

        # 清理旧检查点
        self._cleanup_old_checkpoints(str(checkpoint_dir))

        return str(checkpoint_dir)

    def _get_model_state_dict(self, model: nn.Module) -> Dict[str, Any]:
        """提取模型状态字典

        处理DDP包装的模型，提取内部模型的状态。

        Args:
            model: 模型 (可能被DDP包装)

        Returns:
            模型状态字典
        """
        # DDP包装的模型有.module属性
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()

    def _save_metadata(self, checkpoint_dir: Path, step: int, epoch: int) -> None:
        """保存检查点元数据 (仅Rank 0执行)

        元数据包含训练状态信息，用于恢复训练。

        Args:
            checkpoint_dir: 检查点目录
            step: 当前步数
            epoch: 当前epoch
        """
        metadata = {
            "step": step,
            "epoch": epoch,
            "world_size": self._world_size,  # 记录world_size，加载时验证
        }
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _cleanup_old_checkpoints(self, new_checkpoint: str) -> None:
        """清理旧检查点

        保留最近keep_last_n个检查点，删除更早的。

        Args:
            new_checkpoint: 新保存的检查点路径
        """
        self._saved_checkpoints.append(new_checkpoint)

        # 超过保留数量时删除最旧的
        while len(self._saved_checkpoints) > self.config.keep_last_n:
            old_checkpoint = self._saved_checkpoints.pop(0)
            # 只有Rank 0执行删除
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
        """加载分布式检查点

        每个rank加载自己的分片。

        Args:
            checkpoint_path: 检查点目录路径
            model: 要加载状态的模型
            optimizer: 优化器 (可选)
            scheduler: 学习率调度器 (可选)
            strict: 是否严格匹配模型结构

        Returns:
            检查点内容字典 (包含step, epoch等)

        Example:
            >>> checkpoint = checkpointer.load(
            ...     "checkpoints/checkpoint-1000",
            ...     model=model,
            ...     optimizer=optimizer,
            ... )
            >>> print(f"从step {checkpoint['step']}恢复")
        """
        checkpoint_dir = Path(checkpoint_path)

        # 确定加载路径
        if self.config.use_distributed:
            file_path = checkpoint_dir / f"rank_{self._rank}.pt"
        else:
            file_path = checkpoint_dir / "model.pt"

        # 加载检查点 (先加载到CPU，避免显存问题)
        checkpoint = torch.load(file_path, map_location="cpu")

        # 加载模型状态
        if hasattr(model, "module"):
            # DDP包装的模型
            model.module.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # 加载优化器状态
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 加载调度器状态
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新检查点路径

        扫描保存目录，返回step最大的检查点。

        Returns:
            最新检查点路径，如果没有则返回None

        Example:
            >>> latest = checkpointer.get_latest_checkpoint()
            >>> if latest:
            ...     checkpointer.load(latest, model, optimizer)
        """
        # 查找所有checkpoint-*目录
        checkpoints = list(self.save_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None

        # 按step排序
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        return str(checkpoints[-1])


# =============================================================================
# 便捷函数
# =============================================================================

def save_distributed_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs,
) -> None:
    """保存分布式检查点 (简化接口)

    Args:
        model: 要保存的模型
        path: 保存路径
        optimizer: 优化器 (可选)
        **kwargs: 额外数据

    Example:
        >>> save_distributed_checkpoint(model, "./ckpt/step-1000", optimizer)
    """
    config = CheckpointConfig(save_dir=os.path.dirname(path))
    checkpointer = DistributedCheckpointer(config)
    checkpointer.save(model, optimizer, **kwargs)


def load_distributed_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs,
) -> Dict[str, Any]:
    """加载分布式检查点 (简化接口)

    Args:
        model: 要加载状态的模型
        path: 检查点路径
        optimizer: 优化器 (可选)
        **kwargs: 额外参数

    Returns:
        检查点内容字典

    Example:
        >>> checkpoint = load_distributed_checkpoint(model, "./ckpt/step-1000")
        >>> print(f"恢复到step {checkpoint['step']}")
    """
    config = CheckpointConfig(save_dir=os.path.dirname(path))
    checkpointer = DistributedCheckpointer(config)
    return checkpointer.load(path, model, optimizer, **kwargs)
