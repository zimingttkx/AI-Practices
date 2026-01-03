"""
PyTorch Distributed Data Parallel (DDP) 实现

DDP 是 PyTorch 中最常用的数据并行方式，通过在多个 GPU 上复制模型，
每个 GPU 处理不同的数据批次，然后同步梯度来实现并行训练。

核心概念:
    - 每个进程持有完整的模型副本
    - 数据在进程间分片
    - 梯度通过 AllReduce 同步
    - 支持单机多卡和多机多卡
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


@dataclass
class DDPConfig:
    """DDP 配置类
    
    Attributes:
        backend: 通信后端 ("nccl", "gloo", "mpi")
        init_method: 初始化方法 ("env://", "tcp://", "file://")
        world_size: 总进程数
        rank: 当前进程排名
        local_rank: 本地 GPU 排名
        find_unused_parameters: 是否查找未使用的参数
        broadcast_buffers: 是否广播缓冲区
        gradient_as_bucket_view: 梯度作为桶视图
        static_graph: 是否使用静态图优化
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
    """初始化 DDP 环境
    
    Args:
        rank: 当前进程排名
        world_size: 总进程数
        backend: 通信后端
        init_method: 初始化方法
        master_addr: 主节点地址
        master_port: 主节点端口
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
    """清理 DDP 环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """获取当前进程排名"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """获取总进程数"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """判断是否为主进程"""
    return get_rank() == 0


def get_local_rank() -> int:
    """获取本地 GPU 排名"""
    return int(os.environ.get("LOCAL_RANK", 0))


def synchronize() -> None:
    """同步所有进程"""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_tensor(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    async_op: bool = False,
) -> torch.Tensor:
    """对张量执行 AllReduce 操作
    
    Args:
        tensor: 输入张量
        op: 归约操作类型
        async_op: 是否异步执行
        
    Returns:
        归约后的张量
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
    """收集所有进程的张量
    
    Args:
        tensor: 输入张量
        world_size: 总进程数
        
    Returns:
        所有进程张量的列表
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
    """从源进程广播张量
    
    Args:
        tensor: 输入张量
        src: 源进程排名
        
    Returns:
        广播后的张量
    """
    if not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


def reduce_dict(
    input_dict: Dict[str, torch.Tensor],
    average: bool = True,
) -> Dict[str, torch.Tensor]:
    """归约字典中的所有张量
    
    Args:
        input_dict: 输入字典
        average: 是否取平均
        
    Returns:
        归约后的字典
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
    """DDP 训练器
    
    封装了 DDP 训练的常用操作，包括模型包装、数据加载、训练循环等。
    
    Args:
        model: PyTorch 模型
        config: DDP 配置
        device: 训练设备
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
        """将模型包装为 DDP 模型"""
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
        """获取模型（DDP 包装后或原始模型）"""
        if self._is_wrapped:
            return self.ddp_model
        return self.model
    
    def get_raw_model(self) -> nn.Module:
        """获取原始模型（去除 DDP 包装）"""
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
        """创建分布式数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 每个 GPU 的批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载工作进程数
            pin_memory: 是否固定内存
            drop_last: 是否丢弃最后不完整的批次
            
        Returns:
            分布式数据加载器
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
        """保存检查点（仅主进程）
        
        Args:
            path: 保存路径
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前轮次
            **kwargs: 其他需要保存的内容
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
        """加载检查点
        
        Args:
            path: 检查点路径
            optimizer: 优化器
            scheduler: 学习率调度器
            strict: 是否严格匹配参数
            
        Returns:
            检查点内容
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
        """日志输出（仅主进程）"""
        if is_main_process():
            print(message, *args, **kwargs)
