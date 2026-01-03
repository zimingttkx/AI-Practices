"""
Distributed Data Parallel (DDP) 分布式数据并行实现

================================================================================
核心思想 (一句话理解)
================================================================================
DDP = 每个GPU存完整模型 + 数据分片 + 梯度AllReduce同步

================================================================================
工作原理 (图解)
================================================================================

    ┌─────────────────────────────────────────────────────────────────┐
    │  训练流程                                                        │
    │                                                                 │
    │  1. 数据分片: 1024条数据 → GPU0(256) + GPU1(256) + ...          │
    │  2. 前向传播: 每个GPU独立计算自己的loss                          │
    │  3. 反向传播: 每个GPU独立计算自己的梯度                          │
    │  4. 梯度同步: AllReduce求平均 → 所有GPU得到相同梯度              │
    │  5. 参数更新: 每个GPU用相同梯度更新 → 模型保持一致               │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
数学原理
================================================================================
设有N个GPU，每个处理mini-batch B_i，梯度更新公式:

    g = (1/N) × Σ ∇L(B_i, θ)    # 所有GPU梯度的平均值

AllReduce通过环形通信高效计算这个平均值，通信量 ≈ 2×参数量 (与GPU数无关)

================================================================================
适用场景
================================================================================
- 模型能完整放入单个GPU显存
- 需要加速训练（多卡并行处理数据）
- 追求简单实现和最小通信开销

================================================================================
前置知识
================================================================================
- PyTorch基础: nn.Module, DataLoader, 优化器
- 多进程概念: 进程、进程间通信
- GPU编程基础: CUDA, 显存概念

================================================================================
参考文献
================================================================================
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


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class DDPConfig:
    """DDP训练配置

    这个类定义了DDP训练需要的所有配置参数。
    使用dataclass可以方便地创建配置对象，并提供默认值。

    Attributes:
        backend: 通信后端
            - "nccl": GPU训练首选，NVIDIA优化的通信库
            - "gloo": CPU训练或不支持NCCL时使用
        init_method: 进程组初始化方式，"env://"表示从环境变量读取
        world_size: 总进程数（通常等于GPU数），-1表示自动检测
        rank: 当前进程的全局编号（0到world_size-1），-1表示自动检测
        local_rank: 当前进程在本机的GPU编号
        find_unused_parameters: 是否检测未使用的参数
            - True: 模型有些参数不参与计算时需要开启
            - False: 默认关闭，性能更好
        broadcast_buffers: 是否同步模型的buffer（如BatchNorm的running_mean）
        gradient_as_bucket_view: 梯度内存优化，减少内存拷贝
        static_graph: 静态图优化，模型结构固定时可开启提升性能
        bucket_cap_mb: 梯度桶大小(MB)，影响通信效率

    Example:
        >>> config = DDPConfig(
        ...     backend="nccl",
        ...     find_unused_parameters=False,
        ...     static_graph=True,  # 模型结构固定时开启
        ... )
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


# =============================================================================
# 分布式环境初始化/清理函数
# =============================================================================

def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    init_method: str = "env://",
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """初始化DDP分布式训练环境

    这是DDP训练的第一步，必须在训练开始前调用。
    它会建立进程间的通信连接，让多个GPU能够互相通信。

    工作原理:
        1. 设置环境变量告诉PyTorch主进程的地址和端口
        2. 初始化进程组，建立进程间通信
        3. 设置当前进程使用的GPU

    Args:
        rank: 当前进程编号（0, 1, 2, ...）
            - rank=0 是主进程，负责保存模型、打印日志等
        world_size: 总进程数（通常等于GPU数量）
        backend: 通信后端，GPU用"nccl"，CPU用"gloo"
        init_method: 初始化方式，"env://"从环境变量读取
        master_addr: 主进程IP地址，单机训练用"localhost"
        master_port: 通信端口，确保未被占用

    Example:
        >>> # 在每个进程中调用
        >>> setup_ddp(rank=0, world_size=4)  # 进程0
        >>> setup_ddp(rank=1, world_size=4)  # 进程1
        >>> # ...

    Note:
        使用torchrun启动时，rank和world_size会自动设置:
        torchrun --nproc_per_node=4 train.py
    """
    # 设置主进程地址，所有进程都需要知道主进程在哪
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # 初始化进程组 - 这一步会阻塞直到所有进程都调用了这个函数
    dist.init_process_group(
        backend=backend,      # 通信后端
        init_method=init_method,
        world_size=world_size,  # 总进程数
        rank=rank,              # 当前进程编号
    )

    # 设置当前进程使用的GPU（进程0用GPU0，进程1用GPU1...）
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """清理DDP分布式环境

    训练结束后调用，释放通信资源。
    如果不调用，可能会导致资源泄漏或下次训练初始化失败。
    """
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# 分布式信息查询函数
# =============================================================================

def get_rank() -> int:
    """获取当前进程的全局编号

    Returns:
        进程编号（0到world_size-1），非分布式环境返回0

    Example:
        >>> if get_rank() == 0:
        ...     print("我是主进程，负责保存模型")
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """获取总进程数

    Returns:
        总进程数，非分布式环境返回1

    Example:
        >>> global_batch_size = batch_size * get_world_size()
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """判断当前是否是主进程（rank=0）

    主进程通常负责:
    - 保存模型检查点
    - 打印训练日志
    - 记录TensorBoard

    Returns:
        True如果是主进程，否则False

    Example:
        >>> if is_main_process():
        ...     torch.save(model.state_dict(), "model.pt")
        ...     print(f"Epoch {epoch}, Loss: {loss}")
    """
    return get_rank() == 0


def get_local_rank() -> int:
    """获取当前进程在本机的GPU编号

    在多机训练中，local_rank和rank不同:
    - rank: 全局编号（0, 1, 2, 3, 4, 5, 6, 7）
    - local_rank: 本机编号（机器1: 0,1,2,3  机器2: 0,1,2,3）

    Returns:
        本机GPU编号，默认0
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def synchronize() -> None:
    """同步所有进程（barrier）

    调用后，所有进程会在这里等待，直到所有进程都到达这个点。
    用于确保所有进程完成某个操作后再继续。

    Example:
        >>> # 确保所有进程都完成数据加载
        >>> load_data()
        >>> synchronize()  # 等待所有进程
        >>> start_training()
    """
    if dist.is_initialized():
        dist.barrier()


# =============================================================================
# 分布式通信原语
# =============================================================================

def all_reduce_tensor(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    async_op: bool = False,
) -> Union[torch.Tensor, tuple]:
    """AllReduce操作：所有进程的张量聚合后分发给所有进程

    这是DDP的核心通信操作。AllReduce = Reduce + Broadcast

    工作原理:
        GPU0: [1, 2, 3]  ─┐
        GPU1: [4, 5, 6]  ─┼─→ AllReduce(SUM) ─→ 每个GPU都得到 [5, 7, 9]
        GPU2: [0, 0, 0]  ─┘

    Args:
        tensor: 要聚合的张量（会被原地修改）
        op: 聚合操作
            - SUM: 求和（默认）
            - AVG: 求平均
            - MAX: 取最大值
            - MIN: 取最小值
        async_op: 是否异步执行
            - False: 阻塞直到完成
            - True: 立即返回，需要手动等待

    Returns:
        聚合后的张量，如果async_op=True则返回(tensor, handle)

    Example:
        >>> # 同步所有GPU的loss
        >>> loss_tensor = torch.tensor([loss], device="cuda")
        >>> all_reduce_tensor(loss_tensor, op=dist.ReduceOp.SUM)
        >>> avg_loss = loss_tensor.item() / get_world_size()
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
    """AllGather操作：收集所有进程的张量到每个进程

    工作原理:
        GPU0: [A]  ─┐
        GPU1: [B]  ─┼─→ AllGather ─→ 每个GPU都得到 [[A], [B], [C]]
        GPU2: [C]  ─┘

    Args:
        tensor: 本进程的张量
        world_size: 总进程数，None则自动获取

    Returns:
        包含所有进程张量的列表，按rank排序

    Example:
        >>> # 收集所有GPU的预测结果
        >>> local_preds = model(local_batch)
        >>> all_preds = all_gather_tensor(local_preds)
        >>> full_preds = torch.cat(all_preds, dim=0)
    """
    if not dist.is_initialized():
        return [tensor]

    if world_size is None:
        world_size = get_world_size()

    # 创建接收缓冲区
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast_tensor(
    tensor: torch.Tensor,
    src: int = 0,
) -> torch.Tensor:
    """Broadcast操作：从源进程广播张量到所有进程

    工作原理:
        GPU0: [A]  ─┐
        GPU1: [?]  ─┼─→ Broadcast(src=0) ─→ 每个GPU都得到 [A]
        GPU2: [?]  ─┘

    Args:
        tensor: 要广播的张量（只有src进程的值有意义）
        src: 源进程的rank

    Returns:
        广播后的张量（所有进程相同）

    Example:
        >>> # 主进程生成随机种子，广播给所有进程
        >>> if is_main_process():
        ...     seed = torch.tensor([42], device="cuda")
        >>> else:
        ...     seed = torch.tensor([0], device="cuda")
        >>> broadcast_tensor(seed, src=0)
        >>> # 现在所有进程的seed都是42
    """
    if not dist.is_initialized():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def reduce_dict(
    input_dict: Dict[str, torch.Tensor],
    average: bool = True,
) -> Dict[str, torch.Tensor]:
    """聚合字典中所有张量

    方便同步多个指标（如loss、accuracy等）

    Args:
        input_dict: 键为指标名，值为张量的字典
        average: True则求平均，False则求和

    Returns:
        聚合后的字典

    Example:
        >>> metrics = {"loss": loss_tensor, "acc": acc_tensor}
        >>> avg_metrics = reduce_dict(metrics, average=True)
    """
    if not dist.is_initialized():
        return input_dict

    world_size = get_world_size()
    names = []
    values = []

    # 按键排序确保所有进程顺序一致
    for k, v in sorted(input_dict.items()):
        names.append(k)
        values.append(v)

    values = torch.stack(values, dim=0)
    dist.all_reduce(values)

    if average:
        values /= world_size

    return {k: v for k, v in zip(names, values)}


# =============================================================================
# DDP训练器
# =============================================================================

class DDPTrainer:
    """DDP分布式训练器

    封装了DDP训练的常用操作，包括:
    - 模型包装
    - 分布式数据加载
    - 检查点保存/加载
    - 日志打印

    使用流程:
        1. 创建训练器: trainer = DDPTrainer(model, config)
        2. 包装模型: ddp_model = trainer.wrap_model()
        3. 创建数据加载器: dataloader = trainer.create_dataloader(dataset)
        4. 训练循环中使用ddp_model
        5. 保存检查点: trainer.save_checkpoint(...)

    Args:
        model: PyTorch模型
        config: DDP配置
        device: 目标设备，None则自动检测

    Example:
        >>> model = MyModel()
        >>> trainer = DDPTrainer(model, DDPConfig())
        >>> ddp_model = trainer.wrap_model()
        >>> dataloader = trainer.create_dataloader(dataset, batch_size=32)
        >>>
        >>> for epoch in range(num_epochs):
        ...     dataloader.sampler.set_epoch(epoch)  # 重要！
        ...     for batch in dataloader:
        ...         loss = ddp_model(batch)
        ...         loss.backward()
        ...         optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[DDPConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or DDPConfig()

        # 获取分布式信息
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()

        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{self.local_rank}")
            else:
                device = torch.device("cpu")
        self.device = device

        # 将模型移到对应设备
        self.model = model.to(self.device)
        self.ddp_model = None
        self._is_wrapped = False

    def wrap_model(self) -> DDP:
        """用DDP包装模型

        包装后的模型会自动在backward()时同步梯度。

        Returns:
            DDP包装后的模型

        Note:
            - 只需调用一次
            - 包装后使用ddp_model进行训练
            - 保存模型时使用ddp_model.module获取原始模型
        """
        if self._is_wrapped:
            return self.ddp_model

        self.ddp_model = DDP(
            self.model,
            # GPU相关设置
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            # 性能相关设置
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph,
            bucket_cap_mb=self.config.bucket_cap_mb,
        )
        self._is_wrapped = True
        return self.ddp_model

    def get_model(self) -> nn.Module:
        """获取模型（优先返回DDP包装后的模型）"""
        if self._is_wrapped:
            return self.ddp_model
        return self.model

    def get_raw_model(self) -> nn.Module:
        """获取原始模型（不带DDP包装）

        用于保存模型、访问模型属性等场景。
        """
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

        使用DistributedSampler自动将数据分片到各个进程。

        Args:
            dataset: 数据集
            batch_size: 每个GPU的batch大小
                - 全局batch = batch_size × world_size
                - 例如: batch_size=32, 4个GPU → 全局batch=128
            shuffle: 是否打乱数据
            num_workers: 数据加载进程数
            pin_memory: 是否使用锁页内存（GPU训练建议开启）
            drop_last: 是否丢弃最后不完整的batch

        Returns:
            配置好的DataLoader

        Important:
            每个epoch开始时必须调用:
            dataloader.sampler.set_epoch(epoch)
            否则每个epoch的数据顺序相同！
        """
        # DistributedSampler会自动将数据分片
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,  # 总进程数
            rank=self.rank,                 # 当前进程编号
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,  # 使用分布式采样器，不能同时设置shuffle
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
        """保存训练检查点（仅主进程执行）

        Args:
            path: 保存路径
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
            epoch: 当前epoch
            **kwargs: 其他要保存的内容

        Note:
            只有rank=0的进程会保存，避免多进程同时写文件
        """
        # 只有主进程保存
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
        """加载训练检查点

        Args:
            path: 检查点路径
            optimizer: 要恢复状态的优化器
            scheduler: 要恢复状态的调度器
            strict: 是否严格匹配参数名

        Returns:
            加载的检查点字典
        """
        # 映射到当前GPU
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
        """打印日志（仅主进程执行）

        避免多进程重复打印相同内容。
        """
        if is_main_process():
            print(message, *args, **kwargs)
