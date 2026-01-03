"""
Fully Sharded Data Parallel (FSDP) 全分片数据并行实现

================================================================================
核心思想 (一句话理解)
================================================================================
FSDP = 参数/梯度/优化器状态全部分片存储 + 计算时按需聚合 + 计算后立即释放

================================================================================
FSDP vs DDP 对比 (图解)
================================================================================

    DDP: 每个GPU存储完整模型 (显存浪费)
    ┌─────────────────────────────────────────────────────────────┐
    │ GPU0: [全部参数] [全部梯度] [全部优化器状态]                  │
    │ GPU1: [全部参数] [全部梯度] [全部优化器状态]  ← 完全重复!     │
    │ GPU2: [全部参数] [全部梯度] [全部优化器状态]                  │
    └─────────────────────────────────────────────────────────────┘

    FSDP: 每个GPU只存储1/N (显存节省N倍)
    ┌─────────────────────────────────────────────────────────────┐
    │ GPU0: [参数分片0] [梯度分片0] [优化器分片0]                   │
    │ GPU1: [参数分片1] [梯度分片1] [优化器分片1]  ← 无重复!        │
    │ GPU2: [参数分片2] [梯度分片2] [优化器分片2]                   │
    └─────────────────────────────────────────────────────────────┘

================================================================================
FSDP 执行流程
================================================================================

    前向传播 (Forward):
    ┌─────────────────────────────────────────────────────────────┐
    │ 对于每一层:                                                  │
    │   1. AllGather: 从所有GPU收集该层的完整参数                  │
    │   2. Forward: 用完整参数计算该层输出                         │
    │   3. 释放: 计算完成后释放非本地参数分片 (节省显存)            │
    └─────────────────────────────────────────────────────────────┘

    反向传播 (Backward):
    ┌─────────────────────────────────────────────────────────────┐
    │ 对于每一层 (逆序):                                           │
    │   1. AllGather: 重新收集该层完整参数                         │
    │   2. Backward: 计算梯度                                      │
    │   3. ReduceScatter: 梯度归约并分片存储到各GPU                │
    │   4. 释放: 释放非本地参数                                    │
    └─────────────────────────────────────────────────────────────┘

================================================================================
显存计算公式
================================================================================
设: P = 参数量, N = GPU数量, k = 优化器状态倍数 (Adam: k=2)

    DDP每卡显存:  P + P + kP = (2+k)P     # 参数 + 梯度 + 优化器
    FSDP每卡显存: P/N + P/N + kP/N = (2+k)P/N   # 全部分片

    → FSDP显存节省N倍！

================================================================================
适用场景
================================================================================
- 模型太大，单卡放不下
- 想用更大的batch size
- 愿意用通信开销换取显存节省

================================================================================
前置知识
================================================================================
- DDP的基本概念
- AllGather/ReduceScatter通信原语
- 梯度检查点(Gradient Checkpointing)概念

================================================================================
参考文献
================================================================================
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

# 尝试导入FSDP相关模块 (需要PyTorch 1.12+)
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


# =============================================================================
# 分片策略枚举
# =============================================================================

class ShardingStrategy(Enum):
    """FSDP分片策略

    不同策略在显存节省和通信开销之间做权衡。

    Attributes:
        FULL_SHARD: 完全分片 (等价于ZeRO-3)
            - 参数、梯度、优化器状态全部分片
            - 显存节省最多，通信开销最大
            - 适合: 模型远超单卡显存

        SHARD_GRAD_OP: 梯度+优化器分片 (等价于ZeRO-2)
            - 只分片梯度和优化器状态，参数不分片
            - 显存节省适中，通信开销较小
            - 适合: 模型略超单卡显存

        NO_SHARD: 不分片 (等价于DDP)
            - 不做任何分片
            - 无额外通信开销
            - 适合: 模型能放入单卡

        HYBRID_SHARD: 混合分片
            - 节点内完全分片，节点间不分片
            - 减少跨机通信
            - 适合: 多机训练场景
    """
    FULL_SHARD = auto()      # 完全分片，最省显存
    SHARD_GRAD_OP = auto()   # 梯度+优化器分片
    NO_SHARD = auto()        # 不分片，等价于DDP
    HYBRID_SHARD = auto()    # 混合分片，多机场景


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class FSDPConfig:
    """FSDP训练配置

    Attributes:
        sharding_strategy: 分片策略，决定如何分片模型状态
        cpu_offload: 是否将参数卸载到CPU (显存极度紧张时使用)
        backward_prefetch: 反向传播预取策略
            - "backward_pre": 提前预取下一层参数 (推荐)
            - "backward_post": 计算完当前层后预取
        mixed_precision: 是否启用混合精度训练
        mixed_precision_dtype: 混合精度数据类型 (float16/bfloat16)
        auto_wrap_policy: 自动包装策略
            - "size_based": 按参数量自动包装
            - "transformer": 按Transformer层包装
            - "none": 不自动包装
        min_num_params: size_based策略的最小参数量阈值
        transformer_layer_cls: transformer策略要包装的层类型
        use_orig_params: 使用原始参数引用 (方便访问参数)
        sync_module_states: 初始化时同步模块状态
        forward_prefetch: 前向传播预取
        limit_all_gathers: 限制并发AllGather数量 (防OOM)
        activation_checkpointing: 激活检查点 (进一步省显存)
        activation_checkpointing_layers: 要应用检查点的层类型

    Example:
        >>> config = FSDPConfig(
        ...     sharding_strategy=ShardingStrategy.FULL_SHARD,
        ...     mixed_precision=True,
        ...     activation_checkpointing=True,
        ... )
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


# =============================================================================
# 辅助函数
# =============================================================================

def _get_sharding_strategy(strategy: ShardingStrategy):
    """将自定义枚举转换为PyTorch FSDP的分片策略"""
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP不可用，请升级到PyTorch 1.12+")

    mapping = {
        ShardingStrategy.FULL_SHARD: TorchShardingStrategy.FULL_SHARD,
        ShardingStrategy.SHARD_GRAD_OP: TorchShardingStrategy.SHARD_GRAD_OP,
        ShardingStrategy.NO_SHARD: TorchShardingStrategy.NO_SHARD,
        ShardingStrategy.HYBRID_SHARD: TorchShardingStrategy.HYBRID_SHARD,
    }
    return mapping[strategy]


def _get_backward_prefetch(prefetch: Optional[str]):
    """将预取策略字符串转换为PyTorch枚举"""
    if not FSDP_AVAILABLE or prefetch is None:
        return None

    mapping = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    return mapping.get(prefetch)


def get_fsdp_wrap_policy(config: FSDPConfig) -> Optional[Callable]:
    """创建FSDP自动包装策略

    自动包装策略决定了模型的哪些部分会被FSDP包装。
    合理的包装可以平衡通信开销和显存节省。

    Args:
        config: FSDP配置

    Returns:
        包装策略函数，或None表示不自动包装

    Raises:
        ValueError: 配置无效时抛出
    """
    if not FSDP_AVAILABLE:
        return None

    if config.auto_wrap_policy == "none":
        return None

    # 按参数量包装: 参数量超过阈值的模块会被单独包装
    if config.auto_wrap_policy == "size_based":
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=config.min_num_params,
        )

    # 按Transformer层包装: 每个Transformer层单独包装
    if config.auto_wrap_policy == "transformer":
        if config.transformer_layer_cls is None:
            raise ValueError(
                "使用transformer包装策略时必须指定transformer_layer_cls"
            )
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(config.transformer_layer_cls),
        )

    raise ValueError(f"未知的auto_wrap_policy: {config.auto_wrap_policy}")


def get_fsdp_mixed_precision(config: FSDPConfig) -> Optional[Any]:
    """创建FSDP混合精度配置"""
    if not FSDP_AVAILABLE or not config.mixed_precision:
        return None

    return MixedPrecision(
        param_dtype=config.mixed_precision_dtype,    # 参数精度
        reduce_dtype=config.mixed_precision_dtype,   # 梯度归约精度
        buffer_dtype=config.mixed_precision_dtype,   # buffer精度
    )


# =============================================================================
# FSDP训练器
# =============================================================================

class FSDPTrainer:
    """FSDP分布式训练器

    封装了FSDP训练的常用操作，简化使用流程。

    核心功能:
        - 模型FSDP包装
        - 激活检查点
        - 分布式数据加载
        - 检查点保存/加载 (支持全量和分片两种模式)

    使用流程:
        1. 创建训练器: trainer = FSDPTrainer(model, config)
        2. 包装模型: fsdp_model = trainer.wrap_model()
        3. 创建数据加载器: dataloader = trainer.create_dataloader(dataset)
        4. 训练循环
        5. 保存检查点: trainer.save_checkpoint(...)

    Args:
        model: PyTorch模型
        config: FSDP配置
        device: 目标设备

    Raises:
        RuntimeError: PyTorch版本不支持FSDP时抛出

    Example:
        >>> config = FSDPConfig(
        ...     sharding_strategy=ShardingStrategy.FULL_SHARD,
        ...     mixed_precision=True,
        ... )
        >>> trainer = FSDPTrainer(model, config)
        >>> fsdp_model = trainer.wrap_model()
        >>>
        >>> optimizer = torch.optim.AdamW(fsdp_model.parameters())
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = fsdp_model(batch)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[FSDPConfig] = None,
        device: Optional[torch.device] = None,
    ):
        # 检查FSDP是否可用
        if not FSDP_AVAILABLE:
            raise RuntimeError(
                "FSDP不可用，请升级到PyTorch 1.12+"
            )

        self.config = config or FSDPConfig()

        # 获取分布式信息
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # 设置设备
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
        """用FSDP包装模型

        包装后的模型会自动处理参数分片、梯度同步等。

        Returns:
            FSDP包装后的模型

        Note:
            - 只需调用一次
            - 包装后模型结构会改变，不能直接访问原始参数
            - 保存/加载检查点需要使用特殊方法
        """
        if self._is_wrapped:
            return self.fsdp_model

        # 如果启用激活检查点，先应用
        if self.config.activation_checkpointing:
            self._apply_activation_checkpointing()

        # 准备FSDP参数
        sharding_strategy = _get_sharding_strategy(self.config.sharding_strategy)
        backward_prefetch = _get_backward_prefetch(self.config.backward_prefetch)
        mixed_precision = get_fsdp_mixed_precision(self.config)
        auto_wrap_policy = get_fsdp_wrap_policy(self.config)

        # CPU卸载配置
        cpu_offload = None
        if self.config.cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)

        # 创建FSDP包装
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
        """应用激活检查点以减少显存使用

        激活检查点通过在反向传播时重新计算激活值来节省显存，
        代价是增加约30%的计算时间。
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            apply_activation_checkpointing,
        )

        # 确定要应用检查点的层
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
        """获取模型（优先返回FSDP包装后的模型）"""
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
        """创建分布式数据加载器

        与DDP相同，使用DistributedSampler自动分片数据。
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
        epoch: int = 0,
        full_state_dict: bool = True,
        **kwargs,
    ) -> None:
        """保存FSDP检查点

        FSDP检查点有两种模式:
        1. full_state_dict=True: 收集完整模型到rank0保存 (方便加载)
        2. full_state_dict=False: 每个rank保存自己的分片 (更快)

        Args:
            path: 保存路径
            optimizer: 优化器
            epoch: 当前epoch
            full_state_dict: 是否保存完整模型
            **kwargs: 其他要保存的内容

        Note:
            full_state_dict=True时，只有rank0会保存文件
            full_state_dict=False时，每个rank保存自己的分片
        """
        if full_state_dict:
            # 模式1: 收集完整模型到rank0
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.FULL_STATE_DICT,
                save_policy,
            ):
                state_dict = self.fsdp_model.state_dict()

                # 只有rank0保存
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
            # 模式2: 每个rank保存自己的分片
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
                # 每个rank保存自己的分片
                shard_path = f"{path}.{self.rank}"
                torch.save(checkpoint, shard_path)

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        full_state_dict: bool = True,
    ) -> Dict[str, Any]:
        """加载FSDP检查点

        Args:
            path: 检查点路径
            optimizer: 要恢复状态的优化器
            full_state_dict: 是否从完整检查点加载

        Returns:
            加载的检查点字典
        """
        if full_state_dict:
            # 从完整检查点加载
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.FULL_STATE_DICT,
                save_policy,
            ):
                # rank0加载并广播
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
            # 从分片检查点加载
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
