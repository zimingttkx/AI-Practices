"""
Megatron-Core Integration Megatron并行状态管理

================================================================================
核心思想 (一句话理解)
================================================================================
Megatron = 3D并行拓扑管理 + 进程组创建 + 通信协调 = 训练万亿参数模型的基础设施

================================================================================
什么是Megatron-LM？
================================================================================

    Megatron-LM是NVIDIA开发的大规模Transformer训练框架:
    ┌─────────────────────────────────────────────────────────────────┐
    │  核心功能:                                                       │
    │  1. 3D并行: 数据并行 × 张量并行 × 流水线并行                     │
    │  2. 进程组管理: 为每种并行创建独立的通信组                        │
    │  3. 优化通信: 针对NVIDIA GPU优化的通信模式                       │
    │  4. 序列并行: 与张量并行配合，进一步节省显存                      │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
3D并行拓扑 (图解)
================================================================================

    假设: 16个GPU，张量并行(TP)=2，流水线并行(PP)=4，数据并行(DP)=2

    GPU编号和3D坐标 (d, t, p):
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  数据并行组0 (d=0):                                              │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  PP Stage 0    PP Stage 1    PP Stage 2    PP Stage 3   │   │
    │  │  GPU0(0,0,0) → GPU2(0,0,1) → GPU4(0,0,2) → GPU6(0,0,3)  │   │
    │  │  GPU1(0,1,0) → GPU3(0,1,1) → GPU5(0,1,2) → GPU7(0,1,3)  │   │
    │  │      ↑ TP组                                              │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  数据并行组1 (d=1):                                              │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  GPU8(1,0,0) → GPU10(1,0,1) → GPU12(1,0,2) → GPU14(1,0,3)│   │
    │  │  GPU9(1,1,0) → GPU11(1,1,1) → GPU13(1,1,2) → GPU15(1,1,3)│   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  通信组划分:                                                     │
    │  - TP组: {GPU0,GPU1}, {GPU2,GPU3}, ... (同一PP阶段内)           │
    │  - PP组: {GPU0,GPU2,GPU4,GPU6}, ... (同一TP位置，沿流水线)       │
    │  - DP组: {GPU0,GPU8}, {GPU1,GPU9}, ... (同一TP和PP位置)         │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
进程组的作用
================================================================================
    - TP组: AllReduce/AllGather用于张量并行通信
    - PP组: P2P Send/Recv用于流水线阶段间传递激活值
    - DP组: AllReduce用于梯度同步

================================================================================
前置知识
================================================================================
- 数据并行、张量并行、流水线并行的概念
- PyTorch分布式通信 (dist.new_group)
- 进程组和rank的概念

================================================================================
参考文献
================================================================================
- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language
  Models Using Model Parallelism", arXiv 2019
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU
  Clusters Using Megatron-LM", SC 2021
"""

from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class MegatronConfig:
    """Megatron并行配置

    Attributes:
        tensor_model_parallel_size: 张量并行度 (TP)
        pipeline_model_parallel_size: 流水线并行度 (PP)
        data_parallel_size: 数据并行度 (DP，通常自动计算)
        sequence_parallel: 是否启用序列并行
        virtual_pipeline_model_parallel_size: 虚拟流水线阶段数 (交错调度)
        context_parallel_size: 上下文并行度 (长序列)

    约束条件:
        world_size = TP × PP × DP

    Example:
        >>> # 16个GPU: TP=2, PP=4, DP=2
        >>> config = MegatronConfig(
        ...     tensor_model_parallel_size=2,
        ...     pipeline_model_parallel_size=4,
        ...     # data_parallel_size自动计算为2
        ... )
    """
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    data_parallel_size: int = 1
    sequence_parallel: bool = False
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1


# =============================================================================
# 并行状态单例
# =============================================================================

class MegatronParallelState:
    """Megatron并行状态管理器 (单例模式)

    管理张量并行、流水线并行、数据并行的进程组和rank信息。

    单例模式确保整个程序中只有一个并行状态实例。

    工作流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. 初始化: initialize(config)                                  │
    │     - 根据配置创建TP、PP、DP进程组                               │
    │     - 计算当前rank在各维度的位置                                 │
    │                                                                 │
    │  2. 使用: 通过属性获取进程组和rank                               │
    │     - tensor_model_parallel_group: TP通信组                     │
    │     - pipeline_model_parallel_group: PP通信组                   │
    │     - data_parallel_group: DP通信组                             │
    └─────────────────────────────────────────────────────────────────┘

    Example:
        >>> state = MegatronParallelState()
        >>> state.initialize(config)
        >>> tp_group = state.tensor_model_parallel_group
        >>> tp_rank = state.tensor_model_parallel_rank
    """

    _instance = None

    def __new__(cls):
        """单例模式: 确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 进程组
        self._tensor_model_parallel_group = None
        self._pipeline_model_parallel_group = None
        self._data_parallel_group = None
        self._model_parallel_group = None

        # 各维度的rank
        self._tensor_model_parallel_rank = 0
        self._pipeline_model_parallel_rank = 0
        self._data_parallel_rank = 0

        # 各维度的world_size
        self._tensor_model_parallel_world_size = 1
        self._pipeline_model_parallel_world_size = 1
        self._data_parallel_world_size = 1

        self._initialized = False

    def initialize(self, config: MegatronConfig) -> None:
        """初始化并行状态

        根据配置创建所有进程组，并计算当前rank在各维度的位置。

        Args:
            config: Megatron并行配置

        Raises:
            AssertionError: 如果world_size != TP × PP × DP
        """
        # 如果分布式未初始化，跳过进程组创建
        if not dist.is_initialized():
            self._initialized = True
            return

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        tp_size = config.tensor_model_parallel_size
        pp_size = config.pipeline_model_parallel_size
        dp_size = world_size // (tp_size * pp_size)

        # 验证配置
        assert world_size == tp_size * pp_size * dp_size, \
            f"world_size({world_size}) != TP({tp_size}) × PP({pp_size}) × DP({dp_size})"

        # 创建进程组
        self._create_tensor_parallel_groups(rank, world_size, tp_size, pp_size, dp_size)
        self._create_pipeline_parallel_groups(rank, world_size, tp_size, pp_size, dp_size)
        self._create_data_parallel_groups(rank, world_size, tp_size, pp_size, dp_size)

        # 保存world_size
        self._tensor_model_parallel_world_size = tp_size
        self._pipeline_model_parallel_world_size = pp_size
        self._data_parallel_world_size = dp_size

        self._initialized = True

    def _create_tensor_parallel_groups(
        self, rank: int, world_size: int, tp_size: int, pp_size: int, dp_size: int
    ) -> None:
        """创建张量并行进程组

        TP组包含同一PP阶段、同一DP组内的GPU。
        这些GPU共同计算一个层的不同部分。

        Args:
            rank: 当前全局rank
            world_size: 总GPU数
            tp_size: 张量并行度
            pp_size: 流水线并行度
            dp_size: 数据并行度
        """
        num_tp_groups = world_size // tp_size

        for i in range(num_tp_groups):
            # 每个TP组包含连续的tp_size个rank
            ranks = list(range(i * tp_size, (i + 1) * tp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                self._tensor_model_parallel_group = group
                self._tensor_model_parallel_rank = ranks.index(rank)

    def _create_pipeline_parallel_groups(
        self, rank: int, world_size: int, tp_size: int, pp_size: int, dp_size: int
    ) -> None:
        """创建流水线并行进程组

        PP组包含同一TP位置、同一DP组内的GPU。
        这些GPU沿流水线方向传递激活值。

        Args:
            rank: 当前全局rank
            world_size: 总GPU数
            tp_size: 张量并行度
            pp_size: 流水线并行度
            dp_size: 数据并行度
        """
        for i in range(dp_size):
            for j in range(tp_size):
                # PP组: 同一DP组、同一TP位置的GPU
                ranks = [i * tp_size * pp_size + j + k * tp_size for k in range(pp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self._pipeline_model_parallel_group = group
                    self._pipeline_model_parallel_rank = ranks.index(rank)

    def _create_data_parallel_groups(
        self, rank: int, world_size: int, tp_size: int, pp_size: int, dp_size: int
    ) -> None:
        """创建数据并行进程组

        DP组包含同一TP位置、同一PP阶段的GPU。
        这些GPU处理不同的数据，需要同步梯度。

        Args:
            rank: 当前全局rank
            world_size: 总GPU数
            tp_size: 张量并行度
            pp_size: 流水线并行度
            dp_size: 数据并行度
        """
        for i in range(pp_size):
            for j in range(tp_size):
                # DP组: 同一PP阶段、同一TP位置的GPU
                ranks = [i * tp_size + j + k * tp_size * pp_size for k in range(dp_size)]
                group = dist.new_group(ranks)
                if rank in ranks:
                    self._data_parallel_group = group
                    self._data_parallel_rank = ranks.index(rank)

    # =========================================================================
    # 进程组属性
    # =========================================================================

    @property
    def tensor_model_parallel_group(self):
        """获取张量并行进程组"""
        return self._tensor_model_parallel_group

    @property
    def pipeline_model_parallel_group(self):
        """获取流水线并行进程组"""
        return self._pipeline_model_parallel_group

    @property
    def data_parallel_group(self):
        """获取数据并行进程组"""
        return self._data_parallel_group

    # =========================================================================
    # Rank属性
    # =========================================================================

    @property
    def tensor_model_parallel_rank(self) -> int:
        """获取在张量并行组内的rank"""
        return self._tensor_model_parallel_rank

    @property
    def pipeline_model_parallel_rank(self) -> int:
        """获取在流水线并行组内的rank (即PP阶段编号)"""
        return self._pipeline_model_parallel_rank

    @property
    def data_parallel_rank(self) -> int:
        """获取在数据并行组内的rank"""
        return self._data_parallel_rank

    # =========================================================================
    # World Size属性
    # =========================================================================

    @property
    def tensor_model_parallel_world_size(self) -> int:
        """获取张量并行度"""
        return self._tensor_model_parallel_world_size

    @property
    def pipeline_model_parallel_world_size(self) -> int:
        """获取流水线并行度"""
        return self._pipeline_model_parallel_world_size

    @property
    def data_parallel_world_size(self) -> int:
        """获取数据并行度"""
        return self._data_parallel_world_size

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def is_pipeline_first_stage(self) -> bool:
        """检查当前rank是否是流水线第一阶段

        第一阶段负责接收原始输入数据。

        Returns:
            True如果是第一阶段
        """
        return self._pipeline_model_parallel_rank == 0

    def is_pipeline_last_stage(self) -> bool:
        """检查当前rank是否是流水线最后阶段

        最后阶段负责计算损失。

        Returns:
            True如果是最后阶段
        """
        return self._pipeline_model_parallel_rank == self._pipeline_model_parallel_world_size - 1


# =============================================================================
# 便捷函数
# =============================================================================

def initialize_megatron(config: MegatronConfig) -> MegatronParallelState:
    """初始化Megatron并行环境

    Args:
        config: Megatron并行配置

    Returns:
        初始化后的并行状态单例

    Example:
        >>> config = MegatronConfig(tensor_model_parallel_size=2)
        >>> state = initialize_megatron(config)
    """
    state = MegatronParallelState()
    state.initialize(config)
    return state


def get_model_parallel_group():
    """获取张量并行进程组 (便捷函数)

    Returns:
        张量并行进程组
    """
    state = MegatronParallelState()
    return state.tensor_model_parallel_group


def get_data_parallel_group():
    """获取数据并行进程组 (便捷函数)

    Returns:
        数据并行进程组
    """
    state = MegatronParallelState()
    return state.data_parallel_group


def get_pipeline_parallel_group():
    """获取流水线并行进程组 (便捷函数)

    Returns:
        流水线并行进程组
    """
    state = MegatronParallelState()
    return state.pipeline_model_parallel_group
