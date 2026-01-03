"""
Pipeline Parallelism 流水线并行实现

================================================================================
核心思想 (一句话理解)
================================================================================
流水线并行 = 模型按层切分到多个GPU + 微批次流水执行 + 减少GPU空闲时间

================================================================================
工作原理 (图解)
================================================================================

    模型切分示例 (24层Transformer，4个GPU):
    ┌─────────────────────────────────────────────────────────────────┐
    │  GPU0 (Stage 0): Layer 0-5                                      │
    │  GPU1 (Stage 1): Layer 6-11                                     │
    │  GPU2 (Stage 2): Layer 12-17                                    │
    │  GPU3 (Stage 3): Layer 18-23                                    │
    └─────────────────────────────────────────────────────────────────┘

    朴素执行 (问题: 大量GPU空闲):
    ┌─────────────────────────────────────────────────────────────────┐
    │  时间 →                                                          │
    │  GPU0: [F]                        [B]                           │
    │  GPU1:     [F]                [B]                               │
    │  GPU2:         [F]        [B]                                   │
    │  GPU3:             [F][B]                                       │
    │                                                                 │
    │  F=Forward, B=Backward                                          │
    │  空白 = 气泡 (GPU空闲等待)                                       │
    │  气泡率 ≈ 75% (3/4的时间在等待!)                                 │
    └─────────────────────────────────────────────────────────────────┘

    微批次流水线 (解决方案):
    ┌─────────────────────────────────────────────────────────────────┐
    │  把一个大batch拆成多个micro-batch，流水执行                       │
    │                                                                 │
    │  时间 →                                                          │
    │  GPU0: [F0][F1][F2][F3]      [B3][B2][B1][B0]                   │
    │  GPU1:    [F0][F1][F2][F3][B3][B2][B1][B0]                      │
    │  GPU2:       [F0][F1][F2][F3][B2][B1][B0]                       │
    │  GPU3:          [F0][F1][F2][F3][B1][B0]                        │
    │                                                                 │
    │  气泡率 = (S-1)/(M+S-1)                                          │
    │  S=4阶段, M=4微批次 → 气泡率 = 3/7 ≈ 43%                         │
    │  S=4阶段, M=16微批次 → 气泡率 = 3/19 ≈ 16%                       │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
两种调度策略
================================================================================

    GPipe: 先完成所有Forward，再做所有Backward
    - 优点: 实现简单
    - 缺点: 需要存储所有微批次的激活值，显存占用大

    1F1B (One Forward One Backward): 交替执行
    - 优点: 显存占用恒定 (只需存储S个微批次的激活值)
    - 缺点: 实现复杂

================================================================================
通信模式
================================================================================
    - 点对点通信 (P2P): Stage之间传递激活值和梯度
    - 通信量: O(batch × seq × hidden) per micro-batch

================================================================================
前置知识
================================================================================
- 深度学习的前向/反向传播
- 分布式通信基础 (send/recv)
- 梯度累积概念

================================================================================
参考文献
================================================================================
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using
  Pipeline Parallelism", NeurIPS 2019
- Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN
  Training", SOSP 2019
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class PipelineConfig:
    """流水线并行配置

    Attributes:
        num_stages: 流水线阶段数 (通常等于GPU数)
        num_micro_batches: 微批次数量
            - 越大气泡率越低，但显存占用越大 (GPipe)
            - 建议: num_micro_batches >= 4 * num_stages
        stage_id: 当前阶段ID (0 到 num_stages-1)
        process_group: 分布式进程组
        chunks: 每个batch的chunk数 (等于num_micro_batches)
        checkpoint_activations: 是否启用激活检查点 (省显存)

    Example:
        >>> config = PipelineConfig(
        ...     num_stages=4,
        ...     num_micro_batches=16,  # 气泡率 ≈ 16%
        ...     stage_id=0,
        ... )
    """
    num_stages: int = 1
    num_micro_batches: int = 1
    stage_id: int = 0
    process_group: Optional[dist.ProcessGroup] = None
    chunks: int = 1
    checkpoint_activations: bool = False


# =============================================================================
# 流水线阶段
# =============================================================================

class PipelineStage(nn.Module):
    """流水线阶段包装器

    包装模型的一部分层，处理与相邻阶段的通信和激活值缓存。

    工作流程:
        1. 从上一阶段接收输入 (或使用原始输入，如果是第一阶段)
        2. 执行本阶段的前向传播
        3. 将输出发送给下一阶段 (或计算loss，如果是最后阶段)
        4. 反向传播时，接收梯度，计算本地梯度，发送给上一阶段

    Args:
        module: 本阶段包含的模型层
        stage_id: 阶段编号 (0 到 num_stages-1)
        num_stages: 总阶段数
        config: 流水线配置

    Example:
        >>> # 假设有一个24层的模型，分成4个阶段
        >>> stage0_layers = nn.Sequential(*layers[0:6])
        >>> stage = PipelineStage(stage0_layers, stage_id=0, num_stages=4)
    """

    def __init__(
        self,
        module: nn.Module,
        stage_id: int,
        num_stages: int,
        config: Optional[PipelineConfig] = None,
    ):
        super().__init__()

        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.config = config or PipelineConfig(
            num_stages=num_stages, stage_id=stage_id
        )

        # 判断是否是首尾阶段
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1

        # 激活值缓存 (用于反向传播)
        # key: micro_batch_id, value: tensor
        self._input_cache: Dict[int, torch.Tensor] = {}
        self._output_cache: Dict[int, torch.Tensor] = {}

    def forward(self, input_: torch.Tensor, micro_batch_id: int = 0) -> torch.Tensor:
        """前向传播，缓存激活值用于反向传播

        Args:
            input_: 输入张量
            micro_batch_id: 微批次编号，用于缓存管理

        Returns:
            本阶段的输出
        """
        # 缓存输入用于反向传播
        if self.config.checkpoint_activations:
            self._input_cache[micro_batch_id] = input_.detach()

        output = self.module(input_)
        self._output_cache[micro_batch_id] = output

        return output

    def send_forward(self, tensor: torch.Tensor, dst: int) -> None:
        """发送激活值给下一阶段

        Args:
            tensor: 要发送的激活值
            dst: 目标阶段的rank
        """
        if not dist.is_initialized():
            return
        dist.send(tensor, dst, group=self.config.process_group)

    def recv_forward(self, src: int, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """从上一阶段接收激活值

        Args:
            src: 源阶段的rank
            shape: 期望的张量形状
            dtype: 期望的数据类型

        Returns:
            接收到的激活值
        """
        if not dist.is_initialized():
            return torch.zeros(shape, dtype=dtype)

        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        dist.recv(tensor, src, group=self.config.process_group)
        return tensor

    def send_backward(self, tensor: torch.Tensor, dst: int) -> None:
        """发送梯度给上一阶段"""
        if not dist.is_initialized():
            return
        dist.send(tensor, dst, group=self.config.process_group)

    def recv_backward(self, src: int, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """从下一阶段接收梯度"""
        if not dist.is_initialized():
            return torch.zeros(shape, dtype=dtype)

        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        dist.recv(tensor, src, group=self.config.process_group)
        return tensor

    def clear_cache(self) -> None:
        """清空激活值缓存，释放显存"""
        self._input_cache.clear()
        self._output_cache.clear()


# =============================================================================
# 调度器基类
# =============================================================================

class PipelineScheduler(ABC):
    """流水线调度器抽象基类

    调度器决定了微批次的执行顺序，不同的调度策略有不同的
    显存占用和气泡率特性。
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.num_stages = config.num_stages
        self.num_micro_batches = config.num_micro_batches
        self.stage_id = config.stage_id

    @abstractmethod
    def get_schedule(self) -> List[Tuple[str, int]]:
        """返回调度计划

        Returns:
            (操作类型, 微批次ID) 的列表
            操作类型: "forward" 或 "backward"
        """
        pass

    @abstractmethod
    def forward_backward(
        self,
        stage: PipelineStage,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """执行完整的前向和反向传播

        Args:
            stage: 流水线阶段
            batch: 输入数据
            loss_fn: 损失函数

        Returns:
            平均损失值
        """
        pass


# =============================================================================
# GPipe调度器
# =============================================================================

class GPipeScheduler(PipelineScheduler):
    """GPipe调度器: 先完成所有Forward，再做所有Backward

    执行顺序:
        F0 → F1 → F2 → F3 → B3 → B2 → B1 → B0

    特点:
        - 实现简单
        - 需要存储所有微批次的激活值
        - 显存占用 = O(M × 激活值大小)，M为微批次数

    气泡率: (S-1)/(M+S-1)，S=阶段数，M=微批次数
    """

    def get_schedule(self) -> List[Tuple[str, int]]:
        """生成GPipe调度计划"""
        schedule = []

        # 先执行所有forward
        for mb in range(self.num_micro_batches):
            schedule.append(("forward", mb))

        # 再执行所有backward (逆序)
        for mb in reversed(range(self.num_micro_batches)):
            schedule.append(("backward", mb))

        return schedule

    def forward_backward(
        self,
        stage: PipelineStage,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """执行GPipe前向-反向传播"""
        # 将batch拆分成微批次
        micro_batches = torch.chunk(batch, self.num_micro_batches, dim=0)
        outputs = []
        losses = []

        # ===== Forward阶段 =====
        for mb_id, mb in enumerate(micro_batches):
            # 获取输入
            if stage.is_first_stage:
                input_ = mb  # 第一阶段直接使用原始输入
            else:
                # 从上一阶段接收
                input_ = stage.recv_forward(
                    stage.stage_id - 1,
                    mb.shape,
                    mb.dtype,
                )

            # 前向传播
            output = stage.forward(input_, mb_id)
            outputs.append(output)

            # 发送给下一阶段或计算loss
            if not stage.is_last_stage:
                stage.send_forward(output, stage.stage_id + 1)
            else:
                loss = loss_fn(output)
                losses.append(loss)

        # ===== Backward阶段 =====
        for mb_id in reversed(range(self.num_micro_batches)):
            if stage.is_last_stage:
                # 最后阶段从loss开始反向传播
                losses[mb_id].backward()
                grad = outputs[mb_id].grad
            else:
                # 从下一阶段接收梯度
                grad = stage.recv_backward(
                    stage.stage_id + 1,
                    outputs[mb_id].shape,
                    outputs[mb_id].dtype,
                )
                outputs[mb_id].backward(grad)

            # 发送梯度给上一阶段
            if not stage.is_first_stage:
                input_grad = stage._input_cache[mb_id].grad
                stage.send_backward(input_grad, stage.stage_id - 1)

        # 清理缓存
        stage.clear_cache()

        # 返回平均loss
        if losses:
            return sum(losses) / len(losses)
        return torch.tensor(0.0)


# =============================================================================
# PipeDream (1F1B) 调度器
# =============================================================================

class PipeDreamScheduler(PipelineScheduler):
    """PipeDream (1F1B) 调度器: 交替执行Forward和Backward

    执行顺序 (以Stage 0为例):
        预热: F0 → F1 → F2 → F3
        稳态: F4 → B0 → F5 → B1 → ...
        收尾: B(M-4) → B(M-3) → B(M-2) → B(M-1)

    特点:
        - 显存占用恒定 = O(S × 激活值大小)，S为阶段数
        - 实现较复杂
        - 与GPipe相同的气泡率

    适用场景:
        - 显存紧张时优先选择
        - 微批次数量较大时
    """

    def get_schedule(self) -> List[Tuple[str, int]]:
        """生成1F1B调度计划"""
        schedule = []

        # 预热阶段: 只做forward，数量 = num_stages - stage_id - 1
        num_warmup = self.num_stages - self.stage_id - 1
        num_1f1b = self.num_micro_batches - num_warmup

        # 预热
        for mb in range(num_warmup):
            schedule.append(("forward", mb))

        # 稳态: 1F1B交替
        for i in range(num_1f1b):
            schedule.append(("forward", num_warmup + i))
            schedule.append(("backward", i))

        # 收尾: 只做backward
        for i in range(num_warmup):
            schedule.append(("backward", num_1f1b + i))

        return schedule

    def forward_backward(
        self,
        stage: PipelineStage,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """执行PipeDream 1F1B前向-反向传播"""
        micro_batches = torch.chunk(batch, self.num_micro_batches, dim=0)
        schedule = self.get_schedule()

        outputs: Dict[int, torch.Tensor] = {}
        losses: List[torch.Tensor] = []

        for op, mb_id in schedule:
            if op == "forward":
                mb = micro_batches[mb_id]

                if stage.is_first_stage:
                    input_ = mb
                else:
                    input_ = stage.recv_forward(
                        stage.stage_id - 1, mb.shape, mb.dtype
                    )

                output = stage.forward(input_, mb_id)
                outputs[mb_id] = output

                if not stage.is_last_stage:
                    stage.send_forward(output, stage.stage_id + 1)
                else:
                    loss = loss_fn(output)
                    losses.append(loss)

            elif op == "backward":
                if stage.is_last_stage:
                    losses[mb_id].backward()
                    grad = outputs[mb_id].grad
                else:
                    grad = stage.recv_backward(
                        stage.stage_id + 1,
                        outputs[mb_id].shape,
                        outputs[mb_id].dtype,
                    )
                    outputs[mb_id].backward(grad)

                if not stage.is_first_stage:
                    input_grad = stage._input_cache[mb_id].grad
                    stage.send_backward(input_grad, stage.stage_id - 1)

        stage.clear_cache()

        if losses:
            return sum(losses) / len(losses)
        return torch.tensor(0.0)
