"""
流水线并行 (Pipeline Parallelism) 实现

流水线并行将模型按层分割到多个 GPU，通过微批次流水线化执行
来提高 GPU 利用率。

核心概念:
    - 阶段 (Stage): 模型的一部分层
    - 微批次 (Micro-batch): 将批次分割为更小的单元
    - 调度策略: GPipe, PipeDream 等
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class PipelineConfig:
    """流水线并行配置
    
    Attributes:
        num_stages: 流水线阶段数
        num_micro_batches: 微批次数量
        stage_id: 当前阶段 ID
        process_group: 进程组
        chunks: 每个批次的块数
        checkpoint_activations: 是否检查点激活
    """
    num_stages: int = 1
    num_micro_batches: int = 1
    stage_id: int = 0
    process_group: Optional[dist.ProcessGroup] = None
    chunks: int = 1
    checkpoint_activations: bool = False


class PipelineStage(nn.Module):
    """流水线阶段
    
    封装模型的一部分层，处理与相邻阶段的通信。
    
    Args:
        module: 本阶段的模型层
        stage_id: 阶段 ID
        num_stages: 总阶段数
        config: 流水线配置
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
        
        self.is_first_stage = stage_id == 0
        self.is_last_stage = stage_id == num_stages - 1
        
        # 激活缓存
        self._input_cache: Dict[int, torch.Tensor] = {}
        self._output_cache: Dict[int, torch.Tensor] = {}
    
    def forward(self, input_: torch.Tensor, micro_batch_id: int = 0) -> torch.Tensor:
        """前向传播
        
        Args:
            input_: 输入张量
            micro_batch_id: 微批次 ID
            
        Returns:
            输出张量
        """
        # 缓存输入用于反向传播
        if self.config.checkpoint_activations:
            self._input_cache[micro_batch_id] = input_.detach()
        
        output = self.module(input_)
        
        # 缓存输出
        self._output_cache[micro_batch_id] = output
        
        return output
    
    def send_forward(self, tensor: torch.Tensor, dst: int) -> None:
        """发送张量到下一阶段"""
        if not dist.is_initialized():
            return
        dist.send(tensor, dst, group=self.config.process_group)
    
    def recv_forward(self, src: int, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """从上一阶段接收张量"""
        if not dist.is_initialized():
            return torch.zeros(shape, dtype=dtype)
        
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        dist.recv(tensor, src, group=self.config.process_group)
        return tensor
    
    def send_backward(self, tensor: torch.Tensor, dst: int) -> None:
        """发送梯度到上一阶段"""
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
        """清除激活缓存"""
        self._input_cache.clear()
        self._output_cache.clear()


class PipelineScheduler(ABC):
    """流水线调度器基类"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.num_stages = config.num_stages
        self.num_micro_batches = config.num_micro_batches
        self.stage_id = config.stage_id
    
    @abstractmethod
    def get_schedule(self) -> List[Tuple[str, int]]:
        """获取调度计划
        
        Returns:
            调度计划列表，每项为 (操作类型, 微批次ID)
        """
        pass
    
    @abstractmethod
    def forward_backward(
        self,
        stage: PipelineStage,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """执行前向和反向传播"""
        pass


class GPipeScheduler(PipelineScheduler):
    """GPipe 调度器
    
    GPipe 策略：先完成所有微批次的前向传播，再进行反向传播。
    简单但有较大的内存占用（需要存储所有激活）。
    """
    
    def get_schedule(self) -> List[Tuple[str, int]]:
        schedule = []
        
        # 所有前向传播
        for mb in range(self.num_micro_batches):
            schedule.append(("forward", mb))
        
        # 所有反向传播（逆序）
        for mb in reversed(range(self.num_micro_batches)):
            schedule.append(("backward", mb))
        
        return schedule
    
    def forward_backward(
        self,
        stage: PipelineStage,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """GPipe 前向反向传播"""
        micro_batches = torch.chunk(batch, self.num_micro_batches, dim=0)
        outputs = []
        losses = []
        
        # 前向传播
        for mb_id, mb in enumerate(micro_batches):
            if stage.is_first_stage:
                input_ = mb
            else:
                input_ = stage.recv_forward(
                    stage.stage_id - 1,
                    mb.shape,
                    mb.dtype,
                )
            
            output = stage.forward(input_, mb_id)
            outputs.append(output)
            
            if not stage.is_last_stage:
                stage.send_forward(output, stage.stage_id + 1)
            else:
                loss = loss_fn(output)
                losses.append(loss)
        
        # 反向传播
        for mb_id in reversed(range(self.num_micro_batches)):
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


class PipeDreamScheduler(PipelineScheduler):
    """PipeDream (1F1B) 调度器
    
    1F1B 策略：交替执行前向和反向传播，减少内存占用。
    """
    
    def get_schedule(self) -> List[Tuple[str, int]]:
        schedule = []
        num_warmup = self.num_stages - self.stage_id - 1
        num_1f1b = self.num_micro_batches - num_warmup
        
        # 预热阶段：只有前向
        for mb in range(num_warmup):
            schedule.append(("forward", mb))
        
        # 1F1B 阶段
        for i in range(num_1f1b):
            schedule.append(("forward", num_warmup + i))
            schedule.append(("backward", i))
        
        # 冷却阶段：只有反向
        for i in range(num_warmup):
            schedule.append(("backward", num_1f1b + i))
        
        return schedule
    
    def forward_backward(
        self,
        stage: PipelineStage,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """PipeDream 1F1B 前向反向传播"""
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
