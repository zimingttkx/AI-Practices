"""
Pipeline Parallelism Implementation

Core Idea:
    Pipeline parallelism partitions the model by layers across GPUs, with each
    GPU (stage) responsible for a subset of layers. Micro-batching enables
    concurrent execution across stages to improve GPU utilization.

Mathematical Theory:
    For a model with L layers split into S stages, each stage has L/S layers.
    With M micro-batches, the pipeline bubble ratio is:
    
    .. math::
        \\text{Bubble} = \\frac{S - 1}{M + S - 1}
    
    As M increases, bubble overhead approaches zero.

Problem Statement:
    Very deep models cannot fit on a single GPU. Pipeline parallelism enables
    training by distributing layers across GPUs while maintaining sequential
    dependency through micro-batch scheduling.

Comparison:
    - GPipe: All forwards then all backwards, simple but high memory
    - PipeDream (1F1B): Interleaved schedule, lower memory, same throughput
    - Interleaved: Multiple stages per GPU, reduced bubble

Complexity:
    - Communication: O(B * S * H) per micro-batch between stages
    - Memory: O(M * A) for activation storage, where A is activation size
    - Bubble: O((S-1)/M) fraction of compute wasted

References:
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


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism.
    
    Attributes:
        num_stages: Total number of pipeline stages.
        num_micro_batches: Number of micro-batches per batch.
        stage_id: Current stage ID (0 to num_stages-1).
        process_group: Distributed process group.
        chunks: Number of chunks per batch.
        checkpoint_activations: Enable activation checkpointing.
    """
    num_stages: int = 1
    num_micro_batches: int = 1
    stage_id: int = 0
    process_group: Optional[dist.ProcessGroup] = None
    chunks: int = 1
    checkpoint_activations: bool = False


class PipelineStage(nn.Module):
    """Pipeline stage wrapper for a subset of model layers.
    
    Handles communication with adjacent stages and activation caching
    for backward pass.
    
    Args:
        module: Model layers for this stage.
        stage_id: Stage index (0 to num_stages-1).
        num_stages: Total number of stages.
        config: Pipeline configuration.
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
        
        self._input_cache: Dict[int, torch.Tensor] = {}
        self._output_cache: Dict[int, torch.Tensor] = {}
    
    def forward(self, input_: torch.Tensor, micro_batch_id: int = 0) -> torch.Tensor:
        """Forward pass with activation caching."""
        if self.config.checkpoint_activations:
            self._input_cache[micro_batch_id] = input_.detach()
        
        output = self.module(input_)
        self._output_cache[micro_batch_id] = output
        
        return output
    
    def send_forward(self, tensor: torch.Tensor, dst: int) -> None:
        """Send activation to next stage."""
        if not dist.is_initialized():
            return
        dist.send(tensor, dst, group=self.config.process_group)
    
    def recv_forward(self, src: int, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Receive activation from previous stage."""
        if not dist.is_initialized():
            return torch.zeros(shape, dtype=dtype)
        
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        dist.recv(tensor, src, group=self.config.process_group)
        return tensor
    
    def send_backward(self, tensor: torch.Tensor, dst: int) -> None:
        """Send gradient to previous stage."""
        if not dist.is_initialized():
            return
        dist.send(tensor, dst, group=self.config.process_group)
    
    def recv_backward(self, src: int, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Receive gradient from next stage."""
        if not dist.is_initialized():
            return torch.zeros(shape, dtype=dtype)
        
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        dist.recv(tensor, src, group=self.config.process_group)
        return tensor
    
    def clear_cache(self) -> None:
        """Clear activation caches."""
        self._input_cache.clear()
        self._output_cache.clear()


class PipelineScheduler(ABC):
    """Abstract base class for pipeline schedulers."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.num_stages = config.num_stages
        self.num_micro_batches = config.num_micro_batches
        self.stage_id = config.stage_id
    
    @abstractmethod
    def get_schedule(self) -> List[Tuple[str, int]]:
        """Return schedule as list of (operation, micro_batch_id) tuples."""
        pass
    
    @abstractmethod
    def forward_backward(
        self,
        stage: PipelineStage,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """Execute forward and backward passes according to schedule."""
        pass


class GPipeScheduler(PipelineScheduler):
    """GPipe scheduler: all forwards then all backwards.
    
    Simple but requires storing all activations, leading to high memory usage.
    Bubble ratio: (S-1)/(M+S-1) where S=stages, M=micro-batches.
    """
    
    def get_schedule(self) -> List[Tuple[str, int]]:
        """Generate GPipe schedule."""
        schedule = []
        
        for mb in range(self.num_micro_batches):
            schedule.append(("forward", mb))
        
        for mb in reversed(range(self.num_micro_batches)):
            schedule.append(("backward", mb))
        
        return schedule
    
    def forward_backward(
        self,
        stage: PipelineStage,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """Execute GPipe forward-backward pass."""
        micro_batches = torch.chunk(batch, self.num_micro_batches, dim=0)
        outputs = []
        losses = []
        
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
    """PipeDream (1F1B) scheduler: interleaved forward and backward.
    
    Reduces memory by limiting in-flight micro-batches. After warmup,
    alternates between forward and backward passes.
    """
    
    def get_schedule(self) -> List[Tuple[str, int]]:
        """Generate 1F1B schedule."""
        schedule = []
        num_warmup = self.num_stages - self.stage_id - 1
        num_1f1b = self.num_micro_batches - num_warmup
        
        for mb in range(num_warmup):
            schedule.append(("forward", mb))
        
        for i in range(num_1f1b):
            schedule.append(("forward", num_warmup + i))
            schedule.append(("backward", i))
        
        for i in range(num_warmup):
            schedule.append(("backward", num_1f1b + i))
        
        return schedule
    
    def forward_backward(
        self,
        stage: PipelineStage,
        batch: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """Execute PipeDream 1F1B forward-backward pass."""
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
