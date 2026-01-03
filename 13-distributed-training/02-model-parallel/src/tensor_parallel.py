"""
Tensor Parallelism Implementation

Core Idea:
    Tensor parallelism partitions individual layers across GPUs, with each GPU
    computing a portion of the layer output. This enables training layers that
    are too large for a single GPU's memory.

Mathematical Theory:
    For a linear layer Y = XW + b, tensor parallelism splits W along columns or rows:
    
    Column Parallel (split output features):
    .. math::
        W = [W_1, W_2, ..., W_N], \\quad Y_i = XW_i
    
    Row Parallel (split input features):
    .. math::
        W = \\begin{bmatrix} W_1 \\\\ W_2 \\\\ ... \\\\ W_N \\end{bmatrix}, 
        \\quad Y = \\sum_{i=1}^{N} X_i W_i

Problem Statement:
    Large transformer models have attention and MLP layers with billions of
    parameters. Tensor parallelism distributes these layers across GPUs,
    reducing per-GPU memory while maintaining model semantics.

Comparison:
    - vs Pipeline: Tensor splits layers, Pipeline splits model depth
    - vs FSDP: Tensor keeps activations local, FSDP shards everything
    - vs Sequence: Tensor splits features, Sequence splits sequence length

Complexity:
    - Column Parallel: AllGather in forward, ReduceScatter in backward
    - Row Parallel: ReduceScatter in forward, AllGather in backward
    - Communication: O(B * S * H / N) per layer, where H is hidden size

References:
    - Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language
      Models Using Model Parallelism", arXiv 2019
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism.
    
    Attributes:
        world_size: Number of GPUs for tensor parallelism.
        rank: Current GPU rank in tensor parallel group.
        process_group: Distributed process group.
        sequence_parallel: Enable sequence parallelism integration.
        async_tensor_parallel: Enable asynchronous communication.
    """
    world_size: int = 1
    rank: int = 0
    process_group: Optional[dist.ProcessGroup] = None
    sequence_parallel: bool = False
    async_tensor_parallel: bool = False


def _get_tensor_parallel_world_size(config: Optional[TensorParallelConfig] = None) -> int:
    """Return tensor parallel world size."""
    if config is not None:
        return config.world_size
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _get_tensor_parallel_rank(config: Optional[TensorParallelConfig] = None) -> int:
    """Return tensor parallel rank."""
    if config is not None:
        return config.rank
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def tensor_parallel_split(
    tensor: torch.Tensor,
    dim: int = -1,
    config: Optional[TensorParallelConfig] = None,
) -> torch.Tensor:
    """Split tensor along specified dimension for tensor parallelism.
    
    Args:
        tensor: Input tensor to split.
        dim: Dimension to split along.
        config: Tensor parallel configuration.
        
    Returns:
        Local partition of the tensor.
    """
    world_size = _get_tensor_parallel_world_size(config)
    rank = _get_tensor_parallel_rank(config)
    
    if world_size == 1:
        return tensor
    
    dim_size = tensor.size(dim)
    assert dim_size % world_size == 0, f"Dimension {dim} size {dim_size} not divisible by {world_size}"
    
    chunk_size = dim_size // world_size
    return tensor.narrow(dim, rank * chunk_size, chunk_size).contiguous()


def tensor_parallel_gather(
    tensor: torch.Tensor,
    dim: int = -1,
    config: Optional[TensorParallelConfig] = None,
) -> torch.Tensor:
    """Gather tensor partitions from all ranks.
    
    Args:
        tensor: Local tensor partition.
        dim: Dimension to concatenate along.
        config: Tensor parallel configuration.
        
    Returns:
        Full tensor gathered from all ranks.
    """
    world_size = _get_tensor_parallel_world_size(config)
    
    if world_size == 1:
        return tensor
    
    process_group = config.process_group if config else None
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=process_group)
    
    return torch.cat(tensor_list, dim=dim)


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Copy input to model parallel region (identity forward, AllReduce backward)."""
    
    @staticmethod
    def forward(ctx, input_, process_group):
        ctx.process_group = process_group
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized():
            dist.all_reduce(grad_output, group=ctx.process_group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """Reduce from model parallel region (AllReduce forward, identity backward)."""
    
    @staticmethod
    def forward(ctx, input_, process_group):
        if dist.is_initialized():
            dist.all_reduce(input_, group=process_group)
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather from model parallel region (AllGather forward, split backward)."""
    
    @staticmethod
    def forward(ctx, input_, dim, process_group, world_size, rank):
        ctx.dim = dim
        ctx.process_group = process_group
        ctx.world_size = world_size
        ctx.rank = rank
        
        if world_size == 1:
            return input_
        
        tensor_list = [torch.zeros_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_, group=process_group)
        return torch.cat(tensor_list, dim=dim)
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.world_size == 1:
            return grad_output, None, None, None, None
        
        dim_size = grad_output.size(ctx.dim)
        chunk_size = dim_size // ctx.world_size
        return grad_output.narrow(ctx.dim, ctx.rank * chunk_size, chunk_size).contiguous(), None, None, None, None


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer.
    
    Splits the weight matrix along the output dimension (columns).
    Each GPU computes a portion of the output features.
    
    Mathematical Formulation:
        Y = XW where W is split as [W_1, W_2, ..., W_N]
        Each GPU i computes Y_i = X @ W_i
        Output is gathered if gather_output=True
    
    Args:
        in_features: Size of input features.
        out_features: Total size of output features.
        bias: Whether to include bias.
        gather_output: Whether to gather output from all GPUs.
        config: Tensor parallel configuration.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        config: Optional[TensorParallelConfig] = None,
    ):
        super().__init__()
        
        self.config = config or TensorParallelConfig()
        self.world_size = self.config.world_size
        self.rank = self.config.rank
        
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        assert out_features % self.world_size == 0
        self.out_features_per_partition = out_features // self.world_size
        
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional output gathering."""
        input_parallel = _CopyToModelParallelRegion.apply(
            input_, self.config.process_group
        )
        
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        
        if self.gather_output:
            output = _GatherFromModelParallelRegion.apply(
                output_parallel, -1, self.config.process_group,
                self.world_size, self.rank
            )
        else:
            output = output_parallel
        
        return output


class RowParallelLinear(nn.Module):
    """Row-parallel linear layer.
    
    Splits the weight matrix along the input dimension (rows).
    Each GPU processes a portion of the input features.
    
    Mathematical Formulation:
        Y = XW where W is split row-wise and X is split column-wise
        Each GPU i computes Y_i = X_i @ W_i
        Output is reduced across all GPUs
    
    Args:
        in_features: Total size of input features.
        out_features: Size of output features.
        bias: Whether to include bias.
        input_is_parallel: Whether input is already partitioned.
        config: Tensor parallel configuration.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        config: Optional[TensorParallelConfig] = None,
    ):
        super().__init__()
        
        self.config = config or TensorParallelConfig()
        self.world_size = self.config.world_size
        self.rank = self.config.rank
        
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        
        assert in_features % self.world_size == 0
        self.in_features_per_partition = in_features // self.world_size
        
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with AllReduce for output aggregation."""
        if not self.input_is_parallel:
            input_parallel = tensor_parallel_split(input_, dim=-1, config=self.config)
        else:
            input_parallel = input_
        
        output_parallel = F.linear(input_parallel, self.weight)
        
        output = _ReduceFromModelParallelRegion.apply(
            output_parallel, self.config.process_group
        )
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class VocabParallelEmbedding(nn.Module):
    """Vocabulary-parallel embedding layer.
    
    Partitions the embedding table across GPUs, with each GPU storing
    a subset of the vocabulary embeddings.
    
    Mathematical Formulation:
        E is split row-wise: E = [E_1; E_2; ...; E_N]
        Each GPU i stores embeddings for vocab range [start_i, end_i)
        Lookup results are reduced across all GPUs
    
    Args:
        num_embeddings: Total vocabulary size.
        embedding_dim: Embedding dimension.
        padding_idx: Index for padding token.
        config: Tensor parallel configuration.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        config: Optional[TensorParallelConfig] = None,
    ):
        super().__init__()
        
        self.config = config or TensorParallelConfig()
        self.world_size = self.config.world_size
        self.rank = self.config.rank
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        self.vocab_start_idx = self.rank * (num_embeddings // self.world_size)
        self.vocab_end_idx = (self.rank + 1) * (num_embeddings // self.world_size)
        self.num_embeddings_per_partition = self.vocab_end_idx - self.vocab_start_idx
        
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize embedding weights."""
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            if self.vocab_start_idx <= self.padding_idx < self.vocab_end_idx:
                local_idx = self.padding_idx - self.vocab_start_idx
                with torch.no_grad():
                    self.weight[local_idx].fill_(0)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass with masked embedding lookup and reduction."""
        input_mask = (input_ >= self.vocab_start_idx) & (input_ < self.vocab_end_idx)
        
        masked_input = input_ - self.vocab_start_idx
        masked_input = masked_input.clamp(0, self.num_embeddings_per_partition - 1)
        
        output_parallel = F.embedding(masked_input, self.weight)
        output_parallel = output_parallel * input_mask.unsqueeze(-1).float()
        
        output = _ReduceFromModelParallelRegion.apply(
            output_parallel, self.config.process_group
        )
        
        return output
