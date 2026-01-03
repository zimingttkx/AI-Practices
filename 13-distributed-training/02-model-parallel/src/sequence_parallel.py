"""
Sequence Parallelism Implementation

Core Idea:
    Sequence parallelism partitions the sequence dimension across GPUs,
    complementing tensor parallelism by distributing LayerNorm and Dropout
    operations that would otherwise be replicated.

Mathematical Theory:
    For input X of shape [B, S, H], sequence parallelism splits along S:
    
    .. math::
        X = [X_1, X_2, ..., X_N] \\quad \\text{where } X_i \\in \\mathbb{R}^{B \\times S/N \\times H}
    
    LayerNorm and Dropout operate on local shards, reducing memory by N.
    Attention requires gathering the full sequence, then scattering results.

Problem Statement:
    In tensor parallelism, LayerNorm and Dropout are replicated across GPUs,
    wasting memory. Sequence parallelism distributes these operations by
    partitioning the sequence dimension.

Comparison:
    - vs Tensor Parallel: Sequence splits sequence dim, Tensor splits hidden dim
    - vs Ring Attention: Sequence uses AllGather, Ring uses ring communication
    - Memory savings: Reduces activation memory by factor of N

Complexity:
    - Communication: O(B * S * H) for AllGather/ReduceScatter per layer
    - Memory: O(B * S/N * H) per GPU for activations
    - Computation: Same total FLOPs, distributed across GPUs

References:
    - Korthikanti et al., "Reducing Activation Recomputation in Large
      Transformer Models", MLSys 2023
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


@dataclass
class SequenceParallelConfig:
    """Configuration for sequence parallelism.
    
    Attributes:
        world_size: Number of GPUs for sequence parallelism.
        rank: Current GPU rank.
        process_group: Distributed process group.
        sequence_dim: Dimension to partition (typically 1 for [B, S, H]).
    """
    world_size: int = 1
    rank: int = 0
    process_group: Optional[dist.ProcessGroup] = None
    sequence_dim: int = 1


def scatter_to_sequence_parallel(
    tensor: torch.Tensor,
    config: SequenceParallelConfig,
) -> torch.Tensor:
    """Scatter tensor along sequence dimension to all processes.
    
    Args:
        tensor: Input tensor [batch, seq, hidden].
        config: Sequence parallel configuration.
        
    Returns:
        Local shard [batch, seq/world_size, hidden].
    """
    if config.world_size == 1:
        return tensor
    
    seq_dim = config.sequence_dim
    seq_len = tensor.size(seq_dim)
    
    assert seq_len % config.world_size == 0, \
        f"Sequence length {seq_len} not divisible by {config.world_size}"
    
    chunk_size = seq_len // config.world_size
    start = config.rank * chunk_size
    
    return tensor.narrow(seq_dim, start, chunk_size).contiguous()


def gather_from_sequence_parallel(
    tensor: torch.Tensor,
    config: SequenceParallelConfig,
) -> torch.Tensor:
    """Gather sequence shards from all processes.
    
    Args:
        tensor: Local shard [batch, seq/world_size, hidden].
        config: Sequence parallel configuration.
        
    Returns:
        Full tensor [batch, seq, hidden].
    """
    if config.world_size == 1:
        return tensor
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(config.world_size)]
    dist.all_gather(tensor_list, tensor, group=config.process_group)
    
    return torch.cat(tensor_list, dim=config.sequence_dim)


class _ScatterToSequenceParallel(torch.autograd.Function):
    """Scatter to sequence parallel (scatter forward, gather backward)."""
    
    @staticmethod
    def forward(ctx, input_, config):
        ctx.config = config
        return scatter_to_sequence_parallel(input_, config)
    
    @staticmethod
    def backward(ctx, grad_output):
        return gather_from_sequence_parallel(grad_output, ctx.config), None


class _GatherFromSequenceParallel(torch.autograd.Function):
    """Gather from sequence parallel (gather forward, scatter backward)."""
    
    @staticmethod
    def forward(ctx, input_, config):
        ctx.config = config
        return gather_from_sequence_parallel(input_, config)
    
    @staticmethod
    def backward(ctx, grad_output):
        return scatter_to_sequence_parallel(grad_output, ctx.config), None


class _ReduceScatterToSequenceParallel(torch.autograd.Function):
    """ReduceScatter to sequence parallel."""
    
    @staticmethod
    def forward(ctx, input_, config):
        ctx.config = config
        
        if config.world_size == 1:
            return input_
        
        seq_dim = config.sequence_dim
        seq_len = input_.size(seq_dim)
        
        input_list = list(input_.chunk(config.world_size, dim=seq_dim))
        output = torch.zeros_like(input_list[0])
        
        dist.reduce_scatter(output, input_list, group=config.process_group)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.config.world_size == 1:
            return grad_output, None
        
        tensor_list = [torch.zeros_like(grad_output) for _ in range(ctx.config.world_size)]
        dist.all_gather(tensor_list, grad_output, group=ctx.config.process_group)
        return torch.cat(tensor_list, dim=ctx.config.sequence_dim), None


class SequenceParallelLayerNorm(nn.Module):
    """LayerNorm operating on sequence-parallel shards."""
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        config: Optional[SequenceParallelConfig] = None,
    ):
        super().__init__()
        self.config = config or SequenceParallelConfig()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm to local sequence shard."""
        return F.layer_norm(
            input_, (self.normalized_shape,),
            self.weight, self.bias, self.eps
        )


class SequenceParallelDropout(nn.Module):
    """Dropout operating on sequence-parallel shards."""
    
    def __init__(
        self,
        p: float = 0.1,
        config: Optional[SequenceParallelConfig] = None,
    ):
        super().__init__()
        self.config = config or SequenceParallelConfig()
        self.p = p
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Apply dropout to local sequence shard."""
        return F.dropout(input_, self.p, self.training)


class SequenceParallelAttention(nn.Module):
    """Attention layer with sequence parallelism support.
    
    Gathers full sequence for attention computation, then scatters results
    back to sequence-parallel format.
    
    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        config: Sequence parallel configuration.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        config: Optional[SequenceParallelConfig] = None,
    ):
        super().__init__()
        
        self.config = config or SequenceParallelConfig()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        
        assert hidden_size % num_heads == 0
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with sequence gather/scatter for attention."""
        batch_size, seq_len, _ = hidden_states.shape
        
        if self.config.world_size > 1:
            hidden_states_full = _GatherFromSequenceParallel.apply(
                hidden_states, self.config
            )
        else:
            hidden_states_full = hidden_states
        
        full_seq_len = hidden_states_full.size(1)
        
        qkv = self.qkv_proj(hidden_states_full)
        qkv = qkv.reshape(batch_size, full_seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, full_seq_len, self.hidden_size
        )
        
        output = self.out_proj(attn_output)
        
        if self.config.world_size > 1:
            output = _ScatterToSequenceParallel.apply(output, self.config)
        
        return output
