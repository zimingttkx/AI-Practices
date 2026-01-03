"""
序列并行 (Sequence Parallelism) 实现

序列并行将序列维度分割到多个 GPU，与张量并行配合使用，
进一步减少激活内存占用。

核心概念:
    - 在非张量并行区域按序列维度分片
    - LayerNorm 和 Dropout 在分片序列上执行
    - 与张量并行的 AllReduce 结合
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


@dataclass
class SequenceParallelConfig:
    """序列并行配置
    
    Attributes:
        world_size: 并行大小
        rank: 当前进程排名
        process_group: 进程组
        sequence_dim: 序列维度
    """
    world_size: int = 1
    rank: int = 0
    process_group: Optional[dist.ProcessGroup] = None
    sequence_dim: int = 1  # 通常是 [batch, seq, hidden] 中的 seq


def scatter_to_sequence_parallel(
    tensor: torch.Tensor,
    config: SequenceParallelConfig,
) -> torch.Tensor:
    """将张量按序列维度分散到各进程
    
    Args:
        tensor: 输入张量 [batch, seq, hidden]
        config: 序列并行配置
        
    Returns:
        分片后的张量 [batch, seq/world_size, hidden]
    """
    if config.world_size == 1:
        return tensor
    
    seq_dim = config.sequence_dim
    seq_len = tensor.size(seq_dim)
    
    assert seq_len % config.world_size == 0, \
        f"Sequence length {seq_len} not divisible by {config.world_size}"
    
    chunk_size = seq_len // config.world_size
    start = config.rank * chunk_size
    end = start + chunk_size
    
    return tensor.narrow(seq_dim, start, chunk_size).contiguous()


def gather_from_sequence_parallel(
    tensor: torch.Tensor,
    config: SequenceParallelConfig,
) -> torch.Tensor:
    """从各进程收集序列分片
    
    Args:
        tensor: 本地分片 [batch, seq/world_size, hidden]
        config: 序列并行配置
        
    Returns:
        完整张量 [batch, seq, hidden]
    """
    if config.world_size == 1:
        return tensor
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(config.world_size)]
    dist.all_gather(tensor_list, tensor, group=config.process_group)
    
    return torch.cat(tensor_list, dim=config.sequence_dim)


class _ScatterToSequenceParallel(torch.autograd.Function):
    """分散到序列并行（前向 scatter，反向 gather）"""
    
    @staticmethod
    def forward(ctx, input_, config):
        ctx.config = config
        return scatter_to_sequence_parallel(input_, config)
    
    @staticmethod
    def backward(ctx, grad_output):
        return gather_from_sequence_parallel(grad_output, ctx.config), None


class _GatherFromSequenceParallel(torch.autograd.Function):
    """从序列并行收集（前向 gather，反向 scatter）"""
    
    @staticmethod
    def forward(ctx, input_, config):
        ctx.config = config
        return gather_from_sequence_parallel(input_, config)
    
    @staticmethod
    def backward(ctx, grad_output):
        return scatter_to_sequence_parallel(grad_output, ctx.config), None


class _ReduceScatterToSequenceParallel(torch.autograd.Function):
    """ReduceScatter 到序列并行"""
    
    @staticmethod
    def forward(ctx, input_, config):
        ctx.config = config
        
        if config.world_size == 1:
            return input_
        
        seq_dim = config.sequence_dim
        seq_len = input_.size(seq_dim)
        chunk_size = seq_len // config.world_size
        
        # 分割输入
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
    """序列并行 LayerNorm
    
    在分片的序列上执行 LayerNorm。
    """
    
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
        return F.layer_norm(
            input_, (self.normalized_shape,),
            self.weight, self.bias, self.eps
        )


class SequenceParallelDropout(nn.Module):
    """序列并行 Dropout"""
    
    def __init__(
        self,
        p: float = 0.1,
        config: Optional[SequenceParallelConfig] = None,
    ):
        super().__init__()
        self.config = config or SequenceParallelConfig()
        self.p = p
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return F.dropout(input_, self.p, self.training)


class SequenceParallelAttention(nn.Module):
    """序列并行注意力层
    
    结合张量并行和序列并行的注意力实现。
    
    Args:
        hidden_size: 隐藏层大小
        num_heads: 注意力头数
        dropout: Dropout 概率
        config: 序列并行配置
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
        
        # QKV 投影
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden] (序列已分片)
            attention_mask: 注意力掩码
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 收集完整序列用于注意力计算
        if self.config.world_size > 1:
            hidden_states_full = _GatherFromSequenceParallel.apply(
                hidden_states, self.config
            )
        else:
            hidden_states_full = hidden_states
        
        full_seq_len = hidden_states_full.size(1)
        
        # QKV 投影
        qkv = self.qkv_proj(hidden_states_full)
        qkv = qkv.reshape(batch_size, full_seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, full_seq_len, self.hidden_size
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        # 分散回序列并行
        if self.config.world_size > 1:
            output = _ScatterToSequenceParallel.apply(output, self.config)
        
        return output
