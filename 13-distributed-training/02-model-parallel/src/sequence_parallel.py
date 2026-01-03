"""
Sequence Parallelism 序列并行实现

================================================================================
核心思想 (一句话理解)
================================================================================
序列并行 = 沿序列维度切分激活值 + 与张量并行配合 + 减少LayerNorm/Dropout的显存占用

================================================================================
为什么需要序列并行？(问题背景)
================================================================================

    张量并行的问题:
    ┌─────────────────────────────────────────────────────────────────┐
    │  在张量并行中，LayerNorm和Dropout在每个GPU上重复计算             │
    │                                                                 │
    │  GPU0: [LayerNorm] → [Attention(TP)] → [LayerNorm] → [MLP(TP)] │
    │  GPU1: [LayerNorm] → [Attention(TP)] → [LayerNorm] → [MLP(TP)] │
    │         ↑ 重复!                         ↑ 重复!                 │
    │                                                                 │
    │  问题: LayerNorm/Dropout的激活值在每个GPU上都完整存储            │
    │  浪费: N个GPU，激活值存了N份                                     │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
工作原理 (图解)
================================================================================

    序列并行的解决方案:
    ┌─────────────────────────────────────────────────────────────────┐
    │  输入: X [batch, seq, hidden]                                   │
    │                                                                 │
    │  序列切分:                                                       │
    │  GPU0: X[:, 0:seq/2, :]     GPU1: X[:, seq/2:seq, :]           │
    │                                                                 │
    │  LayerNorm/Dropout: 各自计算自己的序列分片                       │
    │  GPU0: LN(X[:, 0:seq/2, :])                                     │
    │  GPU1: LN(X[:, seq/2:seq, :])                                   │
    │                                                                 │
    │  Attention: 需要完整序列 → AllGather收集                         │
    │  GPU0+GPU1: AllGather → 完整序列 → Attention → ReduceScatter    │
    │                                                                 │
    │  显存节省: LayerNorm/Dropout激活值减少N倍!                       │
    └─────────────────────────────────────────────────────────────────┘

    完整流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  [序列分片] → [LayerNorm] → [AllGather] → [Attention]           │
    │      ↑           ↑             ↑              ↓                 │
    │   seq/N       seq/N          seq           seq                  │
    │                                              ↓                  │
    │  [序列分片] ← [LayerNorm] ← [ReduceScatter] ←                   │
    │      ↓           ↓             ↓                                │
    │   seq/N       seq/N          seq                                │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
通信模式
================================================================================
    - AllGather: 收集所有GPU的序列分片，用于Attention前
    - ReduceScatter: 分发结果并求和，用于Attention后
    - 通信量: O(batch × seq × hidden) per layer

================================================================================
显存分析
================================================================================
    假设: batch=B, seq=S, hidden=H, GPU数=N

    不使用序列并行:
    - LayerNorm激活值: B × S × H (每个GPU都存完整的)
    - 总显存: N × B × S × H

    使用序列并行:
    - LayerNorm激活值: B × (S/N) × H (每个GPU只存1/N)
    - 总显存: B × S × H (减少N倍!)

================================================================================
前置知识
================================================================================
- 张量并行的基本概念
- AllGather/ReduceScatter通信原语
- Transformer的LayerNorm和Attention结构

================================================================================
参考文献
================================================================================
- Korthikanti et al., "Reducing Activation Recomputation in Large
  Transformer Models", MLSys 2023
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class SequenceParallelConfig:
    """序列并行配置

    Attributes:
        world_size: 序列并行的GPU数量
        rank: 当前GPU在序列并行组中的编号 (0 到 world_size-1)
        process_group: 分布式进程组，用于通信
        sequence_dim: 序列维度 (通常是1，对应 [batch, seq, hidden] 中的seq)

    Example:
        >>> config = SequenceParallelConfig(
        ...     world_size=2,  # 2个GPU做序列并行
        ...     rank=0,        # 当前是第0个GPU
        ...     sequence_dim=1,  # 沿第1维(序列维度)切分
        ... )
    """
    world_size: int = 1
    rank: int = 0
    process_group: Optional[dist.ProcessGroup] = None
    sequence_dim: int = 1


# =============================================================================
# 辅助函数
# =============================================================================

def scatter_to_sequence_parallel(
    tensor: torch.Tensor,
    config: SequenceParallelConfig,
) -> torch.Tensor:
    """将张量沿序列维度切分，返回当前GPU的分片

    这是序列并行的核心操作之一：把完整序列切分到各个GPU。

    工作原理:
        输入: [batch, seq, hidden]
        输出: [batch, seq/world_size, hidden]

        GPU0负责: tensor[:, 0:seq/N, :]
        GPU1负责: tensor[:, seq/N:2*seq/N, :]
        ...

    Args:
        tensor: 输入张量 [batch, seq, hidden]
        config: 序列并行配置

    Returns:
        当前GPU负责的序列分片 [batch, seq/world_size, hidden]

    Example:
        >>> x = torch.randn(2, 1024, 4096)  # [batch, seq, hidden]
        >>> x_local = scatter_to_sequence_parallel(x, config)
        >>> # 如果world_size=2，x_local.shape = [2, 512, 4096]
    """
    # 单GPU不需要切分
    if config.world_size == 1:
        return tensor

    seq_dim = config.sequence_dim
    seq_len = tensor.size(seq_dim)

    # 确保序列长度能被GPU数整除
    assert seq_len % config.world_size == 0, \
        f"序列长度{seq_len}必须能被{config.world_size}整除"

    # 计算当前GPU负责的范围
    chunk_size = seq_len // config.world_size
    start = config.rank * chunk_size

    # narrow: 从start开始取chunk_size个元素
    return tensor.narrow(seq_dim, start, chunk_size).contiguous()


def gather_from_sequence_parallel(
    tensor: torch.Tensor,
    config: SequenceParallelConfig,
) -> torch.Tensor:
    """收集所有GPU的序列分片，拼接成完整序列

    这是序列并行的核心操作之一：把各GPU的分片收集起来。

    工作原理:
        输入: [batch, seq/world_size, hidden] (每个GPU的分片)
        输出: [batch, seq, hidden] (完整序列)

        AllGather收集所有GPU的分片，然后沿序列维度拼接

    Args:
        tensor: 当前GPU的序列分片 [batch, seq/world_size, hidden]
        config: 序列并行配置

    Returns:
        拼接后的完整张量 [batch, seq, hidden]

    Example:
        >>> x_local = torch.randn(2, 512, 4096)  # 当前GPU的分片
        >>> x_full = gather_from_sequence_parallel(x_local, config)
        >>> # 如果world_size=2，x_full.shape = [2, 1024, 4096]
    """
    # 单GPU不需要收集
    if config.world_size == 1:
        return tensor

    # 创建接收缓冲区
    tensor_list = [torch.zeros_like(tensor) for _ in range(config.world_size)]
    # AllGather: 收集所有GPU的张量
    dist.all_gather(tensor_list, tensor, group=config.process_group)
    # 沿序列维度拼接
    return torch.cat(tensor_list, dim=config.sequence_dim)


# =============================================================================
# 自动求导函数 (处理前向/反向传播中的通信)
# =============================================================================

class _ScatterToSequenceParallel(torch.autograd.Function):
    """切分到序列并行

    前向: Scatter (切分序列)
    反向: AllGather (收集梯度)

    为什么反向是AllGather?
    - 前向时，完整序列被切分到各GPU
    - 反向时，每个GPU只有自己分片的梯度
    - 需要AllGather收集所有梯度，才能计算完整的输入梯度
    """

    @staticmethod
    def forward(ctx, input_, config):
        ctx.config = config
        return scatter_to_sequence_parallel(input_, config)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向时AllGather收集所有GPU的梯度
        return gather_from_sequence_parallel(grad_output, ctx.config), None


class _GatherFromSequenceParallel(torch.autograd.Function):
    """从序列并行收集

    前向: AllGather (收集序列)
    反向: Scatter (切分梯度)

    为什么反向是Scatter?
    - 前向时，各GPU的分片被收集成完整序列
    - 反向时，完整序列的梯度需要切分回各GPU
    - 每个GPU只需要自己分片对应的梯度
    """

    @staticmethod
    def forward(ctx, input_, config):
        ctx.config = config
        return gather_from_sequence_parallel(input_, config)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向时Scatter切分梯度
        return scatter_to_sequence_parallel(grad_output, ctx.config), None


class _ReduceScatterToSequenceParallel(torch.autograd.Function):
    """ReduceScatter到序列并行

    前向: ReduceScatter (先求和再切分)
    反向: AllGather (收集梯度)

    ReduceScatter vs Scatter:
    - Scatter: 直接切分，每个GPU拿一块
    - ReduceScatter: 先对所有GPU的数据求和，再切分

    使用场景:
    - Attention输出后，需要把各GPU的结果汇总并切分回序列并行格式
    """

    @staticmethod
    def forward(ctx, input_, config):
        ctx.config = config

        if config.world_size == 1:
            return input_

        seq_dim = config.sequence_dim

        # 将输入按序列维度切分成world_size块
        input_list = list(input_.chunk(config.world_size, dim=seq_dim))
        output = torch.zeros_like(input_list[0])

        # ReduceScatter: 每个GPU得到所有GPU对应块的求和结果
        dist.reduce_scatter(output, input_list, group=config.process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.config.world_size == 1:
            return grad_output, None

        # 反向时AllGather收集所有GPU的梯度
        tensor_list = [torch.zeros_like(grad_output) for _ in range(ctx.config.world_size)]
        dist.all_gather(tensor_list, grad_output, group=ctx.config.process_group)
        return torch.cat(tensor_list, dim=ctx.config.sequence_dim), None


# =============================================================================
# 序列并行LayerNorm
# =============================================================================

class SequenceParallelLayerNorm(nn.Module):
    """序列并行LayerNorm

    在序列分片上执行LayerNorm，每个GPU只处理自己的序列部分。

    关键点:
        - LayerNorm是沿hidden维度归一化，不是沿序列维度
        - 所以可以直接在序列分片上计算，无需通信
        - 这就是序列并行能节省显存的原因!

    显存节省:
        - 不使用序列并行: 每个GPU存储 [batch, seq, hidden] 的激活值
        - 使用序列并行: 每个GPU存储 [batch, seq/N, hidden] 的激活值
        - 节省N倍显存!

    Args:
        normalized_shape: 归一化的维度大小 (通常是hidden_size)
        eps: 数值稳定性的小常数
        config: 序列并行配置

    Example:
        >>> ln = SequenceParallelLayerNorm(4096, config=config)
        >>> x = torch.randn(2, 512, 4096)  # 序列分片
        >>> y = ln(x)  # 输出形状不变
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

        # LayerNorm的可学习参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """在序列分片上执行LayerNorm

        Args:
            input_: 序列分片 [batch, seq/world_size, hidden]

        Returns:
            归一化后的张量，形状不变
        """
        # 直接在分片上计算LayerNorm，无需通信
        return F.layer_norm(
            input_, (self.normalized_shape,),
            self.weight, self.bias, self.eps
        )


# =============================================================================
# 序列并行Dropout
# =============================================================================

class SequenceParallelDropout(nn.Module):
    """序列并行Dropout

    在序列分片上执行Dropout，每个GPU独立处理自己的部分。

    关键点:
        - Dropout是逐元素操作，可以直接在分片上计算
        - 无需通信，显存节省N倍

    Args:
        p: Dropout概率
        config: 序列并行配置

    Example:
        >>> dropout = SequenceParallelDropout(0.1, config=config)
        >>> x = torch.randn(2, 512, 4096)  # 序列分片
        >>> y = dropout(x)  # 输出形状不变
    """

    def __init__(
        self,
        p: float = 0.1,
        config: Optional[SequenceParallelConfig] = None,
    ):
        super().__init__()
        self.config = config or SequenceParallelConfig()
        self.p = p

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """在序列分片上执行Dropout

        Args:
            input_: 序列分片 [batch, seq/world_size, hidden]

        Returns:
            Dropout后的张量，形状不变
        """
        return F.dropout(input_, self.p, self.training)


# =============================================================================
# 序列并行Attention
# =============================================================================

class SequenceParallelAttention(nn.Module):
    """序列并行Attention

    Attention需要完整序列来计算注意力分数，所以需要:
    1. AllGather收集完整序列
    2. 计算Attention
    3. Scatter切分结果回序列并行格式

    工作流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │  输入: [batch, seq/N, hidden] (序列分片)                         │
    │                    ↓                                            │
    │  AllGather: [batch, seq, hidden] (完整序列)                      │
    │                    ↓                                            │
    │  QKV投影 + Attention计算                                         │
    │                    ↓                                            │
    │  Output投影: [batch, seq, hidden]                                │
    │                    ↓                                            │
    │  Scatter: [batch, seq/N, hidden] (序列分片)                      │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        dropout: Dropout概率
        config: 序列并行配置

    Example:
        >>> attn = SequenceParallelAttention(4096, 32, config=config)
        >>> x = torch.randn(2, 512, 4096)  # 序列分片
        >>> y = attn(x)  # 输出也是序列分片
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

        assert hidden_size % num_heads == 0, \
            f"hidden_size({hidden_size})必须能被num_heads({num_heads})整除"

        # QKV投影 (合并成一个Linear提高效率)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            hidden_states: 序列分片 [batch, seq/world_size, hidden]
            attention_mask: 注意力掩码 (可选)

        Returns:
            输出序列分片 [batch, seq/world_size, hidden]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ===== 1. AllGather收集完整序列 =====
        if self.config.world_size > 1:
            hidden_states_full = _GatherFromSequenceParallel.apply(
                hidden_states, self.config
            )
        else:
            hidden_states_full = hidden_states

        full_seq_len = hidden_states_full.size(1)

        # ===== 2. QKV投影 =====
        qkv = self.qkv_proj(hidden_states_full)
        # 重塑为 [batch, seq, 3, num_heads, head_dim]
        qkv = qkv.reshape(batch_size, full_seq_len, 3, self.num_heads, self.head_dim)
        # 转置为 [3, batch, num_heads, seq, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ===== 3. 计算注意力分数 =====
        scale = self.head_dim ** -0.5
        # [batch, num_heads, seq, seq]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 应用注意力掩码 (如果有)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax + Dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # ===== 4. 计算注意力输出 =====
        # [batch, num_heads, seq, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        # 转置并重塑为 [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, full_seq_len, self.hidden_size
        )

        # ===== 5. 输出投影 =====
        output = self.out_proj(attn_output)

        # ===== 6. Scatter切分回序列并行格式 =====
        if self.config.world_size > 1:
            output = _ScatterToSequenceParallel.apply(output, self.config)

        return output
