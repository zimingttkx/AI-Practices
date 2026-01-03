"""
张量并行 (Tensor Parallelism) 实现

张量并行将单个层的参数分片到多个 GPU 上，每个 GPU 计算部分结果，
然后通过通信操作合并。主要用于大型 Transformer 模型。

核心概念:
    - 列并行: 按列分割权重矩阵
    - 行并行: 按行分割权重矩阵
    - 词表并行: 分割嵌入层
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


@dataclass
class TensorParallelConfig:
    """张量并行配置
    
    Attributes:
        world_size: 张量并行大小
        rank: 当前进程排名
        process_group: 进程组
        sequence_parallel: 是否启用序列并行
        async_tensor_parallel: 是否异步通信
    """
    world_size: int = 1
    rank: int = 0
    process_group: Optional[dist.ProcessGroup] = None
    sequence_parallel: bool = False
    async_tensor_parallel: bool = False


def _get_tensor_parallel_world_size(config: Optional[TensorParallelConfig] = None) -> int:
    """获取张量并行世界大小"""
    if config is not None:
        return config.world_size
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _get_tensor_parallel_rank(config: Optional[TensorParallelConfig] = None) -> int:
    """获取张量并行排名"""
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
    """将张量按指定维度分割
    
    Args:
        tensor: 输入张量
        dim: 分割维度
        config: 张量并行配置
        
    Returns:
        分割后的本地张量
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
    """收集所有分片并拼接
    
    Args:
        tensor: 本地张量
        dim: 拼接维度
        config: 张量并行配置
        
    Returns:
        完整张量
    """
    world_size = _get_tensor_parallel_world_size(config)
    
    if world_size == 1:
        return tensor
    
    process_group = config.process_group if config else None
    
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=process_group)
    
    return torch.cat(tensor_list, dim=dim)


class _CopyToModelParallelRegion(torch.autograd.Function):
    """复制到模型并行区域（前向无操作，反向 AllReduce）"""
    
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
    """从模型并行区域归约（前向 AllReduce，反向无操作）"""
    
    @staticmethod
    def forward(ctx, input_, process_group):
        if dist.is_initialized():
            dist.all_reduce(input_, group=process_group)
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """从模型并行区域收集（前向 AllGather，反向 Split）"""
    
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
    """列并行线性层
    
    将权重矩阵按列分割，每个 GPU 计算部分输出特征。
    Y = XA，其中 A 按列分割为 [A1, A2, ...]
    
    Args:
        in_features: 输入特征数
        out_features: 输出特征数（总数）
        bias: 是否使用偏置
        gather_output: 是否收集输出
        config: 张量并行配置
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
        
        # 每个 GPU 的输出特征数
        assert out_features % self.world_size == 0
        self.out_features_per_partition = out_features // self.world_size
        
        # 初始化权重
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # 复制输入到并行区域
        input_parallel = _CopyToModelParallelRegion.apply(
            input_, self.config.process_group
        )
        
        # 本地线性计算
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        
        # 收集输出
        if self.gather_output:
            output = _GatherFromModelParallelRegion.apply(
                output_parallel, -1, self.config.process_group,
                self.world_size, self.rank
            )
        else:
            output = output_parallel
        
        return output


class RowParallelLinear(nn.Module):
    """行并行线性层
    
    将权重矩阵按行分割，每个 GPU 计算部分输入特征的贡献。
    Y = XA，其中 A 按行分割，X 也需要按列分割
    
    Args:
        in_features: 输入特征数（总数）
        out_features: 输出特征数
        bias: 是否使用偏置
        input_is_parallel: 输入是否已经并行
        config: 张量并行配置
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
        
        # 每个 GPU 的输入特征数
        assert in_features % self.world_size == 0
        self.in_features_per_partition = in_features // self.world_size
        
        # 初始化权重
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features  # 使用总输入特征数
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # 如果输入未并行，先分割
        if not self.input_is_parallel:
            input_parallel = tensor_parallel_split(input_, dim=-1, config=self.config)
        else:
            input_parallel = input_
        
        # 本地线性计算（不加偏置）
        output_parallel = F.linear(input_parallel, self.weight)
        
        # AllReduce 合并结果
        output = _ReduceFromModelParallelRegion.apply(
            output_parallel, self.config.process_group
        )
        
        # 加偏置
        if self.bias is not None:
            output = output + self.bias
        
        return output


class VocabParallelEmbedding(nn.Module):
    """词表并行嵌入层
    
    将词表按行分割到多个 GPU，每个 GPU 只存储部分词向量。
    
    Args:
        num_embeddings: 词表大小（总数）
        embedding_dim: 嵌入维度
        padding_idx: 填充索引
        config: 张量并行配置
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
        
        # 计算每个 GPU 的词表范围
        self.vocab_start_idx = self.rank * (num_embeddings // self.world_size)
        self.vocab_end_idx = (self.rank + 1) * (num_embeddings // self.world_size)
        self.num_embeddings_per_partition = self.vocab_end_idx - self.vocab_start_idx
        
        # 初始化嵌入
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            # 检查 padding_idx 是否在本分区
            if self.vocab_start_idx <= self.padding_idx < self.vocab_end_idx:
                local_idx = self.padding_idx - self.vocab_start_idx
                with torch.no_grad():
                    self.weight[local_idx].fill_(0)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # 创建掩码：哪些 token 在本分区
        input_mask = (input_ >= self.vocab_start_idx) & (input_ < self.vocab_end_idx)
        
        # 将不在本分区的索引设为 0（避免越界）
        masked_input = input_ - self.vocab_start_idx
        masked_input = masked_input.clamp(0, self.num_embeddings_per_partition - 1)
        
        # 本地嵌入查找
        output_parallel = F.embedding(masked_input, self.weight)
        
        # 将不在本分区的结果置零
        output_parallel = output_parallel * input_mask.unsqueeze(-1).float()
        
        # AllReduce 合并结果
        output = _ReduceFromModelParallelRegion.apply(
            output_parallel, self.config.process_group
        )
        
        return output
