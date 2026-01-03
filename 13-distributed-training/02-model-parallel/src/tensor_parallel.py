"""
Tensor Parallelism 张量并行实现

================================================================================
核心思想 (一句话理解)
================================================================================
张量并行 = 把单层的权重矩阵切分到多个GPU，每个GPU计算一部分输出

================================================================================
两种切分方式 (图解)
================================================================================

    原始线性层: Y = X @ W + b
    其中 X: [batch, seq, in_features], W: [in_features, out_features]

    ┌─────────────────────────────────────────────────────────────────────┐
    │  列并行 (Column Parallel): 按输出维度切分W                           │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │  W = [W1 | W2]  (按列切分，每个GPU存一半列)                   │   │
    │  │                                                              │   │
    │  │  GPU0: Y1 = X @ W1   →  输出 [batch, seq, out/2]            │   │
    │  │  GPU1: Y2 = X @ W2   →  输出 [batch, seq, out/2]            │   │
    │  │                                                              │   │
    │  │  最终: Y = [Y1 | Y2]  (AllGather拼接)                        │   │
    │  │                                                              │   │
    │  │  适用: MLP第一层、QKV投影 (后接逐元素操作如GeLU)              │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │                                                                     │
    │  行并行 (Row Parallel): 按输入维度切分W                             │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │  W = [W1]    X = [X1 | X2]  (输入也要切分)                   │   │
    │  │      [W2]                                                    │   │
    │  │                                                              │   │
    │  │  GPU0: Y1 = X1 @ W1  →  部分结果                             │   │
    │  │  GPU1: Y2 = X2 @ W2  →  部分结果                             │   │
    │  │                                                              │   │
    │  │  最终: Y = Y1 + Y2   (AllReduce求和)                         │   │
    │  │                                                              │   │
    │  │  适用: MLP第二层、Output投影 (需要汇总结果)                   │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────┘

================================================================================
Transformer中的应用 (Megatron-LM方案)
================================================================================

    MLP层: 列并行 → GeLU → 行并行
    ┌─────────────────────────────────────────────────────────────────┐
    │  X → [列并行Linear] → GeLU → [行并行Linear] → Y                  │
    │       (无通信)         ↑        (AllReduce)                      │
    │                    逐元素操作                                    │
    │                    可直接在分片上计算                             │
    └─────────────────────────────────────────────────────────────────┘

    Attention层: QKV列并行 → Attention → Output行并行
    ┌─────────────────────────────────────────────────────────────────┐
    │  X → [QKV列并行] → Attention → [Output行并行] → Y                │
    │       (无通信)        ↑          (AllReduce)                     │
    │                  注意力头天然可并行                               │
    │                  8头2GPU → 每GPU算4头                            │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
通信开销分析
================================================================================
    列并行: 前向AllGather，反向ReduceScatter
    行并行: 前向AllReduce，反向无通信
    每层通信量: O(batch × seq × hidden)

================================================================================
前置知识
================================================================================
- 矩阵乘法: Y = X @ W 的维度变化
- AllReduce/AllGather通信原语
- Transformer的MLP和Attention结构

================================================================================
参考文献
================================================================================
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


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class TensorParallelConfig:
    """张量并行配置

    Attributes:
        world_size: 张量并行的GPU数量
        rank: 当前GPU在张量并行组中的编号 (0 到 world_size-1)
        process_group: 分布式进程组，用于通信
        sequence_parallel: 是否启用序列并行 (与张量并行配合)
        async_tensor_parallel: 是否启用异步通信 (提升性能)

    Example:
        >>> config = TensorParallelConfig(
        ...     world_size=4,  # 4个GPU做张量并行
        ...     rank=0,        # 当前是第0个GPU
        ... )
    """
    world_size: int = 1
    rank: int = 0
    process_group: Optional[dist.ProcessGroup] = None
    sequence_parallel: bool = False
    async_tensor_parallel: bool = False


# =============================================================================
# 辅助函数
# =============================================================================

def _get_tensor_parallel_world_size(config: Optional[TensorParallelConfig] = None) -> int:
    """获取张量并行的GPU数量"""
    if config is not None:
        return config.world_size
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _get_tensor_parallel_rank(config: Optional[TensorParallelConfig] = None) -> int:
    """获取当前GPU在张量并行组中的编号"""
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
    """沿指定维度切分张量，返回当前GPU的分片

    用于将输入数据切分到各个GPU。

    Args:
        tensor: 要切分的张量
        dim: 切分的维度 (-1表示最后一维)
        config: 张量并行配置

    Returns:
        当前GPU负责的张量分片

    Example:
        >>> x = torch.randn(2, 8, 1024)  # [batch, seq, hidden]
        >>> x_local = tensor_parallel_split(x, dim=-1, config)
        >>> # 如果world_size=4，x_local.shape = [2, 8, 256]
    """
    world_size = _get_tensor_parallel_world_size(config)
    rank = _get_tensor_parallel_rank(config)

    # 单GPU不需要切分
    if world_size == 1:
        return tensor

    dim_size = tensor.size(dim)
    assert dim_size % world_size == 0, \
        f"维度{dim}的大小{dim_size}必须能被{world_size}整除"

    # 计算当前GPU负责的范围
    chunk_size = dim_size // world_size
    start = rank * chunk_size
    # narrow: 从start开始取chunk_size个元素
    return tensor.narrow(dim, start, chunk_size).contiguous()


def tensor_parallel_gather(
    tensor: torch.Tensor,
    dim: int = -1,
    config: Optional[TensorParallelConfig] = None,
) -> torch.Tensor:
    """收集所有GPU的张量分片，拼接成完整张量

    Args:
        tensor: 当前GPU的张量分片
        dim: 拼接的维度
        config: 张量并行配置

    Returns:
        拼接后的完整张量

    Example:
        >>> x_local = torch.randn(2, 8, 256)  # 当前GPU的分片
        >>> x_full = tensor_parallel_gather(x_local, dim=-1, config)
        >>> # 如果world_size=4，x_full.shape = [2, 8, 1024]
    """
    world_size = _get_tensor_parallel_world_size(config)

    if world_size == 1:
        return tensor

    process_group = config.process_group if config else None

    # 创建接收缓冲区
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    # AllGather: 收集所有GPU的张量
    dist.all_gather(tensor_list, tensor, group=process_group)
    # 沿指定维度拼接
    return torch.cat(tensor_list, dim=dim)


# =============================================================================
# 自动求导函数 (处理前向/反向传播中的通信)
# =============================================================================

class _CopyToModelParallelRegion(torch.autograd.Function):
    """复制到模型并行区域

    前向: 直接传递 (identity)
    反向: AllReduce梯度 (因为前向时输入被复制到多个GPU)

    用于列并行的输入处理。
    """

    @staticmethod
    def forward(ctx, input_, process_group):
        ctx.process_group = process_group
        return input_  # 前向直接传递

    @staticmethod
    def backward(ctx, grad_output):
        # 反向时需要AllReduce，因为每个GPU都有完整输入的梯度
        if dist.is_initialized():
            dist.all_reduce(grad_output, group=ctx.process_group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """从模型并行区域归约

    前向: AllReduce求和 (汇总各GPU的部分结果)
    反向: 直接传递 (梯度自然分配到各GPU)

    用于行并行的输出处理。
    """

    @staticmethod
    def forward(ctx, input_, process_group):
        # 前向时AllReduce汇总各GPU的部分结果
        if dist.is_initialized():
            dist.all_reduce(input_, group=process_group)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # 反向直接传递


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """从模型并行区域收集

    前向: AllGather收集所有分片
    反向: 切分梯度 (每个GPU只需要自己分片的梯度)

    用于列并行需要完整输出时。
    """

    @staticmethod
    def forward(ctx, input_, dim, process_group, world_size, rank):
        ctx.dim = dim
        ctx.process_group = process_group
        ctx.world_size = world_size
        ctx.rank = rank

        if world_size == 1:
            return input_

        # AllGather收集所有GPU的分片
        tensor_list = [torch.zeros_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_, group=process_group)
        return torch.cat(tensor_list, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.world_size == 1:
            return grad_output, None, None, None, None

        # 反向时切分梯度，每个GPU只取自己的部分
        dim_size = grad_output.size(ctx.dim)
        chunk_size = dim_size // ctx.world_size
        start = ctx.rank * chunk_size
        local_grad = grad_output.narrow(ctx.dim, start, chunk_size).contiguous()
        return local_grad, None, None, None, None


# =============================================================================
# 列并行线性层
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """列并行线性层

    按输出维度(列)切分权重矩阵，每个GPU计算一部分输出特征。

    数学公式:
        原始: Y = X @ W，其中 W: [in, out]
        切分: W = [W1 | W2 | ... | Wn]，每个GPU存储 W_i: [in, out/n]
        计算: GPU_i 计算 Y_i = X @ W_i，得到 [batch, seq, out/n]
        汇总: Y = [Y1 | Y2 | ... | Yn] (如果gather_output=True)

    适用场景:
        - MLP的第一个线性层 (后接GeLU等逐元素操作)
        - Attention的QKV投影 (后接注意力计算)

    Args:
        in_features: 输入特征维度
        out_features: 总输出特征维度 (会被切分)
        bias: 是否使用偏置
        gather_output: 是否收集所有GPU的输出
            - True: 返回完整输出 [batch, seq, out_features]
            - False: 返回分片输出 [batch, seq, out_features/world_size]
        config: 张量并行配置

    Example:
        >>> # 4个GPU，输出维度16384
        >>> linear = ColumnParallelLinear(4096, 16384, config=config)
        >>> # 每个GPU存储 W: [4096, 4096]
        >>> x = torch.randn(2, 512, 4096)
        >>> y = linear(x)  # gather_output=True时 y: [2, 512, 16384]
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

        # 确保输出维度能被GPU数整除
        assert out_features % self.world_size == 0, \
            f"out_features({out_features})必须能被world_size({self.world_size})整除"

        # 每个GPU负责的输出维度
        self.out_features_per_partition = out_features // self.world_size

        # 权重: [out_per_gpu, in] (PyTorch Linear的权重是转置存储的)
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            # 偏置也按输出维度切分
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数 (Kaiming均匀分布)"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            input_: 输入张量 [batch, seq, in_features]

        Returns:
            输出张量，形状取决于gather_output设置
        """
        # 1. 复制输入到模型并行区域 (反向时会AllReduce梯度)
        input_parallel = _CopyToModelParallelRegion.apply(
            input_, self.config.process_group
        )

        # 2. 本地矩阵乘法
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        # 3. 根据配置决定是否收集输出
        if self.gather_output:
            # AllGather收集所有GPU的输出
            output = _GatherFromModelParallelRegion.apply(
                output_parallel, -1, self.config.process_group,
                self.world_size, self.rank
            )
        else:
            # 保持分片状态，传给下一层
            output = output_parallel

        return output


# =============================================================================
# 行并行线性层
# =============================================================================

class RowParallelLinear(nn.Module):
    """行并行线性层

    按输入维度(行)切分权重矩阵，每个GPU处理一部分输入特征。

    数学公式:
        原始: Y = X @ W，其中 W: [in, out]
        切分: W 按行切分，X 按列切分
              W_i: [in/n, out]，X_i: [batch, seq, in/n]
        计算: GPU_i 计算 Y_i = X_i @ W_i，得到部分结果
        汇总: Y = Y1 + Y2 + ... + Yn (AllReduce求和)

    适用场景:
        - MLP的第二个线性层 (需要汇总结果)
        - Attention的Output投影

    Args:
        in_features: 总输入特征维度 (会被切分)
        out_features: 输出特征维度
        bias: 是否使用偏置
        input_is_parallel: 输入是否已经是分片的
            - True: 输入已按最后一维切分，直接使用
            - False: 需要先切分输入
        config: 张量并行配置

    Example:
        >>> # 4个GPU，输入维度16384
        >>> linear = RowParallelLinear(16384, 4096, config=config)
        >>> # 每个GPU存储 W: [4096, 4096]
        >>> x = torch.randn(2, 512, 4096)  # 已分片的输入
        >>> y = linear(x)  # y: [2, 512, 4096]
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

        # 确保输入维度能被GPU数整除
        assert in_features % self.world_size == 0, \
            f"in_features({in_features})必须能被world_size({self.world_size})整除"

        # 每个GPU负责的输入维度
        self.in_features_per_partition = in_features // self.world_size

        # 权重: [out, in_per_gpu]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            # 偏置不切分，只在AllReduce后加一次
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化参数 (Kaiming均匀分布)"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features  # 使用总输入维度
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            input_: 输入张量
                - input_is_parallel=True: [batch, seq, in_features/world_size]
                - input_is_parallel=False: [batch, seq, in_features]

        Returns:
            输出张量 [batch, seq, out_features]
        """
        # 1. 如果输入未分片，先切分
        if not self.input_is_parallel:
            input_parallel = tensor_parallel_split(input_, dim=-1, config=self.config)
        else:
            input_parallel = input_

        # 2. 本地矩阵乘法 (不加偏置)
        output_parallel = F.linear(input_parallel, self.weight)

        # 3. AllReduce汇总所有GPU的部分结果
        output = _ReduceFromModelParallelRegion.apply(
            output_parallel, self.config.process_group
        )

        # 4. 加偏置 (只加一次，在AllReduce之后)
        if self.bias is not None:
            output = output + self.bias

        return output


# =============================================================================
# 词表并行嵌入层
# =============================================================================

class VocabParallelEmbedding(nn.Module):
    """词表并行嵌入层

    将大词表切分到多个GPU，每个GPU存储一部分词的嵌入向量。

    数学公式:
        原始: E: [vocab_size, embed_dim]
        切分: E = [E1; E2; ...; En]，每个GPU存储 vocab_size/n 个词
        查找: 根据token_id确定在哪个GPU，只有对应GPU返回非零结果
        汇总: AllReduce求和得到最终嵌入

    适用场景:
        - 大词表模型 (如多语言模型，词表可能有100k+)
        - 词表太大无法放入单卡

    Args:
        num_embeddings: 总词表大小
        embedding_dim: 嵌入维度
        padding_idx: 填充token的索引
        config: 张量并行配置

    Example:
        >>> # 4个GPU，词表50000
        >>> embed = VocabParallelEmbedding(50000, 4096, config=config)
        >>> # 每个GPU存储 12500 个词的嵌入
        >>> tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> embeddings = embed(tokens)  # [2, 3, 4096]
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

        # 计算当前GPU负责的词表范围
        # GPU0: [0, vocab/n), GPU1: [vocab/n, 2*vocab/n), ...
        self.vocab_start_idx = self.rank * (num_embeddings // self.world_size)
        self.vocab_end_idx = (self.rank + 1) * (num_embeddings // self.world_size)
        self.num_embeddings_per_partition = self.vocab_end_idx - self.vocab_start_idx

        # 只存储本GPU负责的词嵌入
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化嵌入权重"""
        nn.init.normal_(self.weight)
        # 如果padding_idx在当前GPU的范围内，将其嵌入设为0
        if self.padding_idx is not None:
            if self.vocab_start_idx <= self.padding_idx < self.vocab_end_idx:
                local_idx = self.padding_idx - self.vocab_start_idx
                with torch.no_grad():
                    self.weight[local_idx].fill_(0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            input_: token索引张量 [batch, seq]

        Returns:
            嵌入向量 [batch, seq, embedding_dim]
        """
        # 1. 创建掩码：哪些token在当前GPU的词表范围内
        input_mask = (input_ >= self.vocab_start_idx) & (input_ < self.vocab_end_idx)

        # 2. 将token索引转换为本地索引
        # 不在范围内的token会被clamp到有效范围，但会被掩码置零
        masked_input = input_ - self.vocab_start_idx
        masked_input = masked_input.clamp(0, self.num_embeddings_per_partition - 1)

        # 3. 本地嵌入查找
        output_parallel = F.embedding(masked_input, self.weight)

        # 4. 用掩码将不属于当前GPU的token嵌入置零
        output_parallel = output_parallel * input_mask.unsqueeze(-1).float()

        # 5. AllReduce汇总所有GPU的结果
        # 每个token只有一个GPU返回非零结果，求和即得最终嵌入
        output = _ReduceFromModelParallelRegion.apply(
            output_parallel, self.config.process_group
        )

        return output
