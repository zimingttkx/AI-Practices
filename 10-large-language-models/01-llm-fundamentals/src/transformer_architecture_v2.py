"""
Transformer 架构实现

本模块提供 Transformer 架构的生产级实现，基于论文
"Attention Is All You Need" (Vaswani et al., 2017)。

核心组件：
    - 缩放点积注意力
    - 多头注意力
    - 逐位置前馈网络
    - Transformer 编码器/解码器层

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "AttentionConfig",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "PositionwiseFFN",
    "TransformerEncoderLayer",
    "TransformerEncoder",
]


@dataclass(frozen=True)
class AttentionConfig:
    """多头注意力机制配置。

    核心思想：
        定义注意力计算的超参数，同时确保数学一致性。

    数学原理：
        模型维度 d_model 必须能被注意力头数整除：
        d_k = d_model / n_heads

        这确保输出投影可以重构原始维度：
        Concat(head_1, ..., head_h) ∈ ℝ^{batch × seq_len × d_model}

    参数：
        d_model: 模型隐藏层维度。必须能被 n_heads 整除。
                典型值：512（基础）、1024（大型）、2048（超大）
        n_heads: 并行注意力头数。典型值：8、12、16
        d_k: 每个头的维度。若为 None，默认为 d_model // n_heads
        dropout: 注意力权的 Dropout 概率。范围：[0, 1)

    异常：
        ValueError: 若 d_model 不是正数或不能被 n_heads 整除
        ValueError: 若 n_heads 不是正数
        ValueError: 若 dropout 不在 [0, 1) 范围内

    复杂度：
        时间：每个注意力头 O(n² · d_k)，其中 n 是序列长度
        空间：投影矩阵 O(d_model²)

    对比：
        vs 单头注意力：多头允许学习不同的表示子空间
        vs CNN：注意力处理可变距离关系
    """

    d_model: int = 512
    n_heads: int = 8
    d_k: int | None = None
    dropout: float = 0.1

    def __post_init__(self) -> None:
        # 验证 d_model
        if self.d_model <= 0:
            raise ValueError(f"d_model 必须为正数，得到 {self.d_model}")

        # 验证 n_heads
        if self.n_heads <= 0:
            raise ValueError(f"n_heads 必须为正数，得到 {self.n_heads}")

        # 验证整除性
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除"
            )

        # 验证 dropout
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout 必须在 [0, 1) 范围内，得到 {self.dropout}")

        # 若未提供 d_k 则设置
        if self.d_k is None:
            object.__setattr__(self, "d_k", self.d_model // self.n_heads)
        elif self.d_k != self.d_model // self.n_heads:
            raise ValueError(
                f"d_k ({self.d_k}) 必须等于 d_model // n_heads "
                f"({self.d_model // self.n_heads})"
            )


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制。

    核心思想：
        将注意力权值计算为查询和键之间的归一化相似度，
        然后使用这些权值聚合值。

    数学原理：
        Attention(Q, K, V) = softmax(QK^T / √d_k) V

        其中：
        - Q ∈ ℝ^{n × d_k}: 查询矩阵
        - K ∈ ℝ^{m × d_k}: 键矩阵
        - V ∈ ℝ^{m × d_v}: 值矩阵
        - d_k: 缩放因子，防止梯度消失

        缩放因子 √d_k 至关重要，因为：
        对于大的 d_k，点积量级会增大，
        将 softmax 推入梯度极小的区域。

    问题陈述：
        如何衡量序列中不同位置之间的相关性，
        而不考虑它们的距离？

    解决方案：
        使用点积作为相似度度量，并配合缩放以维持
        跨序列位置的梯度流动。

    参数：
        d_k: 键/查询的维度。用于缩放。
        dropout: 应用于注意力权的 Dropout 概率。

    复杂度：
        时间：O(n² · d_k + n² · d_v)，其中 n 是序列长度
        空间：注意力矩阵 O(n²)

    对比：
        vs 加性注意力：点积在优化矩阵乘法时更快
        vs CNN：注意力具有全局感受野，路径长度 O(1)

    总结：
        这是使 Transformer 能够在不使用递归或卷积的情况下
        处理长程依赖的基础操作。
    """

    def __init__(self, d_k: int, dropout: float = 0.1) -> None:
        """初始化缩放点积注意力。

        参数：
            d_k: 缩放因子的维度。必须为正数。
            dropout: Dropout 概率，范围 [0, 1)。

        异常：
            ValueError: 若 d_k <= 0 或 dropout 不在 [0, 1) 内
        """
        super().__init__()

        if d_k <= 0:
            raise ValueError(f"d_k 必须为正数，得到 {d_k}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout 必须在 [0, 1) 范围内，得到 {dropout}")

        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """计算缩放点积注意力。

        参数：
            Q: 查询张量，形状 [batch, n_heads, seq_len_q, d_k]
            K: 键张量，形状 [batch, n_heads, seq_len_k, d_k]
            V: 值张量，形状 [batch, n_heads, seq_len_v, d_v]
            mask: 可选的注意力掩码。
                  形状 [batch, n_heads, seq_len_q, seq_len_k]
                  或 [batch, 1, seq_len_q, seq_len_k]
                  值为 0/False 的位置被掩码。

        返回：
            (输出, 注意力权值) 元组：
                - output: 加权后的值，形状 [batch, n_heads, seq_len_q, d_v]
                - attention_weights: 注意力权值，形状
                  [batch, n_heads, seq_len_q, seq_len_k]

        异常：
            ValueError: 若输入张量形状不兼容
        """
        # 验证输入形状
        if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
            raise ValueError(
                f"Q、K、V 必须是 4D 张量，得到形状 {Q.shape}、{K.shape}、{V.shape}"
            )

        batch_size, n_heads, seq_len_q, d_k = Q.shape
        _, _, seq_len_k, _ = K.shape
        _, _, seq_len_v, d_v = V.shape

        if d_k != self.d_k:
            raise ValueError(
                f"Q 特征维度 ({d_k}) 必须匹配 d_k ({self.d_k})"
            )

        if K.shape[1] != n_heads or K.shape[3] != d_k:
            raise ValueError(
                f"K 形状 {K.shape} 与 Q 形状 {Q.shape} 不兼容"
            )

        if V.shape[1] != n_heads:
            raise ValueError(
                f"V n_heads ({V.shape[1]}) 必须匹配 Q n_heads ({n_heads})"
            )

        if seq_len_k != seq_len_v:
            raise ValueError(
                f"键 seq_len ({seq_len_k}) 必须匹配值 seq_len ({seq_len_v})"
            )

        # 步骤 1：通过矩阵乘法计算注意力分数
        # [batch, n_heads, seq_len_q, d_k] × [batch, n_heads, d_k, seq_len_k]
        # = [batch, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 步骤 2：缩放以防止梯度消失
        # 这是能够稳定训练的关键创新
        scores = scores / math.sqrt(self.d_k)

        # 步骤 3：应用可选掩码
        # 被掩码的位置获得大的负值 -> softmax 后概率为零
        if mask is not None:
            if mask.dim() not in (3, 4):
                raise ValueError(f"mask 必须是 3D 或 4D，得到 {mask.dim()}D")

            # 若需要，将掩码广播到匹配分数形状
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len_q, seq_len_k]

            if mask.shape != scores.shape:
                raise ValueError(
                    f"mask 形状 {mask.shape} 必须匹配 scores 形状 {scores.shape}"
                )

            # 使用大的负值而非 -inf 以保证数值稳定性
            scores = scores.masked_fill(mask == 0, -1e9)

        # 步骤 4：应用 softmax 获得注意力权值
        # 沿键维度（最后一维）进行 softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 步骤 5：使用注意力权值聚合值
        # [batch, n_heads, seq_len_q, seq_len_k] × [batch, n_heads, seq_len_v, d_v]
        # = [batch, n_heads, seq_len_q, d_v]
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制。

    核心思想：
        并行运行多个注意力操作，每个学习不同的
        表示子空间，然后组合它们的输出。

    数学原理：
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

        其中：
        head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

        且：
        - W_i^Q ∈ ℝ^{d_model × d_k}: 头 i 的查询投影
        - W_i^K ∈ ℝ^{d_model × d_k}: 头 i 的键投影
        - W_i^V ∈ ℝ^{d_model × d_v}: 头 i 的值投影
        - W^O ∈ ℝ^{h·d_v × d_model}: 输出投影

        总参数量：4 · d_model²（假设 d_v = d_k）

    问题陈述：
        单个注意力头只能关注一种类型的关系，
        限制了表示能力。

    解决方案：
        多个头同时学习不同的注意力模式：
        - 某些头可能关注句法关系
        - 其他头可能关注语义关系
        - 其他头可能捕获位置依赖

    参数：
        config: 包含 d_model、n_heads、d_k、dropout 的 AttentionConfig

    复杂度：
        时间：O(n² · d_model)，其中 n 是序列长度
        空间：投影矩阵 O(d_model²) + 注意力 O(n²)

    对比：
        vs 单头：多头提供更丰富的表示
        vs 更多头：超过 8-16 个头后收益递减

    总结：
        多头注意力是使 Transformer 能够并行捕获
        多种类型关系的关键创新。
    """

    def __init__(self, config: AttentionConfig) -> None:
        """初始化多头注意力。

        参数：
            config: 已验证的 AttentionConfig

        异常：
            TypeError: 若 config 不是 AttentionConfig 实例
        """
        super().__init__()

        if not isinstance(config, AttentionConfig):
            raise TypeError(
                f"config 必须是 AttentionConfig，得到 {type(config).__name__}"
            )

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_v = config.d_k  # 通常 d_k = d_v

        # Q、K、V 投影：[d_model] -> [n_heads * d_k]
        # 现代实现中通常使用 bias=False
        self.W_q = nn.Linear(config.d_model, config.n_heads * config.d_k, bias=False)
        self.W_k = nn.Linear(config.d_model, config.n_heads * config.d_k, bias=False)
        self.W_v = nn.Linear(config.d_model, config.n_heads * config.d_k, bias=False)

        # 输出投影：[n_heads * d_k] -> [d_model]
        self.W_o = nn.Linear(config.n_heads * config.d_k, config.d_model, bias=False)

        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(config.d_k, config.dropout)

        # 使用 Xavier 初始化权重
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """使用 Xavier 均匀初始化初始化权重。

        总结：
            这有助于维持通过网络的变化量级，
            防止梯度消失/爆炸。
        """
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """应用多头注意力。

        参数：
            query: 输入张量，形状 [batch, seq_len_q, d_model]
            key: 输入张量，形状 [batch, seq_len_k, d_model]
            value: 输入张量，形状 [batch, seq_len_v, d_model]
            mask: 可选的注意力掩码，形状 [batch, seq_len_q, seq_len_k]
                   或 [batch, n_heads, seq_len_q, seq_len_k]

        返回：
            (输出, 注意力权值) 元组：
                - output: 处理后的张量，形状 [batch, seq_len_q, d_model]
                - attention_weights: 来自最后一个头的注意力权值
                  形状 [batch, n_heads, seq_len_q, seq_len_k]

        异常：
            ValueError: 若输入形状不兼容
        """
        batch_size = query.size(0)

        # 验证输入形状
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                f"query、key、value 必须是 3D 张量，"
                f"得到形状 {query.shape}、{key.shape}、{value.shape}"
            )

        if query.shape[2] != self.d_model:
            raise ValueError(
                f"query 特征维度 ({query.shape[2]}) 必须匹配 d_model ({self.d_model})"
            )

        # 步骤 1：线性投影并重塑为多个头
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads * d_k]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 重塑并转置：[batch, seq_len, n_heads * d_k]
        # -> [batch, seq_len, n_heads, d_k]
        # -> [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 形状：[batch, n_heads, seq_len, d_k]

        # 步骤 2：调整掩码形状（若需要）
        if mask is not None:
            if mask.dim() == 3:  # [batch, seq_len_q, seq_len_k]
                # 为头添加维度：[batch, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)
            elif mask.dim() != 4:
                raise ValueError(
                    f"mask 必须是 3D 或 4D，得到 {mask.dim()}D，形状 {mask.shape}"
                )

        # 步骤 3：应用缩放点积注意力
        # x: [batch, n_heads, seq_len_q, d_k]
        # attn_weights: [batch, n_heads, seq_len_q, seq_len_k]
        x, attn_weights = self.attention(Q, K, V, mask)

        # 步骤 4：拼接头
        # [batch, n_heads, seq_len_q, d_k] -> [batch, seq_len_q, n_heads, d_k]
        x = x.transpose(1, 2).contiguous()
        # -> [batch, seq_len_q, n_heads * d_k]
        x = x.view(batch_size, -1, self.n_heads * self.d_k)

        # 步骤 5：输出投影
        # [batch, seq_len_q, n_heads * d_k] -> [batch, seq_len_q, d_model]
        output = self.W_o(x)

        return output, attn_weights


class PositionwiseFFN(nn.Module):
    """逐位置前馈网络。

    核心思想：
        对每个位置独立且相同地应用同一个两层全连接网络。

    数学原理：
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

        其中：
        - x ∈ ℝ^{d_model}: 单个位置的输入
        - W_1 ∈ ℝ^{d_model × d_ff}: 扩展矩阵
        - W_2 ∈ ℝ^{d_ff × d_model}: 投影矩阵
        - d_ff = 4 · d_model（典型扩展因子）

        这等价于带 ReLU 非线性的两层线性变换：
        FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

    问题陈述：
        自注意力对于值是线性的。模型需要
        非线性来近似复杂函数。

    解决方案：
        带 ReLU 激活的两层 MLP 提供必要的
        非线性，同时计算高效。

    参数：
        d_model: 输入和输出维度
        d_ff: 隐藏层维度。典型为 4 × d_model
        dropout: Dropout 概率，范围 [0, 1)

    复杂度：
        时间：O(n · d_model · d_ff)，其中 n 是序列长度
        空间：参数 O(d_model · d_ff)

    对比：
        vs RNN：FFN 并行处理所有位置
        vs CNN：FFN 对单个位置操作，而非空间邻域

    总结：
        尽管简单，FFN 包含约 67% 的 Transformer 参数
        且对模型能力至关重要。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """初始化逐位置 FFN。

        参数：
            d_model: 输入/输出特征维度。必须为正数。
            d_ff: 隐藏层维度。必须为正数。
            dropout: Dropout 概率，范围 [0, 1)。

        异常：
            ValueError: 若维度无效或 dropout 超出范围
        """
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model 必须为正数，得到 {d_model}")
        if d_ff <= 0:
            raise ValueError(f"d_ff 必须为正数，得到 {d_ff}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout 必须在 [0, 1) 范围内，得到 {dropout}")

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """使用 Xavier 初始化初始化权重。"""
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.zeros_(self.w_1.bias)
        nn.init.zeros_(self.w_2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """应用逐位置前馈变换。

        参数：
            x: 输入张量，形状 [batch, seq_len, d_model]

        返回：
            输出张量，形状 [batch, seq_len, d_model]

        异常：
            ValueError: 若输入特征维度不匹配 d_model
        """
        if x.dim() != 3:
            raise ValueError(f"期望 3D 输入，得到 {x.dim()}D，形状 {x.shape}")

        if x.shape[2] != self.w_1.in_features:
            raise ValueError(
                f"输入特征维度 ({x.shape[2]}) 必须匹配 d_model ({self.w_1.in_features})"
            )

        # 应用扩展、ReLU、dropout，然后投影
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoderLayer(nn.Module):
    """单个 Transformer 编码器层。

    核心思想：
        组合自注意力和前馈网络，带残差
        连接和层归一化。

    数学原理（Post-LN）：
        x = x + Dropout(MultiHeadAttention(LayerNorm(x)))
        x = x + Dropout(FFN(LayerNorm(x)))

        其中 LayerNorm(x) = γ · (x - μ) / σ + β

    参数：
        config: 多头注意力的 AttentionConfig
        d_ff: FFN 隐藏维度。典型为 4 × d_model
        dropout: Dropout 概率

    复杂度：
        时间：O(n² · d_model + n · d_model · d_ff)
        空间：O(d_model² + n²)

    总结：
        这是现代 LLM 如 GPT 和 BERT 的基础构建块。
    """

    def __init__(
        self,
        config: AttentionConfig,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """初始化编码器层。

        参数：
            config: 已验证的 AttentionConfig
            d_ff: FFN 隐藏维度
            dropout: Dropout 概率

        异常：
            ValueError: 若 d_ff <= 0 或 dropout 不在 [0, 1) 内
        """
        super().__init__()

        if d_ff <= 0:
            raise ValueError(f"d_ff 必须为正数，得到 {d_ff}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout 必须在 [0, 1) 范围内，得到 {dropout}")

        self.self_attn = MultiHeadAttention(config)
        self.ffn = PositionwiseFFN(config.d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """应用编码器层变换。

        参数：
            x: 输入张量，形状 [batch, seq_len, d_model]
            mask: 可选的注意力掩码，形状 [batch, seq_len, seq_len]

        返回：
            输出张量，形状 [batch, seq_len, d_model]
        """
        # 自注意力块，带残差和层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # FFN 块，带残差和层归一化
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """堆叠的 Transformer 编码器层。

    核心思想：
        堆叠 N 个相同的编码器层以构建深度表示。

    数学原理：
        H^0 = Embedding(Input) + PositionalEncoding
        H^l = EncoderLayer(H^{l-1}) for l = 1, ..., L

    参数：
        config: 每层的 AttentionConfig
        num_layers: 编码器层数（典型为 6、12、24）
        d_ff: FFN 隐藏维度

    总结：
        更深的网络可以学习更复杂的函数但更难训练。
    """

    def __init__(
        self,
        config: AttentionConfig,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """初始化编码器堆叠。

        参数：
            config: 已验证的 AttentionConfig
            num_layers: 层数。必须为正数。
            d_ff: FFN 隐藏维度
            dropout: Dropout 概率

        异常：
            ValueError: 若 num_layers <= 0
        """
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"num_layers 必须为正数，得到 {num_layers}")

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """通过所有编码器层传递输入。

        参数：
            x: 输入张量，形状 [batch, seq_len, d_model]
            mask: 可选的注意力掩码

        返回：
            输出张量，形状 [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ============================================================================
# 单元测试
# ============================================================================

def _test_attention_config() -> None:
    """测试 AttentionConfig 验证。"""
    print("测试 AttentionConfig...")

    # 有效配置
    config = AttentionConfig(d_model=512, n_heads=8)
    assert config.d_k == 64

    # 无效：d_model 不能被 n_heads 整除
    try:
        AttentionConfig(d_model=100, n_heads=8)
        raise AssertionError("应该抛出 ValueError")
    except ValueError:
        pass

    # 无效：负 dropout
    try:
        AttentionConfig(d_model=512, n_heads=8, dropout=-0.1)
        raise AssertionError("应该抛出 ValueError")
    except ValueError:
        pass

    print("✓ AttentionConfig 测试通过")


def _test_scaled_dot_product_attention() -> None:
    """测试缩放点积注意力。"""
    print("测试 ScaledDotProductAttention...")

    batch_size, n_heads, seq_len, d_k = 2, 4, 10, 32

    # 使用 dropout=0 进行确定性测试
    attn = ScaledDotProductAttention(d_k=d_k, dropout=0.0)
    attn.eval()  # 设置为评估模式

    # 创建测试输入
    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = torch.randn(batch_size, n_heads, seq_len, d_k)
    V = torch.randn(batch_size, n_heads, seq_len, d_k)

    # 前向传播
    with torch.no_grad():
        output, weights = attn(Q, K, V)

    # 验证形状
    assert output.shape == (batch_size, n_heads, seq_len, d_k)
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len)

    # 验证注意力权值和约为 1（softmax 性质）
    # 注意：没有掩码时，所有行应和约为 1.0
    weights_sum = weights.sum(dim=-1)
    assert weights_sum.min() > 0.9 and weights_sum.max() < 1.1, \
        f"注意力权值和不约为 1：最小值={weights_sum.min():.6f}，最大值={weights_sum.max():.6f}"

    # 测试带掩码
    mask = torch.ones(batch_size, n_heads, seq_len, seq_len)
    mask[:, :, :, seq_len // 2:] = 0  # 掩码掉后半部分

    with torch.no_grad():
        output_masked, weights_masked = attn(Q, K, V, mask)

    # 检查被掩码的位置具有零注意力
    assert torch.all(weights_masked[:, :, :, :seq_len // 2] >= 0)
    assert torch.all(weights_masked[:, :, :, seq_len // 2:] == 0)

    print("✓ ScaledDotProductAttention 测试通过")


def _test_multi_head_attention() -> None:
    """测试多头注意力。"""
    print("测试 MultiHeadAttention...")

    config = AttentionConfig(d_model=512, n_heads=8, dropout=0.0)
    mha = MultiHeadAttention(config)

    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.d_model)

    # 前向传播
    output, weights = mha(x, x, x)

    # 验证形状
    assert output.shape == (batch_size, seq_len, config.d_model)
    assert weights.shape == (batch_size, config.n_heads, seq_len, seq_len)

    # 测试因果掩码
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.expand(batch_size, config.n_heads, -1, -1)

    output_causal, _ = mha(x, x, x, causal_mask)
    assert output_causal.shape == output.shape

    print("✓ MultiHeadAttention 测试通过")


def _test_positionwise_ffn() -> None:
    """测试逐位置 FFN。"""
    print("测试 PositionwiseFFN...")

    d_model, d_ff = 512, 2048
    ffn = PositionwiseFFN(d_model, d_ff, dropout=0.0)

    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = ffn(x)

    # 验证形状
    assert output.shape == x.shape

    # 检查输出与输入不同（非线性）
    assert not torch.allclose(output, x)

    print("✓ PositionwiseFFN 测试通过")


def _test_transformer_encoder() -> None:
    """测试 Transformer 编码器。"""
    print("测试 TransformerEncoder...")

    config = AttentionConfig(d_model=512, n_heads=8, dropout=0.0)
    encoder = TransformerEncoder(config, num_layers=2, d_ff=2048, dropout=0.0)

    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.d_model)

    # 前向传播
    output = encoder(x)

    # 验证形状
    assert output.shape == x.shape

    # 检查输出不同（应用了变换）
    assert not torch.allclose(output, x)

    # 测试带填充掩码
    padding_mask = torch.ones(batch_size, seq_len, seq_len)
    padding_mask[:, :, 5:] = 0  # 掩码掉填充

    # 为所有头扩展掩码：[batch, seq_len, seq_len] -> [batch, n_heads, seq_len, seq_len]
    padding_mask = padding_mask.unsqueeze(1).expand(batch_size, config.n_heads, seq_len, seq_len)

    output_masked = encoder(x, padding_mask)
    assert output_masked.shape == x.shape

    print("✓ TransformerEncoder 测试通过")


def run_all_tests() -> None:
    """运行所有单元测试。"""
    print("=" * 60)
    print("运行 Transformer 架构单元测试")
    print("=" * 60)

    _test_attention_config()
    _test_scaled_dot_product_attention()
    _test_multi_head_attention()
    _test_positionwise_ffn()
    _test_transformer_encoder()

    print("=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    # 直接执行时运行测试
    run_all_tests()
