"""常用神经网络模型定义。/ Common neural network model definitions.

提供教程中使用的基础模型架构，包括简单的 MLP 和 Wide & Deep 模型。

Provides foundational model architectures used in the tutorial,
including simple MLP and Wide & Deep models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """简单多层感知机。/ Simple Multi-Layer Perceptron.

    使用 nn.Sequential 实现的标准 MLP，适用于分类和回归任务。

    A standard MLP implemented with nn.Sequential,
    suitable for both classification and regression tasks.

    Args:
        input_size: 输入特征维度 / input feature dimension
        hidden_sizes: 隐藏层神经元数量列表 / list of hidden layer sizes
        output_size: 输出维度 / output dimension
        activation: 激活函数类型 / activation function type ('relu', 'elu', 'selu')
        dropout_rate: Dropout 比率 / dropout rate (0 means no dropout)
        use_batchnorm: 是否使用批量归一化 / whether to use batch normalization

    Example:
        >>> model = SimpleMLP(input_size=8, hidden_sizes=[30, 30], output_size=1)
        >>> x = torch.randn(16, 8)
        >>> model(x).shape
        torch.Size([16, 1])
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        output_size: int = 1,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [30, 30]

        # 选择激活函数 / Select activation function
        act_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "leaky_relu": nn.LeakyReLU,
        }
        act_fn = act_map.get(activation, nn.ReLU)

        # 选择初始化方法 / Select initialization method
        init_fn = nn.init.xavier_uniform_
        if activation in ("relu", "leaky_relu"):
            init_fn = nn.init.kaiming_normal_
        elif activation == "selu":
            init_fn = nn.init.kaiming_normal_  # lecun 近似 / approximate lecun

        # 构建层 / Build layers
        layers: list[nn.Module] = []
        prev_size = input_size

        for _, hidden_size in enumerate(hidden_sizes):
            linear = nn.Linear(prev_size, hidden_size, bias=not use_batchnorm)
            init_fn(linear.weight)
            layers.append(linear)

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(act_fn())

            if dropout_rate > 0:
                if activation == "selu":
                    layers.append(nn.AlphaDropout(dropout_rate))
                else:
                    layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # 输出层 / Output layer
        output_layer = nn.Linear(prev_size, output_size)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。/ Forward pass."""
        return self.network(x)


class WideAndDeepModel(nn.Module):
    """Wide & Deep 模型。/ Wide & Deep model.

    实现 Google 2016 年提出的 Wide & Deep 架构，结合线性模型（记忆能力）
    和深度神经网络（泛化能力）。

    Implements the Wide & Deep architecture proposed by Google in 2016,
    combining a linear model (memorization) with a deep neural network (generalization).

    Args:
        wide_size: Wide 路径输入维度 / Wide path input dimension
        deep_size: Deep 路径输入维度 / Deep path input dimension
        hidden_sizes: Deep 路径隐藏层大小 / Deep path hidden layer sizes
        output_size: 输出维度 / output dimension
        aux_output: 是否包含辅助输出 / whether to include auxiliary output

    Reference:
        Cheng et al., "Wide & Deep Learning for Recommender Systems", 2016.
    """

    def __init__(
        self,
        wide_size: int,
        deep_size: int,
        hidden_sizes: list[int] | None = None,
        output_size: int = 1,
        aux_output: bool = False,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [30, 30]

        self.aux_output = aux_output

        # Deep 路径 / Deep path
        deep_layers: list[nn.Module] = []
        prev_size = deep_size
        for h in hidden_sizes:
            deep_layers.append(nn.Linear(prev_size, h))
            deep_layers.append(nn.ReLU())
            prev_size = h
        self.deep_network = nn.Sequential(*deep_layers)

        # 主输出 / Main output (wide + deep concatenated)
        concat_size = wide_size + hidden_sizes[-1]
        self.main_output = nn.Linear(concat_size, output_size)

        # 辅助输出 / Auxiliary output (from deep only)
        if aux_output:
            self.aux_output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(
        self, x_wide: torch.Tensor, x_deep: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播。/ Forward pass.

        Args:
            x_wide: Wide 路径输入 / Wide path input
            x_deep: Deep 路径输入 / Deep path input

        Returns:
            如果 aux_output=True，返回 (main_output, aux_output) 元组；
            否则只返回 main_output。

            If aux_output=True, returns (main_output, aux_output) tuple;
            otherwise returns main_output only.
        """
        deep_out = self.deep_network(x_deep)
        concat = torch.cat([x_wide, deep_out], dim=1)
        main_out = self.main_output(concat)

        if self.aux_output:
            aux_out = self.aux_output_layer(deep_out)
            return main_out, aux_out

        return main_out
