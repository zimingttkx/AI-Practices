"""
模型量化 (Model Quantization)

本模块实现深度学习模型的量化技术，包括：
- 动态量化 (Dynamic Quantization)
- 静态量化 (Static Quantization)
- 量化感知训练 (Quantization-Aware Training, QAT)

=== 量化原理 ===

量化将高精度浮点数 (FP32) 转换为低精度整数 (INT8/INT4)：

    q = round((r - z) / s)
    r = s * q + z

其中：
- r: 原始浮点值
- q: 量化后的整数值
- s: 缩放因子 (scale)
- z: 零点 (zero point)

=== 参考文献 ===

1. Jacob et al. "Quantization and Training of Neural Networks for Efficient
   Integer-Arithmetic-Only Inference" 2018
2. Nagel et al. "A White Paper on Neural Network Quantization" 2021
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizationType(Enum):
    """量化类型"""
    DYNAMIC = "dynamic"      # 动态量化
    STATIC = "static"        # 静态量化
    QAT = "qat"              # 量化感知训练


class QuantizationGranularity(Enum):
    """量化粒度"""
    PER_TENSOR = "per_tensor"    # 整个张量一个 scale
    PER_CHANNEL = "per_channel"  # 每个通道一个 scale
    PER_GROUP = "per_group"      # 每组元素一个 scale


@dataclass
class QuantizationConfig:
    """量化配置"""

    # 量化类型
    quant_type: QuantizationType = QuantizationType.DYNAMIC

    # 量化位数
    weight_bits: int = 8
    activation_bits: int = 8

    # 量化粒度
    weight_granularity: QuantizationGranularity = QuantizationGranularity.PER_CHANNEL
    activation_granularity: QuantizationGranularity = QuantizationGranularity.PER_TENSOR

    # 对称量化
    symmetric: bool = True

    # 量化范围
    weight_qmin: int = -128
    weight_qmax: int = 127
    activation_qmin: int = 0
    activation_qmax: int = 255

    # 校准配置 (静态量化)
    num_calibration_batches: int = 100
    calibration_method: str = "minmax"  # minmax, histogram, entropy

    # QAT 配置
    qat_start_epoch: int = 0
    freeze_bn_epochs: int = 2

    # 要量化的层类型
    quantize_layers: List[str] = field(default_factory=lambda: ["Linear", "Conv2d"])


def compute_scale_zero_point(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    qmin: int,
    qmax: int,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算量化的 scale 和 zero_point

    Args:
        x_min: 最小值
        x_max: 最大值
        qmin: 量化最小值
        qmax: 量化最大值
        symmetric: 是否对称量化

    Returns:
        scale, zero_point
    """
    if symmetric:
        # 对称量化: zero_point = 0
        x_absmax = torch.max(x_min.abs(), x_max.abs())
        scale = x_absmax / ((qmax - qmin) / 2)
        zero_point = torch.zeros_like(scale)
    else:
        # 非对称量化
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = qmin - x_min / scale
        zero_point = torch.round(zero_point)

    # 避免 scale 为 0
    scale = torch.clamp(scale, min=1e-8)

    return scale, zero_point


def quantize_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int
) -> torch.Tensor:
    """
    量化张量

    Args:
        x: 输入张量
        scale: 缩放因子
        zero_point: 零点
        qmin: 量化最小值
        qmax: 量化最大值

    Returns:
        量化后的张量 (整数)
    """
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.int8)


def dequantize_tensor(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    反量化张量

    Args:
        q: 量化后的张量
        scale: 缩放因子
        zero_point: 零点

    Returns:
        反量化后的浮点张量
    """
    return (q.float() - zero_point) * scale


class FakeQuantize(torch.autograd.Function):
    """
    伪量化函数

    前向传播: 量化 -> 反量化 (模拟量化效果)
    反向传播: 直通估计器 (Straight-Through Estimator)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        qmin: int,
        qmax: int
    ) -> torch.Tensor:
        # 量化
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, qmin, qmax)
        # 反量化
        x_q = (q - zero_point) * scale

        # 保存用于反向传播
        ctx.save_for_backward(x, scale)
        ctx.qmin = qmin
        ctx.qmax = qmax

        return x_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, scale = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax

        # 直通估计器: 在量化范围内传递梯度
        x_q = x / scale
        mask = (x_q >= qmin) & (x_q <= qmax)
        grad_input = grad_output * mask.float()

        return grad_input, None, None, None, None


class FakeQuantizeModule(nn.Module):
    """伪量化模块"""

    def __init__(
        self,
        qmin: int = -128,
        qmax: int = 127,
        symmetric: bool = True,
        per_channel: bool = False,
        num_channels: int = 1
    ):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.symmetric = symmetric
        self.per_channel = per_channel

        # 可学习的量化参数
        if per_channel:
            self.register_buffer("scale", torch.ones(num_channels))
            self.register_buffer("zero_point", torch.zeros(num_channels))
        else:
            self.register_buffer("scale", torch.tensor(1.0))
            self.register_buffer("zero_point", torch.tensor(0.0))

        # 统计信息
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def update_stats(self, x: torch.Tensor):
        """更新统计信息"""
        if self.per_channel:
            # 按通道统计
            x_flat = x.transpose(0, 1).reshape(x.shape[1], -1)
            min_val = x_flat.min(dim=1)[0]
            max_val = x_flat.max(dim=1)[0]
        else:
            min_val = x.min()
            max_val = x.max()

        self.min_val = torch.min(self.min_val, min_val)
        self.max_val = torch.max(self.max_val, max_val)

    def compute_qparams(self):
        """计算量化参数"""
        self.scale, self.zero_point = compute_scale_zero_point(
            self.min_val, self.max_val,
            self.qmin, self.qmax,
            self.symmetric
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update_stats(x.detach())
            self.compute_qparams()

        return FakeQuantize.apply(
            x, self.scale, self.zero_point, self.qmin, self.qmax
        )


class QuantizedLinear(nn.Module):
    """量化线性层"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.config = config or QuantizationConfig()

        # 原始权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # 量化参数
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("weight_zero_point", torch.tensor(0.0))
        self.register_buffer("quantized_weight", None)

        # 伪量化模块 (用于 QAT)
        self.weight_fake_quant = FakeQuantizeModule(
            qmin=self.config.weight_qmin,
            qmax=self.config.weight_qmax,
            symmetric=self.config.symmetric,
            per_channel=(self.config.weight_granularity == QuantizationGranularity.PER_CHANNEL),
            num_channels=out_features
        )

        self.input_fake_quant = FakeQuantizeModule(
            qmin=self.config.activation_qmin,
            qmax=self.config.activation_qmax,
            symmetric=False
        )

    def quantize_weights(self):
        """量化权重"""
        with torch.no_grad():
            w = self.weight.data
            w_min = w.min(dim=1)[0] if self.config.weight_granularity == QuantizationGranularity.PER_CHANNEL else w.min()
            w_max = w.max(dim=1)[0] if self.config.weight_granularity == QuantizationGranularity.PER_CHANNEL else w.max()

            self.weight_scale, self.weight_zero_point = compute_scale_zero_point(
                w_min, w_max,
                self.config.weight_qmin, self.config.weight_qmax,
                self.config.symmetric
            )

            # 量化权重
            if self.config.weight_granularity == QuantizationGranularity.PER_CHANNEL:
                scale = self.weight_scale.view(-1, 1)
                zp = self.weight_zero_point.view(-1, 1)
            else:
                scale = self.weight_scale
                zp = self.weight_zero_point

            self.quantized_weight = quantize_tensor(
                w, scale, zp,
                self.config.weight_qmin, self.config.weight_qmax
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.quant_type == QuantizationType.QAT:
            # QAT: 使用伪量化
            w = self.weight_fake_quant(self.weight)
            x = self.input_fake_quant(x)
            return F.linear(x, w, self.bias)
        elif self.quantized_weight is not None:
            # 推理: 使用量化权重
            if self.config.weight_granularity == QuantizationGranularity.PER_CHANNEL:
                scale = self.weight_scale.view(-1, 1)
                zp = self.weight_zero_point.view(-1, 1)
            else:
                scale = self.weight_scale
                zp = self.weight_zero_point

            w = dequantize_tensor(self.quantized_weight, scale, zp)
            return F.linear(x, w, self.bias)
        else:
            # 未量化
            return F.linear(x, self.weight, self.bias)


class QuantizedConv2d(nn.Module):
    """量化卷积层"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        config: Optional[QuantizationConfig] = None
    ):
        super().__init__()
        self.config = config or QuantizationConfig()
        self.stride = stride
        self.padding = padding

        # 原始权重
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # 量化参数
        self.register_buffer("weight_scale", torch.tensor(1.0))
        self.register_buffer("weight_zero_point", torch.tensor(0.0))
        self.register_buffer("quantized_weight", None)

    def quantize_weights(self):
        """量化权重"""
        with torch.no_grad():
            w = self.weight.data
            # Per-channel: 按输出通道
            w_flat = w.view(w.shape[0], -1)
            w_min = w_flat.min(dim=1)[0]
            w_max = w_flat.max(dim=1)[0]

            self.weight_scale, self.weight_zero_point = compute_scale_zero_point(
                w_min, w_max,
                self.config.weight_qmin, self.config.weight_qmax,
                self.config.symmetric
            )

            scale = self.weight_scale.view(-1, 1, 1, 1)
            zp = self.weight_zero_point.view(-1, 1, 1, 1)

            self.quantized_weight = quantize_tensor(
                w, scale, zp,
                self.config.weight_qmin, self.config.weight_qmax
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized_weight is not None:
            scale = self.weight_scale.view(-1, 1, 1, 1)
            zp = self.weight_zero_point.view(-1, 1, 1, 1)
            w = dequantize_tensor(self.quantized_weight, scale, zp)
        else:
            w = self.weight

        return F.conv2d(x, w, self.bias, self.stride, self.padding)


class DynamicQuantizer:
    """
    动态量化器

    动态量化在推理时动态计算激活值的量化参数，
    权重在量化时预先计算好。
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig(quant_type=QuantizationType.DYNAMIC)

    def quantize(
        self,
        model: nn.Module,
        inplace: bool = False
    ) -> nn.Module:
        """
        对模型进行动态量化

        Args:
            model: 要量化的模型
            inplace: 是否原地修改

        Returns:
            量化后的模型
        """
        if not inplace:
            model = self._copy_model(model)

        # 替换 Linear 层
        self._replace_modules(model)

        # 量化权重
        self._quantize_weights(model)

        return model

    def _copy_model(self, model: nn.Module) -> nn.Module:
        """复制模型"""
        import copy
        return copy.deepcopy(model)

    def _replace_modules(self, model: nn.Module, prefix: str = ""):
        """递归替换模块"""
        for name, child in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                # 创建量化版本
                quantized = QuantizedLinear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    config=self.config
                )
                # 复制权重
                quantized.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    quantized.bias.data = child.bias.data.clone()
                setattr(model, name, quantized)

            elif isinstance(child, nn.Conv2d):
                quantized = QuantizedConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size[0],
                    child.stride[0],
                    child.padding[0],
                    bias=child.bias is not None,
                    config=self.config
                )
                quantized.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    quantized.bias.data = child.bias.data.clone()
                setattr(model, name, quantized)

            else:
                self._replace_modules(child, full_name)

    def _quantize_weights(self, model: nn.Module):
        """量化所有权重"""
        for module in model.modules():
            if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
                module.quantize_weights()


class StaticQuantizer:
    """
    静态量化器

    静态量化使用校准数据预先计算激活值的量化参数。
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig(quant_type=QuantizationType.STATIC)
        self.activation_stats: Dict[str, Dict[str, List[float]]] = {}

    def calibrate(
        self,
        model: nn.Module,
        calibration_loader,
        num_batches: Optional[int] = None
    ):
        """
        使用校准数据收集激活值统计

        Args:
            model: 要校准的模型
            calibration_loader: 校准数据加载器
            num_batches: 校准批次数
        """
        num_batches = num_batches or self.config.num_calibration_batches
        model.eval()

        # 注册 hook 收集激活值统计
        hooks = []
        self.activation_stats = {}

        def make_hook(name):
            def hook(module, input, output):
                if name not in self.activation_stats:
                    self.activation_stats[name] = {"min": [], "max": []}
                self.activation_stats[name]["min"].append(output.min().item())
                self.activation_stats[name]["max"].append(output.max().item())
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # 运行校准数据
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                model(batch)

        # 移除 hooks
        for hook in hooks:
            hook.remove()

    def quantize(self, model: nn.Module, inplace: bool = False) -> nn.Module:
        """
        对模型进行静态量化

        Args:
            model: 要量化的模型
            inplace: 是否原地修改

        Returns:
            量化后的模型
        """
        if not self.activation_stats:
            raise RuntimeError("请先调用 calibrate() 进行校准")

        quantizer = DynamicQuantizer(self.config)
        model = quantizer.quantize(model, inplace)

        return model


class QATWrapper(nn.Module):
    """
    量化感知训练包装器

    将普通模型包装为支持 QAT 的模型。
    """

    def __init__(self, model: nn.Module, config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config or QuantizationConfig(quant_type=QuantizationType.QAT)
        self.model = model

        # 为每个需要量化的层添加伪量化模块
        self._prepare_qat()

    def _prepare_qat(self):
        """准备 QAT"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 添加权重伪量化
                module.weight_fake_quant = FakeQuantizeModule(
                    qmin=self.config.weight_qmin,
                    qmax=self.config.weight_qmax,
                    symmetric=self.config.symmetric
                )
                # 添加激活伪量化
                module.input_fake_quant = FakeQuantizeModule(
                    qmin=self.config.activation_qmin,
                    qmax=self.config.activation_qmax,
                    symmetric=False
                )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def convert_to_quantized(self) -> nn.Module:
        """
        将 QAT 模型转换为量化模型

        Returns:
            量化后的模型
        """
        self.eval()
        quantizer = DynamicQuantizer(self.config)
        return quantizer.quantize(self.model)


def quantize_model(
    model: nn.Module,
    quant_type: str = "dynamic",
    calibration_loader=None,
    config: Optional[QuantizationConfig] = None
) -> nn.Module:
    """
    量化模型的便捷函数

    Args:
        model: 要量化的模型
        quant_type: 量化类型 ("dynamic", "static", "qat")
        calibration_loader: 校准数据加载器 (静态量化需要)
        config: 量化配置

    Returns:
        量化后的模型
    """
    if quant_type == "dynamic":
        quantizer = DynamicQuantizer(config)
        return quantizer.quantize(model)

    elif quant_type == "static":
        if calibration_loader is None:
            raise ValueError("静态量化需要提供 calibration_loader")
        quantizer = StaticQuantizer(config)
        quantizer.calibrate(model, calibration_loader)
        return quantizer.quantize(model)

    elif quant_type == "qat":
        return QATWrapper(model, config)

    else:
        raise ValueError(f"未知的量化类型: {quant_type}")


def calibrate_model(
    model: nn.Module,
    calibration_loader,
    num_batches: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    校准模型并返回激活值统计

    Args:
        model: 要校准的模型
        calibration_loader: 校准数据加载器
        num_batches: 校准批次数

    Returns:
        激活值统计字典
    """
    quantizer = StaticQuantizer()
    quantizer.calibrate(model, calibration_loader, num_batches)
    return quantizer.activation_stats
