"""
AWQ: Activation-aware Weight Quantization

============================================================
核心思想 (Core Idea)
============================================================
AWQ 观察到权重的重要性不均匀：少数"显著"通道对激活值影响巨大。
通过保护这些显著通道，可以在极低比特量化下保持模型质量。

============================================================
关键创新 (Key Innovations)
============================================================
1. 激活感知: 根据激活值分布识别显著权重通道
2. 通道缩放: 对显著通道应用缩放因子，减少量化误差
3. 无需训练: 仅需少量校准数据，无需反向传播
4. 硬件友好: 缩放因子可融合到相邻层

============================================================
数学原理 (Mathematical Principle)
============================================================
对于线性层 Y = XW，量化误差为:
    E = ||XW - X·Q(W)||

AWQ 引入缩放因子 s:
    Y = X·(s⁻¹)·(s·W) = X'·W'

对 W' = s·W 量化后:
    E' = ||X'·W' - X'·Q(W')|| = ||X·s⁻¹·(s·W - Q(s·W))||

当 s 较大时，Q(s·W) ≈ s·W，误差减小。

============================================================
参考文献 (References)
============================================================
[1] Lin, J., et al. (2023). AWQ: Activation-aware Weight Quantization 
    for LLM Compression and Acceleration. MLSys 2024.
[2] https://github.com/mit-han-lab/llm-awq
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np

# 尝试导入 torch，如果不可用则使用 numpy 模拟
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

__all__ = [
    "AWQConfig",
    "AWQQuantizer",
    "AWQLinear",
    "ActivationObserver",
    "SalientChannelFinder",
    "create_awq_quantizer",
    "quantize_model_awq",
]


# =============================================================================
# 配置
# =============================================================================

class AWQGranularity(Enum):
    """AWQ 量化粒度"""
    PER_CHANNEL = "per_channel"  # 每通道量化
    PER_GROUP = "per_group"      # 分组量化


@dataclass
class AWQConfig:
    """AWQ 量化配置。
    
    Args:
        w_bit: 权重量化位数
        group_size: 分组大小 (per_group 模式)
        zero_point: 是否使用零点 (非对称量化)
        granularity: 量化粒度
        salient_ratio: 显著通道比例
        alpha: 缩放因子搜索的 alpha 参数
        n_grid: 网格搜索的步数
        max_seq_len: 校准时的最大序列长度
        n_samples: 校准样本数
    """
    w_bit: int = 4
    group_size: int = 128
    zero_point: bool = True
    granularity: AWQGranularity = AWQGranularity.PER_GROUP
    salient_ratio: float = 0.01
    alpha: float = 0.5
    n_grid: int = 20
    max_seq_len: int = 512
    n_samples: int = 128
    
    def __post_init__(self):
        if self.w_bit not in [2, 3, 4, 8]:
            raise ValueError(f"w_bit must be 2, 3, 4, or 8, got {self.w_bit}")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if not 0 < self.salient_ratio < 1:
            raise ValueError("salient_ratio must be in (0, 1)")
        if isinstance(self.granularity, str):
            self.granularity = AWQGranularity(self.granularity)


# =============================================================================
# 激活观察器
# =============================================================================

class ActivationObserver:
    """激活值观察器：收集激活值统计信息。
    
    用于识别显著通道和计算最优缩放因子。
    
    Attributes:
        running_mean: 运行均值
        running_var: 运行方差
        running_max: 运行最大值
        n_samples: 观察的样本数
    """
    
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)
        self.running_max = np.zeros(num_features)
        self.running_min = np.full(num_features, float('inf'))
        self.n_samples = 0
    
    def observe(self, x: np.ndarray) -> None:
        """观察一批激活值。
        
        Args:
            x: 激活值，形状为 (..., num_features)
        """
        # 展平到 (N, num_features)
        x_flat = x.reshape(-1, self.num_features)
        batch_size = x_flat.shape[0]
        
        # 更新统计量
        batch_mean = x_flat.mean(axis=0)
        batch_var = x_flat.var(axis=0)
        batch_max = np.abs(x_flat).max(axis=0)
        batch_min = x_flat.min(axis=0)
        
        # Welford's online algorithm for mean and variance
        if self.n_samples == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
            self.running_max = batch_max
            self.running_min = batch_min
        else:
            delta = batch_mean - self.running_mean
            total = self.n_samples + batch_size
            self.running_mean += delta * batch_size / total
            self.running_var = (
                (self.running_var * self.n_samples + batch_var * batch_size) / total
                + delta ** 2 * self.n_samples * batch_size / total ** 2
            )
            self.running_max = np.maximum(self.running_max, batch_max)
            self.running_min = np.minimum(self.running_min, batch_min)
        
        self.n_samples += batch_size
    
    def get_scale_importance(self) -> np.ndarray:
        """获取每个通道的重要性分数。
        
        基于激活值的绝对值均值，值越大表示该通道越重要。
        """
        return np.abs(self.running_mean) + np.sqrt(self.running_var + 1e-8)
    
    def get_activation_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取激活值范围。"""
        return self.running_min, self.running_max
    
    def reset(self) -> None:
        """重置观察器。"""
        self.running_mean = np.zeros(self.num_features)
        self.running_var = np.zeros(self.num_features)
        self.running_max = np.zeros(self.num_features)
        self.running_min = np.full(self.num_features, float('inf'))
        self.n_samples = 0


# =============================================================================
# 显著通道查找器
# =============================================================================

class SalientChannelFinder:
    """显著通道查找器：识别对输出影响最大的权重通道。
    
    AWQ 的核心思想是保护显著通道，减少量化误差。
    
    Attributes:
        config: AWQ 配置
        observers: 每层的激活观察器
    """
    
    def __init__(self, config: AWQConfig):
        self.config = config
        self.observers: Dict[str, ActivationObserver] = {}
    
    def register_layer(self, name: str, num_features: int) -> None:
        """注册一个层进行观察。"""
        self.observers[name] = ActivationObserver(num_features)
    
    def observe_activation(self, name: str, x: np.ndarray) -> None:
        """观察指定层的激活值。"""
        if name in self.observers:
            self.observers[name].observe(x)
    
    def find_salient_channels(self, name: str) -> np.ndarray:
        """找到指定层的显著通道索引。
        
        Args:
            name: 层名称
            
        Returns:
            显著通道的索引数组
        """
        if name not in self.observers:
            raise ValueError(f"Layer {name} not registered")
        
        observer = self.observers[name]
        importance = observer.get_scale_importance()
        
        # 选择 top-k 显著通道
        k = max(1, int(len(importance) * self.config.salient_ratio))
        salient_indices = np.argsort(importance)[-k:]
        
        return salient_indices
    
    def compute_optimal_scales(
        self,
        name: str,
        weight: np.ndarray
    ) -> np.ndarray:
        """计算最优缩放因子。
        
        通过网格搜索找到最小化量化误差的缩放因子。
        
        Args:
            name: 层名称
            weight: 权重矩阵，形状为 (out_features, in_features)
            
        Returns:
            每个输入通道的缩放因子
        """
        if name not in self.observers:
            return np.ones(weight.shape[1])
        
        observer = self.observers[name]
        importance = observer.get_scale_importance()
        
        # 初始化缩放因子
        scales = np.ones(weight.shape[1])
        
        # 对显著通道搜索最优缩放
        salient_indices = self.find_salient_channels(name)
        
        for idx in salient_indices:
            best_scale = 1.0
            best_error = float('inf')
            
            # 网格搜索
            for i in range(self.config.n_grid):
                ratio = self.config.alpha + (1 - self.config.alpha) * i / self.config.n_grid
                scale = importance[idx] ** ratio
                
                # 计算量化误差
                scaled_weight = weight[:, idx] * scale
                q_weight = self._pseudo_quantize(scaled_weight)
                error = np.mean((scaled_weight - q_weight) ** 2)
                
                if error < best_error:
                    best_error = error
                    best_scale = scale
            
            scales[idx] = best_scale
        
        return scales
    
    def _pseudo_quantize(self, x: np.ndarray) -> np.ndarray:
        """伪量化：模拟量化效果。"""
        qmin = 0 if self.config.zero_point else -(2 ** (self.config.w_bit - 1))
        qmax = 2 ** self.config.w_bit - 1 if self.config.zero_point else 2 ** (self.config.w_bit - 1) - 1
        
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / (qmax - qmin) if x_max != x_min else 1.0
        zero_point = qmin - x_min / scale if self.config.zero_point else 0
        
        q = np.clip(np.round(x / scale + zero_point), qmin, qmax)
        return (q - zero_point) * scale


# =============================================================================
# AWQ 量化线性层
# =============================================================================

class AWQLinear:
    """AWQ 量化的线性层。
    
    存储量化后的权重和缩放因子，支持高效推理。
    
    Attributes:
        in_features: 输入特征数
        out_features: 输出特征数
        w_bit: 量化位数
        group_size: 分组大小
        qweight: 量化后的权重
        scales: 缩放因子
        zeros: 零点
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: AWQConfig,
        bias: bool = True
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.w_bit = config.w_bit
        self.group_size = config.group_size
        
        # 计算分组数
        self.n_groups = (in_features + config.group_size - 1) // config.group_size
        
        # 量化参数
        self.qweight: Optional[np.ndarray] = None  # (out_features, in_features)
        self.scales: Optional[np.ndarray] = None   # (out_features, n_groups)
        self.zeros: Optional[np.ndarray] = None    # (out_features, n_groups)
        self.bias_data: Optional[np.ndarray] = None
        
        # AWQ 缩放因子
        self.awq_scales: Optional[np.ndarray] = None  # (in_features,)
    
    def quantize_weight(
        self,
        weight: np.ndarray,
        awq_scales: Optional[np.ndarray] = None
    ) -> None:
        """量化权重。
        
        Args:
            weight: 原始权重，形状为 (out_features, in_features)
            awq_scales: AWQ 缩放因子
        """
        if awq_scales is not None:
            self.awq_scales = awq_scales
            # 应用 AWQ 缩放
            weight = weight * awq_scales[np.newaxis, :]
        
        out_features, in_features = weight.shape
        
        # 分组量化
        scales_list = []
        zeros_list = []
        qweight_list = []
        
        for g in range(self.n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, in_features)
            w_group = weight[:, start:end]
            
            # 计算量化参数
            w_min = w_group.min(axis=1, keepdims=True)
            w_max = w_group.max(axis=1, keepdims=True)
            
            qmin = 0
            qmax = 2 ** self.w_bit - 1
            
            scale = (w_max - w_min) / (qmax - qmin)
            scale = np.where(scale == 0, 1.0, scale)
            zero = qmin - w_min / scale
            
            # 量化
            q = np.clip(np.round(w_group / scale + zero), qmin, qmax).astype(np.int32)
            
            scales_list.append(scale)
            zeros_list.append(zero)
            qweight_list.append(q)
        
        self.scales = np.concatenate(scales_list, axis=1)
        self.zeros = np.concatenate(zeros_list, axis=1)
        self.qweight = np.concatenate(qweight_list, axis=1)
    
    def dequantize_weight(self) -> np.ndarray:
        """反量化权重。"""
        if self.qweight is None:
            raise ValueError("Weight not quantized yet")
        
        weight = np.zeros((self.out_features, self.in_features))
        
        for g in range(self.n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            
            scale = self.scales[:, g:g+1]
            zero = self.zeros[:, g:g+1]
            
            weight[:, start:end] = (self.qweight[:, start:end] - zero) * scale
        
        # 反向应用 AWQ 缩放
        if self.awq_scales is not None:
            weight = weight / self.awq_scales[np.newaxis, :]
        
        return weight
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。
        
        Args:
            x: 输入，形状为 (..., in_features)
            
        Returns:
            输出，形状为 (..., out_features)
        """
        # 应用 AWQ 缩放到输入
        if self.awq_scales is not None:
            x = x / self.awq_scales
        
        # 反量化并计算
        weight = self.dequantize_weight()
        output = x @ weight.T
        
        if self.bias_data is not None:
            output = output + self.bias_data
        
        return output
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """获取内存占用（字节）。"""
        # 量化权重
        qweight_bits = self.out_features * self.in_features * self.w_bit
        qweight_bytes = (qweight_bits + 7) // 8
        
        # 缩放因子和零点 (FP16)
        scales_bytes = self.out_features * self.n_groups * 2
        zeros_bytes = self.out_features * self.n_groups * 2
        
        # AWQ 缩放因子
        awq_scales_bytes = self.in_features * 2 if self.awq_scales is not None else 0
        
        # 偏置
        bias_bytes = self.out_features * 4 if self.bias_data is not None else 0
        
        return {
            "qweight": qweight_bytes,
            "scales": scales_bytes,
            "zeros": zeros_bytes,
            "awq_scales": awq_scales_bytes,
            "bias": bias_bytes,
            "total": qweight_bytes + scales_bytes + zeros_bytes + awq_scales_bytes + bias_bytes
        }


# =============================================================================
# AWQ 量化器
# =============================================================================

class AWQQuantizer:
    """AWQ 量化器：执行完整的 AWQ 量化流程。
    
    流程:
    1. 收集激活值统计
    2. 识别显著通道
    3. 计算最优缩放因子
    4. 量化权重
    
    Attributes:
        config: AWQ 配置
        finder: 显著通道查找器
        quantized_layers: 量化后的层
    """
    
    def __init__(self, config: AWQConfig):
        self.config = config
        self.finder = SalientChannelFinder(config)
        self.quantized_layers: Dict[str, AWQLinear] = {}
        
        # 统计信息
        self._original_size = 0
        self._quantized_size = 0
    
    def calibrate(
        self,
        layer_name: str,
        activations: List[np.ndarray]
    ) -> None:
        """校准：收集激活值统计。
        
        Args:
            layer_name: 层名称
            activations: 激活值列表
        """
        if not activations:
            return
        
        num_features = activations[0].shape[-1]
        self.finder.register_layer(layer_name, num_features)
        
        for act in activations:
            self.finder.observe_activation(layer_name, act)
    
    def quantize_layer(
        self,
        layer_name: str,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None
    ) -> AWQLinear:
        """量化单个层。
        
        Args:
            layer_name: 层名称
            weight: 权重矩阵
            bias: 偏置向量
            
        Returns:
            量化后的 AWQLinear 层
        """
        out_features, in_features = weight.shape
        
        # 计算最优缩放因子
        awq_scales = self.finder.compute_optimal_scales(layer_name, weight)
        
        # 创建量化层
        q_layer = AWQLinear(in_features, out_features, self.config)
        q_layer.quantize_weight(weight, awq_scales)
        
        if bias is not None:
            q_layer.bias_data = bias
        
        # 记录统计
        self._original_size += weight.size * 4  # FP32
        self._quantized_size += q_layer.get_memory_footprint()["total"]
        
        self.quantized_layers[layer_name] = q_layer
        return q_layer
    
    def get_compression_ratio(self) -> float:
        """获取压缩比。"""
        if self._quantized_size == 0:
            return 1.0
        return self._original_size / self._quantized_size
    
    def get_stats(self) -> Dict[str, Any]:
        """获取量化统计信息。"""
        return {
            "num_layers": len(self.quantized_layers),
            "original_size_mb": self._original_size / (1024 * 1024),
            "quantized_size_mb": self._quantized_size / (1024 * 1024),
            "compression_ratio": self.get_compression_ratio(),
            "w_bit": self.config.w_bit,
            "group_size": self.config.group_size,
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_awq_quantizer(
    w_bit: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    salient_ratio: float = 0.01,
    **kwargs
) -> AWQQuantizer:
    """创建 AWQ 量化器的工厂函数。
    
    Args:
        w_bit: 量化位数 (2, 3, 4, 8)
        group_size: 分组大小
        zero_point: 是否使用零点
        salient_ratio: 显著通道比例
        **kwargs: 其他配置参数
        
    Returns:
        AWQQuantizer 实例
        
    Example:
        >>> quantizer = create_awq_quantizer(w_bit=4, group_size=128)
        >>> quantizer.calibrate("layer1", activations)
        >>> q_layer = quantizer.quantize_layer("layer1", weight)
    """
    config = AWQConfig(
        w_bit=w_bit,
        group_size=group_size,
        zero_point=zero_point,
        salient_ratio=salient_ratio,
        **kwargs
    )
    return AWQQuantizer(config)


def quantize_model_awq(
    weights: Dict[str, np.ndarray],
    activations: Dict[str, List[np.ndarray]],
    w_bit: int = 4,
    group_size: int = 128,
    **kwargs
) -> Tuple[Dict[str, AWQLinear], Dict[str, Any]]:
    """对整个模型进行 AWQ 量化。
    
    Args:
        weights: 层名称到权重的映射
        activations: 层名称到激活值列表的映射
        w_bit: 量化位数
        group_size: 分组大小
        **kwargs: 其他配置参数
        
    Returns:
        (quantized_layers, stats): 量化后的层和统计信息
        
    Example:
        >>> weights = {"fc1": np.random.randn(512, 768)}
        >>> activations = {"fc1": [np.random.randn(32, 768)]}
        >>> q_layers, stats = quantize_model_awq(weights, activations)
    """
    quantizer = create_awq_quantizer(w_bit=w_bit, group_size=group_size, **kwargs)
    
    # 校准
    for name, acts in activations.items():
        quantizer.calibrate(name, acts)
    
    # 量化
    for name, weight in weights.items():
        quantizer.quantize_layer(name, weight)
    
    return quantizer.quantized_layers, quantizer.get_stats()


# =============================================================================
# 辅助函数
# =============================================================================

def compute_quantization_error(
    original: np.ndarray,
    quantized: np.ndarray
) -> Dict[str, float]:
    """计算量化误差。
    
    Args:
        original: 原始权重
        quantized: 量化后的权重
        
    Returns:
        误差统计字典
    """
    diff = original - quantized
    
    return {
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(np.abs(diff))),
        "max_error": float(np.max(np.abs(diff))),
        "relative_error": float(np.linalg.norm(diff) / (np.linalg.norm(original) + 1e-8)),
        "snr_db": float(10 * np.log10(np.mean(original ** 2) / (np.mean(diff ** 2) + 1e-10))),
    }


def estimate_model_size(
    num_params: int,
    w_bit: int = 4,
    group_size: int = 128
) -> Dict[str, float]:
    """估算量化后的模型大小。
    
    Args:
        num_params: 参数数量
        w_bit: 量化位数
        group_size: 分组大小
        
    Returns:
        大小估算字典 (MB)
    """
    # 原始大小 (FP32)
    original_size = num_params * 4
    
    # 量化权重大小
    qweight_size = num_params * w_bit / 8
    
    # 缩放因子和零点 (FP16)
    n_groups = num_params / group_size
    scales_size = n_groups * 2
    zeros_size = n_groups * 2
    
    quantized_size = qweight_size + scales_size + zeros_size
    
    return {
        "original_mb": original_size / (1024 * 1024),
        "quantized_mb": quantized_size / (1024 * 1024),
        "compression_ratio": original_size / quantized_size,
        "memory_saved_mb": (original_size - quantized_size) / (1024 * 1024),
    }


def pack_int4_weights(weights: np.ndarray) -> np.ndarray:
    """将 INT4 权重打包为 INT8 存储。
    
    两个 INT4 值打包到一个 INT8 中。
    
    Args:
        weights: INT4 权重 (值范围 0-15)
        
    Returns:
        打包后的 INT8 数组
    """
    assert weights.dtype in [np.int32, np.int64, np.uint8]
    weights = weights.astype(np.uint8)
    
    # 确保偶数长度
    flat = weights.flatten()
    if len(flat) % 2 != 0:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    
    # 打包: 低 4 位 + 高 4 位
    packed = (flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)
    
    return packed


def unpack_int4_weights(packed: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
    """解包 INT4 权重。
    
    Args:
        packed: 打包的 INT8 数组
        original_shape: 原始形状
        
    Returns:
        解包后的 INT4 权重
    """
    # 解包
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    
    unpacked = np.empty(len(packed) * 2, dtype=np.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high
    
    # 恢复形状
    total_elements = np.prod(original_shape)
    return unpacked[:total_elements].reshape(original_shape)
