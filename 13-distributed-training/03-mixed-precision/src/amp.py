"""
Automatic Mixed Precision (AMP) 自动混合精度训练实现

================================================================================
核心思想 (一句话理解)
================================================================================
AMP = 前向/反向用FP16加速 + 权重更新用FP32保精度 + 损失缩放防下溢

================================================================================
为什么需要AMP？(问题背景)
================================================================================

    FP32训练的问题:
    ┌─────────────────────────────────────────────────────────────────┐
    │  - 显存占用大: 每个参数4字节，激活值也是4字节                     │
    │  - 计算速度慢: 无法利用Tensor Core的FP16加速                     │
    │  - 带宽瓶颈: 数据传输量大                                        │
    └─────────────────────────────────────────────────────────────────┘

    AMP的解决方案:
    ┌─────────────────────────────────────────────────────────────────┐
    │  前向传播: FP16 (快，省显存)                                     │
    │  反向传播: FP16 (快，省显存)                                     │
    │  权重更新: FP32 (保持精度)                                       │
    │                                                                 │
    │  效果: 显存减半，速度翻倍，精度不损失                             │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
工作原理 (图解)
================================================================================

    AMP训练流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  1. autocast上下文:                                              │
    │     ┌─────────────────────────────────────────────────────┐    │
    │     │  with autocast():                                    │    │
    │     │      # 矩阵乘法、卷积 → 自动用FP16                    │    │
    │     │      # LayerNorm、Softmax → 保持FP32                 │    │
    │     │      output = model(input)                           │    │
    │     └─────────────────────────────────────────────────────┘    │
    │                                                                 │
    │  2. 损失缩放 (防止梯度下溢):                                     │
    │     ┌─────────────────────────────────────────────────────┐    │
    │     │  loss = criterion(output, target)                    │    │
    │     │  scaled_loss = scaler.scale(loss)  # loss × 65536   │    │
    │     │  scaled_loss.backward()            # 梯度也×65536    │    │
    │     └─────────────────────────────────────────────────────┘    │
    │                                                                 │
    │  3. 梯度还原 + 更新:                                             │
    │     ┌─────────────────────────────────────────────────────┐    │
    │     │  scaler.unscale_(optimizer)  # 梯度 ÷ 65536         │    │
    │     │  # 检查是否有inf/nan                                  │    │
    │     │  scaler.step(optimizer)      # 无溢出则更新          │    │
    │     │  scaler.update()             # 调整缩放因子          │    │
    │     └─────────────────────────────────────────────────────┘    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
autocast自动选择精度的规则
================================================================================
    FP16操作 (加速):
    - 矩阵乘法 (Linear, Conv)
    - 批量矩阵乘法 (BMM)

    FP32操作 (保精度):
    - LayerNorm, BatchNorm
    - Softmax
    - 损失函数
    - 小张量运算

================================================================================
前置知识
================================================================================
- 浮点数表示 (FP32 vs FP16的范围和精度)
- 深度学习训练流程 (前向、反向、优化器更新)
- 梯度下溢/溢出的概念

================================================================================
参考文献
================================================================================
- Micikevicius et al., "Mixed Precision Training", ICLR 2018
- NVIDIA Apex documentation
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class AMPConfig:
    """自动混合精度配置

    Attributes:
        enabled: 是否启用AMP
        dtype: 混合精度数据类型 (torch.float16 或 torch.bfloat16)
        cache_enabled: 是否缓存autocast的kernel选择 (提升性能)
        use_grad_scaler: 是否使用梯度缩放 (FP16必须，BF16不需要)
        init_scale: 初始损失缩放因子 (默认65536)
        growth_factor: 缩放因子增长倍数 (默认2.0)
        backoff_factor: 溢出时缩放因子回退倍数 (默认0.5)
        growth_interval: 连续多少步无溢出后增大缩放因子 (默认2000)
        max_scale: 最大缩放因子

    Example:
        >>> config = AMPConfig(
        ...     enabled=True,
        ...     dtype=torch.float16,
        ...     use_grad_scaler=True,
        ... )
    """
    enabled: bool = True
    dtype: torch.dtype = torch.float16
    cache_enabled: bool = True
    use_grad_scaler: bool = True
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_scale: float = 2.0 ** 24


# =============================================================================
# 辅助函数
# =============================================================================

def get_autocast_dtype(dtype_str: str = "float16") -> torch.dtype:
    """将字符串转换为torch数据类型

    Args:
        dtype_str: 数据类型字符串 ("float16", "fp16", "bfloat16", "bf16")

    Returns:
        对应的torch.dtype

    Example:
        >>> dtype = get_autocast_dtype("fp16")
        >>> dtype
        torch.float16
    """
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.float16)


@contextmanager
def autocast_context(
    device_type: str = "cuda",
    dtype: torch.dtype = torch.float16,
    enabled: bool = True,
    cache_enabled: bool = True,
) -> Generator[None, None, None]:
    """自动混合精度上下文管理器

    在这个上下文中，PyTorch会自动选择合适的精度执行操作。

    Args:
        device_type: 设备类型 ("cuda" 或 "cpu")
        dtype: 目标数据类型
        enabled: 是否启用autocast
        cache_enabled: 是否缓存kernel选择

    Example:
        >>> with autocast_context(dtype=torch.float16):
        ...     output = model(input)  # 自动使用FP16
    """
    with autocast(
        device_type=device_type,
        dtype=dtype,
        enabled=enabled,
        cache_enabled=cache_enabled,
    ):
        yield


# =============================================================================
# AMP训练器
# =============================================================================

class AMPTrainer:
    """自动混合精度训练器

    封装了autocast上下文和梯度缩放，简化混合精度训练流程。

    工作流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. with trainer.autocast():  # 进入混合精度上下文              │
    │         output = model(input)                                   │
    │         loss = criterion(output, target)                        │
    │                                                                 │
    │  2. trainer.backward(loss)    # 缩放损失并反向传播              │
    │                                                                 │
    │  3. trainer.step(optimizer)   # 还原梯度并更新权重              │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        model: PyTorch模型
        config: AMP配置
        device: 训练设备

    Example:
        >>> trainer = AMPTrainer(model, AMPConfig())
        >>> with trainer.autocast():
        ...     loss = model(batch)
        >>> trainer.backward(loss)
        >>> trainer.step(optimizer)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AMPConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or AMPConfig()

        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.device_type = "cuda" if device.type == "cuda" else "cpu"

        # 将模型移到设备
        self.model = model.to(self.device)

        # 初始化梯度缩放器 (仅FP16需要)
        self.scaler = None
        if self.config.use_grad_scaler and self.config.enabled:
            self.scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                enabled=self.config.enabled,
            )

    def autocast(self) -> autocast:
        """获取autocast上下文管理器

        Returns:
            autocast上下文管理器

        Example:
            >>> with trainer.autocast():
            ...     output = model(input)
        """
        return autocast(
            device_type=self.device_type,
            dtype=self.config.dtype,
            enabled=self.config.enabled,
            cache_enabled=self.config.cache_enabled,
        )

    def forward(self, *args, **kwargs) -> Any:
        """带autocast的前向传播

        Args:
            *args: 模型输入参数
            **kwargs: 模型输入关键字参数

        Returns:
            模型输出
        """
        with self.autocast():
            return self.model(*args, **kwargs)

    def backward(self, loss: torch.Tensor) -> None:
        """带梯度缩放的反向传播

        如果启用了梯度缩放，会先放大损失再反向传播。

        Args:
            loss: 损失张量
        """
        if self.scaler is not None:
            # 缩放损失: loss × scale_factor
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """优化器更新步骤

        如果启用了梯度缩放:
        1. 还原梯度 (÷ scale_factor)
        2. 检查是否有inf/nan
        3. 无溢出则更新权重
        4. 调整缩放因子

        Args:
            optimizer: 优化器
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """手动还原梯度

        在梯度裁剪之前需要先还原梯度。

        Args:
            optimizer: 优化器

        Example:
            >>> trainer.backward(loss)
            >>> trainer.unscale_gradients(optimizer)  # 先还原
            >>> torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 再裁剪
            >>> trainer.step(optimizer)
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def clip_gradients(
        self,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """梯度裁剪

        注意: 调用此方法前需要先调用unscale_gradients()

        Args:
            max_norm: 最大梯度范数
            norm_type: 范数类型 (默认L2范数)

        Returns:
            裁剪前的梯度范数
        """
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
        )

    def get_scale(self) -> float:
        """获取当前损失缩放因子

        Returns:
            当前缩放因子，未启用缩放时返回1.0
        """
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0

    def state_dict(self) -> Dict[str, Any]:
        """获取训练器状态 (用于保存检查点)

        Returns:
            包含模型和缩放器状态的字典
        """
        state = {"model": self.model.state_dict()}
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """加载训练器状态 (用于恢复检查点)

        Args:
            state: 状态字典
        """
        self.model.load_state_dict(state["model"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
