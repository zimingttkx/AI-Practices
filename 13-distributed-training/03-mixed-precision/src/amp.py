"""
自动混合精度 (AMP) 训练实现

AMP 通过在前向传播中使用低精度（FP16/BF16），在反向传播中使用高精度（FP32），
来加速训练并减少内存占用，同时保持模型精度。

核心概念:
    - autocast: 自动选择合适的精度
    - GradScaler: 梯度缩放防止下溢
    - 白名单/黑名单: 控制哪些操作使用低精度
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


@dataclass
class AMPConfig:
    """AMP 配置类
    
    Attributes:
        enabled: 是否启用 AMP
        dtype: 混合精度数据类型 (float16, bfloat16)
        cache_enabled: 是否启用 autocast 缓存
        use_grad_scaler: 是否使用梯度缩放器
        init_scale: 初始缩放因子
        growth_factor: 缩放因子增长率
        backoff_factor: 缩放因子回退率
        growth_interval: 增长间隔
        max_scale: 最大缩放因子
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


def get_autocast_dtype(dtype_str: str = "float16") -> torch.dtype:
    """获取 autocast 数据类型
    
    Args:
        dtype_str: 数据类型字符串
        
    Returns:
        torch.dtype
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
    """autocast 上下文管理器
    
    Args:
        device_type: 设备类型 ("cuda", "cpu")
        dtype: 数据类型
        enabled: 是否启用
        cache_enabled: 是否启用缓存
    """
    with autocast(
        device_type=device_type,
        dtype=dtype,
        enabled=enabled,
        cache_enabled=cache_enabled,
    ):
        yield


class AMPTrainer:
    """AMP 训练器
    
    封装了 AMP 训练的常用操作。
    
    Args:
        model: PyTorch 模型
        config: AMP 配置
        device: 训练设备
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[AMPConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or AMPConfig()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        
        self.model = model.to(self.device)
        
        # 初始化梯度缩放器
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
        """获取 autocast 上下文"""
        return autocast(
            device_type=self.device_type,
            dtype=self.config.dtype,
            enabled=self.config.enabled,
            cache_enabled=self.config.cache_enabled,
        )
    
    def forward(self, *args, **kwargs) -> Any:
        """带 autocast 的前向传播"""
        with self.autocast():
            return self.model(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """带梯度缩放的反向传播"""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """带梯度缩放的优化器步骤"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """反缩放梯度（用于梯度裁剪）"""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def clip_gradients(
        self,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """梯度裁剪"""
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
        )
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = {"model": self.model.state_dict()}
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """加载状态字典"""
        self.model.load_state_dict(state["model"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
