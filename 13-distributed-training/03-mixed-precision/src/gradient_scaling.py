"""
梯度缩放 (Gradient Scaling) 实现

梯度缩放用于防止 FP16 训练中的梯度下溢问题。
通过放大损失值来放大梯度，然后在优化器步骤前反缩放。

核心概念:
    - 损失缩放: 放大损失以放大梯度
    - 梯度反缩放: 优化前恢复原始梯度
    - 动态缩放: 根据梯度溢出情况自动调整缩放因子
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class GradScalerConfig:
    """梯度缩放器配置
    
    Attributes:
        init_scale: 初始缩放因子
        growth_factor: 增长因子
        backoff_factor: 回退因子
        growth_interval: 增长间隔（步数）
        max_scale: 最大缩放因子
        min_scale: 最小缩放因子
        enabled: 是否启用
    """
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_scale: float = 2.0 ** 24
    min_scale: float = 1.0
    enabled: bool = True


class SmartGradScaler:
    """智能梯度缩放器
    
    相比 PyTorch 原生 GradScaler，增加了更多控制选项。
    
    Args:
        config: 缩放器配置
    """
    
    def __init__(self, config: Optional[GradScalerConfig] = None):
        self.config = config or GradScalerConfig()
        
        self._scale = torch.tensor(
            self.config.init_scale,
            dtype=torch.float32,
        )
        self._growth_tracker = 0
        self._found_inf = torch.tensor(0.0)
        self._enabled = self.config.enabled
        
        # 统计信息
        self._overflow_count = 0
        self._total_steps = 0
    
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失值
        
        Args:
            loss: 原始损失
            
        Returns:
            缩放后的损失
        """
        if not self._enabled:
            return loss
        
        return loss * self._scale.to(loss.device)
    
    def unscale_(self, optimizer: Optimizer) -> None:
        """反缩放优化器中的梯度
        
        Args:
            optimizer: 优化器
        """
        if not self._enabled:
            return
        
        inv_scale = 1.0 / self._scale
        self._found_inf.zero_()
        
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    # 反缩放梯度
                    param.grad.mul_(inv_scale.to(param.grad.device))
                    
                    # 检查 inf/nan
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        self._found_inf.fill_(1.0)
    
    def step(self, optimizer: Optimizer) -> bool:
        """执行优化器步骤
        
        Args:
            optimizer: 优化器
            
        Returns:
            是否成功执行步骤（无溢出）
        """
        if not self._enabled:
            optimizer.step()
            return True
        
        self._total_steps += 1
        
        if self._found_inf.item() > 0:
            # 发现溢出，跳过此步骤
            self._overflow_count += 1
            return False
        
        optimizer.step()
        return True
    
    def update(self) -> None:
        """更新缩放因子"""
        if not self._enabled:
            return
        
        if self._found_inf.item() > 0:
            # 发现溢出，减小缩放因子
            new_scale = self._scale * self.config.backoff_factor
            self._scale = torch.clamp(
                new_scale,
                min=self.config.min_scale,
                max=self.config.max_scale,
            )
            self._growth_tracker = 0
        else:
            # 无溢出，考虑增大缩放因子
            self._growth_tracker += 1
            if self._growth_tracker >= self.config.growth_interval:
                new_scale = self._scale * self.config.growth_factor
                self._scale = torch.clamp(
                    new_scale,
                    min=self.config.min_scale,
                    max=self.config.max_scale,
                )
                self._growth_tracker = 0
        
        self._found_inf.zero_()
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        return self._scale.item()
    
    def get_overflow_ratio(self) -> float:
        """获取溢出比例"""
        if self._total_steps == 0:
            return 0.0
        return self._overflow_count / self._total_steps
    
    def state_dict(self) -> Dict:
        """获取状态字典"""
        return {
            "scale": self._scale.item(),
            "growth_tracker": self._growth_tracker,
            "overflow_count": self._overflow_count,
            "total_steps": self._total_steps,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """加载状态字典"""
        self._scale = torch.tensor(state["scale"], dtype=torch.float32)
        self._growth_tracker = state["growth_tracker"]
        self._overflow_count = state.get("overflow_count", 0)
        self._total_steps = state.get("total_steps", 0)


class DynamicLossScaler:
    """动态损失缩放器
    
    更激进的动态缩放策略，适用于不稳定的训练。
    
    Args:
        init_scale: 初始缩放因子
        scale_window: 缩放窗口大小
        min_scale: 最小缩放因子
        max_scale: 最大缩放因子
    """
    
    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        scale_window: int = 1000,
        min_scale: float = 1.0,
        max_scale: float = 2.0 ** 24,
    ):
        self.scale = init_scale
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self._overflow_history: List[bool] = []
        self._step_count = 0
    
    def loss_scale(self) -> float:
        """获取当前损失缩放因子"""
        return self.scale
    
    def update_scale(self, overflow: bool) -> None:
        """更新缩放因子
        
        Args:
            overflow: 是否发生溢出
        """
        self._overflow_history.append(overflow)
        self._step_count += 1
        
        # 保持窗口大小
        if len(self._overflow_history) > self.scale_window:
            self._overflow_history.pop(0)
        
        if overflow:
            # 立即减半
            self.scale = max(self.scale / 2, self.min_scale)
        elif self._step_count % self.scale_window == 0:
            # 检查窗口内的溢出率
            overflow_rate = sum(self._overflow_history) / len(self._overflow_history)
            
            if overflow_rate < 0.01:
                # 溢出率很低，增大缩放因子
                self.scale = min(self.scale * 2, self.max_scale)
            elif overflow_rate > 0.1:
                # 溢出率较高，减小缩放因子
                self.scale = max(self.scale / 2, self.min_scale)
    
    def has_overflow(self, params: List[nn.Parameter]) -> bool:
        """检查参数梯度是否溢出
        
        Args:
            params: 参数列表
            
        Returns:
            是否存在溢出
        """
        for param in params:
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    return True
        return False
    
    def state_dict(self) -> Dict:
        """获取状态字典"""
        return {
            "scale": self.scale,
            "step_count": self._step_count,
            "overflow_history": self._overflow_history[-100:],  # 只保存最近100个
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """加载状态字典"""
        self.scale = state["scale"]
        self._step_count = state["step_count"]
        self._overflow_history = state.get("overflow_history", [])
