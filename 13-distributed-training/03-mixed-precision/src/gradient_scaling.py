"""
Gradient Scaling 梯度缩放实现

================================================================================
核心思想 (一句话理解)
================================================================================
梯度缩放 = 放大损失防止梯度下溢 + 还原梯度保持正确性 + 动态调整找最优缩放因子

================================================================================
为什么需要梯度缩放？(问题背景)
================================================================================

    FP16的数值范围问题:
    ┌─────────────────────────────────────────────────────────────────┐
    │  FP16可表示的最小正数: 约 6×10⁻⁵                                 │
    │  深度学习中常见的梯度值: 10⁻⁶ ~ 10⁻⁸                             │
    │                                                                 │
    │  问题: 梯度下溢 (Underflow)                                      │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  grad = 1×10⁻⁷                                          │   │
    │  │  FP16表示 → 0.0  (信息完全丢失!)                         │   │
    │  │  模型无法学习!                                           │   │
    │  └─────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
工作原理 (图解)
================================================================================

    损失缩放的数学原理:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  原始: L → backward → g = ∂L/∂w                                 │
    │                                                                 │
    │  缩放: L' = s × L → backward → g' = s × g                       │
    │                                                                 │
    │  还原: g = g' / s                                               │
    │                                                                 │
    │  效果:                                                          │
    │  - 原始梯度 g = 1×10⁻⁷ (FP16下溢为0)                            │
    │  - 缩放后 g' = 65536 × 1×10⁻⁷ = 6.5×10⁻³ (可以表示!)           │
    │  - 还原后 g = 6.5×10⁻³ / 65536 = 1×10⁻⁷ (正确值)               │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    动态缩放算法:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  初始: scale = 65536 (2¹⁶)                                      │
    │                                                                 │
    │  每步:                                                          │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  1. 缩放损失: scaled_loss = loss × scale                 │   │
    │  │  2. 反向传播: scaled_loss.backward()                     │   │
    │  │  3. 还原梯度: grad = grad / scale                        │   │
    │  │  4. 检查溢出: 是否有 inf 或 nan?                          │   │
    │  │                                                          │   │
    │  │  if 有溢出:                                               │   │
    │  │      scale = scale × 0.5  # 缩小                         │   │
    │  │      跳过本次更新                                         │   │
    │  │  else:                                                   │   │
    │  │      正常更新权重                                         │   │
    │  │      连续成功次数 += 1                                    │   │
    │  │      if 连续成功次数 >= 2000:                             │   │
    │  │          scale = scale × 2  # 增大                       │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  目标: 自动找到最大的scale，使梯度既不下溢也不溢出               │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
静态缩放 vs 动态缩放
================================================================================
    静态缩放:
    - 固定scale值
    - 简单但可能不适应训练过程
    - 可能溢出或下溢

    动态缩放:
    - 自动调整scale值
    - 适应训练过程中的梯度变化
    - 更鲁棒，推荐使用

================================================================================
前置知识
================================================================================
- FP16的数值范围和精度限制
- 深度学习的反向传播
- 梯度下溢/溢出的概念

================================================================================
参考文献
================================================================================
- Micikevicius et al., "Mixed Precision Training", ICLR 2018
- NVIDIA Apex AMP documentation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class GradScalerConfig:
    """梯度缩放器配置

    Attributes:
        init_scale: 初始缩放因子 (默认65536 = 2¹⁶)
        growth_factor: 缩放因子增长倍数 (默认2.0)
        backoff_factor: 溢出时缩放因子回退倍数 (默认0.5)
        growth_interval: 连续多少步无溢出后增大缩放因子 (默认2000)
        max_scale: 最大缩放因子 (默认2²⁴)
        min_scale: 最小缩放因子 (默认1.0)
        enabled: 是否启用缩放

    参数选择建议:
        - init_scale: 65536是个好的起点，大多数情况下工作良好
        - growth_interval: 2000步是PyTorch默认值，可根据训练稳定性调整
        - 如果频繁溢出，可以减小init_scale或增大growth_interval

    Example:
        >>> config = GradScalerConfig(
        ...     init_scale=65536.0,
        ...     growth_interval=2000,
        ... )
    """
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_scale: float = 2.0 ** 24
    min_scale: float = 1.0
    enabled: bool = True


# =============================================================================
# 智能梯度缩放器
# =============================================================================

class SmartGradScaler:
    """增强版梯度缩放器

    相比PyTorch原生GradScaler，提供更多控制选项和统计信息。

    功能:
    - 动态调整缩放因子
    - 溢出检测和跳过
    - 溢出率统计
    - 检查点保存/加载

    工作流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. scaled_loss = scaler.scale(loss)  # 缩放损失               │
    │  2. scaled_loss.backward()            # 反向传播               │
    │  3. scaler.unscale_(optimizer)        # 还原梯度               │
    │  4. scaler.step(optimizer)            # 更新 (如果无溢出)      │
    │  5. scaler.update()                   # 调整缩放因子           │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        config: 缩放器配置

    Example:
        >>> scaler = SmartGradScaler(GradScalerConfig())
        >>> scaled_loss = scaler.scale(loss)
        >>> scaled_loss.backward()
        >>> scaler.unscale_(optimizer)
        >>> scaler.step(optimizer)
        >>> scaler.update()
    """

    def __init__(self, config: Optional[GradScalerConfig] = None):
        self.config = config or GradScalerConfig()

        # 当前缩放因子
        self._scale = torch.tensor(
            self.config.init_scale,
            dtype=torch.float32,
        )
        # 连续无溢出的步数
        self._growth_tracker = 0
        # 是否检测到溢出
        self._found_inf = torch.tensor(0.0)
        self._enabled = self.config.enabled

        # 统计信息
        self._overflow_count = 0  # 溢出次数
        self._total_steps = 0     # 总步数

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失值

        在反向传播之前调用，将损失乘以缩放因子。

        Args:
            loss: 原始损失张量

        Returns:
            缩放后的损失张量

        Example:
            >>> loss = criterion(output, target)
            >>> scaled_loss = scaler.scale(loss)
            >>> scaled_loss.backward()
        """
        if not self._enabled:
            return loss

        # loss × scale_factor
        return loss * self._scale.to(loss.device)

    def unscale_(self, optimizer: Optimizer) -> None:
        """还原梯度

        将优化器中所有参数的梯度除以缩放因子，并检测溢出。

        Args:
            optimizer: 包含参数梯度的优化器

        Example:
            >>> scaler.unscale_(optimizer)
            >>> # 现在可以进行梯度裁剪
            >>> torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        """
        if not self._enabled:
            return

        # 计算缩放因子的倒数
        inv_scale = 1.0 / self._scale
        self._found_inf.zero_()

        # 遍历所有参数，还原梯度并检测溢出
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    # 梯度 ÷ scale_factor
                    param.grad.mul_(inv_scale.to(param.grad.device))

                    # 检测inf或nan
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        self._found_inf.fill_(1.0)

    def step(self, optimizer: Optimizer) -> bool:
        """执行优化器更新

        如果检测到溢出，跳过本次更新。

        Args:
            optimizer: 优化器

        Returns:
            True如果执行了更新，False如果因溢出跳过

        Example:
            >>> if scaler.step(optimizer):
            ...     print("更新成功")
            ... else:
            ...     print("检测到溢出，跳过更新")
        """
        if not self._enabled:
            optimizer.step()
            return True

        self._total_steps += 1

        # 如果检测到溢出，跳过更新
        if self._found_inf.item() > 0:
            self._overflow_count += 1
            return False

        # 正常更新
        optimizer.step()
        return True

    def update(self) -> None:
        """更新缩放因子

        根据溢出历史动态调整缩放因子:
        - 如果溢出: scale × 0.5
        - 如果连续2000步无溢出: scale × 2
        """
        if not self._enabled:
            return

        if self._found_inf.item() > 0:
            # 溢出: 减小缩放因子
            new_scale = self._scale * self.config.backoff_factor
            self._scale = torch.clamp(
                new_scale,
                min=self.config.min_scale,
                max=self.config.max_scale,
            )
            self._growth_tracker = 0
        else:
            # 无溢出: 增加计数器
            self._growth_tracker += 1
            if self._growth_tracker >= self.config.growth_interval:
                # 连续足够多步无溢出: 增大缩放因子
                new_scale = self._scale * self.config.growth_factor
                self._scale = torch.clamp(
                    new_scale,
                    min=self.config.min_scale,
                    max=self.config.max_scale,
                )
                self._growth_tracker = 0

        # 重置溢出标志
        self._found_inf.zero_()

    def get_scale(self) -> float:
        """获取当前缩放因子

        Returns:
            当前缩放因子值
        """
        return self._scale.item()

    def get_overflow_ratio(self) -> float:
        """获取溢出率

        Returns:
            溢出步数占总步数的比例

        Example:
            >>> ratio = scaler.get_overflow_ratio()
            >>> print(f"溢出率: {ratio:.2%}")
        """
        if self._total_steps == 0:
            return 0.0
        return self._overflow_count / self._total_steps

    def state_dict(self) -> Dict:
        """获取缩放器状态 (用于保存检查点)

        Returns:
            状态字典
        """
        return {
            "scale": self._scale.item(),
            "growth_tracker": self._growth_tracker,
            "overflow_count": self._overflow_count,
            "total_steps": self._total_steps,
        }

    def load_state_dict(self, state: Dict) -> None:
        """加载缩放器状态 (用于恢复检查点)

        Args:
            state: 状态字典
        """
        self._scale = torch.tensor(state["scale"], dtype=torch.float32)
        self._growth_tracker = state["growth_tracker"]
        self._overflow_count = state.get("overflow_count", 0)
        self._total_steps = state.get("total_steps", 0)


# =============================================================================
# 动态损失缩放器 (激进版)
# =============================================================================

class DynamicLossScaler:
    """激进的动态损失缩放器

    使用滑动窗口计算溢出率，比标准动态缩放更激进地调整缩放因子。
    适用于训练不稳定的场景。

    与SmartGradScaler的区别:
    - 使用滑动窗口统计溢出率
    - 根据溢出率而非连续步数调整
    - 更快响应训练动态变化

    Args:
        init_scale: 初始缩放因子
        scale_window: 滑动窗口大小
        min_scale: 最小缩放因子
        max_scale: 最大缩放因子

    Example:
        >>> scaler = DynamicLossScaler(init_scale=65536, scale_window=1000)
        >>> # 训练循环中
        >>> overflow = scaler.has_overflow(model.parameters())
        >>> scaler.update_scale(overflow)
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

        # 溢出历史 (滑动窗口)
        self._overflow_history: List[bool] = []
        self._step_count = 0

    def loss_scale(self) -> float:
        """获取当前损失缩放因子

        Returns:
            当前缩放因子
        """
        return self.scale

    def update_scale(self, overflow: bool) -> None:
        """根据溢出状态更新缩放因子

        调整策略:
        - 溢出时立即减半
        - 每scale_window步检查溢出率:
          - 溢出率 < 1%: 缩放因子翻倍
          - 溢出率 > 10%: 缩放因子减半

        Args:
            overflow: 当前步是否溢出
        """
        # 记录溢出历史
        self._overflow_history.append(overflow)
        self._step_count += 1

        # 保持滑动窗口大小
        if len(self._overflow_history) > self.scale_window:
            self._overflow_history.pop(0)

        if overflow:
            # 溢出时立即减半
            self.scale = max(self.scale / 2, self.min_scale)
        elif self._step_count % self.scale_window == 0:
            # 每scale_window步检查溢出率
            overflow_rate = sum(self._overflow_history) / len(self._overflow_history)

            if overflow_rate < 0.01:
                # 溢出率很低，可以增大缩放因子
                self.scale = min(self.scale * 2, self.max_scale)
            elif overflow_rate > 0.1:
                # 溢出率太高，需要减小缩放因子
                self.scale = max(self.scale / 2, self.min_scale)

    def has_overflow(self, params: List[nn.Parameter]) -> bool:
        """检查参数梯度是否有溢出

        Args:
            params: 要检查的参数列表

        Returns:
            True如果任何梯度包含inf或nan

        Example:
            >>> overflow = scaler.has_overflow(list(model.parameters()))
            >>> if overflow:
            ...     print("检测到梯度溢出")
        """
        for param in params:
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    return True
        return False

    def state_dict(self) -> Dict:
        """获取缩放器状态 (用于保存检查点)

        Returns:
            状态字典
        """
        return {
            "scale": self.scale,
            "step_count": self._step_count,
            # 只保存最近100条历史
            "overflow_history": self._overflow_history[-100:],
        }

    def load_state_dict(self, state: Dict) -> None:
        """加载缩放器状态 (用于恢复检查点)

        Args:
            state: 状态字典
        """
        self.scale = state["scale"]
        self._step_count = state["step_count"]
        self._overflow_history = state.get("overflow_history", [])
