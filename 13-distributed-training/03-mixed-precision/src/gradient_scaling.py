"""
Gradient Scaling Implementation

Core Idea:
    Gradient scaling prevents underflow in FP16 training by multiplying the
    loss before backward pass, then unscaling gradients before optimizer step.
    Dynamic scaling adjusts the scale factor based on overflow detection.

Mathematical Theory:
    Loss scaling multiplies loss by scale factor s before backward:
    
    .. math::
        \\tilde{L} = s \\cdot L \\implies \\tilde{g} = s \\cdot \\nabla L
    
    Before optimizer step, gradients are unscaled: g = tilde{g} / s
    
    Dynamic scaling algorithm:
    - If overflow detected: s = s * backoff_factor (typically 0.5)
    - If N consecutive steps without overflow: s = s * growth_factor (typically 2.0)

Problem Statement:
    FP16 gradients can underflow to zero for small values (< 6e-5). Loss
    scaling shifts gradients into representable range, while dynamic scaling
    adapts to training dynamics.

Comparison:
    - Static scaling: Fixed scale, may overflow or underflow
    - Dynamic scaling: Adapts to training, more robust
    - No scaling (BF16): Not needed due to larger exponent range

Complexity:
    - Overhead: O(P) for overflow check across P parameters
    - Memory: O(1) for scale factor and counters
    - Compute: Negligible compared to forward/backward

References:
    - Micikevicius et al., "Mixed Precision Training", ICLR 2018
    - NVIDIA Apex AMP documentation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class GradScalerConfig:
    """Configuration for gradient scaler.
    
    Attributes:
        init_scale: Initial loss scale factor.
        growth_factor: Multiplier for scale increase.
        backoff_factor: Multiplier for scale decrease on overflow.
        growth_interval: Steps between scale increases.
        max_scale: Maximum allowed scale factor.
        min_scale: Minimum allowed scale factor.
        enabled: Whether scaling is enabled.
    """
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_scale: float = 2.0 ** 24
    min_scale: float = 1.0
    enabled: bool = True


class SmartGradScaler:
    """Enhanced gradient scaler with overflow tracking and statistics.
    
    Provides more control options compared to PyTorch's native GradScaler.
    
    Args:
        config: Scaler configuration.
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
        
        self._overflow_count = 0
        self._total_steps = 0
    
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss value before backward pass.
        
        Args:
            loss: Original loss tensor.
            
        Returns:
            Scaled loss tensor.
        """
        if not self._enabled:
            return loss
        
        return loss * self._scale.to(loss.device)
    
    def unscale_(self, optimizer: Optimizer) -> None:
        """Unscale gradients in optimizer parameter groups.
        
        Args:
            optimizer: Optimizer containing parameters with gradients.
        """
        if not self._enabled:
            return
        
        inv_scale = 1.0 / self._scale
        self._found_inf.zero_()
        
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.mul_(inv_scale.to(param.grad.device))
                    
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        self._found_inf.fill_(1.0)
    
    def step(self, optimizer: Optimizer) -> bool:
        """Execute optimizer step if no overflow detected.
        
        Args:
            optimizer: Optimizer to step.
            
        Returns:
            True if step was executed, False if skipped due to overflow.
        """
        if not self._enabled:
            optimizer.step()
            return True
        
        self._total_steps += 1
        
        if self._found_inf.item() > 0:
            self._overflow_count += 1
            return False
        
        optimizer.step()
        return True
    
    def update(self) -> None:
        """Update scale factor based on overflow history."""
        if not self._enabled:
            return
        
        if self._found_inf.item() > 0:
            new_scale = self._scale * self.config.backoff_factor
            self._scale = torch.clamp(
                new_scale,
                min=self.config.min_scale,
                max=self.config.max_scale,
            )
            self._growth_tracker = 0
        else:
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
        """Get current scale factor."""
        return self._scale.item()
    
    def get_overflow_ratio(self) -> float:
        """Get ratio of steps with overflow."""
        if self._total_steps == 0:
            return 0.0
        return self._overflow_count / self._total_steps
    
    def state_dict(self) -> Dict:
        """Get scaler state for checkpointing."""
        return {
            "scale": self._scale.item(),
            "growth_tracker": self._growth_tracker,
            "overflow_count": self._overflow_count,
            "total_steps": self._total_steps,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load scaler state from checkpoint."""
        self._scale = torch.tensor(state["scale"], dtype=torch.float32)
        self._growth_tracker = state["growth_tracker"]
        self._overflow_count = state.get("overflow_count", 0)
        self._total_steps = state.get("total_steps", 0)


class DynamicLossScaler:
    """Aggressive dynamic loss scaler for unstable training.
    
    Uses window-based overflow rate to adjust scaling more aggressively
    than standard dynamic scaling.
    
    Args:
        init_scale: Initial loss scale factor.
        scale_window: Window size for overflow rate calculation.
        min_scale: Minimum allowed scale factor.
        max_scale: Maximum allowed scale factor.
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
        """Get current loss scale factor."""
        return self.scale
    
    def update_scale(self, overflow: bool) -> None:
        """Update scale based on overflow status.
        
        Args:
            overflow: Whether overflow occurred in current step.
        """
        self._overflow_history.append(overflow)
        self._step_count += 1
        
        if len(self._overflow_history) > self.scale_window:
            self._overflow_history.pop(0)
        
        if overflow:
            self.scale = max(self.scale / 2, self.min_scale)
        elif self._step_count % self.scale_window == 0:
            overflow_rate = sum(self._overflow_history) / len(self._overflow_history)
            
            if overflow_rate < 0.01:
                self.scale = min(self.scale * 2, self.max_scale)
            elif overflow_rate > 0.1:
                self.scale = max(self.scale / 2, self.min_scale)
    
    def has_overflow(self, params: List[nn.Parameter]) -> bool:
        """Check if any parameter gradients have overflow.
        
        Args:
            params: List of parameters to check.
            
        Returns:
            True if any gradient contains inf or nan.
        """
        for param in params:
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    return True
        return False
    
    def state_dict(self) -> Dict:
        """Get scaler state for checkpointing."""
        return {
            "scale": self.scale,
            "step_count": self._step_count,
            "overflow_history": self._overflow_history[-100:],
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load scaler state from checkpoint."""
        self.scale = state["scale"]
        self._step_count = state["step_count"]
        self._overflow_history = state.get("overflow_history", [])
