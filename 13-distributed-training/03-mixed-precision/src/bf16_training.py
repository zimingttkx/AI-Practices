"""
BF16 (Brain Floating Point 16) Training Implementation

Core Idea:
    BF16 uses 8 exponent bits (same as FP32) with 7 mantissa bits, providing
    the same dynamic range as FP32 without requiring loss scaling. This makes
    training more stable compared to FP16.

Mathematical Theory:
    BF16 format: 1 sign + 8 exponent + 7 mantissa bits
    
    .. math::
        \\text{Range: } \\pm 3.4 \\times 10^{38} \\quad \\text{(same as FP32)}
        
        \\text{Precision: } \\epsilon \\approx 0.0078 \\quad \\text{(vs FP16: } 0.001\\text{)}
    
    The larger exponent range eliminates gradient underflow/overflow issues.

Problem Statement:
    FP16 training requires careful loss scaling due to limited dynamic range.
    BF16 provides FP32-equivalent range, eliminating scaling complexity while
    maintaining memory and compute benefits of 16-bit training.

Comparison:
    - FP16: 5 exp + 10 mantissa, range ~6e-5 to 65504, needs scaling
    - BF16: 8 exp + 7 mantissa, range ~1e-38 to 3e38, no scaling needed
    - FP32: 8 exp + 23 mantissa, full precision baseline

Complexity:
    - Memory: Same as FP16 (~50% reduction vs FP32)
    - Compute: ~2x speedup on Ampere+ Tensor Cores
    - Stability: No loss scaling overhead, simpler training loop

References:
    - Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training", 2019
    - Google TPU documentation on bfloat16
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def is_bf16_supported() -> bool:
    """Check if current device supports BF16.
    
    Returns:
        True if BF16 is supported (requires Ampere or newer GPU).
    """
    if not torch.cuda.is_available():
        return False
    
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return capability[0] >= 8


@dataclass
class BF16Config:
    """Configuration for BF16 training.
    
    Attributes:
        enabled: Enable BF16 training.
        convert_weights: Convert model weights to BF16.
        keep_batchnorm_fp32: Keep BatchNorm layers in FP32.
        keep_layernorm_fp32: Keep LayerNorm layers in FP32.
        master_weights: Maintain FP32 master weights for updates.
        patch_torch_functions: Patch torch functions for BF16.
    """
    enabled: bool = True
    convert_weights: bool = True
    keep_batchnorm_fp32: bool = True
    keep_layernorm_fp32: bool = True
    master_weights: bool = True
    patch_torch_functions: bool = False


def convert_to_bf16(
    model: nn.Module,
    keep_batchnorm_fp32: bool = True,
    keep_layernorm_fp32: bool = True,
) -> nn.Module:
    """Convert model to BF16 while keeping normalization layers in FP32.
    
    Args:
        model: PyTorch model to convert.
        keep_batchnorm_fp32: Keep BatchNorm layers in FP32 for stability.
        keep_layernorm_fp32: Keep LayerNorm layers in FP32 for stability.
        
    Returns:
        Model with BF16 weights (except specified layers).
    """
    fp32_modules = []
    
    for name, module in model.named_modules():
        if keep_batchnorm_fp32 and isinstance(module, (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )):
            fp32_modules.append(name)
        
        if keep_layernorm_fp32 and isinstance(module, nn.LayerNorm):
            fp32_modules.append(name)
    
    model = model.to(torch.bfloat16)
    
    for name, module in model.named_modules():
        if name in fp32_modules:
            module.float()
    
    return model


class BF16Trainer:
    """Trainer for BF16 mixed precision training.
    
    Unlike FP16, BF16 does not require gradient scaling due to its
    FP32-equivalent dynamic range.
    
    Args:
        model: PyTorch model to train.
        config: BF16 configuration.
        device: Target device for training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[BF16Config] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or BF16Config()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        if self.config.enabled and not is_bf16_supported():
            print("Warning: BF16 not supported, falling back to FP32")
            self.config.enabled = False
        
        if self.config.enabled and self.config.convert_weights:
            model = convert_to_bf16(
                model,
                keep_batchnorm_fp32=self.config.keep_batchnorm_fp32,
                keep_layernorm_fp32=self.config.keep_layernorm_fp32,
            )
        
        self.model = model.to(self.device)
        
        self.master_weights: Optional[Dict[str, torch.Tensor]] = None
        if self.config.enabled and self.config.master_weights:
            self._init_master_weights()
    
    def _init_master_weights(self) -> None:
        """Initialize FP32 master weights for stable updates."""
        self.master_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.master_weights[name] = param.data.float().clone()
    
    def autocast(self):
        """Get BF16 autocast context manager."""
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.config.enabled,
        )
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with BF16 autocast."""
        with self.autocast():
            return self.model(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass (no scaling needed for BF16)."""
        loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optimizer step with optional master weight update."""
        if self.master_weights is not None:
            self._step_with_master_weights(optimizer)
        else:
            optimizer.step()
    
    def _step_with_master_weights(self, optimizer: torch.optim.Optimizer) -> None:
        """Update using FP32 master weights for numerical stability."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.master_weights:
                    self.master_weights[name].grad = param.grad.float()
        
        optimizer.step()
        
        for name, param in self.model.named_parameters():
            if name in self.master_weights:
                param.data.copy_(self.master_weights[name].to(torch.bfloat16))
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for model and master weights."""
        self.model.zero_grad(set_to_none=set_to_none)
        if self.master_weights is not None:
            for tensor in self.master_weights.values():
                if tensor.grad is not None:
                    if set_to_none:
                        tensor.grad = None
                    else:
                        tensor.grad.zero_()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing."""
        state = {"model": self.model.state_dict()}
        if self.master_weights is not None:
            state["master_weights"] = self.master_weights
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load trainer state from checkpoint."""
        self.model.load_state_dict(state["model"])
        if "master_weights" in state:
            self.master_weights = state["master_weights"]
