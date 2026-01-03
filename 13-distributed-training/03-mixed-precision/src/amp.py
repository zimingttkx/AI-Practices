"""
Automatic Mixed Precision (AMP) Training Implementation

Core Idea:
    AMP accelerates training by using lower precision (FP16/BF16) for forward
    pass computations while maintaining FP32 for gradient accumulation and
    weight updates, reducing memory and increasing throughput.

Mathematical Theory:
    FP16 has limited dynamic range [6e-5, 65504]. Loss scaling prevents
    gradient underflow by multiplying loss before backward:
    
    .. math::
        \\tilde{L} = s \\cdot L, \\quad \\tilde{g} = s \\cdot g, \\quad g = \\tilde{g} / s
    
    where s is the scale factor, dynamically adjusted based on overflow.

Problem Statement:
    FP16 training suffers from gradient underflow (small gradients become zero)
    and overflow (large values exceed FP16 range). AMP with loss scaling
    addresses these issues while maintaining training stability.

Comparison:
    - FP32: Full precision, baseline memory and speed
    - FP16 + AMP: ~2x memory reduction, ~2x speedup on Tensor Cores
    - BF16: Same range as FP32, no scaling needed, requires Ampere+

Complexity:
    - Memory: ~50% reduction for activations and gradients
    - Compute: ~2x speedup on Tensor Core operations
    - Overhead: Minimal from autocast and scaling operations

References:
    - Micikevicius et al., "Mixed Precision Training", ICLR 2018
    - NVIDIA Apex documentation
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


@dataclass
class AMPConfig:
    """Configuration for automatic mixed precision training.
    
    Attributes:
        enabled: Enable AMP training.
        dtype: Mixed precision dtype (float16 or bfloat16).
        cache_enabled: Enable autocast kernel caching.
        use_grad_scaler: Enable gradient scaling for FP16.
        init_scale: Initial loss scale factor.
        growth_factor: Scale increase multiplier.
        backoff_factor: Scale decrease multiplier on overflow.
        growth_interval: Steps between scale increases.
        max_scale: Maximum allowed scale factor.
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
    """Convert string to torch dtype for autocast.
    
    Args:
        dtype_str: Dtype string ("float16", "fp16", "bfloat16", "bf16").
        
    Returns:
        Corresponding torch.dtype.
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
    """Context manager for automatic mixed precision.
    
    Args:
        device_type: Device type ("cuda" or "cpu").
        dtype: Target dtype for autocast.
        enabled: Whether autocast is enabled.
        cache_enabled: Whether to cache autocast kernels.
    """
    with autocast(
        device_type=device_type,
        dtype=dtype,
        enabled=enabled,
        cache_enabled=cache_enabled,
    ):
        yield


class AMPTrainer:
    """Trainer with automatic mixed precision support.
    
    Encapsulates autocast context and gradient scaling for simplified
    mixed precision training workflow.
    
    Args:
        model: PyTorch model to train.
        config: AMP configuration.
        device: Target device for training.
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
        """Get autocast context manager."""
        return autocast(
            device_type=self.device_type,
            dtype=self.config.dtype,
            enabled=self.config.enabled,
            cache_enabled=self.config.cache_enabled,
        )
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with autocast enabled."""
        with self.autocast():
            return self.model(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient scaling."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optimizer step with gradient unscaling and scale update."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients before gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def clip_gradients(
        self,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """Clip gradients by global norm."""
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
        )
    
    def get_scale(self) -> float:
        """Get current loss scale factor."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing."""
        state = {"model": self.model.state_dict()}
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load trainer state from checkpoint."""
        self.model.load_state_dict(state["model"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
