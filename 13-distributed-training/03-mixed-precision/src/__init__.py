"""
混合精度训练模块

包含 AMP、BF16 训练和梯度缩放实现。
"""

from .amp import (
    AMPConfig,
    AMPTrainer,
    autocast_context,
    get_autocast_dtype,
)

from .bf16_training import (
    BF16Config,
    BF16Trainer,
    convert_to_bf16,
    is_bf16_supported,
)

from .gradient_scaling import (
    GradScalerConfig,
    SmartGradScaler,
    DynamicLossScaler,
)

__all__ = [
    # AMP
    "AMPConfig",
    "AMPTrainer",
    "autocast_context",
    "get_autocast_dtype",
    # BF16
    "BF16Config",
    "BF16Trainer",
    "convert_to_bf16",
    "is_bf16_supported",
    # Gradient Scaling
    "GradScalerConfig",
    "SmartGradScaler",
    "DynamicLossScaler",
]
