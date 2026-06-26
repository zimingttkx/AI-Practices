"""神经网络基础模块 — PyTorch 实现。/ Neural network fundamentals — PyTorch implementation.

提供常用的模型定义、训练工具和回调机制，用于 02-neural-networks 教程。

Provides common model definitions, training utilities, and callback mechanisms
for the 02-neural-networks tutorial.
"""

from __future__ import annotations

__all__ = [
    "SimpleMLP",
    "WideAndDeepModel",
    "EarlyStopping",
    "ModelCheckpoint",
    "train_epoch",
    "evaluate_epoch",
]
