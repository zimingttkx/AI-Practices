"""02-neural-networks 测试共享 fixture。/ Shared test fixtures for 02-neural-networks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 将 src 目录加入 Python 路径 / Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def device() -> str:
    """检测并返回可用的计算设备。/ Detect and return available compute device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def seed() -> int:
    """设置随机种子。/ Set random seeds."""
    import numpy as np
    import torch

    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    return seed_value
