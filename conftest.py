"""项目级 pytest 共享配置和 fixture。/ Project-level pytest shared configuration and fixtures."""

from __future__ import annotations

import random

import numpy as np
import pytest


@pytest.fixture
def device() -> str:
    """检测并返回可用的计算设备。/ Detect and return available compute device.

    Returns:
        str: 'cuda', 'mps', 或 'cpu'
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


@pytest.fixture
def seed() -> int:
    """设置所有框架的随机种子以确保可复现性。/ Set random seeds for all frameworks for reproducibility.

    Returns:
        int: 使用的种子值
    """
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import torch

        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
    except ImportError:
        pass
    return seed_value


@pytest.fixture
def small_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """生成小型回归数据集用于快速测试。/ Generate small regression dataset for fast testing.

    Returns:
        tuple: (X, y) 其中 X 形状为 (100, 8), y 形状为 (100,)
    """
    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing()
    # 取前 100 个样本
    return housing.data[:100].astype(np.float32), housing.target[:100].astype(np.float32)


@pytest.fixture
def small_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """生成小型分类数据集用于快速测试。/ Generate small classification dataset for fast testing.

    Returns:
        tuple: (X, y) 其中 X 形状为 (200, 28, 28), y 形状为 (200,)
    """
    try:
        import torchvision

        dataset = torchvision.datasets.FashionMNIST(
            root="/tmp/fashion_mnist",
            train=True,
            download=True,
        )
        # 取前 200 个样本
        X = dataset.data[:200].float() / 255.0
        y = dataset.targets[:200]
        return X.numpy(), y.numpy()
    except ImportError:
        # 回退到 sklearn 的 digits 数据集
        from sklearn.datasets import load_digits

        digits = load_digits()
        return digits.data[:200].astype(np.float32), digits.target[:200]
