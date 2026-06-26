"""训练工具和回调机制测试。/ Tests for training utilities and callbacks."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from models import SimpleMLP
from torch.utils.data import DataLoader, TensorDataset
from training import EarlyStopping, ModelCheckpoint, evaluate_epoch, train_epoch


@pytest.fixture
def simple_data(device: str) -> tuple[DataLoader, DataLoader]:
    """创建简单的训练和验证数据。/ Create simple train and val data."""
    torch.manual_seed(42)
    X_train = torch.randn(100, 8)
    y_train = torch.randn(100, 1)
    X_val = torch.randn(20, 8)
    y_val = torch.randn(20, 1)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)

    return train_loader, val_loader


@pytest.fixture
def simple_model(device: str) -> SimpleMLP:
    """创建简单模型。/ Create a simple model."""
    return SimpleMLP(input_size=8, hidden_sizes=[30], output_size=1).to(device)


class TestTrainEpoch:
    """train_epoch 函数测试。/ Tests for train_epoch function."""

    def test_returns_float(self, simple_model: SimpleMLP, simple_data: tuple, device: str) -> None:
        """测试返回值为浮点数。/ Test return value is a float."""
        train_loader, _ = simple_data
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        loss = train_epoch(simple_model, train_loader, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self, simple_model: SimpleMLP, simple_data: tuple, device: str) -> None:
        """测试损失在多轮训练后下降。/ Test loss decreases over multiple epochs."""
        train_loader, _ = simple_data
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)

        losses = []
        for _ in range(5):
            loss = train_epoch(simple_model, train_loader, criterion, optimizer, device)
            losses.append(loss)

        # 最后一个损失应该比第一个小 / Last loss should be less than first
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


class TestEvaluateEpoch:
    """evaluate_epoch 函数测试。/ Tests for evaluate_epoch function."""

    def test_returns_dict(self, simple_model: SimpleMLP, simple_data: tuple, device: str) -> None:
        """测试返回值为字典。/ Test return value is a dict."""
        _, val_loader = simple_data
        criterion = nn.MSELoss()

        result = evaluate_epoch(simple_model, val_loader, criterion, device)
        assert isinstance(result, dict)
        assert "loss" in result

    def test_eval_no_gradient(
        self, simple_model: SimpleMLP, simple_data: tuple, device: str
    ) -> None:
        """测试评估时无梯度计算。/ Test no gradient computation during evaluation."""
        _, val_loader = simple_data
        criterion = nn.MSELoss()

        # 确保在评估模式下没有梯度 / Ensure no gradients in eval mode
        simple_model.eval()
        with torch.no_grad():
            evaluate_epoch(simple_model, val_loader, criterion, device)

        # 验证参数没有梯度 / Verify parameters have no gradients
        for param in simple_model.parameters():
            assert param.grad is None or not param.grad.requires_grad


class TestEarlyStopping:
    """EarlyStopping 回调测试。/ Tests for EarlyStopping callback."""

    def test_initial_state(self) -> None:
        """测试初始状态。/ Test initial state."""
        es = EarlyStopping(patience=5, monitor="val_loss")
        assert es.counter == 0
        assert not es.should_stop

    def test_triggers_on_patience(self) -> None:
        """测试耐心值耗尽时触发。/ Test triggers when patience is exhausted."""
        es = EarlyStopping(patience=3, monitor="val_loss", mode="min")

        es.on_train_begin()
        # 模拟改善 / Simulate improvement
        es.on_epoch_end(0, {"val_loss": 1.0, "model": SimpleMLP(8, [10], 1)})
        assert not es.should_stop

        # 模拟不改善 / Simulate no improvement
        es.on_epoch_end(1, {"val_loss": 1.1, "model": SimpleMLP(8, [10], 1)})
        es.on_epoch_end(2, {"val_loss": 1.2, "model": SimpleMLP(8, [10], 1)})
        es.on_epoch_end(3, {"val_loss": 1.3, "model": SimpleMLP(8, [10], 1)})

        assert es.should_stop
        assert es.stopped_epoch == 3

    def test_resets_on_improvement(self) -> None:
        """测试改善时重置计数器。/ Test counter resets on improvement."""
        es = EarlyStopping(patience=3, monitor="val_loss", mode="min")
        dummy_model = SimpleMLP(8, [10], 1)

        es.on_train_begin()
        es.on_epoch_end(0, {"val_loss": 1.0, "model": dummy_model})
        es.on_epoch_end(1, {"val_loss": 1.1, "model": dummy_model})  # no improvement
        assert es.counter == 1

        es.on_epoch_end(2, {"val_loss": 0.9, "model": dummy_model})  # improvement
        assert es.counter == 0


class TestModelCheckpoint:
    """ModelCheckpoint 回调测试。/ Tests for ModelCheckpoint callback."""

    def test_saves_best_model(self, simple_model: SimpleMLP) -> None:
        """测试保存最佳模型。/ Test saving best model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "best_model.pth")
            mc = ModelCheckpoint(filepath=filepath, monitor="val_loss", mode="min")

            # 第一次保存 / First save
            mc.on_epoch_end(0, {"val_loss": 1.0, "model": simple_model})
            assert os.path.exists(filepath)

            # 更好的结果 / Better result
            mc.on_epoch_end(1, {"val_loss": 0.5, "model": simple_model})
            assert os.path.exists(filepath)

    def test_only_saves_on_improvement(self, simple_model: SimpleMLP) -> None:
        """测试只在改善时保存。/ Test only saves on improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "best_model.pth")
            mc = ModelCheckpoint(
                filepath=filepath, monitor="val_loss", mode="min", save_best_only=True
            )

            mc.on_epoch_end(0, {"val_loss": 1.0, "model": simple_model})
            initial_mtime = os.path.getmtime(filepath)

            # 不改善 / No improvement - file should not be updated
            import time

            time.sleep(0.1)
            mc.on_epoch_end(1, {"val_loss": 1.5, "model": simple_model})

            # 改善 / Improvement - file should be updated
            time.sleep(0.1)
            mc.on_epoch_end(2, {"val_loss": 0.5, "model": simple_model})
            assert os.path.getmtime(filepath) > initial_mtime
