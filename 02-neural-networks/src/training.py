"""训练工具和回调机制。/ Training utilities and callback mechanisms.

提供标准化训练循环、评估函数和 PyTorch 版的回调系统，
等价于 Keras 的 ModelCheckpoint、EarlyStopping 等。

Provides standardized training loop, evaluation functions, and a PyTorch
callback system equivalent to Keras's ModelCheckpoint, EarlyStopping, etc.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """训练一个 epoch。/ Train for one epoch.

    执行单轮前向传播、损失计算、反向传播和参数更新。

    Performs one round of forward pass, loss computation,
    backward propagation, and parameter update.

    Args:
        model: 待训练的 PyTorch 模型 / PyTorch model to train
        dataloader: 训练数据加载器 / training data loader
        criterion: 损失函数 / loss function
        optimizer: 优化器 / optimizer
        device: 计算设备 / compute device ('cpu', 'cuda', 'mps')

    Returns:
        float: 平均训练损失 / average training loss
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

    return total_loss / total_samples


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> dict[str, float]:
    """评估模型性能。/ Evaluate model performance.

    在验证/测试集上计算损失和指标。

    Computes loss and metrics on validation/test set.

    Args:
        model: 待评估的模型 / model to evaluate
        dataloader: 评估数据加载器 / evaluation data loader
        criterion: 损失函数 / loss function
        device: 计算设备 / compute device

    Returns:
        dict: 包含 'loss' 和可选指标的字典 / dict with 'loss' and optional metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

            # 分类任务：计算准确率 / Classification: compute accuracy
            if outputs.dim() > 1 and outputs.size(1) > 1:
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()

    result = {"loss": total_loss / total_samples}
    if total_samples > 0 and correct > 0:
        result["accuracy"] = correct / total_samples

    return result


class Callback:
    """回调基类。/ Base callback class.

    提供与 Keras 回调类似的接口，用于在训练过程中执行自定义操作。

    Provides a Keras-like callback interface for executing
    custom operations during training.

    子类可以重写以下方法 / Subclasses can override:
        on_train_begin(logs): 训练开始时调用 / called when training begins
        on_epoch_end(epoch, logs): 每个 epoch 结束时调用 / called at end of each epoch
        on_train_end(logs): 训练结束时调用 / called when training ends
    """

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """训练开始时调用。/ Called when training begins."""
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """每个 epoch 结束时调用。/ Called at the end of each epoch."""
        pass

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """训练结束时调用。/ Called when training ends."""
        pass


class EarlyStopping(Callback):
    """早停回调。/ Early stopping callback.

    当监控指标连续 patience 个 epoch 不改善时停止训练。

    Stops training when the monitored metric hasn't improved
    for patience consecutive epochs.

    Args:
        monitor: 监控指标名称 / monitored metric name ('val_loss', 'val_accuracy')
        patience: 等待改善的 epoch 数 / number of epochs to wait for improvement
        min_delta: 最小改善阈值 / minimum improvement threshold
        mode: 'min' 指标越小越好, 'max' 指标越大越好 / 'min' for lower-is-better, 'max' for higher-is-better
        restore_best_weights: 是否恢复最佳权重 / whether to restore best weights
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.best_epoch: int = 0
        self.counter: int = 0
        self.stopped_epoch: int = 0
        self.best_weights: dict[str, Any] | None = None
        self.should_stop: bool = False

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """重置状态。/ Reset state."""
        self.counter = 0
        self.should_stop = False
        self.best_score = float("inf") if self.mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """检查是否应该停止训练。/ Check whether training should stop."""
        if logs is None or self.monitor not in logs:
            return

        current_score = logs[self.monitor]

        # 判断是否改善 / Determine if improvement occurred
        if self.mode == "min":
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            # 保存最佳权重 / Save best weights
            if self.restore_best_weights:
                model = logs.get("model")
                if model is not None:
                    self.best_weights = copy.deepcopy(
                        {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    )
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                self.stopped_epoch = epoch
                logger.info(
                    f"早停触发 / EarlyStopping triggered at epoch {epoch + 1}. "
                    f"Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch + 1}"
                )

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """训练结束时恢复最佳权重。/ Restore best weights when training ends."""
        if self.restore_best_weights and self.best_weights and logs is not None and "model" in logs:
            logs["model"].load_state_dict(self.best_weights)
            logger.info(f"恢复最佳权重 / Restored best weights from epoch {self.best_epoch + 1}")


class ModelCheckpoint(Callback):
    """模型检查点回调。/ Model checkpoint callback.

    在训练过程中保存最佳模型权重。

    Saves best model weights during training.

    Args:
        filepath: 保存路径 / save path
        monitor: 监控指标 / monitored metric
        save_best_only: 只保存最佳模型 / save only the best model
        mode: 'min' 或 'max' / 'min' or 'max'
    """

    def __init__(
        self,
        filepath: str = "best_model.pth",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
    ) -> None:
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode

        self.best_score: float = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """根据监控指标保存模型。/ Save model based on monitored metric."""
        if logs is None or self.monitor not in logs:
            return

        current_score = logs[self.monitor]

        if self.mode == "min":
            improved = current_score < self.best_score
        else:
            improved = current_score > self.best_score

        if (improved or not self.save_best_only) and "model" in logs and logs["model"] is not None:
            os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
            torch.save(logs["model"].state_dict(), self.filepath)
            if improved:
                self.best_score = current_score
                logger.info(f"保存最佳模型 / Saved best model: {self.monitor}={current_score:.4f}")
