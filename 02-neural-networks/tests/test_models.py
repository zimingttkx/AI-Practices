"""SimpleMLP 和 WideAndDeepModel 模型测试。/ Tests for SimpleMLP and WideAndDeepModel."""

from __future__ import annotations

import torch
import torch.nn as nn
from models import SimpleMLP, WideAndDeepModel


class TestSimpleMLP:
    """SimpleMLP 模型测试。/ Tests for SimpleMLP model."""

    def test_default_construction(self) -> None:
        """测试默认构造。/ Test default construction."""
        model = SimpleMLP(input_size=8, output_size=1)
        assert isinstance(model, nn.Module)

    def test_forward_shape_regression(self, seed: int, device: str) -> None:
        """测试回归任务前向传播形状。/ Test forward pass shape for regression."""
        torch.manual_seed(seed)
        model = SimpleMLP(input_size=8, hidden_sizes=[30, 30], output_size=1).to(device)
        x = torch.randn(16, 8, device=device)
        output = model(x)
        assert output.shape == (16, 1)

    def test_forward_shape_classification(self, seed: int, device: str) -> None:
        """测试分类任务前向传播形状。/ Test forward pass shape for classification."""
        torch.manual_seed(seed)
        model = SimpleMLP(input_size=784, hidden_sizes=[300, 100], output_size=10).to(device)
        x = torch.randn(32, 784, device=device)
        output = model(x)
        assert output.shape == (32, 10)

    def test_dropout(self, seed: int, device: str) -> None:
        """测试 Dropout 是否生效。/ Test that Dropout is active."""
        torch.manual_seed(seed)
        model = SimpleMLP(input_size=8, hidden_sizes=[100], dropout_rate=0.5).to(device)
        x = torch.randn(64, 8, device=device)
        # 训练模式：输出应该因 Dropout 而不同 / Training mode: outputs should differ due to Dropout
        model.train()
        out1 = model(x)
        out2 = model(x)
        assert not torch.allclose(
            out1, out2
        ), "Dropout should cause different outputs in training mode"

    def test_batchnorm(self, seed: int, device: str) -> None:
        """测试批量归一化层是否包含。/ Test that BatchNorm layers are included."""
        model = SimpleMLP(input_size=8, hidden_sizes=[30, 30], use_batchnorm=True)
        bn_count = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm1d))
        assert bn_count == 2, f"Expected 2 BatchNorm1d layers, got {bn_count}"

    def test_custom_activation(self) -> None:
        """测试不同激活函数。/ Test different activation functions."""
        for act in ["relu", "elu", "selu", "leaky_relu"]:
            model = SimpleMLP(input_size=8, hidden_sizes=[30], activation=act)
            x = torch.randn(4, 8)
            output = model(x)
            assert output.shape == (4, 1), f"Failed for activation={act}"

    def test_gradient_flow(self, seed: int, device: str) -> None:
        """测试梯度是否正常流动。/ Test that gradients flow properly."""
        torch.manual_seed(seed)
        model = SimpleMLP(input_size=8, hidden_sizes=[30], output_size=1).to(device)
        x = torch.randn(16, 8, device=device)
        y = torch.randn(16, 1, device=device)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestWideAndDeepModel:
    """WideAndDeepModel 模型测试。/ Tests for WideAndDeepModel."""

    def test_single_output(self, seed: int, device: str) -> None:
        """测试单输出前向传播。/ Test single output forward pass."""
        torch.manual_seed(seed)
        model = WideAndDeepModel(wide_size=5, deep_size=6, hidden_sizes=[30, 30], output_size=1).to(
            device
        )

        x_wide = torch.randn(16, 5, device=device)
        x_deep = torch.randn(16, 6, device=device)
        output = model(x_wide, x_deep)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (16, 1)

    def test_aux_output(self, seed: int, device: str) -> None:
        """测试辅助输出。/ Test auxiliary output."""
        torch.manual_seed(seed)
        model = WideAndDeepModel(
            wide_size=5, deep_size=6, hidden_sizes=[30, 30], output_size=1, aux_output=True
        ).to(device)

        x_wide = torch.randn(16, 5, device=device)
        x_deep = torch.randn(16, 6, device=device)
        main_out, aux_out = model(x_wide, x_deep)

        assert main_out.shape == (16, 1)
        assert aux_out.shape == (16, 1)

    def test_gradient_flow(self, seed: int, device: str) -> None:
        """测试梯度流动。/ Test gradient flow."""
        torch.manual_seed(seed)
        model = WideAndDeepModel(wide_size=5, deep_size=6, hidden_sizes=[30, 30], output_size=1).to(
            device
        )

        x_wide = torch.randn(16, 5, device=device)
        x_deep = torch.randn(16, 6, device=device)
        y = torch.randn(16, 1, device=device)

        output = model(x_wide, x_deep)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
