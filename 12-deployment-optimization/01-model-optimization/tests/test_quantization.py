"""
量化模块单元测试
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantization import (
    QuantizationConfig,
    QuantizationType,
    QuantizationGranularity,
    FakeQuantize,
    FakeQuantizeModule,
    QuantizedLinear,
    QuantizedConv2d,
    DynamicQuantizer,
    StaticQuantizer,
    QATWrapper,
    quantize_model,
    calibrate_model,
    compute_scale_zero_point,
    quantize_tensor,
    dequantize_tensor,
)


# ==================== 测试模型 ====================

class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvModel(nn.Module):
    """卷积测试模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==================== 基础函数测试 ====================

class TestScaleZeroPoint:
    """测试 scale 和 zero_point 计算"""

    def test_symmetric_quantization(self):
        """测试对称量化"""
        x_min = torch.tensor(-1.0)
        x_max = torch.tensor(1.0)

        scale, zero_point = compute_scale_zero_point(
            x_min, x_max, -128, 127, symmetric=True
        )

        assert scale > 0
        assert zero_point == 0

    def test_asymmetric_quantization(self):
        """测试非对称量化"""
        x_min = torch.tensor(0.0)
        x_max = torch.tensor(1.0)

        scale, zero_point = compute_scale_zero_point(
            x_min, x_max, 0, 255, symmetric=False
        )

        assert scale > 0
        assert zero_point >= 0

    def test_scale_not_zero(self):
        """测试 scale 不为零"""
        x_min = torch.tensor(0.0)
        x_max = torch.tensor(0.0)

        scale, zero_point = compute_scale_zero_point(
            x_min, x_max, -128, 127, symmetric=True
        )

        assert scale > 0  # 应该被 clamp 到最小值


class TestQuantizeDequantize:
    """测试量化和反量化"""

    def test_quantize_tensor(self):
        """测试张量量化"""
        x = torch.randn(10, 10)
        scale = torch.tensor(0.1)
        zero_point = torch.tensor(0.0)

        q = quantize_tensor(x, scale, zero_point, -128, 127)

        assert q.dtype == torch.int8
        assert q.min() >= -128
        assert q.max() <= 127

    def test_dequantize_tensor(self):
        """测试张量反量化"""
        q = torch.randint(-128, 127, (10, 10), dtype=torch.int8)
        scale = torch.tensor(0.1)
        zero_point = torch.tensor(0.0)

        x = dequantize_tensor(q, scale, zero_point)

        assert x.dtype == torch.float32

    def test_quantize_dequantize_roundtrip(self):
        """测试量化-反量化往返"""
        x = torch.randn(10, 10)
        x_min, x_max = x.min(), x.max()

        scale, zero_point = compute_scale_zero_point(
            x_min, x_max, -128, 127, symmetric=True
        )

        q = quantize_tensor(x, scale, zero_point, -128, 127)
        x_reconstructed = dequantize_tensor(q, scale, zero_point)

        # 量化误差应该在合理范围内
        error = (x - x_reconstructed).abs().max()
        assert error < scale * 2  # 最大误差约为 1 个量化步长


# ==================== FakeQuantize 测试 ====================

class TestFakeQuantize:
    """测试伪量化"""

    def test_fake_quantize_forward(self):
        """测试伪量化前向传播"""
        x = torch.randn(10, 10, requires_grad=True)
        scale = torch.tensor(0.1)
        zero_point = torch.tensor(0.0)

        x_q = FakeQuantize.apply(x, scale, zero_point, -128, 127)

        assert x_q.shape == x.shape
        assert x_q.requires_grad

    def test_fake_quantize_backward(self):
        """测试伪量化反向传播 (STE)"""
        x = torch.randn(10, 10, requires_grad=True)
        scale = torch.tensor(0.1)
        zero_point = torch.tensor(0.0)

        x_q = FakeQuantize.apply(x, scale, zero_point, -128, 127)
        loss = x_q.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestFakeQuantizeModule:
    """测试伪量化模块"""

    def test_module_forward(self):
        """测试模块前向传播"""
        module = FakeQuantizeModule(qmin=-128, qmax=127)
        x = torch.randn(10, 10)

        # 训练模式
        module.train()
        y = module(x)
        assert y.shape == x.shape

        # 评估模式
        module.eval()
        y = module(x)
        assert y.shape == x.shape

    def test_stats_update(self):
        """测试统计信息更新"""
        module = FakeQuantizeModule()
        module.train()

        x = torch.randn(10, 10)
        _ = module(x)

        assert module.min_val < float('inf')
        assert module.max_val > float('-inf')


# ==================== 量化层测试 ====================

class TestQuantizedLinear:
    """测试量化线性层"""

    def test_forward(self):
        """测试前向传播"""
        layer = QuantizedLinear(64, 32)
        x = torch.randn(8, 64)

        y = layer(x)

        assert y.shape == (8, 32)

    def test_quantize_weights(self):
        """测试权重量化"""
        layer = QuantizedLinear(64, 32)
        layer.quantize_weights()

        assert layer.quantized_weight is not None
        assert layer.quantized_weight.dtype == torch.int8

    def test_qat_mode(self):
        """测试 QAT 模式"""
        config = QuantizationConfig(quant_type=QuantizationType.QAT)
        layer = QuantizedLinear(64, 32, config=config)

        x = torch.randn(8, 64)
        layer.train()
        y = layer(x)

        assert y.shape == (8, 32)


class TestQuantizedConv2d:
    """测试量化卷积层"""

    def test_forward(self):
        """测试前向传播"""
        layer = QuantizedConv2d(3, 16, 3, padding=1)
        x = torch.randn(4, 3, 32, 32)

        y = layer(x)

        assert y.shape == (4, 16, 32, 32)

    def test_quantize_weights(self):
        """测试权重量化"""
        layer = QuantizedConv2d(3, 16, 3)
        layer.quantize_weights()

        assert layer.quantized_weight is not None
        assert layer.quantized_weight.dtype == torch.int8


# ==================== 量化器测试 ====================

class TestDynamicQuantizer:
    """测试动态量化器"""

    def test_quantize_model(self):
        """测试模型量化"""
        model = SimpleModel()
        quantizer = DynamicQuantizer()

        quantized_model = quantizer.quantize(model)

        # 检查层是否被替换
        has_quantized_layer = False
        for module in quantized_model.modules():
            if isinstance(module, QuantizedLinear):
                has_quantized_layer = True
                break

        assert has_quantized_layer

    def test_quantized_model_forward(self):
        """测试量化模型前向传播"""
        model = SimpleModel()
        quantizer = DynamicQuantizer()

        quantized_model = quantizer.quantize(model)
        x = torch.randn(8, 64)

        y = quantized_model(x)

        assert y.shape == (8, 10)


class TestStaticQuantizer:
    """测试静态量化器"""

    def test_calibrate(self):
        """测试校准"""
        model = SimpleModel()
        quantizer = StaticQuantizer()

        # 创建校准数据
        calibration_data = [torch.randn(8, 64) for _ in range(10)]

        quantizer.calibrate(model, calibration_data, num_batches=5)

        assert len(quantizer.activation_stats) > 0

    def test_quantize_after_calibrate(self):
        """测试校准后量化"""
        model = SimpleModel()
        quantizer = StaticQuantizer()

        calibration_data = [torch.randn(8, 64) for _ in range(10)]
        quantizer.calibrate(model, calibration_data, num_batches=5)

        quantized_model = quantizer.quantize(model)

        x = torch.randn(8, 64)
        y = quantized_model(x)

        assert y.shape == (8, 10)


class TestQATWrapper:
    """测试 QAT 包装器"""

    def test_wrapper_forward(self):
        """测试包装器前向传播"""
        model = SimpleModel()
        qat_model = QATWrapper(model)

        x = torch.randn(8, 64)
        y = qat_model(x)

        assert y.shape == (8, 10)

    def test_convert_to_quantized(self):
        """测试转换为量化模型"""
        model = SimpleModel()
        qat_model = QATWrapper(model)

        # 模拟训练
        x = torch.randn(8, 64)
        _ = qat_model(x)

        quantized_model = qat_model.convert_to_quantized()

        y = quantized_model(x)
        assert y.shape == (8, 10)


# ==================== 便捷函数测试 ====================

class TestQuantizeModel:
    """测试 quantize_model 便捷函数"""

    def test_dynamic_quantization(self):
        """测试动态量化"""
        model = SimpleModel()
        quantized = quantize_model(model, quant_type="dynamic")

        x = torch.randn(8, 64)
        y = quantized(x)

        assert y.shape == (8, 10)

    def test_qat_quantization(self):
        """测试 QAT 量化"""
        model = SimpleModel()
        qat_model = quantize_model(model, quant_type="qat")

        assert isinstance(qat_model, QATWrapper)


class TestCalibrateModel:
    """测试 calibrate_model 便捷函数"""

    def test_calibrate(self):
        """测试校准"""
        model = SimpleModel()
        calibration_data = [torch.randn(8, 64) for _ in range(10)]

        stats = calibrate_model(model, calibration_data, num_batches=5)

        assert isinstance(stats, dict)


# ==================== 配置测试 ====================

class TestQuantizationConfig:
    """测试量化配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = QuantizationConfig()

        assert config.quant_type == QuantizationType.DYNAMIC
        assert config.weight_bits == 8
        assert config.activation_bits == 8

    def test_custom_config(self):
        """测试自定义配置"""
        config = QuantizationConfig(
            quant_type=QuantizationType.QAT,
            weight_bits=4,
            symmetric=False
        )

        assert config.quant_type == QuantizationType.QAT
        assert config.weight_bits == 4
        assert config.symmetric is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
