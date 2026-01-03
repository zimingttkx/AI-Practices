"""
模型导出模块单元测试
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from export import (
    ExportConfig,
    ExportFormat,
    ONNXExporter,
    TorchScriptExporter,
    ModelAnalyzer,
    export_to_onnx,
    export_to_torchscript,
    compare_model_outputs,
)

# 检查 ONNX 是否可用
try:
    import torch.onnx
    # 尝试一个简单的导出来检查 onnxscript
    _test_model = nn.Linear(2, 2)
    _test_input = torch.randn(1, 2)
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=True) as f:
        torch.onnx.export(_test_model, _test_input, f.name)
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

# 跳过 ONNX 测试的装饰器
skip_if_no_onnx = pytest.mark.skipif(
    not ONNX_AVAILABLE,
    reason="ONNX export not available (missing onnxscript or other dependencies)"
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


class ModelWithControlFlow(nn.Module):
    """带控制流的模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        if x.sum() > 0:
            x = torch.relu(x)
        else:
            x = torch.sigmoid(x)
        x = self.fc2(x)
        return x


# ==================== ONNX 导出测试 ====================

class TestONNXExporter:
    """测试 ONNX 导出器"""

    @skip_if_no_onnx
    def test_export_simple_model(self):
        """测试导出简单模型"""
        model = SimpleModel()
        dummy_input = torch.randn(1, 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            config = ExportConfig(verify_export=False, optimize=False)
            exporter = ONNXExporter(config)
            result_path = exporter.export(model, dummy_input, output_path)

            assert os.path.exists(result_path)
            assert result_path == output_path

    @skip_if_no_onnx
    def test_export_conv_model(self):
        """测试导出卷积模型"""
        model = ConvModel()
        dummy_input = torch.randn(1, 3, 32, 32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "conv_model.onnx")

            config = ExportConfig(verify_export=False, optimize=False)
            exporter = ONNXExporter(config)
            result_path = exporter.export(model, dummy_input, output_path)

            assert os.path.exists(result_path)

    @skip_if_no_onnx
    def test_export_with_dynamic_axes(self):
        """测试带动态维度的导出"""
        model = SimpleModel()
        dummy_input = torch.randn(1, 64)

        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "dynamic_model.onnx")

            config = ExportConfig(
                verify_export=False,
                optimize=False,
                dynamic_axes=dynamic_axes
            )
            exporter = ONNXExporter(config)
            result_path = exporter.export(model, dummy_input, output_path)

            assert os.path.exists(result_path)

    @skip_if_no_onnx
    def test_export_with_custom_names(self):
        """测试自定义输入输出名称"""
        model = SimpleModel()
        dummy_input = torch.randn(1, 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "named_model.onnx")

            config = ExportConfig(
                verify_export=False,
                optimize=False,
                input_names=["features"],
                output_names=["logits"]
            )
            exporter = ONNXExporter(config)
            result_path = exporter.export(model, dummy_input, output_path)

            assert os.path.exists(result_path)


# ==================== TorchScript 导出测试 ====================

class TestTorchScriptExporter:
    """测试 TorchScript 导出器"""

    def test_export_trace(self):
        """测试追踪导出"""
        model = SimpleModel()
        example_input = torch.randn(1, 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "traced_model.pt")

            exporter = TorchScriptExporter()
            result_path = exporter.export_trace(
                model, example_input, output_path, optimize=False
            )

            assert os.path.exists(result_path)

    def test_export_script(self):
        """测试脚本化导出"""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "scripted_model.pt")

            exporter = TorchScriptExporter()
            result_path = exporter.export_script(model, output_path, optimize=False)

            assert os.path.exists(result_path)

    def test_traced_model_inference(self):
        """测试追踪模型推理"""
        model = SimpleModel()
        model.eval()
        example_input = torch.randn(1, 64)

        # 直接使用 torch.jit.trace 而不是通过文件加载
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)

            # 验证追踪模型可以运行
            output = traced_model(example_input)
            assert output.shape == (1, 10)

    def test_scripted_model_inference(self):
        """测试脚本化模型推理"""
        model = SimpleModel()
        model.eval()

        # 直接使用 torch.jit.script
        scripted_model = torch.jit.script(model)

        # 验证脚本化模型可以运行
        x = torch.randn(1, 64)
        output = scripted_model(x)
        assert output.shape == (1, 10)

    def test_traced_model_output_consistency(self):
        """测试追踪模型输出一致性"""
        model = SimpleModel()
        model.eval()
        example_input = torch.randn(1, 64)

        # 直接追踪而不是通过文件
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)

            # 比较输出
            original_output = model(example_input)
            traced_output = traced_model(example_input)

        assert torch.allclose(original_output, traced_output, atol=1e-5)


# ==================== ModelAnalyzer 测试 ====================

class TestModelAnalyzer:
    """测试模型分析器"""

    def test_count_parameters(self):
        """测试参数统计"""
        model = SimpleModel()

        stats = ModelAnalyzer.count_parameters(model)

        assert "total" in stats
        assert "trainable" in stats
        assert "non_trainable" in stats
        assert "layers" in stats
        assert stats["total"] > 0
        assert stats["trainable"] == stats["total"]  # 所有参数都可训练

    def test_count_parameters_with_frozen(self):
        """测试带冻结参数的统计"""
        model = SimpleModel()

        # 冻结第一层
        for param in model.fc1.parameters():
            param.requires_grad = False

        stats = ModelAnalyzer.count_parameters(model)

        assert stats["trainable"] < stats["total"]
        assert stats["non_trainable"] > 0

    def test_estimate_model_size(self):
        """测试模型大小估算"""
        model = SimpleModel()

        size_fp32 = ModelAnalyzer.estimate_model_size(model, torch.float32)
        size_fp16 = ModelAnalyzer.estimate_model_size(model, torch.float16)
        size_int8 = ModelAnalyzer.estimate_model_size(model, torch.int8)

        assert "params_mb" in size_fp32
        assert "total_mb" in size_fp32
        assert size_fp32["params_mb"] > size_fp16["params_mb"]
        assert size_fp16["params_mb"] > size_int8["params_mb"]

    def test_profile_inference(self):
        """测试推理性能分析"""
        model = SimpleModel()

        stats = ModelAnalyzer.profile_inference(
            model,
            input_shape=(1, 64),
            num_runs=10,
            warmup_runs=2,
            device="cpu"
        )

        assert "mean_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert "std_ms" in stats
        assert "throughput" in stats
        assert stats["mean_ms"] > 0
        assert stats["throughput"] > 0


# ==================== 便捷函数测试 ====================

class TestExportToONNX:
    """测试 export_to_onnx 便捷函数"""

    @skip_if_no_onnx
    def test_basic_export(self):
        """测试基本导出"""
        model = SimpleModel()
        dummy_input = torch.randn(1, 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            result_path = export_to_onnx(
                model, dummy_input, output_path, verify=False
            )

            assert os.path.exists(result_path)


class TestExportToTorchScript:
    """测试 export_to_torchscript 便捷函数"""

    def test_trace_export(self):
        """测试追踪导出"""
        model = SimpleModel()
        example_input = torch.randn(1, 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            result_path = export_to_torchscript(
                model, output_path,
                method="trace",
                example_input=example_input,
                optimize=False
            )

            assert os.path.exists(result_path)

    def test_script_export(self):
        """测试脚本化导出"""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            result_path = export_to_torchscript(
                model, output_path,
                method="script",
                optimize=False
            )

            assert os.path.exists(result_path)

    def test_trace_without_input_raises(self):
        """测试追踪导出无输入时抛出异常"""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            with pytest.raises(ValueError):
                export_to_torchscript(
                    model, output_path,
                    method="trace",
                    example_input=None
                )

    def test_invalid_method_raises(self):
        """测试无效方法抛出异常"""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            with pytest.raises(ValueError):
                export_to_torchscript(
                    model, output_path,
                    method="invalid"
                )


class TestCompareModelOutputs:
    """测试模型输出比较"""

    def test_identical_models(self):
        """测试相同模型"""
        model1 = SimpleModel()
        model2 = model1  # 同一个模型

        test_inputs = [torch.randn(1, 64) for _ in range(5)]

        results = compare_model_outputs(model1, model2, test_inputs)

        assert results["all_match"] is True
        assert results["max_diff"] == 0.0
        assert len(results["mismatches"]) == 0

    def test_different_models(self):
        """测试不同模型"""
        model1 = SimpleModel()
        model2 = SimpleModel()  # 不同的随机初始化

        test_inputs = [torch.randn(1, 64) for _ in range(5)]

        results = compare_model_outputs(model1, model2, test_inputs)

        # 不同初始化的模型输出应该不同
        assert results["max_diff"] > 0


# ==================== 配置测试 ====================

class TestExportConfig:
    """测试导出配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = ExportConfig()

        assert config.format == ExportFormat.ONNX
        assert config.opset_version == 14
        assert config.verify_export is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            opset_version=13,
            verify_export=False,
            input_names=["x", "y"],
            output_names=["z"]
        )

        assert config.format == ExportFormat.TORCHSCRIPT
        assert config.opset_version == 13
        assert config.verify_export is False
        assert config.input_names == ["x", "y"]
        assert config.output_names == ["z"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
