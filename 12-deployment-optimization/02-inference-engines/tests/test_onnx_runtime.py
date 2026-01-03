"""
ONNX Runtime 模块单元测试
"""

import pytest
import numpy as np
import tempfile
import os
import sys

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from onnx_runtime import (
    SessionConfig,
    GraphOptimizationLevel,
    ExecutionMode,
    ExecutionProvider,
    ONNX_RUNTIME_AVAILABLE,
)

# 检查 ONNX Runtime 是否可用
skip_if_no_onnx_runtime = pytest.mark.skipif(
    not ONNX_RUNTIME_AVAILABLE,
    reason="ONNX Runtime not available"
)

# 检查是否可以创建测试模型
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

skip_if_no_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not available for creating test models"
)


# ==================== 测试模型 ====================

def create_simple_onnx_model(output_path: str):
    """创建简单的 ONNX 测试模型"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required to create test model")

    class SimpleModel(nn.Module):
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

    model = SimpleModel()
    model.eval()
    dummy_input = torch.randn(1, 64)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=14
    )

    return model, dummy_input


# ==================== SessionConfig 测试 ====================

class TestSessionConfig:
    """测试会话配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = SessionConfig()

        assert config.providers == ["CPUExecutionProvider"]
        assert config.graph_optimization_level == GraphOptimizationLevel.ENABLE_ALL
        assert config.intra_op_num_threads == 0
        assert config.inter_op_num_threads == 0
        assert config.execution_mode == ExecutionMode.SEQUENTIAL
        assert config.enable_mem_pattern is True
        assert config.enable_mem_reuse is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = SessionConfig(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            graph_optimization_level=GraphOptimizationLevel.ENABLE_BASIC,
            intra_op_num_threads=4,
            inter_op_num_threads=2,
            execution_mode=ExecutionMode.PARALLEL,
            cuda_device_id=0
        )

        assert "CUDAExecutionProvider" in config.providers
        assert config.graph_optimization_level == GraphOptimizationLevel.ENABLE_BASIC
        assert config.intra_op_num_threads == 4
        assert config.inter_op_num_threads == 2
        assert config.execution_mode == ExecutionMode.PARALLEL
        assert config.cuda_device_id == 0

    @skip_if_no_onnx_runtime
    def test_to_session_options(self):
        """测试转换为 SessionOptions"""
        config = SessionConfig(
            graph_optimization_level=GraphOptimizationLevel.ENABLE_ALL,
            intra_op_num_threads=4,
            enable_mem_pattern=True
        )

        options = config.to_session_options()
        assert options is not None

    def test_get_provider_options(self):
        """测试获取提供者选项"""
        config = SessionConfig(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            cuda_device_id=1,
            cuda_mem_limit=1024 * 1024 * 1024  # 1GB
        )

        provider_options = config.get_provider_options()

        assert len(provider_options) == 2
        assert provider_options[0]["device_id"] == 1
        assert provider_options[0]["gpu_mem_limit"] == 1024 * 1024 * 1024


# ==================== ONNXInferenceSession 测试 ====================

class TestONNXInferenceSession:
    """测试 ONNX 推理会话"""

    @skip_if_no_onnx_runtime
    @skip_if_no_torch
    def test_create_session(self):
        """测试创建会话"""
        from onnx_runtime import ONNXInferenceSession

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            create_simple_onnx_model(model_path)

            session = ONNXInferenceSession(model_path)

            assert session is not None
            assert len(session.input_names) == 1
            assert len(session.output_names) == 1
            assert session.input_names[0] == "input"
            assert session.output_names[0] == "output"

    @skip_if_no_onnx_runtime
    @skip_if_no_torch
    def test_run_inference(self):
        """测试推理"""
        from onnx_runtime import ONNXInferenceSession

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            create_simple_onnx_model(model_path)

            session = ONNXInferenceSession(model_path)

            # 执行推理
            input_data = np.random.randn(1, 64).astype(np.float32)
            outputs = session.run({"input": input_data})

            assert len(outputs) == 1
            assert outputs[0].shape == (1, 10)

    @skip_if_no_onnx_runtime
    @skip_if_no_torch
    def test_run_single(self):
        """测试单输入推理"""
        from onnx_runtime import ONNXInferenceSession

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            create_simple_onnx_model(model_path)

            session = ONNXInferenceSession(model_path)

            # 执行推理
            input_data = np.random.randn(1, 64).astype(np.float32)
            output = session.run_single(input_data)

            assert output.shape == (1, 10)

    @skip_if_no_onnx_runtime
    @skip_if_no_torch
    def test_dynamic_batch_size(self):
        """测试动态批次大小"""
        from onnx_runtime import ONNXInferenceSession

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            create_simple_onnx_model(model_path)

            session = ONNXInferenceSession(model_path)

            # 不同批次大小
            for batch_size in [1, 4, 8, 16]:
                input_data = np.random.randn(batch_size, 64).astype(np.float32)
                output = session.run_single(input_data)
                assert output.shape == (batch_size, 10)

    @skip_if_no_onnx_runtime
    @skip_if_no_torch
    def test_get_input_output_info(self):
        """测试获取输入输出信息"""
        from onnx_runtime import ONNXInferenceSession

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            create_simple_onnx_model(model_path)

            session = ONNXInferenceSession(model_path)

            # 输入信息
            input_shape = session.get_input_shape("input")
            input_dtype = session.get_input_dtype("input")

            assert input_shape is not None
            assert input_dtype is not None

            # 输出信息
            output_shape = session.get_output_shape("output")
            output_dtype = session.get_output_dtype("output")

            assert output_shape is not None
            assert output_dtype is not None


# ==================== Benchmarker 测试 ====================

class TestBenchmarker:
    """测试基准测试器"""

    @skip_if_no_onnx_runtime
    @skip_if_no_torch
    def test_benchmark(self):
        """测试基准测试"""
        from onnx_runtime import ONNXInferenceSession, Benchmarker

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            create_simple_onnx_model(model_path)

            session = ONNXInferenceSession(model_path)
            benchmarker = Benchmarker(session)

            input_data = np.random.randn(1, 64).astype(np.float32)
            stats = benchmarker.benchmark(
                {"input": input_data},
                num_runs=10,
                warmup_runs=2
            )

            assert "mean_ms" in stats
            assert "std_ms" in stats
            assert "min_ms" in stats
            assert "max_ms" in stats
            assert "p50_ms" in stats
            assert "p90_ms" in stats
            assert "p99_ms" in stats
            assert "throughput" in stats
            assert stats["mean_ms"] > 0
            assert stats["throughput"] > 0

    @skip_if_no_onnx_runtime
    @skip_if_no_torch
    def test_benchmark_batch_sizes(self):
        """测试不同批次大小的基准测试"""
        from onnx_runtime import ONNXInferenceSession, Benchmarker

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            create_simple_onnx_model(model_path)

            session = ONNXInferenceSession(model_path)
            benchmarker = Benchmarker(session)

            results = benchmarker.benchmark_batch_sizes(
                input_shape=(64,),
                batch_sizes=[1, 4, 8],
                num_runs=5
            )

            assert len(results) == 3
            for result in results:
                assert "batch_size" in result
                assert "mean_ms" in result
                assert "total_throughput" in result


# ==================== 便捷函数测试 ====================

class TestConvenienceFunctions:
    """测试便捷函数"""

    @skip_if_no_onnx_runtime
    @skip_if_no_torch
    def test_create_session(self):
        """测试 create_session 函数"""
        from onnx_runtime import create_session

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            create_simple_onnx_model(model_path)

            session = create_session(model_path)
            assert session is not None

    @skip_if_no_onnx_runtime
    @skip_if_no_torch
    def test_run_inference_function(self):
        """测试 run_inference 函数"""
        from onnx_runtime import run_inference

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            create_simple_onnx_model(model_path)

            input_data = np.random.randn(1, 64).astype(np.float32)
            outputs = run_inference(model_path, {"input": input_data})

            assert len(outputs) == 1
            assert outputs[0].shape == (1, 10)

    @skip_if_no_onnx_runtime
    def test_get_available_providers(self):
        """测试获取可用提供者"""
        from onnx_runtime import get_available_providers

        providers = get_available_providers()

        assert isinstance(providers, list)
        assert "CPUExecutionProvider" in providers


# ==================== 枚举测试 ====================

class TestEnums:
    """测试枚举类型"""

    def test_execution_provider(self):
        """测试执行提供者枚举"""
        assert ExecutionProvider.CPU.value == "CPUExecutionProvider"
        assert ExecutionProvider.CUDA.value == "CUDAExecutionProvider"
        assert ExecutionProvider.TENSORRT.value == "TensorrtExecutionProvider"

    def test_graph_optimization_level(self):
        """测试图优化级别枚举"""
        assert GraphOptimizationLevel.DISABLE_ALL.value == 0
        assert GraphOptimizationLevel.ENABLE_BASIC.value == 1
        assert GraphOptimizationLevel.ENABLE_EXTENDED.value == 2
        assert GraphOptimizationLevel.ENABLE_ALL.value == 99

    def test_execution_mode(self):
        """测试执行模式枚举"""
        assert ExecutionMode.SEQUENTIAL.value == 0
        assert ExecutionMode.PARALLEL.value == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
