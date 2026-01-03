"""
Triton 客户端模块测试
"""

import pytest
import sys
import os
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from triton_client import (
    TRITON_AVAILABLE,
    TRITON_HTTP_AVAILABLE,
    TRITON_GRPC_AVAILABLE,
    TritonDataType,
    numpy_to_triton_dtype,
    ModelInput,
    ModelOutput,
    ModelMetadata,
    InferenceResult,
    ModelConfigGenerator,
)


# ==================== 数据类型转换测试 ====================

class TestDataTypeConversion:
    """数据类型转换测试"""

    def test_numpy_to_triton_float32(self):
        """测试 float32 转换"""
        arr = np.array([1.0], dtype=np.float32)
        dtype = numpy_to_triton_dtype(arr.dtype)
        assert dtype == "FP32"

    def test_numpy_to_triton_float64(self):
        """测试 float64 转换"""
        arr = np.array([1.0], dtype=np.float64)
        dtype = numpy_to_triton_dtype(arr.dtype)
        assert dtype == "FP64"

    def test_numpy_to_triton_int32(self):
        """测试 int32 转换"""
        arr = np.array([1], dtype=np.int32)
        dtype = numpy_to_triton_dtype(arr.dtype)
        assert dtype == "INT32"

    def test_numpy_to_triton_int64(self):
        """测试 int64 转换"""
        arr = np.array([1], dtype=np.int64)
        dtype = numpy_to_triton_dtype(arr.dtype)
        assert dtype == "INT64"

    def test_numpy_to_triton_uint8(self):
        """测试 uint8 转换"""
        arr = np.array([1], dtype=np.uint8)
        dtype = numpy_to_triton_dtype(arr.dtype)
        assert dtype == "UINT8"

    def test_numpy_to_triton_bool(self):
        """测试 bool 转换"""
        arr = np.array([True], dtype=np.bool_)
        dtype = numpy_to_triton_dtype(arr.dtype)
        assert dtype == "BOOL"

    def test_triton_data_type_enum(self):
        """测试 TritonDataType 枚举"""
        assert TritonDataType.FP32.value == "FP32"
        assert TritonDataType.INT64.value == "INT64"
        assert TritonDataType.BOOL.value == "BOOL"


# ==================== 数据类测试 ====================

class TestDataClasses:
    """数据类测试"""

    def test_model_input(self):
        """测试 ModelInput"""
        inp = ModelInput(
            name="input",
            datatype="FP32",
            shape=[1, 3, 224, 224]
        )
        assert inp.name == "input"
        assert inp.datatype == "FP32"
        assert inp.shape == [1, 3, 224, 224]

    def test_model_output(self):
        """测试 ModelOutput"""
        out = ModelOutput(
            name="output",
            datatype="FP32",
            shape=[1, 1000]
        )
        assert out.name == "output"
        assert out.datatype == "FP32"
        assert out.shape == [1, 1000]

    def test_model_metadata(self):
        """测试 ModelMetadata"""
        metadata = ModelMetadata(
            name="resnet50",
            versions=["1", "2"],
            platform="onnxruntime_onnx",
            inputs=[ModelInput("input", "FP32", [1, 3, 224, 224])],
            outputs=[ModelOutput("output", "FP32", [1, 1000])]
        )
        assert metadata.name == "resnet50"
        assert len(metadata.versions) == 2
        assert len(metadata.inputs) == 1
        assert len(metadata.outputs) == 1

    def test_inference_result(self):
        """测试 InferenceResult"""
        result = InferenceResult(
            outputs={"output": np.array([0.1, 0.9])},
            model_name="test_model",
            model_version="1",
            latency_ms=5.0
        )
        assert result.model_name == "test_model"
        assert result.model_version == "1"
        assert result.latency_ms == 5.0
        assert "output" in result.outputs


# ==================== 配置生成器测试 ====================

class TestModelConfigGenerator:
    """ModelConfigGenerator 测试"""

    def test_generate_basic_config(self):
        """测试生成基本配置"""
        config = ModelConfigGenerator.generate_config(
            name="test_model",
            platform="onnxruntime_onnx",
            max_batch_size=32
        )

        assert 'name: "test_model"' in config
        assert 'platform: "onnxruntime_onnx"' in config
        assert 'max_batch_size: 32' in config

    def test_generate_config_with_inputs_outputs(self):
        """测试生成带输入输出的配置"""
        config = ModelConfigGenerator.generate_config(
            name="image_classifier",
            platform="onnxruntime_onnx",
            max_batch_size=32,
            inputs=[{
                "name": "input",
                "data_type": "TYPE_FP32",
                "dims": [3, 224, 224]
            }],
            outputs=[{
                "name": "output",
                "data_type": "TYPE_FP32",
                "dims": [1000]
            }]
        )

        assert 'name: "input"' in config
        assert 'data_type: TYPE_FP32' in config
        assert 'dims: [ 3, 224, 224 ]' in config
        assert 'name: "output"' in config

    def test_generate_config_with_dynamic_batching(self):
        """测试生成带动态批处理的配置"""
        config = ModelConfigGenerator.generate_config(
            name="test_model",
            platform="onnxruntime_onnx",
            dynamic_batching=True,
            preferred_batch_sizes=[4, 8, 16],
            max_queue_delay_microseconds=100
        )

        assert 'dynamic_batching {' in config
        assert 'preferred_batch_size: [ 4, 8, 16 ]' in config
        assert 'max_queue_delay_microseconds: 100' in config

    def test_generate_config_gpu_instance(self):
        """测试生成 GPU 实例配置"""
        config = ModelConfigGenerator.generate_config(
            name="test_model",
            platform="tensorrt_plan",
            instance_count=2,
            device="GPU"
        )

        assert 'instance_group [' in config
        assert 'count: 2' in config
        assert 'kind: KIND_GPU' in config
        assert 'gpus: [ 0 ]' in config

    def test_generate_config_cpu_instance(self):
        """测试生成 CPU 实例配置"""
        config = ModelConfigGenerator.generate_config(
            name="test_model",
            platform="onnxruntime_onnx",
            device="CPU"
        )

        assert 'kind: KIND_CPU' in config
        assert 'gpus' not in config

    def test_generate_ensemble_config(self):
        """测试生成 Ensemble 配置"""
        config = ModelConfigGenerator.generate_ensemble_config(
            name="pipeline",
            max_batch_size=32,
            inputs=[{
                "name": "raw_image",
                "data_type": "TYPE_UINT8",
                "dims": [-1, -1, 3]
            }],
            outputs=[{
                "name": "classification",
                "data_type": "TYPE_FP32",
                "dims": [1000]
            }],
            steps=[
                {
                    "model_name": "preprocessing",
                    "model_version": -1,
                    "input_map": {"raw_input": "raw_image"},
                    "output_map": {"processed_output": "preprocessed"}
                },
                {
                    "model_name": "classifier",
                    "model_version": -1,
                    "input_map": {"input": "preprocessed"},
                    "output_map": {"output": "classification"}
                }
            ]
        )

        assert 'platform: "ensemble"' in config
        assert 'ensemble_scheduling {' in config
        assert 'model_name: "preprocessing"' in config
        assert 'model_name: "classifier"' in config
        assert 'input_map {' in config
        assert 'output_map {' in config


# ==================== 客户端测试 (需要 Triton 服务器) ====================

@pytest.mark.skipif(not TRITON_HTTP_AVAILABLE, reason="Triton HTTP client not available")
class TestTritonHTTPClient:
    """TritonHTTPClient 测试 (需要运行中的 Triton 服务器)"""

    def test_client_creation(self):
        """测试客户端创建"""
        from triton_client import TritonHTTPClient

        # 只测试创建，不连接服务器
        client = TritonHTTPClient(url="localhost:8000")
        assert client.url == "localhost:8000"
        client.close()

    def test_server_not_available(self):
        """测试服务器不可用"""
        from triton_client import TritonHTTPClient

        client = TritonHTTPClient(url="localhost:9999")
        assert client.is_server_live() is False
        client.close()


@pytest.mark.skipif(not TRITON_GRPC_AVAILABLE, reason="Triton gRPC client not available")
class TestTritonGRPCClient:
    """TritonGRPCClient 测试"""

    def test_client_creation(self):
        """测试客户端创建"""
        from triton_client import TritonGRPCClient

        client = TritonGRPCClient(url="localhost:8001")
        assert client.url == "localhost:8001"
        client.close()


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton client not available")
class TestTritonClient:
    """统一 TritonClient 测试"""

    def test_http_protocol(self):
        """测试 HTTP 协议"""
        from triton_client import TritonClient

        client = TritonClient(url="localhost:8000", protocol="http")
        assert client.protocol == "http"
        client.close()

    @pytest.mark.skipif(not TRITON_GRPC_AVAILABLE, reason="gRPC not available")
    def test_grpc_protocol(self):
        """测试 gRPC 协议"""
        from triton_client import TritonClient

        client = TritonClient(url="localhost:8001", protocol="grpc")
        assert client.protocol == "grpc"
        client.close()

    def test_invalid_protocol(self):
        """测试无效协议"""
        from triton_client import TritonClient

        with pytest.raises(ValueError):
            TritonClient(url="localhost:8000", protocol="invalid")

    def test_context_manager(self):
        """测试上下文管理器"""
        from triton_client import TritonClient

        with TritonClient(url="localhost:8000", protocol="http") as client:
            assert client is not None


# ==================== 便捷函数测试 ====================

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton client not available")
class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_create_triton_client(self):
        """测试 create_triton_client"""
        from triton_client import create_triton_client

        client = create_triton_client(url="localhost:8000", protocol="http")
        assert client is not None
        client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
