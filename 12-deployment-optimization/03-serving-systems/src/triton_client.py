"""
Triton Inference Server 客户端模块

提供 Triton Inference Server 的客户端封装。

主要功能:
1. TritonHTTPClient: HTTP 客户端
2. TritonGRPCClient: gRPC 客户端
3. 模型推理和管理
4. 健康检查
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import asyncio
import time
import json

import numpy as np

# 检查 Triton 客户端是否可用
try:
    import tritonclient.http as httpclient
    TRITON_HTTP_AVAILABLE = True
except ImportError:
    TRITON_HTTP_AVAILABLE = False
    httpclient = None

try:
    import tritonclient.grpc as grpcclient
    TRITON_GRPC_AVAILABLE = True
except ImportError:
    TRITON_GRPC_AVAILABLE = False
    grpcclient = None

TRITON_AVAILABLE = TRITON_HTTP_AVAILABLE or TRITON_GRPC_AVAILABLE


# ==================== 数据类型映射 ====================

class TritonDataType(Enum):
    """Triton 数据类型"""
    BOOL = "BOOL"
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FP16 = "FP16"
    FP32 = "FP32"
    FP64 = "FP64"
    BYTES = "BYTES"


NUMPY_TO_TRITON = {
    np.bool_: TritonDataType.BOOL,
    np.uint8: TritonDataType.UINT8,
    np.uint16: TritonDataType.UINT16,
    np.uint32: TritonDataType.UINT32,
    np.uint64: TritonDataType.UINT64,
    np.int8: TritonDataType.INT8,
    np.int16: TritonDataType.INT16,
    np.int32: TritonDataType.INT32,
    np.int64: TritonDataType.INT64,
    np.float16: TritonDataType.FP16,
    np.float32: TritonDataType.FP32,
    np.float64: TritonDataType.FP64,
}


def numpy_to_triton_dtype(np_dtype: np.dtype) -> str:
    """将 NumPy 数据类型转换为 Triton 数据类型字符串"""
    np_type = np_dtype.type
    if np_type in NUMPY_TO_TRITON:
        return NUMPY_TO_TRITON[np_type].value
    raise ValueError(f"Unsupported numpy dtype: {np_dtype}")


# ==================== 模型信息 ====================

@dataclass
class ModelInput:
    """模型输入信息"""
    name: str
    datatype: str
    shape: List[int]


@dataclass
class ModelOutput:
    """模型输出信息"""
    name: str
    datatype: str
    shape: List[int]


@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str
    versions: List[str]
    platform: str
    inputs: List[ModelInput]
    outputs: List[ModelOutput]


@dataclass
class InferenceResult:
    """推理结果"""
    outputs: Dict[str, np.ndarray]
    model_name: str
    model_version: str
    latency_ms: float


# ==================== HTTP 客户端 ====================

class TritonHTTPClient:
    """Triton HTTP 客户端"""

    def __init__(
        self,
        url: str = "localhost:8000",
        verbose: bool = False,
        connection_timeout: float = 60.0,
        network_timeout: float = 60.0
    ):
        """
        初始化 HTTP 客户端

        Args:
            url: Triton 服务器地址
            verbose: 是否输出详细日志
            connection_timeout: 连接超时时间
            network_timeout: 网络超时时间
        """
        if not TRITON_HTTP_AVAILABLE:
            raise RuntimeError(
                "Triton HTTP client not available. "
                "Install with: pip install tritonclient[http]"
            )

        self.url = url
        self.verbose = verbose
        self.client = httpclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            connection_timeout=connection_timeout,
            network_timeout=network_timeout
        )

    def is_server_live(self) -> bool:
        """检查服务器是否存活"""
        try:
            return self.client.is_server_live()
        except Exception:
            return False

    def is_server_ready(self) -> bool:
        """检查服务器是否就绪"""
        try:
            return self.client.is_server_ready()
        except Exception:
            return False

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """检查模型是否就绪"""
        try:
            return self.client.is_model_ready(model_name, model_version)
        except Exception:
            return False

    def get_model_metadata(self, model_name: str, model_version: str = "") -> ModelMetadata:
        """获取模型元数据"""
        metadata = self.client.get_model_metadata(model_name, model_version)

        inputs = [
            ModelInput(
                name=inp["name"],
                datatype=inp["datatype"],
                shape=inp["shape"]
            )
            for inp in metadata.get("inputs", [])
        ]

        outputs = [
            ModelOutput(
                name=out["name"],
                datatype=out["datatype"],
                shape=out["shape"]
            )
            for out in metadata.get("outputs", [])
        ]

        return ModelMetadata(
            name=metadata["name"],
            versions=metadata.get("versions", []),
            platform=metadata.get("platform", ""),
            inputs=inputs,
            outputs=outputs
        )

    def get_model_config(self, model_name: str, model_version: str = "") -> Dict:
        """获取模型配置"""
        return self.client.get_model_config(model_name, model_version)

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        model_version: str = "",
        outputs: Optional[List[str]] = None,
        request_id: str = "",
        timeout: Optional[float] = None
    ) -> InferenceResult:
        """
        执行推理

        Args:
            model_name: 模型名称
            inputs: 输入数据字典 {name: ndarray}
            model_version: 模型版本
            outputs: 请求的输出名称列表
            request_id: 请求 ID
            timeout: 超时时间

        Returns:
            推理结果
        """
        start_time = time.time()

        # 准备输入
        triton_inputs = []
        for name, data in inputs.items():
            inp = httpclient.InferInput(
                name,
                data.shape,
                numpy_to_triton_dtype(data.dtype)
            )
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)

        # 准备输出
        triton_outputs = None
        if outputs:
            triton_outputs = [
                httpclient.InferRequestedOutput(name)
                for name in outputs
            ]

        # 执行推理
        result = self.client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            model_version=model_version,
            outputs=triton_outputs,
            request_id=request_id,
            timeout=timeout
        )

        # 解析输出
        output_dict = {}
        response = result.get_response()
        for output in response.get("outputs", []):
            name = output["name"]
            output_dict[name] = result.as_numpy(name)

        latency = (time.time() - start_time) * 1000

        return InferenceResult(
            outputs=output_dict,
            model_name=model_name,
            model_version=model_version or "1",
            latency_ms=latency
        )

    def async_infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        callback: callable,
        model_version: str = "",
        outputs: Optional[List[str]] = None,
        request_id: str = ""
    ):
        """
        异步推理

        Args:
            model_name: 模型名称
            inputs: 输入数据
            callback: 回调函数
            model_version: 模型版本
            outputs: 输出名称列表
            request_id: 请求 ID
        """
        # 准备输入
        triton_inputs = []
        for name, data in inputs.items():
            inp = httpclient.InferInput(
                name,
                data.shape,
                numpy_to_triton_dtype(data.dtype)
            )
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)

        # 准备输出
        triton_outputs = None
        if outputs:
            triton_outputs = [
                httpclient.InferRequestedOutput(name)
                for name in outputs
            ]

        # 异步推理
        self.client.async_infer(
            model_name=model_name,
            inputs=triton_inputs,
            callback=callback,
            model_version=model_version,
            outputs=triton_outputs,
            request_id=request_id
        )

    def close(self):
        """关闭客户端"""
        self.client.close()


# ==================== gRPC 客户端 ====================

class TritonGRPCClient:
    """Triton gRPC 客户端 (更高性能)"""

    def __init__(
        self,
        url: str = "localhost:8001",
        verbose: bool = False,
        ssl: bool = False,
        root_certificates: Optional[str] = None,
        private_key: Optional[str] = None,
        certificate_chain: Optional[str] = None
    ):
        """
        初始化 gRPC 客户端

        Args:
            url: Triton 服务器地址
            verbose: 是否输出详细日志
            ssl: 是否使用 SSL
            root_certificates: 根证书
            private_key: 私钥
            certificate_chain: 证书链
        """
        if not TRITON_GRPC_AVAILABLE:
            raise RuntimeError(
                "Triton gRPC client not available. "
                "Install with: pip install tritonclient[grpc]"
            )

        self.url = url
        self.verbose = verbose
        self.client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain
        )

    def is_server_live(self) -> bool:
        """检查服务器是否存活"""
        try:
            return self.client.is_server_live()
        except Exception:
            return False

    def is_server_ready(self) -> bool:
        """检查服务器是否就绪"""
        try:
            return self.client.is_server_ready()
        except Exception:
            return False

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """检查模型是否就绪"""
        try:
            return self.client.is_model_ready(model_name, model_version)
        except Exception:
            return False

    def get_model_metadata(self, model_name: str, model_version: str = "") -> ModelMetadata:
        """获取模型元数据"""
        metadata = self.client.get_model_metadata(model_name, model_version)

        inputs = [
            ModelInput(
                name=inp.name,
                datatype=inp.datatype,
                shape=list(inp.shape)
            )
            for inp in metadata.inputs
        ]

        outputs = [
            ModelOutput(
                name=out.name,
                datatype=out.datatype,
                shape=list(out.shape)
            )
            for out in metadata.outputs
        ]

        return ModelMetadata(
            name=metadata.name,
            versions=list(metadata.versions),
            platform=metadata.platform,
            inputs=inputs,
            outputs=outputs
        )

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        model_version: str = "",
        outputs: Optional[List[str]] = None,
        request_id: str = "",
        timeout: Optional[float] = None
    ) -> InferenceResult:
        """
        执行推理

        Args:
            model_name: 模型名称
            inputs: 输入数据字典
            model_version: 模型版本
            outputs: 输出名称列表
            request_id: 请求 ID
            timeout: 超时时间

        Returns:
            推理结果
        """
        start_time = time.time()

        # 准备输入
        triton_inputs = []
        for name, data in inputs.items():
            inp = grpcclient.InferInput(
                name,
                data.shape,
                numpy_to_triton_dtype(data.dtype)
            )
            inp.set_data_from_numpy(data)
            triton_inputs.append(inp)

        # 准备输出
        triton_outputs = None
        if outputs:
            triton_outputs = [
                grpcclient.InferRequestedOutput(name)
                for name in outputs
            ]

        # 执行推理
        result = self.client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            model_version=model_version,
            outputs=triton_outputs,
            request_id=request_id,
            client_timeout=timeout
        )

        # 解析输出
        output_dict = {}
        for output in result.get_response().outputs:
            name = output.name
            output_dict[name] = result.as_numpy(name)

        latency = (time.time() - start_time) * 1000

        return InferenceResult(
            outputs=output_dict,
            model_name=model_name,
            model_version=model_version or "1",
            latency_ms=latency
        )

    def close(self):
        """关闭客户端"""
        self.client.close()


# ==================== 统一客户端接口 ====================

class TritonClient:
    """统一的 Triton 客户端接口"""

    def __init__(
        self,
        url: str = "localhost:8000",
        protocol: str = "http",
        verbose: bool = False,
        **kwargs
    ):
        """
        初始化统一客户端

        Args:
            url: Triton 服务器地址
            protocol: 协议类型 ("http" 或 "grpc")
            verbose: 是否输出详细日志
            **kwargs: 其他客户端参数
        """
        self.protocol = protocol.lower()

        if self.protocol == "http":
            self.client = TritonHTTPClient(url=url, verbose=verbose, **kwargs)
        elif self.protocol == "grpc":
            self.client = TritonGRPCClient(url=url, verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

    def is_server_live(self) -> bool:
        """检查服务器是否存活"""
        return self.client.is_server_live()

    def is_server_ready(self) -> bool:
        """检查服务器是否就绪"""
        return self.client.is_server_ready()

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """检查模型是否就绪"""
        return self.client.is_model_ready(model_name, model_version)

    def get_model_metadata(self, model_name: str, model_version: str = "") -> ModelMetadata:
        """获取模型元数据"""
        return self.client.get_model_metadata(model_name, model_version)

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        model_version: str = "",
        outputs: Optional[List[str]] = None,
        request_id: str = "",
        timeout: Optional[float] = None
    ) -> InferenceResult:
        """执行推理"""
        return self.client.infer(
            model_name=model_name,
            inputs=inputs,
            model_version=model_version,
            outputs=outputs,
            request_id=request_id,
            timeout=timeout
        )

    def batch_infer(
        self,
        model_name: str,
        batch_inputs: List[Dict[str, np.ndarray]],
        model_version: str = "",
        outputs: Optional[List[str]] = None
    ) -> List[InferenceResult]:
        """
        批量推理

        Args:
            model_name: 模型名称
            batch_inputs: 批量输入列表
            model_version: 模型版本
            outputs: 输出名称列表

        Returns:
            推理结果列表
        """
        results = []
        for i, inputs in enumerate(batch_inputs):
            result = self.infer(
                model_name=model_name,
                inputs=inputs,
                model_version=model_version,
                outputs=outputs,
                request_id=f"batch_{i}"
            )
            results.append(result)
        return results

    def close(self):
        """关闭客户端"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== 模型仓库配置生成 ====================

class ModelConfigGenerator:
    """模型配置生成器"""

    @staticmethod
    def generate_config(
        name: str,
        platform: str,
        max_batch_size: int = 32,
        inputs: List[Dict] = None,
        outputs: List[Dict] = None,
        instance_count: int = 1,
        device: str = "GPU",
        dynamic_batching: bool = True,
        preferred_batch_sizes: List[int] = None,
        max_queue_delay_microseconds: int = 100
    ) -> str:
        """
        生成 Triton 模型配置 (config.pbtxt)

        Args:
            name: 模型名称
            platform: 平台 (onnxruntime_onnx, tensorrt_plan, pytorch_libtorch 等)
            max_batch_size: 最大批次大小
            inputs: 输入配置列表
            outputs: 输出配置列表
            instance_count: 实例数量
            device: 设备类型 (GPU/CPU)
            dynamic_batching: 是否启用动态批处理
            preferred_batch_sizes: 首选批次大小
            max_queue_delay_microseconds: 最大队列延迟

        Returns:
            配置文件内容
        """
        config_lines = [
            f'name: "{name}"',
            f'platform: "{platform}"',
            f'max_batch_size: {max_batch_size}',
            ''
        ]

        # 输入配置
        if inputs:
            for inp in inputs:
                config_lines.append('input [')
                config_lines.append('  {')
                config_lines.append(f'    name: "{inp["name"]}"')
                config_lines.append(f'    data_type: {inp["data_type"]}')
                dims = ', '.join(str(d) for d in inp["dims"])
                config_lines.append(f'    dims: [ {dims} ]')
                config_lines.append('  }')
                config_lines.append(']')
                config_lines.append('')

        # 输出配置
        if outputs:
            for out in outputs:
                config_lines.append('output [')
                config_lines.append('  {')
                config_lines.append(f'    name: "{out["name"]}"')
                config_lines.append(f'    data_type: {out["data_type"]}')
                dims = ', '.join(str(d) for d in out["dims"])
                config_lines.append(f'    dims: [ {dims} ]')
                config_lines.append('  }')
                config_lines.append(']')
                config_lines.append('')

        # 动态批处理配置
        if dynamic_batching:
            config_lines.append('dynamic_batching {')
            if preferred_batch_sizes:
                sizes = ', '.join(str(s) for s in preferred_batch_sizes)
                config_lines.append(f'  preferred_batch_size: [ {sizes} ]')
            config_lines.append(f'  max_queue_delay_microseconds: {max_queue_delay_microseconds}')
            config_lines.append('}')
            config_lines.append('')

        # 实例配置
        kind = "KIND_GPU" if device.upper() == "GPU" else "KIND_CPU"
        config_lines.append('instance_group [')
        config_lines.append('  {')
        config_lines.append(f'    count: {instance_count}')
        config_lines.append(f'    kind: {kind}')
        if device.upper() == "GPU":
            config_lines.append('    gpus: [ 0 ]')
        config_lines.append('  }')
        config_lines.append(']')

        return '\n'.join(config_lines)

    @staticmethod
    def generate_ensemble_config(
        name: str,
        max_batch_size: int,
        inputs: List[Dict],
        outputs: List[Dict],
        steps: List[Dict]
    ) -> str:
        """
        生成 Ensemble 模型配置

        Args:
            name: 模型名称
            max_batch_size: 最大批次大小
            inputs: 输入配置
            outputs: 输出配置
            steps: 处理步骤

        Returns:
            配置文件内容
        """
        config_lines = [
            f'name: "{name}"',
            'platform: "ensemble"',
            f'max_batch_size: {max_batch_size}',
            ''
        ]

        # 输入配置
        for inp in inputs:
            config_lines.append('input [')
            config_lines.append('  {')
            config_lines.append(f'    name: "{inp["name"]}"')
            config_lines.append(f'    data_type: {inp["data_type"]}')
            dims = ', '.join(str(d) for d in inp["dims"])
            config_lines.append(f'    dims: [ {dims} ]')
            config_lines.append('  }')
            config_lines.append(']')
            config_lines.append('')

        # 输出配置
        for out in outputs:
            config_lines.append('output [')
            config_lines.append('  {')
            config_lines.append(f'    name: "{out["name"]}"')
            config_lines.append(f'    data_type: {out["data_type"]}')
            dims = ', '.join(str(d) for d in out["dims"])
            config_lines.append(f'    dims: [ {dims} ]')
            config_lines.append('  }')
            config_lines.append(']')
            config_lines.append('')

        # Ensemble 调度配置
        config_lines.append('ensemble_scheduling {')
        config_lines.append('  step [')

        for step in steps:
            config_lines.append('    {')
            config_lines.append(f'      model_name: "{step["model_name"]}"')
            config_lines.append(f'      model_version: {step.get("model_version", -1)}')

            # 输入映射
            for key, value in step.get("input_map", {}).items():
                config_lines.append('      input_map {')
                config_lines.append(f'        key: "{key}"')
                config_lines.append(f'        value: "{value}"')
                config_lines.append('      }')

            # 输出映射
            for key, value in step.get("output_map", {}).items():
                config_lines.append('      output_map {')
                config_lines.append(f'        key: "{key}"')
                config_lines.append(f'        value: "{value}"')
                config_lines.append('      }')

            config_lines.append('    }')

        config_lines.append('  ]')
        config_lines.append('}')

        return '\n'.join(config_lines)


# ==================== 便捷函数 ====================

def create_triton_client(
    url: str = "localhost:8000",
    protocol: str = "http",
    **kwargs
) -> TritonClient:
    """
    创建 Triton 客户端

    Args:
        url: 服务器地址
        protocol: 协议类型
        **kwargs: 其他参数

    Returns:
        Triton 客户端实例
    """
    return TritonClient(url=url, protocol=protocol, **kwargs)


def quick_infer(
    url: str,
    model_name: str,
    inputs: Dict[str, np.ndarray],
    protocol: str = "http"
) -> Dict[str, np.ndarray]:
    """
    快速推理

    Args:
        url: 服务器地址
        model_name: 模型名称
        inputs: 输入数据
        protocol: 协议类型

    Returns:
        输出数据字典
    """
    with TritonClient(url=url, protocol=protocol) as client:
        result = client.infer(model_name=model_name, inputs=inputs)
        return result.outputs
