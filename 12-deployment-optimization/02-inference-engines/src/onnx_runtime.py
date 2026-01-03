"""
ONNX Runtime 推理模块

提供 ONNX Runtime 推理的封装和优化功能。

主要功能:
1. ONNXInferenceSession: ONNX 推理会话封装
2. SessionConfig: 会话配置
3. IOBinding: 高性能 IO 绑定
4. 性能分析和基准测试
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np

# 检查 ONNX Runtime 是否可用
try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    ort = None


class ExecutionProvider(Enum):
    """执行提供者枚举"""
    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    ROCM = "ROCMExecutionProvider"
    DML = "DmlExecutionProvider"
    COREML = "CoreMLExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"


class GraphOptimizationLevel(Enum):
    """图优化级别"""
    DISABLE_ALL = 0
    ENABLE_BASIC = 1
    ENABLE_EXTENDED = 2
    ENABLE_ALL = 99


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = 0
    PARALLEL = 1


@dataclass
class SessionConfig:
    """ONNX Runtime 会话配置"""
    # 执行提供者
    providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])

    # 图优化
    graph_optimization_level: GraphOptimizationLevel = GraphOptimizationLevel.ENABLE_ALL

    # 线程配置
    intra_op_num_threads: int = 0  # 0 表示自动
    inter_op_num_threads: int = 0

    # 执行模式
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL

    # 内存优化
    enable_mem_pattern: bool = True
    enable_mem_reuse: bool = True

    # 其他选项
    log_severity_level: int = 2  # 0=Verbose, 1=Info, 2=Warning, 3=Error
    optimized_model_filepath: Optional[str] = None

    # CUDA 特定选项
    cuda_device_id: int = 0
    cuda_mem_limit: Optional[int] = None  # 字节
    cuda_arena_extend_strategy: int = 0  # 0=NextPowerOfTwo, 1=SameAsRequested

    def to_session_options(self) -> "ort.SessionOptions":
        """转换为 SessionOptions"""
        if not ONNX_RUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")

        options = ort.SessionOptions()

        # 图优化级别
        opt_levels = {
            GraphOptimizationLevel.DISABLE_ALL: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            GraphOptimizationLevel.ENABLE_BASIC: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            GraphOptimizationLevel.ENABLE_EXTENDED: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            GraphOptimizationLevel.ENABLE_ALL: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        options.graph_optimization_level = opt_levels[self.graph_optimization_level]

        # 线程配置
        if self.intra_op_num_threads > 0:
            options.intra_op_num_threads = self.intra_op_num_threads
        if self.inter_op_num_threads > 0:
            options.inter_op_num_threads = self.inter_op_num_threads

        # 执行模式
        exec_modes = {
            ExecutionMode.SEQUENTIAL: ort.ExecutionMode.ORT_SEQUENTIAL,
            ExecutionMode.PARALLEL: ort.ExecutionMode.ORT_PARALLEL,
        }
        options.execution_mode = exec_modes[self.execution_mode]

        # 内存优化
        options.enable_mem_pattern = self.enable_mem_pattern
        options.enable_mem_reuse = self.enable_mem_reuse

        # 日志级别
        options.log_severity_level = self.log_severity_level

        # 保存优化后的模型
        if self.optimized_model_filepath:
            options.optimized_model_filepath = self.optimized_model_filepath

        return options

    def get_provider_options(self) -> List[Dict[str, Any]]:
        """获取提供者选项"""
        provider_options = []

        for provider in self.providers:
            if provider == "CUDAExecutionProvider":
                cuda_options = {
                    "device_id": self.cuda_device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo" if self.cuda_arena_extend_strategy == 0 else "kSameAsRequested",
                }
                if self.cuda_mem_limit is not None:
                    cuda_options["gpu_mem_limit"] = self.cuda_mem_limit
                provider_options.append(cuda_options)
            else:
                provider_options.append({})

        return provider_options


class ONNXInferenceSession:
    """ONNX Runtime 推理会话封装"""

    def __init__(
        self,
        model_path: str,
        config: Optional[SessionConfig] = None
    ):
        """
        初始化推理会话

        Args:
            model_path: ONNX 模型路径
            config: 会话配置
        """
        if not ONNX_RUNTIME_AVAILABLE:
            raise RuntimeError(
                "ONNX Runtime not available. "
                "Install with: pip install onnxruntime or pip install onnxruntime-gpu"
            )

        self.model_path = model_path
        self.config = config or SessionConfig()

        # 创建会话
        session_options = self.config.to_session_options()
        provider_options = self.config.get_provider_options()

        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=list(zip(self.config.providers, provider_options))
        )

        # 缓存输入输出信息
        self._input_info = {inp.name: inp for inp in self.session.get_inputs()}
        self._output_info = {out.name: out for out in self.session.get_outputs()}

    @property
    def input_names(self) -> List[str]:
        """获取输入名称列表"""
        return list(self._input_info.keys())

    @property
    def output_names(self) -> List[str]:
        """获取输出名称列表"""
        return list(self._output_info.keys())

    def get_input_shape(self, name: str) -> List[Union[int, str]]:
        """获取输入形状"""
        return self._input_info[name].shape

    def get_input_dtype(self, name: str) -> str:
        """获取输入数据类型"""
        return self._input_info[name].type

    def get_output_shape(self, name: str) -> List[Union[int, str]]:
        """获取输出形状"""
        return self._output_info[name].shape

    def get_output_dtype(self, name: str) -> str:
        """获取输出数据类型"""
        return self._output_info[name].type

    def run(
        self,
        inputs: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        执行推理

        Args:
            inputs: 输入字典 {name: array}
            output_names: 要返回的输出名称，None 表示全部

        Returns:
            输出数组列表
        """
        return self.session.run(output_names, inputs)

    def run_single(
        self,
        input_array: np.ndarray,
        input_name: Optional[str] = None
    ) -> np.ndarray:
        """
        单输入单输出推理

        Args:
            input_array: 输入数组
            input_name: 输入名称，None 使用第一个

        Returns:
            输出数组
        """
        if input_name is None:
            input_name = self.input_names[0]

        outputs = self.run({input_name: input_array})
        return outputs[0]

    def get_providers(self) -> List[str]:
        """获取当前使用的执行提供者"""
        return self.session.get_providers()

    def get_session_options(self) -> Dict[str, Any]:
        """获取会话选项信息"""
        return {
            "providers": self.get_providers(),
            "input_names": self.input_names,
            "output_names": self.output_names,
        }


class IOBindingSession:
    """IO Binding 高性能推理会话"""

    def __init__(
        self,
        session: ONNXInferenceSession,
        device_type: str = "cuda",
        device_id: int = 0
    ):
        """
        初始化 IO Binding 会话

        Args:
            session: ONNX 推理会话
            device_type: 设备类型 (cuda, cpu)
            device_id: 设备 ID
        """
        if not ONNX_RUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")

        self.session = session
        self.device_type = device_type
        self.device_id = device_id

        # 创建 IO Binding
        self.io_binding = session.session.io_binding()

        # 缓存绑定的张量
        self._bound_inputs: Dict[str, Any] = {}
        self._bound_outputs: Dict[str, Any] = {}

    def bind_input(
        self,
        name: str,
        data: np.ndarray,
        device_type: Optional[str] = None,
        device_id: Optional[int] = None
    ):
        """
        绑定输入

        Args:
            name: 输入名称
            data: 输入数据
            device_type: 设备类型
            device_id: 设备 ID
        """
        device_type = device_type or self.device_type
        device_id = device_id if device_id is not None else self.device_id

        # 创建 OrtValue
        ort_value = ort.OrtValue.ortvalue_from_numpy(
            data,
            device_type=device_type,
            device_id=device_id
        )

        # 绑定
        self.io_binding.bind_input(
            name=name,
            device_type=device_type,
            device_id=device_id,
            element_type=data.dtype,
            shape=data.shape,
            buffer_ptr=ort_value.data_ptr()
        )

        self._bound_inputs[name] = ort_value

    def bind_output(
        self,
        name: str,
        device_type: Optional[str] = None,
        device_id: Optional[int] = None
    ):
        """
        绑定输出

        Args:
            name: 输出名称
            device_type: 设备类型
            device_id: 设备 ID
        """
        device_type = device_type or self.device_type
        device_id = device_id if device_id is not None else self.device_id

        self.io_binding.bind_output(
            name=name,
            device_type=device_type,
            device_id=device_id
        )

    def run(self) -> List[np.ndarray]:
        """
        执行推理

        Returns:
            输出数组列表
        """
        self.session.session.run_with_iobinding(self.io_binding)
        return [out.numpy() for out in self.io_binding.get_outputs()]

    def clear_bindings(self):
        """清除所有绑定"""
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()
        self._bound_inputs.clear()
        self._bound_outputs.clear()


class ModelOptimizer:
    """ONNX 模型优化器"""

    def __init__(self):
        """初始化优化器"""
        if not ONNX_RUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")

    @staticmethod
    def optimize_model(
        input_path: str,
        output_path: str,
        optimization_level: GraphOptimizationLevel = GraphOptimizationLevel.ENABLE_ALL
    ) -> str:
        """
        优化 ONNX 模型

        Args:
            input_path: 输入模型路径
            output_path: 输出模型路径
            optimization_level: 优化级别

        Returns:
            优化后的模型路径
        """
        if not ONNX_RUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")

        options = ort.SessionOptions()

        opt_levels = {
            GraphOptimizationLevel.DISABLE_ALL: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            GraphOptimizationLevel.ENABLE_BASIC: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            GraphOptimizationLevel.ENABLE_EXTENDED: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            GraphOptimizationLevel.ENABLE_ALL: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        options.graph_optimization_level = opt_levels[optimization_level]
        options.optimized_model_filepath = output_path

        # 创建会话触发优化
        _ = ort.InferenceSession(input_path, sess_options=options)

        return output_path


class Benchmarker:
    """推理性能基准测试"""

    def __init__(self, session: ONNXInferenceSession):
        """
        初始化基准测试器

        Args:
            session: ONNX 推理会话
        """
        self.session = session

    def benchmark(
        self,
        inputs: Dict[str, np.ndarray],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        执行基准测试

        Args:
            inputs: 输入数据
            num_runs: 测试次数
            warmup_runs: 预热次数

        Returns:
            性能统计字典
        """
        # 预热
        for _ in range(warmup_runs):
            self.session.run(inputs)

        # 计时
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.session.run(inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)

        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "throughput": float(1000 / np.mean(latencies)),  # samples/sec
            "num_runs": num_runs,
        }

    def benchmark_batch_sizes(
        self,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int],
        input_name: Optional[str] = None,
        dtype: np.dtype = np.float32,
        num_runs: int = 50
    ) -> List[Dict[str, Any]]:
        """
        测试不同批次大小的性能

        Args:
            input_shape: 输入形状 (不含 batch 维度)
            batch_sizes: 要测试的批次大小列表
            input_name: 输入名称
            dtype: 数据类型
            num_runs: 每个批次的测试次数

        Returns:
            各批次大小的性能统计列表
        """
        if input_name is None:
            input_name = self.session.input_names[0]

        results = []
        for batch_size in batch_sizes:
            full_shape = (batch_size,) + input_shape
            inputs = {input_name: np.random.randn(*full_shape).astype(dtype)}

            stats = self.benchmark(inputs, num_runs=num_runs, warmup_runs=5)
            stats["batch_size"] = batch_size
            stats["total_throughput"] = stats["throughput"] * batch_size

            results.append(stats)

        return results


# 便捷函数
def create_session(
    model_path: str,
    providers: Optional[List[str]] = None,
    **kwargs
) -> ONNXInferenceSession:
    """
    创建 ONNX 推理会话

    Args:
        model_path: 模型路径
        providers: 执行提供者列表
        **kwargs: 其他配置参数

    Returns:
        推理会话
    """
    if providers is None:
        providers = ["CPUExecutionProvider"]

    config = SessionConfig(providers=providers, **kwargs)
    return ONNXInferenceSession(model_path, config)


def run_inference(
    model_path: str,
    inputs: Dict[str, np.ndarray],
    providers: Optional[List[str]] = None
) -> List[np.ndarray]:
    """
    执行单次推理

    Args:
        model_path: 模型路径
        inputs: 输入数据
        providers: 执行提供者

    Returns:
        输出数组列表
    """
    session = create_session(model_path, providers)
    return session.run(inputs)


def get_available_providers() -> List[str]:
    """获取可用的执行提供者"""
    if not ONNX_RUNTIME_AVAILABLE:
        return []
    return ort.get_available_providers()
