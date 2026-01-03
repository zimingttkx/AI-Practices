"""
TensorRT 推理引擎模块

提供 TensorRT 推理的封装和优化功能。

主要功能:
1. TensorRTEngine: TensorRT 推理引擎封装
2. EngineBuilder: 引擎构建器
3. Calibrator: INT8 校准器
4. 性能分析和基准测试

注意: TensorRT 仅支持 NVIDIA GPU
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import numpy as np
import os

# 检查 TensorRT 是否可用
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None

# 检查 PyCUDA 是否可用
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    cuda = None


class Precision(Enum):
    """精度模式"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class CalibrationAlgorithm(Enum):
    """INT8 校准算法"""
    ENTROPY = "entropy"
    MINMAX = "minmax"
    PERCENTILE = "percentile"


@dataclass
class EngineConfig:
    """TensorRT 引擎配置"""
    # 精度设置
    precision: Precision = Precision.FP16

    # 工作空间大小 (字节)
    workspace_size: int = 1 << 30  # 1GB

    # 动态形状配置
    min_batch_size: int = 1
    opt_batch_size: int = 8
    max_batch_size: int = 32

    # INT8 校准
    calibration_data: Optional[List[np.ndarray]] = None
    calibration_cache_file: Optional[str] = None
    calibration_algorithm: CalibrationAlgorithm = CalibrationAlgorithm.ENTROPY

    # 优化选项
    strict_type_constraints: bool = False
    max_aux_streams: int = -1

    # 日志级别
    log_level: int = 2  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR


class TRTLogger:
    """TensorRT 日志记录器"""

    def __init__(self, log_level: int = 2):
        """
        初始化日志记录器

        Args:
            log_level: 日志级别
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        severity_map = {
            0: trt.Logger.VERBOSE,
            1: trt.Logger.INFO,
            2: trt.Logger.WARNING,
            3: trt.Logger.ERROR,
        }
        self.logger = trt.Logger(severity_map.get(log_level, trt.Logger.WARNING))

    def get_logger(self) -> "trt.Logger":
        """获取 TensorRT Logger"""
        return self.logger


class Int8Calibrator:
    """INT8 校准器"""

    def __init__(
        self,
        calibration_data: List[np.ndarray],
        cache_file: Optional[str] = None,
        algorithm: CalibrationAlgorithm = CalibrationAlgorithm.ENTROPY
    ):
        """
        初始化校准器

        Args:
            calibration_data: 校准数据列表
            cache_file: 校准缓存文件路径
            algorithm: 校准算法
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available")

        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.algorithm = algorithm
        self.current_index = 0
        self.batch_size = calibration_data[0].shape[0] if calibration_data else 1

        # 分配设备内存
        self.device_input = None
        if calibration_data:
            self.device_input = cuda.mem_alloc(calibration_data[0].nbytes)

    def get_batch_size(self) -> int:
        """获取批次大小"""
        return self.batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """
        获取下一批校准数据

        Args:
            names: 输入名称列表

        Returns:
            设备内存指针列表，或 None 表示结束
        """
        if self.current_index >= len(self.calibration_data):
            return None

        # 拷贝数据到设备
        batch = self.calibration_data[self.current_index]
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += 1

        return [int(self.device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        """读取校准缓存"""
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes):
        """写入校准缓存"""
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                f.write(cache)


class EngineBuilder:
    """TensorRT 引擎构建器"""

    def __init__(self, config: Optional[EngineConfig] = None):
        """
        初始化构建器

        Args:
            config: 引擎配置
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT not available. "
                "Install with: pip install tensorrt"
            )

        self.config = config or EngineConfig()
        self.trt_logger = TRTLogger(self.config.log_level)

    def build_from_onnx(
        self,
        onnx_path: str,
        engine_path: Optional[str] = None,
        input_shapes: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None
    ) -> bytes:
        """
        从 ONNX 模型构建 TensorRT 引擎

        Args:
            onnx_path: ONNX 模型路径
            engine_path: 引擎保存路径 (可选)
            input_shapes: 输入形状配置 {name: (min_shape, opt_shape, max_shape)}

        Returns:
            序列化的引擎字节
        """
        logger = self.trt_logger.get_logger()

        # 创建 Builder
        builder = trt.Builder(logger)

        # 创建网络 (显式批次)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)

        # 解析 ONNX
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"ONNX Parser Error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        # 创建配置
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.config.workspace_size)

        # 设置精度
        if self.config.precision == Precision.FP16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.precision == Precision.INT8:
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                # 设置校准器
                if self.config.calibration_data:
                    calibrator = Int8Calibrator(
                        self.config.calibration_data,
                        self.config.calibration_cache_file,
                        self.config.calibration_algorithm
                    )
                    config.int8_calibrator = calibrator

        # 配置动态形状
        if input_shapes:
            profile = builder.create_optimization_profile()
            for name, (min_shape, opt_shape, max_shape) in input_shapes.items():
                profile.set_shape(name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
        else:
            # 使用默认的批次大小配置
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                shape = input_tensor.shape
                if shape[0] == -1:  # 动态批次
                    min_shape = (self.config.min_batch_size,) + tuple(shape[1:])
                    opt_shape = (self.config.opt_batch_size,) + tuple(shape[1:])
                    max_shape = (self.config.max_batch_size,) + tuple(shape[1:])
                    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

        # 构建引擎
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # 保存引擎
        if engine_path:
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

        return serialized_engine


class TensorRTEngine:
    """TensorRT 推理引擎"""

    def __init__(
        self,
        engine_path: Optional[str] = None,
        serialized_engine: Optional[bytes] = None,
        log_level: int = 2
    ):
        """
        初始化推理引擎

        Args:
            engine_path: 引擎文件路径
            serialized_engine: 序列化的引擎字节
            log_level: 日志级别
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        if not PYCUDA_AVAILABLE:
            raise RuntimeError("PyCUDA not available")

        self.trt_logger = TRTLogger(log_level)
        logger = self.trt_logger.get_logger()

        # 加载引擎
        runtime = trt.Runtime(logger)

        if engine_path:
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()

        if serialized_engine is None:
            raise ValueError("Must provide either engine_path or serialized_engine")

        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 分配内存
        self._allocate_buffers()

        # 创建 CUDA 流
        self.stream = cuda.Stream()

    def _allocate_buffers(self):
        """分配输入输出缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)

            # 处理动态形状
            if -1 in shape:
                # 使用最大形状分配内存
                shape = tuple(max(1, s) for s in shape)

            size = int(np.prod(shape))

            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            buffer_info = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "host": host_mem,
                "device": device_mem,
            }

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(buffer_info)
            else:
                self.outputs.append(buffer_info)

    @property
    def input_names(self) -> List[str]:
        """获取输入名称列表"""
        return [inp["name"] for inp in self.inputs]

    @property
    def output_names(self) -> List[str]:
        """获取输出名称列表"""
        return [out["name"] for out in self.outputs]

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
        sync: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        执行推理

        Args:
            inputs: 输入数据字典 {name: array}
            sync: 是否同步执行

        Returns:
            输出数据字典 {name: array}
        """
        # 设置输入形状并拷贝数据
        for inp_info in self.inputs:
            name = inp_info["name"]
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")

            data = inputs[name]

            # 设置动态形状
            self.context.set_input_shape(name, data.shape)

            # 拷贝到设备
            np.copyto(inp_info["host"][:data.size], data.ravel())
            cuda.memcpy_htod_async(inp_info["device"], inp_info["host"], self.stream)

        # 设置张量地址
        for inp_info in self.inputs:
            self.context.set_tensor_address(inp_info["name"], int(inp_info["device"]))
        for out_info in self.outputs:
            self.context.set_tensor_address(out_info["name"], int(out_info["device"]))

        # 执行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 拷贝输出到主机
        outputs = {}
        for out_info in self.outputs:
            name = out_info["name"]
            shape = self.context.get_tensor_shape(name)
            size = int(np.prod(shape))

            cuda.memcpy_dtoh_async(out_info["host"], out_info["device"], self.stream)

            if sync:
                self.stream.synchronize()

            outputs[name] = out_info["host"][:size].reshape(shape).copy()

        return outputs

    def infer_single(
        self,
        input_array: np.ndarray,
        input_name: Optional[str] = None
    ) -> np.ndarray:
        """
        单输入单输出推理

        Args:
            input_array: 输入数组
            input_name: 输入名称

        Returns:
            输出数组
        """
        if input_name is None:
            input_name = self.input_names[0]

        outputs = self.infer({input_name: input_array})
        return list(outputs.values())[0]
