"""
推理引擎模块 (Inference Engines)

提供主流深度学习推理引擎的封装和优化功能。

模块:
- onnx_runtime: ONNX Runtime 推理
- tensorrt_engine: TensorRT 推理
- vllm_inference: vLLM 大语言模型推理
"""

# ONNX Runtime
from .onnx_runtime import (
    # 类
    ONNXInferenceSession,
    IOBindingSession,
    SessionConfig,
    ModelOptimizer,
    Benchmarker,
    # 枚举
    ExecutionProvider,
    GraphOptimizationLevel,
    ExecutionMode,
    # 函数
    create_session,
    run_inference,
    get_available_providers,
    # 常量
    ONNX_RUNTIME_AVAILABLE,
)

# TensorRT
from .tensorrt_engine import (
    # 类
    TensorRTEngine,
    EngineBuilder,
    EngineConfig as TRTEngineConfig,
    Int8Calibrator,
    TRTLogger,
    # 枚举
    Precision,
    CalibrationAlgorithm,
    # 常量
    TENSORRT_AVAILABLE,
    PYCUDA_AVAILABLE,
)

# vLLM
from .vllm_inference import (
    # 类
    VLLMEngine,
    SamplingConfig,
    EngineConfig as VLLMEngineConfig,
    GenerationOutput,
    LoRAAdapter,
    # 枚举
    QuantizationMethod,
    # 函数
    create_engine,
    generate_text,
    # 常量
    VLLM_AVAILABLE,
)

__all__ = [
    # ONNX Runtime
    "ONNXInferenceSession",
    "IOBindingSession",
    "SessionConfig",
    "ModelOptimizer",
    "Benchmarker",
    "ExecutionProvider",
    "GraphOptimizationLevel",
    "ExecutionMode",
    "create_session",
    "run_inference",
    "get_available_providers",
    "ONNX_RUNTIME_AVAILABLE",
    # TensorRT
    "TensorRTEngine",
    "EngineBuilder",
    "TRTEngineConfig",
    "Int8Calibrator",
    "TRTLogger",
    "Precision",
    "CalibrationAlgorithm",
    "TENSORRT_AVAILABLE",
    "PYCUDA_AVAILABLE",
    # vLLM
    "VLLMEngine",
    "SamplingConfig",
    "VLLMEngineConfig",
    "GenerationOutput",
    "LoRAAdapter",
    "QuantizationMethod",
    "create_engine",
    "generate_text",
    "VLLM_AVAILABLE",
]
