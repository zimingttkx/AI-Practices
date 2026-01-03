# 02-inference-engines 推理引擎

本模块介绍主流深度学习推理引擎的使用和优化技术。

## 学习目标

1. **ONNX Runtime**: 跨平台推理引擎
2. **TensorRT**: NVIDIA GPU 高性能推理
3. **vLLM**: 大语言模型推理优化

## 目录结构

```
02-inference-engines/
├── README.md
├── 知识点.md              # 理论知识文档
├── src/
│   ├── __init__.py
│   ├── onnx_runtime.py    # ONNX Runtime 推理
│   ├── tensorrt_engine.py # TensorRT 推理
│   └── vllm_inference.py  # vLLM 推理
├── notebooks/
│   ├── 01_ONNX_Runtime_tutorial.ipynb
│   ├── 02_TensorRT_tutorial.ipynb
│   └── 03_vLLM_tutorial.ipynb
└── tests/
    ├── test_onnx_runtime.py
    ├── test_tensorrt.py
    └── test_vllm.py
```

## 推理引擎对比

| 引擎 | 平台 | 优势 | 适用场景 |
|:-----|:-----|:-----|:---------|
| ONNX Runtime | 跨平台 | 通用性强 | 通用部署 |
| TensorRT | NVIDIA GPU | 极致性能 | GPU 推理 |
| vLLM | GPU | LLM 优化 | 大模型推理 |

## 快速开始

```python
# ONNX Runtime 推理
from inference_engines import ONNXInferenceSession

session = ONNXInferenceSession("model.onnx")
output = session.run(input_data)

# TensorRT 推理
from inference_engines import TensorRTEngine

engine = TensorRTEngine("model.onnx")
output = engine.infer(input_data)

# vLLM 推理
from inference_engines import VLLMEngine

engine = VLLMEngine("meta-llama/Llama-2-7b")
output = engine.generate("Hello, world!")
```

## 依赖安装

```bash
# ONNX Runtime
pip install onnxruntime  # CPU
pip install onnxruntime-gpu  # GPU

# TensorRT (需要 NVIDIA GPU)
pip install tensorrt

# vLLM
pip install vllm
```

## 参考资源

- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [vLLM 文档](https://docs.vllm.ai/)
