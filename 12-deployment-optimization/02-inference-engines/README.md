# 推理引擎 (Inference Engines)

> **前置知识**: PyTorch 基础、ONNX 模型格式、GPU 编程基础
>
> **学习目标**: 掌握主流推理引擎的使用和优化，实现高性能模型部署

---

## 为什么需要推理引擎？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      训练框架 vs 推理引擎                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  训练框架 (PyTorch/TensorFlow)        推理引擎 (ORT/TensorRT/vLLM)      │
│  ┌─────────────────────────┐          ┌─────────────────────────┐       │
│  │  灵活的动态图            │          │  优化的静态图            │       │
│  │  支持自动微分            │          │  算子融合                │       │
│  │  开发调试友好            │    →     │  内存优化                │       │
│  │  性能非最优              │          │  硬件加速                │       │
│  │  通用性强                │          │  极致性能                │       │
│  └─────────────────────────┘          └─────────────────────────┘       │
│                                                                         │
│  核心优化技术:                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  算子融合: Conv + BN + ReLU → 单一融合算子，减少内存访问        │   │
│  │  内存优化: 复用中间张量内存，降低显存占用                       │   │
│  │  量化推理: INT8/FP16 计算，2-4x 加速                            │   │
│  │  并行执行: 多流/多线程，提高吞吐量                              │   │
│  │  图优化: 常量折叠、死代码消除，减少计算量                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 推理引擎全景

```
                         推理引擎选择
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ONNX Runtime          TensorRT              vLLM
    (跨平台通用)         (NVIDIA极致)         (LLM专用)
         │                    │                    │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    │         │         │         │         │         │
   CPU      GPU       FP16     INT8      Paged    Continuous
  推理     推理       推理     推理    Attention  Batching
    │         │         │         │         │         │
 多平台   CUDA EP    层融合   校准量化   内存优化   动态批处理
 部署     加速       优化     加速      高吞吐     低延迟
```

### 推理引擎对比

| 引擎 | 平台支持 | 性能 | 易用性 | 适用场景 |
|:-----|:--------:|:----:|:------:|:---------|
| ONNX Runtime | CPU/GPU/NPU | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 通用部署、跨平台 |
| TensorRT | NVIDIA GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | GPU 极致性能 |
| vLLM | GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | LLM 推理服务 |
| OpenVINO | Intel CPU/GPU | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Intel 硬件优化 |

---

## 核心技术详解

### 1. ONNX Runtime

**核心优势**: 跨平台、跨框架、易于集成

```
┌─────────────────────────────────────────────────────────────┐
│                    ONNX Runtime 架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PyTorch ─────┐                                             │
│               │                                             │
│  TensorFlow ──┼──→ ONNX ──→ ONNX Runtime                   │
│               │              │                              │
│  Keras ───────┘              ├─→ CPUExecutionProvider       │
│                              ├─→ CUDAExecutionProvider      │
│                              ├─→ TensorrtExecutionProvider  │
│                              ├─→ CoreMLExecutionProvider    │
│                              └─→ OpenVINOExecutionProvider  │
│                                                             │
│  Execution Provider (EP) 机制:                              │
│  - 统一 API，多后端支持                                     │
│  - 自动选择最优执行路径                                     │
│  - 支持混合执行 (部分算子 GPU，部分 CPU)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**快速使用**:
```python
import onnxruntime as ort

# 创建推理会话
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 配置优化选项
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4

# 执行推理
output = session.run(None, {'input': input_data})
```

### 2. TensorRT

**核心优势**: NVIDIA GPU 极致性能优化

```
┌─────────────────────────────────────────────────────────────┐
│                    TensorRT 优化流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ONNX Model                                                 │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              TensorRT Builder                        │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  1. 层融合 (Layer Fusion)                   │    │   │
│  │  │     Conv + BN + ReLU → ConvBNReLU          │    │   │
│  │  │                                             │    │   │
│  │  │  2. 精度校准 (Calibration)                  │    │   │
│  │  │     FP32 → FP16/INT8                       │    │   │
│  │  │                                             │    │   │
│  │  │  3. 内核自动调优 (Auto-Tuning)              │    │   │
│  │  │     选择最优 CUDA 内核实现                  │    │   │
│  │  │                                             │    │   │
│  │  │  4. 内存优化                                │    │   │
│  │  │     张量复用、内存池                        │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  TensorRT Engine (.engine/.plan)                           │
│  (序列化的优化模型，可直接加载推理)                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**快速使用**:
```python
import tensorrt as trt

# 构建引擎
builder = trt.Builder(trt.Logger())
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, trt.Logger())

# 解析 ONNX
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# 配置优化
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16

# 构建并保存引擎
engine = builder.build_serialized_network(network, config)
```

### 3. vLLM

**核心优势**: LLM 推理专用优化，PagedAttention 技术

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM 核心技术                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  传统 KV Cache 问题:                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  请求 1: [████████████░░░░░░░░]  预分配但未使用     │   │
│  │  请求 2: [██████░░░░░░░░░░░░░░]  大量内存浪费       │   │
│  │  请求 3: [████████████████░░░░]                     │   │
│  │          ↑ 实际使用    ↑ 浪费                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  PagedAttention 解决方案:                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  物理内存块 (Pages):                                │   │
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┐         │   │
│  │  │ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │         │   │
│  │  └────┴────┴────┴────┴────┴────┴────┴────┘         │   │
│  │    ↑    ↑    ↑    ↑    ↑    ↑                       │   │
│  │  请求1: [P0, P1, P2]                                │   │
│  │  请求2: [P3, P4]                                    │   │
│  │  请求3: [P5, P6, P7]                                │   │
│  │                                                     │   │
│  │  - 按需分配，无浪费                                 │   │
│  │  - 支持动态增长                                     │   │
│  │  - 内存利用率接近 100%                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Continuous Batching:                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  传统: 等最长请求完成才处理新请求                   │   │
│  │  vLLM: 请求完成后立即处理新请求，动态调度           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**快速使用**:
```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256
)

# 批量推理
outputs = llm.generate(prompts, sampling_params)
```

---

## 文件结构

```
02-inference-engines/
├── README.md                 # 本文件
├── 知识点.md                 # 详细知识点文档
├── src/
│   ├── __init__.py
│   ├── onnx_runtime.py       # ONNX Runtime 封装
│   ├── tensorrt_engine.py    # TensorRT 封装
│   └── vllm_inference.py     # vLLM 封装
├── notebooks/
│   ├── 01_ONNX_Runtime_tutorial.ipynb    # ONNX Runtime 教程
│   ├── 02_TensorRT_tutorial.ipynb        # TensorRT 教程
│   ├── 03_vLLM_tutorial.ipynb            # vLLM 教程
│   └── 04_Advanced_Inference_tutorial.ipynb  # 高级推理技术
└── tests/
    ├── test_onnx_runtime.py
    ├── test_tensorrt.py
    └── test_vllm.py
```

---

## 快速开始

### 安装依赖

```bash
# ONNX Runtime
pip install onnxruntime           # CPU 版本
pip install onnxruntime-gpu       # GPU 版本

# TensorRT (需要 NVIDIA GPU 和 CUDA)
pip install tensorrt

# vLLM (需要 GPU)
pip install vllm
```

### 使用示例

```python
# ============================================================
# ONNX Runtime 推理
# ============================================================
from inference_engines import ONNXInferenceSession, SessionConfig

config = SessionConfig(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    graph_optimization_level='ORT_ENABLE_ALL'
)
session = ONNXInferenceSession("model.onnx", config)
output = session.run_single(input_data)

# ============================================================
# TensorRT 推理
# ============================================================
from inference_engines import TensorRTEngine

engine = TensorRTEngine("model.onnx", precision="fp16")
output = engine.infer(input_data)

# ============================================================
# vLLM 推理
# ============================================================
from inference_engines import VLLMEngine

engine = VLLMEngine("meta-llama/Llama-2-7b-hf")
output = engine.generate("What is machine learning?")
```

---

## 引擎选择指南

```
┌─────────────────────────────────────────────────────────────┐
│                    推理引擎选择决策树                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                        开始                                 │
│                          │                                  │
│                          ▼                                  │
│                 ┌────────────────┐                          │
│                 │  是否是 LLM？   │                          │
│                 └────────┬───────┘                          │
│                          │                                  │
│              ┌───────────┴───────────┐                      │
│              │                       │                      │
│              ▼                       ▼                      │
│             是                      否                      │
│              │                       │                      │
│              ▼                       ▼                      │
│           vLLM              ┌────────────────┐              │
│                             │ 需要跨平台？    │              │
│                             └────────┬───────┘              │
│                                      │                      │
│                          ┌───────────┴───────────┐          │
│                          │                       │          │
│                          ▼                       ▼          │
│                         是                      否          │
│                          │                       │          │
│                          ▼                       ▼          │
│                   ONNX Runtime          ┌────────────────┐  │
│                                         │ NVIDIA GPU？    │  │
│                                         └────────┬───────┘  │
│                                                  │          │
│                                      ┌───────────┴───────┐  │
│                                      │                   │  │
│                                      ▼                   ▼  │
│                                     是                  否  │
│                                      │                   │  │
│                                      ▼                   ▼  │
│                                 TensorRT          ONNX RT   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 场景推荐

| 场景 | 推荐引擎 | 原因 |
|:-----|:---------|:-----|
| 通用模型部署 | ONNX Runtime + CUDA EP | 易用性高，性能好 |
| NVIDIA GPU 极致性能 | TensorRT | 最优 GPU 性能 |
| LLM 推理服务 | vLLM | PagedAttention 优化 |
| 边缘设备部署 | ONNX Runtime | 跨平台支持 |
| Intel 硬件 | OpenVINO | Intel 专用优化 |

---

## 学习路径

1. **01_ONNX_Runtime_tutorial.ipynb** - ONNX Runtime 基础与优化
2. **02_TensorRT_tutorial.ipynb** - TensorRT 构建与推理
3. **03_vLLM_tutorial.ipynb** - vLLM LLM 推理优化
4. **04_Advanced_Inference_tutorial.ipynb** - 高级推理技术

---

## 参考资源

- [ONNX Runtime 官方文档](https://onnxruntime.ai/docs/)
- [TensorRT 开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
