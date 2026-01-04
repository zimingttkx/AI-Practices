# 12 - Deployment Optimization

This module covers model compression, inference acceleration, serving, and MLOps.

## Module Structure

```
12-deployment-optimization/
├── 01-model-optimization/      # Model Optimization
├── 02-inference-engines/       # Inference Engines
├── 03-serving-systems/         # Serving Systems
└── 04-mlops/                   # MLOps
```

## Core Content

### 01 - Model Optimization

| Technology | Description | Compression |
|:-----------|:------------|:------------|
| Quantization | FP32→INT8/INT4 | 2-4x |
| Pruning | Remove redundant params | 2-10x |
| Distillation | Large→Small model | Variable |
| ONNX Export | Cross-platform | - |

### 02 - Inference Engines

| Engine | Features | Use Case |
|:-------|:---------|:---------|
| TensorRT | NVIDIA GPU optimized | High-performance |
| ONNX Runtime | Cross-platform | General deployment |
| vLLM | LLM specialized | LLM serving |
| Triton | Multi-model serving | Production |

### 03 - Serving Systems

| Technology | Function |
|:-----------|:---------|
| FastAPI | REST API |
| gRPC | High-performance RPC |
| Load Balancing | Traffic distribution |
| Batching | Throughput optimization |

### 04 - MLOps

| Component | Tools |
|:----------|:------|
| Experiment Tracking | MLflow, W&B |
| Model Registry | MLflow Registry |
| Monitoring | Prometheus, Grafana |
| CI/CD | GitHub Actions |

## Learning Path

```
Quantization → ONNX → TensorRT → FastAPI → MLOps
```
