# 12 - 部署优化

本模块涵盖模型压缩、推理加速、服务部署和MLOps全流程。

## 模块结构

```
12-deployment-optimization/
├── 01-model-optimization/      # 模型优化
├── 02-inference-engines/       # 推理引擎
├── 03-serving-systems/         # 服务系统
└── 04-mlops/                   # MLOps
```

## 核心内容

### 01 - 模型优化

| 技术 | 说明 | 压缩比 |
|:-----|:-----|:-------|
| 量化 | FP32→INT8/INT4 | 2-4x |
| 剪枝 | 移除冗余参数 | 2-10x |
| 蒸馏 | 大模型→小模型 | 可变 |
| ONNX导出 | 跨平台部署 | - |

**量化类型**：
- 动态量化：推理时量化
- 静态量化：校准后量化
- 量化感知训练：训练时模拟量化

### 02 - 推理引擎

| 引擎 | 特点 | 适用场景 |
|:-----|:-----|:---------|
| TensorRT | NVIDIA GPU优化 | 高性能推理 |
| ONNX Runtime | 跨平台 | 通用部署 |
| vLLM | LLM专用 | 大模型服务 |
| Triton | 多模型服务 | 生产环境 |

### 03 - 服务系统

| 技术 | 功能 |
|:-----|:-----|
| FastAPI | REST API服务 |
| gRPC | 高性能RPC |
| 负载均衡 | 流量分发 |
| 批处理 | 提高吞吐量 |

### 04 - MLOps

| 组件 | 工具 |
|:-----|:-----|
| 实验追踪 | MLflow、W&B |
| 模型注册 | MLflow Registry |
| 监控告警 | Prometheus、Grafana |
| CI/CD | GitHub Actions |

## 学习路线

```
量化剪枝 → ONNX导出 → TensorRT → FastAPI → MLOps
```

## 依赖库

```python
onnx>=1.14.0
onnxruntime>=1.15.0
tensorrt>=8.6.0  # NVIDIA GPU
fastapi>=0.100.0
uvicorn>=0.23.0
mlflow>=2.5.0
```
