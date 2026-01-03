# 模型服务系统 (Model Serving Systems)

> **前置知识**: Python Web 开发基础、REST API、Docker 基础
>
> **学习目标**: 掌握模型服务化部署，实现高可用、高性能的推理服务

---

## 为什么需要模型服务化？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      训练环境 vs 生产环境                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  训练环境                              生产环境                         │
│  ┌─────────────────────┐              ┌─────────────────────┐          │
│  │  Python 脚本         │              │  REST/gRPC API      │          │
│  │  单次推理            │      →       │  高并发处理         │          │
│  │  本地运行            │              │  分布式部署         │          │
│  │  开发调试            │              │  高可用保障         │          │
│  └─────────────────────┘              └─────────────────────┘          │
│                                                                         │
│  服务化解决的问题:                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. 接口标准化: 统一的 API 接口，便于集成                       │   │
│  │  2. 高并发: 支持多用户同时访问                                  │   │
│  │  3. 高可用: 故障自动恢复，服务不中断                            │   │
│  │  4. 可扩展: 根据负载动态扩缩容                                  │   │
│  │  5. 监控: 实时监控服务状态和性能                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 服务架构全景

```
                         模型服务架构
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    FastAPI 服务         Triton Server          负载均衡
    (快速开发)           (高性能)              (高可用)
         │                    │                    │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    │         │         │         │         │         │
  REST API  异步处理   多模型    动态批处理  轮询    健康检查
  Pydantic  批处理     gRPC     GPU调度    加权    自动恢复
    │         │         │         │         │         │
 简单部署   高吞吐    极致性能   资源优化   流量分发  故障转移
```

### 服务方案对比

| 方案 | 性能 | 易用性 | 功能 | 适用场景 |
|:-----|:----:|:------:|:----:|:---------|
| FastAPI | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 快速原型、小规模部署 |
| Triton | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 生产环境、大规模部署 |
| TorchServe | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | PyTorch 项目 |
| TF Serving | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | TensorFlow 项目 |

---

## 核心技术详解

### 1. FastAPI 服务

**核心优势**: 简单灵活、开发快速、自动文档

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI 服务架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  客户端请求                                                 │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FastAPI 应用                                       │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  路由层: @app.post("/predict")              │   │   │
│  │  │  验证层: Pydantic 模型验证                  │   │   │
│  │  │  业务层: 异步推理处理                       │   │   │
│  │  │  响应层: JSON 序列化                        │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  推理引擎 (ONNX Runtime / TensorRT)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**快速使用**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class PredictRequest(BaseModel):
    data: list[float]

@app.post("/predict")
async def predict(request: PredictRequest):
    result = model.predict(request.data)
    return {"prediction": result}

# 启动: uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Triton Inference Server

**核心优势**: 高性能、多模型、动态批处理

```
┌─────────────────────────────────────────────────────────────┐
│                  Triton Inference Server                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Model A │  │ Model B │  │ Model C │  │ Model D │        │
│  │ (ONNX)  │  │ (TRT)   │  │(PyTorch)│  │  (TF)   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  核心功能:                                          │   │
│  │  - 动态批处理: 自动收集请求批量处理                │   │
│  │  - 模型并发: 多模型同时服务                        │   │
│  │  - 版本管理: 支持模型热更新                        │   │
│  │  - GPU 调度: 智能分配 GPU 资源                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  接口: HTTP/REST │ gRPC │ C API                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**模型仓库结构**:
```
model_repository/
├── model_a/
│   ├── config.pbtxt      # 模型配置
│   ├── 1/                # 版本 1
│   │   └── model.onnx
│   └── 2/                # 版本 2
│       └── model.onnx
└── model_b/
    ├── config.pbtxt
    └── 1/
        └── model.plan    # TensorRT 引擎
```

### 3. 负载均衡

**核心优势**: 高可用、可扩展、故障转移

```
┌─────────────────────────────────────────────────────────────┐
│                    负载均衡架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    ┌─────────────────┐                      │
│                    │   负载均衡器     │                      │
│                    │ (Nginx/HAProxy) │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│            ┌────────────────┼────────────────┐              │
│            │                │                │              │
│            ▼                ▼                ▼              │
│     ┌───────────┐    ┌───────────┐    ┌───────────┐        │
│     │  Server 1  │    │  Server 2  │    │  Server 3  │        │
│     │  (健康)    │    │  (健康)    │    │  (故障)    │        │
│     └───────────┘    └───────────┘    └───────────┘        │
│            ↑                ↑                ✗              │
│            └────────────────┴── 流量自动转移               │
│                                                             │
│  负载均衡策略:                                              │
│  - 轮询 (Round Robin): 依次分配                            │
│  - 加权轮询: 按权重分配                                    │
│  - 最少连接: 选择连接数最少的服务器                        │
│  - IP 哈希: 相同 IP 路由到相同服务器                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 文件结构

```
03-serving-systems/
├── README.md                 # 本文件
├── 知识点.md                 # 详细知识点文档
├── src/
│   ├── __init__.py
│   ├── fastapi_server.py     # FastAPI 服务实现
│   ├── triton_client.py      # Triton 客户端
│   └── load_balancer.py      # 负载均衡器
├── notebooks/
│   ├── 01_FastAPI_tutorial.ipynb        # FastAPI 教程
│   ├── 02_Triton_tutorial.ipynb         # Triton 教程
│   ├── 03_LoadBalancing_tutorial.ipynb  # 负载均衡教程
│   └── 04_Advanced_Serving_tutorial.ipynb  # 高级服务技术
└── tests/
    ├── test_fastapi_server.py
    ├── test_triton_client.py
    └── test_load_balancer.py
```

---

## 快速开始

### 安装依赖

```bash
# FastAPI
pip install fastapi uvicorn pydantic

# Triton 客户端
pip install tritonclient[all]

# 负载均衡相关
pip install httpx aiohttp

# 监控
pip install prometheus-client
```

### 使用示例

```python
# ============================================================
# FastAPI 服务
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Model Serving API")

class PredictRequest(BaseModel):
    data: list[float]

class PredictResponse(BaseModel):
    prediction: list[float]
    latency_ms: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start = time.time()
    result = model.predict(request.data)
    latency = (time.time() - start) * 1000
    return PredictResponse(prediction=result, latency_ms=latency)

# 启动: uvicorn main:app --host 0.0.0.0 --port 8000

# ============================================================
# Triton 客户端
# ============================================================
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

# 准备输入
inputs = [httpclient.InferInput("input", data.shape, "FP32")]
inputs[0].set_data_from_numpy(data)

# 执行推理
result = client.infer("model_name", inputs)
output = result.as_numpy("output")

# ============================================================
# 负载均衡
# ============================================================
from load_balancer import LoadBalancer

balancer = LoadBalancer(
    servers=["server1:8000", "server2:8000", "server3:8000"],
    strategy="least_connections"
)

response = await balancer.request("/predict", data={"input": [1.0, 2.0]})
```

---

## 服务选择指南

```
┌─────────────────────────────────────────────────────────────┐
│                    服务方案选择决策树                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                        开始                                 │
│                          │                                  │
│                          ▼                                  │
│                 ┌────────────────┐                          │
│                 │  快速原型？     │                          │
│                 └────────┬───────┘                          │
│                          │                                  │
│              ┌───────────┴───────────┐                      │
│              │                       │                      │
│              ▼                       ▼                      │
│             是                      否                      │
│              │                       │                      │
│              ▼                       ▼                      │
│          FastAPI            ┌────────────────┐              │
│                             │ 需要多模型？    │              │
│                             └────────┬───────┘              │
│                                      │                      │
│                          ┌───────────┴───────────┐          │
│                          │                       │          │
│                          ▼                       ▼          │
│                         是                      否          │
│                          │                       │          │
│                          ▼                       ▼          │
│                       Triton            ┌────────────────┐  │
│                                         │ 高并发需求？    │  │
│                                         └────────┬───────┘  │
│                                                  │          │
│                                      ┌───────────┴───────┐  │
│                                      │                   │  │
│                                      ▼                   ▼  │
│                                     是                  否  │
│                                      │                   │  │
│                                      ▼                   ▼  │
│                              Triton + 负载均衡      FastAPI │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 场景推荐

| 场景 | 推荐方案 | 原因 |
|:-----|:---------|:-----|
| 快速原型 | FastAPI | 开发速度快，易于调试 |
| 单模型生产 | FastAPI + Gunicorn | 简单可靠 |
| 多模型生产 | Triton | 统一管理，资源优化 |
| 高并发场景 | Triton + 负载均衡 | 高性能，可扩展 |
| 边缘部署 | FastAPI + ONNX Runtime | 轻量级 |

---

## 学习路径

1. **01_FastAPI_tutorial.ipynb** - FastAPI 服务基础
2. **02_Triton_tutorial.ipynb** - Triton 服务器使用
3. **03_LoadBalancing_tutorial.ipynb** - 负载均衡实现
4. **04_Advanced_Serving_tutorial.ipynb** - 高级服务技术

---

## 参考资源

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [NGINX 负载均衡](https://docs.nginx.com/nginx/admin-guide/load-balancer/)
- [Kubernetes 服务部署](https://kubernetes.io/docs/concepts/services-networking/)
