# 03-serving-systems 模型服务系统

本模块介绍深度学习模型的服务化部署技术。

## 学习目标

1. **FastAPI 服务**: 快速构建 REST API 服务
2. **Triton Inference Server**: 高性能推理服务器
3. **负载均衡**: 多实例部署和流量分发

## 目录结构

```
03-serving-systems/
├── README.md
├── 知识点.md              # 理论知识文档
├── src/
│   ├── __init__.py
│   ├── fastapi_server.py  # FastAPI 服务实现
│   ├── triton_client.py   # Triton 客户端
│   └── load_balancer.py   # 负载均衡器
├── notebooks/
│   ├── 01_FastAPI_tutorial.ipynb
│   ├── 02_Triton_tutorial.ipynb
│   └── 03_LoadBalancing_tutorial.ipynb
└── tests/
    ├── test_fastapi_server.py
    ├── test_triton_client.py
    └── test_load_balancer.py
```

## 服务架构对比

| 方案 | 特点 | 适用场景 |
|:-----|:-----|:---------|
| FastAPI | 简单灵活 | 快速原型、小规模 |
| Triton | 高性能、多模型 | 生产环境、大规模 |
| 负载均衡 | 高可用、可扩展 | 分布式部署 |

## 快速开始

```python
# FastAPI 服务
from serving import create_model_server

app = create_model_server(model, "/predict")
# uvicorn main:app --host 0.0.0.0 --port 8000

# Triton 客户端
from serving import TritonClient

client = TritonClient("localhost:8001")
result = client.infer("model_name", inputs)

# 负载均衡
from serving import LoadBalancer

balancer = LoadBalancer(["server1:8000", "server2:8000"])
response = balancer.request("/predict", data)
```

## 依赖安装

```bash
# FastAPI
pip install fastapi uvicorn

# Triton 客户端
pip install tritonclient[all]

# 其他依赖
pip install httpx aiohttp
```

## 参考资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [NGINX 负载均衡](https://docs.nginx.com/nginx/admin-guide/load-balancer/)
