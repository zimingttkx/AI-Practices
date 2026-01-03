# 04-mlops: 机器学习运维实践

> **前置知识**: Python基础、机器学习训练流程、基本的软件工程概念（版本控制、测试）
>
> **学习目标**: 掌握MLOps三大核心能力——实验追踪、模型注册、生产监控

---

## 一、MLOps 是什么？

**一句话定义**: MLOps = 机器学习 + DevOps，让模型从"能跑"变成"能用"。

```
传统ML开发的痛点:
┌─────────────────────────────────────────────────────────────┐
│  "上周那个效果好的模型用的什么参数？" → 忘了              │
│  "线上模型效果突然变差了？"           → 不知道为什么      │
│  "新模型比旧模型好吗？"               → 没法比较          │
└─────────────────────────────────────────────────────────────┘

MLOps 解决方案:
┌─────────────────────────────────────────────────────────────┐
│  实验追踪 → 记录每次训练的参数、指标、模型文件            │
│  模型注册 → 管理模型版本，控制上线流程                    │
│  生产监控 → 实时监控模型表现，发现问题自动告警            │
└─────────────────────────────────────────────────────────────┘
```

### 核心概念对照表

| 概念 | 类比 | 作用 |
|:-----|:-----|:-----|
| 实验追踪 | 实验室笔记本 | 记录每次"实验"的配置和结果 |
| 模型注册 | 软件版本库 | 管理模型的v1.0、v2.0... |
| 生产监控 | 汽车仪表盘 | 实时显示"引擎"状态 |

---

## 二、模块结构

```
04-mlops/
├── src/                          # 源代码
│   ├── experiment_tracker.py     # 实验追踪器 (记录训练过程)
│   ├── model_registry.py         # 模型注册中心 (版本管理)
│   └── monitoring.py             # 监控系统 (漂移检测+告警)
│
├── notebooks/                    # 教程 (按顺序学习)
│   ├── 01_ExperimentTracking_tutorial.ipynb   # 入门：实验追踪
│   ├── 02_ModelRegistry_tutorial.ipynb        # 进阶：模型注册
│   ├── 03_Monitoring_tutorial.ipynb           # 核心：生产监控
│   ├── 04_DriftDetection_tutorial.ipynb       # 重点：漂移检测
│   ├── 05_ABTesting_tutorial.ipynb            # 实战：A/B测试
│   ├── 06_AutoRetraining_tutorial.ipynb       # 高级：自动重训练
│   └── 07_FeatureStore_tutorial.ipynb         # 扩展：特征存储
│
├── tests/                        # 单元测试 (77个测试用例)
└── 知识点.md                     # 理论知识速查
```

---

## 三、快速开始

### 3.1 环境准备

```bash
# 必需依赖 (本模块核心功能)
pip install numpy scipy

# 可选依赖 (企业级工具集成)
pip install mlflow wandb prometheus-client
```

### 3.2 三分钟上手

```python
import sys
sys.path.insert(0, 'src')  # 添加源码路径

# ========== 1. 实验追踪：记录训练过程 ==========
from experiment_tracker import ExperimentTracker

# 创建追踪器，指定实验名称和保存目录
tracker = ExperimentTracker(
    experiment_name="my_first_experiment",  # 实验名称（用于分组）
    save_dir="./experiments"                # 保存位置
)

# 记录超参数（训练前设置的配置）
tracker.log_params({
    "learning_rate": 0.001,  # 学习率
    "batch_size": 32,        # 批次大小
    "epochs": 100            # 训练轮数
})

# 记录指标（训练过程中产生的数值）
for epoch in range(100):
    loss = 1.0 / (epoch + 1)  # 模拟损失下降
    tracker.log_metrics({"loss": loss, "accuracy": 0.5 + epoch * 0.005}, step=epoch)

tracker.end_run()  # 结束并保存

# ========== 2. 模型注册：版本管理 ==========
from model_registry import ModelRegistry, ModelStage

registry = ModelRegistry(registry_path="./model_registry")

# 注册模型（关联模型文件和性能指标）
version = registry.register_model(
    name="my_classifier",           # 模型名称
    model_path="model.pt",          # 模型文件路径
    version="1.0",                  # 版本号
    metrics={"accuracy": 0.95}      # 性能指标
)

# 推送到生产环境
registry.transition_stage("my_classifier", "1.0", ModelStage.PRODUCTION)

# ========== 3. 生产监控：实时监控 ==========
from monitoring import ModelMonitor

monitor = ModelMonitor(model_name="my_classifier", model_version="1.0")

# 记录每次推理
monitor.record_inference(
    input_data=[1.0, 2.0, 3.0],  # 输入特征
    prediction=1,                # 模型预测
    label=1,                     # 真实标签（如果有）
    latency_ms=15.5              # 推理耗时
)

# 获取统计信息
stats = monitor.get_stats()
print(f"准确率: {stats['accuracy']:.2%}")
print(f"平均延迟: {stats['latency_avg']:.2f}ms")
```

---

## 四、学习路径

```
初学者路径 (建议顺序):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  01_实验追踪 → 02_模型注册 → 03_监控基础 → 04_漂移检测    │
│      ↓            ↓            ↓            ↓              │
│   记录实验     版本管理     指标收集     发现问题          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

进阶路径 (掌握基础后):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  05_A/B测试 → 06_自动重训练 → 07_特征存储                  │
│      ↓            ↓              ↓                         │
│   对比模型     自动更新       特征复用                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、工具对比

| 功能 | 本模块实现 | MLflow | Weights & Biases |
|:-----|:----------|:-------|:-----------------|
| 实验追踪 | ✓ 本地文件 | ✓ 服务器 | ✓ 云端 |
| 模型注册 | ✓ 本地 | ✓ 服务器 | ✓ 云端 |
| 漂移检测 | ✓ KS/PSI | 需插件 | 需插件 |
| 部署难度 | 零配置 | 需服务器 | 需账号 |
| 适用场景 | 学习/小项目 | 团队协作 | 大规模实验 |

**建议**: 先用本模块理解原理，再根据需求选择企业级工具。

---

## 六、常见问题

### Q1: 实验追踪和Git有什么区别？
- **Git**: 管理代码版本
- **实验追踪**: 管理训练过程（参数+指标+模型文件）
- **关系**: 一次Git提交可能对应多次实验（调参）

### Q2: 什么时候需要模型注册？
- 模型要上线时
- 需要回滚到旧版本时
- 多人协作开发模型时

### Q3: 数据漂移是什么？
- **定义**: 线上数据分布与训练数据不同
- **后果**: 模型效果下降
- **检测**: KS检验、PSI指数

---

## 七、参考资源

- [MLflow 官方文档](https://mlflow.org/docs/latest/index.html) - 企业级实验追踪
- [Weights & Biases](https://docs.wandb.ai/) - 云端实验管理
- [Evidently AI](https://evidentlyai.com/) - 数据漂移检测
- [MLOps 社区](https://ml-ops.org/) - 最佳实践指南
