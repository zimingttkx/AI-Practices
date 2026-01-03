# 模型优化 (Model Optimization)

> **前置知识**: PyTorch 基础、深度学习模型训练流程
>
> **学习目标**: 掌握量化、剪枝、蒸馏三大模型压缩技术，实现高效模型部署

---

## 为什么需要模型优化？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        模型部署的现实挑战                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  训练环境                              部署环境                          │
│  ┌─────────────────┐                  ┌─────────────────┐               │
│  │ GPU 集群        │                  │ 边缘设备/手机    │               │
│  │ 数百 GB 内存    │      VS          │ 数 GB 内存      │               │
│  │ 秒级延迟可接受  │                  │ 毫秒级延迟要求  │               │
│  │ 不限能耗        │                  │ 电池供电        │               │
│  └─────────────────┘                  └─────────────────┘               │
│                                                                         │
│  核心矛盾: 模型越大精度越高，但部署成本越高                              │
│                                                                         │
│  解决方案: 模型优化技术                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  量化: FP32 → INT8/INT4，减少 4-8x 内存，加速 2-4x              │   │
│  │  剪枝: 移除冗余参数，减少 2-10x 计算量                          │   │
│  │  蒸馏: 大模型知识 → 小模型，保持精度的同时大幅压缩              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 优化技术全景

```
                         模型优化技术
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
      量化                  剪枝                 蒸馏
  (Quantization)        (Pruning)         (Distillation)
         │                    │                    │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    │         │         │         │         │         │
   PTQ      QAT      结构化    非结构化    响应      特征
 (训练后)  (感知)     剪枝       剪枝      蒸馏      蒸馏
    │         │         │         │         │         │
 无需训练  需要训练  直接加速  需要稀疏  软标签    中间层
 精度中等  精度最高  硬件友好  计算支持  KL散度    MSE损失
```

### 优化效果对比

| 技术 | 模型大小 | 推理速度 | 精度损失 | 实现难度 | 适用场景 |
|:-----|:--------:|:--------:|:--------:|:--------:|:---------|
| FP16 量化 | 2x↓ | 1.5-2x↑ | <0.1% | ⭐ | 通用 GPU 部署 |
| INT8 PTQ | 4x↓ | 2-4x↑ | <1% | ⭐⭐ | 服务器快速部署 |
| INT8 QAT | 4x↓ | 2-4x↑ | <0.5% | ⭐⭐⭐ | 精度敏感场景 |
| INT4 量化 | 8x↓ | 3-6x↑ | 1-3% | ⭐⭐⭐⭐ | LLM 推理 |
| 结构化剪枝 | 2-4x↓ | 2-4x↑ | <1% | ⭐⭐ | CNN 模型 |
| 非结构化剪枝 | 5-10x↓ | 依赖硬件 | <1% | ⭐⭐ | 稀疏计算硬件 |
| 知识蒸馏 | 可变 | 可变 | <1% | ⭐⭐⭐ | 模型架构压缩 |

---

## 核心技术详解

### 1. 量化 (Quantization)

**核心思想**: 用低精度数值表示高精度参数

```
数值精度对比:
┌─────────────────────────────────────────────────────────────┐
│  FP32 (32位浮点)                                            │
│  ├── 内存: 4 字节/参数                                      │
│  ├── 范围: ±3.4×10³⁸                                        │
│  └── 精度: 最高                                             │
│                                                             │
│  FP16 (16位浮点)                                            │
│  ├── 内存: 2 字节/参数 (2x 压缩)                            │
│  ├── 范围: ±65504                                           │
│  └── 精度: 高                                               │
│                                                             │
│  INT8 (8位整数)                                             │
│  ├── 内存: 1 字节/参数 (4x 压缩)                            │
│  ├── 范围: [-128, 127]                                      │
│  └── 精度: 中                                               │
│                                                             │
│  INT4 (4位整数)                                             │
│  ├── 内存: 0.5 字节/参数 (8x 压缩)                          │
│  ├── 范围: [-8, 7]                                          │
│  └── 精度: 低                                               │
└─────────────────────────────────────────────────────────────┘
```

**量化公式**:
```
量化:   q = round(r / scale) + zero_point
反量化: r = (q - zero_point) × scale

其中:
- r: 原始浮点值
- q: 量化后整数值
- scale: 缩放因子 = (r_max - r_min) / (q_max - q_min)
- zero_point: 零点偏移
```

**量化方法选择**:

```python
# 方法1: 动态量化 - 最简单，无需校准数据
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 方法2: 静态量化 - 需要校准数据，精度更高
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model)
# 运行校准数据...
model_int8 = torch.quantization.convert(model_prepared)

# 方法3: QAT - 训练中模拟量化，精度最高
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_qat = torch.quantization.prepare_qat(model)
# 继续训练...
model_int8 = torch.quantization.convert(model_qat)
```

### 2. 剪枝 (Pruning)

**核心思想**: 移除模型中不重要的参数

```
剪枝类型对比:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  非结构化剪枝 (Unstructured)                                │
│  ┌─────────┐      ┌─────────┐                               │
│  │●●●●●●●●●│      │●○●○●●○●│                               │
│  │●●●●●●●●●│  →   │●○●○●●○●│  移除单个权重                 │
│  │●●●●●●●●●│      │●○●○●●○●│  产生稀疏矩阵                 │
│  └─────────┘      └─────────┘  需要特殊硬件支持             │
│                                                             │
│  结构化剪枝 (Structured)                                    │
│  ┌─────────┐      ┌───────┐                                 │
│  │●●●●●●●●●│      │●●●●●●●│                                 │
│  │●●●●●●●●●│  →   │●●●●●●●│  移除整行/列/通道              │
│  │●●●●●●●●●│      │●●●●●●●│  产生规则小模型                │
│  └─────────┘      └───────┘  无需特殊硬件                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**重要性评估方法**:

| 方法 | 公式 | 优点 | 缺点 |
|:-----|:-----|:-----|:-----|
| 幅度剪枝 | `importance = |w|` | 简单高效 | 可能不准确 |
| 梯度剪枝 | `importance = |w × ∇w|` | 更准确 | 需要额外计算 |
| Taylor 展开 | `importance = |g × w|` | 理论基础强 | 计算开销大 |

```python
import torch.nn.utils.prune as prune

# 非结构化剪枝: 移除 30% 最小权重
prune.l1_unstructured(module, name='weight', amount=0.3)

# 结构化剪枝: 移除 30% 最不重要的通道
prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
```

### 3. 知识蒸馏 (Knowledge Distillation)

**核心思想**: 将大模型的知识迁移到小模型

```
知识蒸馏框架:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  教师模型 (Teacher)                                         │
│  ┌─────────────────┐                                        │
│  │  大型预训练模型  │ ──→ 软标签 (Soft Labels)              │
│  │  参数: 数十亿    │     P_T = softmax(z_T / T)            │
│  └─────────────────┘                                        │
│           │                                                 │
│           │ 知识迁移 (KL 散度)                              │
│           ↓                                                 │
│  学生模型 (Student)                                         │
│  ┌─────────────────┐                                        │
│  │  轻量级模型      │ ──→ 学习软标签分布                    │
│  │  参数: 数百万    │                                       │
│  └─────────────────┘                                        │
│                                                             │
│  损失函数:                                                  │
│  L = α × L_soft + (1-α) × L_hard                           │
│                                                             │
│  其中:                                                      │
│  - L_soft: KL(student_soft || teacher_soft) × T²           │
│  - L_hard: CrossEntropy(student, labels)                   │
│  - T: 温度参数 (通常 4-20)                                  │
│  - α: 软标签权重 (通常 0.5-0.9)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """知识蒸馏损失函数"""
    # 软标签损失 (KL 散度)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)  # 温度缩放

    # 硬标签损失 (交叉熵)
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

---

## 模型导出

### ONNX 导出

```python
import torch.onnx

# 导出为 ONNX 格式 (跨框架通用)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}},  # 支持动态 batch
    opset_version=14
)
```

### TorchScript 导出

```python
# 方法1: 脚本化 - 支持动态控制流
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 方法2: 追踪 - 更简单，但不支持动态控制流
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")
```

---

## 文件结构

```
01-model-optimization/
├── src/                              # 源代码
│   ├── __init__.py
│   ├── quantization.py               # 量化实现
│   ├── pruning.py                    # 剪枝实现
│   ├── distillation.py               # 知识蒸馏
│   └── export.py                     # 模型导出
├── tests/                            # 单元测试
│   ├── test_quantization.py
│   ├── test_pruning.py
│   ├── test_distillation.py
│   └── test_export.py
├── notebooks/                        # 教程笔记本
│   ├── 01_Quantization_tutorial.ipynb    # 量化教程
│   ├── 02_Pruning_tutorial.ipynb         # 剪枝教程
│   ├── 03_Distillation_tutorial.ipynb    # 蒸馏教程
│   ├── 04_Export_tutorial.ipynb          # 导出教程
│   └── 05_Advanced_Optimization_tutorial.ipynb  # 高级优化
├── 知识点.md                         # 详细知识点
└── README.md                         # 本文件
```

---

## 快速开始

### 安装依赖

```bash
pip install torch onnx onnxruntime
```

### 运行测试

```bash
pytest tests/ -v
```

### 使用示例

```python
from model_optimization import (
    DynamicQuantizer,
    StructuredPruner,
    KnowledgeDistiller,
    ONNXExporter
)

# 1. 量化 - 最快的优化方式
quantizer = DynamicQuantizer()
quantized_model = quantizer.quantize(model)

# 2. 剪枝 - 减少计算量
pruner = StructuredPruner(sparsity=0.3)
pruned_model = pruner.prune(model)

# 3. 蒸馏 - 训练小模型
distiller = KnowledgeDistiller(teacher_model, student_model)
distilled_model = distiller.train(train_loader)

# 4. 导出 - 部署到生产环境
exporter = ONNXExporter()
exporter.export(model, "model.onnx")
```

---

## 优化策略选择指南

```
┌─────────────────────────────────────────────────────────────┐
│                    如何选择优化技术？                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  场景 1: 快速部署，精度要求高                               │
│  └── 推荐: FP16 量化                                        │
│      原因: 几乎无精度损失，实现简单                         │
│                                                             │
│  场景 2: 服务器部署，追求吞吐量                             │
│  └── 推荐: INT8 静态量化 + TensorRT                         │
│      原因: 4x 压缩，2-4x 加速                               │
│                                                             │
│  场景 3: 边缘设备，资源受限                                 │
│  └── 推荐: INT8 QAT + 结构化剪枝                            │
│      原因: 最大化压缩，保持精度                             │
│                                                             │
│  场景 4: LLM 推理，显存受限                                 │
│  └── 推荐: GPTQ/AWQ INT4 量化                               │
│      原因: 8x 压缩，专为 LLM 设计                           │
│                                                             │
│  场景 5: 需要更小的模型架构                                 │
│  └── 推荐: 知识蒸馏                                         │
│      原因: 可以训练任意小模型                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 学习路径

1. **01_Quantization_tutorial.ipynb** - 量化基础与实践
2. **02_Pruning_tutorial.ipynb** - 剪枝技术详解
3. **03_Distillation_tutorial.ipynb** - 知识蒸馏实战
4. **04_Export_tutorial.ipynb** - 模型导出与部署
5. **05_Advanced_Optimization_tutorial.ipynb** - 高级优化技术
