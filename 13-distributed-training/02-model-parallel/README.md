# 02-model-parallel 模型并行

本模块实现了模型并行技术，包括张量并行、流水线并行和序列并行。

## 模块结构

```
02-model-parallel/
├── src/
│   ├── __init__.py           # 模块导出
│   ├── tensor_parallel.py    # 张量并行
│   ├── pipeline_parallel.py  # 流水线并行
│   └── sequence_parallel.py  # 序列并行
├── tests/
│   └── test_model_parallel.py
└── README.md
```

## 核心概念

### 张量并行 (Tensor Parallelism)
- 列并行: 按列分割权重矩阵
- 行并行: 按行分割权重矩阵
- 词表并行: 分割嵌入层

### 流水线并行 (Pipeline Parallelism)
- GPipe: 先前向后反向
- PipeDream (1F1B): 交替执行

### 序列并行 (Sequence Parallelism)
- 按序列维度分片
- 与张量并行配合使用

## 运行测试

```bash
pytest tests/test_model_parallel.py -v
```
