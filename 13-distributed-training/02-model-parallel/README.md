# 02-model-parallel

> **Prerequisites**: Data parallelism basics (DDP), matrix multiplication, Transformer architecture

## Core Concept: What is Model Parallelism?

```
┌─────────────────────────────────────────────────────────────────────────┐
│  数据并行 vs 模型并行                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  数据并行: 模型复制到每个GPU，数据分片                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ GPU0: [完整模型] ← batch_0                                       │   │
│  │ GPU1: [完整模型] ← batch_1                                       │   │
│  │ 问题: 模型太大放不下单卡怎么办?                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  模型并行: 模型切分到多个GPU                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ GPU0: [模型第1部分]  ─→  GPU1: [模型第2部分]  ─→  输出           │   │
│  │ 解决: 单卡放不下的大模型可以跨卡训练                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 三种模型并行方案对比

| 特性 | 张量并行 (TP) | 流水线并行 (PP) | 序列并行 (SP) |
|------|--------------|----------------|---------------|
| **切分维度** | 层内权重矩阵 | 层间（按层切分） | 序列维度 |
| **通信模式** | AllReduce/AllGather | 点对点(P2P) | AllGather/ReduceScatter |
| **通信频率** | 每层2次 | 每个微批次 | 每层2次 |
| **适用场景** | 单层参数大 | 层数多 | 长序列 |
| **气泡开销** | 无 | 有（可优化） | 无 |
| **典型框架** | Megatron-LM | GPipe/PipeDream | Megatron-LM |

## 模块结构

```
02-model-parallel/
├── src/
│   ├── tensor_parallel.py     # 张量并行：列并行/行并行/词表并行
│   ├── pipeline_parallel.py   # 流水线并行：GPipe/1F1B调度
│   └── sequence_parallel.py   # 序列并行：序列维度分片
└── tests/
    └── test_model_parallel.py # 26个测试用例
```

---

## 张量并行 (Tensor Parallelism)

### 核心原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│  张量并行的核心思想: 把单层的权重矩阵切分到多个GPU                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  原始线性层: Y = XW + b                                                  │
│  其中 X: [batch, seq, hidden_in], W: [hidden_in, hidden_out]            │
│                                                                         │
│  列并行 (Column Parallel): 按输出维度切分W                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │         W = [W1 | W2]  (按列切分)                                │   │
│  │                                                                  │   │
│  │  GPU0: Y1 = X @ W1    GPU1: Y2 = X @ W2                         │   │
│  │                                                                  │   │
│  │  最终: Y = [Y1 | Y2]  (拼接结果)                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  行并行 (Row Parallel): 按输入维度切分W                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │         W = [W1]      X = [X1 | X2]  (按列切分输入)              │   │
│  │             [W2]                                                 │   │
│  │                                                                  │   │
│  │  GPU0: Y1 = X1 @ W1   GPU1: Y2 = X2 @ W2                        │   │
│  │                                                                  │   │
│  │  最终: Y = Y1 + Y2    (AllReduce求和)                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Transformer中的张量并行

```
Transformer层的张量并行切分策略 (Megatron-LM方案):

┌─────────────────────────────────────────────────────────────────────────┐
│  MLP层: 列并行 → 行并行                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  X ─→ [列并行Linear] ─→ GeLU ─→ [行并行Linear] ─→ Y              │   │
│  │       (无通信)              (AllReduce)                          │   │
│  │                                                                  │   │
│  │  为什么这样设计?                                                  │   │
│  │  - 列并行后GeLU可以直接在分片上计算 (逐元素操作)                   │   │
│  │  - 行并行后需要AllReduce汇总结果                                  │   │
│  │  - 整个MLP只需要1次AllReduce通信                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Attention层: QKV列并行 → Output行并行                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  X ─→ [QKV列并行] ─→ Attention ─→ [Output行并行] ─→ Y            │   │
│  │       (无通信)                    (AllReduce)                    │   │
│  │                                                                  │   │
│  │  注意力头天然可并行:                                              │   │
│  │  - 8个头，2个GPU → 每个GPU计算4个头                               │   │
│  │  - 头之间独立计算，无需通信                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.tensor_parallel import (
    TensorParallelConfig,
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

# 配置张量并行
config = TensorParallelConfig(
    world_size=4,           # 4个GPU
    rank=0,                 # 当前GPU编号
    parallel_mode="1d",     # 1D张量并行
)

# 创建列并行线性层 (用于MLP第一层、QKV投影)
col_linear = ColumnParallelLinear(
    in_features=4096,
    out_features=16384,
    config=config,
    gather_output=False,  # 不收集输出，直接传给下一层
)

# 创建行并行线性层 (用于MLP第二层、Output投影)
row_linear = RowParallelLinear(
    in_features=16384,
    out_features=4096,
    config=config,
    input_is_parallel=True,  # 输入已经是分片的
)

# 词表并行嵌入 (大词表切分)
embedding = VocabParallelEmbedding(
    num_embeddings=50000,
    embedding_dim=4096,
    config=config,
)
```

---

## 流水线并行 (Pipeline Parallelism)

### 核心原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│  流水线并行的核心思想: 把模型按层切分到多个GPU，像流水线一样处理数据        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  模型切分:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  24层Transformer，4个GPU:                                        │   │
│  │  GPU0: Layer 0-5   (Stage 0)                                    │   │
│  │  GPU1: Layer 6-11  (Stage 1)                                    │   │
│  │  GPU2: Layer 12-17 (Stage 2)                                    │   │
│  │  GPU3: Layer 18-23 (Stage 3)                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  朴素流水线 (问题: 气泡太多):                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  时间 →                                                          │   │
│  │  GPU0: [F0]                    [B0]                              │   │
│  │  GPU1:      [F0]          [B0]                                   │   │
│  │  GPU2:           [F0][B0]                                        │   │
│  │  GPU3:                [F0][B0]                                   │   │
│  │                                                                  │   │
│  │  F=Forward, B=Backward                                          │   │
│  │  空白部分 = 气泡 (GPU空闲等待)                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 微批次调度策略

```
GPipe调度: 先完成所有Forward，再做所有Backward
┌─────────────────────────────────────────────────────────────────────────┐
│  时间 →                                                                  │
│  GPU0: [F0][F1][F2][F3]          [B3][B2][B1][B0]                        │
│  GPU1:    [F0][F1][F2][F3]    [B3][B2][B1][B0]                           │
│  GPU2:       [F0][F1][F2][F3][B3][B2][B1][B0]                            │
│  GPU3:          [F0][F1][F2][F3][B3][B2][B1][B0]                         │
│                                                                         │
│  优点: 实现简单                                                          │
│  缺点: 需要存储所有微批次的激活值，显存占用大                              │
│  气泡率: (p-1) / m，其中p=阶段数，m=微批次数                              │
└─────────────────────────────────────────────────────────────────────────┘

1F1B调度 (One Forward One Backward): 交替执行Forward和Backward
┌─────────────────────────────────────────────────────────────────────────┐
│  时间 →                                                                  │
│  GPU0: [F0][F1][F2][F3][B0][F4][B1][F5][B2]...                           │
│  GPU1:    [F0][F1][F2][B0][F3][B1][F4][B2]...                            │
│  GPU2:       [F0][F1][B0][F2][B1][F3][B2]...                             │
│  GPU3:          [F0][B0][F1][B1][F2][B2]...                              │
│                                                                         │
│  优点: 显存占用恒定 (只需存储p个微批次的激活值)                            │
│  缺点: 实现复杂                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.pipeline_parallel import (
    PipelineConfig,
    PipelineStage,
    GPipeScheduler,
)

# 配置流水线并行
config = PipelineConfig(
    num_stages=4,           # 4个阶段
    num_micro_batches=8,    # 8个微批次
    stage_id=0,             # 当前阶段ID
)

# 创建流水线阶段
stage = PipelineStage(
    module=my_layers,       # 当前阶段的层
    config=config,
)

# 选择调度器
scheduler = GPipeScheduler(config)

# 执行流水线
for micro_batch in scheduler.generate_schedule():
    if micro_batch.is_forward:
        output = stage.forward(micro_batch.input)
        stage.send_forward(output)
    else:
        grad = stage.backward(micro_batch.grad)
        stage.send_backward(grad)
```

---

## 序列并行 (Sequence Parallelism)

### 核心原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│  序列并行的核心思想: 沿序列维度切分，与张量并行配合                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题: 张量并行中，LayerNorm和Dropout在每个GPU上重复计算                  │
│                                                                         │
│  解决: 序列并行 - LayerNorm/Dropout也按序列切分                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GPU0: [LayerNorm(seq/2)] → AllGather → [Attention] → ReduceScatter │
│  │  GPU1: [LayerNorm(seq/2)] → AllGather → [Attention] → ReduceScatter │
│  │        ↑ 各自计算一半序列                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  显存节省: LayerNorm/Dropout的激活值减少N倍                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.sequence_parallel import (
    SequenceParallelConfig,
    SequenceParallelLayerNorm,
    SequenceParallelAttention,
    scatter_to_sequence_parallel,
    gather_from_sequence_parallel,
)

# 配置序列并行
config = SequenceParallelConfig(
    world_size=2,
    rank=0,
    sequence_dim=1,
)

# 序列并行LayerNorm
ln = SequenceParallelLayerNorm(normalized_shape=4096, config=config)

# 序列并行Attention
attn = SequenceParallelAttention(hidden_size=4096, num_heads=32, config=config)

# 使用
x = torch.randn(batch, seq, hidden)
x_sp = scatter_to_sequence_parallel(x, config)  # 分片
x_ln = ln(x_sp)
x_attn = attn(x_ln)
x_full = gather_from_sequence_parallel(x_attn, config)  # 收集
```

---

## 常见问题与最佳实践

### 如何选择并行策略？

```
模型能放入单卡? → 是 → 使用DDP
                → 否 → 单层能放入单卡? → 是 → 流水线并行 (PP)
                                       → 否 → 张量并行 (TP)
                                              → 还不够? → 3D并行
```

### 微批次数量选择

```python
# 气泡率 = (p-1) / m，p=阶段数，m=微批次数
# 建议: num_micro_batches >= 4 * num_stages
num_stages = 4
num_micro_batches = 16  # 气泡率 = 3/16 = 18.75%
```

## 运行测试

```bash
pytest tests/test_model_parallel.py -v
```

## 参考资料

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [GPipe Paper](https://arxiv.org/abs/1811.06965)
- [Sequence Parallelism Paper](https://arxiv.org/abs/2205.05198)
