# 04-large-scale-training 大规模训练

> **前置知识**: 数据并行、模型并行、混合精度训练

## 核心概念：什么是大规模训练？

```
┌─────────────────────────────────────────────────────────────────────────┐
│  大规模训练的挑战                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  模型规模增长:                                                           │
│  - GPT-3: 175B参数 → 需要350GB显存 (FP16)                               │
│  - GPT-4: 估计1.8T参数 → 需要3.6TB显存                                  │
│  - 单卡最大: 80GB (A100/H100)                                           │
│                                                                         │
│  解决方案: 3D并行 + 内存优化 + 高效检查点                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  数据并行 (DP): 复制模型，分割数据                                │   │
│  │  张量并行 (TP): 切分单层权重                                      │   │
│  │  流水线并行 (PP): 切分模型层                                      │   │
│  │  ZeRO优化: 分片优化器状态/梯度/参数                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 主流框架对比

| 特性 | DeepSpeed | Megatron-LM | PyTorch FSDP |
|------|-----------|-------------|--------------|
| **开发者** | Microsoft | NVIDIA | Meta |
| **核心优势** | ZeRO内存优化 | 3D并行优化 | 原生集成 |
| **ZeRO支持** | Stage 1/2/3 | 部分支持 | 类似ZeRO-3 |
| **张量并行** | 需配合Megatron | 原生支持 | 不支持 |
| **流水线并行** | 支持 | 原生支持 | 不支持 |
| **CPU Offload** | 支持 | 不支持 | 支持 |
| **易用性** | 配置文件驱动 | 需要代码修改 | API简单 |
| **典型场景** | 通用大模型 | 超大Transformer | 中等规模 |

## 模块结构

```
04-large-scale-training/
├── src/
│   ├── deepspeed_config.py  # DeepSpeed配置生成
│   ├── megatron_core.py     # Megatron并行状态管理
│   └── checkpoint_utils.py  # 分布式检查点
└── tests/
    └── test_large_scale.py
```

---

## DeepSpeed 配置

### ZeRO优化阶段

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ZeRO (Zero Redundancy Optimizer) 内存优化                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  传统数据并行的问题:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  每个GPU都存储: 模型参数 + 梯度 + 优化器状态                      │   │
│  │  Adam优化器: 参数(4B) + 梯度(4B) + 动量(4B) + 方差(4B) = 16B/参数 │   │
│  │  1B参数模型 → 每GPU需要16GB (N个GPU总共N×16GB，大量冗余!)        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ZeRO的解决方案 (分片消除冗余):                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Stage 1: 分片优化器状态                                         │   │
│  │  - 每GPU: 参数(4B) + 梯度(4B) + 优化器状态(8B/N)                 │   │
│  │  - 内存: 8 + 8/N GB/B参数                                        │   │
│  │                                                                  │   │
│  │  Stage 2: 分片优化器状态 + 梯度                                   │   │
│  │  - 每GPU: 参数(4B) + 梯度(4B/N) + 优化器状态(8B/N)               │   │
│  │  - 内存: 4 + 12/N GB/B参数                                       │   │
│  │                                                                  │   │
│  │  Stage 3: 分片优化器状态 + 梯度 + 参数                            │   │
│  │  - 每GPU: 参数(4B/N) + 梯度(4B/N) + 优化器状态(8B/N)             │   │
│  │  - 内存: 16/N GB/B参数 (线性扩展!)                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.deepspeed_config import DeepSpeedConfig, create_deepspeed_config

# 配置DeepSpeed
config = DeepSpeedConfig(
    train_batch_size=256,
    train_micro_batch_size_per_gpu=4,
    gradient_accumulation_steps=8,
    zero_stage=2,                    # ZeRO Stage 2
    fp16_enabled=True,               # 混合精度
    offload_optimizer=False,         # CPU卸载 (Stage 3可用)
    learning_rate=1e-4,
    warmup_steps=1000,
)

# 生成配置字典
ds_config = create_deepspeed_config(config)

# 保存为JSON
from src.deepspeed_config import save_deepspeed_config
save_deepspeed_config(ds_config, "ds_config.json")

# 使用DeepSpeed初始化
import deepspeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)
```

---

## Megatron 并行状态

### 3D并行拓扑

```
┌─────────────────────────────────────────────────────────────────────────┐
│  3D并行: 数据并行 × 张量并行 × 流水线并行                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  示例: 16个GPU，TP=2, PP=4, DP=2                                        │
│                                                                         │
│  GPU编号和坐标 (d, t, p):                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  DP组0:                                                          │   │
│  │  GPU0(0,0,0) ─→ GPU2(0,0,1) ─→ GPU4(0,0,2) ─→ GPU6(0,0,3)       │   │
│  │  GPU1(0,1,0) ─→ GPU3(0,1,1) ─→ GPU5(0,1,2) ─→ GPU7(0,1,3)       │   │
│  │       ↑ TP组                                                     │   │
│  │                                                                  │   │
│  │  DP组1:                                                          │   │
│  │  GPU8(1,0,0) ─→ GPU10(1,0,1) ─→ GPU12(1,0,2) ─→ GPU14(1,0,3)    │   │
│  │  GPU9(1,1,0) ─→ GPU11(1,1,1) ─→ GPU13(1,1,2) ─→ GPU15(1,1,3)    │   │
│  │                      ↑ PP组 (流水线方向)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  通信组:                                                                 │
│  - TP组: 同一PP阶段、同一DP组内的GPU (如 GPU0, GPU1)                    │
│  - PP组: 同一TP位置、同一DP组内的GPU (如 GPU0, GPU2, GPU4, GPU6)        │
│  - DP组: 同一TP位置、同一PP阶段的GPU (如 GPU0, GPU8)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.megatron_core import MegatronConfig, initialize_megatron

# 配置3D并行
config = MegatronConfig(
    tensor_model_parallel_size=2,    # 张量并行度
    pipeline_model_parallel_size=4,  # 流水线并行度
    data_parallel_size=2,            # 数据并行度 (自动计算)
    sequence_parallel=True,          # 序列并行
)

# 初始化并行状态
state = initialize_megatron(config)

# 获取进程组
tp_group = state.tensor_model_parallel_group
pp_group = state.pipeline_model_parallel_group
dp_group = state.data_parallel_group

# 检查当前位置
if state.is_pipeline_first_stage():
    print("这是流水线第一阶段")
if state.is_pipeline_last_stage():
    print("这是流水线最后阶段")

# 获取各维度的rank
print(f"TP rank: {state.tensor_model_parallel_rank}")
print(f"PP rank: {state.pipeline_model_parallel_rank}")
print(f"DP rank: {state.data_parallel_rank}")
```

---

## 分布式检查点

### 为什么需要分布式检查点？

```
┌─────────────────────────────────────────────────────────────────────────┐
│  大模型检查点的挑战                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  传统检查点:                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - 单文件保存: 175B参数 → 350GB文件                              │   │
│  │  - 串行I/O: 保存需要几十分钟                                      │   │
│  │  - 内存峰值: 需要额外显存存储完整状态                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  分布式检查点:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - 分片保存: 每个rank保存自己的部分                               │   │
│  │  - 并行I/O: N个rank同时写入，速度提升N倍                         │   │
│  │  - 内存高效: 无需额外显存                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  检查点目录结构:                                                         │
│  checkpoint-1000/                                                       │
│  ├── metadata.json      # 元数据 (step, epoch, world_size)             │
│  ├── rank_0.pt          # Rank 0的状态                                  │
│  ├── rank_1.pt          # Rank 1的状态                                  │
│  └── ...                                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.checkpoint_utils import CheckpointConfig, DistributedCheckpointer

# 配置检查点
config = CheckpointConfig(
    save_dir="./checkpoints",
    save_interval=1000,      # 每1000步保存
    keep_last_n=3,           # 保留最近3个检查点
    use_distributed=True,    # 分布式保存
)

checkpointer = DistributedCheckpointer(config)

# 保存检查点
checkpoint_path = checkpointer.save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    step=1000,
    epoch=1,
    # 可以添加自定义数据
    loss=current_loss,
    metrics=metrics_dict,
)

# 加载检查点
checkpoint = checkpointer.load(
    checkpoint_path,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
)
print(f"从step {checkpoint['step']}恢复训练")

# 获取最新检查点
latest = checkpointer.get_latest_checkpoint()
if latest:
    checkpointer.load(latest, model, optimizer)
```

---

## 常见问题与最佳实践

### 如何选择并行策略？

```
模型参数量?
├── < 10B → 数据并行 + ZeRO-2
├── 10B-100B → 数据并行 + ZeRO-3 或 张量并行
└── > 100B → 3D并行 (DP + TP + PP)

单层参数量?
├── 能放入单卡 → 流水线并行
└── 不能放入单卡 → 张量并行

显存不足?
├── 先尝试ZeRO-2
├── 再尝试ZeRO-3
├── 还不够 → CPU Offload
└── 最后 → 减小batch size或模型
```

### ZeRO Stage选择

```python
# ZeRO-1: 优化器状态分片
# - 通信量与DDP相同
# - 内存节省有限
# - 适合: 优化器状态占用大的场景

# ZeRO-2: 优化器状态 + 梯度分片 (推荐)
# - 通信量与DDP相同
# - 内存节省显著
# - 适合: 大多数场景

# ZeRO-3: 全分片
# - 通信量增加 (需要AllGather参数)
# - 内存节省最大
# - 适合: 超大模型、显存极度紧张
```

## 运行测试

```bash
pytest tests/test_large_scale.py -v
```

## 参考资料

- [ZeRO Paper (SC 2020)](https://arxiv.org/abs/1910.02054)
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PyTorch Distributed Checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html)
