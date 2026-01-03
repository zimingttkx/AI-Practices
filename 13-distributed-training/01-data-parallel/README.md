# 01-data-parallel 数据并行

本模块实现了分布式数据并行训练的核心技术，包括 DDP、FSDP 和 ZeRO。

## 模块结构

```
01-data-parallel/
├── src/
│   ├── __init__.py      # 模块导出
│   ├── ddp.py           # PyTorch DDP 实现
│   ├── fsdp.py          # Fully Sharded Data Parallel
│   └── zero.py          # ZeRO 优化器
├── tests/
│   └── test_data_parallel.py
└── README.md
```

## 核心概念

### DDP (Distributed Data Parallel)
- 每个 GPU 持有完整模型副本
- 数据在进程间分片
- 梯度通过 AllReduce 同步

### FSDP (Fully Sharded Data Parallel)
- 参数、梯度、优化器状态全部分片
- 按需聚合完整参数
- 大幅减少内存占用

### ZeRO (Zero Redundancy Optimizer)
- ZeRO-1: 优化器状态分片
- ZeRO-2: + 梯度分片
- ZeRO-3: + 参数分片

## 使用示例

### DDP 训练

```python
from src.ddp import DDPConfig, DDPTrainer, setup_ddp, cleanup_ddp

# 初始化分布式环境
setup_ddp(rank, world_size)

# 创建训练器
config = DDPConfig(find_unused_parameters=True)
trainer = DDPTrainer(model, config)
ddp_model = trainer.wrap_model()

# 创建分布式数据加载器
dataloader = trainer.create_dataloader(dataset, batch_size=32)

# 训练循环
for batch in dataloader:
    loss = ddp_model(batch)
    loss.backward()
    optimizer.step()

cleanup_ddp()
```

### FSDP 训练

```python
from src.fsdp import FSDPConfig, FSDPTrainer, ShardingStrategy

config = FSDPConfig(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=True,
    activation_checkpointing=True,
)
trainer = FSDPTrainer(model, config)
fsdp_model = trainer.wrap_model()
```

### ZeRO 优化器

```python
from src.zero import ZeROConfig, ZeROOptimizer, ZeROStage

config = ZeROConfig(stage=ZeROStage.GRADIENTS)
optimizer = ZeROOptimizer(base_optimizer, config, model)
```

## 运行测试

```bash
pytest tests/test_data_parallel.py -v
```
