# 01-data-parallel 数据并行

> **前置知识**: PyTorch 基础（nn.Module、DataLoader、优化器）、多进程概念、基本的 GPU 编程概念

## 核心概念：什么是数据并行？

```
┌─────────────────────────────────────────────────────────────────────────┐
│  数据并行的核心思想                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  问题: 单GPU处理大批量数据太慢                                           │
│                                                                         │
│  解决: 把数据分给多个GPU，每个GPU处理一部分，最后汇总结果                  │
│                                                                         │
│  ┌─────────┐    数据分片     ┌─────────┐                               │
│  │ 批次数据 │ ─────────────→ │ GPU 0   │ ──┐                           │
│  │ (1024)  │                │ (256条) │   │                           │
│  └─────────┘                └─────────┘   │                           │
│       │                     ┌─────────┐   │  梯度同步                  │
│       ├───────────────────→ │ GPU 1   │ ──┼─────→ 更新模型             │
│       │                     │ (256条) │   │      (AllReduce)          │
│       │                     └─────────┘   │                           │
│       │                     ┌─────────┐   │                           │
│       ├───────────────────→ │ GPU 2   │ ──┤                           │
│       │                     │ (256条) │   │                           │
│       │                     └─────────┘   │                           │
│       │                     ┌─────────┐   │                           │
│       └───────────────────→ │ GPU 3   │ ──┘                           │
│                             │ (256条) │                               │
│                             └─────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 三种数据并行方案对比

| 特性 | DDP | FSDP | ZeRO |
|------|-----|------|------|
| **每卡存储** | 完整模型 | 模型分片 | 可配置分片 |
| **显存效率** | 1x | N倍节省 | 1x~N倍节省 |
| **通信开销** | 低 | 中~高 | 中~高 |
| **实现复杂度** | 简单 | 中等 | 中等 |
| **适用场景** | 模型能放入单卡 | 模型超出单卡 | 模型超出单卡 |
| **框架** | PyTorch原生 | PyTorch原生 | DeepSpeed |

### 显存占用公式

```
设: P = 参数量, N = GPU数量, Adam优化器(2个状态)

┌─────────────────────────────────────────────────────────────┐
│ DDP 每卡显存:                                                │
│   参数(2P) + 梯度(2P) + 优化器状态(4P) = 8P bytes (FP16)     │
│   → 所有GPU存储相同内容，显存无节省                           │
├─────────────────────────────────────────────────────────────┤
│ ZeRO-1 每卡显存:                                             │
│   参数(2P) + 梯度(2P) + 优化器状态(4P/N) = 4P + 4P/N         │
│   → 只分片优化器状态                                         │
├─────────────────────────────────────────────────────────────┤
│ ZeRO-2 每卡显存:                                             │
│   参数(2P) + 梯度(2P/N) + 优化器状态(4P/N) = 2P + 6P/N       │
│   → 分片优化器状态 + 梯度                                    │
├─────────────────────────────────────────────────────────────┤
│ ZeRO-3 / FSDP 每卡显存:                                      │
│   参数(2P/N) + 梯度(2P/N) + 优化器状态(4P/N) = 8P/N          │
│   → 全部分片，显存节省N倍                                    │
└─────────────────────────────────────────────────────────────┘
```

## 模块结构

```
01-data-parallel/
├── src/
│   ├── ddp.py      # DDP: 最基础的数据并行，每卡完整模型
│   ├── fsdp.py     # FSDP: PyTorch原生全分片数据并行
│   └── zero.py     # ZeRO: DeepSpeed的内存优化方案
└── tests/
    └── test_data_parallel.py  # 28个测试用例
```

---

## DDP (Distributed Data Parallel)

### 核心原理

```
┌────────────────────────────────────────────────────────────────┐
│  DDP 工作流程                                                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. 初始化: 每个GPU启动一个进程，加载完整模型                    │
│                                                                │
│  2. 前向传播: 每个GPU处理自己的数据分片                          │
│     GPU0: forward(batch_0) → loss_0                            │
│     GPU1: forward(batch_1) → loss_1                            │
│     ...                                                        │
│                                                                │
│  3. 反向传播: 每个GPU计算自己的梯度                              │
│     GPU0: backward() → grad_0                                  │
│     GPU1: backward() → grad_1                                  │
│     ...                                                        │
│                                                                │
│  4. 梯度同步 (AllReduce): 所有GPU的梯度求平均                    │
│     avg_grad = (grad_0 + grad_1 + ... + grad_N) / N            │
│     → 每个GPU都得到相同的平均梯度                               │
│                                                                │
│  5. 参数更新: 每个GPU用相同梯度更新，保持模型一致                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### AllReduce 通信原理

```
Ring AllReduce (环形全归约) - DDP默认使用的高效通信算法

步骤1: ReduceScatter (归约分散)
┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
│GPU0 │───→│GPU1 │───→│GPU2 │───→│GPU3 │───→ (环形)
│g0   │    │g1   │    │g2   │    │g3   │
└─────┘    └─────┘    └─────┘    └─────┘
    ↓          ↓          ↓          ↓
每个GPU得到 1/N 的完整归约结果

步骤2: AllGather (全收集)
将归约结果广播给所有GPU

通信量: 2(N-1)/N × 参数量 ≈ 2×参数量 (与GPU数量无关!)
```

### 使用示例

```python
import torch
import torch.distributed as dist
from src.ddp import DDPConfig, DDPTrainer, setup_ddp, cleanup_ddp

def train(rank, world_size):
    # 1. 初始化分布式环境
    setup_ddp(rank, world_size)

    # 2. 创建模型和训练器
    model = MyModel()
    config = DDPConfig(
        find_unused_parameters=False,  # 如果模型有未使用的参数，设为True
        gradient_as_bucket_view=True,  # 内存优化
    )
    trainer = DDPTrainer(model, config)

    # 3. 包装模型
    ddp_model = trainer.wrap_model()

    # 4. 创建分布式数据加载器 (自动分片数据)
    dataloader = trainer.create_dataloader(
        dataset,
        batch_size=32,  # 每个GPU的batch_size，全局batch = 32 × world_size
    )

    # 5. 训练循环
    optimizer = torch.optim.Adam(ddp_model.parameters())
    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)  # 重要! 确保每个epoch数据打乱不同
        for batch in dataloader:
            optimizer.zero_grad()
            loss = ddp_model(batch)
            loss.backward()  # 梯度自动同步
            optimizer.step()

    # 6. 清理
    cleanup_ddp()

# 启动多进程训练
# torchrun --nproc_per_node=4 train.py
```

---

## FSDP (Fully Sharded Data Parallel)

### 核心原理

```
┌────────────────────────────────────────────────────────────────┐
│  FSDP vs DDP 的关键区别                                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  DDP: 每个GPU存储完整模型                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ GPU0: [全部参数] [全部梯度] [全部优化器状态]              │   │
│  │ GPU1: [全部参数] [全部梯度] [全部优化器状态]  ← 重复存储! │   │
│  │ GPU2: [全部参数] [全部梯度] [全部优化器状态]              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  FSDP: 每个GPU只存储模型的一部分                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ GPU0: [参数分片0] [梯度分片0] [优化器分片0]               │   │
│  │ GPU1: [参数分片1] [梯度分片1] [优化器分片1]  ← 无重复!    │   │
│  │ GPU2: [参数分片2] [梯度分片2] [优化器分片2]               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### FSDP 执行流程

```
前向传播时:
┌─────────────────────────────────────────────────────────────┐
│ Layer 1 计算:                                                │
│   1. AllGather: 收集所有GPU的Layer1参数 → 得到完整Layer1     │
│   2. Forward: 用完整参数计算                                 │
│   3. 释放: 计算完成后释放非本地参数 (节省显存)                │
│                                                              │
│ Layer 2 计算:                                                │
│   1. AllGather: 收集Layer2参数                               │
│   2. Forward: 计算                                           │
│   3. 释放                                                    │
│   ...                                                        │
└─────────────────────────────────────────────────────────────┘

反向传播时:
┌─────────────────────────────────────────────────────────────┐
│ Layer N 梯度计算:                                            │
│   1. AllGather: 重新收集Layer N参数                          │
│   2. Backward: 计算梯度                                      │
│   3. ReduceScatter: 梯度归约并分片存储                       │
│   4. 释放非本地参数                                          │
│   ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

### 分片策略

```python
from src.fsdp import FSDPConfig, FSDPTrainer, ShardingStrategy

# 策略1: FULL_SHARD (完全分片) - 最省显存
# 等价于 ZeRO-3，参数/梯度/优化器全部分片
config = FSDPConfig(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)

# 策略2: SHARD_GRAD_OP (梯度+优化器分片) - 平衡方案
# 等价于 ZeRO-2，参数不分片，减少通信
config = FSDPConfig(
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
)

# 策略3: HYBRID_SHARD (混合分片) - 多机场景
# 节点内全分片，节点间不分片，减少跨机通信
config = FSDPConfig(
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
)
```

### 使用示例

```python
from src.fsdp import FSDPConfig, FSDPTrainer, ShardingStrategy

# 配置FSDP
config = FSDPConfig(
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
    mixed_precision=True,                            # 启用混合精度
    activation_checkpointing=True,                   # 激活检查点(进一步省显存)
    cpu_offload=False,                               # CPU卸载(显存极度紧张时启用)
)

# 创建训练器并包装模型
trainer = FSDPTrainer(model, config)
fsdp_model = trainer.wrap_model()

# 训练循环与DDP类似
optimizer = torch.optim.AdamW(fsdp_model.parameters())
for batch in dataloader:
    optimizer.zero_grad()
    with trainer.autocast():  # 混合精度上下文
        loss = fsdp_model(batch)
    loss.backward()
    optimizer.step()

# 保存检查点 (FSDP需要特殊处理)
trainer.save_checkpoint("checkpoint.pt", optimizer, full_state_dict=True)
```

---

## ZeRO (Zero Redundancy Optimizer)

### 三个阶段详解

```
┌─────────────────────────────────────────────────────────────────┐
│  ZeRO 渐进式优化                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ZeRO-1: 优化器状态分片                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 每个GPU: [完整参数] [完整梯度] [1/N优化器状态]            │    │
│  │ 显存节省: ~4x (Adam有2个状态，FP32存储)                   │    │
│  │ 通信增加: 优化器step时需要AllGather                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ZeRO-2: + 梯度分片                                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 每个GPU: [完整参数] [1/N梯度] [1/N优化器状态]             │    │
│  │ 显存节省: ~8x                                             │    │
│  │ 通信: 用ReduceScatter替代AllReduce                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ZeRO-3: + 参数分片 (等价于FSDP)                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 每个GPU: [1/N参数] [1/N梯度] [1/N优化器状态]              │    │
│  │ 显存节省: ~N倍 (线性扩展)                                 │    │
│  │ 通信: 每层forward/backward都需要AllGather参数             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.zero import ZeROConfig, ZeROOptimizer, ZeROStage

# 创建基础优化器
base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 包装为ZeRO优化器
config = ZeROConfig(
    stage=ZeROStage.GRADIENTS,  # ZeRO-2
    overlap_comm=True,           # 通信计算重叠
    contiguous_gradients=True,   # 连续梯度缓冲区
    cpu_offload=False,           # CPU卸载
)
optimizer = ZeROOptimizer(base_optimizer, config, model)

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()  # ZeRO自动处理梯度同步和参数更新
```

---

## 常见问题与最佳实践

### 1. 如何选择数据并行方案？

```
模型能放入单卡显存?
    ├── 是 → 使用 DDP (最简单，通信最少)
    └── 否 → 超出多少?
              ├── 略超 → ZeRO-2 / FSDP(SHARD_GRAD_OP)
              └── 远超 → ZeRO-3 / FSDP(FULL_SHARD)
                         └── 还不够? → + CPU Offload
```

### 2. 关键参数调优

```python
# DDP 调优
DDPConfig(
    bucket_cap_mb=25,              # 梯度桶大小，影响通信效率
    find_unused_parameters=False,  # 有未使用参数时设True，但会降低性能
    static_graph=True,             # 模型结构固定时启用，提升性能
)

# FSDP 调优
FSDPConfig(
    backward_prefetch="backward_pre",  # 预取策略，减少等待
    forward_prefetch=True,              # 前向预取
    limit_all_gathers=True,             # 限制并发AllGather，防OOM
)
```

### 3. 常见错误

```python
# 错误1: 忘记设置epoch导致数据重复
for epoch in range(num_epochs):
    # dataloader.sampler.set_epoch(epoch)  # 忘记这行!
    for batch in dataloader:
        ...

# 错误2: 在非rank0进程打印/保存
if dist.get_rank() == 0:  # 只在主进程执行
    print(f"Loss: {loss}")
    torch.save(model.state_dict(), "model.pt")

# 错误3: FSDP保存检查点方式错误
# 错误: torch.save(fsdp_model.state_dict(), "model.pt")
# 正确: 使用FSDP的state_dict上下文管理器
with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
    state_dict = fsdp_model.state_dict()
    if rank == 0:
        torch.save(state_dict, "model.pt")
```

## 运行测试

```bash
# 运行所有数据并行测试
pytest tests/test_data_parallel.py -v

# 运行特定测试
pytest tests/test_data_parallel.py::test_ddp_trainer -v
```

## 参考资料

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
