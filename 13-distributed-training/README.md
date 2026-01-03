# 13-distributed-training 分布式训练

本模块实现了分布式训练的核心技术。

## 模块结构

```
13-distributed-training/
├── 01-data-parallel/        # DDP、FSDP、ZeRO (28 tests)
├── 02-model-parallel/       # 张量/流水线/序列并行 (26 tests)
├── 03-mixed-precision/      # AMP、BF16、梯度缩放 (31 tests)
└── 04-large-scale-training/ # DeepSpeed、Megatron (20 tests)
```

## 运行测试

```bash
pytest 13-distributed-training/ -v
```

## 总测试数: 105
