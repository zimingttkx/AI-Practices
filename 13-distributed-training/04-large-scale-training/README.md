# 04-large-scale-training 大规模训练

本模块实现了大规模训练工具，包括 DeepSpeed 配置、Megatron-Core 集成和分布式检查点。

## 模块结构

```
04-large-scale-training/
├── src/
│   ├── __init__.py          # 模块导出
│   ├── deepspeed_config.py  # DeepSpeed 配置生成
│   ├── megatron_core.py     # Megatron 并行状态
│   └── checkpoint_utils.py  # 分布式检查点
├── tests/
│   └── test_large_scale.py
└── README.md
```

## 核心功能

### DeepSpeed
- ZeRO Stage 1/2/3 配置
- 混合精度 (FP16/BF16)
- CPU Offload

### Megatron-Core
- 张量/流水线/数据并行组管理
- 并行状态单例

### 分布式检查点
- 分片保存/加载
- 自动清理旧检查点

## 运行测试

```bash
pytest tests/test_large_scale.py -v
```
