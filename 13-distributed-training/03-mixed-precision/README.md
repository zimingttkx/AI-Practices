# 03-mixed-precision 混合精度训练

本模块实现了混合精度训练技术，包括 AMP、BF16 和梯度缩放。

## 模块结构

```
03-mixed-precision/
├── src/
│   ├── __init__.py          # 模块导出
│   ├── amp.py               # 自动混合精度
│   ├── bf16_training.py     # BF16 训练
│   └── gradient_scaling.py  # 梯度缩放
├── tests/
│   └── test_mixed_precision.py
└── README.md
```

## 核心概念

### AMP (Automatic Mixed Precision)
- 自动选择 FP16/FP32 精度
- 使用 GradScaler 防止梯度下溢
- 加速训练，减少内存

### BF16 (Brain Floating Point 16)
- 8位指数 + 7位尾数
- 动态范围与 FP32 相同
- 无需梯度缩放

### Gradient Scaling
- 损失缩放防止下溢
- 动态调整缩放因子
- 溢出检测和恢复

## 使用示例

### AMP 训练

```python
from src.amp import AMPConfig, AMPTrainer

config = AMPConfig(dtype=torch.float16)
trainer = AMPTrainer(model, config)

for batch in dataloader:
    with trainer.autocast():
        loss = model(batch)
    trainer.backward(loss)
    trainer.step(optimizer)
```

### BF16 训练

```python
from src.bf16_training import BF16Config, BF16Trainer

config = BF16Config(keep_batchnorm_fp32=True)
trainer = BF16Trainer(model, config)

with trainer.autocast():
    output = trainer.forward(input)
```

## 运行测试

```bash
pytest tests/test_mixed_precision.py -v
```
