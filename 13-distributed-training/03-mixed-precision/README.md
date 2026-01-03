# 03-mixed-precision

> **Prerequisites**: Floating point representation, deep learning training, gradient descent

## Core Concept: What is Mixed Precision Training?

```
┌─────────────────────────────────────────────────────────────────────────┐
│  为什么需要混合精度？                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FP32训练的问题:                                                         │
│  - 显存占用大: 每个参数4字节                                              │
│  - 计算速度慢: Tensor Core无法加速                                       │
│                                                                         │
│  混合精度的解决方案:                                                      │
│  - 前向/反向: 使用FP16/BF16 (2字节)                                      │
│  - 权重更新: 保持FP32精度                                                │
│  - 效果: 显存减半，速度翻倍，精度不损失                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 三种数据类型对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│  浮点数格式对比                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FP32 (32位):  1位符号 + 8位指数 + 23位尾数                              │
│  ├── 范围: ±3.4×10³⁸                                                    │
│  ├── 精度: 约7位有效数字                                                 │
│  └── 用途: 权重更新、梯度累积                                            │
│                                                                         │
│  FP16 (16位):  1位符号 + 5位指数 + 10位尾数                              │
│  ├── 范围: ±65504 (很小!)                                               │
│  ├── 精度: 约3位有效数字                                                 │
│  ├── 问题: 容易溢出/下溢                                                 │
│  └── 需要: 梯度缩放 (Loss Scaling)                                       │
│                                                                         │
│  BF16 (16位):  1位符号 + 8位指数 + 7位尾数                               │
│  ├── 范围: ±3.4×10³⁸ (与FP32相同!)                                      │
│  ├── 精度: 约2位有效数字                                                 │
│  ├── 优点: 无需梯度缩放                                                  │
│  └── 要求: Ampere及以上GPU (A100, RTX 30系列+)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 技术方案对比

| 特性 | FP16 + AMP | BF16 | 纯FP32 |
|------|-----------|------|--------|
| **显存占用** | ~50% | ~50% | 100% |
| **计算速度** | ~2x | ~2x | 1x |
| **需要梯度缩放** | ✅ 是 | ❌ 否 | ❌ 否 |
| **数值稳定性** | 需要小心 | 较好 | 最好 |
| **GPU要求** | Volta+ | Ampere+ | 任意 |
| **典型场景** | 通用训练 | 大模型训练 | 调试/基线 |

## 模块结构

```
03-mixed-precision/
├── src/
│   ├── amp.py               # 自动混合精度 (FP16)
│   ├── bf16_training.py     # BF16训练
│   └── gradient_scaling.py  # 梯度缩放
└── tests/
    └── test_mixed_precision.py
```

---

## AMP (Automatic Mixed Precision)

### 核心原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│  AMP工作流程                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. 前向传播 (autocast):                                                 │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │  with autocast():                                            │    │
│     │      output = model(input)  # 自动选择FP16/FP32              │    │
│     │      loss = criterion(output, target)                        │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  2. 损失缩放 (防止梯度下溢):                                             │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │  scaled_loss = loss * scale_factor  # 放大损失               │    │
│     │  scaled_loss.backward()             # 梯度也被放大           │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  3. 梯度还原 + 优化器更新:                                               │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │  scaler.unscale_(optimizer)  # 梯度除以scale_factor          │    │
│     │  scaler.step(optimizer)      # 检查溢出，更新权重            │    │
│     │  scaler.update()             # 动态调整scale_factor          │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 为什么需要损失缩放？

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FP16的数值范围问题                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FP16可表示的最小正数: 6×10⁻⁵                                            │
│  很多梯度值: 10⁻⁶ ~ 10⁻⁸ (比最小值还小!)                                 │
│                                                                         │
│  问题: 梯度下溢 (underflow)                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  grad = 1e-7  →  FP16表示  →  0.0  (信息丢失!)                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  解决: 损失缩放 (Loss Scaling)                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  loss × 65536 → backward → grad × 65536 → grad / 65536          │   │
│  │  1e-7 × 65536 = 6.5e-3  (可以表示!)                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.amp import AMPConfig, AMPTrainer

# 配置AMP
config = AMPConfig(
    enabled=True,
    dtype=torch.float16,      # 使用FP16
    use_grad_scaler=True,     # 启用梯度缩放
    init_scale=65536.0,       # 初始缩放因子
)

# 创建训练器
trainer = AMPTrainer(model, config)

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()

    # 前向传播 (自动混合精度)
    with trainer.autocast():
        output = model(batch)
        loss = criterion(output, target)

    # 反向传播 (带梯度缩放)
    trainer.backward(loss)

    # 梯度裁剪 (可选)
    trainer.unscale_gradients(optimizer)
    trainer.clip_gradients(max_norm=1.0)

    # 优化器更新
    trainer.step(optimizer)
```

---

## BF16 训练

### 核心原理

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BF16 vs FP16                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FP16:  □ □ □ □ □ | □ □ □ □ □ □ □ □ □ □                                │
│         ─────────   ─────────────────────                               │
│          5位指数        10位尾数                                         │
│          范围小         精度高                                           │
│                                                                         │
│  BF16:  □ □ □ □ □ □ □ □ | □ □ □ □ □ □ □                                │
│         ─────────────────   ─────────────                               │
│            8位指数           7位尾数                                     │
│            范围大            精度低                                      │
│                                                                         │
│  关键区别:                                                               │
│  - BF16指数位与FP32相同 → 数值范围相同 → 无需梯度缩放!                   │
│  - BF16尾数位较少 → 精度略低 → 但对训练影响很小                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.bf16_training import BF16Config, BF16Trainer

# 配置BF16
config = BF16Config(
    enabled=True,
    convert_weights=True,       # 转换模型权重为BF16
    keep_batchnorm_fp32=True,   # BatchNorm保持FP32 (数值稳定)
    keep_layernorm_fp32=True,   # LayerNorm保持FP32
    master_weights=True,        # 保持FP32主权重 (更新精度)
)

# 创建训练器
trainer = BF16Trainer(model, config)

# 训练循环 (无需梯度缩放!)
for batch in dataloader:
    trainer.zero_grad()

    with trainer.autocast():
        output = trainer.forward(batch)
        loss = criterion(output, target)

    trainer.backward(loss)  # 直接反向传播
    trainer.step(optimizer)  # 直接更新
```

---

## 梯度缩放 (Gradient Scaling)

### 动态缩放算法

```
┌─────────────────────────────────────────────────────────────────────────┐
│  动态损失缩放 (Dynamic Loss Scaling)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  初始: scale = 65536                                                    │
│                                                                         │
│  每步检查:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  if 梯度有inf/nan (溢出):                                        │   │
│  │      scale = scale × 0.5    # 缩小缩放因子                       │   │
│  │      跳过本次更新                                                 │   │
│  │  else:                                                           │   │
│  │      正常更新权重                                                 │   │
│  │      连续成功次数 += 1                                            │   │
│  │      if 连续成功次数 >= 2000:                                     │   │
│  │          scale = scale × 2  # 增大缩放因子                        │   │
│  │          连续成功次数 = 0                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  目标: 找到最大的scale，使梯度既不下溢也不溢出                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 使用示例

```python
from src.gradient_scaling import SmartGradScaler, GradScalerConfig

# 配置梯度缩放器
config = GradScalerConfig(
    init_scale=65536.0,      # 初始缩放因子
    growth_factor=2.0,       # 增长倍数
    backoff_factor=0.5,      # 回退倍数
    growth_interval=2000,    # 增长间隔
)

scaler = SmartGradScaler(config)

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        loss = model(batch)

    # 缩放损失并反向传播
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()

    # 还原梯度
    scaler.unscale_(optimizer)

    # 更新 (如果没有溢出)
    if scaler.step(optimizer):
        print("更新成功")
    else:
        print("检测到溢出，跳过更新")

    # 更新缩放因子
    scaler.update()

    # 监控
    print(f"当前scale: {scaler.get_scale()}")
    print(f"溢出率: {scaler.get_overflow_ratio():.2%}")
```

---

## 常见问题与最佳实践

### 如何选择FP16还是BF16？

```
有Ampere+GPU (A100, RTX 30系列+)?
    ├── 是 → 优先使用BF16 (更简单，更稳定)
    └── 否 → 使用FP16 + AMP (需要梯度缩放)

训练不稳定 (loss爆炸/NaN)?
    ├── 检查学习率是否过大
    ├── 检查梯度裁剪是否启用
    ├── 尝试增大init_scale
    └── 考虑某些层保持FP32 (如LayerNorm)
```

### 哪些操作应该保持FP32？

```python
# 这些操作对精度敏感，建议保持FP32:
- BatchNorm / LayerNorm  # 归一化统计量
- Softmax               # 指数运算
- Loss计算              # 累积求和
- 小张量运算            # 精度损失相对更大
```

## 运行测试

```bash
pytest tests/test_mixed_precision.py -v
```

## 参考资料

- [Mixed Precision Training (ICLR 2018)](https://arxiv.org/abs/1710.03740)
- [NVIDIA AMP Documentation](https://nvidia.github.io/apex/amp.html)
- [PyTorch AMP Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [BF16 Training Study](https://arxiv.org/abs/1905.12322)
