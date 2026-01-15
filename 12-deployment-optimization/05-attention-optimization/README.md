# Flash Attention 优化模块

Flash Attention 系列算法的教育性实现，包含 V1/V2/V3 三个版本。

## 目录结构

```
05-attention-optimization/
├── src/
│   ├── __init__.py
│   └── flash_attn.py      # 核心实现 (~1300行)
├── tests/
│   ├── __init__.py
│   └── test_flash_attn.py # 单元测试 (44个测试)
├── notebooks/
│   └── 01_FlashAttention3_tutorial.ipynb
├── 知识点.md
└── README.md
```

## 快速开始

```python
from src.flash_attn import create_flash_attention
import numpy as np

# 创建 Flash Attention V3
attn = create_flash_attention("v3", block_size=128, causal=True)

# 准备数据
batch, seq, dim = 2, 512, 64
query = np.random.randn(batch, seq, dim).astype(np.float32)
key = np.random.randn(batch, seq, dim).astype(np.float32)
value = np.random.randn(batch, seq, dim).astype(np.float32)

# 前向传播
output = attn(query, key, value)
```

## 主要组件

| 组件 | 描述 |
|:-----|:-----|
| `OnlineSoftmax` | 在线 softmax 算法 |
| `BlockwiseAttention` | 分块注意力计算 |
| `WarpScheduler` | Producer-Consumer 异步调度 |
| `FP8Quantizer` | FP8 E4M3/E5M2 量化 |
| `IncoherentProcessor` | Hadamard 变换降低量化误差 |
| `FlashAttentionV1/V2/V3` | 三个版本的完整实现 |

## 运行测试

```bash
pytest tests/test_flash_attn.py -v
```

## 参考文献

- [Flash Attention V1](https://arxiv.org/abs/2205.14135)
- [Flash Attention V2](https://arxiv.org/abs/2307.08691)
- [Flash Attention V3](https://arxiv.org/abs/2407.08608)
