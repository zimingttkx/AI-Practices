# 长上下文处理 (Long Context)

本模块实现分布式长序列处理技术，支持百万级 token 的序列长度。

## 核心组件

### Ring Attention
- **RingAttentionConfig**: 配置类
- **BlockwiseAttention**: 分块注意力 (在线 softmax)
- **RingAttention**: 环形注意力主类
- **SequenceParallel**: 序列并行工具

## 快速开始

```python
from src import RingAttentionConfig, RingAttention, create_ring_attention

# 创建配置
config = RingAttentionConfig(
    d_model=768,
    n_heads=12,
    block_size=1024,
    n_devices=8,
)

# 最大序列长度: 8192 tokens
print(f"Max seq len: {config.max_seq_len}")

# 创建模型
attn = create_ring_attention(d_model=768, n_heads=12)
```

## 目录结构

```
05-long-context/
├── src/
│   ├── __init__.py
│   └── ring_attention.py    # Ring Attention 实现
├── tests/
│   └── test_ring_attention.py
├── notebooks/
│   └── 03_RingAttention_tutorial.ipynb
├── 知识点.md
└── README.md
```

## 测试

```bash
pytest tests/ -v
```

## 参考文献

1. [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)
2. [Sequence Parallelism](https://arxiv.org/abs/2105.13120)
