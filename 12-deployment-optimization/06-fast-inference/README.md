# 06-fast-inference: 高效推理模块

本模块实现 LLM 高效推理的核心技术，包括 PagedAttention、Continuous Batching 和 Speculative Decoding。

## 目录结构

```
06-fast-inference/
├── src/
│   ├── paged_attention.py    # PagedAttention 实现
│   ├── continuous_batch.py   # 连续批处理实现
│   ├── speculative.py        # 推测解码实现
│   └── __init__.py
├── tests/
│   ├── test_paged_attention.py
│   ├── test_continuous_batch.py
│   ├── test_speculative.py
│   └── __init__.py
├── notebooks/
│   ├── 01_PagedAttention_tutorial.ipynb
│   ├── 02_ContinuousBatching_tutorial.ipynb
│   ├── 03_SpeculativeDecoding_tutorial.ipynb
│   └── 04_AWQ_tutorial.ipynb
├── 知识点.md
└── README.md
```

## 核心技术

### 1. PagedAttention
借鉴操作系统虚拟内存分页，实现高效 KV Cache 管理：
- 按需分配物理块
- Copy-on-Write 支持
- 内存效率提升 60-80%

### 2. Continuous Batching
迭代级调度，动态调整批次：
- FCFS/SJF/Priority 调度策略
- GPU 利用率 >90%
- 吞吐量提升 2-3x

### 3. Speculative Decoding
小模型推测 + 大模型验证：
- 拒绝采样保证分布一致
- 无损加速 2-3x

## 快速开始

```python
from src.paged_attention import create_paged_attention
from src.continuous_batch import create_continuous_batcher
from src.speculative import create_speculative_decoder

# PagedAttention
paged_attn = create_paged_attention(
    num_heads=32, head_dim=128, num_layers=32,
    block_size=16, num_blocks=1000
)

# Continuous Batching
batcher = create_continuous_batcher(
    max_batch_size=256,
    scheduling_policy="fcfs"
)

# Speculative Decoding
decoder = create_speculative_decoder(
    num_speculative_tokens=4,
    vocab_size=32000
)
```

## 运行测试

```bash
cd 12-deployment-optimization/06-fast-inference
python -m pytest tests/ -v
```

## 参考资料

- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
