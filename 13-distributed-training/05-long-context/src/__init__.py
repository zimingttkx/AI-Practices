"""
长上下文模块 (Long Context)

本模块实现分布式长序列处理技术：
- Ring Attention: 环形注意力，支持近乎无限序列长度
- Sequence Parallelism: 序列并行工具

参考文献:
1. Liu, H., et al. (2023). Ring Attention with Blockwise 
   Transformers for Near-Infinite Context. arXiv:2310.01889
"""

from .ring_attention import (
    RingAttentionConfig,
    BlockwiseAttention,
    RingCommunicator,
    RingAttention,
    RingAttentionLayer,
    SequenceParallel,
    create_ring_attention,
    compute_blockwise_attention,
)

__all__ = [
    "RingAttentionConfig",
    "BlockwiseAttention",
    "RingCommunicator",
    "RingAttention",
    "RingAttentionLayer",
    "SequenceParallel",
    "create_ring_attention",
    "compute_blockwise_attention",
]
