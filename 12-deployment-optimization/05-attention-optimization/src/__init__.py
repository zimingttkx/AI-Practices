"""
Flash Attention 优化模块

本模块实现 Flash Attention 系列算法，包括：
- Flash Attention 3: 异步流水线 + 低精度优化
- Online Softmax: 分块在线计算
- FP8 量化: Block Quantization + Incoherent Processing
"""

from .flash_attn import (
    # 配置
    FlashAttentionConfig,
    # 核心组件
    OnlineSoftmax,
    BlockwiseAttention,
    WarpScheduler,
    # FP8 量化
    FP8Quantizer,
    IncoherentProcessor,
    # 主类
    FlashAttentionV1,
    FlashAttentionV2,
    FlashAttentionV3,
    # 工厂函数
    create_flash_attention,
    # 工具函数
    standard_attention,
    compute_attention_flops,
    validate_attention_inputs,
)

__all__ = [
    "FlashAttentionConfig",
    "OnlineSoftmax",
    "BlockwiseAttention",
    "WarpScheduler",
    "FP8Quantizer",
    "IncoherentProcessor",
    "FlashAttentionV1",
    "FlashAttentionV2",
    "FlashAttentionV3",
    "create_flash_attention",
    "standard_attention",
    "compute_attention_flops",
    "validate_attention_inputs",
]
