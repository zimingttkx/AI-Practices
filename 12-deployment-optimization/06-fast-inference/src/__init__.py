"""
快速推理模块 (Fast Inference)

本模块实现大语言模型的高效推理技术：
1. PagedAttention - 分页注意力，优化 KV Cache 内存管理
2. Continuous Batching - 连续批处理，提高吞吐量
3. Speculative Decoding - 投机解码，加速自回归生成

参考文献:
[1] Kwon et al. "Efficient Memory Management for Large Language Model 
    Serving with PagedAttention" SOSP 2023
[2] Yu et al. "ORCA: A Distributed Serving System for Transformer-Based 
    Generative Models" OSDI 2022
[3] Leviathan et al. "Fast Inference from Transformers via Speculative 
    Decoding" ICML 2023
"""

from .paged_attention import (
    PagedAttentionConfig,
    BlockStatus,
    KVBlock,
    BlockTable,
    BlockAllocator,
    PagedKVCache,
    PagedAttention,
    MultiLayerPagedKVCache,
    create_paged_attention,
)

from .continuous_batch import (
    RequestStatus,
    SchedulingPolicy,
    SchedulerConfig,
    SamplingParams,
    Request,
    RequestQueue,
    SchedulerOutput,
    BatchScheduler,
    BatcherOutput,
    ContinuousBatcher,
    create_continuous_batcher,
    create_request,
)

from .speculative import (
    SpeculativeConfig,
    DraftModel,
    TargetModel,
    MockDraftModel,
    MockTargetModel,
    TokenVerifier,
    SpeculativeOutput,
    SpeculativeDecoder,
    TreeNode,
    TreeSpeculation,
    create_speculative_decoder,
    create_tree_speculation,
    compute_acceptance_rate,
    estimate_speedup,
)

__all__ = [
    # PagedAttention
    "PagedAttentionConfig",
    "BlockStatus",
    "KVBlock",
    "BlockTable", 
    "BlockAllocator",
    "PagedKVCache",
    "PagedAttention",
    "MultiLayerPagedKVCache",
    "create_paged_attention",
    # Continuous Batching
    "RequestStatus",
    "SchedulingPolicy",
    "SchedulerConfig",
    "SamplingParams",
    "Request",
    "RequestQueue",
    "SchedulerOutput",
    "BatchScheduler",
    "BatcherOutput",
    "ContinuousBatcher",
    "create_continuous_batcher",
    "create_request",
    # Speculative Decoding
    "SpeculativeConfig",
    "DraftModel",
    "TargetModel",
    "MockDraftModel",
    "MockTargetModel",
    "TokenVerifier",
    "SpeculativeOutput",
    "SpeculativeDecoder",
    "TreeNode",
    "TreeSpeculation",
    "create_speculative_decoder",
    "create_tree_speculation",
    "compute_acceptance_rate",
    "estimate_speedup",
]
