"""
高效架构模块 (Efficient Architectures)

本模块实现了 2024-2026 年最前沿的高效 LLM 架构：
- Mamba: 选择性状态空间模型，线性复杂度
- MoE: 稀疏专家混合，条件计算

参考文献:
1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling 
   with Selective State Spaces. arXiv:2312.00752
2. Fedus, W., et al. (2022). Switch Transformers: Scaling to 
   Trillion Parameter Models. JMLR.
"""

from .mamba import (
    MambaConfig,
    SelectiveSSM,
    MambaBlock,
    MambaLayer,
    MambaModel,
    MambaForCausalLM,
    create_mamba_model,
)

from .moe import (
    MoEConfig,
    RouterType,
    Expert,
    Router,
    TopKRouter,
    SwitchRouter,
    ExpertChoiceRouter,
    MoELayer,
    MoETransformerBlock,
    MoEModel,
    create_moe_model,
    compute_load_balancing_loss,
)

__all__ = [
    # Mamba
    "MambaConfig",
    "SelectiveSSM",
    "MambaBlock",
    "MambaLayer",
    "MambaModel",
    "MambaForCausalLM",
    "create_mamba_model",
    # MoE
    "MoEConfig",
    "RouterType",
    "Expert",
    "Router",
    "TopKRouter",
    "SwitchRouter",
    "ExpertChoiceRouter",
    "MoELayer",
    "MoETransformerBlock",
    "MoEModel",
    "create_moe_model",
    "compute_load_balancing_loss",
]
