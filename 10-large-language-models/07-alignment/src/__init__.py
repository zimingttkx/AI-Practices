"""
对齐模块 (Alignment Module)

============================================================
核心思想 (Core Idea)
============================================================
AI对齐旨在使语言模型的行为与人类意图和价值观保持一致。
通过人类反馈强化学习(RLHF)和直接偏好优化(DPO)等技术，
可以让模型生成更有帮助、更安全、更诚实的回复。

============================================================
参考文献 (References)
============================================================
[1] Ouyang, L., et al. (2022). Training language models to follow
    instructions with human feedback. NeurIPS 2022.
[2] Rafailov, R., et al. (2023). Direct Preference Optimization:
    Your Language Model is Secretly a Reward Model. NeurIPS 2023.
[3] Bai, Y., et al. (2022). Constitutional AI: Harmlessness from
    AI Feedback. arXiv:2212.08073.
"""

from .reward_model import (
    RewardModel,
    RewardModelConfig,
    RewardModelOutput,
    PairwiseRewardModel,
    PreferenceDataset,
)

from .rlhf import (
    RLHFConfig,
    RLHFTrainer,
    PPOTrainer,
    ValueHead,
    compute_advantages,
    compute_rewards,
)

from .dpo import (
    DPOConfig,
    DPOTrainer,
    DPOLoss,
    PreferenceDataCollator,
    compute_dpo_loss,
    compute_reference_logprobs,
)

from .constitutional import (
    ConstitutionalAI,
    ConstitutionalBatch,
    ConstitutionalConfig,
    ConstitutionalPrinciple,
    RevisionBatch,
    SelfCriticTrainer,
    create_constitutional_ai,
)

from .rlaif import (
    RLAIFConfig,
    AIFeedbackGenerator,
    RLAIFTrainer,
    create_rlaif_trainer,
)

from .kto import (
    KTOConfig,
    KTOLoss,
    KTOTrainer,
    KTOBatch,
    create_kto_trainer,
)

from .orpo import (
    ORPOConfig,
    ORPOLoss,
    ORPOTrainer,
    PreferenceBatch,
    create_orpo_trainer,
)


__all__ = [
    # Reward Model
    "RewardModel",
    "RewardModelConfig",
    "RewardModelOutput",
    "PairwiseRewardModel",
    "PreferenceDataset",
    # RLHF
    "RLHFConfig",
    "RLHFTrainer",
    "PPOTrainer",
    "ValueHead",
    "compute_advantages",
    "compute_rewards",
    # DPO
    "DPOConfig",
    "DPOTrainer",
    "DPOLoss",
    "PreferenceDataCollator",
    "compute_dpo_loss",
    "compute_reference_logprobs",
    # Constitutional AI
    "ConstitutionalAI",
    "ConstitutionalBatch",
    "ConstitutionalConfig",
    "ConstitutionalPrinciple",
    "RevisionBatch",
    "SelfCriticTrainer",
    "create_constitutional_ai",
    # RLAIF
    "RLAIFConfig",
    "AIFeedbackGenerator",
    "RLAIFTrainer",
    "create_rlaif_trainer",
    # KTO
    "KTOConfig",
    "KTOLoss",
    "KTOTrainer",
    "KTOBatch",
    "create_kto_trainer",
    # ORPO
    "ORPOConfig",
    "ORPOLoss",
    "ORPOTrainer",
    "PreferenceBatch",
    "create_orpo_trainer",
]
