"""
Vision-Language Models 视觉-语言模型

包含 CLIP、BLIP、LLaVA、SigLIP、CogVLM、Qwen-VL 等模型实现和评估工具。
"""

from .clip import (
    CLIP, CLIPConfig, clip_loss, siglip_loss, create_clip_model,
    ZeroShotClassifier, LinearProbe, CLIPFineTuner
)
from .blip import BLIP, BLIPConfig, create_blip_model
from .llava import LLaVA, LLaVAConfig, create_llava_model
from .siglip import (
    SigLIP, SigLIPConfig, siglip_loss as siglip_sigmoid_loss,
    create_siglip_model, SigLIPZeroShotClassifier
)
from .cogvlm import CogVLM, CogVLMConfig, create_cogvlm_model
from .qwen_vl import QwenVL, QwenVLConfig, create_qwen_vl_model
from .evaluation import (
    MultimodalEvaluator, RetrievalMetrics, BLEU, ROUGE, CIDEr,
    VQAMetrics, GroundingMetrics, ClassificationMetrics, EvaluationResult
)

__all__ = [
    # CLIP
    "CLIP", "CLIPConfig", "clip_loss", "siglip_loss", "create_clip_model",
    "ZeroShotClassifier", "LinearProbe", "CLIPFineTuner",
    # BLIP
    "BLIP", "BLIPConfig", "create_blip_model",
    # LLaVA
    "LLaVA", "LLaVAConfig", "create_llava_model",
    # SigLIP
    "SigLIP", "SigLIPConfig", "siglip_sigmoid_loss", "create_siglip_model", "SigLIPZeroShotClassifier",
    # CogVLM
    "CogVLM", "CogVLMConfig", "create_cogvlm_model",
    # Qwen-VL
    "QwenVL", "QwenVLConfig", "create_qwen_vl_model",
    # Evaluation
    "MultimodalEvaluator", "RetrievalMetrics", "BLEU", "ROUGE", "CIDEr",
    "VQAMetrics", "GroundingMetrics", "ClassificationMetrics", "EvaluationResult",
]
