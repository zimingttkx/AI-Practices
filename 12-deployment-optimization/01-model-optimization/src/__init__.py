"""
模型优化模块

提供量化、剪枝、知识蒸馏、模型导出和 AWQ 量化功能。
"""

from .quantization import (
    QuantizationConfig,
    QuantizationType,
    QuantizationGranularity,
    FakeQuantize,
    FakeQuantizeModule,
    QuantizedLinear,
    QuantizedConv2d,
    DynamicQuantizer,
    StaticQuantizer,
    QATWrapper,
    quantize_model,
    calibrate_model,
    compute_scale_zero_point,
    quantize_tensor,
    dequantize_tensor,
)

from .pruning import (
    PruningConfig,
    PruningType,
    ImportanceMetric,
    PruningMask,
    MagnitudePruner,
    StructuredPruner,
    GradientPruner,
    IterativePruner,
    prune_model,
    compute_model_sparsity,
    compute_magnitude_importance,
    create_pruning_mask,
)

from .distillation import (
    DistillationConfig,
    DistillationType,
    DistillationLoss,
    FeatureDistillation,
    RelationDistillation,
    AttentionTransfer,
    KnowledgeDistiller,
    distill_model,
    soft_cross_entropy,
)

from .export import (
    ExportConfig,
    ExportFormat,
    ONNXExporter,
    TorchScriptExporter,
    ModelAnalyzer,
    export_to_onnx,
    export_to_torchscript,
    compare_model_outputs,
)

from .awq import (
    AWQConfig,
    AWQGranularity,
    ActivationObserver,
    SalientChannelFinder,
    AWQLinear,
    AWQQuantizer,
    create_awq_quantizer,
    quantize_model_awq,
    compute_quantization_error,
    estimate_model_size,
    pack_int4_weights,
    unpack_int4_weights,
)

__all__ = [
    # Quantization
    "QuantizationConfig",
    "QuantizationType",
    "QuantizationGranularity",
    "FakeQuantize",
    "FakeQuantizeModule",
    "QuantizedLinear",
    "QuantizedConv2d",
    "DynamicQuantizer",
    "StaticQuantizer",
    "QATWrapper",
    "quantize_model",
    "calibrate_model",
    "compute_scale_zero_point",
    "quantize_tensor",
    "dequantize_tensor",
    # Pruning
    "PruningConfig",
    "PruningType",
    "ImportanceMetric",
    "PruningMask",
    "MagnitudePruner",
    "StructuredPruner",
    "GradientPruner",
    "IterativePruner",
    "prune_model",
    "compute_model_sparsity",
    "compute_magnitude_importance",
    "create_pruning_mask",
    # Distillation
    "DistillationConfig",
    "DistillationType",
    "DistillationLoss",
    "FeatureDistillation",
    "RelationDistillation",
    "AttentionTransfer",
    "KnowledgeDistiller",
    "distill_model",
    "soft_cross_entropy",
    # Export
    "ExportConfig",
    "ExportFormat",
    "ONNXExporter",
    "TorchScriptExporter",
    "ModelAnalyzer",
    "export_to_onnx",
    "export_to_torchscript",
    "compare_model_outputs",
    # AWQ Quantization
    "AWQConfig",
    "AWQGranularity",
    "ActivationObserver",
    "SalientChannelFinder",
    "AWQLinear",
    "AWQQuantizer",
    "create_awq_quantizer",
    "quantize_model_awq",
    "compute_quantization_error",
    "estimate_model_size",
    "pack_int4_weights",
    "unpack_int4_weights",
]
