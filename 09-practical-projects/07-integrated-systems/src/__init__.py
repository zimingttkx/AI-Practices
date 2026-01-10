"""
跨模块集成系统。

整合多模态检索、RAG、Agent推理和性能测试的端到端系统。

模块组成:
    - 多模态检索: MultimodalRetriever, CLIPEncoder
    - 视觉问答: VisionQAAgent
    - 代码助手: CodeRetriever, CodeAgent, ReviewAgent
    - 性能测试: RAGBenchmark, AgentBenchmark, MultimodalBenchmark
"""

# 多模态检索
from .multimodal_retriever import (
    ModalityType,
    MultimodalDocument,
    MultimodalRetriever,
    ImageEncoder,
    TextEncoder,
    CLIPEncoder,
    SearchResult,
)

# 视觉问答智能体
from .vision_qa_agent import (
    ActionType,
    VisionAction,
    VisionObservation,
    VisionQAAgent,
    AgentResult,
)

# 端到端流水线
from .pipeline import (
    PipelineConfig,
    PipelineResult,
    MultimodalRAGPipeline,
)

# 代码检索
from .code_retriever import (
    CodeLanguage,
    CodeBlockType,
    CodeDocument,
    CodeChunker,
    CodeEmbedding,
    CodeRetriever,
)

# 代码生成智能体
from .code_agent import (
    CodeActionType,
    CodeAction,
    CodeResult,
    CodeAgent,
)

# 代码审查智能体
from .review_agent import (
    IssueSeverity,
    IssueCategory,
    ReviewIssue,
    ReviewResult,
    ReviewAgent,
)

# 性能基准测试
from .rag_benchmark import RAGBenchmark, RAGMetrics
from .agent_benchmark import AgentBenchmark, AgentMetrics, TaskResult
from .multimodal_benchmark import MultimodalBenchmark, MultimodalMetrics

__all__ = [
    # 多模态
    "ModalityType",
    "MultimodalDocument",
    "MultimodalRetriever",
    "ImageEncoder",
    "TextEncoder",
    "CLIPEncoder",
    "SearchResult",
    # 视觉问答
    "ActionType",
    "VisionAction",
    "VisionObservation",
    "VisionQAAgent",
    "AgentResult",
    # 流水线
    "PipelineConfig",
    "PipelineResult",
    "MultimodalRAGPipeline",
    # 代码检索
    "CodeLanguage",
    "CodeBlockType",
    "CodeDocument",
    "CodeChunker",
    "CodeEmbedding",
    "CodeRetriever",
    # 代码生成
    "CodeActionType",
    "CodeAction",
    "CodeResult",
    "CodeAgent",
    # 代码审查
    "IssueSeverity",
    "IssueCategory",
    "ReviewIssue",
    "ReviewResult",
    "ReviewAgent",
    # 基准测试
    "RAGBenchmark",
    "RAGMetrics",
    "AgentBenchmark",
    "AgentMetrics",
    "TaskResult",
    "MultimodalBenchmark",
    "MultimodalMetrics",
]
