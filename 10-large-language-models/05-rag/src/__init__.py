"""
RAG (Retrieval-Augmented Generation) 模块

本模块提供检索增强生成的完整实现，基于论文
"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)。

核心组件：
    - Embeddings: 文本向量嵌入 (稠密/稀疏/混合)
    - VectorStore: 向量数据库 (内存存储)
    - Retriever: 文档检索器 (稠密/稀疏/混合)
    - RAGPipeline: RAG流水线 (基础/高级)

作者: AI-Practices
许可证: MIT
"""

from __future__ import annotations

# 向量嵌入
from .embeddings import (
    EmbeddingConfig,
    BaseEmbedding,
    DenseEmbedding,
    SparseEmbedding,
    HybridEmbedding,
)

# 向量存储
from .vector_store import (
    Document,
    SearchResult,
    VectorStore,
    SimpleVectorStore,
)

# 检索器
from .retriever import (
    RetrieverConfig,
    BaseRetriever,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
)

# RAG流水线
from .rag_pipeline import (
    RAGConfig,
    RAGResponse,
    TextSplitter,
    RecursiveTextSplitter,
    RAGPipeline,
    AdvancedRAGPipeline,
)


__all__ = [
    # 向量嵌入
    "EmbeddingConfig",
    "BaseEmbedding",
    "DenseEmbedding",
    "SparseEmbedding",
    "HybridEmbedding",
    # 向量存储
    "Document",
    "SearchResult",
    "VectorStore",
    "SimpleVectorStore",
    # 检索器
    "RetrieverConfig",
    "BaseRetriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    # RAG流水线
    "RAGConfig",
    "RAGResponse",
    "TextSplitter",
    "RecursiveTextSplitter",
    "RAGPipeline",
    "AdvancedRAGPipeline",
]

__version__ = "1.0.0"
