"""
检索器 (Retriever) 实现

本模块提供多种文档检索器的实现，是RAG系统的检索层核心。

=== 核心思想 ===

检索器负责根据用户查询从文档库中检索相关文档。
本模块实现三种检索策略：

1. 稠密检索 (Dense Retrieval)
   - 使用向量嵌入进行语义检索
   - 适合同义词/近义词/语义相似查询
   - 计算密集，召回率高

2. 稀疏检索 (Sparse Retrieval)
   - 基于BM25的关键词检索
   - 适合精确关键词匹配
   - 计算高效，召回率低

3. 混合检索 (Hybrid Retrieval)
   - 结合稠密和稀疏检索的优势
   - 通过RRF或加权融合合并结果
   - 平衡召回率和精度

=== 数学基础 ===

稠密检索:
    query_vector = embedding(query)
    score = cosine_similarity(query_vector, doc_vector)

稀疏检索 (BM25):
    score(D, Q) = Σ IDF(q_i) * TF(q_i, D)
    
    其中:
    IDF(q) = log((N - df(q) + 0.5) / (df(q) + 0.5) + 1)
    TF(q, D) = f(q, D) * (k1 + 1) / (f(q, D) + k1 * (1 - b + b * |D|/avgdl))

混合检索 - RRF (Reciprocal Rank Fusion):
    score(doc) = Σ 1/(k + rank_i(doc))
    
    其中:
    - k: 常数，推荐值60
    - rank_i: 第i个检索器中文档的排名

混合检索 - 加权融合:
    score(doc) = α * norm(dense_score) + (1-α) * norm(sparse_score)
    
    归一化:
    norm(x) = (x - min) / (max - min)

=== 算法流程 ===

稠密检索流程:
    输入: 查询文本
      ↓
    嵌入化: query → vector
      ↓
    向量搜索: vector_store.search(vector, top_k)
      ↓
    过滤: score >= threshold
      ↓
    输出: List[SearchResult]

稀疏检索流程:
    输入: 文档集合
      ↓
    拟合: 计算IDF和平均文档长度
      ↓
    预计算: 所有文档的BM25向量
      ↓
    输入: 查询文本
      ↓
    计算查询向量
      ↓
    点积: scores = doc_vectors · query_vector
      ↓
    Top-K排序
      ↓
    输出: List[SearchResult]

混合检索流程:
    输入: 查询文本
      ↓
    ┌────────────────┬────────────────┐
    ↓                ↓                ↓
稠密检索          稀疏检索          融合合并
    ↓                ↓                ↓
dense_results  sparse_results  final_results
    └────────────────┴────────────────┘
                      ↓
              RRF或加权融合
                      ↓
              输出: List[SearchResult]

=== 参考文献 ===

1. Karpukhin et al. "Dense Passage Retrieval for Open-Domain QA" EMNLP 2020
2. Robertson & Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond" 2009
3. Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" SIGIR 2009
4. Izacard & Grave. "Leveraging Passage Retrieval with Generative Models" EMNLP 2021

=== 核心组件 ===

    - RetrieverConfig: 检索器配置类
    - BaseRetriever: 检索器基类 (抽象接口)
    - DenseRetriever: 稠密检索器 (语义检索)
    - SparseRetriever: 稀疏检索器 (BM25)
    - HybridRetriever: 混合检索器 (RRF/加权融合)

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .embeddings import BaseEmbedding, DenseEmbedding, SparseEmbedding
    from .vector_store import Document, SearchResult, VectorStore
except ImportError:
    from embeddings import BaseEmbedding, DenseEmbedding, SparseEmbedding
    from vector_store import Document, SearchResult, VectorStore


__all__ = [
    "RetrieverConfig",
    "BaseRetriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
]


@dataclass
class RetrieverConfig:
    """检索器配置。

    参数：
        top_k: 返回结果数量
        score_threshold: 分数阈值
        use_reranker: 是否使用重排序
        rerank_top_k: 重排序候选数量
    """
    top_k: int = 5
    score_threshold: float = 0.0
    use_reranker: bool = False
    rerank_top_k: int = 20

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"top_k必须为正数，得到 {self.top_k}")
        if self.rerank_top_k < self.top_k:
            raise ValueError("rerank_top_k必须大于等于top_k")


class BaseRetriever(ABC):
    """检索器基类。

    定义所有检索器的通用接口。
    """

    def __init__(self, config: Optional[RetrieverConfig] = None) -> None:
        """初始化检索器。

        参数：
            config: 检索器配置
        """
        self.config = config or RetrieverConfig()

    @abstractmethod
    def retrieve(self, query: str) -> List[SearchResult]:
        """检索相关文档。

        参数：
            query: 查询文本

        返回：
            搜索结果列表
        """
        pass

    def _filter_by_threshold(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """按分数阈值过滤结果。"""
        return [r for r in results if r.score >= self.config.score_threshold]


class DenseRetriever(BaseRetriever):
    """稠密检索器。

    使用稠密向量嵌入进行语义检索。

    示例：
        >>> embedding = DenseEmbedding()
        >>> store = SimpleVectorStore()
        >>> retriever = DenseRetriever(embedding, store)
        >>> results = retriever.retrieve("What is machine learning?")
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: VectorStore,
        config: Optional[RetrieverConfig] = None,
    ) -> None:
        """初始化稠密检索器。

        参数：
            embedding: 嵌入模型
            vector_store: 向量存储
            config: 检索器配置
        """
        super().__init__(config)
        self.embedding = embedding
        self.vector_store = vector_store

    def retrieve(self, query: str) -> List[SearchResult]:
        """检索相关文档。"""
        # 将查询转换为向量
        query_vector = self.embedding.embed_query(query)
        # 在向量存储中搜索
        results = self.vector_store.search(query_vector, top_k=self.config.top_k)
        # 按阈值过滤
        return self._filter_by_threshold(results)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到检索器。

        参数：
            documents: 文档列表

        返回：
            文档ID列表
        """
        # 为缺少嵌入的文档生成嵌入
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedding.embed_text(doc.content)
        return self.vector_store.add_documents(documents)

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(top_k={self.config.top_k}, "
            f"store_count={self.vector_store.count})"
        )


class SparseRetriever(BaseRetriever):
    """稀疏检索器。

    使用BM25等稀疏向量进行关键词检索。

    示例：
        >>> retriever = SparseRetriever()
        >>> retriever.fit(documents)
        >>> results = retriever.retrieve("machine learning")
    """

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """初始化稀疏检索器。

        参数：
            config: 检索器配置
            k1: BM25 k1参数
            b: BM25 b参数
        """
        super().__init__(config)
        self.embedding = SparseEmbedding(k1=k1, b=b)
        self._documents: List[Document] = []
        self._doc_vectors: Optional[np.ndarray] = None

    def fit(self, documents: List[Document]) -> "SparseRetriever":
        """拟合检索器。

        参数：
            documents: 文档列表

        返回：
            self
        """
        self._documents = documents
        # 提取文档内容
        texts = [doc.content for doc in documents]
        # 拟合BM25模型（计算IDF和平均文档长度）
        self.embedding.fit(texts)
        # 预计算所有文档的稀疏向量
        self._doc_vectors = self.embedding.embed_texts(texts)
        return self

    def retrieve(self, query: str) -> List[SearchResult]:
        """检索相关文档。"""
        if self._doc_vectors is None:
            raise RuntimeError("必须先调用fit()方法")

        # 将查询转换为稀疏向量
        query_vector = self.embedding.embed_text(query)
        # 计算查询与所有文档的BM25分数
        scores = np.dot(self._doc_vectors, query_vector)

        # 获取top-k结果索引
        top_indices = np.argsort(scores)[::-1][:self.config.top_k]

        # 构建搜索结果
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(SearchResult(
                document=self._documents[idx],
                score=float(scores[idx]),
                rank=rank + 1,
            ))

        return self._filter_by_threshold(results)

    def __repr__(self) -> str:
        return (
            f"SparseRetriever(top_k={self.config.top_k}, "
            f"doc_count={len(self._documents)})"
        )


class HybridRetriever(BaseRetriever):
    """混合检索器。

    结合稠密检索和稀疏检索的优势。

    融合策略：
        - RRF (Reciprocal Rank Fusion): 1/(k + rank)
        - 加权平均: alpha * dense_score + (1-alpha) * sparse_score

    示例：
        >>> dense_retriever = DenseRetriever(embedding, store)
        >>> sparse_retriever = SparseRetriever()
        >>> hybrid = HybridRetriever(dense_retriever, sparse_retriever, alpha=0.7)
        >>> results = hybrid.retrieve("machine learning")
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        config: Optional[RetrieverConfig] = None,
        alpha: float = 0.5,
        fusion_method: str = "rrf",
    ) -> None:
        """初始化混合检索器。

        参数：
            dense_retriever: 稠密检索器
            sparse_retriever: 稀疏检索器
            config: 检索器配置
            alpha: 稠密检索权重 (0-1)
            fusion_method: 融合方法 ("rrf", "weighted")
        """
        super().__init__(config)
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha
        self.fusion_method = fusion_method

        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha必须在[0,1]范围内，得到 {alpha}")
        if fusion_method not in ("rrf", "weighted"):
            raise ValueError(f"不支持的融合方法: {fusion_method}")

    def retrieve(self, query: str) -> List[SearchResult]:
        """检索相关文档。"""
        # 分别执行稠密检索和稀疏检索
        dense_results = self.dense_retriever.retrieve(query)
        sparse_results = self.sparse_retriever.retrieve(query)

        # 根据融合方法合并结果
        if self.fusion_method == "rrf":
            return self._rrf_fusion(dense_results, sparse_results)
        else:
            return self._weighted_fusion(dense_results, sparse_results)

    def _rrf_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: int = 60,
    ) -> List[SearchResult]:
        """RRF融合。
        
        Reciprocal Rank Fusion: score = Σ 1/(k + rank)
        k=60 是论文推荐的默认值。
        """
        doc_scores: Dict[str, Tuple[float, Document]] = {}

        # 计算稠密检索的RRF分数
        for result in dense_results:
            doc_id = result.document.doc_id
            rrf_score = self.alpha / (k + result.rank)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, result.document)
            else:
                doc_scores[doc_id] = (rrf_score, result.document)

        # 计算稀疏检索的RRF分数
        for result in sparse_results:
            doc_id = result.document.doc_id
            rrf_score = (1 - self.alpha) / (k + result.rank)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + rrf_score, result.document)
            else:
                doc_scores[doc_id] = (rrf_score, result.document)

        # 按分数降序排列
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)

        # 构建最终结果
        results = []
        for rank, (doc_id, (score, doc)) in enumerate(sorted_docs[:self.config.top_k]):
            results.append(SearchResult(document=doc, score=score, rank=rank + 1))

        return results

    def _weighted_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
    ) -> List[SearchResult]:
        """加权融合。
        
        score = alpha * norm(dense_score) + (1-alpha) * norm(sparse_score)
        """
        doc_scores: Dict[str, Tuple[float, Document]] = {}

        # 获取最大分数用于归一化
        dense_max = max((r.score for r in dense_results), default=1.0)
        sparse_max = max((r.score for r in sparse_results), default=1.0)

        # 计算稠密检索的加权分数
        for result in dense_results:
            doc_id = result.document.doc_id
            norm_score = result.score / dense_max if dense_max > 0 else 0
            weighted_score = self.alpha * norm_score
            doc_scores[doc_id] = (weighted_score, result.document)

        # 计算稀疏检索的加权分数并累加
        for result in sparse_results:
            doc_id = result.document.doc_id
            norm_score = result.score / sparse_max if sparse_max > 0 else 0
            weighted_score = (1 - self.alpha) * norm_score
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc_scores[doc_id][0] + weighted_score, result.document)
            else:
                doc_scores[doc_id] = (weighted_score, result.document)

        # 按分数降序排列
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)

        # 构建最终结果
        results = []
        for rank, (doc_id, (score, doc)) in enumerate(sorted_docs[:self.config.top_k]):
            results.append(SearchResult(document=doc, score=score, rank=rank + 1))

        return results

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(alpha={self.alpha}, fusion='{self.fusion_method}', "
            f"top_k={self.config.top_k})"
        )
