"""
向量存储 (Vector Store) 实现

本模块提供向量数据库的实现，支持文档存储、索引构建和相似度搜索。

=== 核心思想 ===

向量存储是RAG系统的核心组件，负责存储和检索文档的向量表示。
本模块实现了一个简单高效的内存向量存储，支持：

1. 文档存储管理
   - 添加/删除/查询文档
   - 元数据管理
   - 自动生成文档ID

2. 相似度搜索
   - 余弦相似度 (语义检索)
   - 点积相似度 (高效检索)
   - 欧氏距离 (绝对距离)

3. 持久化支持
   - 保存到JSON文件
   - 从文件加载

=== 数学基础 ===

余弦相似度 (Cosine Similarity):
    cos(A, B) = (A·B) / (||A|| × ||B||)
             = Σ(ai × bi) / (√Σai² × √Σbi²)
    
    取值范围: [-1, 1]
    特点: 关注方向，忽略模长
    适用: 文本语义检索

点积相似度 (Dot Product):
    A·B = Σ ai × bi
    
    取值范围: [-∞, +∞]
    特点: 计算高效，需要归一化
    适用: 高性能检索

欧氏距离 (Euclidean Distance):
    d(A, B) = √Σ(ai - bi)²
    
    取值范围: [0, +∞]
    特点: 绝对距离，距离越小越相似
    适用: 图像特征匹配

归一化:
    v_norm = v / ||v||_2 = v / √Σvi²

=== 算法流程 ===

添加文档:
    输入: 文档列表 (带嵌入向量)
      ↓
    验证嵌入向量存在
      ↓
    存储到字典: doc_id → Document
      ↓
    存储向量: doc_id → vector
      ↓
    输出: 文档ID列表

相似度搜索:
    输入: 查询向量, top_k
      ↓
    构建向量矩阵 [N, dim]
      ↓
    计算相似度分数
      ↓
    排序获取Top-K索引
      ↓
    构建SearchResult对象
      ↓
    输出: 搜索结果列表 (按分数降序)

=== 参考文献 ===

1. Johnson et al. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" 2019
2. Malkov & Yashunin. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" 2018
3. FAISS: "A library for efficient similarity search and clustering of dense vectors" Meta Research

=== 核心组件 ===

    - Document: 文档数据结构
    - SearchResult: 搜索结果数据结构
    - VectorStore: 向量存储基类 (抽象接口)
    - SimpleVectorStore: 简单向量存储 (内存实现)

"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


__all__ = [
    "Document",
    "SearchResult",
    "VectorStore",
    "SimpleVectorStore",
]


@dataclass
class Document:
    """文档数据结构。

    参数：
        content: 文档内容
        metadata: 元数据
        doc_id: 文档ID（自动生成）
        embedding: 嵌入向量
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("文档内容不能为空")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """从字典创建文档。"""
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id", str(uuid.uuid4())),
        )


@dataclass
class SearchResult:
    """搜索结果。

    参数：
        document: 匹配的文档
        score: 相似度分数
        rank: 排名
    """
    document: Document
    score: float
    rank: int = 0

    def __repr__(self) -> str:
        preview = self.document.content[:50] + "..." if len(self.document.content) > 50 else self.document.content
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, content='{preview}')"


class VectorStore(ABC):
    """向量存储基类。

    定义向量数据库的通用接口。
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到存储。

        参数：
            documents: 文档列表

        返回：
            文档ID列表
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """相似度搜索。

        参数：
            query_vector: 查询向量
            top_k: 返回结果数量

        返回：
            搜索结果列表
        """
        pass

    @abstractmethod
    def delete(self, doc_ids: List[str]) -> int:
        """删除文档。

        参数：
            doc_ids: 要删除的文档ID列表

        返回：
            删除的文档数量
        """
        pass

    @abstractmethod
    def get(self, doc_id: str) -> Optional[Document]:
        """获取文档。

        参数：
            doc_id: 文档ID

        返回：
            文档对象，如果不存在则返回None
        """
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """文档数量。"""
        pass


class SimpleVectorStore(VectorStore):
    """简单向量存储。

    使用NumPy实现的内存向量存储，适用于小规模数据。

    示例：
        >>> store = SimpleVectorStore()
        >>> docs = [Document(content="Hello world", embedding=np.random.randn(384))]
        >>> store.add_documents(docs)
        >>> results = store.search(np.random.randn(384), top_k=1)
    """

    def __init__(self, metric: str = "cosine") -> None:
        """初始化简单向量存储。

        参数：
            metric: 距离度量 ("cosine", "euclidean", "dot")
        """
        if metric not in ("cosine", "euclidean", "dot"):
            raise ValueError(f"不支持的距离度量: {metric}")
        self.metric = metric
        self._documents: Dict[str, Document] = {}
        self._vectors: Dict[str, np.ndarray] = {}

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到存储。"""
        doc_ids = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"文档 {doc.doc_id} 缺少嵌入向量")
            self._documents[doc.doc_id] = doc
            self._vectors[doc.doc_id] = doc.embedding
            doc_ids.append(doc.doc_id)
        return doc_ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """相似度搜索。"""
        if not self._vectors:
            return []

        # 构建向量矩阵
        doc_ids = list(self._vectors.keys())
        vectors = np.array([self._vectors[doc_id] for doc_id in doc_ids])

        # 计算相似度分数
        scores = self._compute_similarity(query_vector, vectors)

        # 获取top-k结果索引（降序排列）
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 构建搜索结果
        results = []
        for rank, idx in enumerate(top_indices):
            doc_id = doc_ids[idx]
            results.append(SearchResult(
                document=self._documents[doc_id],
                score=float(scores[idx]),
                rank=rank + 1,
            ))

        return results

    def _compute_similarity(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
    ) -> np.ndarray:
        """计算相似度。
        
        支持三种度量方式：
        - cosine: 余弦相似度 = dot(a, b) / (||a|| * ||b||)
        - dot: 点积相似度 = dot(a, b)
        - euclidean: 负欧氏距离 = -||a - b||
        """
        if self.metric == "cosine":
            # 余弦相似度：归一化后点积
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            return np.dot(vectors_norm, query_norm)
        elif self.metric == "dot":
            # 点积相似度
            return np.dot(vectors, query)
        else:  # euclidean
            # 负欧氏距离（距离越小，相似度越高）
            distances = np.linalg.norm(vectors - query, axis=1)
            return -distances

    def delete(self, doc_ids: List[str]) -> int:
        """删除文档。"""
        deleted = 0
        for doc_id in doc_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                del self._vectors[doc_id]
                deleted += 1
        return deleted

    def get(self, doc_id: str) -> Optional[Document]:
        """获取文档。"""
        return self._documents.get(doc_id)

    @property
    def count(self) -> int:
        """文档数量。"""
        return len(self._documents)

    def __repr__(self) -> str:
        return f"SimpleVectorStore(metric='{self.metric}', count={self.count})"

    def save(self, path: Union[str, Path]) -> None:
        """保存到文件。"""
        path = Path(path)
        data = {
            "metric": self.metric,
            "documents": [doc.to_dict() for doc in self._documents.values()],
            "vectors": {doc_id: vec.tolist() for doc_id, vec in self._vectors.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SimpleVectorStore":
        """从文件加载。"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls(metric=data["metric"])
        for doc_data in data["documents"]:
            doc = Document.from_dict(doc_data)
            doc.embedding = np.array(data["vectors"][doc.doc_id])
            store._documents[doc.doc_id] = doc
            store._vectors[doc.doc_id] = doc.embedding

        return store
