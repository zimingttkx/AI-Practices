"""
向量嵌入 (Embeddings) 实现

本模块提供多种文本向量嵌入方法的实现，是RAG系统的核心组件。

=== 核心思想 ===

向量嵌入将文本映射到连续向量空间，使语义相似的文本在空间中距离相近。
本模块实现三种嵌入方式：

1. 稠密嵌入 (Dense Embedding)
   - 使用神经网络生成连续向量
   - 捕获语义相似性，适用于同义词/近义词匹配
   - 计算成本较高，但语义理解能力强

2. 稀疏嵌入 (Sparse Embedding)
   - 基于词频统计的向量表示 (BM25)
   - 适用于关键词精确匹配
   - 计算效率高，但存在词汇鸿沟问题

3. 混合嵌入 (Hybrid Embedding)
   - 结合稠密和稀疏嵌入的优势
   - 通过加权融合或RRF实现
   - 平衡语义理解和关键词匹配

=== 数学基础 ===

稠密嵌入 - Mean Pooling:
    E(text) = (1/n) * Σ E(token_i)

稠密嵌入 - Attention Pooling:
    E(text) = Σ α_i * E(token_i)
    α_i = softmax(u^T * E(token_i))

BM25评分公式:
    score(D, Q) = Σ IDF(q_i) * TF(q_i, D)
    
    其中:
    IDF(q) = log((N - df(q) + 0.5) / (df(q) + 0.5) + 1)
    TF(q, D) = f(q, D) * (k1 + 1) / (f(q, D) + k1 * (1 - b + b * |D|/avgdl))
    
    参数说明:
    - N: 文档总数
    - df(q): 包含词q的文档数
    - f(q, D): 词q在文档D中的频率
    - |D|: 文档D的长度
    - avgdl: 平均文档长度
    - k1: 词频饱和参数 (默认1.5)
    - b: 长度归一化参数 (默认0.75)

混合嵌入 - 线性加权:
    hybrid_score = α * dense_score + (1-α) * sparse_score

混合嵌入 - RRF (Reciprocal Rank Fusion):
    rrf_score = Σ 1/(k + rank_i)  # k=60为推荐值

=== 算法流程 ===

稠密嵌入流程:
    输入: 文本
      ↓
    分词 (Tokenization)
      ↓
    词嵌入查找 (Token Embedding Lookup)
      ↓
    平均池化 (Mean Pooling)
      ↓
    L2归一化 (可选)
      ↓
    输出: 稠密向量 [dimension]

稀疏嵌入 (BM25) 流程:
    输入: 文档集合
      ↓
    分词并统计词频
      ↓
    计算IDF值 (预计算)
      ↓
    计算平均文档长度
      ↓
    输入: 查询文本
      ↓
    计算TF-IDF分数
      ↓
    输出: 稀疏向量 [vocab_size]

混合嵌入流程:
    输入: 文本
      ↓
    ┌─────────────┬─────────────┐
    ↓             ↓             ↓
稠密嵌入        稀疏嵌入        加权融合
    ↓             ↓             ↓
dense_vec    sparse_vec   hybrid_vec
    └─────────────┴─────────────┘
                    ↓
              输出: 混合向量

=== 参考文献 ===

1. Dense Embedding:
   - Reimers & Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" EMNLP 2019
   - Muennighoff. "Sentence Transformers: Multilingual Sentence Embeddings" 2020

2. Sparse Embedding:
   - Robertson & Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond" 2009
   - Robertson et al. "Okapi/ BM25 at TREC" 1995

3. Hybrid Embedding:
   - Luan et al. "Sparse, Dense, and Attentional Representations for Knowledge Base Embedding" 2021
   - Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" SIGIR 2009

=== 核心组件 ===

    - EmbeddingConfig: 嵌入配置类
    - BaseEmbedding: 嵌入模型基类
    - DenseEmbedding: 稠密向量嵌入 (基于词向量平均)
    - SparseEmbedding: 稀疏向量嵌入 (基于BM25)
    - HybridEmbedding: 混合嵌入 (稠密+稀疏融合)

作者: AI-Practices
许可证: MIT
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


__all__ = [
    "EmbeddingConfig",
    "BaseEmbedding",
    "DenseEmbedding",
    "SparseEmbedding",
    "HybridEmbedding",
]


@dataclass
class EmbeddingConfig:
    """嵌入配置。

    参数：
        model_name: 模型名称
        dimension: 嵌入维度
        normalize: 是否归一化
        batch_size: 批处理大小
        max_length: 最大序列长度
    """
    model_name: str = "text-embedding-ada-002"
    dimension: int = 1536
    normalize: bool = True
    batch_size: int = 32
    max_length: int = 512

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError(f"dimension必须为正数，得到 {self.dimension}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size必须为正数，得到 {self.batch_size}")
        if self.max_length <= 0:
            raise ValueError(f"max_length必须为正数，得到 {self.max_length}")


class BaseEmbedding(ABC):
    """嵌入基类。

    定义所有嵌入模型的通用接口。

    属性：
        config: 嵌入配置
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        """初始化嵌入模型。

        参数：
            config: 嵌入配置，如果为None则使用默认配置
        """
        self.config = config or EmbeddingConfig()

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """将单个文本转换为向量。

        参数：
            text: 输入文本

        返回：
            嵌入向量
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将多个文本转换为向量。

        参数：
            texts: 输入文本列表

        返回：
            嵌入向量矩阵 [num_texts, dimension]
        """
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """将查询文本转换为向量。

        某些模型对查询和文档使用不同的嵌入方式。

        参数：
            query: 查询文本

        返回：
            查询嵌入向量
        """
        return self.embed_text(query)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2归一化向量。

        参数：
            vectors: 输入向量

        返回：
            归一化后的向量
        """
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / norm if norm > 0 else vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        return vectors / norms


class DenseEmbedding(BaseEmbedding):
    """稠密向量嵌入。

    使用神经网络模型生成稠密向量表示。
    本实现提供简化版本，实际应用中应使用预训练模型。

    数学原理：
        文本嵌入 = mean(token_embeddings)
        
        归一化: v_norm = v / ||v||_2

    示例：
        >>> config = EmbeddingConfig(dimension=384)
        >>> embedding = DenseEmbedding(config)
        >>> vector = embedding.embed_text("Hello world")
        >>> print(vector.shape)
        (384,)
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        random_seed: int = 42,
    ) -> None:
        """初始化稠密嵌入。

        参数：
            config: 嵌入配置
            random_seed: 随机种子（用于演示）
        """
        super().__init__(config)
        self._rng = np.random.RandomState(random_seed)
        self._vocab: Dict[str, np.ndarray] = {}

    def _tokenize(self, text: str) -> List[str]:
        """简单分词。

        参数：
            text: 输入文本

        返回：
            词元列表
        """
        # 转小写并提取单词
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # 截断到最大长度
        return tokens[:self.config.max_length]

    def _get_token_embedding(self, token: str) -> np.ndarray:
        """获取词元嵌入。

        参数：
            token: 词元

        返回：
            词元嵌入向量
        """
        # 懒加载：首次遇到的词元生成随机向量
        if token not in self._vocab:
            self._vocab[token] = self._rng.randn(self.config.dimension).astype(np.float32)
        return self._vocab[token]

    def embed_text(self, text: str) -> np.ndarray:
        """将单个文本转换为向量。

        使用词元嵌入的平均值作为文本嵌入。

        参数：
            text: 输入文本

        返回：
            嵌入向量
        """
        tokens = self._tokenize(text)
        if not tokens:
            # 空文本返回零向量
            return np.zeros(self.config.dimension, dtype=np.float32)

        # 平均池化：mean(token_embeddings)
        embeddings = [self._get_token_embedding(token) for token in tokens]
        vector = np.mean(embeddings, axis=0)

        # L2归一化
        if self.config.normalize:
            vector = self._normalize(vector)

        return vector

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将多个文本转换为向量。

        参数：
            texts: 输入文本列表

        返回：
            嵌入向量矩阵 [num_texts, dimension]
        """
        vectors = [self.embed_text(text) for text in texts]
        return np.array(vectors)

    def __repr__(self) -> str:
        return (
            f"DenseEmbedding(dimension={self.config.dimension}, "
            f"normalize={self.config.normalize}, vocab_size={len(self._vocab)})"
        )


class SparseEmbedding(BaseEmbedding):
    """稀疏向量嵌入 (BM25)。

    基于BM25算法的稀疏向量表示，适用于关键词匹配。

    数学原理：
        BM25评分公式：
        score(D, Q) = Σ IDF(qi) * TF(qi, D)
        
        其中：
        - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
        - TF(qi, D) = f(qi, D) * (k1 + 1) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
        
        参数说明：
        - N: 文档总数
        - df(qi): 包含词qi的文档数
        - f(qi, D): 词qi在文档D中的频率
        - |D|: 文档D的长度
        - avgdl: 平均文档长度
        - k1: 词频饱和参数 (默认1.5)
        - b: 文档长度归一化参数 (默认0.75)

    示例：
        >>> embedding = SparseEmbedding()
        >>> embedding.fit(["hello world", "world peace"])
        >>> vector = embedding.embed_text("hello")
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """初始化BM25嵌入。

        参数：
            config: 嵌入配置
            k1: 词频饱和参数
            b: 文档长度归一化参数
        """
        super().__init__(config)
        self.k1 = k1
        self.b = b
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._doc_count: int = 0
        self._fitted: bool = False

    def _tokenize(self, text: str) -> List[str]:
        """简单分词。"""
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def fit(self, documents: List[str]) -> "SparseEmbedding":
        """拟合BM25模型。

        计算IDF值和平均文档长度。

        参数：
            documents: 文档列表

        返回：
            self
        """
        self._doc_count = len(documents)
        doc_freqs: Dict[str, int] = {}
        total_length = 0

        # 第一遍：统计文档频率和总长度
        for doc in documents:
            tokens = self._tokenize(doc)
            total_length += len(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freqs[token] = doc_freqs.get(token, 0) + 1

        # 计算平均文档长度
        self._avgdl = total_length / self._doc_count if self._doc_count > 0 else 0

        # 第二遍：构建词汇表和计算IDF
        # IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
        for idx, token in enumerate(doc_freqs.keys()):
            self._vocab[token] = idx
            df = doc_freqs[token]
            self._idf[token] = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1)

        self._fitted = True
        return self

    def embed_text(self, text: str) -> np.ndarray:
        """将文本转换为稀疏向量。

        参数：
            text: 输入文本

        返回：
            稀疏嵌入向量
        """
        if not self._fitted:
            raise RuntimeError("必须先调用fit()方法")

        tokens = self._tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)

        vector = np.zeros(len(self._vocab), dtype=np.float32)

        # 计算BM25分数
        # TF(qi, D) = f(qi, D) * (k1 + 1) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
        for token, freq in tf.items():
            if token in self._vocab:
                idx = self._vocab[token]
                idf = self._idf.get(token, 0)
                # 词频饱和项
                numerator = freq * (self.k1 + 1)
                # 文档长度归一化项
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
                # BM25 = IDF * TF
                vector[idx] = idf * numerator / denominator

        return vector

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将多个文本转换为稀疏向量。"""
        return np.array([self.embed_text(text) for text in texts])

    @property
    def vocab_size(self) -> int:
        """词汇表大小。"""
        return len(self._vocab)

    def __repr__(self) -> str:
        return (
            f"SparseEmbedding(k1={self.k1}, b={self.b}, "
            f"vocab_size={self.vocab_size}, fitted={self._fitted})"
        )


class HybridEmbedding(BaseEmbedding):
    """混合嵌入。

    结合稠密嵌入和稀疏嵌入的优势。

    数学原理：
        混合向量 = [dense_vec * alpha, sparse_vec * (1 - alpha)]
        
        其中 alpha 控制稠密/稀疏嵌入的权重比例。

    示例：
        >>> dense = DenseEmbedding()
        >>> sparse = SparseEmbedding()
        >>> hybrid = HybridEmbedding(dense, sparse, alpha=0.7)
        >>> vector = hybrid.embed_text("Hello world")
    """

    def __init__(
        self,
        dense_embedding: DenseEmbedding,
        sparse_embedding: SparseEmbedding,
        config: Optional[EmbeddingConfig] = None,
        alpha: float = 0.5,
    ) -> None:
        """初始化混合嵌入。

        参数：
            dense_embedding: 稠密嵌入模型
            sparse_embedding: 稀疏嵌入模型
            config: 嵌入配置
            alpha: 稠密嵌入权重 (0-1)
        """
        super().__init__(config)
        self.dense_embedding = dense_embedding
        self.sparse_embedding = sparse_embedding
        self.alpha = alpha

        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha必须在[0,1]范围内，得到 {alpha}")

    def embed_text(self, text: str) -> np.ndarray:
        """将文本转换为混合向量。

        返回稠密向量和稀疏向量的加权拼接。

        参数：
            text: 输入文本

        返回：
            混合嵌入向量
        """
        # 分别获取稠密和稀疏嵌入
        dense_vec = self.dense_embedding.embed_text(text)
        sparse_vec = self.sparse_embedding.embed_text(text)
        # 加权拼接: [dense * alpha, sparse * (1-alpha)]
        return np.concatenate([
            dense_vec * self.alpha,
            sparse_vec * (1 - self.alpha),
        ])

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将多个文本转换为混合向量。"""
        return np.array([self.embed_text(text) for text in texts])

    def get_dense_embedding(self, text: str) -> np.ndarray:
        """获取稠密嵌入。"""
        return self.dense_embedding.embed_text(text)

    def get_sparse_embedding(self, text: str) -> np.ndarray:
        """获取稀疏嵌入。"""
        return self.sparse_embedding.embed_text(text)

    def __repr__(self) -> str:
        return (
            f"HybridEmbedding(alpha={self.alpha}, "
            f"dense={self.dense_embedding.__class__.__name__}, "
            f"sparse={self.sparse_embedding.__class__.__name__})"
        )
