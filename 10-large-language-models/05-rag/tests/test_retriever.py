"""
检索器模块单元测试

测试覆盖：
    - RetrieverConfig: 配置验证
    - DenseRetriever: 稠密检索器
    - SparseRetriever: 稀疏检索器
    - HybridRetriever: 混合检索器
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from embeddings import DenseEmbedding, SparseEmbedding, EmbeddingConfig
from vector_store import Document, SimpleVectorStore
from retriever import (
    RetrieverConfig,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
)


class TestRetrieverConfig(unittest.TestCase):
    """RetrieverConfig测试类。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = RetrieverConfig()
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.score_threshold, 0.0)

    def test_custom_config(self):
        """测试自定义配置。"""
        config = RetrieverConfig(top_k=10, score_threshold=0.5)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.score_threshold, 0.5)

    def test_invalid_top_k(self):
        """测试无效top_k。"""
        with self.assertRaises(ValueError):
            RetrieverConfig(top_k=0)
        with self.assertRaises(ValueError):
            RetrieverConfig(top_k=-1)

    def test_invalid_rerank_top_k(self):
        """测试无效rerank_top_k。"""
        with self.assertRaises(ValueError):
            RetrieverConfig(top_k=10, rerank_top_k=5)


class TestDenseRetriever(unittest.TestCase):
    """DenseRetriever测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.config = EmbeddingConfig(dimension=64)
        self.embedding = DenseEmbedding(self.config, random_seed=42)
        self.store = SimpleVectorStore()
        self.retriever = DenseRetriever(
            embedding=self.embedding,
            vector_store=self.store,
            config=RetrieverConfig(top_k=3),
        )
        self.documents = [
            Document(content="机器学习是人工智能的核心技术"),
            Document(content="深度学习使用神经网络进行学习"),
            Document(content="自然语言处理用于理解人类语言"),
            Document(content="计算机视觉处理图像和视频"),
            Document(content="强化学习通过奖励信号学习"),
        ]
        self.retriever.add_documents(self.documents)

    def test_add_documents(self):
        """测试添加文档。"""
        self.assertEqual(self.store.count, 5)

    def test_retrieve(self):
        """测试检索。"""
        results = self.retriever.retrieve("机器学习")
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].rank, 1)

    def test_retrieve_with_threshold(self):
        """测试带阈值检索。"""
        retriever = DenseRetriever(
            embedding=self.embedding,
            vector_store=self.store,
            config=RetrieverConfig(top_k=5, score_threshold=0.99),
        )
        results = retriever.retrieve("机器学习")
        self.assertLessEqual(len(results), 5)


class TestSparseRetriever(unittest.TestCase):
    """SparseRetriever测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.retriever = SparseRetriever(config=RetrieverConfig(top_k=3))
        self.documents = [
            Document(content="机器学习是人工智能的核心技术"),
            Document(content="深度学习使用神经网络进行学习"),
            Document(content="自然语言处理用于理解人类语言"),
        ]
        self.retriever.fit(self.documents)

    def test_fit(self):
        """测试拟合。"""
        self.assertEqual(len(self.retriever._documents), 3)
        self.assertIsNotNone(self.retriever._doc_vectors)

    def test_retrieve(self):
        """测试检索。"""
        results = self.retriever.retrieve("机器学习")
        self.assertLessEqual(len(results), 3)

    def test_not_fitted_error(self):
        """测试未拟合错误。"""
        retriever = SparseRetriever()
        with self.assertRaises(RuntimeError):
            retriever.retrieve("Test")


class TestHybridRetriever(unittest.TestCase):
    """HybridRetriever测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.config = EmbeddingConfig(dimension=64)
        self.embedding = DenseEmbedding(self.config, random_seed=42)
        self.store = SimpleVectorStore()
        self.dense_retriever = DenseRetriever(
            embedding=self.embedding,
            vector_store=self.store,
            config=RetrieverConfig(top_k=5),
        )
        self.sparse_retriever = SparseRetriever(config=RetrieverConfig(top_k=5))
        
        self.documents = [
            Document(content="机器学习是人工智能的核心技术"),
            Document(content="深度学习使用神经网络进行学习"),
            Document(content="自然语言处理用于理解人类语言"),
        ]
        self.dense_retriever.add_documents(self.documents)
        self.sparse_retriever.fit(self.documents)
        
        self.hybrid = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            config=RetrieverConfig(top_k=3),
            alpha=0.7,
            fusion_method="rrf",
        )

    def test_retrieve_rrf(self):
        """测试RRF融合检索。"""
        results = self.hybrid.retrieve("机器学习")
        self.assertLessEqual(len(results), 3)

    def test_retrieve_weighted(self):
        """测试加权融合检索。"""
        hybrid = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            config=RetrieverConfig(top_k=3),
            alpha=0.5,
            fusion_method="weighted",
        )
        results = hybrid.retrieve("机器学习")
        self.assertLessEqual(len(results), 3)

    def test_invalid_alpha(self):
        """测试无效alpha。"""
        with self.assertRaises(ValueError):
            HybridRetriever(
                dense_retriever=self.dense_retriever,
                sparse_retriever=self.sparse_retriever,
                alpha=1.5,
            )

    def test_invalid_fusion_method(self):
        """测试无效融合方法。"""
        with self.assertRaises(ValueError):
            HybridRetriever(
                dense_retriever=self.dense_retriever,
                sparse_retriever=self.sparse_retriever,
                fusion_method="invalid",
            )


if __name__ == "__main__":
    unittest.main()
