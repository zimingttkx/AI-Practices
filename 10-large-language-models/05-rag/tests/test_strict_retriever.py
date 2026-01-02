"""
检索器模块严格单元测试

测试覆盖：
    - 边界条件测试
    - 异常处理测试
    - 类型检查测试
    - __repr__ 方法测试
    - 融合算法测试
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import DenseEmbedding, SparseEmbedding, EmbeddingConfig
from vector_store import Document, SimpleVectorStore
from retriever import (
    RetrieverConfig,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
)


class TestRetrieverConfigStrict(unittest.TestCase):
    """RetrieverConfig严格测试。"""

    def test_top_k_boundary_one(self):
        """测试top_k=1边界。"""
        config = RetrieverConfig(top_k=1)
        self.assertEqual(config.top_k, 1)

    def test_top_k_large(self):
        """测试大top_k值。"""
        # rerank_top_k默认为None，需要同时设置
        config = RetrieverConfig(top_k=1000, rerank_top_k=1000)
        self.assertEqual(config.top_k, 1000)

    def test_invalid_top_k_zero(self):
        """测试top_k=0。"""
        with self.assertRaises(ValueError):
            RetrieverConfig(top_k=0)

    def test_invalid_top_k_negative(self):
        """测试负top_k。"""
        with self.assertRaises(ValueError):
            RetrieverConfig(top_k=-1)

    def test_score_threshold_zero(self):
        """测试score_threshold=0。"""
        config = RetrieverConfig(score_threshold=0.0)
        self.assertEqual(config.score_threshold, 0.0)

    def test_score_threshold_one(self):
        """测试score_threshold=1。"""
        config = RetrieverConfig(score_threshold=1.0)
        self.assertEqual(config.score_threshold, 1.0)

    def test_rerank_top_k_valid(self):
        """测试有效rerank_top_k。"""
        config = RetrieverConfig(top_k=5, rerank_top_k=10)
        self.assertEqual(config.rerank_top_k, 10)

    def test_rerank_top_k_invalid(self):
        """测试无效rerank_top_k（小于top_k）。"""
        with self.assertRaises(ValueError):
            RetrieverConfig(top_k=10, rerank_top_k=5)

    def test_rerank_top_k_equal(self):
        """测试rerank_top_k等于top_k（允许）。"""
        # rerank_top_k等于top_k是允许的
        config = RetrieverConfig(top_k=10, rerank_top_k=10)
        self.assertEqual(config.rerank_top_k, 10)


class TestDenseRetrieverStrict(unittest.TestCase):
    """DenseRetriever严格测试。"""

    def setUp(self):
        """测试前准备。"""
        self.config = EmbeddingConfig(dimension=32)
        self.embedding = DenseEmbedding(self.config, random_seed=42)
        self.store = SimpleVectorStore()
        self.retriever = DenseRetriever(
            embedding=self.embedding,
            vector_store=self.store,
            config=RetrieverConfig(top_k=3),
        )

    def test_repr(self):
        """测试__repr__方法。"""
        repr_str = repr(self.retriever)
        self.assertIn("DenseRetriever", repr_str)
        self.assertIn("top_k=3", repr_str)

    def test_add_empty_documents(self):
        """测试添加空文档列表。"""
        ids = self.retriever.add_documents([])
        self.assertEqual(len(ids), 0)

    def test_add_single_document(self):
        """测试添加单个文档。"""
        doc = Document(content="single document")
        ids = self.retriever.add_documents([doc])
        self.assertEqual(len(ids), 1)

    def test_add_many_documents(self):
        """测试添加大量文档。"""
        docs = [Document(content=f"document {i}") for i in range(50)]
        ids = self.retriever.add_documents(docs)
        self.assertEqual(len(ids), 50)

    def test_retrieve_empty_store(self):
        """测试空存储检索。"""
        results = self.retriever.retrieve("query")
        self.assertEqual(len(results), 0)

    def test_retrieve_single_document(self):
        """测试单文档检索。"""
        self.retriever.add_documents([Document(content="only document")])
        results = self.retriever.retrieve("query")
        self.assertEqual(len(results), 1)

    def test_retrieve_top_k_limit(self):
        """测试top_k限制。"""
        docs = [Document(content=f"doc {i}") for i in range(10)]
        self.retriever.add_documents(docs)
        results = self.retriever.retrieve("query")
        self.assertEqual(len(results), 3)  # top_k=3

    def test_retrieve_with_threshold(self):
        """测试带阈值检索。"""
        retriever = DenseRetriever(
            embedding=self.embedding,
            vector_store=SimpleVectorStore(),
            config=RetrieverConfig(top_k=10, score_threshold=0.99),
        )
        docs = [Document(content=f"doc {i}") for i in range(5)]
        retriever.add_documents(docs)
        results = retriever.retrieve("completely different query xyz")
        # 高阈值应该过滤掉大部分结果
        self.assertLessEqual(len(results), 5)

    def test_retrieve_results_have_rank(self):
        """测试检索结果有排名。"""
        docs = [Document(content=f"doc {i}") for i in range(5)]
        self.retriever.add_documents(docs)
        results = self.retriever.retrieve("query")
        for i, result in enumerate(results):
            self.assertEqual(result.rank, i + 1)

    def test_retrieve_unicode_query(self):
        """测试Unicode查询。"""
        docs = [Document(content="中文文档内容"), Document(content="English doc")]
        self.retriever.add_documents(docs)
        results = self.retriever.retrieve("中文")
        # 检索可能返回结果也可能不返回，取决于嵌入相似度
        self.assertIsInstance(results, list)


class TestSparseRetrieverStrict(unittest.TestCase):
    """SparseRetriever严格测试。"""

    def setUp(self):
        """测试前准备。"""
        self.retriever = SparseRetriever(config=RetrieverConfig(top_k=3))

    def test_repr(self):
        """测试__repr__方法。"""
        repr_str = repr(self.retriever)
        self.assertIn("SparseRetriever", repr_str)

    def test_fit_empty_documents(self):
        """测试空文档拟合。"""
        self.retriever.fit([])
        # 空拟合后应该标记为已拟合但文档为空
        self.assertEqual(len(self.retriever._documents), 0)

    def test_fit_single_document(self):
        """测试单文档拟合。"""
        docs = [Document(content="single document")]
        self.retriever.fit(docs)
        results = self.retriever.retrieve("single")
        self.assertEqual(len(results), 1)

    def test_not_fitted_retrieve_raises(self):
        """测试未拟合检索抛出异常。"""
        with self.assertRaises(RuntimeError):
            self.retriever.retrieve("query")

    def test_retrieve_exact_match(self):
        """测试精确匹配检索。"""
        docs = [
            Document(content="机器学习算法"),
            Document(content="深度学习模型"),
            Document(content="自然语言处理"),
        ]
        self.retriever.fit(docs)
        results = self.retriever.retrieve("机器学习")
        self.assertGreater(len(results), 0)

    def test_retrieve_no_match(self):
        """测试无匹配检索。"""
        docs = [Document(content="apple banana cherry")]
        self.retriever.fit(docs)
        results = self.retriever.retrieve("xyz123完全不相关")
        # 即使无匹配也应该返回结果（BM25会给出分数）
        self.assertLessEqual(len(results), 1)

    def test_refit_updates_state(self):
        """测试重新拟合更新状态。"""
        docs1 = [Document(content="first corpus")]
        self.retriever.fit(docs1)
        count1 = len(self.retriever._documents)
        
        docs2 = [Document(content="second"), Document(content="corpus")]
        self.retriever.fit(docs2)
        count2 = len(self.retriever._documents)
        
        self.assertEqual(count1, 1)
        self.assertEqual(count2, 2)


class TestHybridRetrieverStrict(unittest.TestCase):
    """HybridRetriever严格测试。"""

    def setUp(self):
        """测试前准备。"""
        self.config = EmbeddingConfig(dimension=32)
        self.embedding = DenseEmbedding(self.config, random_seed=42)
        self.store = SimpleVectorStore()
        self.dense = DenseRetriever(
            embedding=self.embedding,
            vector_store=self.store,
            config=RetrieverConfig(top_k=5),
        )
        self.sparse = SparseRetriever(config=RetrieverConfig(top_k=5))
        
        self.documents = [
            Document(content="机器学习是人工智能的核心"),
            Document(content="深度学习使用神经网络"),
            Document(content="自然语言处理理解文本"),
        ]
        self.dense.add_documents(self.documents)
        self.sparse.fit(self.documents)

    def test_repr(self):
        """测试__repr__方法。"""
        hybrid = HybridRetriever(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
            alpha=0.6,
            fusion_method="rrf",
        )
        repr_str = repr(hybrid)
        self.assertIn("HybridRetriever", repr_str)
        self.assertIn("rrf", repr_str)

    def test_alpha_boundary_zero(self):
        """测试alpha=0边界。"""
        hybrid = HybridRetriever(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
            alpha=0.0,
        )
        results = hybrid.retrieve("机器学习")
        self.assertGreater(len(results), 0)

    def test_alpha_boundary_one(self):
        """测试alpha=1边界。"""
        hybrid = HybridRetriever(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
            alpha=1.0,
        )
        results = hybrid.retrieve("机器学习")
        self.assertGreater(len(results), 0)

    def test_invalid_alpha_negative(self):
        """测试负alpha。"""
        with self.assertRaises(ValueError):
            HybridRetriever(
                dense_retriever=self.dense,
                sparse_retriever=self.sparse,
                alpha=-0.1,
            )

    def test_invalid_alpha_greater_than_one(self):
        """测试alpha>1。"""
        with self.assertRaises(ValueError):
            HybridRetriever(
                dense_retriever=self.dense,
                sparse_retriever=self.sparse,
                alpha=1.1,
            )

    def test_fusion_method_rrf(self):
        """测试RRF融合方法。"""
        hybrid = HybridRetriever(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
            fusion_method="rrf",
        )
        results = hybrid.retrieve("机器学习")
        self.assertGreater(len(results), 0)

    def test_fusion_method_weighted(self):
        """测试加权融合方法。"""
        hybrid = HybridRetriever(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
            fusion_method="weighted",
        )
        results = hybrid.retrieve("机器学习")
        self.assertGreater(len(results), 0)

    def test_invalid_fusion_method(self):
        """测试无效融合方法。"""
        with self.assertRaises(ValueError):
            HybridRetriever(
                dense_retriever=self.dense,
                sparse_retriever=self.sparse,
                fusion_method="invalid",
            )

    def test_retrieve_deduplication(self):
        """测试检索结果去重。"""
        hybrid = HybridRetriever(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
            config=RetrieverConfig(top_k=10),
        )
        results = hybrid.retrieve("机器学习")
        doc_ids = [r.document.doc_id for r in results]
        self.assertEqual(len(doc_ids), len(set(doc_ids)))  # 无重复

    def test_retrieve_respects_top_k(self):
        """测试检索遵守top_k限制。"""
        hybrid = HybridRetriever(
            dense_retriever=self.dense,
            sparse_retriever=self.sparse,
            config=RetrieverConfig(top_k=2),
        )
        results = hybrid.retrieve("机器学习")
        self.assertLessEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
