"""
向量嵌入模块严格单元测试

测试覆盖：
    - 边界条件测试
    - 异常处理测试
    - 类型检查测试
    - __repr__ 方法测试
    - 数值稳定性测试
    - 内存安全测试
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import (
    EmbeddingConfig,
    DenseEmbedding,
    SparseEmbedding,
    HybridEmbedding,
)


class TestEmbeddingConfigStrict(unittest.TestCase):
    """EmbeddingConfig严格测试。"""

    def test_boundary_dimension_min(self):
        """测试最小维度边界。"""
        config = EmbeddingConfig(dimension=1)
        self.assertEqual(config.dimension, 1)

    def test_boundary_dimension_large(self):
        """测试大维度。"""
        config = EmbeddingConfig(dimension=10000)
        self.assertEqual(config.dimension, 10000)

    def test_boundary_batch_size_min(self):
        """测试最小批大小。"""
        config = EmbeddingConfig(batch_size=1)
        self.assertEqual(config.batch_size, 1)

    def test_boundary_max_length_one(self):
        """测试max_length=1边界。"""
        config = EmbeddingConfig(max_length=1)
        self.assertEqual(config.max_length, 1)

    def test_invalid_dimension_zero(self):
        """测试维度为0。"""
        with self.assertRaises(ValueError) as ctx:
            EmbeddingConfig(dimension=0)
        self.assertIn("dimension", str(ctx.exception).lower())

    def test_invalid_dimension_negative(self):
        """测试负维度。"""
        with self.assertRaises(ValueError):
            EmbeddingConfig(dimension=-100)

    def test_invalid_batch_size_zero(self):
        """测试批大小为0。"""
        with self.assertRaises(ValueError):
            EmbeddingConfig(batch_size=0)

    def test_invalid_batch_size_negative(self):
        """测试负批大小。"""
        with self.assertRaises(ValueError):
            EmbeddingConfig(batch_size=-1)

    def test_invalid_max_length_negative(self):
        """测试负max_length。"""
        with self.assertRaises(ValueError):
            EmbeddingConfig(max_length=-1)


class TestDenseEmbeddingStrict(unittest.TestCase):
    """DenseEmbedding严格测试。"""

    def setUp(self):
        """测试前准备。"""
        self.config = EmbeddingConfig(dimension=64)
        self.embedding = DenseEmbedding(self.config, random_seed=42)

    def test_repr(self):
        """测试__repr__方法。"""
        repr_str = repr(self.embedding)
        self.assertIn("DenseEmbedding", repr_str)
        self.assertIn("64", repr_str)

    def test_embed_single_char(self):
        """测试单字符嵌入。"""
        vector = self.embedding.embed_text("a")
        self.assertEqual(vector.shape, (64,))
        self.assertFalse(np.any(np.isnan(vector)))

    def test_embed_unicode(self):
        """测试Unicode字符嵌入。"""
        vector = self.embedding.embed_text("你好世界🌍")
        self.assertEqual(vector.shape, (64,))
        self.assertFalse(np.any(np.isnan(vector)))

    def test_embed_special_chars(self):
        """测试特殊字符嵌入。"""
        vector = self.embedding.embed_text("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        self.assertEqual(vector.shape, (64,))

    def test_embed_whitespace_only(self):
        """测试纯空白字符。"""
        vector = self.embedding.embed_text("   \t\n  ")
        self.assertEqual(vector.shape, (64,))

    def test_embed_very_long_text(self):
        """测试超长文本。"""
        long_text = "word " * 10000
        vector = self.embedding.embed_text(long_text)
        self.assertEqual(vector.shape, (64,))
        self.assertFalse(np.any(np.isnan(vector)))
        self.assertFalse(np.any(np.isinf(vector)))

    def test_embed_texts_empty_list(self):
        """测试空列表嵌入。"""
        vectors = self.embedding.embed_texts([])
        self.assertEqual(vectors.shape[0], 0)

    def test_embed_texts_single_item(self):
        """测试单元素列表。"""
        vectors = self.embedding.embed_texts(["hello"])
        self.assertEqual(vectors.shape, (1, 64))

    def test_embed_texts_large_batch(self):
        """测试大批量嵌入。"""
        texts = [f"text {i}" for i in range(100)]
        vectors = self.embedding.embed_texts(texts)
        self.assertEqual(vectors.shape, (100, 64))

    def test_normalization_unit_vector(self):
        """测试归一化后为单位向量。"""
        for text in ["hello", "world", "test", "机器学习"]:
            vector = self.embedding.embed_text(text)
            norm = np.linalg.norm(vector)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_dtype_float32(self):
        """测试输出类型为float32。"""
        vector = self.embedding.embed_text("test")
        self.assertEqual(vector.dtype, np.float32)

    def test_deterministic_output(self):
        """测试确定性输出。"""
        emb1 = DenseEmbedding(self.config, random_seed=123)
        emb2 = DenseEmbedding(self.config, random_seed=123)
        v1 = emb1.embed_text("same text")
        v2 = emb2.embed_text("same text")
        np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_different_output(self):
        """测试不同种子产生不同输出。"""
        emb1 = DenseEmbedding(self.config, random_seed=1)
        emb2 = DenseEmbedding(self.config, random_seed=2)
        v1 = emb1.embed_text("same text")
        v2 = emb2.embed_text("same text")
        self.assertFalse(np.allclose(v1, v2))

    def test_embed_query_same_as_embed_text(self):
        """测试embed_query与embed_text一致。"""
        v1 = self.embedding.embed_text("query")
        v2 = self.embedding.embed_query("query")
        np.testing.assert_array_equal(v1, v2)


class TestSparseEmbeddingStrict(unittest.TestCase):
    """SparseEmbedding严格测试。"""

    def setUp(self):
        """测试前准备。"""
        self.embedding = SparseEmbedding()
        self.documents = [
            "机器学习是人工智能的一个分支",
            "深度学习是机器学习的子领域",
            "自然语言处理用于理解文本",
        ]
        self.embedding.fit(self.documents)

    def test_repr(self):
        """测试__repr__方法。"""
        repr_str = repr(self.embedding)
        self.assertIn("SparseEmbedding", repr_str)
        self.assertIn("vocab_size", repr_str)

    def test_fit_empty_corpus(self):
        """测试空语料库拟合。"""
        emb = SparseEmbedding()
        emb.fit([])
        self.assertEqual(emb.vocab_size, 0)

    def test_fit_single_document(self):
        """测试单文档拟合。"""
        emb = SparseEmbedding()
        emb.fit(["单个文档"])
        self.assertTrue(emb._fitted)

    def test_fit_duplicate_documents(self):
        """测试重复文档拟合。"""
        emb = SparseEmbedding()
        emb.fit(["same", "same", "same"])
        self.assertTrue(emb._fitted)

    def test_embed_unknown_words(self):
        """测试未知词嵌入。"""
        vector = self.embedding.embed_text("完全未知的词汇xyz123")
        self.assertEqual(len(vector), self.embedding.vocab_size)

    def test_embed_empty_after_fit(self):
        """测试拟合后嵌入空文本。"""
        vector = self.embedding.embed_text("")
        self.assertEqual(len(vector), self.embedding.vocab_size)
        self.assertTrue(np.allclose(vector, 0))

    def test_bm25_k1_boundary(self):
        """测试BM25 k1边界值。"""
        emb = SparseEmbedding(k1=0.0)
        emb.fit(self.documents)
        vector = emb.embed_text("机器学习")
        self.assertFalse(np.any(np.isnan(vector)))

    def test_bm25_b_boundary_zero(self):
        """测试BM25 b=0边界。"""
        emb = SparseEmbedding(b=0.0)
        emb.fit(self.documents)
        vector = emb.embed_text("机器学习")
        self.assertFalse(np.any(np.isnan(vector)))

    def test_bm25_b_boundary_one(self):
        """测试BM25 b=1边界。"""
        emb = SparseEmbedding(b=1.0)
        emb.fit(self.documents)
        vector = emb.embed_text("机器学习")
        self.assertFalse(np.any(np.isnan(vector)))

    def test_idf_values_positive(self):
        """测试IDF值为正。"""
        for word, idf in self.embedding._idf.items():
            self.assertGreaterEqual(idf, 0)

    def test_vocab_consistency(self):
        """测试词汇表一致性。"""
        self.assertEqual(len(self.embedding._vocab), self.embedding.vocab_size)

    def test_not_fitted_embed_text(self):
        """测试未拟合时embed_text。"""
        emb = SparseEmbedding()
        with self.assertRaises(RuntimeError):
            emb.embed_text("test")

    def test_not_fitted_embed_texts(self):
        """测试未拟合时embed_texts。"""
        emb = SparseEmbedding()
        with self.assertRaises(RuntimeError):
            emb.embed_texts(["test"])

    def test_refit_clears_state(self):
        """测试重新拟合清除状态。"""
        old_vocab_size = self.embedding.vocab_size
        self.embedding.fit(["完全不同的新文档"])
        self.assertNotEqual(self.embedding.vocab_size, old_vocab_size)


class TestHybridEmbeddingStrict(unittest.TestCase):
    """HybridEmbedding严格测试。"""

    def setUp(self):
        """测试前准备。"""
        self.dense = DenseEmbedding(EmbeddingConfig(dimension=32), random_seed=42)
        self.sparse = SparseEmbedding()
        self.sparse.fit(["hello world", "test document", "sample text"])
        self.hybrid = HybridEmbedding(self.dense, self.sparse, alpha=0.5)

    def test_repr(self):
        """测试__repr__方法。"""
        repr_str = repr(self.hybrid)
        self.assertIn("HybridEmbedding", repr_str)
        self.assertIn("0.5", repr_str)

    def test_alpha_boundary_zero(self):
        """测试alpha=0边界。"""
        hybrid = HybridEmbedding(self.dense, self.sparse, alpha=0.0)
        vector = hybrid.embed_text("test")
        self.assertFalse(np.any(np.isnan(vector)))

    def test_alpha_boundary_one(self):
        """测试alpha=1边界。"""
        hybrid = HybridEmbedding(self.dense, self.sparse, alpha=1.0)
        vector = hybrid.embed_text("test")
        self.assertFalse(np.any(np.isnan(vector)))

    def test_invalid_alpha_negative(self):
        """测试负alpha。"""
        with self.assertRaises(ValueError):
            HybridEmbedding(self.dense, self.sparse, alpha=-0.001)

    def test_invalid_alpha_greater_than_one(self):
        """测试alpha>1。"""
        with self.assertRaises(ValueError):
            HybridEmbedding(self.dense, self.sparse, alpha=1.001)

    def test_output_dimension(self):
        """测试输出维度正确。"""
        vector = self.hybrid.embed_text("test")
        expected_dim = 32 + self.sparse.vocab_size
        self.assertEqual(len(vector), expected_dim)

    def test_embed_texts_consistency(self):
        """测试批量嵌入一致性。"""
        texts = ["hello", "world"]
        batch_vectors = self.hybrid.embed_texts(texts)
        single_vectors = [self.hybrid.embed_text(t) for t in texts]
        for i, text in enumerate(texts):
            np.testing.assert_array_almost_equal(batch_vectors[i], single_vectors[i])

    def test_dense_sparse_separation(self):
        """测试稠密稀疏分离获取。"""
        dense_vec = self.hybrid.get_dense_embedding("test")
        sparse_vec = self.hybrid.get_sparse_embedding("test")
        self.assertEqual(len(dense_vec), 32)
        self.assertEqual(len(sparse_vec), self.sparse.vocab_size)

    def test_numerical_stability(self):
        """测试数值稳定性。"""
        for _ in range(100):
            vector = self.hybrid.embed_text("random test text")
            self.assertFalse(np.any(np.isnan(vector)))
            self.assertFalse(np.any(np.isinf(vector)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
