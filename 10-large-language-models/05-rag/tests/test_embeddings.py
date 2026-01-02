"""
向量嵌入模块单元测试

测试覆盖：
    - EmbeddingConfig: 配置验证
    - DenseEmbedding: 稠密嵌入
    - SparseEmbedding: BM25稀疏嵌入
    - HybridEmbedding: 混合嵌入
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import (
    EmbeddingConfig,
    BaseEmbedding,
    DenseEmbedding,
    SparseEmbedding,
    HybridEmbedding,
)


class TestEmbeddingConfig(unittest.TestCase):
    """EmbeddingConfig测试类。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = EmbeddingConfig()
        self.assertEqual(config.dimension, 1536)
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.normalize)

    def test_custom_config(self):
        """测试自定义配置。"""
        config = EmbeddingConfig(dimension=384, batch_size=64, normalize=False)
        self.assertEqual(config.dimension, 384)
        self.assertEqual(config.batch_size, 64)
        self.assertFalse(config.normalize)

    def test_invalid_dimension(self):
        """测试无效维度。"""
        with self.assertRaises(ValueError):
            EmbeddingConfig(dimension=0)
        with self.assertRaises(ValueError):
            EmbeddingConfig(dimension=-1)

    def test_invalid_batch_size(self):
        """测试无效批大小。"""
        with self.assertRaises(ValueError):
            EmbeddingConfig(batch_size=0)

    def test_invalid_max_length(self):
        """测试无效最大长度。"""
        with self.assertRaises(ValueError):
            EmbeddingConfig(max_length=-1)


class TestDenseEmbedding(unittest.TestCase):
    """DenseEmbedding测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.config = EmbeddingConfig(dimension=128)
        self.embedding = DenseEmbedding(self.config, random_seed=42)

    def test_embed_text(self):
        """测试单文本嵌入。"""
        vector = self.embedding.embed_text("Hello world")
        self.assertEqual(vector.shape, (128,))
        self.assertEqual(vector.dtype, np.float32)

    def test_embed_texts(self):
        """测试多文本嵌入。"""
        texts = ["Hello", "World", "Test"]
        vectors = self.embedding.embed_texts(texts)
        self.assertEqual(vectors.shape, (3, 128))

    def test_empty_text(self):
        """测试空文本。"""
        vector = self.embedding.embed_text("")
        self.assertEqual(vector.shape, (128,))
        self.assertTrue(np.allclose(vector, 0))

    def test_normalization(self):
        """测试归一化。"""
        vector = self.embedding.embed_text("Test normalization")
        norm = np.linalg.norm(vector)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_no_normalization(self):
        """测试不归一化。"""
        config = EmbeddingConfig(dimension=128, normalize=False)
        embedding = DenseEmbedding(config, random_seed=42)
        vector = embedding.embed_text("Test no normalization")
        norm = np.linalg.norm(vector)
        self.assertNotAlmostEqual(norm, 1.0, places=1)

    def test_reproducibility(self):
        """测试可重复性。"""
        embedding1 = DenseEmbedding(self.config, random_seed=42)
        embedding2 = DenseEmbedding(self.config, random_seed=42)
        v1 = embedding1.embed_text("Test")
        v2 = embedding2.embed_text("Test")
        np.testing.assert_array_almost_equal(v1, v2)

    def test_embed_query(self):
        """测试查询嵌入。"""
        vector = self.embedding.embed_query("Query text")
        self.assertEqual(vector.shape, (128,))


class TestSparseEmbedding(unittest.TestCase):
    """SparseEmbedding测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.embedding = SparseEmbedding()
        self.documents = [
            "机器学习是人工智能的一个分支",
            "深度学习是机器学习的子领域",
            "自然语言处理用于理解文本",
        ]
        self.embedding.fit(self.documents)

    def test_fit(self):
        """测试拟合。"""
        self.assertTrue(self.embedding._fitted)
        self.assertGreater(self.embedding.vocab_size, 0)

    def test_embed_text(self):
        """测试文本嵌入。"""
        vector = self.embedding.embed_text("机器学习")
        self.assertEqual(len(vector), self.embedding.vocab_size)

    def test_embed_texts(self):
        """测试多文本嵌入。"""
        vectors = self.embedding.embed_texts(["机器学习", "深度学习"])
        self.assertEqual(vectors.shape[0], 2)

    def test_not_fitted_error(self):
        """测试未拟合错误。"""
        embedding = SparseEmbedding()
        with self.assertRaises(RuntimeError):
            embedding.embed_text("Test")

    def test_bm25_parameters(self):
        """测试BM25参数。"""
        embedding = SparseEmbedding(k1=2.0, b=0.5)
        embedding.fit(self.documents)
        self.assertEqual(embedding.k1, 2.0)
        self.assertEqual(embedding.b, 0.5)

    def test_idf_calculation(self):
        """测试IDF计算。"""
        self.assertGreater(len(self.embedding._idf), 0)


class TestHybridEmbedding(unittest.TestCase):
    """HybridEmbedding测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.dense = DenseEmbedding(EmbeddingConfig(dimension=64), random_seed=42)
        self.sparse = SparseEmbedding()
        self.sparse.fit(["Hello world", "Test document"])
        self.hybrid = HybridEmbedding(self.dense, self.sparse, alpha=0.7)

    def test_embed_text(self):
        """测试混合嵌入。"""
        vector = self.hybrid.embed_text("Hello world")
        expected_dim = 64 + self.sparse.vocab_size
        self.assertEqual(len(vector), expected_dim)

    def test_embed_texts(self):
        """测试多文本混合嵌入。"""
        vectors = self.hybrid.embed_texts(["Hello", "World"])
        self.assertEqual(vectors.shape[0], 2)

    def test_invalid_alpha(self):
        """测试无效alpha。"""
        with self.assertRaises(ValueError):
            HybridEmbedding(self.dense, self.sparse, alpha=1.5)
        with self.assertRaises(ValueError):
            HybridEmbedding(self.dense, self.sparse, alpha=-0.1)

    def test_get_dense_embedding(self):
        """测试获取稠密嵌入。"""
        vector = self.hybrid.get_dense_embedding("Test")
        self.assertEqual(len(vector), 64)

    def test_get_sparse_embedding(self):
        """测试获取稀疏嵌入。"""
        vector = self.hybrid.get_sparse_embedding("Test")
        self.assertEqual(len(vector), self.sparse.vocab_size)


if __name__ == "__main__":
    unittest.main()
