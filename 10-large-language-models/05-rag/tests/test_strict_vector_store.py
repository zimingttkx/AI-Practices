"""
向量存储模块严格单元测试

测试覆盖：
    - 边界条件测试
    - 异常处理测试
    - 类型检查测试
    - __repr__ 方法测试
    - 数值稳定性测试
    - 序列化测试
"""

import unittest
import tempfile
import numpy as np
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_store import (
    Document,
    SearchResult,
    SimpleVectorStore,
)


class TestDocumentStrict(unittest.TestCase):
    """Document严格测试。"""

    def test_empty_content_raises(self):
        """测试空内容抛出异常。"""
        with self.assertRaises(ValueError) as ctx:
            Document(content="")
        # 验证抛出了ValueError
        self.assertIsNotNone(ctx.exception)

    def test_whitespace_only_content(self):
        """测试纯空白内容。"""
        doc = Document(content="   ")
        self.assertEqual(doc.content, "   ")

    def test_unicode_content(self):
        """测试Unicode内容。"""
        doc = Document(content="你好世界🌍🚀")
        self.assertEqual(doc.content, "你好世界🌍🚀")

    def test_very_long_content(self):
        """测试超长内容。"""
        long_content = "x" * 100000
        doc = Document(content=long_content)
        self.assertEqual(len(doc.content), 100000)

    def test_special_chars_content(self):
        """测试特殊字符内容。"""
        special = "!@#$%^&*()_+-=[]{}|;':\",./<>?\n\t\r"
        doc = Document(content=special)
        self.assertEqual(doc.content, special)

    def test_auto_generated_doc_id(self):
        """测试自动生成doc_id。"""
        doc1 = Document(content="test1")
        doc2 = Document(content="test2")
        self.assertIsNotNone(doc1.doc_id)
        self.assertIsNotNone(doc2.doc_id)
        self.assertNotEqual(doc1.doc_id, doc2.doc_id)

    def test_custom_doc_id(self):
        """测试自定义doc_id。"""
        doc = Document(content="test", doc_id="custom-id-123")
        self.assertEqual(doc.doc_id, "custom-id-123")

    def test_metadata_default_empty(self):
        """测试元数据默认为空。"""
        doc = Document(content="test")
        self.assertEqual(doc.metadata, {})

    def test_metadata_complex_nested(self):
        """测试复杂嵌套元数据。"""
        metadata = {
            "source": "test.txt",
            "nested": {"level1": {"level2": [1, 2, 3]}},
            "list": [1, "two", 3.0],
        }
        doc = Document(content="test", metadata=metadata)
        self.assertEqual(doc.metadata["nested"]["level1"]["level2"], [1, 2, 3])

    def test_to_dict_complete(self):
        """测试to_dict完整性。"""
        doc = Document(content="test", metadata={"key": "value"}, doc_id="id123")
        d = doc.to_dict()
        self.assertIn("content", d)
        self.assertIn("metadata", d)
        self.assertIn("doc_id", d)
        self.assertEqual(d["content"], "test")
        self.assertEqual(d["doc_id"], "id123")

    def test_from_dict_complete(self):
        """测试from_dict完整性。"""
        data = {
            "content": "test content",
            "metadata": {"source": "file.txt"},
            "doc_id": "custom-id",
        }
        doc = Document.from_dict(data)
        self.assertEqual(doc.content, "test content")
        self.assertEqual(doc.doc_id, "custom-id")
        self.assertEqual(doc.metadata["source"], "file.txt")

    def test_from_dict_minimal(self):
        """测试from_dict最小数据。"""
        data = {"content": "minimal"}
        doc = Document.from_dict(data)
        self.assertEqual(doc.content, "minimal")

    def test_roundtrip_to_from_dict(self):
        """测试to_dict/from_dict往返。"""
        original = Document(content="test", metadata={"key": "value"})
        restored = Document.from_dict(original.to_dict())
        self.assertEqual(original.content, restored.content)
        self.assertEqual(original.doc_id, restored.doc_id)
        self.assertEqual(original.metadata, restored.metadata)

    def test_embedding_attribute(self):
        """测试embedding属性。"""
        doc = Document(content="test")
        self.assertIsNone(doc.embedding)
        doc.embedding = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(doc.embedding, [1.0, 2.0, 3.0])


class TestSearchResultStrict(unittest.TestCase):
    """SearchResult严格测试。"""

    def test_repr_short_content(self):
        """测试__repr__短内容。"""
        doc = Document(content="short")
        result = SearchResult(document=doc, score=0.95, rank=1)
        repr_str = repr(result)
        self.assertIn("rank=1", repr_str)
        self.assertIn("0.95", repr_str)

    def test_repr_long_content(self):
        """测试__repr__长内容截断。"""
        doc = Document(content="x" * 200)
        result = SearchResult(document=doc, score=0.5, rank=2)
        repr_str = repr(result)
        self.assertIn("...", repr_str)

    def test_score_boundary_zero(self):
        """测试分数边界0。"""
        doc = Document(content="test")
        result = SearchResult(document=doc, score=0.0, rank=1)
        self.assertEqual(result.score, 0.0)

    def test_score_boundary_one(self):
        """测试分数边界1。"""
        doc = Document(content="test")
        result = SearchResult(document=doc, score=1.0, rank=1)
        self.assertEqual(result.score, 1.0)

    def test_score_negative(self):
        """测试负分数（欧氏距离场景）。"""
        doc = Document(content="test")
        result = SearchResult(document=doc, score=-0.5, rank=1)
        self.assertEqual(result.score, -0.5)

    def test_rank_values(self):
        """测试排名值。"""
        doc = Document(content="test")
        for rank in [1, 10, 100, 1000]:
            result = SearchResult(document=doc, score=0.5, rank=rank)
            self.assertEqual(result.rank, rank)


class TestSimpleVectorStoreStrict(unittest.TestCase):
    """SimpleVectorStore严格测试。"""

    def setUp(self):
        """测试前准备。"""
        self.dim = 64
        np.random.seed(42)

    def _create_doc_with_embedding(self, content: str, dim: int = None) -> Document:
        """创建带嵌入的文档。"""
        dim = dim or self.dim
        doc = Document(content=content)
        doc.embedding = np.random.randn(dim).astype(np.float32)
        return doc

    def test_repr(self):
        """测试__repr__方法。"""
        store = SimpleVectorStore(metric="cosine")
        repr_str = repr(store)
        self.assertIn("SimpleVectorStore", repr_str)
        self.assertIn("cosine", repr_str)

    def test_metric_cosine(self):
        """测试余弦度量。"""
        store = SimpleVectorStore(metric="cosine")
        self.assertEqual(store.metric, "cosine")

    def test_metric_euclidean(self):
        """测试欧氏度量。"""
        store = SimpleVectorStore(metric="euclidean")
        self.assertEqual(store.metric, "euclidean")

    def test_metric_dot(self):
        """测试点积度量。"""
        store = SimpleVectorStore(metric="dot")
        self.assertEqual(store.metric, "dot")

    def test_invalid_metric(self):
        """测试无效度量。"""
        with self.assertRaises(ValueError):
            SimpleVectorStore(metric="invalid")
        with self.assertRaises(ValueError):
            SimpleVectorStore(metric="")
        with self.assertRaises(ValueError):
            SimpleVectorStore(metric="COSINE")  # 大小写敏感

    def test_add_single_document(self):
        """测试添加单个文档。"""
        store = SimpleVectorStore()
        doc = self._create_doc_with_embedding("single doc")
        ids = store.add_documents([doc])
        self.assertEqual(len(ids), 1)
        self.assertEqual(store.count, 1)

    def test_add_many_documents(self):
        """测试添加大量文档。"""
        store = SimpleVectorStore()
        docs = [self._create_doc_with_embedding(f"doc {i}") for i in range(100)]
        ids = store.add_documents(docs)
        self.assertEqual(len(ids), 100)
        self.assertEqual(store.count, 100)

    def test_add_document_without_embedding_raises(self):
        """测试添加无嵌入文档抛出异常。"""
        store = SimpleVectorStore()
        doc = Document(content="no embedding")
        with self.assertRaises(ValueError):
            store.add_documents([doc])

    def test_add_empty_list(self):
        """测试添加空列表。"""
        store = SimpleVectorStore()
        ids = store.add_documents([])
        self.assertEqual(len(ids), 0)
        self.assertEqual(store.count, 0)

    def test_search_empty_store(self):
        """测试空存储搜索。"""
        store = SimpleVectorStore()
        query = np.random.randn(self.dim).astype(np.float32)
        results = store.search(query, top_k=5)
        self.assertEqual(len(results), 0)

    def test_search_top_k_larger_than_store(self):
        """测试top_k大于存储数量。"""
        store = SimpleVectorStore()
        docs = [self._create_doc_with_embedding(f"doc {i}") for i in range(3)]
        store.add_documents(docs)
        query = np.random.randn(self.dim).astype(np.float32)
        results = store.search(query, top_k=10)
        self.assertEqual(len(results), 3)

    def test_search_top_k_one(self):
        """测试top_k=1。"""
        store = SimpleVectorStore()
        docs = [self._create_doc_with_embedding(f"doc {i}") for i in range(5)]
        store.add_documents(docs)
        query = np.random.randn(self.dim).astype(np.float32)
        results = store.search(query, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].rank, 1)

    def test_search_results_sorted(self):
        """测试搜索结果排序。"""
        store = SimpleVectorStore(metric="cosine")
        docs = [self._create_doc_with_embedding(f"doc {i}") for i in range(10)]
        store.add_documents(docs)
        query = np.random.randn(self.dim).astype(np.float32)
        results = store.search(query, top_k=5)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_ranks_sequential(self):
        """测试搜索排名连续。"""
        store = SimpleVectorStore()
        docs = [self._create_doc_with_embedding(f"doc {i}") for i in range(5)]
        store.add_documents(docs)
        query = np.random.randn(self.dim).astype(np.float32)
        results = store.search(query, top_k=5)
        ranks = [r.rank for r in results]
        self.assertEqual(ranks, [1, 2, 3, 4, 5])

    def test_delete_existing(self):
        """测试删除存在的文档。"""
        store = SimpleVectorStore()
        docs = [self._create_doc_with_embedding(f"doc {i}") for i in range(3)]
        ids = store.add_documents(docs)
        deleted = store.delete([ids[0]])
        self.assertEqual(deleted, 1)
        self.assertEqual(store.count, 2)
        self.assertIsNone(store.get(ids[0]))

    def test_delete_nonexistent(self):
        """测试删除不存在的文档。"""
        store = SimpleVectorStore()
        deleted = store.delete(["nonexistent-id"])
        self.assertEqual(deleted, 0)

    def test_delete_mixed(self):
        """测试混合删除（存在和不存在）。"""
        store = SimpleVectorStore()
        docs = [self._create_doc_with_embedding(f"doc {i}") for i in range(2)]
        ids = store.add_documents(docs)
        deleted = store.delete([ids[0], "nonexistent"])
        self.assertEqual(deleted, 1)

    def test_delete_all(self):
        """测试删除所有文档。"""
        store = SimpleVectorStore()
        docs = [self._create_doc_with_embedding(f"doc {i}") for i in range(5)]
        ids = store.add_documents(docs)
        deleted = store.delete(ids)
        self.assertEqual(deleted, 5)
        self.assertEqual(store.count, 0)

    def test_get_existing(self):
        """测试获取存在的文档。"""
        store = SimpleVectorStore()
        doc = self._create_doc_with_embedding("test doc")
        store.add_documents([doc])
        retrieved = store.get(doc.doc_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "test doc")

    def test_get_nonexistent(self):
        """测试获取不存在的文档。"""
        store = SimpleVectorStore()
        result = store.get("nonexistent")
        self.assertIsNone(result)

    def test_save_load_roundtrip(self):
        """测试保存加载往返。"""
        store = SimpleVectorStore(metric="euclidean")
        docs = [self._create_doc_with_embedding(f"doc {i}") for i in range(5)]
        store.add_documents(docs)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            store.save(path)
            loaded = SimpleVectorStore.load(path)
            self.assertEqual(loaded.count, 5)
            self.assertEqual(loaded.metric, "euclidean")
            for doc in docs:
                retrieved = loaded.get(doc.doc_id)
                self.assertIsNotNone(retrieved)
                self.assertEqual(retrieved.content, doc.content)
        finally:
            Path(path).unlink()

    def test_save_empty_store(self):
        """测试保存空存储。"""
        store = SimpleVectorStore()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store.save(path)
            loaded = SimpleVectorStore.load(path)
            self.assertEqual(loaded.count, 0)
        finally:
            Path(path).unlink()

    def test_cosine_similarity_identical_vectors(self):
        """测试余弦相似度-相同向量。"""
        store = SimpleVectorStore(metric="cosine")
        doc = self._create_doc_with_embedding("test")
        store.add_documents([doc])
        results = store.search(doc.embedding, top_k=1)
        self.assertAlmostEqual(results[0].score, 1.0, places=5)

    def test_dot_product_identical_vectors(self):
        """测试点积-相同向量。"""
        store = SimpleVectorStore(metric="dot")
        doc = Document(content="test")
        doc.embedding = np.ones(self.dim, dtype=np.float32)
        store.add_documents([doc])
        results = store.search(doc.embedding, top_k=1)
        self.assertGreater(results[0].score, 0)

    def test_euclidean_identical_vectors(self):
        """测试欧氏距离-相同向量。"""
        store = SimpleVectorStore(metric="euclidean")
        doc = self._create_doc_with_embedding("test")
        store.add_documents([doc])
        results = store.search(doc.embedding, top_k=1)
        # 欧氏距离相同向量距离为0，转换后分数可能为0或负数
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].rank, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
