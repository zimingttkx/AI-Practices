"""
向量存储模块单元测试

测试覆盖：
    - Document: 文档数据结构
    - SearchResult: 搜索结果
    - SimpleVectorStore: 简单向量存储
"""

import unittest
import tempfile
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_store import (
    Document,
    SearchResult,
    SimpleVectorStore,
)


class TestDocument(unittest.TestCase):
    """Document测试类。"""

    def test_create_document(self):
        """测试创建文档。"""
        doc = Document(content="Test content")
        self.assertEqual(doc.content, "Test content")
        self.assertIsNotNone(doc.doc_id)
        self.assertEqual(doc.metadata, {})

    def test_document_with_metadata(self):
        """测试带元数据的文档。"""
        doc = Document(content="Test", metadata={"source": "test.txt"})
        self.assertEqual(doc.metadata["source"], "test.txt")

    def test_empty_content_error(self):
        """测试空内容错误。"""
        with self.assertRaises(ValueError):
            Document(content="")

    def test_to_dict(self):
        """测试转换为字典。"""
        doc = Document(content="Test", metadata={"key": "value"})
        d = doc.to_dict()
        self.assertEqual(d["content"], "Test")
        self.assertEqual(d["metadata"]["key"], "value")

    def test_from_dict(self):
        """测试从字典创建。"""
        data = {"content": "Test", "metadata": {"key": "value"}}
        doc = Document.from_dict(data)
        self.assertEqual(doc.content, "Test")
        self.assertEqual(doc.metadata["key"], "value")


class TestSearchResult(unittest.TestCase):
    """SearchResult测试类。"""

    def test_create_result(self):
        """测试创建搜索结果。"""
        doc = Document(content="Test content")
        result = SearchResult(document=doc, score=0.95, rank=1)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.rank, 1)

    def test_repr(self):
        """测试字符串表示。"""
        doc = Document(content="Short content")
        result = SearchResult(document=doc, score=0.9, rank=1)
        repr_str = repr(result)
        self.assertIn("rank=1", repr_str)
        self.assertIn("0.9", repr_str)


class TestSimpleVectorStore(unittest.TestCase):
    """SimpleVectorStore测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.store = SimpleVectorStore(metric="cosine")
        self.dim = 64
        np.random.seed(42)

    def _create_doc_with_embedding(self, content: str) -> Document:
        """创建带嵌入的文档。"""
        doc = Document(content=content)
        doc.embedding = np.random.randn(self.dim).astype(np.float32)
        return doc

    def test_add_documents(self):
        """测试添加文档。"""
        docs = [self._create_doc_with_embedding(f"Doc {i}") for i in range(3)]
        ids = self.store.add_documents(docs)
        self.assertEqual(len(ids), 3)
        self.assertEqual(self.store.count, 3)

    def test_add_document_without_embedding(self):
        """测试添加无嵌入文档。"""
        doc = Document(content="No embedding")
        with self.assertRaises(ValueError):
            self.store.add_documents([doc])

    def test_search_cosine(self):
        """测试余弦相似度搜索。"""
        docs = [self._create_doc_with_embedding(f"Doc {i}") for i in range(5)]
        self.store.add_documents(docs)
        query = np.random.randn(self.dim).astype(np.float32)
        results = self.store.search(query, top_k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].rank, 1)

    def test_search_euclidean(self):
        """测试欧氏距离搜索。"""
        store = SimpleVectorStore(metric="euclidean")
        docs = [self._create_doc_with_embedding(f"Doc {i}") for i in range(5)]
        store.add_documents(docs)
        query = np.random.randn(self.dim).astype(np.float32)
        results = store.search(query, top_k=3)
        self.assertEqual(len(results), 3)

    def test_search_dot(self):
        """测试点积搜索。"""
        store = SimpleVectorStore(metric="dot")
        docs = [self._create_doc_with_embedding(f"Doc {i}") for i in range(5)]
        store.add_documents(docs)
        query = np.random.randn(self.dim).astype(np.float32)
        results = store.search(query, top_k=3)
        self.assertEqual(len(results), 3)

    def test_invalid_metric(self):
        """测试无效度量。"""
        with self.assertRaises(ValueError):
            SimpleVectorStore(metric="invalid")

    def test_delete(self):
        """测试删除文档。"""
        docs = [self._create_doc_with_embedding(f"Doc {i}") for i in range(3)]
        ids = self.store.add_documents(docs)
        deleted = self.store.delete([ids[0]])
        self.assertEqual(deleted, 1)
        self.assertEqual(self.store.count, 2)

    def test_get(self):
        """测试获取文档。"""
        doc = self._create_doc_with_embedding("Test doc")
        self.store.add_documents([doc])
        retrieved = self.store.get(doc.doc_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, "Test doc")

    def test_get_nonexistent(self):
        """测试获取不存在的文档。"""
        result = self.store.get("nonexistent-id")
        self.assertIsNone(result)

    def test_search_empty_store(self):
        """测试空存储搜索。"""
        query = np.random.randn(self.dim).astype(np.float32)
        results = self.store.search(query, top_k=3)
        self.assertEqual(len(results), 0)

    def test_save_and_load(self):
        """测试保存和加载。"""
        docs = [self._create_doc_with_embedding(f"Doc {i}") for i in range(3)]
        self.store.add_documents(docs)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        self.store.save(path)
        loaded_store = SimpleVectorStore.load(path)

        self.assertEqual(loaded_store.count, 3)
        self.assertEqual(loaded_store.metric, "cosine")

        Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
