"""
RAG流水线模块单元测试

测试覆盖：
    - RAGConfig: 配置验证
    - RAGResponse: 响应数据结构
    - TextSplitter: 文本分割器
    - RAGPipeline: 基础RAG流水线
    - AdvancedRAGPipeline: 高级RAG流水线
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from vector_store import Document
from rag_pipeline import (
    RAGConfig,
    RAGResponse,
    RecursiveTextSplitter,
    RAGPipeline,
    AdvancedRAGPipeline,
)


class TestRAGConfig(unittest.TestCase):
    """RAGConfig测试类。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = RAGConfig()
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 50)
        self.assertEqual(config.top_k, 5)

    def test_custom_config(self):
        """测试自定义配置。"""
        config = RAGConfig(chunk_size=256, chunk_overlap=25, top_k=3)
        self.assertEqual(config.chunk_size, 256)
        self.assertEqual(config.chunk_overlap, 25)
        self.assertEqual(config.top_k, 3)

    def test_invalid_chunk_size(self):
        """测试无效chunk_size。"""
        with self.assertRaises(ValueError):
            RAGConfig(chunk_size=0)

    def test_invalid_chunk_overlap(self):
        """测试无效chunk_overlap。"""
        with self.assertRaises(ValueError):
            RAGConfig(chunk_overlap=-1)
        with self.assertRaises(ValueError):
            RAGConfig(chunk_size=100, chunk_overlap=100)

    def test_invalid_top_k(self):
        """测试无效top_k。"""
        with self.assertRaises(ValueError):
            RAGConfig(top_k=0)


class TestRAGResponse(unittest.TestCase):
    """RAGResponse测试类。"""

    def test_create_response(self):
        """测试创建响应。"""
        doc = Document(content="Source content")
        response = RAGResponse(
            answer="Test answer",
            source_documents=[doc],
            context="Test context",
            prompt="Test prompt",
        )
        self.assertEqual(response.answer, "Test answer")
        self.assertEqual(len(response.source_documents), 1)

    def test_repr(self):
        """测试字符串表示。"""
        response = RAGResponse(
            answer="Short answer",
            source_documents=[],
            context="",
            prompt="",
        )
        repr_str = repr(response)
        self.assertIn("Short answer", repr_str)
        self.assertIn("sources=0", repr_str)


class TestRecursiveTextSplitter(unittest.TestCase):
    """RecursiveTextSplitter测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)

    def test_split_short_text(self):
        """测试分割短文本。"""
        text = "This is a short text."
        chunks = self.splitter.split_text(text)
        self.assertEqual(len(chunks), 1)

    def test_split_long_text(self):
        """测试分割长文本。"""
        text = "这是一段很长的文本。" * 50
        chunks = self.splitter.split_text(text)
        self.assertGreater(len(chunks), 1)

    def test_split_with_paragraphs(self):
        """测试按段落分割。"""
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = self.splitter.split_text(text)
        self.assertGreaterEqual(len(chunks), 1)

    def test_split_documents(self):
        """测试分割文档列表。"""
        docs = [
            Document(content="短文档内容"),
            Document(content="这是一段较长的文档内容。" * 20),
        ]
        chunks = self.splitter.split_documents(docs)
        self.assertGreater(len(chunks), 2)

    def test_chunk_metadata(self):
        """测试分块元数据。"""
        doc = Document(content="内容" * 100, metadata={"source": "test.txt"})
        chunks = self.splitter.split_documents([doc])
        for chunk in chunks:
            self.assertIn("source_doc_id", chunk.metadata)
            self.assertIn("chunk_index", chunk.metadata)
            self.assertEqual(chunk.metadata["source"], "test.txt")


class TestRAGPipeline(unittest.TestCase):
    """RAGPipeline测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.config = RAGConfig(chunk_size=100, top_k=2)
        self.pipeline = RAGPipeline(config=self.config)

    def test_add_documents(self):
        """测试添加文档。"""
        docs = [
            Document(content="机器学习是人工智能的核心技术"),
            Document(content="深度学习使用神经网络"),
        ]
        ids = self.pipeline.add_documents(docs)
        self.assertGreater(len(ids), 0)

    def test_add_texts(self):
        """测试添加文本。"""
        texts = ["文本一", "文本二"]
        metadatas = [{"source": "a"}, {"source": "b"}]
        ids = self.pipeline.add_texts(texts, metadatas)
        self.assertEqual(len(ids), 2)

    def test_query(self):
        """测试查询。"""
        self.pipeline.add_texts(["机器学习基础", "深度学习入门"])
        response = self.pipeline.query("什么是机器学习？")
        self.assertIsInstance(response, RAGResponse)
        self.assertIsNotNone(response.answer)
        self.assertIsNotNone(response.context)

    def test_query_empty_store(self):
        """测试空存储查询。"""
        response = self.pipeline.query("测试问题")
        self.assertIsInstance(response, RAGResponse)


class TestAdvancedRAGPipeline(unittest.TestCase):
    """AdvancedRAGPipeline测试类。"""

    def setUp(self):
        """测试前准备。"""
        self.config = RAGConfig(chunk_size=100, top_k=2)
        self.pipeline = AdvancedRAGPipeline(config=self.config)
        self.pipeline.add_texts([
            "机器学习是人工智能的核心技术",
            "深度学习使用神经网络进行学习",
            "自然语言处理用于理解人类语言",
        ])

    def test_query_basic(self):
        """测试基础查询。"""
        response = self.pipeline.query("什么是机器学习？")
        self.assertIsInstance(response, RAGResponse)

    def test_query_with_rewrite(self):
        """测试带查询重写的查询。"""
        response = self.pipeline.query("机器学习", use_query_rewrite=True)
        self.assertIsInstance(response, RAGResponse)

    def test_custom_query_rewriter(self):
        """测试自定义查询重写器。"""
        def custom_rewriter(query):
            return [query, f"{query}是什么", f"{query}的应用"]
        
        pipeline = AdvancedRAGPipeline(
            config=self.config,
            query_rewriter=custom_rewriter,
        )
        pipeline.add_texts(["机器学习基础知识"])
        response = pipeline.query("机器学习", use_query_rewrite=True)
        self.assertIsInstance(response, RAGResponse)


if __name__ == "__main__":
    unittest.main()
