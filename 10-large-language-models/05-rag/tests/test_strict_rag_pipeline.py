"""
RAG流水线模块严格单元测试

测试覆盖：
    - 边界条件测试
    - 异常处理测试
    - 类型检查测试
    - __repr__ 方法测试
    - 流水线完整性测试
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_store import Document
from rag_pipeline import (
    RAGConfig,
    RAGResponse,
    RecursiveTextSplitter,
    RAGPipeline,
    AdvancedRAGPipeline,
)


class TestRAGConfigStrict(unittest.TestCase):
    """RAGConfig严格测试。"""

    def test_chunk_size_boundary_one(self):
        """测试chunk_size=1边界。"""
        # chunk_overlap必须小于chunk_size
        config = RAGConfig(chunk_size=1, chunk_overlap=0)
        self.assertEqual(config.chunk_size, 1)

    def test_chunk_size_large(self):
        """测试大chunk_size。"""
        config = RAGConfig(chunk_size=10000)
        self.assertEqual(config.chunk_size, 10000)

    def test_invalid_chunk_size_zero(self):
        """测试chunk_size=0。"""
        with self.assertRaises(ValueError):
            RAGConfig(chunk_size=0)

    def test_invalid_chunk_size_negative(self):
        """测试负chunk_size。"""
        with self.assertRaises(ValueError):
            RAGConfig(chunk_size=-1)

    def test_chunk_overlap_zero(self):
        """测试chunk_overlap=0。"""
        config = RAGConfig(chunk_overlap=0)
        self.assertEqual(config.chunk_overlap, 0)

    def test_invalid_chunk_overlap_negative(self):
        """测试负chunk_overlap。"""
        with self.assertRaises(ValueError):
            RAGConfig(chunk_overlap=-1)

    def test_invalid_chunk_overlap_equals_size(self):
        """测试chunk_overlap等于chunk_size。"""
        with self.assertRaises(ValueError):
            RAGConfig(chunk_size=100, chunk_overlap=100)

    def test_invalid_chunk_overlap_greater_than_size(self):
        """测试chunk_overlap大于chunk_size。"""
        with self.assertRaises(ValueError):
            RAGConfig(chunk_size=100, chunk_overlap=150)

    def test_top_k_boundary_one(self):
        """测试top_k=1边界。"""
        config = RAGConfig(top_k=1)
        self.assertEqual(config.top_k, 1)

    def test_invalid_top_k_zero(self):
        """测试top_k=0。"""
        with self.assertRaises(ValueError):
            RAGConfig(top_k=0)

    def test_invalid_top_k_negative(self):
        """测试负top_k。"""
        with self.assertRaises(ValueError):
            RAGConfig(top_k=-1)

    def test_custom_prompt_template(self):
        """测试自定义提示模板。"""
        template = "Context: {context}\nQ: {question}\nA:"
        config = RAGConfig(prompt_template=template)
        self.assertEqual(config.prompt_template, template)


class TestRAGResponseStrict(unittest.TestCase):
    """RAGResponse严格测试。"""

    def test_repr_short_answer(self):
        """测试__repr__短回答。"""
        response = RAGResponse(
            answer="short",
            source_documents=[],
            context="",
            prompt="",
        )
        repr_str = repr(response)
        self.assertIn("short", repr_str)
        self.assertIn("sources=0", repr_str)

    def test_repr_long_answer_truncated(self):
        """测试__repr__长回答截断。"""
        long_answer = "x" * 200
        response = RAGResponse(
            answer=long_answer,
            source_documents=[],
            context="",
            prompt="",
        )
        repr_str = repr(response)
        self.assertIn("...", repr_str)
        self.assertLess(len(repr_str), 200)

    def test_empty_source_documents(self):
        """测试空来源文档。"""
        response = RAGResponse(
            answer="answer",
            source_documents=[],
            context="context",
            prompt="prompt",
        )
        self.assertEqual(len(response.source_documents), 0)

    def test_multiple_source_documents(self):
        """测试多个来源文档。"""
        docs = [Document(content=f"doc {i}") for i in range(5)]
        response = RAGResponse(
            answer="answer",
            source_documents=docs,
            context="context",
            prompt="prompt",
        )
        self.assertEqual(len(response.source_documents), 5)


class TestRecursiveTextSplitterStrict(unittest.TestCase):
    """RecursiveTextSplitter严格测试。"""

    def test_repr(self):
        """测试__repr__方法。"""
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
        repr_str = repr(splitter)
        self.assertIn("RecursiveTextSplitter", repr_str)
        self.assertIn("100", repr_str)
        self.assertIn("20", repr_str)

    def test_split_empty_text(self):
        """测试分割空文本。"""
        splitter = RecursiveTextSplitter(chunk_size=100)
        chunks = splitter.split_text("")
        # 空文本分割后返回空列表
        self.assertEqual(len(chunks), 0)

    def test_split_single_char(self):
        """测试分割单字符。"""
        splitter = RecursiveTextSplitter(chunk_size=100)
        chunks = splitter.split_text("a")
        self.assertEqual(len(chunks), 1)

    def test_split_exact_chunk_size(self):
        """测试精确chunk_size文本。"""
        splitter = RecursiveTextSplitter(chunk_size=10, chunk_overlap=0)
        text = "a" * 10
        chunks = splitter.split_text(text)
        self.assertGreaterEqual(len(chunks), 1)

    def test_split_preserves_content(self):
        """测试分割保留内容。"""
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "Hello world. This is a test. Another sentence here."
        chunks = splitter.split_text(text)
        # 所有内容应该在某个块中
        combined = "".join(chunks)
        # 由于分隔符处理，可能有细微差异，但核心词应该存在
        self.assertIn("Hello", combined)
        self.assertIn("test", combined)

    def test_split_by_paragraph(self):
        """测试按段落分割。"""
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter.split_text(text)
        self.assertGreaterEqual(len(chunks), 1)

    def test_split_by_newline(self):
        """测试按换行分割。"""
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "Line one.\nLine two.\nLine three.\nLine four."
        chunks = splitter.split_text(text)
        self.assertGreaterEqual(len(chunks), 1)

    def test_split_by_sentence(self):
        """测试按句子分割。"""
        splitter = RecursiveTextSplitter(chunk_size=30, chunk_overlap=0)
        text = "First sentence。Second sentence。Third sentence。"
        chunks = splitter.split_text(text)
        self.assertGreaterEqual(len(chunks), 1)

    def test_split_unicode(self):
        """测试Unicode文本分割。"""
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "你好世界。这是测试。另一个句子。" * 10
        chunks = splitter.split_text(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertFalse(any(c == '\ufffd' for c in chunk))  # 无乱码

    def test_split_documents_empty_list(self):
        """测试分割空文档列表。"""
        splitter = RecursiveTextSplitter(chunk_size=100)
        chunks = splitter.split_documents([])
        self.assertEqual(len(chunks), 0)

    def test_split_documents_preserves_metadata(self):
        """测试分割保留元数据。"""
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
        doc = Document(content="x" * 200, metadata={"source": "test.txt", "author": "test"})
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "test.txt")
            self.assertEqual(chunk.metadata["author"], "test")
            self.assertIn("source_doc_id", chunk.metadata)
            self.assertIn("chunk_index", chunk.metadata)

    def test_split_documents_chunk_indices(self):
        """测试分割块索引正确。"""
        splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=0)
        doc = Document(content="word " * 50)
        chunks = splitter.split_documents([doc])
        indices = [c.metadata["chunk_index"] for c in chunks]
        self.assertEqual(indices, list(range(len(chunks))))

    def test_custom_separators(self):
        """测试自定义分隔符。"""
        splitter = RecursiveTextSplitter(
            chunk_size=50,
            chunk_overlap=0,
            separators=["|||", "---", " "],
        )
        text = "Part1|||Part2|||Part3"
        chunks = splitter.split_text(text)
        self.assertGreaterEqual(len(chunks), 1)


class TestRAGPipelineStrict(unittest.TestCase):
    """RAGPipeline严格测试。"""

    def setUp(self):
        """测试前准备。"""
        self.config = RAGConfig(chunk_size=100, chunk_overlap=20, top_k=2)
        self.pipeline = RAGPipeline(config=self.config)

    def test_repr(self):
        """测试__repr__方法。"""
        repr_str = repr(self.pipeline)
        self.assertIn("RAGPipeline", repr_str)
        self.assertIn("100", repr_str)
        self.assertIn("2", repr_str)

    def test_add_empty_documents(self):
        """测试添加空文档列表。"""
        ids = self.pipeline.add_documents([])
        self.assertEqual(len(ids), 0)

    def test_add_single_document(self):
        """测试添加单个文档。"""
        doc = Document(content="single document content")
        ids = self.pipeline.add_documents([doc])
        self.assertGreater(len(ids), 0)

    def test_add_texts_empty(self):
        """测试添加空文本列表。"""
        ids = self.pipeline.add_texts([])
        self.assertEqual(len(ids), 0)

    def test_add_texts_with_metadata(self):
        """测试添加带元数据的文本。"""
        texts = ["text1", "text2"]
        metadatas = [{"source": "a"}, {"source": "b"}]
        ids = self.pipeline.add_texts(texts, metadatas)
        self.assertEqual(len(ids), 2)

    def test_add_texts_without_metadata(self):
        """测试添加不带元数据的文本。"""
        texts = ["text1", "text2"]
        ids = self.pipeline.add_texts(texts)
        self.assertEqual(len(ids), 2)

    def test_query_empty_store(self):
        """测试空存储查询。"""
        response = self.pipeline.query("test question")
        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(len(response.source_documents), 0)

    def test_query_returns_response(self):
        """测试查询返回响应。"""
        self.pipeline.add_texts(["machine learning basics"])
        response = self.pipeline.query("what is ML?")
        self.assertIsInstance(response, RAGResponse)
        self.assertIsNotNone(response.answer)
        self.assertIsNotNone(response.context)
        self.assertIsNotNone(response.prompt)

    def test_query_unicode(self):
        """测试Unicode查询。"""
        self.pipeline.add_texts(["机器学习基础知识"])
        response = self.pipeline.query("什么是机器学习？")
        self.assertIsInstance(response, RAGResponse)

    def test_custom_generator(self):
        """测试自定义生成器。"""
        def custom_gen(prompt):
            return f"Custom: {len(prompt)} chars"
        
        pipeline = RAGPipeline(
            config=self.config,
            generator=custom_gen,
        )
        pipeline.add_texts(["test content"])
        response = pipeline.query("question")
        self.assertIn("Custom:", response.answer)

    def test_context_length_limit(self):
        """测试上下文长度限制。"""
        config = RAGConfig(chunk_size=100, chunk_overlap=20, top_k=10, max_context_length=100)
        pipeline = RAGPipeline(config=config)
        # 添加大量文本
        pipeline.add_texts([f"document {i} " * 20 for i in range(10)])
        response = pipeline.query("test")
        self.assertLessEqual(len(response.context), 150)  # 允许一些余量


class TestAdvancedRAGPipelineStrict(unittest.TestCase):
    """AdvancedRAGPipeline严格测试。"""

    def setUp(self):
        """测试前准备。"""
        self.config = RAGConfig(chunk_size=100, chunk_overlap=20, top_k=2)
        self.pipeline = AdvancedRAGPipeline(config=self.config)
        self.pipeline.add_texts([
            "机器学习是人工智能的核心",
            "深度学习使用神经网络",
            "自然语言处理理解文本",
        ])

    def test_repr(self):
        """测试__repr__方法。"""
        repr_str = repr(self.pipeline)
        self.assertIn("AdvancedRAGPipeline", repr_str)

    def test_query_without_rewrite(self):
        """测试不带重写的查询。"""
        response = self.pipeline.query("机器学习", use_query_rewrite=False)
        self.assertIsInstance(response, RAGResponse)

    def test_query_with_rewrite(self):
        """测试带重写的查询。"""
        response = self.pipeline.query("机器学习", use_query_rewrite=True)
        self.assertIsInstance(response, RAGResponse)

    def test_custom_query_rewriter(self):
        """测试自定义查询重写器。"""
        def rewriter(q):
            return [q, f"{q}是什么", f"{q}的应用"]
        
        pipeline = AdvancedRAGPipeline(
            config=self.config,
            query_rewriter=rewriter,
        )
        pipeline.add_texts(["测试内容"])
        response = pipeline.query("测试", use_query_rewrite=True)
        self.assertIsInstance(response, RAGResponse)

    def test_context_has_citations(self):
        """测试上下文包含引用。"""
        response = self.pipeline.query("机器学习")
        # 高级流水线应该有引用格式
        if response.source_documents:
            self.assertIn("[", response.context)

    def test_deduplication(self):
        """测试结果去重。"""
        # 使用会产生重复的查询重写器
        def dup_rewriter(q):
            return [q, q, q]  # 相同查询
        
        pipeline = AdvancedRAGPipeline(
            config=RAGConfig(chunk_size=100, chunk_overlap=20, top_k=5),
            query_rewriter=dup_rewriter,
        )
        pipeline.add_texts(["doc1", "doc2", "doc3"])
        response = pipeline.query("test", use_query_rewrite=True)
        doc_ids = [d.doc_id for d in response.source_documents]
        self.assertEqual(len(doc_ids), len(set(doc_ids)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
