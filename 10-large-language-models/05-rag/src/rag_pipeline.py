"""
RAG流水线 (RAG Pipeline) 实现

本模块提供完整的RAG流水线实现，整合文档处理、检索和生成。

=== 核心思想 ===

RAG流水线是端到端的检索增强生成框架，协调以下组件：

1. 文档处理
   - 文本分块 (Chunking)
   - 向量化嵌入
   - 索引构建

2. 检索增强
   - 相似度检索
   - 上下文构建
   - 查询优化

3. 生成回答
   - 提示模板
   - LLM生成
   - 答案返回

=== 数学基础 ===

递归文本分块:
    chunks = split_by_separators(text, separators)
    
    分隔符优先级:
    1. 段落 (\n\n)
    2. 换行 (\n)
    3. 句号 (。！？.!?)
    4. 空格
    5. 字符

上下文构建:
    context = join(top_k_docs, separator)
    total_length = Σ len(doc) ≤ max_context_length

=== 算法流程 ===

基础RAG流水线:
    输入: 用户问题
      ↓
    ┌─────────────────────────┐
    │  1. 文档处理 (离线)       │
    │  documents → chunks      │
    │  chunks → embeddings     │
    └─────────────────────────┘
      ↓
    ┌─────────────────────────┐
    │  2. 检索 (在线)          │
    │  query → embedding       │
    │  embedding → top_k_docs  │
    └─────────────────────────┘
      ↓
    ┌─────────────────────────┐
    │  3. 上下文构建           │
    │  docs → context          │
    │  context + query → prompt│
    └─────────────────────────┘
      ↓
    ┌─────────────────────────┐
    │  4. 生成回答             │
    │  prompt → LLM → answer   │
    └─────────────────────────┘
      ↓
    输出: RAGResponse(answer, sources, context)

高级RAG流水线:
    输入: 用户问题 + 选项
      ↓
    [可选] 查询重写
    query → [query_1, query_2, ...]
      ↓
    多查询检索与合并
    for q in queries:
        results.append(retrieve(q))
    merged = merge(results)
      ↓
    [可选] 重排序
    reranked = rerank(merged[:top_m])[:top_k]
      ↓
    构建带引用的上下文
    context = build_with_citations(reranked)
      ↓
    生成回答
    answer = llm(context + query)
      ↓
    输出: RAGResponse

=== 参考文献 ===

1. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" NeurIPS 2020
2. Izacard & Grave. "Leveraging Passage Retrieval with Generative Models" EMNLP 2021
3. Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" ICLR 2024
4. Mallen et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation" 2023

=== 核心组件 ===

    - RAGConfig: RAG配置类
    - RAGResponse: RAG响应数据结构
    - TextSplitter: 文本分割器基类
    - RecursiveTextSplitter: 递归文本分割器
    - RAGPipeline: 基础RAG流水线
    - AdvancedRAGPipeline: 高级RAG流水线 (查询重写/多轮检索)

作者: AI-Practices
许可证: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from .embeddings import BaseEmbedding, DenseEmbedding
    from .vector_store import Document, SearchResult, SimpleVectorStore, VectorStore
    from .retriever import DenseRetriever, RetrieverConfig
except ImportError:
    from embeddings import BaseEmbedding, DenseEmbedding
    from vector_store import Document, SearchResult, SimpleVectorStore, VectorStore
    from retriever import DenseRetriever, RetrieverConfig


__all__ = [
    "RAGConfig",
    "RAGPipeline",
    "AdvancedRAGPipeline",
    "RAGResponse",
    "TextSplitter",
    "RecursiveTextSplitter",
]


@dataclass
class RAGConfig:
    """RAG配置。

    参数：
        chunk_size: 文本分块大小
        chunk_overlap: 分块重叠大小
        top_k: 检索结果数量
        max_context_length: 最大上下文长度
        prompt_template: 提示模板
    """
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    max_context_length: int = 4096
    prompt_template: str = """基于以下上下文回答问题。

上下文：
{context}

问题：{question}

回答："""

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size必须为正数，得到 {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap不能为负数，得到 {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
        if self.top_k <= 0:
            raise ValueError(f"top_k必须为正数，得到 {self.top_k}")


@dataclass
class RAGResponse:
    """RAG响应。

    参数：
        answer: 生成的回答
        source_documents: 来源文档
        context: 使用的上下文
        prompt: 完整提示
    """
    answer: str
    source_documents: List[Document]
    context: str
    prompt: str

    def __repr__(self) -> str:
        preview = self.answer[:100] + "..." if len(self.answer) > 100 else self.answer
        return f"RAGResponse(answer='{preview}', sources={len(self.source_documents)})"


class TextSplitter(ABC):
    """文本分割器基类。"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """初始化文本分割器。

        参数：
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """分割文本。

        参数：
            text: 输入文本

        返回：
            文本块列表
        """
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表。

        参数：
            documents: 文档列表

        返回：
            分割后的文档列表
        """
        result = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "source_doc_id": doc.doc_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                )
                result.append(new_doc)
        return result


class RecursiveTextSplitter(TextSplitter):
    """递归文本分割器。

    按照分隔符层级递归分割文本，保持语义完整性。

    算法原理：
        1. 按优先级选择分隔符（段落 > 换行 > 句子 > 空格 > 字符）
        2. 使用选定分隔符分割文本
        3. 对超长块递归使用下一级分隔符
        4. 合并小块并保持重叠

    分隔符优先级：
        1. 段落 (\\n\\n)
        2. 换行 (\\n)
        3. 句子 (。！？.!?)
        4. 空格
        5. 字符

    示例：
        >>> splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
        >>> chunks = splitter.split_text("长文本...")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ) -> None:
        """初始化递归文本分割器。

        参数：
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
            separators: 分隔符列表
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """递归分割文本。"""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归分割实现。"""
        final_chunks = []

        # 选择当前最优分隔符
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # 使用分隔符分割
        if separator:
            splits = text.split(separator)
        else:
            # 字符级分割
            splits = list(text)

        # 处理分割结果
        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                # 小块暂存
                good_splits.append(split)
            else:
                # 大块需要进一步分割
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    # 递归使用下一级分隔符
                    other_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(other_chunks)
                else:
                    final_chunks.append(split)

        # 合并剩余小块
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """合并小块，保持重叠。"""
        merged = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)
            # 检查是否超过块大小
            if current_length + split_length + len(separator) > self.chunk_size:
                if current_chunk:
                    # 保存当前块
                    merged.append(separator.join(current_chunk))
                    # 保留部分内容作为重叠
                    overlap_start = max(0, len(current_chunk) - 2)
                    current_chunk = current_chunk[overlap_start:]
                    current_length = sum(len(s) for s in current_chunk) + len(separator) * (len(current_chunk) - 1)

            current_chunk.append(split)
            current_length += split_length + len(separator)

        # 保存最后一块
        if current_chunk:
            merged.append(separator.join(current_chunk))

        return merged

    def __repr__(self) -> str:
        return (
            f"RecursiveTextSplitter(chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )


class RAGPipeline:
    """基础RAG流水线。

    实现完整的检索增强生成流程：
        1. 文档分块
        2. 向量化存储
        3. 相似度检索
        4. 上下文构建
        5. 答案生成

    示例：
        >>> pipeline = RAGPipeline()
        >>> pipeline.add_documents([Document(content="...")])
        >>> response = pipeline.query("What is machine learning?")
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None,
        generator: Optional[Callable[[str], str]] = None,
    ) -> None:
        """初始化RAG流水线。

        参数：
            config: RAG配置
            embedding: 嵌入模型
            vector_store: 向量存储
            generator: 生成器函数
        """
        self.config = config or RAGConfig()
        self.embedding = embedding or DenseEmbedding()
        self.vector_store = vector_store or SimpleVectorStore()
        self.generator = generator or self._default_generator
        self.splitter = RecursiveTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.retriever = DenseRetriever(
            embedding=self.embedding,
            vector_store=self.vector_store,
            config=RetrieverConfig(top_k=self.config.top_k),
        )

    def _default_generator(self, prompt: str) -> str:
        """默认生成器（占位符）。"""
        return f"[模拟生成] 基于提供的上下文，这是一个示例回答。实际应用中请接入LLM API。"

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到流水线。

        参数：
            documents: 文档列表

        返回：
            文档ID列表
        """
        # 分割文档为小块
        chunks = self.splitter.split_documents(documents)
        # 添加到检索器（自动生成嵌入）
        return self.retriever.add_documents(chunks)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """添加文本到流水线。

        参数：
            texts: 文本列表
            metadatas: 元数据列表

        返回：
            文档ID列表
        """
        # 创建默认元数据
        metadatas = metadatas or [{} for _ in texts]
        # 转换为文档对象
        documents = [
            Document(content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        return self.add_documents(documents)

    def query(self, question: str) -> RAGResponse:
        """查询RAG流水线。

        RAG流程：
            1. 检索相关文档
            2. 构建上下文
            3. 生成提示
            4. 调用生成器

        参数：
            question: 用户问题

        返回：
            RAG响应
        """
        # Step 1: 检索相关文档
        results = self.retriever.retrieve(question)
        # Step 2: 构建上下文
        context = self._build_context(results)
        # Step 3: 生成提示
        prompt = self.config.prompt_template.format(
            context=context,
            question=question,
        )
        # Step 4: 调用生成器
        answer = self.generator(prompt)

        return RAGResponse(
            answer=answer,
            source_documents=[r.document for r in results],
            context=context,
            prompt=prompt,
        )

    def _build_context(self, results: List[SearchResult]) -> str:
        """构建上下文。
        
        将检索结果拼接为上下文字符串，控制总长度不超过max_context_length。
        """
        context_parts = []
        total_length = 0

        for result in results:
            content = result.document.content
            # 检查是否超过最大长度
            if total_length + len(content) > self.config.max_context_length:
                remaining = self.config.max_context_length - total_length
                if remaining > 100:
                    # 截断并添加省略号
                    context_parts.append(content[:remaining] + "...")
                break
            context_parts.append(content)
            total_length += len(content)

        # 使用分隔符连接
        return "\n\n---\n\n".join(context_parts)

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(chunk_size={self.config.chunk_size}, "
            f"top_k={self.config.top_k})"
        )


class AdvancedRAGPipeline(RAGPipeline):
    """高级RAG流水线。

    在基础RAG基础上增加：
        - 查询重写: 将复杂查询分解为多个子查询
        - 多轮检索: 合并多个查询的检索结果
        - 来源引用: 在上下文中标注来源

    示例：
        >>> pipeline = AdvancedRAGPipeline()
        >>> pipeline.add_documents(documents)
        >>> response = pipeline.query("复杂问题", use_query_rewrite=True)
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None,
        generator: Optional[Callable[[str], str]] = None,
        query_rewriter: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        """初始化高级RAG流水线。

        参数：
            config: RAG配置
            embedding: 嵌入模型
            vector_store: 向量存储
            generator: 生成器函数
            query_rewriter: 查询重写函数
        """
        super().__init__(config, embedding, vector_store, generator)
        self.query_rewriter = query_rewriter or self._default_query_rewriter

    def _default_query_rewriter(self, query: str) -> List[str]:
        """默认查询重写器（返回原查询）。"""
        return [query]

    def query(
        self,
        question: str,
        use_query_rewrite: bool = False,
        max_iterations: int = 1,
    ) -> RAGResponse:
        """高级查询。

        高级RAG流程：
            1. 查询重写（可选）
            2. 多查询检索
            3. 结果去重与排序
            4. 构建带引用的上下文
            5. 生成回答

        参数：
            question: 用户问题
            use_query_rewrite: 是否使用查询重写
            max_iterations: 最大迭代次数

        返回：
            RAG响应
        """
        # Step 1: 查询重写
        if use_query_rewrite:
            queries = self.query_rewriter(question)
        else:
            queries = [question]

        # Step 2: 多查询检索，合并结果
        all_results: Dict[str, SearchResult] = {}
        for query in queries:
            results = self.retriever.retrieve(query)
            for result in results:
                doc_id = result.document.doc_id
                # 保留分数最高的结果
                if doc_id not in all_results or result.score > all_results[doc_id].score:
                    all_results[doc_id] = result

        # Step 3: 按分数排序，取top_k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True,
        )[:self.config.top_k]

        # Step 4: 构建带引用的上下文
        context = self._build_context_with_citations(sorted_results)
        prompt = self.config.prompt_template.format(
            context=context,
            question=question,
        )
        # Step 5: 生成回答
        answer = self.generator(prompt)

        return RAGResponse(
            answer=answer,
            source_documents=[r.document for r in sorted_results],
            context=context,
            prompt=prompt,
        )

    def _build_context_with_citations(self, results: List[SearchResult]) -> str:
        """构建带引用的上下文。
        
        格式: [序号] 内容
              来源: 来源信息
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.document.metadata.get("source", "未知来源")
            context_parts.append(f"[{i}] {result.document.content}\n来源: {source}")
        return "\n\n---\n\n".join(context_parts)

    def __repr__(self) -> str:
        return (
            f"AdvancedRAGPipeline(chunk_size={self.config.chunk_size}, "
            f"top_k={self.config.top_k})"
        )
