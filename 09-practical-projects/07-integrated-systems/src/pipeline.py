"""
多模态RAG流水线。

整合多模态检索和视觉问答智能体的端到端流水线。

核心组件:
    - PipelineConfig: 流水线配置
    - PipelineResult: 流水线结果
    - MultimodalRAGPipeline: 多模态RAG流水线
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .multimodal_retriever import (
    CLIPEncoder,
    ModalityType,
    MultimodalDocument,
    MultimodalRetriever,
    SearchResult,
)
from .vision_qa_agent import AgentResult, VisionQAAgent


@dataclass
class PipelineConfig:
    """流水线配置。"""
    top_k: int = 5
    max_context_length: int = 4096
    max_agent_steps: int = 10
    use_agent: bool = True
    similarity_threshold: float = 0.0
    verbose: bool = False
    
    prompt_template: str = """基于以下检索到的内容回答问题。

检索内容:
{context}

问题: {question}

回答:"""


@dataclass
class PipelineResult:
    """流水线结果。"""
    question: str
    answer: str
    sources: List[MultimodalDocument] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    agent_result: Optional[AgentResult] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_sources(self) -> int:
        return len(self.sources)
    
    def __repr__(self) -> str:
        preview = self.answer[:80] + "..." if len(self.answer) > 80 else self.answer
        return f"PipelineResult(answer='{preview}', sources={self.num_sources})"


class MultimodalRAGPipeline:
    """多模态RAG流水线。
    
    整合检索和生成的端到端流水线:
        1. 接收多模态查询 (文本/图像)
        2. 检索相关文档
        3. 构建上下文
        4. 使用Agent或直接生成回答
    
    示例:
        >>> pipeline = MultimodalRAGPipeline()
        >>> pipeline.add_documents(docs)
        >>> result = pipeline.query("这张图片里有什么?", image=img)
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        encoder: Optional[CLIPEncoder] = None,
        llm_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.encoder = encoder or CLIPEncoder()
        self.llm_func = llm_func or self._mock_llm
        
        self.retriever = MultimodalRetriever(
            encoder=self.encoder,
            similarity_threshold=self.config.similarity_threshold,
        )
        
        self.agent = VisionQAAgent(
            retriever=self.retriever,
            llm_func=self.llm_func,
            max_steps=self.config.max_agent_steps,
            verbose=self.config.verbose,
        )
    
    def _mock_llm(self, prompt: str) -> str:
        """模拟LLM响应。"""
        return "基于检索到的内容，这是一个示例回答。"
    
    def add_document(self, doc: MultimodalDocument) -> str:
        """添加单个文档。"""
        return self.retriever.add_document(doc)
    
    def add_documents(self, docs: List[MultimodalDocument]) -> List[str]:
        """批量添加文档。"""
        return self.retriever.add_documents(docs)
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加文本文档。"""
        doc = MultimodalDocument(content=text, metadata=metadata or {})
        return self.add_document(doc)
    
    def add_image(
        self,
        image: np.ndarray,
        caption: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """添加图像文档。"""
        doc = MultimodalDocument(
            content=caption,
            image_data=image,
            metadata=metadata or {},
        )
        return self.add_document(doc)
    
    def query(
        self,
        question: str,
        image: Optional[np.ndarray] = None,
        use_agent: Optional[bool] = None,
    ) -> PipelineResult:
        """执行查询。
        
        参数:
            question: 问题文本
            image: 查询图像 (可选)
            use_agent: 是否使用Agent (默认使用配置)
            
        返回:
            流水线结果
        """
        start_time = time.time()
        use_agent = use_agent if use_agent is not None else self.config.use_agent
        
        # 检索
        if image is not None:
            search_results = self.retriever.search(image, top_k=self.config.top_k)
        else:
            search_results = self.retriever.search(question, top_k=self.config.top_k)
        
        sources = [r.document for r in search_results]
        
        # 生成回答
        if use_agent:
            context = [r.document.content for r in search_results if r.document.content]
            agent_result = self.agent.answer(question, image=image, context=context)
            answer = agent_result.answer
        else:
            context = self._build_context(search_results)
            prompt = self.config.prompt_template.format(
                context=context,
                question=question,
            )
            answer = self.llm_func(prompt)
            agent_result = None
        
        latency_ms = (time.time() - start_time) * 1000
        
        return PipelineResult(
            question=question,
            answer=answer,
            sources=sources,
            search_results=search_results,
            agent_result=agent_result,
            latency_ms=latency_ms,
            metadata={
                "use_agent": use_agent,
                "num_retrieved": len(search_results),
            },
        )
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """构建上下文。"""
        parts = []
        total_len = 0
        
        for i, result in enumerate(results, 1):
            content = result.document.content
            if not content:
                content = f"[图像文档 {result.document.doc_id[:8]}]"
            
            entry = f"[{i}] {content}"
            
            if total_len + len(entry) > self.config.max_context_length:
                remaining = self.config.max_context_length - total_len
                if remaining > 50:
                    parts.append(entry[:remaining] + "...")
                break
            
            parts.append(entry)
            total_len += len(entry)
        
        return "\n\n".join(parts)
    
    def batch_query(
        self,
        questions: List[str],
        images: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[PipelineResult]:
        """批量查询。"""
        if images is None:
            images = [None] * len(questions)
        
        results = []
        for q, img in zip(questions, images):
            results.append(self.query(q, image=img))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return {
            "num_documents": self.retriever.num_documents,
            "encoder_dim": self.encoder.embedding_dim,
            "config": {
                "top_k": self.config.top_k,
                "max_context_length": self.config.max_context_length,
                "use_agent": self.config.use_agent,
            },
        }
    
    def clear(self) -> None:
        """清空所有文档。"""
        self.retriever.clear()
    
    def __repr__(self) -> str:
        return f"MultimodalRAGPipeline(documents={self.retriever.num_documents})"
