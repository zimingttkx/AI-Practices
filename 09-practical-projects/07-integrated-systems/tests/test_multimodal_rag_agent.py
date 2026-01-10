"""
多模态RAG智能体单元测试。
"""

import numpy as np
import pytest

from src.multimodal_retriever import (
    CLIPEncoder,
    ImageEncoder,
    ModalityType,
    MultimodalDocument,
    MultimodalRetriever,
    SearchResult,
    TextEncoder,
)
from src.vision_qa_agent import (
    ActionType,
    AgentResult,
    AgentStep,
    VisionAction,
    VisionObservation,
    VisionQAAgent,
)
from src.pipeline import (
    MultimodalRAGPipeline,
    PipelineConfig,
    PipelineResult,
)


# =============================================================================
# MultimodalDocument Tests
# =============================================================================

class TestMultimodalDocument:
    """MultimodalDocument测试。"""
    
    def test_create_text_document(self):
        doc = MultimodalDocument(content="测试文本")
        assert doc.content == "测试文本"
        assert doc.modality == ModalityType.TEXT
        assert doc.has_text
        assert not doc.has_image
    
    def test_create_image_document(self):
        img = np.random.rand(64, 64, 3).astype(np.float32)
        doc = MultimodalDocument(image_data=img)
        assert doc.has_image
        assert doc.modality == ModalityType.IMAGE
        assert doc.image_data.shape == (64, 64, 3)
    
    def test_create_multimodal_document(self):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        doc = MultimodalDocument(content="图像描述", image_data=img)
        assert doc.has_text
        assert doc.has_image
    
    def test_document_id_unique(self):
        doc1 = MultimodalDocument(content="a")
        doc2 = MultimodalDocument(content="b")
        assert doc1.doc_id != doc2.doc_id
    
    def test_content_hash(self):
        doc1 = MultimodalDocument(content="相同内容")
        doc2 = MultimodalDocument(content="相同内容")
        assert doc1.content_hash() == doc2.content_hash()
        
        doc3 = MultimodalDocument(content="不同内容")
        assert doc1.content_hash() != doc3.content_hash()


# =============================================================================
# Encoder Tests
# =============================================================================

class TestTextEncoder:
    """TextEncoder测试。"""
    
    def test_encode_single_text(self):
        encoder = TextEncoder(dim=128)
        emb = encoder.encode("hello world")
        assert emb.shape == (1, 128)
        assert np.isclose(np.linalg.norm(emb[0]), 1.0, atol=1e-5)
    
    def test_encode_multiple_texts(self):
        encoder = TextEncoder(dim=256)
        texts = ["text one", "text two", "text three"]
        emb = encoder.encode(texts)
        assert emb.shape == (3, 256)
    
    def test_encode_empty_text(self):
        encoder = TextEncoder(dim=64)
        emb = encoder.encode("")
        assert emb.shape == (1, 64)
        assert np.allclose(emb[0], 0)
    
    def test_encode_batch(self):
        encoder = TextEncoder(dim=128)
        texts = [f"text {i}" for i in range(100)]
        emb = encoder.encode_batch(texts, batch_size=32)
        assert emb.shape == (100, 128)


class TestImageEncoder:
    """ImageEncoder测试。"""
    
    def test_encode_single_image(self):
        encoder = ImageEncoder(dim=128)
        img = np.random.rand(64, 64, 3).astype(np.float32)
        emb = encoder.encode(img)
        assert emb.shape == (1, 128)
    
    def test_encode_grayscale_image(self):
        encoder = ImageEncoder(dim=64)
        img = np.random.rand(32, 32).astype(np.float32)
        emb = encoder.encode(img)
        assert emb.shape == (1, 64)
    
    def test_encode_multiple_images(self):
        encoder = ImageEncoder(dim=128)
        images = [np.random.rand(48, 48, 3).astype(np.float32) for _ in range(5)]
        emb = encoder.encode(images)
        assert emb.shape == (5, 128)


class TestCLIPEncoder:
    """CLIPEncoder测试。"""
    
    def test_encode_text(self):
        encoder = CLIPEncoder(dim=256)
        emb = encoder.encode_text("a photo of a cat")
        assert emb.shape == (1, 256)
    
    def test_encode_image(self):
        encoder = CLIPEncoder(dim=256)
        img = np.random.rand(64, 64, 3).astype(np.float32)
        emb = encoder.encode_image(img)
        assert emb.shape == (1, 256)
    
    def test_encode_document_text_only(self):
        encoder = CLIPEncoder(dim=128)
        doc = MultimodalDocument(content="test content")
        emb = encoder.encode_document(doc)
        assert emb.shape == (128,)
    
    def test_encode_document_image_only(self):
        encoder = CLIPEncoder(dim=128)
        img = np.random.rand(32, 32, 3).astype(np.float32)
        doc = MultimodalDocument(image_data=img)
        emb = encoder.encode_document(doc)
        assert emb.shape == (128,)
    
    def test_encode_document_multimodal(self):
        encoder = CLIPEncoder(dim=128)
        img = np.random.rand(32, 32, 3).astype(np.float32)
        doc = MultimodalDocument(content="caption", image_data=img)
        emb = encoder.encode_document(doc)
        assert emb.shape == (128,)


# =============================================================================
# MultimodalRetriever Tests
# =============================================================================

class TestMultimodalRetriever:
    """MultimodalRetriever测试。"""
    
    @pytest.fixture
    def retriever(self):
        return MultimodalRetriever()
    
    @pytest.fixture
    def sample_docs(self):
        return [
            MultimodalDocument(content="机器学习是人工智能的一个分支"),
            MultimodalDocument(content="深度学习使用神经网络"),
            MultimodalDocument(content="自然语言处理研究文本理解"),
        ]
    
    def test_add_document(self, retriever):
        doc = MultimodalDocument(content="test")
        doc_id = retriever.add_document(doc)
        assert retriever.num_documents == 1
        assert doc_id == doc.doc_id
    
    def test_add_documents(self, retriever, sample_docs):
        doc_ids = retriever.add_documents(sample_docs)
        assert len(doc_ids) == 3
        assert retriever.num_documents == 3
    
    def test_search_by_text(self, retriever, sample_docs):
        retriever.add_documents(sample_docs)
        results = retriever.search_by_text("神经网络", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_by_image(self, retriever):
        img1 = np.random.rand(32, 32, 3).astype(np.float32)
        img2 = np.random.rand(32, 32, 3).astype(np.float32)
        retriever.add_document(MultimodalDocument(image_data=img1, content="cat"))
        retriever.add_document(MultimodalDocument(image_data=img2, content="dog"))
        
        query_img = np.random.rand(32, 32, 3).astype(np.float32)
        results = retriever.search_by_image(query_img, top_k=2)
        assert len(results) == 2
    
    def test_search_empty_retriever(self, retriever):
        results = retriever.search("query")
        assert results == []
    
    def test_get_document(self, retriever):
        doc = MultimodalDocument(content="findme")
        retriever.add_document(doc)
        found = retriever.get_document(doc.doc_id)
        assert found is not None
        assert found.content == "findme"
    
    def test_remove_document(self, retriever, sample_docs):
        retriever.add_documents(sample_docs)
        assert retriever.num_documents == 3
        
        removed = retriever.remove_document(sample_docs[0].doc_id)
        assert removed
        assert retriever.num_documents == 2
    
    def test_clear(self, retriever, sample_docs):
        retriever.add_documents(sample_docs)
        retriever.clear()
        assert retriever.num_documents == 0
    
    def test_modality_filter(self, retriever):
        retriever.add_document(MultimodalDocument(content="text only"))
        img = np.random.rand(32, 32, 3).astype(np.float32)
        retriever.add_document(MultimodalDocument(image_data=img))
        
        text_results = retriever.search("query", modality_filter=ModalityType.TEXT)
        assert all(r.document.modality == ModalityType.TEXT for r in text_results)


# =============================================================================
# VisionQAAgent Tests
# =============================================================================

class TestVisionAction:
    """VisionAction测试。"""
    
    def test_create_action(self):
        action = VisionAction(
            action_type=ActionType.SEARCH,
            params={"query": "test"},
        )
        assert action.action_type == ActionType.SEARCH
        assert action.params["query"] == "test"


class TestVisionObservation:
    """VisionObservation测试。"""
    
    def test_successful_observation(self):
        action = VisionAction(action_type=ActionType.DESCRIBE)
        obs = VisionObservation(action=action, result="描述结果", success=True)
        assert obs.success
        assert obs.error is None
    
    def test_failed_observation(self):
        action = VisionAction(action_type=ActionType.LOCATE)
        obs = VisionObservation(action=action, result=None, success=False, error="定位失败")
        assert not obs.success
        assert obs.error == "定位失败"


class TestVisionQAAgent:
    """VisionQAAgent测试。"""
    
    @pytest.fixture
    def agent(self):
        return VisionQAAgent(max_steps=5)
    
    def test_answer_text_question(self, agent):
        result = agent.answer("什么是机器学习?")
        assert isinstance(result, AgentResult)
        assert result.question == "什么是机器学习?"
        assert result.num_steps > 0
    
    def test_answer_with_image(self, agent):
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = agent.answer("图中有什么?", image=img)
        assert isinstance(result, AgentResult)
    
    def test_answer_with_context(self, agent):
        context = ["背景信息1", "背景信息2"]
        result = agent.answer("问题", context=context)
        assert isinstance(result, AgentResult)
    
    def test_max_steps_limit(self):
        agent = VisionQAAgent(max_steps=2)
        result = agent.answer("复杂问题")
        assert result.num_steps <= 2
    
    def test_register_custom_action(self, agent):
        def custom_handler(params):
            return "custom result"
        
        agent.register_action(ActionType.THINK, custom_handler)
        action = VisionAction(action_type=ActionType.THINK)
        obs = agent._execute_action(action)
        assert obs.result == "custom result"


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestPipelineConfig:
    """PipelineConfig测试。"""
    
    def test_default_config(self):
        config = PipelineConfig()
        assert config.top_k == 5
        assert config.use_agent
    
    def test_custom_config(self):
        config = PipelineConfig(top_k=10, use_agent=False)
        assert config.top_k == 10
        assert not config.use_agent


class TestPipelineResult:
    """PipelineResult测试。"""
    
    def test_result_properties(self):
        result = PipelineResult(
            question="test",
            answer="answer",
            sources=[MultimodalDocument(content="src")],
        )
        assert result.num_sources == 1


class TestMultimodalRAGPipeline:
    """MultimodalRAGPipeline测试。"""
    
    @pytest.fixture
    def pipeline(self):
        return MultimodalRAGPipeline()
    
    def test_add_text(self, pipeline):
        doc_id = pipeline.add_text("测试文本")
        assert pipeline.retriever.num_documents == 1
    
    def test_add_image(self, pipeline):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        doc_id = pipeline.add_image(img, caption="测试图像")
        assert pipeline.retriever.num_documents == 1
    
    def test_query_text(self, pipeline):
        pipeline.add_text("机器学习是AI的分支")
        pipeline.add_text("深度学习使用神经网络")
        
        result = pipeline.query("什么是机器学习?", use_agent=False)
        assert isinstance(result, PipelineResult)
        assert result.answer
    
    def test_query_with_agent(self, pipeline):
        pipeline.add_text("测试内容")
        result = pipeline.query("问题", use_agent=True)
        assert result.agent_result is not None
    
    def test_query_with_image(self, pipeline):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        pipeline.add_image(img, caption="cat")
        
        query_img = np.random.rand(32, 32, 3).astype(np.float32)
        result = pipeline.query("这是什么?", image=query_img)
        assert isinstance(result, PipelineResult)
    
    def test_batch_query(self, pipeline):
        pipeline.add_text("内容1")
        pipeline.add_text("内容2")
        
        questions = ["问题1", "问题2", "问题3"]
        results = pipeline.batch_query(questions)
        assert len(results) == 3
    
    def test_get_stats(self, pipeline):
        pipeline.add_text("test")
        stats = pipeline.get_stats()
        assert stats["num_documents"] == 1
        assert "config" in stats
    
    def test_clear(self, pipeline):
        pipeline.add_text("test")
        pipeline.clear()
        assert pipeline.retriever.num_documents == 0
    
    def test_latency_tracking(self, pipeline):
        pipeline.add_text("test")
        result = pipeline.query("query")
        assert result.latency_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
