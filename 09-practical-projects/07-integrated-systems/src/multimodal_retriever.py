"""
多模态检索器实现。

支持图像和文本的联合检索，基于CLIP对比学习框架。

核心组件:
    - ImageEncoder: 图像编码器
    - TextEncoder: 文本编码器
    - CLIPEncoder: CLIP联合编码器
    - MultimodalDocument: 多模态文档
    - MultimodalRetriever: 多模态检索器
"""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ModalityType(Enum):
    """模态类型。"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultimodalDocument:
    """多模态文档。
    
    支持文本、图像等多种模态的统一表示。
    
    属性:
        doc_id: 文档唯一标识
        content: 文本内容
        image_data: 图像数据 (numpy array, HWC格式)
        modality: 主要模态类型
        metadata: 元数据
        embedding: 预计算的嵌入向量
    """
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    image_data: Optional[np.ndarray] = None
    modality: ModalityType = ModalityType.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        if self.image_data is not None:
            self.modality = ModalityType.IMAGE
        elif self.content:
            self.modality = ModalityType.TEXT
    
    @property
    def has_image(self) -> bool:
        return self.image_data is not None
    
    @property
    def has_text(self) -> bool:
        return bool(self.content)
    
    def content_hash(self) -> str:
        """计算内容哈希。"""
        data = self.content.encode("utf-8")
        if self.image_data is not None:
            data += self.image_data.tobytes()
        return hashlib.md5(data).hexdigest()[:16]
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        img_info = f", image={self.image_data.shape}" if self.has_image else ""
        return f"MultimodalDocument(id={self.doc_id[:8]}, text='{preview}'{img_info})"


@dataclass
class SearchResult:
    """检索结果。"""
    document: MultimodalDocument
    score: float
    rank: int = 0
    
    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, doc={self.document.doc_id[:8]})"


class BaseEncoder(ABC):
    """编码器基类。"""
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """嵌入维度。"""
        pass
    
    @abstractmethod
    def encode(self, inputs: Any) -> np.ndarray:
        """编码输入。"""
        pass


class TextEncoder(BaseEncoder):
    """文本编码器。
    
    使用简化的词袋模型 + 随机投影生成文本嵌入。
    实际应用中应替换为预训练模型。
    """
    
    def __init__(self, dim: int = 512, vocab_size: int = 10000) -> None:
        self.dim = dim
        self.vocab_size = vocab_size
        np.random.seed(42)
        self._projection = np.random.randn(vocab_size, dim).astype(np.float32)
        self._projection /= np.linalg.norm(self._projection, axis=1, keepdims=True)
    
    @property
    def embedding_dim(self) -> int:
        return self.dim
    
    def _tokenize(self, text: str) -> List[int]:
        """简单分词，返回token id列表。"""
        tokens = text.lower().split()
        return [hash(t) % self.vocab_size for t in tokens]
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本。
        
        参数:
            texts: 单个文本或文本列表
            
        返回:
            嵌入向量 (N, dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            token_ids = self._tokenize(text)
            if not token_ids:
                emb = np.zeros(self.dim, dtype=np.float32)
            else:
                emb = np.mean(self._projection[token_ids], axis=0)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
            embeddings.append(emb)
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码。"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.append(self.encode(batch))
        return np.vstack(all_embeddings)


class ImageEncoder(BaseEncoder):
    """图像编码器。
    
    使用简化的特征提取生成图像嵌入。
    实际应用中应替换为ViT等预训练模型。
    """
    
    def __init__(self, dim: int = 512, patch_size: int = 16) -> None:
        self.dim = dim
        self.patch_size = patch_size
        np.random.seed(43)
        self._projection = np.random.randn(patch_size * patch_size * 3, dim).astype(np.float32)
        self._projection /= np.linalg.norm(self._projection, axis=1, keepdims=True)
    
    @property
    def embedding_dim(self) -> int:
        return self.dim
    
    def _extract_patches(self, image: np.ndarray) -> np.ndarray:
        """提取图像块。"""
        if image.ndim == 1:
            # 一维数组，reshape为方形图像
            size = int(np.sqrt(len(image)))
            image = image[:size*size].reshape(size, size)
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        
        h, w, c = image.shape
        ph, pw = self.patch_size, self.patch_size
        
        # 调整大小到可整除
        nh = (h // ph) * ph
        nw = (w // pw) * pw
        if nh == 0:
            nh = ph
        if nw == 0:
            nw = pw
        
        # 简单resize
        image_resized = self._simple_resize(image, (nh, nw))
        
        # 提取patches
        patches = []
        for i in range(0, nh, ph):
            for j in range(0, nw, pw):
                patch = image_resized[i:i+ph, j:j+pw, :]
                patches.append(patch.flatten())
        
        return np.array(patches, dtype=np.float32)
    
    def _simple_resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """简单图像缩放。"""
        h, w = size
        oh, ow = image.shape[:2]
        
        y_indices = np.linspace(0, oh - 1, h).astype(int)
        x_indices = np.linspace(0, ow - 1, w).astype(int)
        
        return image[np.ix_(y_indices, x_indices)]
    
    def encode(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """编码图像。
        
        参数:
            images: 单个图像或图像列表 (HWC格式)
            
        返回:
            嵌入向量 (N, dim)
        """
        if isinstance(images, np.ndarray) and images.ndim in (2, 3):
            images = [images]
        
        embeddings = []
        for image in images:
            patches = self._extract_patches(image)
            # 投影到嵌入空间
            patch_dim = min(patches.shape[1], self._projection.shape[0])
            proj = self._projection[:patch_dim, :]
            patch_embs = patches[:, :patch_dim] @ proj
            # 平均池化
            emb = np.mean(patch_embs, axis=0)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            embeddings.append(emb)
        
        return np.array(embeddings, dtype=np.float32)


class CLIPEncoder(BaseEncoder):
    """CLIP联合编码器。
    
    将文本和图像编码到同一向量空间。
    """
    
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.text_encoder = TextEncoder(dim=dim)
        self.image_encoder = ImageEncoder(dim=dim)
        
        # 模态对齐投影
        np.random.seed(44)
        self._text_proj = np.eye(dim, dtype=np.float32)
        self._image_proj = np.eye(dim, dtype=np.float32)
    
    @property
    def embedding_dim(self) -> int:
        return self.dim
    
    def encode(self, inputs: Any) -> np.ndarray:
        """统一编码接口。"""
        if isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], str)):
            return self.encode_text(inputs)
        else:
            return self.encode_image(inputs)
    
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本。"""
        emb = self.text_encoder.encode(texts)
        return emb @ self._text_proj
    
    def encode_image(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """编码图像。"""
        emb = self.image_encoder.encode(images)
        return emb @ self._image_proj
    
    def encode_document(self, doc: MultimodalDocument) -> np.ndarray:
        """编码多模态文档。
        
        对于同时包含文本和图像的文档，取两者嵌入的平均。
        """
        embeddings = []
        
        if doc.has_text:
            text_emb = self.encode_text(doc.content)
            embeddings.append(text_emb[0])
        
        if doc.has_image:
            image_emb = self.encode_image(doc.image_data)
            embeddings.append(image_emb[0])
        
        if not embeddings:
            return np.zeros(self.dim, dtype=np.float32)
        
        combined = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        
        return combined


class MultimodalRetriever:
    """多模态检索器。
    
    支持文本查询图像、图像查询文本、跨模态检索。
    
    示例:
        >>> retriever = MultimodalRetriever()
        >>> retriever.add_documents([doc1, doc2, doc3])
        >>> results = retriever.search("a cat sitting on a chair", top_k=5)
    """
    
    def __init__(
        self,
        encoder: Optional[CLIPEncoder] = None,
        similarity_threshold: float = 0.0,
    ) -> None:
        self.encoder = encoder or CLIPEncoder()
        self.similarity_threshold = similarity_threshold
        
        self._documents: Dict[str, MultimodalDocument] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._index_matrix: Optional[np.ndarray] = None
        self._index_ids: List[str] = []
    
    @property
    def num_documents(self) -> int:
        return len(self._documents)
    
    def add_document(self, doc: MultimodalDocument) -> str:
        """添加单个文档。"""
        if doc.embedding is None:
            doc.embedding = self.encoder.encode_document(doc)
        
        self._documents[doc.doc_id] = doc
        self._embeddings[doc.doc_id] = doc.embedding
        self._rebuild_index()
        
        return doc.doc_id
    
    def add_documents(self, docs: List[MultimodalDocument]) -> List[str]:
        """批量添加文档。"""
        doc_ids = []
        for doc in docs:
            if doc.embedding is None:
                doc.embedding = self.encoder.encode_document(doc)
            self._documents[doc.doc_id] = doc
            self._embeddings[doc.doc_id] = doc.embedding
            doc_ids.append(doc.doc_id)
        
        self._rebuild_index()
        return doc_ids
    
    def _rebuild_index(self) -> None:
        """重建索引矩阵。"""
        if not self._embeddings:
            self._index_matrix = None
            self._index_ids = []
            return
        
        self._index_ids = list(self._embeddings.keys())
        self._index_matrix = np.array(
            [self._embeddings[doc_id] for doc_id in self._index_ids],
            dtype=np.float32
        )
    
    def search(
        self,
        query: Union[str, np.ndarray, MultimodalDocument],
        top_k: int = 10,
        modality_filter: Optional[ModalityType] = None,
    ) -> List[SearchResult]:
        """检索相关文档。
        
        参数:
            query: 查询 (文本/图像/文档)
            top_k: 返回结果数
            modality_filter: 模态过滤
            
        返回:
            检索结果列表
        """
        if self._index_matrix is None or len(self._index_ids) == 0:
            return []
        
        # 编码查询
        if isinstance(query, str):
            query_emb = self.encoder.encode_text(query)[0]
        elif isinstance(query, np.ndarray):
            if query.ndim == 1:
                query_emb = query
            else:
                query_emb = self.encoder.encode_image(query)[0]
        elif isinstance(query, MultimodalDocument):
            query_emb = self.encoder.encode_document(query)
        else:
            raise ValueError(f"不支持的查询类型: {type(query)}")
        
        # 计算相似度
        scores = self._index_matrix @ query_emb
        
        # 应用模态过滤
        if modality_filter is not None:
            mask = np.array([
                self._documents[doc_id].modality == modality_filter
                for doc_id in self._index_ids
            ])
            scores = np.where(mask, scores, -np.inf)
        
        # 排序
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            score = scores[idx]
            if score < self.similarity_threshold:
                continue
            doc_id = self._index_ids[idx]
            results.append(SearchResult(
                document=self._documents[doc_id],
                score=float(score),
                rank=rank,
            ))
        
        return results
    
    def search_by_image(
        self,
        image: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """图像检索。"""
        return self.search(image, top_k=top_k)
    
    def search_by_text(
        self,
        text: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """文本检索。"""
        return self.search(text, top_k=top_k)
    
    def get_document(self, doc_id: str) -> Optional[MultimodalDocument]:
        """获取文档。"""
        return self._documents.get(doc_id)
    
    def remove_document(self, doc_id: str) -> bool:
        """删除文档。"""
        if doc_id not in self._documents:
            return False
        
        del self._documents[doc_id]
        del self._embeddings[doc_id]
        self._rebuild_index()
        return True
    
    def clear(self) -> None:
        """清空所有文档。"""
        self._documents.clear()
        self._embeddings.clear()
        self._index_matrix = None
        self._index_ids = []
    
    def __repr__(self) -> str:
        return f"MultimodalRetriever(documents={self.num_documents})"
