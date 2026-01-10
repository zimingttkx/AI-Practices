"""
代码检索器实现。

支持代码片段的语义检索和结构化索引。

核心组件:
    - CodeDocument: 代码文档
    - CodeChunker: 代码分块器
    - CodeEmbedding: 代码嵌入
    - CodeRetriever: 代码检索器
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class CodeLanguage(Enum):
    """编程语言类型。"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


class CodeBlockType(Enum):
    """代码块类型。"""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    SNIPPET = "snippet"


@dataclass
class CodeDocument:
    """代码文档。
    
    属性:
        doc_id: 文档唯一标识
        content: 代码内容
        language: 编程语言
        block_type: 代码块类型
        name: 函数/类名称
        file_path: 文件路径
        start_line: 起始行号
        end_line: 结束行号
        metadata: 元数据
        embedding: 嵌入向量
    """
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    language: CodeLanguage = CodeLanguage.UNKNOWN
    block_type: CodeBlockType = CodeBlockType.SNIPPET
    name: str = ""
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    @property
    def num_lines(self) -> int:
        return self.content.count("\n") + 1
    
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode("utf-8")).hexdigest()[:16]
    
    def __repr__(self) -> str:
        preview = self.name or self.content[:30].replace("\n", " ")
        return f"CodeDocument({self.language.value}, {self.block_type.value}, '{preview}')"


@dataclass
class SearchResult:
    """检索结果。"""
    document: CodeDocument
    score: float
    rank: int = 0


class CodeChunker:
    """代码分块器。
    
    将代码文件分割为语义完整的代码块。
    """
    
    LANGUAGE_PATTERNS = {
        CodeLanguage.PYTHON: {
            "function": r"^(\s*)def\s+(\w+)\s*\(",
            "class": r"^(\s*)class\s+(\w+)",
            "method": r"^(\s{4,})def\s+(\w+)\s*\(",
        },
        CodeLanguage.JAVASCRIPT: {
            "function": r"^(\s*)(?:async\s+)?function\s+(\w+)\s*\(",
            "class": r"^(\s*)class\s+(\w+)",
            "method": r"^(\s{2,})(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{",
        },
        CodeLanguage.TYPESCRIPT: {
            "function": r"^(\s*)(?:export\s+)?(?:async\s+)?function\s+(\w+)",
            "class": r"^(\s*)(?:export\s+)?class\s+(\w+)",
            "method": r"^(\s{2,})(?:public|private|protected)?\s*(?:async\s+)?(\w+)\s*\(",
        },
    }
    
    EXTENSION_MAP = {
        ".py": CodeLanguage.PYTHON,
        ".js": CodeLanguage.JAVASCRIPT,
        ".ts": CodeLanguage.TYPESCRIPT,
        ".tsx": CodeLanguage.TYPESCRIPT,
        ".java": CodeLanguage.JAVA,
        ".cpp": CodeLanguage.CPP,
        ".go": CodeLanguage.GO,
        ".rs": CodeLanguage.RUST,
    }
    
    def __init__(self, max_chunk_lines: int = 100, min_chunk_lines: int = 5) -> None:
        self.max_chunk_lines = max_chunk_lines
        self.min_chunk_lines = min_chunk_lines
    
    def detect_language(self, file_path: str) -> CodeLanguage:
        """检测编程语言。"""
        for ext, lang in self.EXTENSION_MAP.items():
            if file_path.endswith(ext):
                return lang
        return CodeLanguage.UNKNOWN
    
    def chunk_file(self, content: str, file_path: str = "") -> List[CodeDocument]:
        """分割代码文件。"""
        language = self.detect_language(file_path)
        lines = content.split("\n")
        
        if language == CodeLanguage.UNKNOWN or language not in self.LANGUAGE_PATTERNS:
            return self._chunk_by_lines(content, file_path, language)
        
        return self._chunk_by_structure(lines, file_path, language)
    
    def _chunk_by_structure(
        self,
        lines: List[str],
        file_path: str,
        language: CodeLanguage,
    ) -> List[CodeDocument]:
        """按代码结构分块。"""
        patterns = self.LANGUAGE_PATTERNS[language]
        chunks = []
        current_block: List[Tuple[int, str]] = []
        current_type = CodeBlockType.SNIPPET
        current_name = ""
        current_indent = 0
        
        for i, line in enumerate(lines):
            # 检测函数/类定义
            for block_type, pattern in patterns.items():
                match = re.match(pattern, line)
                if match:
                    # 保存之前的块
                    if current_block:
                        doc = self._create_document(
                            current_block, file_path, language,
                            current_type, current_name
                        )
                        if doc:
                            chunks.append(doc)
                    
                    current_block = [(i, line)]
                    current_indent = len(match.group(1))
                    current_name = match.group(2)
                    current_type = CodeBlockType(block_type)
                    break
            else:
                # 继续当前块
                if current_block:
                    # 检查是否退出当前块
                    stripped = line.lstrip()
                    if stripped and not line.startswith(" " * (current_indent + 1)):
                        if not stripped.startswith("#") and not stripped.startswith("//"):
                            # 可能是新的顶层定义
                            pass
                    current_block.append((i, line))
                else:
                    current_block.append((i, line))
        
        # 保存最后一个块
        if current_block:
            doc = self._create_document(
                current_block, file_path, language,
                current_type, current_name
            )
            if doc:
                chunks.append(doc)
        
        return chunks
    
    def _chunk_by_lines(
        self,
        content: str,
        file_path: str,
        language: CodeLanguage,
    ) -> List[CodeDocument]:
        """按行数分块。"""
        lines = content.split("\n")
        chunks = []
        
        for i in range(0, len(lines), self.max_chunk_lines):
            chunk_lines = lines[i:i + self.max_chunk_lines]
            if len(chunk_lines) >= self.min_chunk_lines:
                doc = CodeDocument(
                    content="\n".join(chunk_lines),
                    language=language,
                    block_type=CodeBlockType.SNIPPET,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=i + len(chunk_lines),
                )
                chunks.append(doc)
        
        return chunks
    
    def _create_document(
        self,
        block: List[Tuple[int, str]],
        file_path: str,
        language: CodeLanguage,
        block_type: CodeBlockType,
        name: str,
    ) -> Optional[CodeDocument]:
        """创建代码文档。"""
        if len(block) < self.min_chunk_lines:
            return None
        
        content = "\n".join(line for _, line in block)
        start_line = block[0][0] + 1
        end_line = block[-1][0] + 1
        
        return CodeDocument(
            content=content,
            language=language,
            block_type=block_type,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
        )


class CodeEmbedding:
    """代码嵌入模型。
    
    将代码转换为向量表示。
    """
    
    KEYWORDS = {
        CodeLanguage.PYTHON: [
            "def", "class", "import", "from", "return", "if", "else", "elif",
            "for", "while", "try", "except", "with", "as", "yield", "async",
            "await", "lambda", "self", "None", "True", "False",
        ],
        CodeLanguage.JAVASCRIPT: [
            "function", "class", "const", "let", "var", "return", "if", "else",
            "for", "while", "try", "catch", "async", "await", "import", "export",
            "this", "null", "undefined", "true", "false",
        ],
    }
    
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        np.random.seed(42)
        self._token_embeddings = np.random.randn(10000, dim).astype(np.float32)
        self._token_embeddings /= np.linalg.norm(
            self._token_embeddings, axis=1, keepdims=True
        )
    
    def _tokenize(self, code: str) -> List[int]:
        """简单分词。"""
        tokens = re.findall(r"\w+|[^\w\s]", code.lower())
        return [hash(t) % 10000 for t in tokens]
    
    def encode(self, codes: Union[str, List[str]]) -> np.ndarray:
        """编码代码。"""
        if isinstance(codes, str):
            codes = [codes]
        
        embeddings = []
        for code in codes:
            token_ids = self._tokenize(code)
            if not token_ids:
                emb = np.zeros(self.dim, dtype=np.float32)
            else:
                emb = np.mean(self._token_embeddings[token_ids], axis=0)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
            embeddings.append(emb)
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode_document(self, doc: CodeDocument) -> np.ndarray:
        """编码代码文档。"""
        # 组合代码内容和元信息
        text_parts = [doc.content]
        if doc.name:
            text_parts.append(doc.name)
        combined = " ".join(text_parts)
        return self.encode(combined)[0]


class CodeRetriever:
    """代码检索器。
    
    支持语义检索和结构化过滤。
    
    示例:
        >>> retriever = CodeRetriever()
        >>> retriever.index_file(code_content, "main.py")
        >>> results = retriever.search("排序算法", top_k=5)
    """
    
    def __init__(
        self,
        embedding: Optional[CodeEmbedding] = None,
        chunker: Optional[CodeChunker] = None,
    ) -> None:
        self.embedding = embedding or CodeEmbedding()
        self.chunker = chunker or CodeChunker()
        
        self._documents: Dict[str, CodeDocument] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._index_matrix: Optional[np.ndarray] = None
        self._index_ids: List[str] = []
    
    @property
    def num_documents(self) -> int:
        return len(self._documents)
    
    def index_file(self, content: str, file_path: str) -> List[str]:
        """索引代码文件。"""
        chunks = self.chunker.chunk_file(content, file_path)
        return self.add_documents(chunks)
    
    def add_document(self, doc: CodeDocument) -> str:
        """添加单个文档。"""
        if doc.embedding is None:
            doc.embedding = self.embedding.encode_document(doc)
        
        self._documents[doc.doc_id] = doc
        self._embeddings[doc.doc_id] = doc.embedding
        self._rebuild_index()
        
        return doc.doc_id
    
    def add_documents(self, docs: List[CodeDocument]) -> List[str]:
        """批量添加文档。"""
        doc_ids = []
        for doc in docs:
            if doc.embedding is None:
                doc.embedding = self.embedding.encode_document(doc)
            self._documents[doc.doc_id] = doc
            self._embeddings[doc.doc_id] = doc.embedding
            doc_ids.append(doc.doc_id)
        
        self._rebuild_index()
        return doc_ids
    
    def _rebuild_index(self) -> None:
        """重建索引。"""
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
        query: str,
        top_k: int = 10,
        language_filter: Optional[CodeLanguage] = None,
        block_type_filter: Optional[CodeBlockType] = None,
    ) -> List[SearchResult]:
        """检索代码。"""
        if self._index_matrix is None:
            return []
        
        query_emb = self.embedding.encode(query)[0]
        scores = self._index_matrix @ query_emb
        
        # 应用过滤
        if language_filter or block_type_filter:
            mask = np.ones(len(self._index_ids), dtype=bool)
            for i, doc_id in enumerate(self._index_ids):
                doc = self._documents[doc_id]
                if language_filter and doc.language != language_filter:
                    mask[i] = False
                if block_type_filter and doc.block_type != block_type_filter:
                    mask[i] = False
            scores = np.where(mask, scores, -np.inf)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] == -np.inf:
                continue
            doc_id = self._index_ids[idx]
            results.append(SearchResult(
                document=self._documents[doc_id],
                score=float(scores[idx]),
                rank=rank,
            ))
        
        return results
    
    def search_by_name(self, name: str) -> List[CodeDocument]:
        """按名称搜索。"""
        results = []
        name_lower = name.lower()
        for doc in self._documents.values():
            if name_lower in doc.name.lower():
                results.append(doc)
        return results
    
    def get_document(self, doc_id: str) -> Optional[CodeDocument]:
        """获取文档。"""
        return self._documents.get(doc_id)
    
    def clear(self) -> None:
        """清空索引。"""
        self._documents.clear()
        self._embeddings.clear()
        self._index_matrix = None
        self._index_ids = []
    
    def __repr__(self) -> str:
        return f"CodeRetriever(documents={self.num_documents})"
