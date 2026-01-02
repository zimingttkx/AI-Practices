# 05-RAG (检索增强生成)

> Retrieval-Augmented Generation - 结合检索与生成的知识增强技术

## 目录结构

```
05-rag/
├── README.md                    # 本文件
├── knowledge_points.md          # 知识点详解
├── src/
│   ├── __init__.py              # 模块导出
│   ├── embeddings.py            # 向量嵌入 (405行)
│   ├── vector_store.py          # 向量存储 (240行)
│   ├── retriever.py             # 检索器 (310行)
│   └── rag_pipeline.py          # RAG流水线 (467行)
├── notebooks/
│   ├── 01_embedding_models.ipynb
│   ├── 02_vector_databases.ipynb
│   ├── 03_rag_pipeline.ipynb
│   └── 04_advanced_rag.ipynb
└── tests/
    ├── test_embeddings.py       # 嵌入测试 (23个)
    ├── test_vector_store.py     # 存储测试 (17个)
    ├── test_retriever.py        # 检索测试 (15个)
    ├── test_rag_pipeline.py     # 流水线测试 (19个)
    └── run_tests.py             # 测试运行器
```

## 快速开始

### 基础RAG流水线

```python
from src import RAGPipeline, RAGConfig, Document

# 创建RAG流水线
config = RAGConfig(chunk_size=512, top_k=3)
pipeline = RAGPipeline(config=config)

# 添加文档
documents = [
    Document(content="机器学习是人工智能的核心技术..."),
    Document(content="深度学习使用神经网络进行学习..."),
]
pipeline.add_documents(documents)

# 查询
response = pipeline.query("什么是机器学习？")
print(response.answer)
print(f"来源文档: {len(response.source_documents)}")
```

### 向量嵌入

```python
from src import DenseEmbedding, SparseEmbedding, EmbeddingConfig

# 稠密嵌入
config = EmbeddingConfig(dimension=384)
dense = DenseEmbedding(config)
vector = dense.embed_text("Hello world")

# 稀疏嵌入 (BM25)
sparse = SparseEmbedding(k1=1.5, b=0.75)
sparse.fit(["文档1", "文档2", "文档3"])
vector = sparse.embed_text("查询文本")
```

### 向量存储

```python
from src import SimpleVectorStore, Document
import numpy as np

# 创建存储
store = SimpleVectorStore(metric="cosine")

# 添加文档
doc = Document(content="测试内容")
doc.embedding = np.random.randn(384)
store.add_documents([doc])

# 搜索
results = store.search(query_vector, top_k=5)
```

### 混合检索

```python
from src import DenseRetriever, SparseRetriever, HybridRetriever

# 创建混合检索器
hybrid = HybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    alpha=0.7,  # 稠密检索权重
    fusion_method="rrf",  # RRF融合
)

results = hybrid.retrieve("查询问题")
```

## 核心组件

### 1. 向量嵌入 (embeddings.py)

| 类 | 描述 |
|---|---|
| `EmbeddingConfig` | 嵌入配置 |
| `BaseEmbedding` | 嵌入基类 |
| `DenseEmbedding` | 稠密向量嵌入 |
| `SparseEmbedding` | BM25稀疏嵌入 |
| `HybridEmbedding` | 混合嵌入 |

### 2. 向量存储 (vector_store.py)

| 类 | 描述 |
|---|---|
| `Document` | 文档数据结构 |
| `SearchResult` | 搜索结果 |
| `VectorStore` | 存储基类 |
| `SimpleVectorStore` | 简单向量存储 |

### 3. 检索器 (retriever.py)

| 类 | 描述 |
|---|---|
| `RetrieverConfig` | 检索器配置 |
| `BaseRetriever` | 检索器基类 |
| `DenseRetriever` | 稠密检索器 |
| `SparseRetriever` | 稀疏检索器 |
| `HybridRetriever` | 混合检索器 |

### 4. RAG流水线 (rag_pipeline.py)

| 类 | 描述 |
|---|---|
| `RAGConfig` | RAG配置 |
| `RAGResponse` | RAG响应 |
| `TextSplitter` | 文本分割器基类 |
| `RecursiveTextSplitter` | 递归文本分割器 |
| `RAGPipeline` | 基础RAG流水线 |
| `AdvancedRAGPipeline` | 高级RAG流水线 |

## 检索策略对比

| 策略 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| 稠密检索 | 语义理解强 | 需要训练 | 语义相似度 |
| 稀疏检索 | 精确匹配 | 无语义理解 | 关键词匹配 |
| 混合检索 | 综合优势 | 复杂度高 | 通用场景 |

## 融合方法

### RRF (Reciprocal Rank Fusion)

```
score = Σ 1/(k + rank_i)
```

- 不依赖原始分数
- 对异构结果鲁棒
- k通常取60

### 加权融合

```
score = α * dense_score + (1-α) * sparse_score
```

- 需要分数归一化
- α控制稠密/稀疏权重

## 运行测试

```bash
cd 05-rag
python tests/run_tests.py
```

**测试统计**: 74个测试，100%通过

## 参考资料

1. [RAG原始论文](https://arxiv.org/abs/2005.11401) - Lewis et al., 2020
2. [BM25算法](https://en.wikipedia.org/wiki/Okapi_BM25)
3. [FAISS](https://github.com/facebookresearch/faiss)
4. [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
5. [LlamaIndex](https://docs.llamaindex.ai/)
