# RAG (检索增强生成) 知识点详解

> 本文档涵盖RAG技术的核心概念、数学原理和工程实践

## 目录

- [第一部分：RAG基础原理](#第一部分rag基础原理)
- [第二部分：向量嵌入技术](#第二部分向量嵌入技术)
- [第三部分：向量数据库与索引](#第三部分向量数据库与索引)
- [第四部分：检索策略优化](#第四部分检索策略优化)
- [第五部分：高级RAG技术](#第五部分高级rag技术)
- [第六部分：评估与工程实践](#第六部分评估与工程实践)

---

## 第一部分：RAG基础原理

### 1.1 RAG核心思想

**核心问题**: LLM存在知识幻觉、知识过期、专业知识缺失等问题

**RAG解决方案**: 
```
Query → Retrieve → Augment → Generate
  ↓        ↓           ↓          ↓
 问题   检索相关文档  构建上下文  生成答案
```

**数学形式化**:
```
P(answer|query) = Σ P(answer|context) P(context|query)
```

其中:
- `P(context|query)`: 检索器模型 (Retriever)
- `P(answer|context)`: 生成器模型 (Generator)

### 1.2 RAG vs 微调对比

| 维度 | RAG | 微调 (Fine-tuning) |
|------|-----|-------------------|
| **知识更新** | 实时更新索引 | 需要重新训练 |
| **成本** | 低 (存储+检索) | 高 (GPU训练) |
| **可解释性** | 高 (可追溯来源) | 低 (黑盒) |
| **幻觉** | 显著减少 | 可能增加 |
| **适用场景** | 知识密集型任务 | 风格/格式调整 |
| **数据要求** | 非结构化文档 | QA对数据 |

### 1.3 RAG架构演进

**RAG-1 (Lewis et al., 2020)**:
- 端到端联合训练检索器和生成器
- 使用参数化记忆模块

**RAG-2 (Real-world RAG)**:
- 独立检索器 (BM25/Dense Retriever)
- 独立生成器 (LLaMA/GPT等)
- 模块化设计，易于工程落地

**RAG-3 (Advanced RAG)**:
- 查询重写 (Query Rewriting)
- 多轮检索 (Multi-hop Retrieval)
- 重排序 (Reranking)

---

## 第二部分：向量嵌入技术

### 2.1 稠密嵌入 (Dense Embedding)

#### 2.1.1 核心思想

将文本映射到连续向量空间，相似语义的文本在空间中距离相近。

#### 2.1.2 数学原理

**词嵌入平均 (Mean Pooling)**:
```
E(text) = (1/n) Σ E(token_i)
```

**注意力加权池化 (Attention Pooling)**:
```
E(text) = Σ α_i E(token_i)
α_i = softmax(u^T E(token_i))
```

#### 2.1.3 主流模型

**BERT系列**:
- BERT-base: 768维, 12层
- RoBERTa: 优化预训练策略
- E5: 基于对比学习的嵌入

**Sentence-BERT (Reimers & Gurevych, 2019)**:
```
Loss = -log(exp(sim(q,d+)/τ) / Σ exp(sim(q,d_i)/τ))
```

其中:
- `sim(u,v) = u^T v / (||u|| ||v||)` (余弦相似度)
- `τ`: 温度参数

**Contriever (Izacard et al., 2022)**:
- 无需对比训练
- 使用FFF (Fused Favorite Families) 损失

### 2.2 稀疏嵌入 (Sparse Embedding)

#### 2.2.1 BM25算法

**经典BM25公式**:
```
score(D, Q) = Σ IDF(q_i) × TF(q_i, D)
```

**IDF (逆文档频率)**:
```
IDF(q) = log((N - df(q) + 0.5) / (df(q) + 0.5) + 1)
```

**TF (词频饱和)**:
```
TF(q, D) = f(q, D) × (k1 + 1) / (f(q, D) + k1 × (1 - b + b × |D|/avgdl))
```

**参数说明**:
- `N`: 文档总数
- `df(q)`: 包含词q的文档数
- `f(q, D)`: 词q在文档D中的频率
- `|D|`: 文档D的长度
- `avgdl`: 平均文档长度
- `k1`: 词频饱和参数 (1.2-2.0, 默认1.5)
- `b`: 长度归一化参数 (0-1, 默认0.75)

**参数调优指南**:
- `k1` 越大: 词频影响越大 (适合短查询)
- `b` 越大: 长文档惩罚越大 (适合长文档)

#### 2.2.2 TF-IDF变体

**BM25 (Okapi)**: 考虑文档长度归一化
**BM25L**: 改进长文档性能
**BM25+**: 增加额外的词频项

### 2.3 混合嵌入 (Hybrid Embedding)

#### 2.3.1 核心思想

结合稠密嵌入 (语义相似) 和稀疏嵌入 (关键词匹配) 的优势。

#### 2.3.2 融合策略

**线性加权融合**:
```
score = α × dense_score + (1-α) × sparse_score
```

**RRF (Reciprocal Rank Fusion)**:
```
score = Σ 1/(k + rank_i)
```

典型值: `k=60` (Cormack et al., 2009)

**归一化分数融合**:
```
score = α × norm(dense) + (1-α) × norm(sparse)
norm(x) = (x - min) / (max - min)
```

#### 2.3.3 最佳实践

| 场景 | 推荐α值 | 说明 |
|------|---------|------|
| 语义搜索 | 0.7-0.9 | 偏向稠密 |
| 关键词搜索 | 0.3-0.5 | 偏向稀疏 |
| 平衡搜索 | 0.5-0.7 | 混合模式 |

---

## 第三部分：向量数据库与索引

### 3.1 相似度度量

#### 3.1.1 余弦相似度 (Cosine Similarity)

**公式**:
```
cos(A, B) = (A·B) / (||A|| × ||B||)
         = Σ(ai × bi) / (√Σai² × √Σbi²)
```

**特点**:
- 关注方向，忽略模长
- 归一化后等价于点积
- 取值范围: [-1, 1]

**适用场景**: 文本语义相似度

#### 3.1.2 欧氏距离 (Euclidean Distance)

**公式**:
```
d(A, B) = √Σ(ai - bi)²
```

**特点**:
- 考虑绝对距离
- 取值范围: [0, +∞]
- 距离越小越相似

**适用场景**: 图像特征匹配

#### 3.1.3 点积 (Dot Product)

**公式**:
```
A·B = Σ ai × bi
```

**特点**:
- 计算效率高
- 需要向量归一化
- 取值范围: [-∞, +∞]

**适用场景**: 高维归一化向量

#### 3.1.4 度量选择指南

| 度量 | 计算复杂度 | 归一化要求 | 推荐场景 |
|------|-----------|-----------|----------|
| 余弦 | O(d) | 无 | 文本检索 |
| 点积 | O(d) | 需要 | 高性能检索 |
| 欧氏 | O(d) | 无 | 图像检索 |

### 3.2 高效索引结构

#### 3.2.1 HNSW (Hierarchical Navigable Small World)

**核心思想**: 多层图结构，类似跳表

**算法流程**:
```
1. 构建多层图，顶层稀疏，底层稠密
2. 查询时从顶层随机入口开始
3. 每层贪婪搜索最近邻
4. 进入下一层，直到最底层
```

**时间复杂度**:
- 构建: O(N log N)
- 查询: O(log N)

**参数**:
- `M`: 每个节点的最大连接数 (默认16)
- `ef_construction`: 构建时的搜索宽度 (默认200)
- `ef`: 查询时的搜索宽度 (默认10)

**优点**: 高召回率，查询速度快
**缺点**: 内存占用高

#### 3.2.2 IVF (Inverted File Index)

**核心思想**: 聚类 + 倒排索引

**算法流程**:
```
1. 使用K-means聚类得到V个中心点
2. 将向量分配到最近的中心点
3. 每个中心点维护一个倒排列表
4. 查询时只搜索最近的nprobe个中心点
```

**时间复杂度**:
- 构建: O(N × V × iter)
- 查询: O(N/V × nprobe)

**参数**:
- `nlist`: 聚类中心数量 (默认sqrt(N))
- `nprobe`: 搜索中心数量 (默认1-10)

**优点**: 内存效率高，适合大规模
**缺点**: 召回率略低

#### 3.2.3 PQ (Product Quantization)

**核心思想**: 向量量化压缩

**算法流程**:
```
1. 将D维向量分成m段
2. 每段独立聚类成256个中心
3. 每段用8bit编码 (256个中心)
4. 压缩比: D×4字节 → m×1字节
```

**压缩比**:
```
原始: 768维 × 4字节 = 3KB
PQ: 96段 × 1字节 = 96字节 (压缩32倍)
```

**优点**: 显著降低内存占用
**缺点**: 精度损失

### 3.3 主流向量数据库

| 数据库 | 索引方法 | 特点 | 适用场景 |
|--------|----------|------|----------|
| **FAISS** | IVF/HNSW/PQ | Meta开源，C++实现 | 超大规模检索 |
| **Chroma** | HNSW | 轻量级，易用 | 原型开发 |
| **Pinecone** | 专有算法 | 托管服务 | 生产环境 |
| **Milvus** | IVF/HNSW/DiskANN | 分布式，云原生 | 企业级应用 |
| **Weaviate** | HNSW | 支持多模态 | 混合检索 |
| **Qdrant** | HNSW | Rust实现，高性能 | 实时检索 |

---

## 第四部分：检索策略优化

### 4.1 稠密检索 (Dense Retrieval)

**流程**:
```
Query → Embedding → Vector Store → Top-K Results
```

**优点**: 捕获语义相似性
**缺点**: 关键词匹配弱，计算成本高

### 4.2 稀疏检索 (Sparse Retrieval)

**流程**:
```
Query → BM25 → Lexical Search → Top-K Results
```

**优点**: 关键词精确匹配，速度快
**缺点**: 语义理解弱，词汇鸿沟

### 4.3 混合检索 (Hybrid Retrieval)

**RRF算法 (Cormack et al., 2009)**:

```
for doc in union(dense_results, sparse_results):
    score[doc] = Σ 1/(k + rank_i(doc))
    其中 k=60, rank_i从1开始
```

**优势**: 结合语义和关键词，提高召回率

### 4.4 重排序 (Reranking)

**两阶段检索**:
```
Stage 1: Bi-Encoder → 检索Top-100 (召回)
Stage 2: Cross-Encoder → 精排Top-10 (精排)
```

**Cross-Encoder (Nogueira & Cho, 2019)**:
```
score = CrossEncoder([CLS] Query [SEP] Doc [SEP])
```

**计算成本**: 
- Bi-Encoder: O(1) 单次计算
- Cross-Encoder: O(N) 需要对每个候选计算

**最佳实践**: 
- 初检: 20-50个候选
- 精排: 5-10个结果

---

## 第五部分：高级RAG技术

### 5.1 查询改写 (Query Rewriting)

#### 5.1.1 查询扩展

**方法1: 同义词扩展**
```
原查询: "ML algorithms"
扩展后: ["机器学习算法", "machine learning methods", "ML techniques"]
```

**方法2: 伪相关反馈**
```
1. 初始检索Top-K文档
2. 从中提取高频词
3. 将高频词加入查询重新检索
```

#### 5.1.2 查询分解 (Query Decomposition)

**复杂查询分解**:
```
原查询: "RAG和微调的区别是什么？"
分解: ["RAG的特点", "微调的特点", "RAG vs 微调对比"]
```

#### 5.1.3 HyDE (Hypothetical Document Embeddings)

**核心思想**: 生成假设性答案作为查询向量

```
Query: "What is the capital of France?"
↓ LLM生成假设答案
Hypothetical: "The capital of France is Paris."
↓ 嵌入化
Embedding → Vector Search
```

**论文**: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)

### 5.2 多轮检索 (Multi-hop Retrieval)

**递归检索**:
```
Query → Retrieve Docs → Extract Info → 
New Query → Retrieve Docs → Extract Info → ...
```

**Self-Query (Metzler et al., 2021)**:
模型自主决定是否需要更多检索。

### 5.3 Self-RAG (Self-Reflective Retrieval-Augmented Generation)

**论文**: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique" (2023)

**核心思想**: 模型生成特殊tokens控制检索流程

**控制Tokens**:
- `[Retrieve]`: 需要检索
- `[No Retrieve]`: 直接回答
- `[Critique]`: 评估答案质量
- `[Relevant] / [Irrelevant]`: 检索结果相关性

**训练目标**:
```
L = Σ L_gen + λ × L_retrieve + μ × L_critique
```

### 5.4 文档分块策略

#### 5.4.1 固定大小分块

```python
chunks = [text[i:i+chunk_size] 
          for i in range(0, len(text), chunk_size-overlap)]
```

**优点**: 简单高效
**缺点**: 可能切断语义

#### 5.4.2 递归分块 (Recursive Splitting)

**分隔符优先级**:
```
1. \n\n (段落)
2. \n (句子)
3. . ! ? (标点)
4. 空格
5. 字符
```

**优点**: 保持语义完整性
**缺点**: 实现复杂

#### 5.4.3 语义分块 (Semantic Splitting)

**方法**: 基于嵌入相似度确定分块边界

```
1. 计算每个句子的嵌入
2. 计算相邻句子的余弦相似度
3. 在相似度低谷处切分
```

**论文**: Liu et al., "Dense X Retrieval" (2022)

#### 5.4.4 分块参数调优

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `chunk_size` | 256-512 tokens | 太大: 信息冗余; 太小: 上下文不足 |
| `chunk_overlap` | 10-20% | 保持上下文连续性 |
| `max_context` | 2000-4000 tokens | LLM的上下文窗口限制 |

### 5.5 上下文压缩

**信息提取**:
```
原始文档: "机器学习是人工智能的一个分支..." (1000字)
压缩后: "机器学习核心: 监督学习、无监督学习、强化学习" (50字)
```

**方法**:
1. **LLM压缩**: 让LLM提取关键信息
2. **关键词提取**: 使用TF-IDF/TextRank
3. **句子选择**: 选择与查询最相关的句子

---

## 第六部分：评估与工程实践

### 6.1 检索质量评估

#### 6.1.1 召回率 (Recall@K)

**定义**: Top-K中相关文档的比例

```
Recall@K = |{相关文档} ∩ {Top-K}| / |{相关文档}|
```

#### 6.1.2 平均倒数排名 (MRR)

**定义**: 第一个相关文档排名倒数的平均值

```
MRR = (1/Q) Σ (1/rank_i)
```

其中`rank_i`是查询i的第一个相关文档排名。

#### 6.1.3 NDCG (Normalized Discounted Cumulative Gain)

**定义**: 考虑位置权重的排序质量

```
DCG@K = Σ (2^rel_i - 1) / log2(i+1)
NDCG@K = DCG@K / IDCG@K
```

其中`rel_i`是第i个文档的相关性等级。

### 6.2 生成质量评估

#### 6.2.1 Faithfulness (忠实度)

**定义**: 答案与检索上下文的一致性

```
Faithfulness = (一致陈述数) / (总陈述数)
```

**评估方法**: 
- NLI模型判断
- 人工标注

#### 6.2.2 Relevance (相关性)

**定义**: 答案与问题的相关性

**评估方法**:
- LLM打分 (GPT-4作为裁判)
- 人工评分 (1-5分)

#### 6.2.3 RAGAS框架

**论文**: Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)

**指标组合**:
```
RAGAS = (Faithfulness + Relevance + Context Recall) / 3
```

### 6.3 工程最佳实践

#### 6.3.1 分块策略

```
chunk_size = 512
chunk_overlap = 50 (10%)
```

#### 6.3.2 Top-K选择

```
top_k = 3-5  # 平衡质量和速度
```

#### 6.3.3 混合检索权重

```
alpha = 0.5-0.7  # 偏向稠密检索
```

#### 6.3.4 重排序策略

```
Stage 1 (召回): Top-20
Stage 2 (精排): Top-5
```

#### 6.3.5 提示模板设计

```
基于以下上下文回答问题。如果上下文中没有相关信息，请明确说明。

上下文：
{context}

问题：{question}

要求：
1. 仅使用提供的上下文信息
2. 答案要准确、简洁
3. 如果不确定，请说明

回答：
```

---

## 参考文献

1. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" NeurIPS 2020
2. Karpukhin et al. "Dense Passage Retrieval for Open-Domain QA" EMNLP 2020
3. Reimers & Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" EMNLP 2019
4. Izacard & Grave. "Leveraging Passage Retrieval with Generative Models" EMNLP 2021
5. Robertson & Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond" Foundations and Trends in IR 2009
6. Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" SIGIR 2009
7. Nogueira & Cho. "Passage Re-ranking with BERT" BERTRank 2019
8. Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" ICLR 2024
9. Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels" EMNLP 2022
10. Mallen et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation" 2023

---

**最后更新**: 2026-01-02
**版本**: 2.0
**作者**: AI-Practices
