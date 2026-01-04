# 10 - 大语言模型

本模块系统讲解大语言模型（LLM）的核心技术，从Transformer架构到实际应用。

## 模块结构

```
10-large-language-models/
├── 01-llm-fundamentals/        # LLM基础
├── 02-pretrained-models/       # 预训练模型
├── 03-fine-tuning/             # 微调技术
├── 04-prompt-engineering/      # 提示工程
├── 05-rag/                     # 检索增强生成
├── 06-agents/                  # Agent系统
└── 07-alignment/               # 对齐训练
```

## 核心内容

### 01 - LLM基础

| 主题 | 内容 |
|:-----|:-----|
| Transformer架构 | 自注意力机制、位置编码、多头注意力 |
| Tokenizer | BPE、WordPiece、SentencePiece |
| 预训练目标 | 语言建模、掩码语言建模 |

### 02 - 预训练模型

| 模型 | 特点 |
|:-----|:-----|
| GPT系列 | 自回归生成、因果注意力 |
| BERT | 双向编码、掩码预测 |
| LLaMA | 高效架构、开源可用 |

### 03 - 微调技术

| 技术 | 说明 |
|:-----|:-----|
| 全参数微调 | 更新所有参数 |
| LoRA | 低秩适配，参数高效 |
| QLoRA | 量化+LoRA，显存友好 |
| Adapter | 插入适配层 |

### 04 - 提示工程

| 技术 | 应用 |
|:-----|:-----|
| Zero-shot | 无示例推理 |
| Few-shot | 少样本学习 |
| Chain-of-Thought | 思维链推理 |
| ReAct | 推理+行动 |

### 05 - 检索增强生成 (RAG)

| 组件 | 功能 |
|:-----|:-----|
| 文档切分 | 文本分块策略 |
| 向量嵌入 | 文本向量化 |
| 向量数据库 | FAISS、Chroma |
| 检索策略 | 相似度检索、混合检索 |

### 06 - Agent系统

| 组件 | 功能 |
|:-----|:-----|
| 工具调用 | Function Calling |
| 记忆管理 | 上下文管理 |
| 任务规划 | 任务分解与执行 |

### 07 - 对齐训练

| 技术 | 说明 |
|:-----|:-----|
| RLHF | 人类反馈强化学习 |
| DPO | 直接偏好优化 |
| Constitutional AI | 宪法AI |

## 学习路线

```
LLM基础 → 预训练模型 → 微调技术 → 提示工程 → RAG → Agent → 对齐
```

## 依赖库

```python
transformers>=4.30.0
peft>=0.5.0
accelerate>=0.21.0
langchain>=0.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
```
