# 03-memory-systems: AI Agent 记忆系统

## 概述

本模块实现了 AI Agent 的完整记忆系统框架，包括短期记忆（对话上下文管理）、长期记忆（持久知识存储）和智能检索策略。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Systems                            │
├─────────────────────────────────────────────────────────────┤
│  Short-Term Memory    │    Long-Term Memory                 │
│  ├─ ConversationBuffer│    ├─ VectorStore                   │
│  ├─ SlidingWindow     │    ├─ MemoryEntry                   │
│  ├─ SummaryMemory     │    └─ Embedding                     │
│  └─ TokenBasedMemory  │                                     │
├─────────────────────────────────────────────────────────────┤
│                    Memory Retrieval                          │
│  ├─ SimilarityRetrieval  ├─ RecencyRetrieval                │
│  ├─ ImportanceRetrieval  └─ HybridRetrieval                 │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

```python
from src import (
    ConversationBuffer, SlidingWindowMemory,
    LongTermMemory, MemoryType,
    MemoryRetriever, HybridRetrieval
)

# 短期记忆
memory = SlidingWindowMemory(window_size=10)
memory.add_user_message("你好")
memory.add_assistant_message("你好！有什么可以帮助你的？")

# 长期记忆
ltm = LongTermMemory()
ltm.store("用户偏好深色模式", memory_type=MemoryType.PREFERENCE)
results = ltm.recall("用户界面偏好", k=5)

# 智能检索
retriever = MemoryRetriever(ltm, strategy=HybridRetrieval())
results = retriever.retrieve("偏好设置", k=3)
```

## 模块结构

```
03-memory-systems/
├── src/
│   ├── __init__.py           # 模块入口
│   ├── short_term_memory.py  # 短期记忆实现
│   ├── long_term_memory.py   # 长期记忆实现
│   └── memory_retrieval.py   # 检索策略
├── tests/
│   ├── test_short_term_memory.py
│   ├── test_long_term_memory.py
│   └── test_memory_retrieval.py
├── notebooks/
│   ├── 01_ShortTermMemory_tutorial.ipynb
│   ├── 02_LongTermMemory_tutorial.ipynb
│   └── 03_MemoryRetrieval_tutorial.ipynb
├── 知识点.md
└── README.md
```

## 核心组件

### 短期记忆策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| ConversationBuffer | 保存所有消息 | 短对话 |
| SlidingWindowMemory | 保留最近 k 条 | 长对话 |
| SummaryMemory | 摘要旧消息 | 超长会话 |
| TokenBasedMemory | Token 预算管理 | 成本控制 |

### 检索策略

| 策略 | 公式 | 适用场景 |
|------|------|----------|
| Similarity | sim(q,m) | 知识检索 |
| Recency | e^(-λt) | 最近上下文 |
| Importance | importance | 关键信息 |
| Hybrid | α·sim + β·rec + γ·imp | 通用 |

## 运行测试

```bash
cd 14-agents-reasoning/03-memory-systems
python -m pytest tests/ -v
```

## 参考文献

- MemGPT: Towards LLMs as Operating Systems (2023)
- Generative Agents: Interactive Simulacra (2023)
- RAG: Retrieval-Augmented Generation (2020)
