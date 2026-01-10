# 07-integrated-systems | 跨模块集成系统

> 整合多模态检索、RAG、Agent推理的端到端系统

---

## 目录结构

```
07-integrated-systems/
├── src/                          # 源代码
│   ├── multimodal_retriever.py   # 多模态检索器
│   ├── vision_qa_agent.py        # 视觉问答智能体
│   ├── pipeline.py               # 端到端流水线
│   ├── code_retriever.py         # 代码检索器
│   ├── code_agent.py             # 代码生成智能体
│   ├── review_agent.py           # 代码审查智能体
│   ├── rag_benchmark.py          # RAG性能测试
│   ├── agent_benchmark.py        # Agent性能测试
│   └── multimodal_benchmark.py   # 多模态性能测试
├── tests/                        # 单元测试 (109 tests)
├── notebooks/                    # Jupyter教程
│   ├── 01_MultimodalRetrieval_tutorial.ipynb
│   ├── 02_VisionQA_tutorial.ipynb
│   ├── 03_CodeAssistant_tutorial.ipynb
│   ├── 04_Benchmarks_tutorial.ipynb
│   └── 05_EndToEnd_tutorial.ipynb
├── 知识点.md                      # 技术知识文档
├── 使用教程.md                    # 使用指南
└── README.md
```

---

## 核心功能

| 模块 | 功能 | 关键类 |
|------|------|--------|
| 多模态检索 | CLIP图文联合检索 | MultimodalRetriever, CLIPEncoder |
| 视觉问答 | ReAct推理框架 | VisionQAAgent |
| 代码助手 | 检索+生成+审查 | CodeRetriever, CodeAgent, ReviewAgent |
| 性能测试 | 延迟/吞吐量/准确率 | RAGBenchmark, AgentBenchmark |

---

## 快速开始

```python
import sys
sys.path.append('src')

from multimodal_retriever import MultimodalRetriever, MultimodalDocument
from vision_qa_agent import VisionQAAgent

# 创建检索器
retriever = MultimodalRetriever()
retriever.add_document(MultimodalDocument(content="深度学习使用神经网络"))

# 创建智能体
agent = VisionQAAgent(retriever=retriever)
result = agent.answer("什么是深度学习?")
print(result.answer)
```

---

## 学习路径

1. **入门**: 阅读 `使用教程.md`
2. **实践**: 运行 `notebooks/` 中的Jupyter教程
3. **深入**: 阅读 `知识点.md` 理解原理
4. **测试**: 运行 `pytest tests/ -v`

---

## 运行测试

```bash
cd 09-practical-projects/07-integrated-systems
python -m pytest tests/ -v
```

---

[返回上级](../README.md)
