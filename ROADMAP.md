# AI-Practices 项目路线图

> **最后更新**: 2026-01-02 | **当前阶段**: LLM模块完善

---

## 进度总览

| 模块 | 状态 | 完成度 |
|:-----|:-----|:-------|
| 01-foundations | 完成 | 100% |
| 02-neural-networks | 完成 | 100% |
| 03-computer-vision | 完成 | 100% |
| 04-sequence-models | 完成 | 100% |
| 05-advanced-topics | 完成 | 100% |
| 06-generative-models | 完成 | 100% |
| 07-reinforcement-learning | 完成 | 100% |
| 08-theory-notes | 完成 | 100% |
| 09-practical-projects | 完成 | 100% |
| 10-large-language-models | 完成 | 100% |

---

## 一、已完成内容

### 10-large-language-models 模块

```
10-large-language-models/
├── 01-llm-fundamentals/          # Transformer架构、Tokenizer
├── 02-pretrained-models/         # GPT/BERT/LLaMA实现
├── 03-fine-tuning/               # LoRA/QLoRA微调
├── 04-prompt-engineering/        # 提示工程技术
├── 05-rag/                       # 检索增强生成
├── 06-agents/                    # Agent系统与工具调用
└── 07-alignment/                 # RLHF/DPO对齐训练
```

**各子模块内容**:

- **04-prompt-engineering**: 模板系统、Few-shot学习、思维链推理
- **05-rag**: 嵌入模型、向量存储、检索器、RAG流水线
- **06-agents**: 工具系统、记忆机制、ReAct/多Agent模式
- **07-alignment**: 奖励模型、PPO训练、DPO直接偏好优化

---

## 二、下一步计划

### 11-multimodal-learning (规划中)

```
11-multimodal-learning/
├── 01-vision-language/           # CLIP、BLIP、LLaVA
├── 02-image-generation/          # Stable Diffusion、ControlNet
└── 03-audio-models/              # Whisper、TTS
```

### 工程化提升

- 测试覆盖率提升
- Docker容器化支持
- CI/CD流程完善

---

## 三、时间规划

| 阶段 | 内容 | 状态 |
|:----:|:-----|:----:|
| Phase 1 | Transformer模块 | 完成 |
| Phase 2 | 生成式模型 | 完成 |
| Phase 3 | LLM完整模块 | 完成 |
| Phase 4 | 多模态学习 | 待开发 |

---

## 四、依赖说明

主要依赖包括:
- transformers, peft, accelerate (LLM相关)
- langchain, chromadb, faiss-cpu (RAG相关)
- sentence-transformers (向量嵌入)
- pytest, black, ruff (开发工具)

详见 `requirements.txt`
