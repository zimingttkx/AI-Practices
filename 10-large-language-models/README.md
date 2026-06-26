# 大语言模型 (Large Language Models)

## 前置知识

- ✅ 必需：深度学习基础、PyTorch、Transformer 架构
- 📖 推荐：自然语言处理基础、注意力机制原理
- 🔗 前序模块：[04-sequence-models](../04-sequence-models)、[02-neural-networks](../02-neural-networks)

## 概述

从基础理论到前沿技术，涵盖 Transformer 架构、预训练、微调、对齐和部署。

## 核心概念

| 技术 | 描述 | 代表模型 |
|:-----|:-----|:---------|
| **Transformer** | 自注意力架构 | GPT, BERT, LLaMA |
| **预训练** | 大规模自监督学习 | GPT-3, LLaMA-2 |
| **微调** | 适配下游任务 | LoRA, QLoRA |
| **对齐** | 人类偏好对齐 | RLHF, DPO |
| **推理加速** | 高效部署 | KV Cache, Speculative Decoding |

## 学习路径

### 01. LLM 基础
- `tokenizer_architecture.ipynb` - 分词器原理 (BPE, WordPiece, SentencePiece)
- `transformer_architecture.ipynb` - Transformer 架构详解
- `positional_encoding.ipynb` - 位置编码 (Sinusoidal, RoPE, ALiBi)
- `attention_mechanisms.ipynb` - 注意力机制 (MHA, GQA, MQA, Flash Attention)

### 02. 预训练模型
- `gpt_architecture.ipynb` - GPT 系列 (GPT-2, GPT-3, GPT-4)
- `bert_architecture.ipynb` - BERT 系列 (BERT, RoBERTa, DeBERTa)
- `llama_architecture.ipynb` - LLaMA 系列 (LLaMA, LLaMA-2/3)
- `scaling_laws.ipynb` - 缩放定律

### 03. 微调方法
- `full_finetuning.ipynb` - 全量微调
- `peft_finetuning.ipynb` - 参数高效微调 (LoRA, Adapter, Prefix)
- `qlora_finetuning.ipynb` - QLoRA 量化微调
- `instruction_tuning.ipynb` - 指令微调

### 04. 对齐技术
- `rlhf.ipynb` - 基于人类反馈的强化学习
- `dpo.ipynb` - 直接偏好优化
- `ppo_training.ipynb` - PPO 训练

### 05. 推理优化
- `kv_cache.ipynb` - KV Cache 优化
- `speculative_decoding.ipynb` - 推测解码
- `quantization_inference.ipynb` - 推理量化

### 06. 应用模式
- `rag.ipynb` - 检索增强生成
- `agents.ipynb` - LLM Agents
- `function_calling.ipynb` - Function Calling
- `multiturn_conversation.ipynb` - 多轮对话

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
