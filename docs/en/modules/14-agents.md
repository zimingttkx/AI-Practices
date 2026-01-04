# 14 - Agents & Reasoning

This module covers core LLM agent technologies including tool use, reasoning strategies, memory systems, and multi-agent collaboration.

## Module Structure

```
14-agents-reasoning/
├── 01-tool-use/                # Tool Calling
├── 02-reasoning/               # Reasoning Strategies
├── 03-memory-systems/          # Memory Systems
├── 04-planning/                # Task Planning
└── 05-multi-agent/             # Multi-Agent
```

## Core Content

### 01 - Tool Use

| Technology | Description |
|:-----------|:------------|
| Function Calling | OpenAI-style function definitions |
| Tool Registry | Decorator registration, tag management |
| Tool Executor | Timeout control, retry mechanism |
| Structured Output | JSON Schema constraints |

### 02 - Reasoning Strategies

| Technology | Description | Use Case |
|:-----------|:------------|:---------|
| Chain-of-Thought | Step-by-step reasoning | Complex reasoning |
| ReAct | Reasoning + Action loop | Tool usage |
| Tree-of-Thoughts | Tree search exploration | Exploratory problems |
| Self-Consistency | Multi-path voting | Improve accuracy |
| Reflection | Self-correction | Error fixing |

### 03 - Memory Systems

| Type | Description |
|:-----|:------------|
| Short-term | Conversation context window |
| Long-term | Vector database storage |
| Episodic | Historical interaction records |
| Semantic | Knowledge graphs |

### 04 - Task Planning

| Technology | Description |
|:-----------|:------------|
| Task Decomposition | Break down complex tasks |
| Plan Generation | Step planning |
| Plan Execution | Monitoring and adjustment |
| Re-planning | Failure recovery |

### 05 - Multi-Agent

| Pattern | Description |
|:--------|:------------|
| Debate | Multiple agents discuss to reach consensus |
| Collaborative | Division of labor |
| Hierarchical | Main agent coordinates sub-agents |

## Learning Path

```
Function Calling → CoT/ReAct → ToT → Memory → Planning → Multi-Agent
```
