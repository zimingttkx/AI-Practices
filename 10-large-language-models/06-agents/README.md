# 06-agents: AI Agent 系统

本模块实现了基于大语言模型的AI Agent系统，包括工具使用、记忆管理和多种Agent架构。深入理解Agent的工作原理，掌握构建智能Agent的核心技术。

## 目录

- [核心概念](#核心概念)
- [理论背景](#理论背景)
- [模块结构](#模块结构)
- [快速开始](#快速开始)
- [组件详解](#组件详解)
- [Agent模式](#agent模式)
- [最佳实践](#最佳实践)
- [扩展阅读](#扩展阅读)

---

## 核心概念

### 什么是AI Agent？

AI Agent是能够感知环境、推理决策并采取行动的自主系统。与传统的问答系统不同，Agent具有：

```
┌──────────────────────────────────────────────────────────┐
│                    AI Agent                              │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  输入感知 → 推理决策 → 行动执行 → 结果反馈                │
│     ↓          ↓          ↓          ↓                   │
│  文本/环境    LLM核心    工具调用    观察/学习             │
│                                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│  │  LLM    │  │  Tools  │  │ Memory  │                   │
│  │ (大脑)  │  │ (工具)  │  │ (记忆)  │                   │
│  └─────────┘  └─────────┘  └─────────┘                   │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Agent 核心能力

| 能力 | 描述 | 实现方式 |
|------|------|----------|
| **自主性** | 独立决策，无需持续人类干预 | 规划-执行循环 |
| **工具使用** | 调用外部API和服务 | 工具注册和调用机制 |
| **记忆管理** | 保存和检索历史信息 | 多层次记忆系统 |
| **推理能力** | 分解复杂任务 | Chain-of-Thought |
| **自我反思** | 从错误中学习 | Reflexion模式 |

---

## 理论背景

### 核心论文

本模块基于以下开创性研究：

| 论文 | 作者/年份 | 核心贡献 |
|------|-----------|----------|
| **ReAct** | Yao et al., 2022 | 推理与行动交织协同 |
| **Toolformer** | Schick et al., 2023 | 自主学习工具使用 |
| **Reflexion** | Shinn et al., 2023 | 自我反思改进 |
| **CoT** | Wei et al., 2022 | 思维链推理 |
| **ToT** | Yao et al., 2023 | 思维树探索 |

### ReAct 模式

ReAct (Reasoning + Acting) 是最经典的Agent模式：

```
循环:
    Thought:  我需要搜索关于Python的信息
    Action:   search
    Action Input: {"query": "Python programming"}
    ──────────────────────────────────────
    Observation: Python is a high-level...
    
    Thought:  现在我需要计算一个表达式
    Action:   calculator
    Action Input: {"expression": "2**10"}
    ──────────────────────────────────────
    Observation: 1024
    
    Thought:  我有足够信息回答了
    Final Answer: Python是高级编程语言，2的10次方是1024
```

---

## 模块结构

```
06-agents/
├── src/
│   ├── __init__.py         # 模块导出
│   ├── tools.py            # 工具系统实现
│   ├── memory.py           # 记忆管理实现
│   └── agent.py            # Agent核心实现
│
├── tests/
│   ├── test_tools.py       # 工具测试
│   ├── test_memory.py      # 记忆测试
│   ├── test_agent.py       # Agent测试
│   └── run_tests.py        # 测试运行器
│
├── notebooks/
│   ├── 01_tools_basics.ipynb       # 工具系统教程
│   ├── 02_memory_systems.ipynb     # 记忆系统教程
│   └── 03_agent_patterns.ipynb     # Agent模式教程
│
├── knowledge_points.md     # 知识点总结
└── README.md               # 本文档
```

---

## 快速开始

### 安装依赖

```bash
# 无需额外依赖，纯Python实现
# Python 3.8+ 即可运行
```

### 基础使用

#### 1. 使用工具系统

```python
from src.tools import (
    CalculatorTool, 
    SearchTool, 
    PythonREPLTool,
    ToolRegistry
)

# 创建工具
calc = CalculatorTool()
result = calc.run(expression="sqrt(144) + 10")
print(result.output)  # "22.0"

# 工具注册表
registry = ToolRegistry()
registry.register(CalculatorTool())
registry.register(SearchTool())
registry.register(PythonREPLTool())

# 查看所有工具
print(registry.get_tools_description())
```

#### 2. 使用记忆系统

```python
from src.memory import (
    BufferMemory, 
    WindowMemory,
    SummaryMemory,
    VectorMemory
)

# 缓冲记忆 - 保存所有历史
buffer = BufferMemory(system_message="你是Python专家")
buffer.add_user_message("什么是列表推导式？")
buffer.add_assistant_message("列表推导式是...")
print(f"消息数: {buffer.message_count}")

# 窗口记忆 - 只保留最近k轮
window = WindowMemory(k=5, system_message="你是助手")
# 自动滑动，保持最近5轮

# 向量记忆 - 语义检索
vector = VectorMemory(top_k=3)
vector.add_user_message("Python列表和元组的区别")
vector.add_user_message("如何使用pandas读取CSV")
# 检索相关记忆
results = vector.retrieve("Python数据结构")
```

#### 3. 使用Agent

```python
from src.agent import (
    ReActAgent,
    ToolCallingAgent,
    PlanAndExecuteAgent,
    AgentConfig
)

# 创建ReAct Agent
agent = ReActAgent(
    tools=[CalculatorTool(), SearchTool()],
    config=AgentConfig(
        max_iterations=10,
        return_intermediate_steps=True
    ),
    llm=your_llm_function  # 需要提供LLM函数
)

# 运行Agent
result = agent.run("计算 sqrt(144) + 10，然后搜索Python教程")
print(result)
```

---

## 组件详解

### 工具系统 (tools.py)

#### 内置工具

| 类 | 功能 | 安全特性 |
|---|------|----------|
| `CalculatorTool` | 数学计算 | AST安全解析，防注入 |
| `SearchTool` | 网络搜索 | 参数验证，结果过滤 |
| `PythonREPLTool` | 代码执行 | 受限环境，超时限制 |
| `WikipediaTool` | 百科查询 | API限流，缓存 |
| `DateTimeTool` | 日期时间 | 时区处理 |

#### 自定义工具

```python
from src.tools import Tool, ToolConfig, ToolResult

class WeatherTool(Tool):
    """天气查询工具"""
    
    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="weather",
            description="查询指定城市的天气信息",
            parameters={
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            required_params=["city"]
        )
    
    def _run(self, city: str, **kwargs) -> str:
        # 实现天气查询逻辑
        # 这里可以调用真实API
        weather_data = self._fetch_weather(city)
        return f"{city}: {weather_data}"

# 使用自定义工具
weather = WeatherTool()
result = weather.run(city="北京")
print(result.output)
```

### 记忆系统 (memory.py)

#### 记忆类型对比

| 类型 | Token消耗 | 信息保留 | 适用场景 |
|------|-----------|----------|----------|
| `BufferMemory` | 高 | 完整 | 短对话(<10轮) |
| `WindowMemory` | 固定 | 最近k轮 | 中等对话(10-50轮) |
| `SummaryMemory` | 低 | 摘要+最新 | 长对话(>50轮) |
| `VectorMemory` | 中 | 语义相关 | 知识检索 |

#### Message结构

```python
from src.memory import Message, MessageRole

# 创建消息
msg = Message(
    role=MessageRole.USER,
    content="帮我分析这段代码",
    metadata={"language": "python", "priority": "high"}
)

# 转换为字典格式（用于LLM）
dict_msg = msg.to_dict()
# {"role": "user", "content": "...", ...}
```

### Agent系统 (agent.py)

#### 配置选项

```python
from src.agent import AgentConfig

config = AgentConfig(
    # 最大迭代次数，防止无限循环
    max_iterations=10,
    
    # 最大执行时间（秒）
    max_execution_time=120.0,
    
    # 提前停止策略
    early_stopping=True,
    
    # 返回中间步骤（便于调试）
    return_intermediate_steps=True,
    
    # 处理解析错误
    handle_parsing_errors=True,
    
    # 详细输出（调试用）
    verbose=False
)
```

#### 状态管理

Agent运行时的状态转换：

```
IDLE → THINKING → ACTING → FINISHED
  ↑                          ↓
  └──────────────────────────┘
        (或 ERROR)
```

---

## Agent模式

### ReAct Agent

**特点**: 推理过程透明，易于调试

```python
agent = ReActAgent(
    tools=[CalculatorTool(), SearchTool()],
    config=AgentConfig(max_iterations=10)
)

# 运行
result = agent.run("计算100的平方根，然后搜索平方根的应用")
```

**执行流程**:
1. Thought: 思考下一步
2. Action: 选择工具
3. Action Input: 工具参数
4. Observation: 观察结果
5. 重复直到完成

### ToolCalling Agent

**特点**: 结构化输出，解析可靠

```python
agent = ToolCallingAgent(
    tools=[CalculatorTool(), DateTimeTool()],
    llm=your_llm_function
)

# 依赖LLM的Function Calling能力
result = agent.run("计算2+2，并给出7天后的日期")
```

**LLM响应格式**:
```json
{
  "tool": "calculator",
  "arguments": {
    "expression": "2+2"
  }
}
```

### PlanAndExecute Agent

**特点**: 先规划后执行，适合复杂任务

```python
agent = PlanAndExecuteAgent(
    tools=[CalculatorTool(), SearchTool()],
    config=AgentConfig(max_iterations=15)
)

# 自动分解任务并执行
result = agent.run("研究Python性能优化技巧并写一份报告")
```

**执行流程**:
1. 规划阶段: 分解任务为步骤
2. 执行阶段: 逐步执行每个步骤
3. 调整阶段: 根据结果调整计划

### 模式对比

| 特性 | ReAct | ToolCalling | PlanAndExecute |
|------|-------|-------------|----------------|
| 推理透明度 | 高 | 低 | 中 |
| Token效率 | 低 | 高 | 中 |
| 解析可靠性 | 中 | 高 | 中 |
| 复杂任务能力 | 强 | 中 | 最强 |
| 调试难度 | 低 | 中 | 高 |

---

## 最佳实践

### 工具设计原则

1. **单一职责**: 每个工具只做一件事
2. **清晰描述**: 描述要让LLM能理解
3. **参数验证**: 验证所有必需参数
4. **错误处理**: 返回清晰的错误信息
5. **安全第一**: 防止注入和资源滥用

### 记忆选择策略

```
对话长度 → 推荐记忆类型
────────────────────────────
< 10轮    → BufferMemory
10-50轮   → WindowMemory(k=5-10)
> 50轮    → SummaryMemory
知识检索  → VectorMemory
混合需求  → 组合使用
```

### Agent配置建议

| 场景 | max_iterations | return_intermediate_steps | verbose |
|------|----------------|---------------------------|---------|
| 开发调试 | 15 | True | True |
| 生产环境 | 5-10 | False | False |
| 复杂任务 | 15-20 | True | False |

### 常见问题解决

#### 问题1: Agent陷入无限循环

**解决方案**:
- 设置合理的`max_iterations`
- 启用`early_stopping`
- 检查工具是否返回有效结果

#### 问题2: 工具调用解析失败

**解决方案**:
- 启用`handle_parsing_errors`
- 优化提示模板
- 使用ToolCalling模式（如果模型支持）

#### 问题3: Token超出限制

**解决方案**:
- 使用WindowMemory或SummaryMemory
- 减少工具描述长度
- 启用自动摘要

---

## 运行测试

```bash
cd 06-agents

# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_tools.py -v

# 运行测试并显示覆盖率
python -m pytest tests/ --cov=src --cov-report=html
```

### 测试覆盖

- `test_tools.py`: 工具系统的所有功能
- `test_memory.py`: 各种记忆类型
- `test_agent.py`: Agent运行循环

---

## 扩展阅读

### 论文

- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Reasoning and Acting
- [Toolformer Paper](https://arxiv.org/abs/2302.04761) - Language Models Can Teach Themselves to Use Tools
- [Reflexion Paper](https://arxiv.org/abs/2303.11366) - Language Agents with Verbal Reinforcement Learning
- [Chain of Thought](https://arxiv.org/abs/2201.11903) - Elicits Reasoning in LLMs
- [Tree of Thoughts](https://arxiv.org/abs/2303.11112) - Deliberate Problem Solving

### 框架和库

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/) - 流行的Agent框架
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) - 官方函数调用指南
- [LangGraph](https://github.com/langchain-ai/langgraph) - 构建有状态的Agent
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) - 自主Agent
- [BabyAGI](https://github.com/yoheinakajima/babyagi) - 任务管理Agent

### 资源

- [Prompt Engineering Guide](https://www.promptingguide.ai/) - 提示工程指南
- [Awesome AI Agents](https://github.com/e2b-dev/awesome-ai-agents) - Agent资源列表
- [LLM Visualization](https://bbycroft.net/llm) - LLM可视化

---

## 依赖

- Python 3.8+
- 无外部依赖（纯Python实现）

---

## 许可证

MIT License

---

## 贡献

欢迎提交Issue和Pull Request！

---

## 作者

AI-Practices 项目组

---

**最后更新**: 2024年1月
