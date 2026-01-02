# 06-agents 知识点总结

## 1. Agent 基础概念

### 1.1 什么是AI Agent

AI Agent是能够感知环境、做出决策并采取行动的自主系统。与传统的问答系统不同，Agent具有：

- **自主性 (Autonomy)**: 能够独立决策，无需持续人类干预
- **反应性 (Reactivity)**: 能够响应环境变化
- **目标导向 (Goal-oriented)**: 朝着特定目标行动
- **工具使用 (Tool Use)**: 能够调用外部工具扩展能力
- **社交能力 (Social Ability)**: 可以与其他Agent或人类协作

### 1.2 Agent 与传统LLM的区别

| 特性 | 传统LLM | AI Agent |
|------|---------|----------|
| 输入 | 文本提示 | 文本 + 工具 + 环境 |
| 输出 | 文本响应 | 文本 + 动作 + 状态变化 |
| 记忆 | 上下文窗口 | 多层次记忆系统 |
| 能力 | 静态知识 | 动态工具调用 |
| 目标 | 回答问题 | 完成任务 |

### 1.3 Agent 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                      AI Agent 架构                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │             │  │             │  │             │        │
│  │  LLM Core   │  │    Tools    │  │   Memory    │        │
│  │  (大脑)     │  │   (工具)    │  │   (记忆)    │        │
│  │             │  │             │  │             │        │
│  │ - 推理      │  │ - Calculator│  │ - 短期记忆  │        │
│  │ - 规划      │  │ - Search    │  │ - 长期记忆  │        │
│  │ - 决策      │  │ - Code      │  │ - 向量记忆  │        │
│  │             │  │ - API       │  │             │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│                    ┌─────┴────────┐                         │
│                    │    Planner   │                         │
│                    │   (规划器)   │                         │
│                    │              │                         │
│                    │  - 任务分解  │                         │
│                    │  - 工具选择  │                         │
│                    │  - 步骤排序  │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## 2. 工具系统 (Tool System)

### 2.1 理论基础

#### Toolformer 核心思想

Toolformer (Schick et al., 2023) 的关键创新：

1. **Self-Supervised Learning**: 模型自己决定何时调用API
2. **Minimal Intervention**: 只在必要时插入API调用
3. **Text Interface**: 所有工具通过文本输入输出交互
4. **No Fine-tuning Required**: 基于现有LLM，无需微调

#### 工具调用流程

```
User Query → LLM → Tool Decision → Tool Execution → Result → LLM → Final Answer
                  ↓
            - Is tool needed?
            - Which tool?
            - What parameters?
```

### 2.2 工具设计原则

1. **明确边界**: 每个工具有清晰的功能范围
2. **文本接口**: 输入输出都是文本格式
3. **清晰描述**: 工具描述要让LLM能理解
4. **幂等性**: 相同输入产生相同输出
5. **错误恢复**: 失败时有清晰的错误信息
6. **安全第一**: 防止注入攻击和资源滥用

### 2.3 工具实现模式

```python
class Tool(ABC):
    """工具基类定义"""
    
    @property
    @abstractmethod
    def config(self) -> ToolConfig:
        """返回工具配置，包含名称、描述、参数定义"""
        pass
    
    @abstractmethod
    def _run(self, **kwargs) -> str:
        """执行工具逻辑的核心方法"""
        pass
    
    def run(self, **kwargs) -> ToolResult:
        """统一的执行入口，处理参数验证、异常处理、结果封装"""
        # 1. 参数验证
        self._validate_params(**kwargs)
        
        # 2. 执行逻辑
        try:
            output = self._run(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
        
        # 3. 结果封装
        return ToolResult(success=True, output=output)
```

### 2.4 内置工具详解

#### CalculatorTool

**设计要点**:
- 使用AST解析而非eval，防止代码注入
- 支持基本运算: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- 支持数学函数: `sqrt`, `sin`, `cos`, `log`, `exp`, `abs`, `round`, `ceil`, `floor`
- 常量支持: `pi`, `e`

**安全考虑**:
- 只允许安全的数学表达式
- 拒绝任何函数调用或导入语句
- 限制可用符号表

#### PythonREPLTool

**设计要点**:
- 受限的Python执行环境
- 限制可用内置函数和模块
- 设置执行超时
- 捕获所有异常

**安全边界**:
- 禁止文件系统操作
- 禁止网络请求
- 禁止子进程执行
- 限制内存使用

#### SearchTool

**功能**:
- 网络搜索接口
- 支持结构化结果返回
- 可配置结果数量
- 模拟实现（生产环境接入真实API）

#### VectorMemory

**技术原理**:
- 使用TF-IDF进行文本向量化
- 余弦相似度计算
- Top-K检索

**应用场景**:
- 语义搜索
- 知识检索
- 相关历史查找

### 2.5 工具安全考虑

| 安全威胁 | 防护措施 |
|----------|----------|
| 代码注入 | AST解析，禁用eval |
| 资源耗尽 | 执行超时，内存限制 |
| 恶意参数 | 类型验证，白名单过滤 |
| 无限循环 | 最大迭代限制 |
| 隐私泄露 | 敏感信息过滤 |

## 3. 记忆系统 (Memory System)

### 3.1 认知科学基础

#### 记忆的层次结构

```
┌─────────────────────────────────────────────────────┐
│          感知记忆 (Sensory Memory)                  │
│     ← 瞬时，0.5-3秒，容量大但衰减快                 │
├─────────────────────────────────────────────────────┤
│          工作记忆 (Working Memory)                 │
│     ← 短期，15-30秒，7±2项信息                     │
├─────────────────────────────────────────────────────┤
│          长期记忆 (Long-term Memory)               │
│  ┌──────────────┬────────────────────────────────┐  │
│  │  陈述性记忆  │          程序性记忆            │  │
│  │ (事实/知识)  │          (技能/习惯)           │  │
│  │              │                                │  │
│  │ - 语义记忆   │  - 条件反射                    │  │
│  │ - 情景记忆   │  - 运动技能                    │  │
│  └──────────────┴────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

#### LLM中的记忆映射

| 认知记忆 | LLM实现 | 特点 |
|----------|---------|------|
| 感知记忆 | 当前输入 | 单次交互 |
| 工作记忆 | 上下文窗口 | 短期保持 |
| 语义记忆 | 向量数据库 | 知识检索 |
| 情景记忆 | 对话历史 | 顺序记录 |

### 3.2 记忆类型详解

#### BufferMemory

**特点**:
- 保存所有对话历史
- 无信息丢失
- Token消耗线性增长

**适用场景**:
- 短对话（<10轮）
- 需要完整历史
- Token充足

**时间复杂度**:
- 添加: O(1)
- 检索: O(1)

#### WindowMemory

**特点**:
- 滑动窗口，保留最近k轮
- 固定Token消耗
- 早期信息丢失

**适用场景**:
- 中等对话（10-50轮）
- Token受限
- 关注近期上下文

**窗口大小选择**:
```
Token限制 → 推荐k值
────────────────────
4000     → k=3-5
8000     → k=5-10
16000+   → k=10-20
```

#### SummaryMemory

**特点**:
- 自动摘要压缩
- 保留关键信息
- 信息有损压缩

**摘要策略**:
1. **触发条件**: 消息数超过阈值
2. **摘要方法**: 
   - 提取关键信息
   - 压缩重复内容
   - 保留重要决策
3. **存储结构**: 摘要 + 最新消息

**适用场景**:
- 长对话（>50轮）
- 需要保留早期关键信息
- Token受限

#### VectorMemory

**技术原理**:
```
文本 → 分词 → TF-IDF向量 → 相似度计算 → Top-K检索
```

**相似度计算**:
```python
similarity = cos_similarity(query_vector, doc_vector)
           = (query · doc) / (||query|| * ||doc||)
```

**适用场景**:
- 知识密集型任务
- 语义检索需求
- 大量历史消息

### 3.3 记忆选择决策树

```
                    ┌─────────────┐
                    │  对话长度？  │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
         <10              10-50             >50
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │Token充足？│    │Token限制？│    │保留早期？ │
    └─────┬────┘    └─────┬────┘    └─────┬────┘
          │               │               │
     是/否│          是/否│          是/否│
          ▼               ▼               ▼
    BufferMemory   WindowMemory   SummaryMemory
    (或Vector)        (k=5-10)      (或混合)
```

### 3.4 记忆优化技术

#### 压缩策略

1. **去重**: 移除重复或相似消息
2. **摘要**: 提取关键信息
3. **分层**: 热数据在内存，冷数据持久化
4. **索引**: 建立快速检索索引

#### Token估算

```python
# 粗略估算（中英文混合）
def estimate_tokens(text: str) -> int:
    # 英文: 1 token ≈ 4 characters
    # 中文: 1 token ≈ 1.5-2 characters
    english_chars = len([c for c in text if ord(c) < 128])
    chinese_chars = len(text) - english_chars
    return int(english_chars / 4 + chinese_chars / 1.5)
```

## 4. Agent 架构模式

### 4.1 ReAct 模式

#### 论文核心

**"ReAct: Synergizing Reasoning and Acting in Language Models"** (Yao et al., 2022)

核心思想: 将推理（Reasoning）和行动（Acting）交织进行

#### 执行流程

```
循环:
    Thought:  思考当前情况
    Action:   选择工具
    Action Input: 工具参数
    ───────────────────────
    Observation: 观察结果
    ...
    Final Answer: 最终答案
```

#### 优点

- 推理过程透明
- 易于调试
- 适合复杂推理
- 可以自我纠正

#### 缺点

- Token消耗大
- 解析可能不稳定
- 需要模型有强推理能力

#### 提示模板

```
Answer the following question as best you can. You have access to the following tools:

{tools_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
```

### 4.2 ToolCalling 模式

#### 基础

OpenAI Function Calling API

#### 执行流程

```
User Message → LLM → Structured Output → Tool Execution → Result → LLM → Final Answer
                        ↓
                  {
                    "tool": "calculator",
                    "arguments": {"expression": "2+2"}
                  }
```

#### 优点

- 结构化输出
- 解析可靠性高
- Token效率高
- 不依赖特殊格式

#### 缺点

- 推理过程不透明
- 依赖模型的函数调用能力
- 调试困难

#### 适用模型

- GPT-3.5+/GPT-4
- Claude 3+
- 其他支持Function Calling的模型

### 4.3 Plan-and-Execute 模式

#### 执行流程

```
1. 规划阶段:
   - 分析任务
   - 分解步骤
   - 选择工具
   - 生成计划

2. 执行阶段:
   - 逐步执行
   - 观察结果
   - 调整计划

3. 调整阶段:
   - 评估进度
   - 修正计划
   - 继续执行
```

#### 优点

- 适合复杂任务
- 计划可调整
- 步骤清晰

#### 缺点

- 规划可能不准
- 需要多次迭代
- 时间成本高

#### 计划示例

```
任务: "帮我研究Python的最新发展并写一份报告"

计划:
1. 搜索"Python 2024新特性"
2. 搜索"Python性能改进"
3. 搜索"Python生态系统"
4. 整理信息
5. 生成报告
```

### 4.4 其他高级模式

#### Reflexion

**论文**: "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023)

核心: 自我反思和改进

```
Task → Attempt → Evaluate → Reflect → Adjust → Retry
                      ↓
                 What went wrong?
                 How to improve?
```

#### ToT (Tree of Thoughts)

**论文**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)

核心: 将思维过程表示为树，探索多种可能

```
                    Root
                  /  |  \
               B1   B2   B3
              / \       |
            C1  C2     C3
```

#### CoT (Chain of Thoughts)

**论文**: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)

核心: 通过中间推理步骤引导模型

```
Q: Roger有5个网球。他又买了2罐网球。
   每罐有3个网球。现在他有多少个网球？

A: Roger开始有5个球。
   2罐 × 3个/罐 = 6个球。
   5 + 6 = 11个球。
   答案: 11
```

## 5. Agent 运行机制

### 5.1 核心循环

```python
def agent_run(input_text: str) -> str:
    state = AgentState.IDLE
    intermediate_steps = []
    
    while iteration < max_iterations:
        # 1. 规划阶段
        state = AgentState.THINKING
        output = llm(generate_prompt(input_text, intermediate_steps))
        
        # 2. 解析响应
        parsed = parse_response(output)
        
        # 3. 检查是否完成
        if isinstance(parsed, AgentFinish):
            state = AgentState.FINISHED
            return parsed.output
        
        # 4. 执行动作
        state = AgentState.ACTING
        observation = execute_tool(parsed.tool, parsed.tool_input)
        intermediate_steps.append((parsed, observation))
        
        # 5. 更新状态
        iteration += 1
    
    return "达到最大迭代次数"
```

### 5.2 状态转换

```
IDLE → THINKING → ACTING → FINISHED
  ↑                          ↓
  └──────────────────────────┘
        (或 ERROR)
```

### 5.3 错误处理

```python
try:
    result = tool.run(**kwargs)
except ToolError as e:
    if config.handle_parsing_errors:
        # 将错误作为观察结果继续
        observation = f"错误: {str(e)}。请重试或使用其他工具。"
        intermediate_steps.append((action, observation))
    else:
        # 抛出异常终止
        raise AgentRuntimeError(f"Tool execution failed: {e}")
except Exception as e:
    # 未预期的错误
    observation = f"Unexpected error: {str(e)}"
    if config.early_stopping:
        raise AgentRuntimeError(observation)
```

## 6. Agent 设计最佳实践

### 6.1 工具设计

1. **单一职责原则**: 每个工具只做一件事
2. **清晰命名**: 名称应该描述功能
3. **详细描述**: 描述应包含使用场景
4. **参数验证**: 验证所有必需参数
5. **错误处理**: 返回清晰的错误信息
6. **幂等性**: 相同输入产生相同输出
7. **性能考虑**: 避免长时间运行

### 6.2 记忆管理

1. **合理选择**: 根据对话长度选择记忆类型
2. **系统消息**: 始终保留系统消息
3. **Token监控**: 定期检查Token使用
4. **重要信息**: 关键信息单独存储
5. **持久化**: 考虑长期记忆持久化
6. **压缩策略**: 对长对话使用摘要

### 6.3 Agent配置

1. **max_iterations**: 设置合理的上限防止无限循环
2. **max_execution_time**: 控制最大执行时间
3. **early_stopping**: 启用以提前终止
4. **handle_parsing_errors**: 提高鲁棒性
5. **return_intermediate_steps**: 便于调试
6. **verbose**: 开发时启用

### 6.4 调试技巧

1. **启用详细日志**: 打印中间步骤
2. **追踪执行**: 记录每个决策点
3. **测试工具**: 单独测试每个工具
4. **小步迭代**: 先简单后复杂
5. **边界测试**: 测试错误情况

### 6.5 性能优化

1. **并行执行**: 独立工具调用可并行
2. **缓存结果**: 相同输入使用缓存
3. **批量处理**: 合并相似操作
4. **延迟加载**: 按需加载工具
5. **资源限制**: 限制并发和资源使用

## 7. 多Agent协作

### 7.1 协作模式

#### 层次协作

```
Manager Agent
    ├── Research Agent
    ├── Analysis Agent
    └── Report Agent
```

#### 平行协作

```
Task 1 → Agent 1 ──┐
                  ├── Aggregator → Result
Task 2 → Agent 2 ──┘
```

#### 顺序协作

```
Agent 1 → Output 1 → Agent 2 → Output 2 → Agent 3 → Result
```

### 7.2 通信机制

- **共享记忆**: Agent间共享记忆
- **消息传递**: 显式消息通信
- **黑板模式**: 共享状态空间

## 8. 实战应用场景

### 8.1 代码助手

```
工具: CodeExecutor, FileBrowser, DocumentationSearch
记忆: BufferMemory + VectorMemory
模式: ReAct
```

### 8.2 数据分析

```
工具: PythonREPL, DataParser, Visualization
记忆: SummaryMemory
模式: PlanAndExecute
```

### 8.3 客服机器人

```
工具: KnowledgeSearch, OrderLookup, FAQ
记忆: VectorMemory + WindowMemory
模式: ToolCalling
```

### 8.4 研究助手

```
工具: WebSearch, PaperRetrieval, Summarizer
记忆: SummaryMemory + VectorMemory
模式: ReAct + Reflexion
```

## 9. 扩展阅读

### 核心论文

1. **ReAct**: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
2. **Toolformer**: [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
3. **Reflexion**: [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
4. **ToT**: [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2303.11112)
5. **CoT**: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

### 框架和库

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [BabyAGI](https://github.com/yoheinakajima/babyagi)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [LangGraph](https://github.com/langchain-ai/langgraph)

### 相关资源

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LLM Visualization](https://bbycroft.net/llm)
- [AI Agents Research](https://github.com/e2b-dev/awesome-ai-agents)
