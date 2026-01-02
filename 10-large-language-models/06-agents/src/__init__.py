"""
AI Agent 模块

本模块提供完整的AI Agent实现，基于论文
"ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
和 "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)。

核心组件：
    - Tool: 工具基类与内置工具
    - Memory: 对话记忆管理
    - Agent: Agent核心实现

作者: AI-Practices
许可证: MIT
"""

from .tools import (
    Tool,
    ToolConfig,
    ToolResult,
    ToolRegistry,
    CalculatorTool,
    SearchTool,
    PythonREPLTool,
    WikipediaTool,
    DateTimeTool,
)

from .memory import (
    Message,
    MessageRole,
    ConversationMemory,
    BufferMemory,
    WindowMemory,
    SummaryMemory,
    VectorMemory,
)

from .agent import (
    AgentConfig,
    AgentState,
    AgentAction,
    AgentFinish,
    BaseAgent,
    ReActAgent,
    ToolCallingAgent,
    PlanAndExecuteAgent,
)


__all__ = [
    # Tools
    "Tool",
    "ToolConfig",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "SearchTool",
    "PythonREPLTool",
    "WikipediaTool",
    "DateTimeTool",
    # Memory
    "Message",
    "MessageRole",
    "ConversationMemory",
    "BufferMemory",
    "WindowMemory",
    "SummaryMemory",
    "VectorMemory",
    # Agent
    "AgentConfig",
    "AgentState",
    "AgentAction",
    "AgentFinish",
    "BaseAgent",
    "ReActAgent",
    "ToolCallingAgent",
    "PlanAndExecuteAgent",
]
