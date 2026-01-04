"""
Tool Use: Comprehensive Framework for AI Agent Tool Invocation

Core Idea:
    This module provides a complete implementation of the Tool Use paradigm for
    AI agents, enabling Large Language Models to interact with external systems
    through structured function calls. It bridges the gap between natural language
    understanding and programmatic execution.

Architecture Overview:
    The framework consists of four interconnected components:

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Tool Use Framework                        │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  ┌──────────────────┐    ┌──────────────────┐                   │
    │  │ Function Calling │───▶│  Tool Registry   │                   │
    │  │   (Schema Gen)   │    │   (Management)   │                   │
    │  └──────────────────┘    └────────┬─────────┘                   │
    │           │                       │                              │
    │           ▼                       ▼                              │
    │  ┌──────────────────┐    ┌──────────────────┐                   │
    │  │ Structured Output│◀───│  Tool Executor   │                   │
    │  │    (Parsing)     │    │   (Execution)    │                   │
    │  └──────────────────┘    └──────────────────┘                   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    ```

Component Responsibilities:
    1. **Function Calling** (function_calling.py):
       - Schema generation from Python functions
       - Multi-format export (OpenAI, Anthropic)
       - Function call parsing and validation

    2. **Tool Registry** (tool_registry.py):
       - Centralized tool registration and discovery
       - Tag-based filtering and organization
       - Lifecycle management (enable/disable)

    3. **Tool Executor** (tool_executor.py):
       - Safe execution with timeout control
       - Retry mechanisms and error handling
       - Execution history and statistics

    4. **Structured Output** (structured_output.py):
       - JSON extraction from LLM responses
       - Pydantic and JSON Schema validation
       - Automatic error correction

Mathematical Foundation:
    The tool use pipeline can be formalized as:

    $$\text{Agent}: \mathcal{Q} \times \mathcal{T} \rightarrow \mathcal{A}$$

    where:
    - $\mathcal{Q}$ is the query space (user requests)
    - $\mathcal{T}$ is the tool space (available functions)
    - $\mathcal{A}$ is the action space (tool invocations + responses)

Usage Example:
    >>> from src import ToolRegistry, ToolExecutor, FunctionCallParser
    >>>
    >>> # 1. Create and populate registry
    >>> registry = ToolRegistry()
    >>> @registry.register(tags=["math"])
    ... def add(a: int, b: int) -> int:
    ...     '''Add two numbers.'''
    ...     return a + b
    >>>
    >>> # 2. Export schemas for LLM
    >>> tools = registry.to_openai_tools()
    >>>
    >>> # 3. Parse LLM response
    >>> parser = FunctionCallParser([t.to_function_definition() for t in registry])
    >>> calls = parser.parse(llm_response)
    >>>
    >>> # 4. Execute tools
    >>> executor = ToolExecutor(registry)
    >>> results = executor.execute_batch(calls)

References:
    - OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
    - Anthropic Tool Use: https://docs.anthropic.com/claude/docs/tool-use
    - ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)
    - Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)
"""

from __future__ import annotations

# Function Calling: Schema generation and parsing
from .function_calling import (
    ParameterType,
    FunctionParameter,
    FunctionDefinition,
    FunctionCall,
    FunctionCallParser,
    create_function_schema,
)

# Tool Registry: Registration and management
from .tool_registry import (
    Tool,
    ToolRegistry,
    tool,
    get_default_registry,
    reset_default_registry,
)

# Tool Executor: Safe execution engine
from .tool_executor import (
    ExecutionStatus,
    ExecutionResult,
    ExecutionContext,
    ToolExecutor,
)

# Structured Output: Response parsing and validation
from .structured_output import (
    ValidationError,
    OutputSchema,
    StructuredOutputParser,
    OutputFormat,
    create_list_parser,
    create_choice_parser,
    create_extraction_parser,
)


__version__ = "1.0.0"
__author__ = "AI-Practices"


__all__ = [
    # Version info
    "__version__",
    "__author__",
    # function_calling
    "ParameterType",
    "FunctionParameter",
    "FunctionDefinition",
    "FunctionCall",
    "FunctionCallParser",
    "create_function_schema",
    # tool_registry
    "Tool",
    "ToolRegistry",
    "tool",
    "get_default_registry",
    "reset_default_registry",
    # tool_executor
    "ExecutionStatus",
    "ExecutionResult",
    "ExecutionContext",
    "ToolExecutor",
    # structured_output
    "ValidationError",
    "OutputSchema",
    "StructuredOutputParser",
    "OutputFormat",
    "create_list_parser",
    "create_choice_parser",
    "create_extraction_parser",
]
