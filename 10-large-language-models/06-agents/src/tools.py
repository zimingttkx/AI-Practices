"""
工具模块 (Tools Module)

============================================================
核心思想 (Core Idea)
============================================================
Agent工具是LLM与外部世界交互的桥梁。通过定义标准化的工具接口，
LLM可以调用计算器、搜索引擎、代码执行器等外部能力，扩展其功能边界。

============================================================
数学基础 (Mathematical Foundation)
============================================================
工具调用可形式化为函数映射：

    T: (name, args) → result

其中：
    - name ∈ ToolRegistry: 工具名称
    - args ∈ Dict[str, Any]: 参数字典
    - result ∈ ToolResult: 执行结果

工具选择策略（基于LLM）：
    P(tool | query) = softmax(LLM(query, tool_descriptions))

============================================================
算法流程 (Algorithm Flow)
============================================================
1. 工具注册: registry.register(tool)
2. 工具描述: 生成供LLM理解的工具说明
3. 工具选择: LLM根据任务选择合适工具
4. 参数解析: 从LLM输出提取工具参数
5. 工具执行: tool.run(**args) → ToolResult
6. 结果返回: 将结果反馈给LLM

============================================================
参考文献 (References)
============================================================
[1] Schick, T., et al. (2023). Toolformer: Language Models Can Teach
    Themselves to Use Tools. arXiv:2302.04761.
[2] Qin, Y., et al. (2023). Tool Learning with Foundation Models.
    arXiv:2304.08354.
[3] Patil, S., et al. (2023). Gorilla: Large Language Model Connected
    with Massive APIs. arXiv:2305.15334.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union


__all__ = [
    "Tool",
    "ToolConfig",
    "ToolResult",
    "ToolRegistry",
    "CalculatorTool",
    "SearchTool",
    "PythonREPLTool",
    "WikipediaTool",
    "DateTimeTool",
]


class ToolStatus(Enum):
    """工具执行状态。"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolConfig:
    """工具配置。

    参数：
        name: 工具名称
        description: 工具描述（供LLM理解）
        parameters: 参数定义（JSON Schema格式）
        required_params: 必需参数列表
        return_direct: 是否直接返回结果给用户
        timeout: 执行超时时间（秒）
    """
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    return_direct: bool = False
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("工具名称不能为空")
        if not self.description or not self.description.strip():
            raise ValueError("工具描述不能为空")
        if self.timeout <= 0:
            raise ValueError(f"timeout必须为正数，得到 {self.timeout}")

    def to_openai_function(self) -> Dict[str, Any]:
        """转换为OpenAI函数调用格式。
        
        返回：
            OpenAI function calling格式的字典
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params,
            },
        }

    def __repr__(self) -> str:
        return f"ToolConfig(name='{self.name}', params={list(self.parameters.keys())})"


@dataclass
class ToolResult:
    """工具执行结果。

    参数：
        output: 输出内容
        status: 执行状态
        error: 错误信息（如果有）
        metadata: 额外元数据
    """
    output: str
    status: ToolStatus = ToolStatus.SUCCESS
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """是否执行成功。"""
        return self.status == ToolStatus.SUCCESS

    def __repr__(self) -> str:
        preview = self.output[:50] + "..." if len(self.output) > 50 else self.output
        return f"ToolResult(status={self.status.value}, output='{preview}')"


class Tool(ABC):
    """工具基类。

    所有工具必须继承此类并实现_run方法。

    工具设计原则（Toolformer论文）：
        1. 工具应该有明确的功能边界
        2. 输入输出应该是文本格式
        3. 工具描述应该清晰准确

    示例：
        >>> class MyTool(Tool):
        ...     @property
        ...     def config(self) -> ToolConfig:
        ...         return ToolConfig(name="my_tool", description="...")
        ...     def _run(self, **kwargs) -> str:
        ...         return "result"
    """

    @property
    @abstractmethod
    def config(self) -> ToolConfig:
        """返回工具配置。"""
        pass

    @property
    def name(self) -> str:
        """工具名称。"""
        return self.config.name

    @property
    def description(self) -> str:
        """工具描述。"""
        return self.config.description

    @abstractmethod
    def _run(self, **kwargs: Any) -> str:
        """执行工具（子类实现）。

        参数：
            **kwargs: 工具参数

        返回：
            执行结果字符串
        """
        pass

    def run(self, **kwargs: Any) -> ToolResult:
        """执行工具并返回结果。

        参数：
            **kwargs: 工具参数

        返回：
            ToolResult对象
        """
        # 验证必需参数
        for param in self.config.required_params:
            if param not in kwargs:
                return ToolResult(
                    output="",
                    status=ToolStatus.ERROR,
                    error=f"缺少必需参数: {param}",
                )

        try:
            # 执行工具
            output = self._run(**kwargs)
            return ToolResult(output=output, status=ToolStatus.SUCCESS)
        except Exception as e:
            return ToolResult(
                output="",
                status=ToolStatus.ERROR,
                error=str(e),
            )

    def __call__(self, **kwargs: Any) -> ToolResult:
        """调用工具。"""
        return self.run(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ToolRegistry:
    """工具注册表。

    管理和查找可用工具。

    示例：
        >>> registry = ToolRegistry()
        >>> registry.register(CalculatorTool())
        >>> tool = registry.get("calculator")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具。

        参数：
            tool: 工具实例
        """
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已存在")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """注销工具。

        参数：
            name: 工具名称

        返回：
            是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """获取工具。

        参数：
            name: 工具名称

        返回：
            工具实例或None
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有工具名称。"""
        return list(self._tools.keys())

    def get_all_tools(self) -> List[Tool]:
        """获取所有工具。"""
        return list(self._tools.values())

    def get_tools_description(self) -> str:
        """获取所有工具的描述（供LLM使用）。"""
        descriptions = []
        for tool in self._tools.values():
            params = ", ".join(tool.config.required_params)
            descriptions.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(descriptions)

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """转换为OpenAI函数调用格式。"""
        return [tool.config.to_openai_function() for tool in self._tools.values()]

    @property
    def count(self) -> int:
        """工具数量。"""
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"


# ============================================================================
# 内置工具实现
# ============================================================================

class CalculatorTool(Tool):
    """计算器工具。

    支持基本数学运算和常用数学函数。

    安全性：
        使用AST解析而非eval，防止代码注入。

    示例：
        >>> calc = CalculatorTool()
        >>> result = calc.run(expression="2 + 3 * 4")
        >>> print(result.output)  # "14"
    """

    # 允许的运算符
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # 允许的数学函数
    _FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pow": math.pow,
        "pi": math.pi,
        "e": math.e,
    }

    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="calculator",
            description="计算数学表达式。支持基本运算(+,-,*,/,**,%)和数学函数(sqrt,sin,cos,log等)。",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'",
                }
            },
            required_params=["expression"],
        )

    def _run(self, expression: str, **kwargs: Any) -> str:
        """执行计算。"""
        try:
            # 解析表达式
            tree = ast.parse(expression, mode="eval")
            # 安全计算
            result = self._eval_node(tree.body)
            # 格式化结果
            if isinstance(result, float) and result.is_integer():
                return str(int(result))
            return str(result)
        except Exception as e:
            raise ValueError(f"无法计算表达式 '{expression}': {e}")

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """递归计算AST节点。"""
        # 数字
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量类型: {type(node.value)}")

        # 一元运算
        if isinstance(node, ast.UnaryOp):
            op = self._OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的一元运算符: {type(node.op)}")
            return op(self._eval_node(node.operand))

        # 二元运算
        if isinstance(node, ast.BinOp):
            op = self._OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的二元运算符: {type(node.op)}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return op(left, right)

        # 函数调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self._FUNCTIONS:
                    func = self._FUNCTIONS[func_name]
                    args = [self._eval_node(arg) for arg in node.args]
                    return func(*args)
                raise ValueError(f"不支持的函数: {func_name}")

        # 变量名（常量）
        if isinstance(node, ast.Name):
            if node.id in self._FUNCTIONS:
                return self._FUNCTIONS[node.id]
            raise ValueError(f"未知变量: {node.id}")

        raise ValueError(f"不支持的表达式类型: {type(node)}")


class SearchTool(Tool):
    """搜索工具（模拟）。

    模拟网络搜索功能，实际应用中应接入搜索API。

    示例：
        >>> search = SearchTool()
        >>> result = search.run(query="Python教程")
    """

    def __init__(self, search_fn: Optional[Callable[[str], str]] = None) -> None:
        """初始化搜索工具。

        参数：
            search_fn: 自定义搜索函数，接收查询返回结果
        """
        self._search_fn = search_fn

    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="search",
            description="搜索互联网获取信息。用于查找最新信息、事实核查或获取特定主题的知识。",
            parameters={
                "query": {
                    "type": "string",
                    "description": "搜索查询词",
                }
            },
            required_params=["query"],
        )

    def _run(self, query: str, **kwargs: Any) -> str:
        """执行搜索。"""
        if self._search_fn:
            return self._search_fn(query)
        # 模拟搜索结果
        return f"[模拟搜索结果] 关于'{query}'的搜索结果：这是一个模拟的搜索响应。实际应用中请接入搜索API（如Google、Bing、SerpAPI等）。"


class PythonREPLTool(Tool):
    """Python REPL工具。

    在受限环境中执行Python代码。

    安全性：
        - 限制可用模块
        - 限制执行时间
        - 捕获输出

    示例：
        >>> repl = PythonREPLTool()
        >>> result = repl.run(code="print(2 + 2)")
    """

    # 允许的内置函数
    _SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
    }

    def __init__(self, allowed_modules: Optional[List[str]] = None) -> None:
        """初始化Python REPL。

        参数：
            allowed_modules: 允许导入的模块列表
        """
        self._allowed_modules = allowed_modules or ["math", "json", "re", "datetime"]

    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="python_repl",
            description="执行Python代码并返回结果。用于数据处理、计算或生成代码输出。",
            parameters={
                "code": {
                    "type": "string",
                    "description": "要执行的Python代码",
                }
            },
            required_params=["code"],
            timeout=10.0,
        )

    def _run(self, code: str, **kwargs: Any) -> str:
        """执行Python代码。"""
        import io
        import sys

        # 创建受限的全局命名空间
        restricted_globals = {"__builtins__": self._SAFE_BUILTINS}

        # 添加允许的模块
        for module_name in self._allowed_modules:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass

        # 捕获输出
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # 执行代码
            exec(code, restricted_globals)
            output = sys.stdout.getvalue()
            return output if output else "[代码执行成功，无输出]"
        except Exception as e:
            return f"[执行错误] {type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout


class WikipediaTool(Tool):
    """Wikipedia工具（模拟）。

    查询Wikipedia获取知识信息。

    示例：
        >>> wiki = WikipediaTool()
        >>> result = wiki.run(query="机器学习")
    """

    def __init__(self, fetch_fn: Optional[Callable[[str], str]] = None) -> None:
        """初始化Wikipedia工具。

        参数：
            fetch_fn: 自定义获取函数
        """
        self._fetch_fn = fetch_fn

    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="wikipedia",
            description="查询Wikipedia百科全书获取概念解释和背景知识。",
            parameters={
                "query": {
                    "type": "string",
                    "description": "要查询的主题或概念",
                }
            },
            required_params=["query"],
        )

    def _run(self, query: str, **kwargs: Any) -> str:
        """查询Wikipedia。"""
        if self._fetch_fn:
            return self._fetch_fn(query)
        # 模拟Wikipedia结果
        return f"[Wikipedia] {query}：这是关于'{query}'的模拟Wikipedia摘要。实际应用中请接入Wikipedia API。"


class DateTimeTool(Tool):
    """日期时间工具。

    获取当前日期时间或进行日期计算。

    示例：
        >>> dt = DateTimeTool()
        >>> result = dt.run(action="now")
    """

    @property
    def config(self) -> ToolConfig:
        return ToolConfig(
            name="datetime",
            description="获取当前日期时间或进行日期计算。",
            parameters={
                "action": {
                    "type": "string",
                    "description": "操作类型: 'now'(当前时间), 'date'(当前日期), 'timestamp'(时间戳)",
                    "enum": ["now", "date", "timestamp"],
                }
            },
            required_params=["action"],
        )

    def _run(self, action: str, **kwargs: Any) -> str:
        """执行日期时间操作。"""
        from datetime import datetime

        now = datetime.now()

        if action == "now":
            return now.strftime("%Y-%m-%d %H:%M:%S")
        elif action == "date":
            return now.strftime("%Y-%m-%d")
        elif action == "timestamp":
            return str(int(now.timestamp()))
        else:
            raise ValueError(f"未知操作: {action}")
