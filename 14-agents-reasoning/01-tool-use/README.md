# 01-tool-use 工具调用

> **前置知识**: Python 基础、JSON、Pydantic、LLM API 基本概念

## 核心概念

**Tool Use** = 让 LLM 以结构化方式调用外部工具

```
User: "北京今天多少度？"
  ↓
LLM: {"name": "get_weather", "arguments": {"city": "北京"}}  ← 结构化输出
  ↓
程序: 调用天气API → "25°C，晴"
  ↓
LLM: "北京今天25°C，天气晴朗。"  ← 最终回答
```

## 四个核心组件

| 组件 | 文件 | 作用 |
|------|------|------|
| **Function Calling** | `function_calling.py` | 定义函数签名，解析 LLM 输出 |
| **Tool Registry** | `tool_registry.py` | 注册和管理工具 |
| **Tool Executor** | `tool_executor.py` | 安全执行工具 |
| **Structured Output** | `structured_output.py` | 解析结构化输出 |

## 目录结构

```
01-tool-use/
├── src/
│   ├── __init__.py
│   ├── function_calling.py   # 函数定义与调用解析
│   ├── tool_registry.py      # 工具注册与管理
│   ├── tool_executor.py      # 工具执行引擎
│   └── structured_output.py  # 结构化输出解析
├── tests/
│   └── test_tool_use.py      # 单元测试 (~60 cases)
├── notebooks/
│   ├── 01_FunctionCalling_tutorial.ipynb
│   ├── 02_ToolRegistry_tutorial.ipynb
│   └── 03_StructuredOutput_tutorial.ipynb
├── 知识点.md                  # 详细知识点
└── README.md
```

---

## 快速开始

### 1. Function Calling

```python
from src.function_calling import (
    FunctionDefinition,
    FunctionParameter,
    FunctionCallParser,
    ParameterType,
    create_function_schema,
)

# 方式1: 手动定义
weather_func = FunctionDefinition(
    name="get_weather",
    description="获取天气信息",
    parameters=[
        FunctionParameter(
            name="city",
            type=ParameterType.STRING,
            description="城市名称",
        ),
    ],
)

# 方式2: 从 Python 函数自动生成
def calculate(expression: str, precision: int = 2) -> float:
    """计算数学表达式"""
    return round(eval(expression), precision)

calc_func = create_function_schema(calculate)

# 解析 LLM 输出
parser = FunctionCallParser([weather_func, calc_func])
calls = parser.parse('```json\n{"name": "get_weather", "arguments": {"city": "北京"}}\n```')
```

### 2. Tool Registry

```python
from src.tool_registry import ToolRegistry, tool

# 创建注册表
registry = ToolRegistry()

# 使用装饰器注册
@registry.register(tags=["search", "web"])
def web_search(query: str, max_results: int = 5) -> str:
    """搜索网页信息"""
    return f"搜索 '{query}' 的前 {max_results} 条结果"

@registry.register(tags=["math"])
def calculator(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

# 查询工具
print(registry.list_names())        # ['web_search', 'calculator']
print(registry.get_by_tag("math"))  # [Tool(calculator)]

# 导出为 API 格式
openai_tools = registry.to_openai_tools()
anthropic_tools = registry.to_anthropic_tools()
```

### 3. Tool Executor

```python
from src.tool_executor import ToolExecutor, ExecutionContext

# 创建执行器
executor = ToolExecutor(registry, default_timeout=30.0)

# 执行工具
result = executor.execute("calculator", {"expression": "2 + 3 * 4"})
print(result.output)          # 14
print(result.is_success)      # True
print(result.execution_time)  # 0.001

# 带上下文执行
context = ExecutionContext(
    user_id="user123",
    permissions=["read", "write"],
    timeout=10.0,
)
result = executor.execute("web_search", {"query": "Python"}, context)

# 批量执行
from src.function_calling import FunctionCall
calls = [
    FunctionCall(name="calculator", arguments={"expression": "1+1"}),
    FunctionCall(name="calculator", arguments={"expression": "2*2"}),
]
results = executor.execute_batch(calls, parallel=True)
```

### 4. Structured Output

```python
from pydantic import BaseModel, Field
from src.structured_output import StructuredOutputParser

# 定义输出模型
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

# 创建解析器
parser = StructuredOutputParser(Person)

# 解析 LLM 输出
llm_output = '''
分析结果：
```json
{"name": "张三", "age": 25}
```
'''
result = parser.parse(llm_output)
print(result.name)  # 张三
print(result.age)   # 25

# 获取格式说明 (用于 prompt)
instructions = parser.get_format_instructions()
```

---

## 完整示例：构建简单 Agent

```python
from src.tool_registry import ToolRegistry
from src.tool_executor import ToolExecutor
from src.function_calling import FunctionCallParser

# 1. 创建并注册工具
registry = ToolRegistry()

@registry.register(tags=["search"])
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: {query}"

@registry.register(tags=["math"])
def calculate(expression: str) -> float:
    """计算表达式"""
    return eval(expression)

# 2. 创建执行器和解析器
executor = ToolExecutor(registry)
functions = [t.to_function_definition() for t in registry.list_tools()]
parser = FunctionCallParser(functions)

# 3. 处理 LLM 输出
def process_llm_response(response: str):
    calls = parser.parse(response)
    results = []
    for call in calls:
        errors = parser.validate(call)
        if errors:
            results.append(f"Error: {errors}")
        else:
            result = executor.execute_call(call)
            results.append(result.output if result.is_success else result.error)
    return results

# 4. 测试
llm_response = '```json\n{"name": "calculate", "arguments": {"expression": "2**10"}}\n```'
print(process_llm_response(llm_response))  # [1024]
```

---

## 运行测试

```bash
# 运行所有测试
pytest 14-agents-reasoning/01-tool-use/tests/ -v

# 运行特定测试
pytest 14-agents-reasoning/01-tool-use/tests/test_tool_use.py::TestFunctionCallParser -v
```

## 学习路径

1. **01_FunctionCalling_tutorial.ipynb** - 函数定义与解析
2. **02_ToolRegistry_tutorial.ipynb** - 工具注册与管理
3. **03_StructuredOutput_tutorial.ipynb** - 结构化输出解析
4. **知识点.md** - 完整知识点速查

## 参考资料

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [JSON Schema](https://json-schema.org/)
