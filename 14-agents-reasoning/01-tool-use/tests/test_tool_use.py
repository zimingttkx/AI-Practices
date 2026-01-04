"""
01-tool-use 模块单元测试

测试覆盖:
- function_calling: 函数定义、参数、调用解析
- tool_registry: 工具注册、管理、查询
- tool_executor: 工具执行、超时、重试
- structured_output: 结构化输出解析、验证
"""

import json
import pytest
import time
from typing import List, Optional
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.function_calling import (
    FunctionDefinition,
    FunctionParameter,
    FunctionCall,
    FunctionCallParser,
    ParameterType,
    create_function_schema,
)
from src.tool_registry import (
    Tool,
    ToolRegistry,
    tool,
    get_default_registry,
    reset_default_registry,
)
from src.tool_executor import (
    ToolExecutor,
    ExecutionResult,
    ExecutionContext,
    ExecutionStatus,
)
from src.structured_output import (
    StructuredOutputParser,
    OutputSchema,
    ValidationError,
    create_choice_parser,
    create_extraction_parser,
)


# ============== FunctionParameter Tests ==============

class TestFunctionParameter:
    """FunctionParameter 测试"""

    def test_basic_parameter(self):
        """测试基本参数创建"""
        param = FunctionParameter(
            name="query",
            type=ParameterType.STRING,
            description="Search query",
        )
        assert param.name == "query"
        assert param.type == ParameterType.STRING
        assert param.required is True

    def test_optional_parameter(self):
        """测试可选参数"""
        param = FunctionParameter(
            name="limit",
            type=ParameterType.INTEGER,
            description="Max results",
            required=False,
            default=10,
        )
        assert param.required is False
        assert param.default == 10

    def test_enum_parameter(self):
        """测试枚举参数"""
        param = FunctionParameter(
            name="format",
            type=ParameterType.STRING,
            description="Output format",
            enum=["json", "xml", "csv"],
        )
        schema = param.to_schema()
        assert schema["enum"] == ["json", "xml", "csv"]

    def test_to_schema(self):
        """测试转换为 JSON Schema"""
        param = FunctionParameter(
            name="data",
            type=ParameterType.OBJECT,
            description="Input data",
        )
        schema = param.to_schema()
        assert schema["type"] == "object"
        assert schema["description"] == "Input data"


# ============== FunctionDefinition Tests ==============

class TestFunctionDefinition:
    """FunctionDefinition 测试"""

    def test_basic_definition(self):
        """测试基本函数定义"""
        func_def = FunctionDefinition(
            name="search",
            description="Search for information",
            parameters=[
                FunctionParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search query",
                ),
            ],
        )
        assert func_def.name == "search"
        assert len(func_def.parameters) == 1

    def test_to_openai_schema(self):
        """测试转换为 OpenAI 格式"""
        func_def = FunctionDefinition(
            name="get_weather",
            description="Get weather information",
            parameters=[
                FunctionParameter(
                    name="city",
                    type=ParameterType.STRING,
                    description="City name",
                ),
                FunctionParameter(
                    name="unit",
                    type=ParameterType.STRING,
                    description="Temperature unit",
                    required=False,
                    enum=["celsius", "fahrenheit"],
                ),
            ],
        )
        schema = func_def.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "city" in schema["function"]["parameters"]["properties"]
        assert "city" in schema["function"]["parameters"]["required"]
        assert "unit" not in schema["function"]["parameters"]["required"]

    def test_to_anthropic_schema(self):
        """测试转换为 Anthropic 格式"""
        func_def = FunctionDefinition(
            name="calculator",
            description="Perform calculations",
            parameters=[
                FunctionParameter(
                    name="expression",
                    type=ParameterType.STRING,
                    description="Math expression",
                ),
            ],
        )
        schema = func_def.to_anthropic_schema()
        assert schema["name"] == "calculator"
        assert "input_schema" in schema


# ============== FunctionCall Tests ==============

class TestFunctionCall:
    """FunctionCall 测试"""

    def test_basic_call(self):
        """测试基本函数调用"""
        call = FunctionCall(
            name="search",
            arguments={"query": "hello world"},
        )
        assert call.name == "search"
        assert call.arguments["query"] == "hello world"

    def test_to_dict(self):
        """测试转换为字典"""
        call = FunctionCall(
            name="add",
            arguments={"a": 1, "b": 2},
            id="call_123",
        )
        d = call.to_dict()
        assert d["name"] == "add"
        assert d["arguments"] == {"a": 1, "b": 2}
        assert d["id"] == "call_123"

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "name": "multiply",
            "arguments": {"x": 3, "y": 4},
        }
        call = FunctionCall.from_dict(data)
        assert call.name == "multiply"
        assert call.arguments["x"] == 3


# ============== FunctionCallParser Tests ==============

class TestFunctionCallParser:
    """FunctionCallParser 测试"""

    @pytest.fixture
    def parser(self):
        """创建解析器"""
        functions = [
            FunctionDefinition(
                name="search",
                description="Search",
                parameters=[
                    FunctionParameter(
                        name="query",
                        type=ParameterType.STRING,
                        description="Query",
                    ),
                ],
            ),
            FunctionDefinition(
                name="calculate",
                description="Calculate",
                parameters=[
                    FunctionParameter(
                        name="a",
                        type=ParameterType.INTEGER,
                        description="First number",
                    ),
                    FunctionParameter(
                        name="b",
                        type=ParameterType.INTEGER,
                        description="Second number",
                    ),
                ],
            ),
        ]
        return FunctionCallParser(functions)

    def test_parse_json_block(self, parser):
        """测试解析 JSON 代码块"""
        text = '''
        Here is the function call:
        ```json
        {"name": "search", "arguments": {"query": "test"}}
        ```
        '''
        calls = parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments["query"] == "test"

    def test_parse_openai_format(self, parser):
        """测试解析 OpenAI 格式"""
        text = '{"name": "calculate", "arguments": {"a": 5, "b": 3}}'
        calls = parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "calculate"

    def test_parse_anthropic_format(self, parser):
        """测试解析 Anthropic 格式"""
        text = '{"type": "tool_use", "name": "search", "input": {"query": "hello"}}'
        calls = parser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments["query"] == "hello"

    def test_validate_success(self, parser):
        """测试验证成功"""
        call = FunctionCall(name="search", arguments={"query": "test"})
        errors = parser.validate(call)
        assert len(errors) == 0

    def test_validate_missing_required(self, parser):
        """测试验证缺少必需参数"""
        call = FunctionCall(name="calculate", arguments={"a": 1})
        errors = parser.validate(call)
        assert any("Missing required parameter: b" in e for e in errors)

    def test_validate_unknown_function(self, parser):
        """测试验证未知函数"""
        call = FunctionCall(name="unknown", arguments={})
        errors = parser.validate(call)
        assert any("Unknown function" in e for e in errors)

    def test_validate_wrong_type(self, parser):
        """测试验证类型错误"""
        call = FunctionCall(name="calculate", arguments={"a": "not_int", "b": 2})
        errors = parser.validate(call)
        assert any("Expected integer" in e for e in errors)


# ============== create_function_schema Tests ==============

class TestCreateFunctionSchema:
    """create_function_schema 测试"""

    def test_simple_function(self):
        """测试简单函数"""
        def greet(name: str) -> str:
            """Greet someone"""
            return f"Hello, {name}!"

        func_def = create_function_schema(greet)
        assert func_def.name == "greet"
        assert "Greet someone" in func_def.description
        assert len(func_def.parameters) == 1
        assert func_def.parameters[0].name == "name"
        assert func_def.parameters[0].type == ParameterType.STRING

    def test_function_with_defaults(self):
        """测试带默认值的函数"""
        def add(a: int, b: int = 0) -> int:
            """Add two numbers"""
            return a + b

        func_def = create_function_schema(add)
        assert len(func_def.parameters) == 2
        a_param = next(p for p in func_def.parameters if p.name == "a")
        b_param = next(p for p in func_def.parameters if p.name == "b")
        assert a_param.required is True
        assert b_param.required is False
        assert b_param.default == 0

    def test_function_with_list_type(self):
        """测试列表类型参数"""
        def process(items: list) -> int:
            """Process items"""
            return len(items)

        func_def = create_function_schema(process)
        assert func_def.parameters[0].type == ParameterType.ARRAY


# ============== Tool Tests ==============

class TestTool:
    """Tool 测试"""

    def test_basic_tool(self):
        """测试基本工具创建"""
        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool(
            name="add",
            description="Add two numbers",
            func=add,
            parameters=[
                FunctionParameter(name="a", type=ParameterType.INTEGER, description="First"),
                FunctionParameter(name="b", type=ParameterType.INTEGER, description="Second"),
            ],
        )
        assert tool.name == "add"
        assert tool(1, 2) == 3

    def test_tool_disabled(self):
        """测试禁用工具"""
        tool = Tool(
            name="test",
            description="Test",
            func=lambda: "result",
            enabled=False,
        )
        with pytest.raises(RuntimeError, match="disabled"):
            tool()

    def test_tool_tags(self):
        """测试工具标签"""
        tool = Tool(
            name="search",
            description="Search",
            func=lambda q: q,
            tags=["web", "search"],
        )
        assert "web" in tool.tags
        assert "search" in tool.tags

    def test_to_openai_schema(self):
        """测试转换为 OpenAI 格式"""
        tool = Tool(
            name="greet",
            description="Greet someone",
            func=lambda name: f"Hello, {name}",
            parameters=[
                FunctionParameter(name="name", type=ParameterType.STRING, description="Name"),
            ],
        )
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"

    def test_get_signature(self):
        """测试获取签名"""
        tool = Tool(
            name="multiply",
            description="Multiply",
            func=lambda x, y: x * y,
            parameters=[
                FunctionParameter(name="x", type=ParameterType.NUMBER, description="X"),
                FunctionParameter(name="y", type=ParameterType.NUMBER, description="Y", required=False, default=1),
            ],
        )
        sig = tool.get_signature()
        assert "multiply" in sig
        assert "x: number" in sig


# ============== ToolRegistry Tests ==============

class TestToolRegistry:
    """ToolRegistry 测试"""

    @pytest.fixture
    def registry(self):
        """创建注册表"""
        return ToolRegistry()

    def test_register_decorator(self, registry):
        """测试装饰器注册"""
        @registry.register
        def my_tool(x: int) -> int:
            """My tool"""
            return x * 2

        assert "my_tool" in registry
        assert registry.get("my_tool") is not None

    def test_register_with_name(self, registry):
        """测试自定义名称注册"""
        @registry.register(name="custom_name")
        def original_name(x: int) -> int:
            return x

        assert "custom_name" in registry
        assert "original_name" not in registry

    def test_register_with_tags(self, registry):
        """测试带标签注册"""
        @registry.register(tags=["math", "utility"])
        def add(a: int, b: int) -> int:
            return a + b

        tools = registry.get_by_tag("math")
        assert len(tools) == 1
        assert tools[0].name == "add"

    def test_unregister(self, registry):
        """测试注销工具"""
        @registry.register
        def temp_tool():
            pass

        assert "temp_tool" in registry
        registry.unregister("temp_tool")
        assert "temp_tool" not in registry

    def test_list_tools(self, registry):
        """测试列出工具"""
        @registry.register
        def tool1():
            pass

        @registry.register
        def tool2():
            pass

        tools = registry.list_tools()
        assert len(tools) == 2

    def test_enable_disable(self, registry):
        """测试启用/禁用"""
        @registry.register
        def toggleable():
            pass

        registry.disable("toggleable")
        assert registry.get("toggleable").enabled is False

        registry.enable("toggleable")
        assert registry.get("toggleable").enabled is True

    def test_to_openai_tools(self, registry):
        """测试导出 OpenAI 格式"""
        @registry.register
        def func1(x: str) -> str:
            """Function 1"""
            return x

        tools = registry.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    def test_duplicate_registration(self, registry):
        """测试重复注册"""
        @registry.register
        def duplicate():
            pass

        with pytest.raises(ValueError, match="already registered"):
            @registry.register(name="duplicate")
            def another():
                pass

    def test_get_tool_descriptions(self, registry):
        """测试获取工具描述"""
        @registry.register(tags=["test"])
        def described_tool(x: int) -> int:
            """A described tool"""
            return x

        desc = registry.get_tool_descriptions()
        assert "described_tool" in desc
        assert "A described tool" in desc


# ============== ToolExecutor Tests ==============

class TestToolExecutor:
    """ToolExecutor 测试"""

    @pytest.fixture
    def registry(self):
        """创建注册表"""
        reg = ToolRegistry()

        @reg.register
        def add(a: int, b: int) -> int:
            """Add two numbers"""
            return a + b

        @reg.register
        def slow_task(seconds: float) -> str:
            """Slow task"""
            time.sleep(seconds)
            return "done"

        @reg.register
        def failing_task() -> None:
            """Always fails"""
            raise ValueError("Intentional error")

        return reg

    @pytest.fixture
    def executor(self, registry):
        """创建执行器"""
        return ToolExecutor(registry, default_timeout=5.0)

    def test_execute_success(self, executor):
        """测试成功执行"""
        result = executor.execute("add", {"a": 1, "b": 2})
        assert result.is_success
        assert result.output == 3
        assert result.execution_time > 0

    def test_execute_unknown_tool(self, executor):
        """测试执行未知工具"""
        result = executor.execute("unknown", {})
        assert result.is_error
        assert "not found" in result.error

    def test_execute_with_error(self, executor):
        """测试执行出错"""
        result = executor.execute("failing_task", {})
        assert result.status == ExecutionStatus.ERROR
        assert "Intentional error" in result.error

    def test_execute_timeout(self, executor):
        """测试执行超时"""
        result = executor.execute("slow_task", {"seconds": 10.0}, timeout=0.1)
        assert result.status == ExecutionStatus.TIMEOUT

    def test_execute_call(self, executor):
        """测试执行 FunctionCall"""
        call = FunctionCall(name="add", arguments={"a": 5, "b": 3}, id="test_id")
        result = executor.execute_call(call)
        assert result.is_success
        assert result.output == 8
        assert result.call_id == "test_id"

    def test_execute_batch(self, executor):
        """测试批量执行"""
        calls = [
            FunctionCall(name="add", arguments={"a": 1, "b": 1}),
            FunctionCall(name="add", arguments={"a": 2, "b": 2}),
        ]
        results = executor.execute_batch(calls)
        assert len(results) == 2
        assert all(r.is_success for r in results)

    def test_execution_history(self, executor):
        """测试执行历史"""
        executor.execute("add", {"a": 1, "b": 1})
        executor.execute("add", {"a": 2, "b": 2})

        history = executor.get_history()
        assert len(history) == 2

    def test_execution_stats(self, executor):
        """测试执行统计"""
        executor.execute("add", {"a": 1, "b": 1})
        executor.execute("failing_task", {})

        stats = executor.get_stats()
        assert stats["total_executions"] == 2
        assert stats["success_count"] == 1
        assert stats["error_count"] == 1

    def test_hooks(self, executor):
        """测试执行钩子"""
        hook_called = []

        def before_hook(tool, args, ctx):
            hook_called.append("before")

        def after_hook(tool, args, ctx, result):
            hook_called.append("after")

        executor.add_hook("before_execute", before_hook)
        executor.add_hook("after_execute", after_hook)

        executor.execute("add", {"a": 1, "b": 1})

        assert "before" in hook_called
        assert "after" in hook_called

    def test_execution_context(self, executor):
        """测试执行上下文"""
        ctx = ExecutionContext(
            user_id="user123",
            session_id="session456",
            variables={"key": "value"},
            permissions=["read", "write"],
        )
        assert ctx.has_permission("read")
        assert not ctx.has_permission("admin")
        assert ctx.get_variable("key") == "value"


# ============== StructuredOutputParser Tests ==============

class TestStructuredOutputParser:
    """StructuredOutputParser 测试"""

    def test_parse_pydantic_model(self):
        """测试解析 Pydantic 模型"""
        class Person(BaseModel):
            name: str
            age: int

        parser = StructuredOutputParser(Person)
        result = parser.parse('{"name": "Alice", "age": 30}')
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_json_block(self):
        """测试解析 JSON 代码块"""
        class Data(BaseModel):
            value: int

        parser = StructuredOutputParser(Data)
        text = '''
        Here is the result:
        ```json
        {"value": 42}
        ```
        '''
        result = parser.parse(text)
        assert result.value == 42

    def test_parse_json_schema(self):
        """测试使用 JSON Schema 解析"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name"],
        }
        parser = StructuredOutputParser(schema)
        result = parser.parse('{"name": "test", "count": 5}')
        assert result["name"] == "test"
        assert result["count"] == 5

    def test_validation_error(self):
        """测试验证错误"""
        class Strict(BaseModel):
            required_field: str

        parser = StructuredOutputParser(Strict, strict=True)
        with pytest.raises(ValidationError):
            parser.parse('{"wrong_field": "value"}')

    def test_auto_fix_json(self):
        """测试自动修复 JSON"""
        class Simple(BaseModel):
            value: str

        parser = StructuredOutputParser(Simple, auto_fix=True)
        # 尾随逗号
        result = parser.parse('{"value": "test",}')
        assert result.value == "test"

    def test_parse_or_none(self):
        """测试 parse_or_none"""
        class Data(BaseModel):
            x: int

        parser = StructuredOutputParser(Data)
        assert parser.parse_or_none("invalid") is None
        assert parser.parse_or_none('{"x": 1}').x == 1

    def test_get_format_instructions(self):
        """测试获取格式说明"""
        class Output(BaseModel):
            """Output model"""
            result: str

        parser = StructuredOutputParser(Output)
        instructions = parser.get_format_instructions()
        assert "JSON" in instructions
        assert "result" in instructions


# ============== Helper Function Tests ==============

class TestHelperFunctions:
    """辅助函数测试"""

    def test_create_choice_parser(self):
        """测试创建选择解析器"""
        parser = create_choice_parser(["option1", "option2", "option3"])
        result = parser.parse('{"choice": "option1", "reason": "Best option"}')
        assert result["choice"] == "option1"

    def test_create_extraction_parser(self):
        """测试创建提取解析器"""
        parser = create_extraction_parser({
            "name": "Person's name",
            "email": "Email address",
        })
        result = parser.parse('{"name": "John", "email": "john@example.com"}')
        assert result["name"] == "John"
        assert result["email"] == "john@example.com"


# ============== Global Tool Decorator Tests ==============

class TestGlobalToolDecorator:
    """全局 tool 装饰器测试"""

    def setup_method(self):
        """每个测试前重置全局注册表"""
        reset_default_registry()

    def test_global_tool_decorator(self):
        """测试全局装饰器"""
        @tool
        def global_func(x: int) -> int:
            """Global function"""
            return x * 2

        registry = get_default_registry()
        assert "global_func" in registry

    def test_global_tool_with_options(self):
        """测试带选项的全局装饰器"""
        @tool(name="custom", tags=["test"])
        def another_func(x: int) -> int:
            return x

        registry = get_default_registry()
        assert "custom" in registry
        assert "test" in registry.get("custom").tags


# ============== Integration Tests ==============

class TestIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 创建注册表
        registry = ToolRegistry()

        # 2. 注册工具
        @registry.register(tags=["math"])
        def calculator(expression: str) -> float:
            """Calculate a math expression"""
            return eval(expression)

        @registry.register(tags=["text"])
        def uppercase(text: str) -> str:
            """Convert text to uppercase"""
            return text.upper()

        # 3. 创建执行器
        executor = ToolExecutor(registry)

        # 4. 创建解析器
        functions = [t.to_function_definition() for t in registry.list_tools()]
        parser = FunctionCallParser(functions)

        # 5. 模拟 LLM 输出
        llm_output = '''
        I'll calculate that for you.
        ```json
        {"name": "calculator", "arguments": {"expression": "2 + 3 * 4"}}
        ```
        '''

        # 6. 解析函数调用
        calls = parser.parse(llm_output)
        assert len(calls) == 1

        # 7. 验证调用
        errors = parser.validate(calls[0])
        assert len(errors) == 0

        # 8. 执行工具
        result = executor.execute_call(calls[0])
        assert result.is_success
        assert result.output == 14  # 2 + 3 * 4 = 14

    def test_structured_output_with_tools(self):
        """测试结构化输出与工具结合"""
        class ToolCallOutput(BaseModel):
            tool_name: str
            arguments: dict
            reasoning: str

        parser = StructuredOutputParser(ToolCallOutput)

        llm_output = '''
        ```json
        {
            "tool_name": "search",
            "arguments": {"query": "weather today"},
            "reasoning": "User wants to know the weather"
        }
        ```
        '''

        result = parser.parse(llm_output)
        assert result.tool_name == "search"
        assert result.arguments["query"] == "weather today"
