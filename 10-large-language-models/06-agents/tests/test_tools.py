"""
工具模块单元测试 (Tools Module Unit Tests)

测试覆盖：
    - ToolConfig配置验证
    - ToolResult结果处理
    - Tool基类功能
    - ToolRegistry注册管理
    - 内置工具功能测试

"""

import pytest
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools import (
    ToolConfig,
    ToolResult,
    Tool,
    ToolRegistry,
    CalculatorTool,
    SearchTool,
    PythonREPLTool,
    WikipediaTool,
    DateTimeTool,
)


# ==================== ToolConfig Tests ====================

class TestToolConfig:
    """ToolConfig测试类。"""

    def test_basic_config(self):
        """测试基本配置。"""
        config = ToolConfig(name="test", description="A test tool")
        assert config.name == "test"
        assert config.description == "A test tool"
        assert config.return_direct is False
        assert config.timeout == 30.0

    def test_custom_config(self):
        """测试自定义配置。"""
        config = ToolConfig(
            name="my_tool",
            description="A custom tool",
            return_direct=True,
            timeout=60.0,
        )
        assert config.name == "my_tool"
        assert config.description == "A custom tool"
        assert config.return_direct is True
        assert config.timeout == 60.0

    def test_empty_name(self):
        """测试空名称。"""
        with pytest.raises(ValueError, match="工具名称不能为空"):
            ToolConfig(name="", description="test")

    def test_empty_description(self):
        """测试空描述。"""
        with pytest.raises(ValueError, match="工具描述不能为空"):
            ToolConfig(name="test", description="")

    def test_invalid_timeout(self):
        """测试无效的timeout。"""
        with pytest.raises(ValueError, match="timeout必须为正数"):
            ToolConfig(name="test", description="test", timeout=0)
        with pytest.raises(ValueError, match="timeout必须为正数"):
            ToolConfig(name="test", description="test", timeout=-1)

    def test_to_openai_function(self):
        """测试转换为OpenAI函数格式。"""
        config = ToolConfig(
            name="test",
            description="A test tool",
            parameters={"x": {"type": "string"}},
            required_params=["x"],
        )
        func = config.to_openai_function()
        assert func["name"] == "test"
        assert func["description"] == "A test tool"

    def test_config_repr(self):
        """测试配置的字符串表示。"""
        config = ToolConfig(name="test", description="test")
        repr_str = repr(config)
        assert "ToolConfig" in repr_str


# ==================== ToolResult Tests ====================

class TestToolResult:
    """ToolResult测试类。"""

    def test_success_result(self):
        """测试成功结果。"""
        result = ToolResult(output="计算结果: 42")
        assert result.is_success is True
        assert result.output == "计算结果: 42"
        assert result.error is None

    def test_error_result(self):
        """测试错误结果。"""
        from tools import ToolStatus
        result = ToolResult(output="", status=ToolStatus.ERROR, error="除零错误")
        assert result.is_success is False
        assert result.error == "除零错误"

    def test_result_with_metadata(self):
        """测试带元数据的结果。"""
        result = ToolResult(
            output="result",
            metadata={"key": "value"},
        )
        assert result.metadata == {"key": "value"}

    def test_result_repr(self):
        """测试结果的字符串表示。"""
        result = ToolResult(output="test output")
        repr_str = repr(result)
        assert "ToolResult" in repr_str
        assert "success" in repr_str


# ==================== CalculatorTool Tests ====================

class TestCalculatorTool:
    """CalculatorTool测试类。"""

    def setup_method(self):
        """测试前初始化。"""
        self.calc = CalculatorTool()

    def test_basic_addition(self):
        """测试基本加法。"""
        result = self.calc.run(expression="2 + 3")
        assert result.is_success
        assert "5" in result.output

    def test_basic_subtraction(self):
        """测试基本减法。"""
        result = self.calc.run(expression="10 - 4")
        assert result.is_success
        assert "6" in result.output

    def test_multiplication(self):
        """测试乘法。"""
        result = self.calc.run(expression="6 * 7")
        assert result.is_success
        assert "42" in result.output

    def test_division(self):
        """测试除法。"""
        result = self.calc.run(expression="15 / 3")
        assert result.is_success
        assert "5" in result.output

    def test_power(self):
        """测试幂运算。"""
        result = self.calc.run(expression="2 ** 10")
        assert result.is_success
        assert "1024" in result.output

    def test_complex_expression(self):
        """测试复杂表达式。"""
        result = self.calc.run(expression="(2 + 3) * 4 - 6 / 2")
        assert result.is_success
        assert "17" in result.output

    def test_float_result(self):
        """测试浮点数结果。"""
        result = self.calc.run(expression="7 / 2")
        assert result.is_success
        assert "3.5" in result.output

    def test_division_by_zero(self):
        """测试除零错误。"""
        result = self.calc.run(expression="1 / 0")
        assert result.is_success is False
        assert "除零" in result.error or "division" in result.error.lower()

    def test_invalid_expression(self):
        """测试无效表达式。"""
        result = self.calc.run(expression="abc")
        assert result.is_success is False

    def test_unsafe_expression_import(self):
        """测试不安全表达式（import）。"""
        result = self.calc.run(expression="__import__('os')")
        assert result.is_success is False

    def test_unsafe_expression_exec(self):
        """测试不安全表达式（exec）。"""
        result = self.calc.run(expression="exec('print(1)')")
        assert result.is_success is False

    def test_tool_name(self):
        """测试工具名称。"""
        assert self.calc.name == "calculator"

    def test_tool_description(self):
        """测试工具描述。"""
        assert "计算" in self.calc.description or "数学" in self.calc.description


# ==================== SearchTool Tests ====================

class TestSearchTool:
    """SearchTool测试类。"""

    def setup_method(self):
        """测试前初始化。"""
        self.search = SearchTool()

    def test_basic_search(self):
        """测试基本搜索。"""
        result = self.search.run(query="Python编程")
        assert result.is_success
        assert len(result.output) > 0

    def test_search_with_num_results(self):
        """测试指定结果数量。"""
        result = self.search.run(query="机器学习", num_results=5)
        assert result.is_success

    def test_empty_query(self):
        """测试空查询。"""
        result = self.search.run(query="")
        # 模拟搜索对空查询也返回结果
        assert result.is_success

    def test_tool_name(self):
        """测试工具名称。"""
        assert self.search.name == "search"


# ==================== PythonREPLTool Tests ====================

class TestPythonREPLTool:
    """PythonREPLTool测试类。"""

    def setup_method(self):
        """测试前初始化。"""
        self.repl = PythonREPLTool()

    def test_simple_expression(self):
        """测试简单表达式。"""
        result = self.repl.run(code="print(1 + 1)")
        assert result.is_success
        assert "2" in result.output

    def test_print_statement(self):
        """测试print语句。"""
        result = self.repl.run(code="print('Hello, World!')")
        assert result.is_success
        assert "Hello" in result.output

    def test_variable_assignment(self):
        """测试变量赋值。"""
        result = self.repl.run(code="x = 10\nprint(x)")
        assert result.is_success
        assert "10" in result.output

    def test_list_operations(self):
        """测试列表操作。"""
        result = self.repl.run(code="print([i**2 for i in range(5)])")
        assert result.is_success
        assert "0" in result.output and "16" in result.output

    def test_syntax_error(self):
        """测试语法错误。"""
        result = self.repl.run(code="def f(")
        # 语法错误会被捕获并返回成功状态但包含错误信息
        assert "SyntaxError" in result.output

    def test_runtime_error(self):
        """测试运行时错误。"""
        result = self.repl.run(code="undefined_variable")
        # 运行时错误会被捕获并返回成功状态但包含错误信息
        assert "NameError" in result.output

    def test_tool_name(self):
        """测试工具名称。"""
        assert self.repl.name == "python_repl"


# ==================== WikipediaTool Tests ====================

class TestWikipediaTool:
    """WikipediaTool测试类。"""

    def setup_method(self):
        """测试前初始化。"""
        self.wiki = WikipediaTool()

    def test_basic_query(self):
        """测试基本查询。"""
        result = self.wiki.run(query="Python")
        assert result.is_success
        assert len(result.output) > 0

    def test_empty_query(self):
        """测试空查询。"""
        result = self.wiki.run(query="")
        # 模拟Wikipedia对空查询也返回结果
        assert result.is_success

    def test_tool_name(self):
        """测试工具名称。"""
        assert self.wiki.name == "wikipedia"


# ==================== DateTimeTool Tests ====================

class TestDateTimeTool:
    """DateTimeTool测试类。"""

    def setup_method(self):
        """测试前初始化。"""
        self.dt = DateTimeTool()

    def test_current_time(self):
        """测试获取当前时间。"""
        result = self.dt.run(action="now")
        assert result.is_success
        assert len(result.output) > 0

    def test_current_date(self):
        """测试获取当前日期。"""
        result = self.dt.run(action="date")
        assert result.is_success
        assert "-" in result.output  # 日期格式包含-

    def test_timestamp(self):
        """测试获取时间戳。"""
        result = self.dt.run(action="timestamp")
        assert result.is_success
        assert result.output.isdigit()

    def test_invalid_action(self):
        """测试无效操作。"""
        result = self.dt.run(action="invalid_op")
        assert result.is_success is False

    def test_tool_name(self):
        """测试工具名称。"""
        assert self.dt.name == "datetime"


# ==================== ToolRegistry Tests ====================

class TestToolRegistry:
    """ToolRegistry测试类。"""

    def setup_method(self):
        """测试前初始化。"""
        self.registry = ToolRegistry()

    def test_register_tool(self):
        """测试注册工具。"""
        calc = CalculatorTool()
        self.registry.register(calc)
        assert "calculator" in self.registry.list_tools()

    def test_get_tool(self):
        """测试获取工具。"""
        calc = CalculatorTool()
        self.registry.register(calc)
        retrieved = self.registry.get("calculator")
        assert retrieved is calc

    def test_get_nonexistent_tool(self):
        """测试获取不存在的工具。"""
        result = self.registry.get("nonexistent")
        assert result is None

    def test_list_tools(self):
        """测试列出工具。"""
        self.registry.register(CalculatorTool())
        self.registry.register(SearchTool())
        tools = self.registry.list_tools()
        assert "calculator" in tools
        assert "search" in tools

    def test_get_all_tools(self):
        """测试获取所有工具。"""
        calc = CalculatorTool()
        search = SearchTool()
        self.registry.register(calc)
        self.registry.register(search)
        all_tools = self.registry.get_all_tools()
        assert len(all_tools) == 2

    def test_get_tools_description(self):
        """测试获取工具描述。"""
        self.registry.register(CalculatorTool())
        desc = self.registry.get_tools_description()
        assert "calculator" in desc

    def test_duplicate_registration(self):
        """测试重复注册。"""
        calc1 = CalculatorTool()
        self.registry.register(calc1)
        # 重复注册应该抛出异常
        with pytest.raises(ValueError, match="已存在"):
            calc2 = CalculatorTool()
            self.registry.register(calc2)

    def test_registry_repr(self):
        """测试注册表的字符串表示。"""
        self.registry.register(CalculatorTool())
        repr_str = repr(self.registry)
        assert "ToolRegistry" in repr_str


# ==================== Tool Base Class Tests ====================

class TestToolBaseClass:
    """Tool基类测试。"""

    def test_tool_has_name(self):
        """测试工具有名称。"""
        calc = CalculatorTool()
        assert hasattr(calc, 'name')
        assert isinstance(calc.name, str)

    def test_tool_has_description(self):
        """测试工具有描述。"""
        calc = CalculatorTool()
        assert hasattr(calc, 'description')
        assert isinstance(calc.description, str)

    def test_tool_has_run_method(self):
        """测试工具有run方法。"""
        calc = CalculatorTool()
        assert hasattr(calc, 'run')
        assert callable(calc.run)

    def test_tool_repr(self):
        """测试工具的字符串表示。"""
        calc = CalculatorTool()
        repr_str = repr(calc)
        assert "calculator" in repr_str.lower() or "Tool" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
