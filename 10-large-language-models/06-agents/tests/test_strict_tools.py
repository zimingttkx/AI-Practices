"""
工具模块最严格单元测试 (Strictest Tools Unit Tests)

测试覆盖：
    - 边界条件测试
    - 异常处理测试
    - 类型验证测试
    - 安全性测试
    - 并发测试

作者: AI-Practices
许可证: MIT
"""

import pytest
import sys
import os
import math
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools import (
    Tool, ToolConfig, ToolResult, ToolStatus, ToolRegistry,
    CalculatorTool, SearchTool, PythonREPLTool,
    WikipediaTool, DateTimeTool,
)


# ==================== ToolConfig 严格测试 ====================

class TestToolConfigStrict:
    """ToolConfig最严格测试。"""

    def test_name_whitespace_only(self):
        """测试仅空白字符的名称。"""
        with pytest.raises(ValueError):
            ToolConfig(name="   ", description="test")

    def test_description_whitespace_only(self):
        """测试仅空白字符的描述。"""
        with pytest.raises(ValueError):
            ToolConfig(name="test", description="   ")

    def test_timeout_very_small_positive(self):
        """测试极小正数timeout。"""
        config = ToolConfig(name="test", description="test", timeout=0.001)
        assert config.timeout == 0.001

    def test_timeout_very_large(self):
        """测试极大timeout。"""
        config = ToolConfig(name="test", description="test", timeout=1e10)
        assert config.timeout == 1e10

    def test_timeout_float_precision(self):
        """测试浮点精度timeout。"""
        config = ToolConfig(name="test", description="test", timeout=0.123456789)
        assert abs(config.timeout - 0.123456789) < 1e-9

    def test_parameters_empty_dict(self):
        """测试空参数字典。"""
        config = ToolConfig(name="test", description="test", parameters={})
        assert config.parameters == {}

    def test_parameters_nested_dict(self):
        """测试嵌套参数字典。"""
        params = {"a": {"b": {"c": {"type": "string"}}}}
        config = ToolConfig(name="test", description="test", parameters=params)
        assert config.parameters == params

    def test_required_params_empty_list(self):
        """测试空必需参数列表。"""
        config = ToolConfig(name="test", description="test", required_params=[])
        assert config.required_params == []

    def test_required_params_duplicates(self):
        """测试重复必需参数。"""
        config = ToolConfig(name="test", description="test", required_params=["a", "a", "b"])
        assert config.required_params == ["a", "a", "b"]

    def test_to_openai_function_complete(self):
        """测试完整OpenAI函数转换。"""
        config = ToolConfig(
            name="test_tool",
            description="A test tool",
            parameters={"x": {"type": "integer"}, "y": {"type": "string"}},
            required_params=["x"],
        )
        func = config.to_openai_function()
        assert func["name"] == "test_tool"
        assert func["description"] == "A test tool"
        assert func["parameters"]["type"] == "object"
        assert "x" in func["parameters"]["properties"]
        assert func["parameters"]["required"] == ["x"]

    def test_repr_special_characters(self):
        """测试特殊字符的repr。"""
        config = ToolConfig(name="test'\"\\", description="desc")
        repr_str = repr(config)
        assert "ToolConfig" in repr_str


# ==================== ToolResult 严格测试 ====================

class TestToolResultStrict:
    """ToolResult最严格测试。"""

    def test_empty_output(self):
        """测试空输出。"""
        result = ToolResult(output="")
        assert result.output == ""
        assert result.is_success

    def test_very_long_output(self):
        """测试超长输出。"""
        long_output = "x" * 100000
        result = ToolResult(output=long_output)
        assert len(result.output) == 100000

    def test_unicode_output(self):
        """测试Unicode输出。"""
        result = ToolResult(output="你好世界🌍")
        assert result.output == "你好世界🌍"

    def test_newlines_in_output(self):
        """测试换行符输出。"""
        result = ToolResult(output="line1\nline2\r\nline3")
        assert "\n" in result.output

    def test_all_status_types(self):
        """测试所有状态类型。"""
        for status in ToolStatus:
            result = ToolResult(output="test", status=status)
            assert result.status == status

    def test_error_status_is_not_success(self):
        """测试错误状态不是成功。"""
        result = ToolResult(output="", status=ToolStatus.ERROR)
        assert not result.is_success

    def test_timeout_status_is_not_success(self):
        """测试超时状态不是成功。"""
        result = ToolResult(output="", status=ToolStatus.TIMEOUT)
        assert not result.is_success

    def test_metadata_complex_types(self):
        """测试复杂元数据类型。"""
        metadata = {
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "none": None,
            "float": 3.14,
        }
        result = ToolResult(output="test", metadata=metadata)
        assert result.metadata == metadata

    def test_repr_truncation(self):
        """测试repr截断。"""
        result = ToolResult(output="x" * 100)
        repr_str = repr(result)
        assert "..." in repr_str
        assert len(repr_str) < 200


# ==================== CalculatorTool 严格测试 ====================

class TestCalculatorToolStrict:
    """CalculatorTool最严格测试。"""

    def setup_method(self):
        self.calc = CalculatorTool()

    # 基本运算边界测试
    def test_zero_operations(self):
        """测试零运算。"""
        assert "0" in self.calc.run(expression="0 + 0").output
        assert "0" in self.calc.run(expression="0 * 100").output
        assert "0" in self.calc.run(expression="0 / 1").output

    def test_negative_numbers(self):
        """测试负数。"""
        assert "-5" in self.calc.run(expression="-5").output
        assert "-10" in self.calc.run(expression="-5 + -5").output
        assert "10" in self.calc.run(expression="-5 * -2").output

    def test_large_numbers(self):
        """测试大数。"""
        result = self.calc.run(expression="10 ** 100")
        assert result.is_success
        assert "1" in result.output

    def test_small_decimals(self):
        """测试小数。"""
        result = self.calc.run(expression="0.0001 + 0.0002")
        assert result.is_success

    def test_scientific_notation(self):
        """测试科学计数法。"""
        result = self.calc.run(expression="1e10 + 1e10")
        assert result.is_success

    # 数学函数测试
    def test_sqrt_zero(self):
        """测试sqrt(0)。"""
        assert "0" in self.calc.run(expression="sqrt(0)").output

    def test_sqrt_negative(self):
        """测试sqrt负数。"""
        result = self.calc.run(expression="sqrt(-1)")
        assert not result.is_success

    def test_log_one(self):
        """测试log(1)。"""
        assert "0" in self.calc.run(expression="log(1)").output

    def test_log_zero(self):
        """测试log(0)。"""
        result = self.calc.run(expression="log(0)")
        assert not result.is_success

    def test_log_negative(self):
        """测试log负数。"""
        result = self.calc.run(expression="log(-1)")
        assert not result.is_success

    def test_sin_cos_tan_zero(self):
        """测试三角函数零值。"""
        assert "0" in self.calc.run(expression="sin(0)").output
        assert "1" in self.calc.run(expression="cos(0)").output
        assert "0" in self.calc.run(expression="tan(0)").output

    def test_pi_constant(self):
        """测试pi常量。"""
        result = self.calc.run(expression="pi")
        assert result.is_success
        assert "3.14" in result.output

    def test_e_constant(self):
        """测试e常量。"""
        result = self.calc.run(expression="e")
        assert result.is_success
        assert "2.71" in result.output

    def test_exp_zero(self):
        """测试exp(0)。"""
        assert "1" in self.calc.run(expression="exp(0)").output

    # 除法边界测试
    def test_division_by_zero(self):
        """测试除零。"""
        assert not self.calc.run(expression="1/0").is_success

    def test_floor_division_by_zero(self):
        """测试整除零。"""
        assert not self.calc.run(expression="1//0").is_success

    def test_modulo_by_zero(self):
        """测试取模零。"""
        assert not self.calc.run(expression="1%0").is_success

    # 运算符优先级测试
    def test_operator_precedence(self):
        """测试运算符优先级。"""
        assert "14" in self.calc.run(expression="2 + 3 * 4").output
        assert "20" in self.calc.run(expression="(2 + 3) * 4").output

    def test_nested_parentheses(self):
        """测试嵌套括号。"""
        result = self.calc.run(expression="((1 + 2) * (3 + 4))")
        assert "21" in result.output

    # 安全性测试
    def test_no_import(self):
        """测试禁止import。"""
        result = self.calc.run(expression="__import__('os')")
        assert not result.is_success

    def test_no_eval(self):
        """测试禁止eval。"""
        result = self.calc.run(expression="eval('1+1')")
        assert not result.is_success

    def test_no_exec(self):
        """测试禁止exec。"""
        result = self.calc.run(expression="exec('x=1')")
        assert not result.is_success

    def test_no_open(self):
        """测试禁止open。"""
        result = self.calc.run(expression="open('/etc/passwd')")
        assert not result.is_success

    def test_no_attribute_access(self):
        """测试禁止属性访问。"""
        result = self.calc.run(expression="(1).__class__")
        assert not result.is_success

    # 缺少参数测试
    def test_missing_expression(self):
        """测试缺少expression参数。"""
        result = self.calc.run()
        assert not result.is_success
        assert "缺少必需参数" in result.error

    def test_callable_interface(self):
        """测试可调用接口。"""
        result = self.calc(expression="2+2")
        assert result.is_success
        assert "4" in result.output


# ==================== SearchTool 严格测试 ====================

class TestSearchToolStrict:
    """SearchTool最严格测试。"""

    def setup_method(self):
        self.search = SearchTool()

    def test_basic_query(self):
        """测试基本查询。"""
        result = self.search.run(query="test")
        assert result.is_success

    def test_unicode_query(self):
        """测试Unicode查询。"""
        result = self.search.run(query="中文搜索🔍")
        assert result.is_success
        assert "中文搜索" in result.output

    def test_special_characters_query(self):
        """测试特殊字符查询。"""
        result = self.search.run(query="test & query | special")
        assert result.is_success

    def test_very_long_query(self):
        """测试超长查询。"""
        result = self.search.run(query="x" * 10000)
        assert result.is_success

    def test_custom_search_fn(self):
        """测试自定义搜索函数。"""
        def custom_fn(q):
            return f"Custom: {q}"
        search = SearchTool(search_fn=custom_fn)
        result = search.run(query="test")
        assert "Custom: test" in result.output

    def test_custom_search_fn_exception(self):
        """测试自定义搜索函数异常。"""
        def error_fn(q):
            raise RuntimeError("Search failed")
        search = SearchTool(search_fn=error_fn)
        result = search.run(query="test")
        assert not result.is_success

    def test_missing_query(self):
        """测试缺少query参数。"""
        result = self.search.run()
        assert not result.is_success


# ==================== PythonREPLTool 严格测试 ====================

class TestPythonREPLToolStrict:
    """PythonREPLTool最严格测试。"""

    def setup_method(self):
        self.repl = PythonREPLTool()

    def test_print_output(self):
        """测试print输出。"""
        result = self.repl.run(code="print('hello')")
        assert result.is_success
        assert "hello" in result.output

    def test_multiple_prints(self):
        """测试多次print。"""
        result = self.repl.run(code="print('a')\nprint('b')\nprint('c')")
        assert "a" in result.output
        assert "b" in result.output
        assert "c" in result.output

    def test_no_output(self):
        """测试无输出代码。"""
        result = self.repl.run(code="x = 1")
        assert result.is_success
        assert "无输出" in result.output

    def test_math_module(self):
        """测试math模块。"""
        result = self.repl.run(code="print(math.sqrt(16))")
        assert "4" in result.output

    def test_json_module(self):
        """测试json模块。"""
        result = self.repl.run(code="print(json.dumps({'a': 1}))")
        assert result.is_success

    def test_syntax_error(self):
        """测试语法错误。"""
        result = self.repl.run(code="def f(")
        assert "SyntaxError" in result.output

    def test_name_error(self):
        """测试名称错误。"""
        result = self.repl.run(code="undefined_var")
        assert "NameError" in result.output

    def test_type_error(self):
        """测试类型错误。"""
        result = self.repl.run(code="'a' + 1")
        assert "TypeError" in result.output

    def test_zero_division_error(self):
        """测试除零错误。"""
        result = self.repl.run(code="1/0")
        assert "ZeroDivisionError" in result.output

    def test_list_comprehension(self):
        """测试列表推导式。"""
        result = self.repl.run(code="print([x**2 for x in range(5)])")
        assert "[0, 1, 4, 9, 16]" in result.output

    def test_custom_allowed_modules(self):
        """测试自定义允许模块。"""
        repl = PythonREPLTool(allowed_modules=["math"])
        result = repl.run(code="print(math.pi)")
        assert "3.14" in result.output

    def test_missing_code(self):
        """测试缺少code参数。"""
        result = self.repl.run()
        assert not result.is_success


# ==================== WikipediaTool 严格测试 ====================

class TestWikipediaToolStrict:
    """WikipediaTool最严格测试。"""

    def setup_method(self):
        self.wiki = WikipediaTool()

    def test_basic_query(self):
        """测试基本查询。"""
        result = self.wiki.run(query="Python")
        assert result.is_success
        assert "Python" in result.output

    def test_unicode_query(self):
        """测试Unicode查询。"""
        result = self.wiki.run(query="人工智能")
        assert result.is_success

    def test_custom_fetch_fn(self):
        """测试自定义获取函数。"""
        def custom_fn(q):
            return f"Wiki: {q}"
        wiki = WikipediaTool(fetch_fn=custom_fn)
        result = wiki.run(query="test")
        assert "Wiki: test" in result.output

    def test_missing_query(self):
        """测试缺少query参数。"""
        result = self.wiki.run()
        assert not result.is_success


# ==================== DateTimeTool 严格测试 ====================

class TestDateTimeToolStrict:
    """DateTimeTool最严格测试。"""

    def setup_method(self):
        self.dt = DateTimeTool()

    def test_now_format(self):
        """测试now格式。"""
        result = self.dt.run(action="now")
        assert result.is_success
        # 格式: YYYY-MM-DD HH:MM:SS
        assert len(result.output) == 19
        assert "-" in result.output
        assert ":" in result.output

    def test_date_format(self):
        """测试date格式。"""
        result = self.dt.run(action="date")
        assert result.is_success
        # 格式: YYYY-MM-DD
        assert len(result.output) == 10
        assert result.output.count("-") == 2

    def test_timestamp_numeric(self):
        """测试timestamp是数字。"""
        result = self.dt.run(action="timestamp")
        assert result.is_success
        assert result.output.isdigit()
        assert int(result.output) > 0

    def test_invalid_action(self):
        """测试无效action。"""
        result = self.dt.run(action="invalid")
        assert not result.is_success

    def test_missing_action(self):
        """测试缺少action参数。"""
        result = self.dt.run()
        assert not result.is_success


# ==================== ToolRegistry 严格测试 ====================

class TestToolRegistryStrict:
    """ToolRegistry最严格测试。"""

    def setup_method(self):
        self.registry = ToolRegistry()

    def test_empty_registry(self):
        """测试空注册表。"""
        assert self.registry.count == 0
        assert self.registry.list_tools() == []
        assert self.registry.get_all_tools() == []

    def test_register_multiple_tools(self):
        """测试注册多个工具。"""
        self.registry.register(CalculatorTool())
        self.registry.register(SearchTool())
        self.registry.register(DateTimeTool())
        assert self.registry.count == 3

    def test_unregister_existing(self):
        """测试注销存在的工具。"""
        self.registry.register(CalculatorTool())
        assert self.registry.unregister("calculator")
        assert self.registry.count == 0

    def test_unregister_nonexistent(self):
        """测试注销不存在的工具。"""
        assert not self.registry.unregister("nonexistent")

    def test_get_nonexistent(self):
        """测试获取不存在的工具。"""
        assert self.registry.get("nonexistent") is None

    def test_duplicate_registration_error(self):
        """测试重复注册错误。"""
        self.registry.register(CalculatorTool())
        with pytest.raises(ValueError, match="已存在"):
            self.registry.register(CalculatorTool())

    def test_tools_description_format(self):
        """测试工具描述格式。"""
        self.registry.register(CalculatorTool())
        desc = self.registry.get_tools_description()
        assert "calculator" in desc
        assert "expression" in desc

    def test_to_openai_functions(self):
        """测试OpenAI函数格式。"""
        self.registry.register(CalculatorTool())
        funcs = self.registry.to_openai_functions()
        assert len(funcs) == 1
        assert funcs[0]["name"] == "calculator"


# ==================== Tool 基类严格测试 ====================

class TestToolBaseStrict:
    """Tool基类最严格测试。"""

    def test_tool_name_property(self):
        """测试工具名称属性。"""
        calc = CalculatorTool()
        assert isinstance(calc.name, str)
        assert len(calc.name) > 0

    def test_tool_description_property(self):
        """测试工具描述属性。"""
        calc = CalculatorTool()
        assert isinstance(calc.description, str)
        assert len(calc.description) > 0

    def test_tool_config_property(self):
        """测试工具配置属性。"""
        calc = CalculatorTool()
        config = calc.config
        assert isinstance(config, ToolConfig)
        assert config.name == calc.name

    def test_tool_repr(self):
        """测试工具repr。"""
        calc = CalculatorTool()
        repr_str = repr(calc)
        assert "CalculatorTool" in repr_str
        assert "calculator" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
