"""
prompt_templates 模块单元测试

测试覆盖：
    - PromptTemplate 类
    - Message 类
    - ChatPromptTemplate 类
    - PromptLibrary 类
    - OutputParser 类及其子类

"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompt_templates import (
    PromptTemplate,
    Message,
    ChatPromptTemplate,
    PromptLibrary,
    OutputParser,
    JSONOutputParser,
    ListOutputParser,
)


class TestPromptTemplate(unittest.TestCase):
    """PromptTemplate 类测试"""

    def test_basic_format(self):
        """测试基本格式化功能"""
        template = PromptTemplate(
            template="Hello, {name}!",
            input_variables=["name"]
        )
        result = template.format(name="World")
        self.assertEqual(result, "Hello, World!")

    def test_multiple_variables(self):
        """测试多变量格式化"""
        template = PromptTemplate(
            template="{greeting}, {name}! Welcome to {place}.",
            input_variables=["greeting", "name", "place"]
        )
        result = template.format(greeting="Hi", name="Alice", place="Python")
        self.assertEqual(result, "Hi, Alice! Welcome to Python.")

    def test_chinese_content(self):
        """测试中文内容"""
        template = PromptTemplate(
            template="请将以下文本翻译成{target_lang}：\n{text}",
            input_variables=["target_lang", "text"]
        )
        result = template.format(target_lang="英文", text="你好世界")
        self.assertEqual(result, "请将以下文本翻译成英文：\n你好世界")

    def test_missing_variable_raises_error(self):
        """测试缺少变量时抛出异常"""
        template = PromptTemplate(
            template="Hello, {name}!",
            input_variables=["name"]
        )
        with self.assertRaises(ValueError) as context:
            template.format()
        self.assertIn("缺少必需的变量", str(context.exception))

    def test_partial_format(self):
        """测试部分格式化"""
        template = PromptTemplate(
            template="{greeting}, {name}!",
            input_variables=["greeting", "name"]
        )
        partial = template.partial(greeting="Hello")
        self.assertEqual(partial.input_variables, ["name"])
        result = partial.format(name="World")
        self.assertEqual(result, "Hello, World!")

    def test_template_addition(self):
        """测试模板连接"""
        t1 = PromptTemplate(
            template="Part 1: {a}. ",
            input_variables=["a"]
        )
        t2 = PromptTemplate(
            template="Part 2: {b}.",
            input_variables=["b"]
        )
        combined = t1 + t2
        result = combined.format(a="First", b="Second")
        self.assertEqual(result, "Part 1: First. Part 2: Second.")

    def test_save_and_load(self):
        """测试保存和加载"""
        template = PromptTemplate(
            template="Test: {var}",
            input_variables=["var"]
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            temp_path = f.name

        try:
            template.save(temp_path)
            loaded = PromptTemplate.load(temp_path)
            self.assertEqual(loaded.template, template.template)
            self.assertEqual(loaded.input_variables, template.input_variables)
        finally:
            os.unlink(temp_path)

    def test_validation_warning(self):
        """测试变量验证警告"""
        # 声明了但未使用的变量应该产生警告
        template = PromptTemplate(
            template="Hello!",
            input_variables=["unused"],
            validate_template=True
        )
        # 模板创建成功，但会打印警告
        self.assertIsNotNone(template)

    def test_empty_template(self):
        """测试空模板"""
        template = PromptTemplate(
            template="",
            input_variables=[]
        )
        result = template.format()
        self.assertEqual(result, "")


class TestMessage(unittest.TestCase):
    """Message 类测试"""

    def test_message_creation(self):
        """测试消息创建"""
        msg = Message(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")

    def test_to_dict(self):
        """测试转换为字典"""
        msg = Message(role="assistant", content="Hi there!")
        result = msg.to_dict()
        self.assertEqual(result, {"role": "assistant", "content": "Hi there!"})

    def test_system_message(self):
        """测试系统消息"""
        msg = Message(role="system", content="You are a helpful assistant.")
        self.assertEqual(msg.role, "system")


class TestChatPromptTemplate(unittest.TestCase):
    """ChatPromptTemplate 类测试"""

    def test_basic_format(self):
        """测试基本格式化"""
        chat = ChatPromptTemplate(
            messages=[
                Message("system", "You are helpful."),
                Message("user", "Hello, {name}!"),
            ]
        )
        result = chat.format(name="AI")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["content"], "Hello, AI!")

    def test_auto_extract_variables(self):
        """测试自动提取变量"""
        chat = ChatPromptTemplate(
            messages=[
                Message("user", "Translate {text} to {lang}"),
            ]
        )
        self.assertIn("text", chat.input_variables)
        self.assertIn("lang", chat.input_variables)

    def test_format_as_string(self):
        """测试格式化为字符串"""
        chat = ChatPromptTemplate(
            messages=[
                Message("system", "Be helpful."),
                Message("user", "Hi!"),
                Message("assistant", "Hello!"),
            ]
        )
        result = chat.format_as_string()
        self.assertIn("System:", result)
        self.assertIn("User:", result)
        self.assertIn("Assistant:", result)

    def test_from_messages(self):
        """测试从元组列表创建"""
        chat = ChatPromptTemplate.from_messages([
            ("system", "You are helpful."),
            ("user", "Hello!"),
        ])
        self.assertEqual(len(chat.messages), 2)
        self.assertEqual(chat.messages[0].role, "system")


class TestPromptLibrary(unittest.TestCase):
    """PromptLibrary 类测试"""

    def test_classification_template(self):
        """测试分类模板"""
        template = PromptLibrary.CLASSIFICATION
        result = template.format(
            categories="正面, 负面, 中性",
            text="这个产品很好"
        )
        self.assertIn("正面, 负面, 中性", result)
        self.assertIn("这个产品很好", result)

    def test_sentiment_template(self):
        """测试情感分析模板"""
        template = PromptLibrary.SENTIMENT
        result = template.format(text="我很开心")
        self.assertIn("我很开心", result)

    def test_summarization_template(self):
        """测试摘要模板"""
        template = PromptLibrary.SUMMARIZATION
        result = template.format(max_length="100", content="这是一段很长的文本...")
        self.assertIn("这是一段很长的文本...", result)

    def test_translation_template(self):
        """测试翻译模板"""
        template = PromptLibrary.TRANSLATION
        result = template.format(
            source_lang="中文",
            target_lang="英文",
            text="你好"
        )
        self.assertIn("中文", result)
        self.assertIn("英文", result)

    def test_qa_template(self):
        """测试问答模板"""
        template = PromptLibrary.QA
        result = template.format(
            context="Python是一种编程语言。",
            question="Python是什么？"
        )
        self.assertIn("Python是一种编程语言", result)

    def test_code_generation_template(self):
        """测试代码生成模板"""
        template = PromptLibrary.CODE_GENERATION
        result = template.format(
            language="Python",
            requirement="计算两数之和"
        )
        self.assertIn("Python", result)


class TestJSONOutputParser(unittest.TestCase):
    """JSONOutputParser 类测试"""

    def test_parse_json_block(self):
        """测试解析JSON代码块"""
        parser = JSONOutputParser()
        text = '''Here is the result:
```json
{"name": "Alice", "age": 30}
```
'''
        result = parser.parse(text)
        self.assertEqual(result["name"], "Alice")
        self.assertEqual(result["age"], 30)

    def test_parse_raw_json(self):
        """测试解析原始JSON"""
        parser = JSONOutputParser()
        text = '{"key": "value"}'
        result = parser.parse(text)
        self.assertEqual(result["key"], "value")

    def test_parse_invalid_json(self):
        """测试解析无效JSON"""
        parser = JSONOutputParser()
        with self.assertRaises(ValueError) as context:
            parser.parse("not a json")
        self.assertIn("无法解析JSON", str(context.exception))

    def test_format_instructions_with_schema(self):
        """测试带schema的格式说明"""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        parser = JSONOutputParser(schema=schema)
        instructions = parser.get_format_instructions()
        self.assertIn("JSON", instructions)

    def test_format_instructions_without_schema(self):
        """测试不带schema的格式说明"""
        parser = JSONOutputParser()
        instructions = parser.get_format_instructions()
        self.assertIn("JSON", instructions)


class TestListOutputParser(unittest.TestCase):
    """ListOutputParser 类测试"""

    def test_parse_newline_separated(self):
        """测试解析换行分隔的列表"""
        parser = ListOutputParser()
        text = "item1\nitem2\nitem3"
        result = parser.parse(text)
        self.assertEqual(result, ["item1", "item2", "item3"])

    def test_parse_numbered_list(self):
        """测试解析编号列表"""
        parser = ListOutputParser()
        text = "1. First item\n2. Second item\n3. Third item"
        result = parser.parse(text)
        self.assertEqual(result, ["First item", "Second item", "Third item"])

    def test_parse_bullet_list(self):
        """测试解析项目符号列表"""
        parser = ListOutputParser()
        text = "- Item A\n- Item B\n- Item C"
        result = parser.parse(text)
        self.assertEqual(result, ["Item A", "Item B", "Item C"])

    def test_parse_chinese_numbered(self):
        """测试解析中文编号列表"""
        parser = ListOutputParser()
        text = "1、第一项\n2、第二项\n3、第三项"
        result = parser.parse(text)
        self.assertEqual(result, ["第一项", "第二项", "第三项"])

    def test_custom_separator(self):
        """测试自定义分隔符"""
        parser = ListOutputParser(separator=",")
        text = "a, b, c"
        result = parser.parse(text)
        self.assertEqual(result, ["a", "b", "c"])

    def test_format_instructions(self):
        """测试格式说明"""
        parser = ListOutputParser()
        instructions = parser.get_format_instructions()
        self.assertIn("列表", instructions)

    def test_empty_items_filtered(self):
        """测试过滤空项"""
        parser = ListOutputParser()
        text = "item1\n\nitem2\n\n\nitem3"
        result = parser.parse(text)
        self.assertEqual(result, ["item1", "item2", "item3"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
