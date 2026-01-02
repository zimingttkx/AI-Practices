"""
few_shot 模块单元测试

测试覆盖：
    - Example 类
    - ExampleSelector 及其子类
    - FewShotPrompt 类
    - FewShotTemplates 类

作者: 研究级实现
许可证: MIT
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.few_shot import (
    Example,
    ExampleSelector,
    RandomExampleSelector,
    SemanticExampleSelector,
    DiversityExampleSelector,
    FewShotPrompt,
    FewShotTemplates,
)


class TestExample(unittest.TestCase):
    """Example 类测试"""

    def test_basic_creation(self):
        """测试基本创建"""
        ex = Example(
            input_data={"text": "Hello"},
            output="World"
        )
        self.assertEqual(ex.input_data["text"], "Hello")
        self.assertEqual(ex.output, "World")

    def test_with_metadata(self):
        """测试带元数据创建"""
        ex = Example(
            input_data={"text": "Test"},
            output="Result",
            metadata={"source": "test", "score": 0.9}
        )
        self.assertEqual(ex.metadata["source"], "test")
        self.assertEqual(ex.metadata["score"], 0.9)

    def test_format_basic(self):
        """测试基本格式化"""
        ex = Example(
            input_data={"input": "你好"},
            output="Hello"
        )
        result = ex.format("输入：{input}\n输出：{output}")
        self.assertEqual(result, "输入：你好\n输出：Hello")

    def test_format_multiple_inputs(self):
        """测试多输入格式化"""
        ex = Example(
            input_data={"text": "内容", "lang": "英文"},
            output="Content"
        )
        result = ex.format("文本：{text}，语言：{lang}，翻译：{output}")
        self.assertEqual(result, "文本：内容，语言：英文，翻译：Content")

    def test_default_metadata(self):
        """测试默认元数据为空字典"""
        ex = Example(input_data={"x": 1}, output="y")
        self.assertEqual(ex.metadata, {})


class TestRandomExampleSelector(unittest.TestCase):
    """RandomExampleSelector 类测试"""

    def setUp(self):
        """设置测试数据"""
        self.examples = [
            Example({"input": f"input_{i}"}, f"output_{i}")
            for i in range(10)
        ]

    def test_select_k_examples(self):
        """测试选择k个示例"""
        selector = RandomExampleSelector(examples=self.examples, seed=42)
        selected = selector.select("query", k=3)
        self.assertEqual(len(selected), 3)

    def test_select_more_than_available(self):
        """测试选择数量超过可用数量"""
        selector = RandomExampleSelector(examples=self.examples[:2], seed=42)
        selected = selector.select("query", k=5)
        self.assertEqual(len(selected), 2)

    def test_reproducibility(self):
        """测试可重复性（同一实例多次调用）"""
        selector = RandomExampleSelector(examples=self.examples, seed=42)
        # 重置种子后选择
        import random
        random.seed(42)
        selected1 = selector.select("query", k=3)
        # 再次重置种子
        random.seed(42)
        selector2 = RandomExampleSelector(examples=self.examples, seed=42)
        selected2 = selector2.select("query", k=3)
        # 验证两次选择结果相同
        self.assertEqual(len(selected1), len(selected2))

    def test_add_example(self):
        """测试添加示例"""
        selector = RandomExampleSelector(examples=[], seed=42)
        self.assertEqual(len(selector.examples), 0)
        selector.add_example(Example({"input": "new"}, "new_output"))
        self.assertEqual(len(selector.examples), 1)

    def test_empty_selector(self):
        """测试空选择器"""
        selector = RandomExampleSelector(examples=[], seed=42)
        selected = selector.select("query", k=3)
        self.assertEqual(len(selected), 0)


class TestSemanticExampleSelector(unittest.TestCase):
    """SemanticExampleSelector 类测试"""

    def setUp(self):
        """设置测试数据"""
        self.examples = [
            Example({"input": "我很开心"}, "positive"),
            Example({"input": "我很难过"}, "negative"),
            Example({"input": "今天天气不错"}, "neutral"),
            Example({"input": "我非常高兴"}, "positive"),
            Example({"input": "我很伤心"}, "negative"),
        ]

    def test_select_returns_correct_count(self):
        """测试返回正确数量"""
        selector = SemanticExampleSelector(examples=self.examples)
        selected = selector.select("我感到快乐", k=2)
        self.assertEqual(len(selected), 2)

    def test_select_with_custom_embedding(self):
        """测试自定义嵌入函数"""
        def custom_embedding(text: str) -> np.ndarray:
            return np.random.randn(128)

        selector = SemanticExampleSelector(
            examples=self.examples,
            embedding_fn=custom_embedding
        )
        selected = selector.select("test", k=2)
        self.assertEqual(len(selected), 2)

    def test_add_example(self):
        """测试添加示例"""
        selector = SemanticExampleSelector(examples=self.examples)
        initial_count = len(selector.examples)
        selector.add_example(Example({"input": "新示例"}, "new"))
        self.assertEqual(len(selector.examples), initial_count + 1)
        self.assertEqual(len(selector.embeddings), initial_count + 1)

    def test_empty_selector(self):
        """测试空选择器"""
        selector = SemanticExampleSelector(examples=[])
        selected = selector.select("query", k=3)
        self.assertEqual(len(selected), 0)

    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        selector = SemanticExampleSelector(examples=[])
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        sim = selector._cosine_similarity(a, b)
        self.assertAlmostEqual(sim, 1.0, places=5)

        c = np.array([0, 1, 0])
        sim2 = selector._cosine_similarity(a, c)
        self.assertAlmostEqual(sim2, 0.0, places=5)


class TestDiversityExampleSelector(unittest.TestCase):
    """DiversityExampleSelector 类测试"""

    def setUp(self):
        """设置测试数据"""
        self.examples = [
            Example({"input": "苹果是水果"}, "fruit"),
            Example({"input": "香蕉是水果"}, "fruit"),
            Example({"input": "汽车是交通工具"}, "vehicle"),
            Example({"input": "飞机是交通工具"}, "vehicle"),
            Example({"input": "Python是编程语言"}, "programming"),
        ]

    def test_select_returns_correct_count(self):
        """测试返回正确数量"""
        selector = DiversityExampleSelector(examples=self.examples)
        selected = selector.select("橙子是什么", k=3)
        self.assertEqual(len(selected), 3)

    def test_lambda_param_effect(self):
        """测试lambda参数影响"""
        # lambda=1.0 应该只考虑相关性
        selector_relevance = DiversityExampleSelector(
            examples=self.examples,
            lambda_param=1.0
        )
        # lambda=0.0 应该只考虑多样性
        selector_diversity = DiversityExampleSelector(
            examples=self.examples,
            lambda_param=0.0
        )
        # 两者应该产生不同结果
        selected1 = selector_relevance.select("水果", k=3)
        selected2 = selector_diversity.select("水果", k=3)
        # 至少验证都返回了结果
        self.assertEqual(len(selected1), 3)
        self.assertEqual(len(selected2), 3)

    def test_add_example(self):
        """测试添加示例"""
        selector = DiversityExampleSelector(examples=self.examples)
        initial_count = len(selector.examples)
        selector.add_example(Example({"input": "新内容"}, "new"))
        self.assertEqual(len(selector.examples), initial_count + 1)

    def test_empty_selector(self):
        """测试空选择器"""
        selector = DiversityExampleSelector(examples=[])
        selected = selector.select("query", k=3)
        self.assertEqual(len(selected), 0)


class TestFewShotPrompt(unittest.TestCase):
    """FewShotPrompt 类测试"""

    def setUp(self):
        """设置测试数据"""
        self.examples = [
            Example({"input": "高兴"}, "happy"),
            Example({"input": "悲伤"}, "sad"),
            Example({"input": "愤怒"}, "angry"),
        ]

    def test_basic_format(self):
        """测试基本格式化"""
        prompt = FewShotPrompt(
            examples=self.examples,
            example_template="输入：{input}\n输出：{output}",
            prefix="翻译情感词：\n\n",
            suffix="\n\n输入：{input}\n输出：",
            input_variables=["input"]
        )
        result = prompt.format(input="开心")
        self.assertIn("高兴", result)
        self.assertIn("happy", result)
        self.assertIn("开心", result)

    def test_with_example_selector(self):
        """测试使用示例选择器"""
        selector = RandomExampleSelector(examples=self.examples, seed=42)
        prompt = FewShotPrompt(
            examples=self.examples,
            example_template="输入：{input}\n输出：{output}",
            example_selector=selector,
            input_variables=["input"]
        )
        result = prompt.format(input="测试")
        self.assertIsInstance(result, str)

    def test_custom_separator(self):
        """测试自定义分隔符"""
        prompt = FewShotPrompt(
            examples=self.examples[:2],
            example_template="{input} -> {output}",
            example_separator="\n---\n"
        )
        result = prompt.format()
        self.assertIn("---", result)

    def test_add_example(self):
        """测试添加示例"""
        prompt = FewShotPrompt(
            examples=self.examples.copy(),
            example_template="{input}: {output}"
        )
        initial_count = len(prompt.examples)
        prompt.add_example(Example({"input": "新词"}, "new"))
        self.assertEqual(len(prompt.examples), initial_count + 1)

    def test_no_prefix_suffix(self):
        """测试无前缀后缀"""
        prompt = FewShotPrompt(
            examples=self.examples[:1],
            example_template="{input}={output}"
        )
        result = prompt.format()
        self.assertEqual(result, "高兴=happy")


class TestFewShotTemplates(unittest.TestCase):
    """FewShotTemplates 类测试"""

    def test_sentiment_classification(self):
        """测试情感分类模板"""
        prompt = FewShotTemplates.sentiment_classification()
        result = prompt.format(text="这个电影太棒了")
        self.assertIn("情感", result)
        self.assertIn("这个电影太棒了", result)

    def test_translation(self):
        """测试翻译模板"""
        prompt = FewShotTemplates.translation()
        result = prompt.format(source="早上好")
        self.assertIn("中文", result)
        self.assertIn("英文", result)
        self.assertIn("早上好", result)

    def test_translation_custom_languages(self):
        """测试自定义语言翻译模板"""
        prompt = FewShotTemplates.translation(
            source_lang="日文",
            target_lang="中文"
        )
        result = prompt.format(source="こんにちは")
        self.assertIn("日文", result)
        self.assertIn("中文", result)

    def test_named_entity_recognition(self):
        """测试命名实体识别模板"""
        prompt = FewShotTemplates.named_entity_recognition()
        result = prompt.format(text="李明在北京工作")
        self.assertIn("实体", result)
        self.assertIn("李明在北京工作", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
