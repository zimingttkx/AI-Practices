"""
chain_of_thought 模块单元测试

测试覆盖：
    - CoTExample 类
    - CoTPrompt 类
    - SelfConsistency 类
    - TreeOfThought 类
    - CoTExamples 类

"""

from __future__ import annotations

import os
import sys
import unittest
from typing import List
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chain_of_thought import (
    CoTExample,
    CoTPrompt,
    SelfConsistency,
    TreeOfThought,
    CoTExamples,
)


class TestCoTExample(unittest.TestCase):
    """CoTExample 类测试"""

    def test_basic_creation(self):
        """测试基本创建"""
        ex = CoTExample(
            question="1+1=?",
            reasoning="1加1等于2",
            answer="2"
        )
        self.assertEqual(ex.question, "1+1=?")
        self.assertEqual(ex.reasoning, "1加1等于2")
        self.assertEqual(ex.answer, "2")

    def test_default_format(self):
        """测试默认格式化"""
        ex = CoTExample(
            question="问题",
            reasoning="推理过程",
            answer="答案"
        )
        result = ex.format()
        self.assertIn("问题：问题", result)
        self.assertIn("思考：推理过程", result)
        self.assertIn("答案：答案", result)

    def test_custom_format(self):
        """测试自定义格式化"""
        ex = CoTExample(
            question="Q",
            reasoning="R",
            answer="A"
        )
        template = "Q: {question}\nThinking: {reasoning}\nA: {answer}"
        result = ex.format(template)
        self.assertEqual(result, "Q: Q\nThinking: R\nA: A")


class TestCoTPrompt(unittest.TestCase):
    """CoTPrompt 类测试"""

    def test_zero_shot_cot_chinese(self):
        """测试中文 Zero-shot CoT"""
        cot = CoTPrompt(strategy="zero_shot_cot", language="zh")
        result = cot.format("1+1等于多少？")
        self.assertIn("问题：1+1等于多少？", result)
        self.assertIn("让我们一步一步思考", result)

    def test_zero_shot_cot_english(self):
        """测试英文 Zero-shot CoT"""
        cot = CoTPrompt(strategy="zero_shot_cot", language="en")
        result = cot.format("What is 1+1?")
        self.assertIn("What is 1+1?", result)
        self.assertIn("step by step", result)

    def test_few_shot_cot(self):
        """测试 Few-shot CoT"""
        examples = [
            CoTExample("2+2=?", "2加2等于4", "4"),
            CoTExample("3+3=?", "3加3等于6", "6"),
        ]
        cot = CoTPrompt(strategy="few_shot_cot", examples=examples)
        result = cot.format("4+4=?")
        self.assertIn("2+2", result)
        self.assertIn("3+3", result)
        self.assertIn("4+4", result)

    def test_few_shot_cot_no_examples_raises(self):
        """测试 Few-shot CoT 无示例时抛出异常"""
        cot = CoTPrompt(strategy="few_shot_cot", examples=[])
        with self.assertRaises(ValueError) as context:
            cot.format("test")
        self.assertIn("需要提供示例", str(context.exception))

    def test_plan_and_solve(self):
        """测试 Plan-and-Solve 策略"""
        cot = CoTPrompt(strategy="plan_and_solve", language="zh")
        result = cot.format("解决复杂问题")
        self.assertIn("计划", result)
        self.assertIn("执行", result)

    def test_custom_trigger(self):
        """测试自定义触发词"""
        cot = CoTPrompt(
            strategy="zero_shot_cot",
            custom_trigger="请仔细分析："
        )
        result = cot.format("问题")
        self.assertIn("请仔细分析：", result)

    def test_unknown_strategy_raises(self):
        """测试未知策略抛出异常"""
        cot = CoTPrompt(strategy="unknown_strategy")
        with self.assertRaises(ValueError) as context:
            cot.format("test")
        self.assertIn("未知策略", str(context.exception))

    def test_add_example(self):
        """测试添加示例"""
        cot = CoTPrompt(strategy="few_shot_cot")
        cot.add_example(CoTExample("Q", "R", "A"))
        self.assertEqual(len(cot.examples), 1)

    def test_detailed_trigger(self):
        """测试详细触发词"""
        cot = CoTPrompt(strategy="zero_shot_cot", language="zh_detailed")
        result = cot.format("问题")
        self.assertIn("仔细分析", result)


class TestSelfConsistency(unittest.TestCase):
    """SelfConsistency 类测试"""

    def test_basic_generation(self):
        """测试基本生成"""
        def mock_generate(prompt: str, temp: float) -> str:
            return "经过计算，答案：42"

        sc = SelfConsistency(n_samples=3, temperature=0.7)
        result = sc.generate("问题", mock_generate)

        self.assertEqual(result["answer"], "42")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(len(result["all_answers"]), 3)

    def test_voting_mechanism(self):
        """测试投票机制"""
        responses = ["答案：A", "答案：B", "答案：A", "答案：A", "答案：B"]
        idx = [0]

        def mock_generate(prompt: str, temp: float) -> str:
            response = responses[idx[0]]
            idx[0] += 1
            return response

        sc = SelfConsistency(n_samples=5, temperature=0.7)
        result = sc.generate("问题", mock_generate)

        self.assertEqual(result["answer"], "A")
        self.assertEqual(result["confidence"], 0.6)

    def test_custom_extractor(self):
        """测试自定义提取器"""
        def custom_extractor(response: str) -> str:
            return response.split("=")[1].strip()

        sc = SelfConsistency(
            n_samples=2,
            answer_extractor=custom_extractor
        )

        def mock_generate(prompt: str, temp: float) -> str:
            return "result = 100"

        result = sc.generate("问题", mock_generate)
        self.assertEqual(result["answer"], "100")

    def test_default_extractor_patterns(self):
        """测试默认提取器的各种模式"""
        sc = SelfConsistency(n_samples=1)

        # 测试 "答案：" 模式
        result = sc._default_extractor("分析后，答案：42")
        self.assertEqual(result, "42")

        # 测试 "因此" 模式
        result = sc._default_extractor("因此，结果是100")
        self.assertEqual(result, "结果是100")

        # 测试 "所以" 模式
        result = sc._default_extractor("所以答案是50")
        self.assertEqual(result, "答案是50")

        # 测试无匹配时返回最后一行
        result = sc._default_extractor("第一行\n第二行\n最后答案")
        self.assertEqual(result, "最后答案")

    def test_answer_distribution(self):
        """测试答案分布"""
        responses = ["答案：X", "答案：Y", "答案：X"]
        idx = [0]

        def mock_generate(prompt: str, temp: float) -> str:
            response = responses[idx[0]]
            idx[0] += 1
            return response

        sc = SelfConsistency(n_samples=3)
        result = sc.generate("问题", mock_generate)

        self.assertEqual(result["answer_distribution"]["X"], 2)
        self.assertEqual(result["answer_distribution"]["Y"], 1)


class TestTreeOfThought(unittest.TestCase):
    """TreeOfThought 类测试"""

    def test_basic_creation(self):
        """测试基本创建"""
        tot = TreeOfThought(n_branches=3, max_depth=2)
        self.assertEqual(tot.n_branches, 3)
        self.assertEqual(tot.max_depth, 2)

    def test_custom_evaluator(self):
        """测试自定义评估器"""
        def custom_eval(thought: str) -> float:
            return len(thought) / 100.0

        tot = TreeOfThought(evaluator=custom_eval)
        score = tot.evaluate_thought("这是一个测试思考")
        self.assertGreater(score, 0)

    def test_generate_thoughts(self):
        """测试生成思考"""
        tot = TreeOfThought(n_branches=3)

        def mock_generate(prompt: str) -> str:
            return "1. 第一个想法\n2. 第二个想法\n3. 第三个想法"

        thoughts = tot.generate_thoughts("当前状态", mock_generate)
        self.assertLessEqual(len(thoughts), 3)

    def test_generate_thoughts_list_response(self):
        """测试生成思考（列表响应）"""
        tot = TreeOfThought(n_branches=2)

        def mock_generate(prompt: str) -> List[str]:
            return ["想法A", "想法B", "想法C"]

        thoughts = tot.generate_thoughts("状态", mock_generate)
        self.assertEqual(len(thoughts), 2)

    def test_bfs_search(self):
        """测试广度优先搜索"""
        tot = TreeOfThought(n_branches=2, max_depth=2)

        def mock_generate(prompt: str):
            return "1. 思考一\n2. 思考二"

        result = tot.search("问题", mock_generate, strategy="bfs")
        self.assertIn("best_path", result)
        self.assertIn("final_state", result)
        self.assertEqual(result["depth"], 2)

    def test_dfs_search(self):
        """测试深度优先搜索"""
        tot = TreeOfThought(n_branches=2, max_depth=2)

        def mock_generate(prompt: str):
            return "1. 路径A\n2. 路径B"

        result = tot.search("问题", mock_generate, strategy="dfs")
        self.assertIn("path", result)
        self.assertIn("score", result)

    def test_evaluate_thought_default(self):
        """测试默认评估器"""
        tot = TreeOfThought()
        score = tot.evaluate_thought("任意思考")
        self.assertEqual(score, 0.5)


class TestCoTExamples(unittest.TestCase):
    """CoTExamples 类测试"""

    def test_math_examples_exist(self):
        """测试数学示例存在"""
        self.assertGreater(len(CoTExamples.MATH_EXAMPLES), 0)

    def test_math_examples_structure(self):
        """测试数学示例结构"""
        for ex in CoTExamples.MATH_EXAMPLES:
            self.assertIsInstance(ex, CoTExample)
            self.assertTrue(len(ex.question) > 0)
            self.assertTrue(len(ex.reasoning) > 0)
            self.assertTrue(len(ex.answer) > 0)

    def test_logic_examples_exist(self):
        """测试逻辑示例存在"""
        self.assertGreater(len(CoTExamples.LOGIC_EXAMPLES), 0)

    def test_logic_examples_structure(self):
        """测试逻辑示例结构"""
        for ex in CoTExamples.LOGIC_EXAMPLES:
            self.assertIsInstance(ex, CoTExample)
            self.assertTrue(len(ex.question) > 0)

    def test_examples_can_format(self):
        """测试示例可以格式化"""
        for ex in CoTExamples.MATH_EXAMPLES:
            result = ex.format()
            self.assertIn("问题", result)
            self.assertIn("思考", result)
            self.assertIn("答案", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
