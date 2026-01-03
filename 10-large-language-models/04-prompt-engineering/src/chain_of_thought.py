"""
Chain-of-Thought (CoT) 推理实现

本模块提供思维链推理策略，包括 Zero-shot CoT、Few-shot CoT 和 Self-Consistency。

核心组件：
    - CoTExample: CoT 示例
    - CoTPrompt: CoT 提示构建器
    - SelfConsistency: 自洽性推理
    - TreeOfThought: 思维树推理

"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union


__all__ = [
    "CoTExample",
    "CoTPrompt",
    "SelfConsistency",
    "TreeOfThought",
    "CoTExamples",
]


@dataclass
class CoTExample:
    """CoT 示例，包含推理过程。

    参数：
        question: 问题
        reasoning: 推理步骤
        answer: 答案
    """

    question: str
    reasoning: str
    answer: str

    def format(self, template: Optional[str] = None) -> str:
        """格式化 CoT 示例。

        参数：
            template: 可选的格式化模板

        返回：
            格式化后的字符串
        """
        if template:
            return template.format(
                question=self.question,
                reasoning=self.reasoning,
                answer=self.answer
            )
        return f"问题：{self.question}\n思考：{self.reasoning}\n答案：{self.answer}"


class CoTPrompt:
    """Chain-of-Thought 提示构建器。

    支持多种 CoT 策略：
    - zero_shot_cot: 添加 "让我们一步一步思考"
    - few_shot_cot: 提供带推理过程的示例
    - plan_and_solve: 先制定计划再解决

    参数：
        strategy: CoT 策略
        examples: Few-shot CoT 示例
        language: 语言 ("zh" 或 "en")
        custom_trigger: 自定义触发词

    示例：
        >>> cot = CoTPrompt(strategy="zero_shot_cot")
        >>> prompt = cot.format("如果小明有5个苹果，给了小红2个，还剩几个？")
    """

    # 不同语言的 CoT 触发词
    COT_TRIGGERS = {
        "zh": "让我们一步一步思考：",
        "en": "Let's think step by step:",
        "zh_detailed": "让我们仔细分析这个问题，一步一步来思考：",
        "en_detailed": "Let's break this down and think through it carefully step by step:",
    }

    # Plan-and-Solve 触发词
    PLAN_TRIGGERS = {
        "zh": "让我们先制定一个计划，然后逐步解决这个问题：",
        "en": "Let's first devise a plan and then solve the problem step by step:",
    }

    def __init__(
        self,
        strategy: str = "zero_shot_cot",
        examples: Optional[List[CoTExample]] = None,
        language: str = "zh",
        custom_trigger: Optional[str] = None
    ) -> None:
        """初始化 CoT 提示构建器。"""
        self.strategy = strategy
        self.examples = examples or []
        self.language = language
        self.custom_trigger = custom_trigger

    def _get_trigger(self) -> str:
        """获取 CoT 触发词。"""
        if self.custom_trigger:
            return self.custom_trigger

        if self.strategy == "plan_and_solve":
            return self.PLAN_TRIGGERS.get(self.language, self.PLAN_TRIGGERS["en"])

        return self.COT_TRIGGERS.get(self.language, self.COT_TRIGGERS["en"])

    def format(self, question: str, **kwargs) -> str:
        """格式化 CoT 提示。

        参数：
            question: 问题文本
            **kwargs: 额外参数

        返回：
            格式化后的提示字符串

        异常：
            ValueError: 未知策略或缺少示例
        """
        if self.strategy == "zero_shot_cot":
            return self._format_zero_shot(question)
        elif self.strategy == "few_shot_cot":
            return self._format_few_shot(question)
        elif self.strategy == "plan_and_solve":
            return self._format_plan_and_solve(question)
        else:
            raise ValueError(f"未知策略: {self.strategy}")

    def _format_zero_shot(self, question: str) -> str:
        """Zero-shot CoT 格式。"""
        trigger = self._get_trigger()
        return f"问题：{question}\n\n{trigger}\n"

    def _format_few_shot(self, question: str) -> str:
        """Few-shot CoT 格式。"""
        if not self.examples:
            raise ValueError("Few-shot CoT 需要提供示例")

        parts = []
        for ex in self.examples:
            parts.append(ex.format())

        parts.append(f"问题：{question}\n思考：")
        return "\n\n".join(parts)

    def _format_plan_and_solve(self, question: str) -> str:
        """Plan-and-Solve 格式。"""
        trigger = self._get_trigger()
        template = f"""问题：{question}

{trigger}

计划：
1. 首先，理解问题要求
2. 然后，识别关键信息
3. 接着，制定解决步骤
4. 最后，执行计算并验证

执行：
"""
        return template

    def add_example(self, example: CoTExample) -> None:
        """添加 CoT 示例。

        参数：
            example: 要添加的示例
        """
        self.examples.append(example)


class SelfConsistency:
    """Self-Consistency 自洽性推理。

    通过多次采样并投票选择最一致的答案。

    参数：
        n_samples: 采样次数
        temperature: 采样温度
        answer_extractor: 从响应中提取答案的函数

    示例：
        >>> sc = SelfConsistency(n_samples=5, temperature=0.7)
        >>> # 假设有一个生成函数
        >>> answer = sc.generate(prompt, generate_fn)
    """

    def __init__(
        self,
        n_samples: int = 5,
        temperature: float = 0.7,
        answer_extractor: Optional[Callable[[str], str]] = None
    ) -> None:
        """初始化自洽性推理器。"""
        self.n_samples = n_samples
        self.temperature = temperature
        self.answer_extractor = answer_extractor or self._default_extractor

    def _default_extractor(self, response: str) -> str:
        """默认答案提取器。"""
        # 尝试提取 "答案：" 后的内容
        patterns = [
            r"答案[：:]\s*(.+?)(?:\n|$)",
            r"Answer[：:]\s*(.+?)(?:\n|$)",
            r"因此[，,]?\s*(.+?)(?:\n|$)",
            r"所以[，,]?\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 如果没找到，返回最后一行
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else response

    def generate(
        self,
        prompt: str,
        generate_fn: Callable[[str, float], str]
    ) -> Dict[str, Any]:
        """使用 Self-Consistency 生成答案。

        参数：
            prompt: 输入提示
            generate_fn: 生成函数，接受 (prompt, temperature) 返回响应

        返回：
            包含最终答案和详细信息的字典
        """
        responses = []
        answers = []

        for _ in range(self.n_samples):
            response = generate_fn(prompt, self.temperature)
            answer = self.answer_extractor(response)
            responses.append(response)
            answers.append(answer)

        # 投票选择最常见的答案
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]

        return {
            "answer": most_common_answer,
            "confidence": count / self.n_samples,
            "all_answers": answers,
            "answer_distribution": dict(answer_counts),
            "all_responses": responses
        }


class TreeOfThought:
    """Tree-of-Thought (ToT) 思维树推理。

    通过探索多个推理路径并评估选择最优路径。

    参数：
        n_branches: 每个节点的分支数
        max_depth: 最大深度
        evaluator: 评估函数，对推理步骤打分
    """

    def __init__(
        self,
        n_branches: int = 3,
        max_depth: int = 3,
        evaluator: Optional[Callable[[str], float]] = None
    ) -> None:
        """初始化思维树推理器。"""
        self.n_branches = n_branches
        self.max_depth = max_depth
        self.evaluator = evaluator or (lambda x: 0.5)

    def generate_thoughts(
        self,
        state: str,
        generate_fn: Callable[[str], List[str]]
    ) -> List[str]:
        """生成多个思考分支。

        参数：
            state: 当前状态
            generate_fn: 生成函数

        返回：
            思考分支列表
        """
        prompt = f"""当前状态：
{state}

请提出{self.n_branches}个不同的下一步思考方向，每个方向用数字标记：
"""
        response = generate_fn(prompt)

        # 解析多个思考
        thoughts = []
        if isinstance(response, list):
            thoughts = response[:self.n_branches]
        else:
            # 尝试解析编号列表
            pattern = r'\d+[.、)]\s*(.+?)(?=\d+[.、)]|$)'
            matches = re.findall(pattern, response, re.DOTALL)
            thoughts = [m.strip() for m in matches][:self.n_branches]

        return thoughts

    def evaluate_thought(self, thought: str) -> float:
        """评估思考的质量。

        参数：
            thought: 思考内容

        返回：
            质量分数
        """
        return self.evaluator(thought)

    def search(
        self,
        question: str,
        generate_fn: Callable[[str], Union[str, List[str]]],
        strategy: str = "bfs"
    ) -> Dict[str, Any]:
        """搜索最优推理路径。

        参数：
            question: 问题
            generate_fn: 生成函数
            strategy: 搜索策略 ("bfs" 或 "dfs")

        返回：
            最优路径和答案
        """
        if strategy == "bfs":
            return self._bfs_search(question, generate_fn)
        else:
            return self._dfs_search(question, generate_fn)

    def _bfs_search(
        self,
        question: str,
        generate_fn: Callable
    ) -> Dict[str, Any]:
        """广度优先搜索。"""
        # 初始状态
        current_states = [f"问题：{question}"]
        best_path: List[Dict[str, Any]] = []

        for depth in range(self.max_depth):
            all_thoughts: List[Dict[str, Any]] = []

            for state in current_states:
                thoughts = self.generate_thoughts(state, generate_fn)
                for thought in thoughts:
                    score = self.evaluate_thought(thought)
                    all_thoughts.append({
                        "state": state,
                        "thought": thought,
                        "score": score
                    })

            # 选择最好的分支继续
            all_thoughts.sort(key=lambda x: x["score"], reverse=True)
            top_thoughts = all_thoughts[:self.n_branches]

            current_states = [
                f"{t['state']}\n思考{depth+1}：{t['thought']}"
                for t in top_thoughts
            ]

            if top_thoughts:
                best_path.append(top_thoughts[0])

        return {
            "best_path": best_path,
            "final_state": current_states[0] if current_states else "",
            "depth": self.max_depth
        }

    def _dfs_search(
        self,
        question: str,
        generate_fn: Callable
    ) -> Dict[str, Any]:
        """深度优先搜索。"""
        best_result: Dict[str, Any] = {
            "score": -float("inf"),
            "path": [],
            "state": ""
        }

        def dfs(state: str, path: List[str], depth: int) -> None:
            nonlocal best_result

            if depth >= self.max_depth:
                score = (
                    sum(self.evaluate_thought(p) for p in path) / len(path)
                    if path else 0
                )
                if score > best_result["score"]:
                    best_result = {
                        "score": score,
                        "path": path.copy(),
                        "state": state
                    }
                return

            thoughts = self.generate_thoughts(state, generate_fn)

            for thought in thoughts:
                new_state = f"{state}\n思考{depth+1}：{thought}"
                path.append(thought)
                dfs(new_state, path, depth + 1)
                path.pop()

        dfs(f"问题：{question}", [], 0)
        return best_result


class CoTExamples:
    """预定义的 CoT 示例库。

    提供常用任务的标准化 CoT 示例。
    """

    MATH_EXAMPLES = [
        CoTExample(
            question="小明有5个苹果，给了小红2个，又买了3个，现在有几个？",
            reasoning="""1. 初始苹果数：5个
2. 给出后剩余：5 - 2 = 3个
3. 买入后总数：3 + 3 = 6个""",
            answer="6个"
        ),
        CoTExample(
            question="一个班有30人，转走5人，又转来8人，现在有多少人？",
            reasoning="""1. 初始人数：30人
2. 转走后：30 - 5 = 25人
3. 转来后：25 + 8 = 33人""",
            answer="33人"
        ),
    ]
    
    LOGIC_EXAMPLES = [
        CoTExample(
            question="所有的猫都是动物，小花是一只猫，小花是动物吗？",
            reasoning="""1. 前提1：所有的猫都是动物
2. 前提2：小花是一只猫
3. 根据前提1，猫属于动物类别
4. 根据前提2，小花属于猫类别
5. 因此，小花属于动物类别""",
            answer="是的，小花是动物"
        ),
    ]


if __name__ == "__main__":
    # 测试 Zero-shot CoT
    print("=== Zero-shot CoT 测试 ===")
    cot = CoTPrompt(strategy="zero_shot_cot")
    print(cot.format("如果一个商店有15个苹果，卖出8个，又进货12个，现在有多少个？"))
    
    # 测试 Few-shot CoT
    print("\n=== Few-shot CoT 测试 ===")
    cot_few = CoTPrompt(strategy="few_shot_cot", examples=CoTExamples.MATH_EXAMPLES)
    print(cot_few.format("小红有10元钱，买了3元的笔，又买了2元的本子，还剩多少钱？"))
    
    # 测试 Plan-and-Solve
    print("\n=== Plan-and-Solve 测试 ===")
    cot_plan = CoTPrompt(strategy="plan_and_solve")
    print(cot_plan.format("计算 (2+3) × (4+5) - 10 的结果"))
