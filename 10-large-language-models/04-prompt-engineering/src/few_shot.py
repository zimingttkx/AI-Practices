"""
Few-shot 学习实现

本模块提供 Few-shot 学习的核心组件，包括示例选择和提示构建。

核心组件：
    - Example: Few-shot 示例
    - ExampleSelector: 示例选择器基类
    - FewShotPrompt: Few-shot 提示构建器

"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


__all__ = [
    "Example",
    "ExampleSelector",
    "RandomExampleSelector",
    "SemanticExampleSelector",
    "DiversityExampleSelector",
    "FewShotPrompt",
    "FewShotTemplates",
]


@dataclass
class Example:
    """Few-shot 示例。

    参数：
        input_data: 输入数据字典
        output: 期望输出
        metadata: 可选的元数据
    """

    input_data: Dict[str, Any]
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format(self, template: str) -> str:
        """使用模板格式化示例。

        参数：
            template: 格式化模板

        返回：
            格式化后的字符串
        """
        all_vars = {**self.input_data, "output": self.output}
        return template.format(**all_vars)


class ExampleSelector(ABC):
    """示例选择器基类。

    定义示例选择器的接口规范。
    """

    @abstractmethod
    def select(self, query: str, k: int = 3) -> List[Example]:
        """选择最相关的k个示例。

        参数：
            query: 查询文本
            k: 选择的示例数量

        返回：
            选中的示例列表
        """
        pass

    @abstractmethod
    def add_example(self, example: Example) -> None:
        """添加示例到选择器。

        参数：
            example: 要添加的示例
        """
        pass


class RandomExampleSelector(ExampleSelector):
    """随机示例选择器。

    参数：
        examples: 初始示例列表
        seed: 随机种子
    """

    def __init__(
        self,
        examples: Optional[List[Example]] = None,
        seed: int = 42
    ) -> None:
        self.examples = examples or []
        self.seed = seed
        random.seed(seed)

    def select(self, query: str, k: int = 3) -> List[Example]:
        """随机选择k个示例。"""
        k = min(k, len(self.examples))
        return random.sample(self.examples, k)

    def add_example(self, example: Example) -> None:
        """添加示例。"""
        self.examples.append(example)


class SemanticExampleSelector(ExampleSelector):
    """基于语义相似度的示例选择器。

    使用嵌入向量计算相似度，选择最相关的示例。

    参数：
        examples: 初始示例列表
        embedding_fn: 嵌入函数，将文本转换为向量
        input_key: 输入数据中用于计算相似度的键
    """

    def __init__(
        self,
        examples: Optional[List[Example]] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        input_key: str = "input"
    ) -> None:
        self.examples = examples or []
        self.embedding_fn = embedding_fn or self._default_embedding
        self.input_key = input_key
        self.embeddings: List[np.ndarray] = []

        # 预计算示例嵌入
        for ex in self.examples:
            text = ex.input_data.get(self.input_key, str(ex.input_data))
            self.embeddings.append(self.embedding_fn(text))

    def _default_embedding(self, text: str) -> np.ndarray:
        """默认嵌入函数（简单的字符级哈希）。"""
        # 实际使用时应替换为真实的嵌入模型
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度。"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def select(self, query: str, k: int = 3) -> List[Example]:
        """选择与查询最相似的k个示例。"""
        if not self.examples:
            return []

        query_embedding = self.embedding_fn(query)
        similarities = [
            self._cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]

        # 获取top-k索引
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.examples[i] for i in top_k_indices]

    def add_example(self, example: Example) -> None:
        """添加示例。"""
        self.examples.append(example)
        text = example.input_data.get(self.input_key, str(example.input_data))
        self.embeddings.append(self.embedding_fn(text))


class DiversityExampleSelector(ExampleSelector):
    """多样性示例选择器。

    使用 MMR (Maximal Marginal Relevance) 平衡相关性和多样性。

    参数：
        examples: 初始示例列表
        embedding_fn: 嵌入函数
        lambda_param: 相关性与多样性的权衡参数 (0-1)
        input_key: 输入数据中用于计算相似度的键
    """

    def __init__(
        self,
        examples: Optional[List[Example]] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        lambda_param: float = 0.5,
        input_key: str = "input"
    ) -> None:
        self.examples = examples or []
        self.embedding_fn = embedding_fn or self._default_embedding
        self.lambda_param = lambda_param
        self.input_key = input_key
        self.embeddings: List[np.ndarray] = []

        for ex in self.examples:
            text = ex.input_data.get(self.input_key, str(ex.input_data))
            self.embeddings.append(self.embedding_fn(text))

    def _default_embedding(self, text: str) -> np.ndarray:
        """默认嵌入函数。"""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度。"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def select(self, query: str, k: int = 3) -> List[Example]:
        """使用 MMR 算法选择示例。

        参数：
            query: 查询文本
            k: 选择的示例数量

        返回：
            选中的示例列表
        """
        if not self.examples:
            return []

        query_embedding = self.embedding_fn(query)

        # 计算与查询的相似度
        query_similarities = [
            self._cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]

        selected_indices: List[int] = []
        remaining_indices = list(range(len(self.examples)))

        for _ in range(min(k, len(self.examples))):
            mmr_scores = []
            for idx in remaining_indices:
                relevance = query_similarities[idx]

                # 计算与已选示例的最大相似度
                if selected_indices:
                    diversity = max(
                        self._cosine_similarity(
                            self.embeddings[idx], self.embeddings[sel]
                        )
                        for sel in selected_indices
                    )
                else:
                    diversity = 0

                # MMR 分数
                mmr = (
                    self.lambda_param * relevance
                    - (1 - self.lambda_param) * diversity
                )
                mmr_scores.append((idx, mmr))

            # 选择 MMR 分数最高的
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [self.examples[i] for i in selected_indices]

    def add_example(self, example: Example) -> None:
        """添加示例。"""
        self.examples.append(example)
        text = example.input_data.get(self.input_key, str(example.input_data))
        self.embeddings.append(self.embedding_fn(text))


@dataclass
class FewShotPrompt:
    """Few-shot 提示构建器。

    参数：
        examples: 示例列表
        example_template: 示例格式化模板
        prefix: 提示前缀
        suffix: 提示后缀
        example_separator: 示例分隔符
        input_variables: 输入变量列表
        example_selector: 可选的示例选择器

    示例：
        >>> few_shot = FewShotPrompt(
        ...     examples=[
        ...         Example({"input": "高兴"}, "happy"),
        ...         Example({"input": "悲伤"}, "sad"),
        ...     ],
        ...     example_template="输入：{input}\\n输出：{output}",
        ...     prefix="将中文情感词翻译成英文：\\n",
        ...     suffix="\\n输入：{input}\\n输出：",
        ...     input_variables=["input"]
        ... )
        >>> prompt = few_shot.format(input="愤怒")
    """

    examples: List[Example]
    example_template: str
    prefix: str = ""
    suffix: str = ""
    example_separator: str = "\n\n"
    input_variables: List[str] = field(default_factory=list)
    example_selector: Optional[ExampleSelector] = None

    def format(self, **kwargs) -> str:
        """格式化 Few-shot 提示。

        参数：
            **kwargs: 变量名和值的键值对

        返回：
            格式化后的提示字符串
        """
        # 选择示例
        if self.example_selector:
            query = (
                kwargs.get(self.input_variables[0], "")
                if self.input_variables else ""
            )
            selected_examples = self.example_selector.select(str(query))
        else:
            selected_examples = self.examples

        # 格式化示例
        formatted_examples = [
            ex.format(self.example_template) for ex in selected_examples
        ]

        # 组装提示
        parts = [self.prefix] if self.prefix else []
        parts.append(self.example_separator.join(formatted_examples))
        if self.suffix:
            parts.append(self.suffix.format(**kwargs))

        return "".join(parts)

    def add_example(self, example: Example) -> None:
        """添加新示例。

        参数：
            example: 要添加的示例
        """
        self.examples.append(example)
        if self.example_selector:
            self.example_selector.add_example(example)


class FewShotTemplates:
    """预定义的 Few-shot 模板库。

    提供常用任务的标准化 Few-shot 模板。
    """

    @staticmethod
    def sentiment_classification() -> FewShotPrompt:
        """情感分类 Few-shot 模板。

        返回：
            配置好的 FewShotPrompt 实例
        """
        return FewShotPrompt(
            examples=[
                Example({"text": "这个产品太棒了，强烈推荐！"}, "正面"),
                Example({"text": "服务态度很差，不会再来了。"}, "负面"),
                Example({"text": "还行吧，一般般。"}, "中性"),
            ],
            example_template="文本：{text}\n情感：{output}",
            prefix="判断以下文本的情感倾向（正面/负面/中性）：\n\n",
            suffix="\n\n文本：{text}\n情感：",
            input_variables=["text"]
        )

    @staticmethod
    def translation(
        source_lang: str = "中文",
        target_lang: str = "英文"
    ) -> FewShotPrompt:
        """翻译 Few-shot 模板。

        参数：
            source_lang: 源语言
            target_lang: 目标语言

        返回：
            配置好的 FewShotPrompt 实例
        """
        return FewShotPrompt(
            examples=[
                Example({"source": "你好"}, "Hello"),
                Example({"source": "谢谢"}, "Thank you"),
                Example({"source": "再见"}, "Goodbye"),
            ],
            example_template=f"{source_lang}：{{source}}\n{target_lang}：{{output}}",
            prefix=f"将{source_lang}翻译成{target_lang}：\n\n",
            suffix=f"\n\n{source_lang}：{{source}}\n{target_lang}：",
            input_variables=["source"]
        )

    @staticmethod
    def named_entity_recognition() -> FewShotPrompt:
        """命名实体识别 Few-shot 模板。

        返回：
            配置好的 FewShotPrompt 实例
        """
        return FewShotPrompt(
            examples=[
                Example(
                    {"text": "马云创立了阿里巴巴公司。"},
                    "人物：马云\n组织：阿里巴巴公司"
                ),
                Example(
                    {"text": "北京是中国的首都。"},
                    "地点：北京、中国"
                ),
            ],
            example_template="文本：{text}\n实体：\n{output}",
            prefix="从文本中提取命名实体（人物、地点、组织）：\n\n",
            suffix="\n\n文本：{text}\n实体：\n",
            input_variables=["text"]
        )


if __name__ == "__main__":
    # 测试 Few-shot 提示
    print("=== Few-shot 情感分类测试 ===")
    sentiment_prompt = FewShotTemplates.sentiment_classification()
    print(sentiment_prompt.format(text="这家餐厅的菜品非常美味"))
    
    print("\n=== Few-shot 翻译测试 ===")
    translation_prompt = FewShotTemplates.translation()
    print(translation_prompt.format(source="早上好"))
    
    print("\n=== 语义选择器测试 ===")
    selector = SemanticExampleSelector(
        examples=[
            Example({"input": "我很开心"}, "positive"),
            Example({"input": "我很难过"}, "negative"),
            Example({"input": "今天天气不错"}, "neutral"),
        ]
    )
    selected = selector.select("我感到很快乐", k=2)
    for ex in selected:
        print(f"  {ex.input_data} -> {ex.output}")
