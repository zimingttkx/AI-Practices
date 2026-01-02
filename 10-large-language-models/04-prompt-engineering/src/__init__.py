"""
提示工程模块 (Prompt Engineering Module)

本模块提供提示工程的核心组件，包括模板、Few-shot 学习和思维链推理。

核心组件：
    - PromptTemplate: 基础提示模板
    - ChatPromptTemplate: 聊天提示模板
    - FewShotPrompt: Few-shot 提示构建器
    - CoTPrompt: Chain-of-Thought 提示构建器

作者: 研究级实现
许可证: MIT
"""

from .chain_of_thought import (
    CoTExample,
    CoTExamples,
    CoTPrompt,
    SelfConsistency,
    TreeOfThought,
)
from .few_shot import (
    DiversityExampleSelector,
    Example,
    ExampleSelector,
    FewShotPrompt,
    FewShotTemplates,
    RandomExampleSelector,
    SemanticExampleSelector,
)
from .prompt_templates import (
    ChatPromptTemplate,
    JSONOutputParser,
    ListOutputParser,
    Message,
    OutputParser,
    PromptLibrary,
    PromptTemplate,
)

__all__ = [
    # prompt_templates
    "PromptTemplate",
    "Message",
    "ChatPromptTemplate",
    "PromptLibrary",
    "OutputParser",
    "JSONOutputParser",
    "ListOutputParser",
    # few_shot
    "Example",
    "ExampleSelector",
    "RandomExampleSelector",
    "SemanticExampleSelector",
    "DiversityExampleSelector",
    "FewShotPrompt",
    "FewShotTemplates",
    # chain_of_thought
    "CoTExample",
    "CoTPrompt",
    "SelfConsistency",
    "TreeOfThought",
    "CoTExamples",
]
