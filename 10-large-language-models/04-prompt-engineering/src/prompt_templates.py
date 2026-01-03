"""
提示模板库 (Prompt Templates)

本模块提供结构化的提示模板实现，支持变量替换、格式化和输出解析。

核心组件：
    - PromptTemplate: 基础提示模板
    - ChatPromptTemplate: 聊天提示模板
    - PromptLibrary: 预定义模板库
    - OutputParser: 输出解析器

"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


__all__ = [
    "PromptTemplate",
    "Message",
    "ChatPromptTemplate",
    "PromptLibrary",
    "OutputParser",
    "JSONOutputParser",
    "ListOutputParser",
]


@dataclass
class PromptTemplate:
    """基础提示模板类。

    支持变量替换的提示模板，使用 {variable} 语法。

    参数：
        template: 模板字符串，使用 {var} 语法标记变量
        input_variables: 模板中使用的变量名列表
        template_format: 模板格式 ("f-string" 或 "jinja2")
        validate_template: 是否验证模板变量

    示例：
        >>> template = PromptTemplate(
        ...     template="请将以下文本翻译成{target_lang}：\\n{text}",
        ...     input_variables=["target_lang", "text"]
        ... )
        >>> prompt = template.format(target_lang="英文", text="你好")
    """

    template: str
    input_variables: List[str]
    template_format: str = "f-string"
    validate_template: bool = True

    def __post_init__(self) -> None:
        """初始化后验证模板。"""
        if self.validate_template:
            self._validate()

    def _validate(self) -> None:
        """验证模板中的变量与声明的变量一致。"""
        # 提取模板中的变量
        pattern = r'\{(\w+)\}'
        found_vars = set(re.findall(pattern, self.template))
        declared_vars = set(self.input_variables)
        
        # 检查未声明的变量
        undeclared = found_vars - declared_vars
        if undeclared:
            raise ValueError(f"模板中存在未声明的变量: {undeclared}")
        
        # 检查未使用的变量
        unused = declared_vars - found_vars
        if unused:
            print(f"警告: 声明但未使用的变量: {unused}")
    
    def format(self, **kwargs) -> str:
        """格式化模板，替换变量。

        参数：
            **kwargs: 变量名和值的键值对

        返回：
            格式化后的提示字符串

        异常：
            ValueError: 缺少必需的变量
        """
        # 检查必需变量
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"缺少必需的变量: {missing}")

        return self.template.format(**kwargs)

    def partial(self, **kwargs) -> PromptTemplate:
        """部分填充模板，返回新模板。

        参数：
            **kwargs: 要预填充的变量

        返回：
            新的 PromptTemplate 实例
        """
        new_template = self.template.format(**{
            k: f"{{{k}}}" if k not in kwargs else kwargs[k]
            for k in self.input_variables
        })
        new_variables = [v for v in self.input_variables if v not in kwargs]
        
        return PromptTemplate(
            template=new_template,
            input_variables=new_variables,
            template_format=self.template_format,
            validate_template=False
        )

    def __add__(self, other: PromptTemplate) -> PromptTemplate:
        """连接两个模板。

        参数：
            other: 另一个模板

        返回：
            合并后的新模板
        """
        return PromptTemplate(
            template=self.template + other.template,
            input_variables=list(set(self.input_variables + other.input_variables)),
            validate_template=False
        )

    def save(self, path: str) -> None:
        """保存模板到文件。

        参数：
            path: 文件路径
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "template": self.template,
                "input_variables": self.input_variables,
                "template_format": self.template_format
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> PromptTemplate:
        """从文件加载模板。

        参数：
            path: 文件路径

        返回：
            加载的模板实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class Message:
    """聊天消息。

    参数：
        role: 角色 ("system", "user", "assistant")
        content: 消息内容
    """

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式。"""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatPromptTemplate:
    """聊天提示模板。

    用于构建多轮对话格式的提示。

    参数：
        messages: 消息列表
        input_variables: 模板中使用的变量名列表

    示例：
        >>> chat_template = ChatPromptTemplate(
        ...     messages=[
        ...         Message("system", "你是一个有帮助的助手。"),
        ...         Message("user", "请解释{concept}"),
        ...     ]
        ... )
        >>> messages = chat_template.format(concept="机器学习")
    """

    messages: List[Message]
    input_variables: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """初始化后自动提取变量。"""
        if not self.input_variables:
            all_vars = set()
            pattern = r'\{(\w+)\}'
            for msg in self.messages:
                all_vars.update(re.findall(pattern, msg.content))
            self.input_variables = list(all_vars)

    def format(self, **kwargs) -> List[Dict[str, str]]:
        """格式化所有消息。

        参数：
            **kwargs: 变量名和值的键值对

        返回：
            格式化后的消息列表
        """
        formatted_messages = []
        for msg in self.messages:
            content = msg.content.format(**kwargs)
            formatted_messages.append({
                "role": msg.role,
                "content": content
            })
        return formatted_messages

    def format_as_string(self, **kwargs) -> str:
        """格式化为字符串（用于非Chat模型）。

        参数：
            **kwargs: 变量名和值的键值对

        返回：
            格式化后的字符串
        """
        messages = self.format(**kwargs)
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"Assistant: {msg['content']}")
        return "\n\n".join(parts)

    @classmethod
    def from_messages(cls, messages: List[tuple]) -> ChatPromptTemplate:
        """从元组列表创建模板。

        参数：
            messages: (role, content) 元组列表

        返回：
            ChatPromptTemplate 实例
        """
        return cls(
            messages=[Message(role, content) for role, content in messages]
        )


class PromptLibrary:
    """预定义提示模板库。

    提供常用任务的标准化提示模板。
    """

    # 分类任务
    CLASSIFICATION = PromptTemplate(
        template="""请将以下文本分类为 {categories} 之一。

文本：{text}

只输出类别名称，不要解释。

类别：""",
        input_variables=["categories", "text"]
    )
    
    # 情感分析
    SENTIMENT = PromptTemplate(
        template="""分析以下文本的情感倾向。

文本：{text}

请以JSON格式输出：
{{
    "sentiment": "正面/负面/中性",
    "confidence": 0.0-1.0,
    "reason": "简短解释"
}}

分析结果：""",
        input_variables=["text"]
    )
    
    # 摘要生成
    SUMMARIZATION = PromptTemplate(
        template="""请用{max_length}字以内总结以下内容的要点：

{content}

摘要：""",
        input_variables=["max_length", "content"]
    )
    
    # 翻译
    TRANSLATION = PromptTemplate(
        template="""请将以下{source_lang}文本翻译成{target_lang}，保持原文风格：

{text}

翻译：""",
        input_variables=["source_lang", "target_lang", "text"]
    )
    
    # 信息提取
    EXTRACTION = PromptTemplate(
        template="""从以下文本中提取{entities}信息：

文本：{text}

请以JSON格式输出提取结果：""",
        input_variables=["entities", "text"]
    )
    
    # 代码生成
    CODE_GENERATION = PromptTemplate(
        template="""请用{language}实现以下功能：

需求：{requirement}

要求：
- 代码简洁清晰
- 添加必要注释
- 处理边界情况

代码：
```{language}
""",
        input_variables=["language", "requirement"]
    )
    
    # 代码解释
    CODE_EXPLANATION = PromptTemplate(
        template="""请解释以下{language}代码的功能：

```{language}
{code}
```

解释：""",
        input_variables=["language", "code"]
    )
    
    # 问答
    QA = PromptTemplate(
        template="""基于以下上下文回答问题。如果上下文中没有相关信息，请说"根据提供的信息无法回答"。

上下文：
{context}

问题：{question}

回答：""",
        input_variables=["context", "question"]
    )


class OutputParser(ABC):
    """输出解析器基类。

    定义输出解析器的接口规范。
    """

    @abstractmethod
    def parse(self, text: str) -> Any:
        """解析模型输出。

        参数：
            text: 模型输出文本

        返回：
            解析后的结构化数据
        """
        pass

    @abstractmethod
    def get_format_instructions(self) -> str:
        """获取格式说明。

        返回：
            格式说明字符串
        """
        pass


class JSONOutputParser(OutputParser):
    """JSON输出解析器。

    参数：
        schema: 可选的JSON schema定义
    """

    def __init__(self, schema: Optional[Dict] = None) -> None:
        self.schema = schema

    def parse(self, text: str) -> Dict:
        """解析JSON输出。

        参数：
            text: 包含JSON的文本

        返回：
            解析后的字典

        异常：
            ValueError: JSON解析失败
        """
        # 尝试提取JSON块
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # 尝试提取花括号内容
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析JSON: {e}\n原始文本: {text}")
    
    def get_format_instructions(self) -> str:
        """获取格式说明。"""
        if self.schema:
            return f"请以JSON格式输出，schema: {json.dumps(self.schema, ensure_ascii=False)}"
        return "请以JSON格式输出"


class ListOutputParser(OutputParser):
    """列表输出解析器。

    参数：
        separator: 列表项分隔符
    """

    def __init__(self, separator: str = "\n") -> None:
        self.separator = separator

    def parse(self, text: str) -> List[str]:
        """解析列表输出。

        参数：
            text: 包含列表的文本

        返回：
            解析后的字符串列表
        """
        items = text.strip().split(self.separator)
        # 清理每个项目
        cleaned = []
        for item in items:
            item = item.strip()
            # 移除常见的列表标记
            item = re.sub(r'^[\d]+[.、)]\s*', '', item)
            item = re.sub(r'^[-*•]\s*', '', item)
            if item:
                cleaned.append(item)
        return cleaned

    def get_format_instructions(self) -> str:
        """获取格式说明。"""
        return f"请以列表形式输出，每项用'{self.separator}'分隔"


if __name__ == "__main__":
    # 测试基础模板
    template = PromptTemplate(
        template="请将以下文本翻译成{target_lang}：\n{text}",
        input_variables=["target_lang", "text"]
    )
    print("=== 基础模板测试 ===")
    print(template.format(target_lang="英文", text="今天天气很好"))
    
    # 测试聊天模板
    chat = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的翻译助手。"),
        ("user", "请将'{text}'翻译成{lang}"),
    ])
    print("\n=== 聊天模板测试 ===")
    print(chat.format(text="你好世界", lang="英文"))
    
    # 测试预定义模板
    print("\n=== 预定义模板测试 ===")
    print(PromptLibrary.SENTIMENT.format(text="这个产品太棒了！"))
