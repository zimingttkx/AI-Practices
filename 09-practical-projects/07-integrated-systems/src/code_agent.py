"""
代码生成智能体。

基于检索增强的代码生成和补全。

核心组件:
    - CodeAction: 代码动作
    - CodeResult: 代码结果
    - CodeAgent: 代码生成智能体
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .code_retriever import (
    CodeDocument,
    CodeLanguage,
    CodeRetriever,
    SearchResult,
)


class CodeActionType(Enum):
    """代码动作类型。"""
    GENERATE = "generate"
    COMPLETE = "complete"
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    FIX = "fix"
    SEARCH = "search"


@dataclass
class CodeAction:
    """代码动作。"""
    action_type: CodeActionType
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class CodeResult:
    """代码生成结果。"""
    code: str
    language: CodeLanguage = CodeLanguage.UNKNOWN
    explanation: str = ""
    references: List[CodeDocument] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    
    @property
    def num_lines(self) -> int:
        return self.code.count("\n") + 1 if self.code else 0


@dataclass
class AgentStep:
    """智能体步骤。"""
    step_num: int
    thought: str
    action: CodeAction
    result: Any


class CodeAgent:
    """代码生成智能体。
    
    结合代码检索和LLM生成能力。
    
    示例:
        >>> agent = CodeAgent(retriever=retriever)
        >>> result = agent.generate("实现快速排序算法", language=CodeLanguage.PYTHON)
    """
    
    def __init__(
        self,
        retriever: Optional[CodeRetriever] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        max_steps: int = 5,
    ) -> None:
        self.retriever = retriever or CodeRetriever()
        self.llm_func = llm_func or self._mock_llm
        self.max_steps = max_steps
        self._history: List[AgentStep] = []
    
    def _mock_llm(self, prompt: str) -> str:
        """模拟LLM响应。"""
        if "快速排序" in prompt or "quicksort" in prompt.lower():
            return '''def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)'''
        elif "二分查找" in prompt or "binary search" in prompt.lower():
            return '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1'''
        else:
            return "# 代码生成占位符\npass"
    
    def generate(
        self,
        task: str,
        language: CodeLanguage = CodeLanguage.PYTHON,
        context: Optional[str] = None,
    ) -> CodeResult:
        """生成代码。"""
        # 检索相关代码
        references = []
        if self.retriever.num_documents > 0:
            results = self.retriever.search(task, top_k=3, language_filter=language)
            references = [r.document for r in results]
        
        # 构建提示
        prompt = self._build_generate_prompt(task, language, context, references)
        
        # 生成代码
        code = self.llm_func(prompt)
        
        # 记录步骤
        self._history.append(AgentStep(
            step_num=len(self._history) + 1,
            thought=f"生成{language.value}代码: {task}",
            action=CodeAction(CodeActionType.GENERATE, {"task": task}),
            result=code,
        ))
        
        return CodeResult(
            code=code,
            language=language,
            explanation=f"根据任务'{task}'生成的代码",
            references=references,
        )
    
    def complete(
        self,
        code_prefix: str,
        language: CodeLanguage = CodeLanguage.PYTHON,
    ) -> CodeResult:
        """代码补全。"""
        prompt = self._build_complete_prompt(code_prefix, language)
        completion = self.llm_func(prompt)
        
        self._history.append(AgentStep(
            step_num=len(self._history) + 1,
            thought="代码补全",
            action=CodeAction(CodeActionType.COMPLETE, {"prefix": code_prefix[:50]}),
            result=completion,
        ))
        
        return CodeResult(
            code=completion,
            language=language,
        )
    
    def explain(self, code: str) -> CodeResult:
        """解释代码。"""
        prompt = f"解释以下代码的功能:\n\n```\n{code}\n```"
        explanation = self.llm_func(prompt)
        
        self._history.append(AgentStep(
            step_num=len(self._history) + 1,
            thought="解释代码",
            action=CodeAction(CodeActionType.EXPLAIN),
            result=explanation,
        ))
        
        return CodeResult(
            code=code,
            explanation=explanation,
        )
    
    def refactor(
        self,
        code: str,
        instruction: str,
        language: CodeLanguage = CodeLanguage.PYTHON,
    ) -> CodeResult:
        """重构代码。"""
        prompt = self._build_refactor_prompt(code, instruction, language)
        refactored = self.llm_func(prompt)
        
        self._history.append(AgentStep(
            step_num=len(self._history) + 1,
            thought=f"重构: {instruction}",
            action=CodeAction(CodeActionType.REFACTOR, {"instruction": instruction}),
            result=refactored,
        ))
        
        return CodeResult(
            code=refactored,
            language=language,
            explanation=f"按照'{instruction}'重构后的代码",
        )
    
    def fix(
        self,
        code: str,
        error: str,
        language: CodeLanguage = CodeLanguage.PYTHON,
    ) -> CodeResult:
        """修复代码错误。"""
        prompt = f"""修复以下代码中的错误。

代码:
```{language.value}
{code}
```

错误信息:
{error}

修复后的代码:"""
        
        fixed = self.llm_func(prompt)
        
        self._history.append(AgentStep(
            step_num=len(self._history) + 1,
            thought=f"修复错误: {error[:50]}",
            action=CodeAction(CodeActionType.FIX, {"error": error}),
            result=fixed,
        ))
        
        return CodeResult(
            code=fixed,
            language=language,
            explanation=f"修复了错误: {error}",
        )
    
    def _build_generate_prompt(
        self,
        task: str,
        language: CodeLanguage,
        context: Optional[str],
        references: List[CodeDocument],
    ) -> str:
        """构建生成提示。"""
        parts = [f"用{language.value}实现以下功能: {task}"]
        
        if context:
            parts.append(f"\n上下文:\n{context}")
        
        if references:
            parts.append("\n参考代码:")
            for ref in references[:2]:
                parts.append(f"\n```{ref.language.value}\n{ref.content[:200]}\n```")
        
        parts.append("\n生成的代码:")
        return "\n".join(parts)
    
    def _build_complete_prompt(self, prefix: str, language: CodeLanguage) -> str:
        """构建补全提示。"""
        return f"""补全以下{language.value}代码:

```{language.value}
{prefix}
```

补全:"""
    
    def _build_refactor_prompt(
        self,
        code: str,
        instruction: str,
        language: CodeLanguage,
    ) -> str:
        """构建重构提示。"""
        return f"""按照以下要求重构代码:

要求: {instruction}

原代码:
```{language.value}
{code}
```

重构后:"""
    
    def get_history(self) -> List[AgentStep]:
        """获取历史记录。"""
        return self._history.copy()
    
    def clear_history(self) -> None:
        """清空历史。"""
        self._history.clear()
    
    def __repr__(self) -> str:
        return f"CodeAgent(steps={len(self._history)})"
