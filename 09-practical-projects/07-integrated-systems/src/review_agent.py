"""
代码审查智能体。

自动化代码审查和质量检测。

核心组件:
    - IssueSeverity: 问题严重程度
    - ReviewIssue: 审查问题
    - ReviewResult: 审查结果
    - ReviewAgent: 代码审查智能体
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .code_retriever import CodeDocument, CodeLanguage


class IssueSeverity(Enum):
    """问题严重程度。"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class IssueCategory(Enum):
    """问题类别。"""
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"


@dataclass
class ReviewIssue:
    """审查问题。"""
    message: str
    severity: IssueSeverity
    category: IssueCategory
    line: Optional[int] = None
    column: Optional[int] = None
    suggestion: str = ""
    
    def __repr__(self) -> str:
        loc = f"L{self.line}" if self.line else ""
        return f"ReviewIssue({self.severity.value}, {loc}, '{self.message[:40]}')"


@dataclass
class ReviewResult:
    """审查结果。"""
    issues: List[ReviewIssue] = field(default_factory=list)
    summary: str = ""
    score: float = 100.0
    passed: bool = True
    
    @property
    def num_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)
    
    @property
    def num_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)
    
    def __repr__(self) -> str:
        return f"ReviewResult(score={self.score:.1f}, errors={self.num_errors}, warnings={self.num_warnings})"


class ReviewAgent:
    """代码审查智能体。
    
    执行静态分析和智能审查。
    
    示例:
        >>> agent = ReviewAgent()
        >>> result = agent.review(code, language=CodeLanguage.PYTHON)
        >>> print(result.score)
    """
    
    PYTHON_PATTERNS = [
        (r"except\s*:", IssueSeverity.WARNING, IssueCategory.BUG, "避免使用裸except"),
        (r"eval\s*\(", IssueSeverity.ERROR, IssueCategory.SECURITY, "eval存在安全风险"),
        (r"exec\s*\(", IssueSeverity.ERROR, IssueCategory.SECURITY, "exec存在安全风险"),
        (r"import\s+\*", IssueSeverity.WARNING, IssueCategory.STYLE, "避免使用通配符导入"),
        (r"print\s*\(", IssueSeverity.INFO, IssueCategory.STYLE, "生产代码中避免使用print"),
        (r"TODO|FIXME|XXX|HACK", IssueSeverity.INFO, IssueCategory.MAINTAINABILITY, "存在待处理标记"),
        (r"password\s*=\s*['\"]", IssueSeverity.ERROR, IssueCategory.SECURITY, "硬编码密码"),
        (r"api_key\s*=\s*['\"]", IssueSeverity.ERROR, IssueCategory.SECURITY, "硬编码API密钥"),
    ]
    
    JS_PATTERNS = [
        (r"eval\s*\(", IssueSeverity.ERROR, IssueCategory.SECURITY, "eval存在安全风险"),
        (r"innerHTML\s*=", IssueSeverity.WARNING, IssueCategory.SECURITY, "innerHTML可能导致XSS"),
        (r"var\s+\w+", IssueSeverity.INFO, IssueCategory.STYLE, "建议使用let或const"),
        (r"==(?!=)", IssueSeverity.WARNING, IssueCategory.BUG, "建议使用==="),
        (r"console\.log", IssueSeverity.INFO, IssueCategory.STYLE, "生产代码中移除console.log"),
    ]
    
    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
        strict_mode: bool = False,
    ) -> None:
        self.llm_func = llm_func or self._mock_llm
        self.strict_mode = strict_mode
        
        self._patterns = {
            CodeLanguage.PYTHON: self.PYTHON_PATTERNS,
            CodeLanguage.JAVASCRIPT: self.JS_PATTERNS,
            CodeLanguage.TYPESCRIPT: self.JS_PATTERNS,
        }
    
    def _mock_llm(self, prompt: str) -> str:
        """模拟LLM响应。"""
        return "代码结构清晰，建议添加更多注释。"
    
    def review(
        self,
        code: str,
        language: CodeLanguage = CodeLanguage.PYTHON,
        check_style: bool = True,
        check_security: bool = True,
    ) -> ReviewResult:
        """审查代码。"""
        issues = []
        
        # 静态模式检查
        patterns = self._patterns.get(language, [])
        for pattern, severity, category, message in patterns:
            if category == IssueCategory.STYLE and not check_style:
                continue
            if category == IssueCategory.SECURITY and not check_security:
                continue
            
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line = code[:match.start()].count("\n") + 1
                issues.append(ReviewIssue(
                    message=message,
                    severity=severity,
                    category=category,
                    line=line,
                ))
        
        # 代码复杂度检查
        complexity_issues = self._check_complexity(code, language)
        issues.extend(complexity_issues)
        
        # 计算分数
        score = self._calculate_score(issues)
        passed = score >= 60 and not any(
            i.severity == IssueSeverity.ERROR for i in issues
        )
        
        # 生成摘要
        summary = self._generate_summary(issues, score)
        
        return ReviewResult(
            issues=issues,
            summary=summary,
            score=score,
            passed=passed,
        )
    
    def review_document(self, doc: CodeDocument) -> ReviewResult:
        """审查代码文档。"""
        return self.review(doc.content, doc.language)
    
    def _check_complexity(
        self,
        code: str,
        language: CodeLanguage,
    ) -> List[ReviewIssue]:
        """检查代码复杂度。"""
        issues = []
        lines = code.split("\n")
        
        # 检查函数长度
        if language == CodeLanguage.PYTHON:
            func_pattern = r"^\s*def\s+(\w+)"
            current_func = None
            func_start = 0
            
            for i, line in enumerate(lines):
                match = re.match(func_pattern, line)
                if match:
                    if current_func and i - func_start > 50:
                        issues.append(ReviewIssue(
                            message=f"函数'{current_func}'过长({i - func_start}行)",
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.MAINTAINABILITY,
                            line=func_start + 1,
                        ))
                    current_func = match.group(1)
                    func_start = i
        
        # 检查行长度
        for i, line in enumerate(lines):
            if len(line) > 120:
                issues.append(ReviewIssue(
                    message=f"行过长({len(line)}字符)",
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.STYLE,
                    line=i + 1,
                ))
        
        # 检查嵌套深度
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        if max_indent > 20:
            issues.append(ReviewIssue(
                message="嵌套层级过深",
                severity=IssueSeverity.WARNING,
                category=IssueCategory.MAINTAINABILITY,
            ))
        
        return issues
    
    def _calculate_score(self, issues: List[ReviewIssue]) -> float:
        """计算审查分数。"""
        score = 100.0
        
        for issue in issues:
            if issue.severity == IssueSeverity.ERROR:
                score -= 15
            elif issue.severity == IssueSeverity.WARNING:
                score -= 5
            elif issue.severity == IssueSeverity.INFO:
                score -= 1
        
        return max(0.0, score)
    
    def _generate_summary(self, issues: List[ReviewIssue], score: float) -> str:
        """生成审查摘要。"""
        if not issues:
            return "代码审查通过，未发现问题。"
        
        error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)
        
        parts = [f"审查分数: {score:.1f}/100"]
        if error_count:
            parts.append(f"错误: {error_count}")
        if warning_count:
            parts.append(f"警告: {warning_count}")
        
        return ", ".join(parts)
    
    def suggest_fixes(self, code: str, issues: List[ReviewIssue]) -> Dict[int, str]:
        """建议修复方案。"""
        fixes = {}
        lines = code.split("\n")
        
        for issue in issues:
            if issue.line and issue.line <= len(lines):
                line = lines[issue.line - 1]
                
                # 简单的自动修复建议
                if "裸except" in issue.message:
                    fixes[issue.line] = line.replace("except:", "except Exception:")
                elif "通配符导入" in issue.message:
                    fixes[issue.line] = "# TODO: 替换为具体导入"
        
        return fixes
    
    def __repr__(self) -> str:
        mode = "strict" if self.strict_mode else "normal"
        return f"ReviewAgent(mode={mode})"
