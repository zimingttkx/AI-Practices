"""
Code Agent: Code Generation, Debugging, and Refactoring.

Core Idea:
    This module provides agents capable of understanding, generating,
    analyzing, and refactoring code across multiple programming languages.
    It combines static analysis with LLM-based understanding.

Mathematical Foundation:
    Code as structured sequence:
        P(code | spec) = ∏ P(token_i | token_<i, spec)

    Bug localization as ranking:
        score(line) = P(bug | line, context, error_message)

Design Patterns:
    - Visitor Pattern: AST traversal for code analysis
    - Strategy Pattern: Different generation/analysis strategies
    - Template Method: Common code transformation workflow

References:
    - Codex/GPT-4 for Code: https://arxiv.org/abs/2107.03374
    - CodeBERT: https://arxiv.org/abs/2002.08155
    - SWE-Agent: https://arxiv.org/abs/2405.15793

Author: zhangfeng
Version: 1.0.0
"""

from __future__ import annotations

import ast
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
)

__all__ = [
    "CodeLanguage",
    "CodeBlock",
    "CodeAnalysis",
    "CodeIssue",
    "IssueSeverity",
    "RefactorSuggestion",
    "RefactorType",
    "CodeAnalyzer",
    "CodeGenerator",
    "CodeDebugger",
    "CodeRefactorer",
    "CodeAgent",
    "CodeAgentConfig",
    "create_code_agent",
]

logger = logging.getLogger(__name__)


class CodeLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    BASH = "bash"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class IssueSeverity(str, Enum):
    """Severity level of code issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class RefactorType(str, Enum):
    """Types of code refactoring."""
    RENAME = "rename"
    EXTRACT_FUNCTION = "extract_function"
    EXTRACT_VARIABLE = "extract_variable"
    INLINE = "inline"
    MOVE = "move"
    SIMPLIFY = "simplify"
    OPTIMIZE = "optimize"
    FORMAT = "format"
    ADD_TYPES = "add_types"
    ADD_DOCSTRING = "add_docstring"


@dataclass
class CodeBlock:
    """
    Represents a block of code.

    Attributes:
        code: The source code
        language: Programming language
        file_path: Optional file path
        start_line: Starting line number
        end_line: Ending line number
    """
    code: str
    language: CodeLanguage = CodeLanguage.PYTHON
    file_path: str | None = None
    start_line: int = 1
    end_line: int | None = None

    def __post_init__(self):
        if self.end_line is None:
            self.end_line = self.start_line + self.code.count("\n")

    @property
    def line_count(self) -> int:
        return self.code.count("\n") + 1

    def get_line(self, line_num: int) -> str | None:
        """Get a specific line (1-indexed)."""
        lines = self.code.split("\n")
        idx = line_num - self.start_line
        if 0 <= idx < len(lines):
            return lines[idx]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "language": self.language.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_count": self.line_count,
        }


@dataclass
class CodeIssue:
    """
    Represents a code issue (error, warning, etc.).

    Attributes:
        issue_id: Unique identifier
        severity: Issue severity
        message: Issue description
        line: Line number
        column: Column number
        rule: Rule or check that found the issue
        suggestion: Suggested fix
    """
    issue_id: str = field(default_factory=lambda: f"issue_{uuid.uuid4().hex[:8]}")
    severity: IssueSeverity = IssueSeverity.WARNING
    message: str = ""
    line: int = 1
    column: int = 0
    rule: str = ""
    suggestion: str | None = None
    code_snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "severity": self.severity.value,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "rule": self.rule,
            "suggestion": self.suggestion,
        }


@dataclass
class RefactorSuggestion:
    """
    Represents a refactoring suggestion.

    Attributes:
        refactor_type: Type of refactoring
        description: What the refactoring does
        original_code: Code before refactoring
        refactored_code: Code after refactoring
        confidence: Confidence score (0-1)
    """
    refactor_type: RefactorType
    description: str
    original_code: str
    refactored_code: str
    confidence: float = 0.8
    start_line: int = 1
    end_line: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "refactor_type": self.refactor_type.value,
            "description": self.description,
            "original_code": self.original_code,
            "refactored_code": self.refactored_code,
            "confidence": self.confidence,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class CodeAnalysis:
    """
    Result of code analysis.

    Attributes:
        code_block: The analyzed code
        issues: List of detected issues
        metrics: Code metrics (complexity, etc.)
        suggestions: Improvement suggestions
    """
    code_block: CodeBlock
    issues: list[CodeIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    suggestions: list[RefactorSuggestion] = field(default_factory=list)
    analysis_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_block": self.code_block.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "has_errors": self.has_errors,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


class CodeAnalyzer:
    """
    Analyzes code for issues, metrics, and improvement opportunities.

    Core Idea:
        Combines static analysis (AST parsing) with LLM-based
        semantic understanding for comprehensive code analysis.

    Example:
        >>> analyzer = CodeAnalyzer()
        >>> analysis = analyzer.analyze(code_block)
        >>> print(f"Found {analysis.error_count} errors")
    """

    def __init__(
        self,
        llm_func: Callable[[str], str] | None = None,
        enable_static_analysis: bool = True,
    ) -> None:
        self.llm_func = llm_func or self._mock_llm
        self.enable_static_analysis = enable_static_analysis

    def _mock_llm(self, prompt: str) -> str:
        return "Code looks good with minor style issues."

    def analyze(self, code_block: CodeBlock) -> CodeAnalysis:
        """Analyze a code block."""
        import time
        start_time = time.time()

        issues: list[CodeIssue] = []
        metrics: dict[str, Any] = {}
        suggestions: list[RefactorSuggestion] = []

        # Static analysis for Python
        if code_block.language == CodeLanguage.PYTHON and self.enable_static_analysis:
            static_issues = self._python_static_analysis(code_block.code)
            issues.extend(static_issues)
            metrics = self._calculate_python_metrics(code_block.code)

        # LLM-based analysis
        llm_issues = self._llm_analysis(code_block)
        issues.extend(llm_issues)

        return CodeAnalysis(
            code_block=code_block,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions,
            analysis_time=time.time() - start_time,
        )

    def _python_static_analysis(self, code: str) -> list[CodeIssue]:
        """Perform static analysis on Python code."""
        issues: list[CodeIssue] = []

        # Try to parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(CodeIssue(
                severity=IssueSeverity.ERROR,
                message=f"Syntax error: {e.msg}",
                line=e.lineno or 1,
                column=e.offset or 0,
                rule="syntax",
            ))
            return issues

        # Check for common issues
        issues.extend(self._check_unused_imports(tree, code))
        issues.extend(self._check_naming_conventions(tree))
        issues.extend(self._check_complexity(tree))

        return issues

    def _check_unused_imports(self, tree: ast.AST, code: str) -> list[CodeIssue]:
        """Check for potentially unused imports."""
        issues: list[CodeIssue] = []
        imported_names: list[tuple[str, int]] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imported_names.append((name, node.lineno))

        # Simple check: see if name appears elsewhere in code
        for name, lineno in imported_names:
            # Count occurrences (excluding import line itself)
            pattern = rf'\b{re.escape(name)}\b'
            matches = list(re.finditer(pattern, code))
            if len(matches) <= 1:
                issues.append(CodeIssue(
                    severity=IssueSeverity.WARNING,
                    message=f"'{name}' may be unused",
                    line=lineno,
                    rule="unused-import",
                    suggestion=f"Remove unused import: {name}",
                ))

        return issues

    def _check_naming_conventions(self, tree: ast.AST) -> list[CodeIssue]:
        """Check naming conventions."""
        issues: list[CodeIssue] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    issues.append(CodeIssue(
                        severity=IssueSeverity.WARNING,
                        message=f"Function '{node.name}' should use snake_case",
                        line=node.lineno,
                        rule="naming-convention",
                    ))
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append(CodeIssue(
                        severity=IssueSeverity.WARNING,
                        message=f"Class '{node.name}' should use PascalCase",
                        line=node.lineno,
                        rule="naming-convention",
                    ))

        return issues

    def _check_complexity(self, tree: ast.AST) -> list[CodeIssue]:
        """Check code complexity."""
        issues: list[CodeIssue] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count branches (if, for, while, try, etc.)
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                        complexity += 1

                if complexity > 10:
                    issues.append(CodeIssue(
                        severity=IssueSeverity.WARNING,
                        message=f"Function '{node.name}' has high complexity ({complexity})",
                        line=node.lineno,
                        rule="complexity",
                        suggestion="Consider breaking this function into smaller parts",
                    ))

        return issues

    def _calculate_python_metrics(self, code: str) -> dict[str, Any]:
        """Calculate code metrics."""
        lines = code.split("\n")

        # Lines of code
        loc = len(lines)
        sloc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

        # Comment lines
        comment_lines = len([l for l in lines if l.strip().startswith("#")])

        # Function/class counts
        try:
            tree = ast.parse(code)
            functions = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
            classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        except SyntaxError:
            functions = 0
            classes = 0

        return {
            "lines_of_code": loc,
            "source_lines": sloc,
            "comment_lines": comment_lines,
            "function_count": functions,
            "class_count": classes,
            "comment_ratio": comment_lines / max(loc, 1),
        }

    def _llm_analysis(self, code_block: CodeBlock) -> list[CodeIssue]:
        """Use LLM for semantic analysis."""
        # In real implementation, would call LLM
        return []


class CodeGenerator:
    """
    Generates code from natural language descriptions.

    Core Idea:
        Uses LLM to generate code based on specifications,
        with support for context, examples, and constraints.

    Example:
        >>> generator = CodeGenerator()
        >>> code = generator.generate("Create a function to sort a list", language=CodeLanguage.PYTHON)
    """

    def __init__(
        self,
        llm_func: Callable[[str], str] | None = None,
        default_language: CodeLanguage = CodeLanguage.PYTHON,
    ) -> None:
        self.llm_func = llm_func or self._mock_llm
        self.default_language = default_language
        self._generation_history: list[dict[str, Any]] = []

    def _mock_llm(self, prompt: str) -> str:
        """Mock LLM for testing."""
        if "sort" in prompt.lower():
            return """def sort_list(items: list) -> list:
    \"\"\"Sort a list in ascending order.\"\"\"
    return sorted(items)"""
        elif "hello" in prompt.lower():
            return """def hello_world():
    \"\"\"Print Hello World.\"\"\"
    print("Hello, World!")"""
        return """def example():
    \"\"\"Example function.\"\"\"
    pass"""

    def generate(
        self,
        description: str,
        language: CodeLanguage | None = None,
        context: str | None = None,
        examples: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> CodeBlock:
        """
        Generate code from description.

        Args:
            description: What the code should do
            language: Target language
            context: Additional context (existing code, etc.)
            examples: Example inputs/outputs
            constraints: Constraints to follow

        Returns:
            Generated code block
        """
        language = language or self.default_language

        prompt = self._build_prompt(description, language, context, examples, constraints)
        response = self.llm_func(prompt)
        code = self._extract_code(response)

        code_block = CodeBlock(code=code, language=language)

        self._generation_history.append({
            "description": description,
            "language": language.value,
            "code": code,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return code_block

    def _build_prompt(
        self,
        description: str,
        language: CodeLanguage,
        context: str | None,
        examples: list[str] | None,
        constraints: list[str] | None,
    ) -> str:
        """Build prompt for code generation."""
        prompt = f"Generate {language.value} code that: {description}\n"

        if context:
            prompt += f"\nContext:\n{context}\n"

        if examples:
            prompt += "\nExamples:\n"
            for ex in examples:
                prompt += f"- {ex}\n"

        if constraints:
            prompt += "\nConstraints:\n"
            for c in constraints:
                prompt += f"- {c}\n"

        prompt += "\nProvide only the code, no explanations."
        return prompt

    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        # Remove markdown code blocks if present
        code = response.strip()

        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first line (```python) and last line (```)
            lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            code = "\n".join(lines)

        return code.strip()

    def get_history(self) -> list[dict[str, Any]]:
        return self._generation_history.copy()


class CodeDebugger:
    """
    Debugs code by analyzing errors and suggesting fixes.

    Core Idea:
        Combines error message analysis with code context
        to identify bug locations and suggest fixes.

    Example:
        >>> debugger = CodeDebugger()
        >>> fix = debugger.debug(code_block, "NameError: name 'x' is not defined")
    """

    def __init__(
        self,
        llm_func: Callable[[str], str] | None = None,
    ) -> None:
        self.llm_func = llm_func or self._mock_llm

    def _mock_llm(self, prompt: str) -> str:
        return "The variable needs to be defined before use."

    def debug(
        self,
        code_block: CodeBlock,
        error_message: str,
        stack_trace: str | None = None,
    ) -> dict[str, Any]:
        """
        Debug code given an error.

        Args:
            code_block: The problematic code
            error_message: Error message
            stack_trace: Optional stack trace

        Returns:
            Debug result with analysis and suggested fix
        """
        # Parse error message
        error_info = self._parse_error(error_message)

        # Analyze code around error location
        analysis = self._analyze_error_context(code_block, error_info)

        # Generate fix suggestion
        fix = self._suggest_fix(code_block, error_info, analysis)

        return {
            "error_type": error_info.get("type", "Unknown"),
            "error_message": error_message,
            "probable_line": error_info.get("line"),
            "analysis": analysis,
            "suggested_fix": fix,
            "fixed_code": self._apply_fix(code_block.code, fix) if fix else None,
        }

    def _parse_error(self, error_message: str) -> dict[str, Any]:
        """Parse error message to extract information."""
        info: dict[str, Any] = {}

        # Common Python error patterns
        patterns = [
            (r"(\w+Error): (.+)", ["type", "message"]),
            (r"line (\d+)", ["line"]),
            (r"File \"(.+)\", line (\d+)", ["file", "line"]),
        ]

        for pattern, keys in patterns:
            match = re.search(pattern, error_message)
            if match:
                for i, key in enumerate(keys):
                    if i < len(match.groups()):
                        value = match.group(i + 1)
                        info[key] = int(value) if key == "line" else value

        return info

    def _analyze_error_context(
        self,
        code_block: CodeBlock,
        error_info: dict[str, Any],
    ) -> str:
        """Analyze code context around error."""
        line_num = error_info.get("line", 1)
        error_type = error_info.get("type", "Error")

        lines = code_block.code.split("\n")
        context_start = max(0, line_num - 3)
        context_end = min(len(lines), line_num + 2)
        context = "\n".join(f"{i+1}: {lines[i]}" for i in range(context_start, context_end))

        if error_type == "NameError":
            return f"Undefined variable at line {line_num}. Check variable definitions before this line."
        elif error_type == "TypeError":
            return f"Type mismatch at line {line_num}. Check argument types."
        elif error_type == "IndexError":
            return f"Index out of bounds at line {line_num}. Check list/array bounds."
        elif error_type == "SyntaxError":
            return f"Syntax error at line {line_num}. Check brackets, colons, and indentation."

        return f"Error at line {line_num}. Context:\n{context}"

    def _suggest_fix(
        self,
        code_block: CodeBlock,
        error_info: dict[str, Any],
        analysis: str,
    ) -> str | None:
        """Suggest a fix for the error."""
        error_type = error_info.get("type", "")

        if error_type == "NameError":
            # Extract undefined name
            match = re.search(r"name '(\w+)' is not defined", error_info.get("message", ""))
            if match:
                name = match.group(1)
                return f"Define '{name}' before using it, or check for typos."

        elif error_type == "IndentationError":
            return "Fix indentation to use consistent spaces (4 spaces recommended)."

        elif error_type == "SyntaxError":
            return "Check for missing colons, brackets, or parentheses."

        return None

    def _apply_fix(self, code: str, fix_description: str) -> str:
        """Apply fix to code (simplified)."""
        # In real implementation, would use LLM to apply fix
        return code


class CodeRefactorer:
    """
    Suggests and applies code refactoring.

    Core Idea:
        Analyzes code structure to identify refactoring
        opportunities and generates improved versions.

    Example:
        >>> refactorer = CodeRefactorer()
        >>> suggestions = refactorer.suggest(code_block)
    """

    def __init__(
        self,
        llm_func: Callable[[str], str] | None = None,
    ) -> None:
        self.llm_func = llm_func or self._mock_llm

    def _mock_llm(self, prompt: str) -> str:
        return "Consider extracting repeated logic into a separate function."

    def suggest(self, code_block: CodeBlock) -> list[RefactorSuggestion]:
        """Suggest refactorings for code."""
        suggestions: list[RefactorSuggestion] = []

        if code_block.language == CodeLanguage.PYTHON:
            suggestions.extend(self._python_suggestions(code_block))

        return suggestions

    def _python_suggestions(self, code_block: CodeBlock) -> list[RefactorSuggestion]:
        """Generate Python-specific refactoring suggestions."""
        suggestions: list[RefactorSuggestion] = []

        try:
            tree = ast.parse(code_block.code)
        except SyntaxError:
            return suggestions

        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count statements
                stmt_count = len(node.body)
                if stmt_count > 20:
                    suggestions.append(RefactorSuggestion(
                        refactor_type=RefactorType.EXTRACT_FUNCTION,
                        description=f"Function '{node.name}' is too long ({stmt_count} statements). Consider extracting parts.",
                        original_code=ast.unparse(node) if hasattr(ast, 'unparse') else "",
                        refactored_code="",
                        confidence=0.7,
                        start_line=node.lineno,
                    ))

                # Check for missing docstring
                if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)):
                    suggestions.append(RefactorSuggestion(
                        refactor_type=RefactorType.ADD_DOCSTRING,
                        description=f"Function '{node.name}' is missing a docstring.",
                        original_code=f"def {node.name}(...):",
                        refactored_code=f'def {node.name}(...):\n    """Description of {node.name}."""',
                        confidence=0.9,
                        start_line=node.lineno,
                    ))

        # Check for repeated code patterns
        suggestions.extend(self._find_duplicate_code(tree))

        return suggestions

    def _find_duplicate_code(self, tree: ast.AST) -> list[RefactorSuggestion]:
        """Find duplicate code patterns."""
        suggestions: list[RefactorSuggestion] = []

        # Simplified: check for similar function bodies
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        for i, f1 in enumerate(functions):
            for f2 in functions[i + 1:]:
                # Very simplified similarity check
                if len(f1.body) == len(f2.body) and len(f1.body) > 3:
                    suggestions.append(RefactorSuggestion(
                        refactor_type=RefactorType.EXTRACT_FUNCTION,
                        description=f"Functions '{f1.name}' and '{f2.name}' have similar structure. Consider extracting common logic.",
                        original_code="",
                        refactored_code="",
                        confidence=0.5,
                    ))

        return suggestions

    def apply(
        self,
        code_block: CodeBlock,
        suggestion: RefactorSuggestion,
    ) -> CodeBlock:
        """Apply a refactoring suggestion."""
        # In real implementation, would apply the actual refactoring
        if suggestion.refactored_code:
            new_code = code_block.code.replace(
                suggestion.original_code,
                suggestion.refactored_code
            )
            return CodeBlock(code=new_code, language=code_block.language)
        return code_block


@dataclass
class CodeAgentConfig:
    """Configuration for CodeAgent."""
    name: str = "CodeAgent"
    default_language: CodeLanguage = CodeLanguage.PYTHON
    enable_static_analysis: bool = True
    max_generation_tokens: int = 2000
    verbose: bool = False


class CodeAgent:
    """
    Agent for code generation, analysis, debugging, and refactoring.

    Core Idea:
        Provides a unified interface for all code-related tasks,
        combining multiple specialized components.

    Example:
        >>> agent = CodeAgent()
        >>> code = agent.generate("Create a function to calculate factorial")
        >>> analysis = agent.analyze(code)
        >>> if analysis.has_errors:
        ...     fix = agent.debug(code, "Error message")
    """

    def __init__(
        self,
        config: CodeAgentConfig | None = None,
        llm_func: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config or CodeAgentConfig()
        self.llm_func = llm_func or self._mock_llm

        self.analyzer = CodeAnalyzer(
            llm_func=self.llm_func,
            enable_static_analysis=self.config.enable_static_analysis,
        )
        self.generator = CodeGenerator(
            llm_func=self.llm_func,
            default_language=self.config.default_language,
        )
        self.debugger = CodeDebugger(llm_func=self.llm_func)
        self.refactorer = CodeRefactorer(llm_func=self.llm_func)

        self._task_history: list[dict[str, Any]] = []
        logger.info(f"CodeAgent '{self.config.name}' initialized")

    def _mock_llm(self, prompt: str) -> str:
        if "generate" in prompt.lower() or "create" in prompt.lower():
            return "def example():\n    pass"
        return "Code analysis complete."

    def generate(
        self,
        description: str,
        language: CodeLanguage | None = None,
        **kwargs: Any,
    ) -> CodeBlock:
        """Generate code from description."""
        code = self.generator.generate(
            description,
            language=language or self.config.default_language,
            **kwargs,
        )
        self._record_task("generate", description=description, result=code.code)
        return code

    def analyze(self, code: str | CodeBlock) -> CodeAnalysis:
        """Analyze code for issues and metrics."""
        if isinstance(code, str):
            code = CodeBlock(code=code, language=self.config.default_language)

        analysis = self.analyzer.analyze(code)
        self._record_task("analyze", code=code.code[:100], issues=len(analysis.issues))
        return analysis

    def debug(
        self,
        code: str | CodeBlock,
        error_message: str,
        stack_trace: str | None = None,
    ) -> dict[str, Any]:
        """Debug code given an error."""
        if isinstance(code, str):
            code = CodeBlock(code=code, language=self.config.default_language)

        result = self.debugger.debug(code, error_message, stack_trace)
        self._record_task("debug", error=error_message)
        return result

    def refactor(
        self,
        code: str | CodeBlock,
    ) -> tuple[list[RefactorSuggestion], CodeBlock | None]:
        """Get refactoring suggestions."""
        if isinstance(code, str):
            code = CodeBlock(code=code, language=self.config.default_language)

        suggestions = self.refactorer.suggest(code)

        # Apply first high-confidence suggestion if available
        improved_code = None
        for sugg in suggestions:
            if sugg.confidence >= 0.8:
                improved_code = self.refactorer.apply(code, sugg)
                break

        self._record_task("refactor", suggestions=len(suggestions))
        return suggestions, improved_code

    def explain(self, code: str | CodeBlock) -> str:
        """Explain what code does."""
        code_str = code if isinstance(code, str) else code.code

        prompt = f"Explain what this code does:\n\n{code_str}"
        explanation = self.llm_func(prompt)
        self._record_task("explain", code_preview=code_str[:50])
        return explanation

    def improve(self, code: str | CodeBlock, goal: str) -> CodeBlock:
        """Improve code based on a goal."""
        if isinstance(code, str):
            code = CodeBlock(code=code, language=self.config.default_language)

        prompt = f"""Improve this code to: {goal}

Current code:
{code.code}

Provide the improved code only."""

        improved = self.llm_func(prompt)
        result = CodeBlock(code=improved, language=code.language)
        self._record_task("improve", goal=goal)
        return result

    def _record_task(self, task_type: str, **kwargs: Any) -> None:
        """Record task in history."""
        self._task_history.append({
            "type": task_type,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        })

    def get_history(self) -> list[dict[str, Any]]:
        return self._task_history.copy()

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "default_language": self.config.default_language.value,
            "tasks_completed": len(self._task_history),
        }

    def __repr__(self) -> str:
        return f"CodeAgent(name='{self.config.name}')"


def create_code_agent(
    name: str = "CodeAgent",
    default_language: CodeLanguage = CodeLanguage.PYTHON,
    llm_func: Callable[[str], str] | None = None,
    **config_kwargs: Any,
) -> CodeAgent:
    """
    Factory function to create a CodeAgent.

    Args:
        name: Agent name
        default_language: Default programming language
        llm_func: LLM function for generation
        **config_kwargs: Additional config parameters

    Returns:
        Configured CodeAgent instance
    """
    config = CodeAgentConfig(name=name, default_language=default_language, **config_kwargs)
    return CodeAgent(config=config, llm_func=llm_func)
