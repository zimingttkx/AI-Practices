"""
07-computer-use: Computer Use Agent Module.

This module provides agents capable of interacting with computer interfaces,
including screen understanding, GUI automation, and code generation.

Components:
    - ComputerAgent: Screen understanding and GUI automation
    - CodeAgent: Code generation, debugging, and refactoring

Author: zhangfeng
Version: 1.0.0
"""

from .code_agent import (
    CodeAgent,
    CodeAnalysis,
    CodeAnalyzer,
    CodeBlock,
    CodeDebugger,
    CodeGenerator,
    CodeIssue,
    CodeLanguage,
    CodeRefactorer,
    IssueSeverity,
    RefactorSuggestion,
    create_code_agent,
)
from .computer_agent import (
    ActionPlanner,
    ComputerAgent,
    GUIController,
    KeyboardAction,
    MouseAction,
    ScreenAnalyzer,
    ScreenRegion,
    ScreenState,
    UIElement,
    UIElementType,
    create_computer_agent,
)

__all__ = [
    # Computer Agent
    "ScreenRegion",
    "UIElement",
    "UIElementType",
    "ScreenState",
    "MouseAction",
    "KeyboardAction",
    "ScreenAnalyzer",
    "ActionPlanner",
    "GUIController",
    "ComputerAgent",
    "create_computer_agent",
    # Code Agent
    "CodeLanguage",
    "CodeBlock",
    "CodeAnalysis",
    "CodeIssue",
    "IssueSeverity",
    "RefactorSuggestion",
    "CodeAnalyzer",
    "CodeGenerator",
    "CodeDebugger",
    "CodeRefactorer",
    "CodeAgent",
    "create_code_agent",
]
