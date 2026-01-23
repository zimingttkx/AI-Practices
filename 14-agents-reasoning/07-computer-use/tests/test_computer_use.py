"""
Unit tests for the 07-computer-use module.

Tests cover:
- Computer Agent: Screen analysis, action planning, GUI control
- Code Agent: Code generation, analysis, debugging, refactoring

Author: zhangfeng
"""

import pytest
import asyncio
import sys
import os

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio(loop_scope="function")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from computer_agent import (
    ScreenRegion,
    UIElement,
    UIElementType,
    ScreenState,
    MouseAction,
    KeyboardAction,
    MouseButton,
    ActionStatus,
    ActionResult,
    ScreenAnalyzer,
    ActionPlanner,
    GUIController,
    ComputerAgent,
    ComputerAgentConfig,
    MockVisionModel,
    create_computer_agent,
)

from code_agent import (
    CodeLanguage,
    CodeBlock,
    CodeAnalysis,
    CodeIssue,
    IssueSeverity,
    RefactorSuggestion,
    RefactorType,
    CodeAnalyzer,
    CodeGenerator,
    CodeDebugger,
    CodeRefactorer,
    CodeAgent,
    CodeAgentConfig,
    create_code_agent,
)


# =============================================================================
# Screen Region Tests
# =============================================================================

class TestScreenRegion:
    """Tests for ScreenRegion."""

    def test_creation(self):
        region = ScreenRegion(x=100, y=200, width=50, height=30)
        assert region.x == 100
        assert region.y == 200
        assert region.width == 50
        assert region.height == 30

    def test_center(self):
        region = ScreenRegion(x=100, y=100, width=100, height=50)
        assert region.center == (150, 125)

    def test_right_bottom(self):
        region = ScreenRegion(x=10, y=20, width=30, height=40)
        assert region.right == 40
        assert region.bottom == 60

    def test_contains(self):
        region = ScreenRegion(x=0, y=0, width=100, height=100)
        assert region.contains(50, 50) is True
        assert region.contains(0, 0) is True
        assert region.contains(99, 99) is True
        assert region.contains(100, 100) is False
        assert region.contains(-1, 50) is False

    def test_overlaps(self):
        r1 = ScreenRegion(x=0, y=0, width=100, height=100)
        r2 = ScreenRegion(x=50, y=50, width=100, height=100)
        r3 = ScreenRegion(x=200, y=200, width=50, height=50)
        assert r1.overlaps(r2) is True
        assert r1.overlaps(r3) is False

    def test_to_dict(self):
        region = ScreenRegion(x=10, y=20, width=30, height=40)
        d = region.to_dict()
        assert d == {"x": 10, "y": 20, "width": 30, "height": 40}


# =============================================================================
# UI Element Tests
# =============================================================================

class TestUIElement:
    """Tests for UIElement."""

    def test_creation(self):
        elem = UIElement(
            element_type=UIElementType.BUTTON,
            region=ScreenRegion(100, 100, 80, 30),
            text="Submit",
        )
        assert elem.element_type == UIElementType.BUTTON
        assert elem.text == "Submit"
        assert elem.is_enabled is True
        assert elem.element_id.startswith("elem_")

    def test_center(self):
        elem = UIElement(
            region=ScreenRegion(100, 100, 100, 50),
        )
        assert elem.center == (150, 125)

    def test_matches_text(self):
        elem = UIElement(text="Click Here")
        assert elem.matches_text("click") is True
        assert elem.matches_text("CLICK") is True
        assert elem.matches_text("click", case_sensitive=True) is False
        assert elem.matches_text("notfound") is False

    def test_to_dict(self):
        elem = UIElement(
            element_type=UIElementType.TEXT_FIELD,
            text="Input",
        )
        d = elem.to_dict()
        assert d["element_type"] == "text_field"
        assert d["text"] == "Input"
        assert "element_id" in d


# =============================================================================
# Screen State Tests
# =============================================================================

class TestScreenState:
    """Tests for ScreenState."""

    def test_creation(self):
        state = ScreenState(
            active_window="Test Window",
            mouse_position=(500, 300),
        )
        assert state.active_window == "Test Window"
        assert state.mouse_position == (500, 300)
        assert len(state.elements) == 0

    def test_find_element_by_text(self):
        btn = UIElement(element_type=UIElementType.BUTTON, text="OK")
        txt = UIElement(element_type=UIElementType.TEXT_FIELD, text="Name")
        state = ScreenState(elements=[btn, txt])
        
        found = state.find_element_by_text("OK")
        assert found is not None
        assert found.text == "OK"
        
        found = state.find_element_by_text("OK", UIElementType.TEXT_FIELD)
        assert found is None

    def test_find_elements_by_type(self):
        elements = [
            UIElement(element_type=UIElementType.BUTTON, text="A"),
            UIElement(element_type=UIElementType.BUTTON, text="B"),
            UIElement(element_type=UIElementType.TEXT_FIELD, text="C"),
        ]
        state = ScreenState(elements=elements)
        
        buttons = state.find_elements_by_type(UIElementType.BUTTON)
        assert len(buttons) == 2

    def test_get_element_at(self):
        elem = UIElement(region=ScreenRegion(10, 10, 50, 50))
        state = ScreenState(elements=[elem])
        
        found = state.get_element_at(25, 25)
        assert found is not None
        
        found = state.get_element_at(100, 100)
        assert found is None


# =============================================================================
# Mouse/Keyboard Action Tests
# =============================================================================

class TestMouseAction:
    """Tests for MouseAction."""

    def test_creation(self):
        action = MouseAction(
            action_type="click",
            x=100,
            y=200,
            button=MouseButton.LEFT,
        )
        assert action.action_type == "click"
        assert action.x == 100
        assert action.y == 200

    def test_to_dict(self):
        action = MouseAction(action_type="double_click", x=50, y=50)
        d = action.to_dict()
        assert d["action_type"] == "double_click"
        assert d["x"] == 50


class TestKeyboardAction:
    """Tests for KeyboardAction."""

    def test_type_action(self):
        action = KeyboardAction(action_type="type", text="Hello World")
        assert action.text == "Hello World"

    def test_hotkey_action(self):
        action = KeyboardAction(
            action_type="hotkey",
            key="c",
            modifiers=["ctrl"],
        )
        assert action.key == "c"
        assert "ctrl" in action.modifiers


# =============================================================================
# Screen Analyzer Tests
# =============================================================================

class TestScreenAnalyzer:
    """Tests for ScreenAnalyzer."""

    def test_creation(self):
        analyzer = ScreenAnalyzer()
        assert analyzer.vision_model is not None

    def test_with_mock_vision(self):
        mock = MockVisionModel(responses=["Found buttons and text fields"])
        analyzer = ScreenAnalyzer(vision_model=mock)
        assert analyzer.vision_model == mock

    @pytest.mark.asyncio
    async def test_capture_screenshot(self):
        analyzer = ScreenAnalyzer()
        screenshot = await analyzer.capture_screenshot()
        assert isinstance(screenshot, str)
        assert len(screenshot) > 0

    @pytest.mark.asyncio
    async def test_capture_and_analyze(self):
        analyzer = ScreenAnalyzer()
        state = await analyzer.capture_and_analyze()
        assert isinstance(state, ScreenState)
        assert len(state.elements) > 0

    def test_get_last_state(self):
        analyzer = ScreenAnalyzer()
        assert analyzer.get_last_state() is None


# =============================================================================
# Action Planner Tests
# =============================================================================

class TestActionPlanner:
    """Tests for ActionPlanner."""

    def test_creation(self):
        planner = ActionPlanner()
        assert planner.llm_func is not None

    @pytest.mark.asyncio
    async def test_plan_click(self):
        planner = ActionPlanner()
        state = ScreenState(elements=[
            UIElement(
                element_type=UIElementType.BUTTON,
                region=ScreenRegion(100, 100, 80, 30),
                text="Submit",
            )
        ])
        
        actions = await planner.plan("Click the Submit button", state)
        assert len(actions) >= 1
        assert isinstance(actions[0], MouseAction)

    @pytest.mark.asyncio
    async def test_plan_type(self):
        planner = ActionPlanner()
        state = ScreenState(elements=[
            UIElement(
                element_type=UIElementType.TEXT_FIELD,
                region=ScreenRegion(100, 100, 200, 30),
                text="",
            )
        ])
        
        actions = await planner.plan("Type 'Hello' in the field", state)
        assert len(actions) >= 1

    def test_get_action_history(self):
        planner = ActionPlanner()
        history = planner.get_action_history()
        assert isinstance(history, list)


# =============================================================================
# GUI Controller Tests
# =============================================================================

class TestGUIController:
    """Tests for GUIController."""

    def test_creation(self):
        controller = GUIController(simulate=True)
        assert controller.simulate is True

    @pytest.mark.asyncio
    async def test_execute_mouse_action(self):
        controller = GUIController(simulate=True, action_delay=0.01)
        action = MouseAction(action_type="click", x=100, y=100)
        
        result = await controller.execute(action)
        assert result.status == ActionStatus.SUCCESS
        assert "Simulated" in result.output

    @pytest.mark.asyncio
    async def test_execute_keyboard_action(self):
        controller = GUIController(simulate=True, action_delay=0.01)
        action = KeyboardAction(action_type="type", text="Hello")
        
        result = await controller.execute(action)
        assert result.status == ActionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_sequence(self):
        controller = GUIController(simulate=True, action_delay=0.01)
        actions = [
            MouseAction(action_type="click", x=100, y=100),
            KeyboardAction(action_type="type", text="Test"),
        ]
        
        results = await controller.execute_sequence(actions)
        assert len(results) == 2
        assert all(r.is_success for r in results)

    def test_get_execution_history(self):
        controller = GUIController()
        history = controller.get_execution_history()
        assert isinstance(history, list)


# =============================================================================
# Computer Agent Tests
# =============================================================================

class TestComputerAgent:
    """Tests for ComputerAgent."""

    def test_creation(self):
        agent = ComputerAgent()
        assert agent.config.name == "ComputerAgent"
        assert agent.config.simulate is True

    def test_with_config(self):
        config = ComputerAgentConfig(
            name="TestAgent",
            simulate=True,
            max_steps=10,
        )
        agent = ComputerAgent(config=config)
        assert agent.config.name == "TestAgent"
        assert agent.config.max_steps == 10

    @pytest.mark.asyncio
    async def test_execute_task(self):
        agent = ComputerAgent()
        result = await agent.execute_task("Click a button", max_steps=2)
        assert result["status"] in ["completed", "failed"]
        assert "steps" in result

    @pytest.mark.asyncio
    async def test_click_element(self):
        agent = ComputerAgent()
        result = await agent.click_element("OK")
        assert isinstance(result, ActionResult)

    @pytest.mark.asyncio
    async def test_type_text(self):
        agent = ComputerAgent()
        result = await agent.type_text("Hello World")
        assert result.is_success

    @pytest.mark.asyncio
    async def test_press_key(self):
        agent = ComputerAgent()
        result = await agent.press_key("Enter")
        assert result.is_success

    def test_get_status(self):
        agent = ComputerAgent()
        status = agent.get_status()
        assert "name" in status
        assert "is_running" in status

    def test_factory_function(self):
        agent = create_computer_agent(
            name="MyAgent",
            simulate=True,
            max_steps=20,
        )
        assert agent.config.name == "MyAgent"
        assert agent.config.max_steps == 20


# =============================================================================
# Code Block Tests
# =============================================================================

class TestCodeBlock:
    """Tests for CodeBlock."""

    def test_creation(self):
        code = CodeBlock(
            code="def hello():\n    print('Hello')",
            language=CodeLanguage.PYTHON,
        )
        assert code.language == CodeLanguage.PYTHON
        assert code.line_count == 2

    def test_get_line(self):
        code = CodeBlock(code="line1\nline2\nline3")
        assert code.get_line(1) == "line1"
        assert code.get_line(2) == "line2"
        assert code.get_line(10) is None

    def test_to_dict(self):
        code = CodeBlock(code="x = 1")
        d = code.to_dict()
        assert d["code"] == "x = 1"
        assert d["language"] == "python"


# =============================================================================
# Code Issue Tests
# =============================================================================

class TestCodeIssue:
    """Tests for CodeIssue."""

    def test_creation(self):
        issue = CodeIssue(
            severity=IssueSeverity.ERROR,
            message="Syntax error",
            line=5,
        )
        assert issue.severity == IssueSeverity.ERROR
        assert issue.line == 5

    def test_to_dict(self):
        issue = CodeIssue(
            severity=IssueSeverity.WARNING,
            message="Unused import",
            line=1,
            rule="unused-import",
        )
        d = issue.to_dict()
        assert d["severity"] == "warning"
        assert d["rule"] == "unused-import"


# =============================================================================
# Code Analysis Tests
# =============================================================================

class TestCodeAnalysis:
    """Tests for CodeAnalysis."""

    def test_creation(self):
        code = CodeBlock(code="x = 1")
        analysis = CodeAnalysis(code_block=code)
        assert analysis.has_errors is False
        assert analysis.error_count == 0

    def test_with_issues(self):
        code = CodeBlock(code="x = 1")
        issues = [
            CodeIssue(severity=IssueSeverity.ERROR, message="Error"),
            CodeIssue(severity=IssueSeverity.WARNING, message="Warning"),
        ]
        analysis = CodeAnalysis(code_block=code, issues=issues)
        assert analysis.has_errors is True
        assert analysis.error_count == 1
        assert analysis.warning_count == 1


# =============================================================================
# Code Analyzer Tests
# =============================================================================

class TestCodeAnalyzer:
    """Tests for CodeAnalyzer."""

    def test_creation(self):
        analyzer = CodeAnalyzer()
        assert analyzer.enable_static_analysis is True

    def test_analyze_valid_code(self):
        analyzer = CodeAnalyzer()
        code = CodeBlock(code="def hello():\n    print('Hello')")
        analysis = analyzer.analyze(code)
        assert analysis.has_errors is False

    def test_analyze_syntax_error(self):
        analyzer = CodeAnalyzer()
        code = CodeBlock(code="def hello(\n    print('Hello')")
        analysis = analyzer.analyze(code)
        assert analysis.has_errors is True

    def test_naming_convention_check(self):
        analyzer = CodeAnalyzer()
        code = CodeBlock(code="def BadName():\n    pass")
        analysis = analyzer.analyze(code)
        # Should warn about naming convention
        naming_issues = [i for i in analysis.issues if i.rule == "naming-convention"]
        assert len(naming_issues) > 0

    def test_metrics_calculation(self):
        analyzer = CodeAnalyzer()
        code = CodeBlock(code="# Comment\ndef func():\n    pass\n\nclass MyClass:\n    pass")
        analysis = analyzer.analyze(code)
        assert "lines_of_code" in analysis.metrics
        assert "function_count" in analysis.metrics


# =============================================================================
# Code Generator Tests
# =============================================================================

class TestCodeGenerator:
    """Tests for CodeGenerator."""

    def test_creation(self):
        generator = CodeGenerator()
        assert generator.default_language == CodeLanguage.PYTHON

    def test_generate_sort(self):
        generator = CodeGenerator()
        code = generator.generate("Create a function to sort a list")
        assert "def" in code.code
        assert "sort" in code.code.lower()

    def test_generate_with_language(self):
        generator = CodeGenerator()
        code = generator.generate(
            "Create a hello function",
            language=CodeLanguage.PYTHON,
        )
        assert code.language == CodeLanguage.PYTHON

    def test_extract_code_from_markdown(self):
        generator = CodeGenerator()
        response = "```python\ndef test():\n    pass\n```"
        code = generator._extract_code(response)
        assert code == "def test():\n    pass"

    def test_get_history(self):
        generator = CodeGenerator()
        generator.generate("Create a function")
        history = generator.get_history()
        assert len(history) == 1


# =============================================================================
# Code Debugger Tests
# =============================================================================

class TestCodeDebugger:
    """Tests for CodeDebugger."""

    def test_creation(self):
        debugger = CodeDebugger()
        assert debugger.llm_func is not None

    def test_debug_name_error(self):
        debugger = CodeDebugger()
        code = CodeBlock(code="print(x)")
        result = debugger.debug(code, "NameError: name 'x' is not defined")
        assert result["error_type"] == "NameError"
        assert result["analysis"] is not None

    def test_debug_syntax_error(self):
        debugger = CodeDebugger()
        code = CodeBlock(code="def func(\n    pass")
        result = debugger.debug(code, "SyntaxError: unexpected EOF, line 2")
        assert "suggested_fix" in result

    def test_parse_error_message(self):
        debugger = CodeDebugger()
        info = debugger._parse_error("TypeError: unsupported operand type")
        assert info["type"] == "TypeError"


# =============================================================================
# Code Refactorer Tests
# =============================================================================

class TestCodeRefactorer:
    """Tests for CodeRefactorer."""

    def test_creation(self):
        refactorer = CodeRefactorer()
        assert refactorer.llm_func is not None

    def test_suggest_missing_docstring(self):
        refactorer = CodeRefactorer()
        code = CodeBlock(code="def my_function():\n    pass")
        suggestions = refactorer.suggest(code)
        docstring_suggestions = [s for s in suggestions if s.refactor_type == RefactorType.ADD_DOCSTRING]
        assert len(docstring_suggestions) > 0

    def test_suggest_long_function(self):
        refactorer = CodeRefactorer()
        # Create a long function
        long_func = "def long_func():\n" + "    x = 1\n" * 25
        code = CodeBlock(code=long_func)
        suggestions = refactorer.suggest(code)
        extract_suggestions = [s for s in suggestions if s.refactor_type == RefactorType.EXTRACT_FUNCTION]
        assert len(extract_suggestions) > 0

    def test_apply_refactoring(self):
        refactorer = CodeRefactorer()
        code = CodeBlock(code="def old_name():\n    pass")
        suggestion = RefactorSuggestion(
            refactor_type=RefactorType.RENAME,
            description="Rename function",
            original_code="def old_name():",
            refactored_code="def new_name():",
        )
        new_code = refactorer.apply(code, suggestion)
        assert "new_name" in new_code.code


# =============================================================================
# Code Agent Tests
# =============================================================================

class TestCodeAgent:
    """Tests for CodeAgent."""

    def test_creation(self):
        agent = CodeAgent()
        assert agent.config.name == "CodeAgent"

    def test_with_config(self):
        config = CodeAgentConfig(
            name="MyCodeAgent",
            default_language=CodeLanguage.PYTHON,
        )
        agent = CodeAgent(config=config)
        assert agent.config.name == "MyCodeAgent"

    def test_generate(self):
        agent = CodeAgent()
        code = agent.generate("Create a hello world function")
        assert isinstance(code, CodeBlock)

    def test_analyze(self):
        agent = CodeAgent()
        code = "def test():\n    pass"
        analysis = agent.analyze(code)
        assert isinstance(analysis, CodeAnalysis)

    def test_analyze_with_code_block(self):
        agent = CodeAgent()
        code = CodeBlock(code="x = 1", language=CodeLanguage.PYTHON)
        analysis = agent.analyze(code)
        assert analysis.code_block == code

    def test_debug(self):
        agent = CodeAgent()
        result = agent.debug("print(x)", "NameError: name 'x' is not defined")
        assert "error_type" in result

    def test_refactor(self):
        agent = CodeAgent()
        code = "def func():\n    pass"
        suggestions, improved = agent.refactor(code)
        assert isinstance(suggestions, list)

    def test_explain(self):
        agent = CodeAgent()
        explanation = agent.explain("x = 1 + 2")
        assert isinstance(explanation, str)

    def test_improve(self):
        agent = CodeAgent()
        code = "x = 1"
        improved = agent.improve(code, "add type hints")
        assert isinstance(improved, CodeBlock)

    def test_get_history(self):
        agent = CodeAgent()
        agent.generate("Create a function")
        history = agent.get_history()
        assert len(history) >= 1

    def test_get_status(self):
        agent = CodeAgent()
        status = agent.get_status()
        assert "name" in status
        assert "default_language" in status

    def test_factory_function(self):
        agent = create_code_agent(
            name="TestAgent",
            default_language=CodeLanguage.JAVASCRIPT,
        )
        assert agent.config.name == "TestAgent"
        assert agent.config.default_language == CodeLanguage.JAVASCRIPT


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_computer_agent_workflow(self):
        """Test complete computer agent workflow."""
        agent = create_computer_agent(name="IntegrationTest", simulate=True)
        
        # Execute a simple task
        result = await agent.execute_task("Click the OK button", max_steps=2)
        
        assert result["status"] in ["completed", "failed"]
        assert len(result["steps"]) > 0

    def test_code_agent_workflow(self):
        """Test complete code agent workflow."""
        agent = create_code_agent(name="IntegrationTest")
        
        # Generate code
        code = agent.generate("Create a function to add two numbers")
        
        # Analyze it
        analysis = agent.analyze(code)
        
        # Get refactoring suggestions
        suggestions, _ = agent.refactor(code)
        
        assert isinstance(code, CodeBlock)
        assert isinstance(analysis, CodeAnalysis)
        assert isinstance(suggestions, list)

    def test_code_agent_debug_workflow(self):
        """Test code agent debugging workflow."""
        agent = create_code_agent()
        
        # Debug problematic code
        result = agent.debug(
            "print(undefined_var)",
            "NameError: name 'undefined_var' is not defined"
        )
        
        assert result["error_type"] == "NameError"
        assert result["analysis"] is not None
