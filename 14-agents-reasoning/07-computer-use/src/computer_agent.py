"""
Computer Use Agent: Screen Understanding and GUI Automation.

Core Idea:
    This module provides agents capable of understanding screen content,
    identifying UI elements, and performing automated GUI interactions.
    It simulates human-like computer interaction through visual understanding.

Mathematical Foundation:
    Screen understanding as object detection:
        P(element | image) = softmax(f_θ(image))

    Action planning as sequential decision:
        π*(s) = argmax_a Q(s, a) where Q is action-value function

Design Patterns:
    - Observer Pattern: Screen state monitoring
    - Command Pattern: GUI actions as commands
    - Strategy Pattern: Different interaction strategies

References:
    - Anthropic Computer Use: https://docs.anthropic.com/en/docs/agents-and-tools/computer-use
    - GPT-4V for GUI: https://arxiv.org/abs/2312.02003
    - SeeClick: https://arxiv.org/abs/2401.10935

Author: zhangfeng
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Protocol,
    runtime_checkable,
)

__all__ = [
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
    "ComputerAgentConfig",
    "create_computer_agent",
]

logger = logging.getLogger(__name__)


class UIElementType(str, Enum):
    """Types of UI elements that can be detected."""
    BUTTON = "button"
    TEXT_FIELD = "text_field"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    DROPDOWN = "dropdown"
    MENU = "menu"
    MENU_ITEM = "menu_item"
    LINK = "link"
    IMAGE = "image"
    ICON = "icon"
    TAB = "tab"
    WINDOW = "window"
    DIALOG = "dialog"
    SCROLL_BAR = "scroll_bar"
    SLIDER = "slider"
    TEXT = "text"
    LABEL = "label"
    TABLE = "table"
    LIST = "list"
    TREE = "tree"
    UNKNOWN = "unknown"


class MouseButton(str, Enum):
    """Mouse button types."""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class ActionStatus(str, Enum):
    """Status of an executed action."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ScreenRegion:
    """
    Represents a rectangular region on screen.

    Attributes:
        x: Left coordinate
        y: Top coordinate
        width: Region width
        height: Region height
    """
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of region."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains(self, x: int, y: int) -> bool:
        """Check if point is within region."""
        return self.x <= x < self.right and self.y <= y < self.bottom

    def overlaps(self, other: ScreenRegion) -> bool:
        """Check if regions overlap."""
        return not (
            self.right <= other.x or
            other.right <= self.x or
            self.bottom <= other.y or
            other.bottom <= self.y
        )

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class UIElement:
    """
    Represents a detected UI element on screen.

    Attributes:
        element_id: Unique identifier
        element_type: Type of UI element
        region: Bounding box on screen
        text: Text content if any
        confidence: Detection confidence (0-1)
        is_enabled: Whether element is interactive
        is_visible: Whether element is visible
        attributes: Additional element attributes
    """
    element_id: str = field(default_factory=lambda: f"elem_{uuid.uuid4().hex[:8]}")
    element_type: UIElementType = UIElementType.UNKNOWN
    region: ScreenRegion = field(default_factory=lambda: ScreenRegion(0, 0, 0, 0))
    text: str = ""
    confidence: float = 1.0
    is_enabled: bool = True
    is_visible: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of element."""
        return self.region.center

    def matches_text(self, query: str, case_sensitive: bool = False) -> bool:
        """Check if element text matches query."""
        if case_sensitive:
            return query in self.text
        return query.lower() in self.text.lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "region": self.region.to_dict(),
            "text": self.text,
            "confidence": self.confidence,
            "is_enabled": self.is_enabled,
            "is_visible": self.is_visible,
            "attributes": self.attributes,
        }


@dataclass
class ScreenState:
    """
    Current state of the screen.

    Attributes:
        screenshot: Base64 encoded screenshot
        elements: Detected UI elements
        active_window: Currently active window title
        mouse_position: Current mouse position
        timestamp: When state was captured
    """
    screenshot: str | None = None
    elements: list[UIElement] = field(default_factory=list)
    active_window: str = ""
    mouse_position: tuple[int, int] = (0, 0)
    resolution: tuple[int, int] = (1920, 1080)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def find_element_by_text(self, text: str, element_type: UIElementType | None = None) -> UIElement | None:
        """Find element by text content."""
        for elem in self.elements:
            if elem.matches_text(text):
                if element_type is None or elem.element_type == element_type:
                    return elem
        return None

    def find_elements_by_type(self, element_type: UIElementType) -> list[UIElement]:
        """Find all elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]

    def get_element_at(self, x: int, y: int) -> UIElement | None:
        """Get element at specific coordinates."""
        for elem in self.elements:
            if elem.region.contains(x, y):
                return elem
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_window": self.active_window,
            "mouse_position": self.mouse_position,
            "resolution": self.resolution,
            "element_count": len(self.elements),
            "elements": [e.to_dict() for e in self.elements],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MouseAction:
    """
    Represents a mouse action.

    Attributes:
        action_type: Type of mouse action
        x: Target x coordinate
        y: Target y coordinate
        button: Mouse button to use
        clicks: Number of clicks
        duration: Duration for drag operations
    """
    action_type: str  # click, double_click, right_click, drag, scroll, move
    x: int = 0
    y: int = 0
    button: MouseButton = MouseButton.LEFT
    clicks: int = 1
    duration: float = 0.0
    scroll_amount: int = 0
    drag_to: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "x": self.x,
            "y": self.y,
            "button": self.button.value,
            "clicks": self.clicks,
            "duration": self.duration,
            "scroll_amount": self.scroll_amount,
            "drag_to": self.drag_to,
        }


@dataclass
class KeyboardAction:
    """
    Represents a keyboard action.

    Attributes:
        action_type: Type of keyboard action
        text: Text to type
        key: Special key to press
        modifiers: Modifier keys (ctrl, alt, shift)
    """
    action_type: str  # type, press, hotkey
    text: str = ""
    key: str = ""
    modifiers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "text": self.text,
            "key": self.key,
            "modifiers": self.modifiers,
        }


@dataclass
class ActionResult:
    """Result of an executed action."""
    action_id: str
    status: ActionStatus
    output: Any = None
    error: str | None = None
    execution_time: float = 0.0
    before_state: ScreenState | None = None
    after_state: ScreenState | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_success(self) -> bool:
        return self.status == ActionStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "output": str(self.output) if self.output else None,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
        }


@runtime_checkable
class VisionModel(Protocol):
    """Protocol for vision models that analyze screenshots."""

    async def analyze_screenshot(
        self,
        screenshot: str,
        prompt: str,
    ) -> str:
        """Analyze screenshot and return description or structured output."""
        ...


class MockVisionModel:
    """Mock vision model for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["I see a desktop with several icons and windows."]
        self._call_count = 0
        self.call_history: list[dict[str, Any]] = []

    async def analyze_screenshot(self, screenshot: str, prompt: str) -> str:
        self.call_history.append({"screenshot_length": len(screenshot), "prompt": prompt})
        response = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return response


class ScreenAnalyzer:
    """
    Analyzes screen content using vision models.

    Core Idea:
        Uses vision-language models to understand screen content,
        detect UI elements, and extract semantic information.

    Example:
        >>> analyzer = ScreenAnalyzer(vision_model)
        >>> state = await analyzer.capture_and_analyze()
        >>> button = state.find_element_by_text("Submit")
    """

    def __init__(
        self,
        vision_model: VisionModel | None = None,
        element_detection_prompt: str | None = None,
    ) -> None:
        self.vision_model = vision_model or MockVisionModel()
        self.element_detection_prompt = element_detection_prompt or self._default_detection_prompt()
        self._last_state: ScreenState | None = None

    def _default_detection_prompt(self) -> str:
        return """Analyze this screenshot and identify all interactive UI elements.
For each element, provide:
1. Type (button, text_field, checkbox, link, etc.)
2. Text content if visible
3. Approximate position (x, y, width, height)
4. Whether it appears enabled/clickable

Format as JSON list."""

    async def capture_screenshot(self) -> str:
        """Capture current screen as base64 string."""
        # In real implementation, use pyautogui or similar
        # For now, return mock data
        return base64.b64encode(b"mock_screenshot_data").decode()

    async def analyze_screenshot(self, screenshot: str) -> list[UIElement]:
        """Analyze screenshot to detect UI elements."""
        response = await self.vision_model.analyze_screenshot(
            screenshot, self.element_detection_prompt
        )
        return self._parse_elements(response)

    def _parse_elements(self, response: str) -> list[UIElement]:
        """Parse vision model response into UIElement list."""
        # Mock parsing - in real implementation, parse JSON response
        elements = [
            UIElement(
                element_type=UIElementType.BUTTON,
                region=ScreenRegion(100, 100, 80, 30),
                text="OK",
                confidence=0.95,
            ),
            UIElement(
                element_type=UIElementType.TEXT_FIELD,
                region=ScreenRegion(100, 150, 200, 25),
                text="",
                confidence=0.90,
            ),
        ]
        return elements

    async def capture_and_analyze(self) -> ScreenState:
        """Capture screen and analyze in one call."""
        screenshot = await self.capture_screenshot()
        elements = await self.analyze_screenshot(screenshot)

        state = ScreenState(
            screenshot=screenshot,
            elements=elements,
            active_window="Mock Window",
            mouse_position=(500, 300),
        )
        self._last_state = state
        return state

    def get_last_state(self) -> ScreenState | None:
        """Get last captured screen state."""
        return self._last_state


class ActionPlanner:
    """
    Plans sequences of actions to achieve goals.

    Core Idea:
        Given a goal and current screen state, plans a sequence of
        mouse/keyboard actions to achieve the goal.

    Example:
        >>> planner = ActionPlanner(llm_func)
        >>> actions = await planner.plan("Click the Submit button", screen_state)
    """

    def __init__(
        self,
        llm_func: Callable[[str], str] | None = None,
    ) -> None:
        self.llm_func = llm_func or self._mock_llm
        self._action_history: list[dict[str, Any]] = []

    def _mock_llm(self, prompt: str) -> str:
        """Mock LLM for testing."""
        if "click" in prompt.lower():
            return "ACTION: click at (150, 115)"
        elif "type" in prompt.lower():
            return "ACTION: type 'Hello World'"
        return "ACTION: wait"

    async def plan(
        self,
        goal: str,
        screen_state: ScreenState,
        max_steps: int = 10,
    ) -> list[MouseAction | KeyboardAction]:
        """
        Plan actions to achieve goal.

        Args:
            goal: Natural language goal description
            screen_state: Current screen state
            max_steps: Maximum number of actions to plan

        Returns:
            List of planned actions
        """
        actions: list[MouseAction | KeyboardAction] = []

        # Find relevant element
        element = self._find_target_element(goal, screen_state)

        if element:
            if "click" in goal.lower():
                cx, cy = element.center
                actions.append(MouseAction(
                    action_type="click",
                    x=cx,
                    y=cy,
                    button=MouseButton.LEFT,
                ))
            elif "type" in goal.lower() or "enter" in goal.lower():
                # First click to focus
                cx, cy = element.center
                actions.append(MouseAction(
                    action_type="click",
                    x=cx,
                    y=cy,
                ))
                # Then type
                text = self._extract_text_to_type(goal)
                if text:
                    actions.append(KeyboardAction(
                        action_type="type",
                        text=text,
                    ))

        self._action_history.append({
            "goal": goal,
            "actions": [a.to_dict() for a in actions],
            "timestamp": datetime.utcnow().isoformat(),
        })

        return actions

    def _find_target_element(
        self,
        goal: str,
        screen_state: ScreenState,
    ) -> UIElement | None:
        """Find the UI element relevant to the goal."""
        # Extract potential element text from goal
        goal_lower = goal.lower()

        for elem in screen_state.elements:
            if elem.text and elem.text.lower() in goal_lower:
                return elem
            if elem.is_enabled and elem.element_type == UIElementType.BUTTON:
                return elem

        return screen_state.elements[0] if screen_state.elements else None

    def _extract_text_to_type(self, goal: str) -> str:
        """Extract text to type from goal."""
        # Simple extraction - look for quoted text
        import re
        match = re.search(r'"([^"]*)"', goal)
        if match:
            return match.group(1)
        match = re.search(r"'([^']*)'", goal)
        if match:
            return match.group(1)
        return ""

    def get_action_history(self) -> list[dict[str, Any]]:
        return self._action_history.copy()


class GUIController:
    """
    Executes GUI actions (mouse/keyboard).

    Core Idea:
        Provides a safe interface to execute GUI actions,
        with support for simulation mode for testing.

    Example:
        >>> controller = GUIController(simulate=True)
        >>> result = await controller.execute(mouse_action)
    """

    def __init__(
        self,
        simulate: bool = True,
        action_delay: float = 0.1,
    ) -> None:
        self.simulate = simulate
        self.action_delay = action_delay
        self._execution_history: list[ActionResult] = []

    async def execute(
        self,
        action: MouseAction | KeyboardAction,
    ) -> ActionResult:
        """Execute a single action."""
        action_id = f"action_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        try:
            if isinstance(action, MouseAction):
                output = await self._execute_mouse(action)
            else:
                output = await self._execute_keyboard(action)

            result = ActionResult(
                action_id=action_id,
                status=ActionStatus.SUCCESS,
                output=output,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            result = ActionResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )

        self._execution_history.append(result)
        return result

    async def _execute_mouse(self, action: MouseAction) -> str:
        """Execute mouse action."""
        await asyncio.sleep(self.action_delay)

        if self.simulate:
            logger.info(f"[SIMULATE] Mouse {action.action_type} at ({action.x}, {action.y})")
            return f"Simulated: {action.action_type} at ({action.x}, {action.y})"

        # Real implementation would use pyautogui
        # import pyautogui
        # if action.action_type == "click":
        #     pyautogui.click(action.x, action.y, button=action.button.value)
        # ...

        return f"Executed: {action.action_type} at ({action.x}, {action.y})"

    async def _execute_keyboard(self, action: KeyboardAction) -> str:
        """Execute keyboard action."""
        await asyncio.sleep(self.action_delay)

        if self.simulate:
            logger.info(f"[SIMULATE] Keyboard {action.action_type}: {action.text or action.key}")
            return f"Simulated: {action.action_type} '{action.text or action.key}'"

        # Real implementation would use pyautogui
        # import pyautogui
        # if action.action_type == "type":
        #     pyautogui.typewrite(action.text)
        # ...

        return f"Executed: {action.action_type} '{action.text or action.key}'"

    async def execute_sequence(
        self,
        actions: list[MouseAction | KeyboardAction],
    ) -> list[ActionResult]:
        """Execute a sequence of actions."""
        results = []
        for action in actions:
            result = await self.execute(action)
            results.append(result)
            if not result.is_success:
                break
        return results

    def get_execution_history(self) -> list[ActionResult]:
        return self._execution_history.copy()

    def clear_history(self) -> None:
        self._execution_history.clear()


@dataclass
class ComputerAgentConfig:
    """Configuration for ComputerAgent."""
    name: str = "ComputerAgent"
    simulate: bool = True
    max_steps: int = 50
    action_delay: float = 0.1
    screenshot_interval: float = 1.0
    enable_verification: bool = True
    verbose: bool = False


class ComputerAgent:
    """
    Agent capable of computer use through screen understanding and GUI automation.

    Core Idea:
        Combines screen analysis, action planning, and GUI control to
        perform automated tasks on a computer, similar to human interaction.

    Architecture:
        Screen → ScreenAnalyzer → ScreenState
                      ↓
        Goal → ActionPlanner → Actions
                      ↓
        Actions → GUIController → Results

    Example:
        >>> agent = ComputerAgent()
        >>> result = await agent.execute_task("Open Chrome and search for 'Python'")
        >>> print(result)
    """

    def __init__(
        self,
        config: ComputerAgentConfig | None = None,
        vision_model: VisionModel | None = None,
        llm_func: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config or ComputerAgentConfig()

        self.screen_analyzer = ScreenAnalyzer(vision_model=vision_model)
        self.action_planner = ActionPlanner(llm_func=llm_func)
        self.gui_controller = GUIController(
            simulate=self.config.simulate,
            action_delay=self.config.action_delay,
        )

        self._task_history: list[dict[str, Any]] = []
        self._is_running = False
        logger.info(f"ComputerAgent '{self.config.name}' initialized")

    async def execute_task(
        self,
        task: str,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute a task described in natural language.

        Args:
            task: Natural language task description
            max_steps: Maximum steps to take

        Returns:
            Task execution result
        """
        max_steps = max_steps or self.config.max_steps
        self._is_running = True

        task_record = {
            "task": task,
            "start_time": datetime.utcnow().isoformat(),
            "steps": [],
            "status": "running",
        }

        try:
            for step in range(max_steps):
                # Capture current state
                screen_state = await self.screen_analyzer.capture_and_analyze()

                # Plan actions
                actions = await self.action_planner.plan(task, screen_state)

                if not actions:
                    logger.info("No more actions to perform")
                    break

                # Execute actions
                results = await self.gui_controller.execute_sequence(actions)

                step_record = {
                    "step": step + 1,
                    "actions": [a.to_dict() for a in actions],
                    "results": [r.to_dict() for r in results],
                }
                task_record["steps"].append(step_record)

                # Check if task is complete (simplified)
                if self._is_task_complete(task, screen_state, results):
                    break

                # Wait before next iteration
                await asyncio.sleep(self.config.screenshot_interval)

            task_record["status"] = "completed"
            task_record["end_time"] = datetime.utcnow().isoformat()

        except Exception as e:
            task_record["status"] = "failed"
            task_record["error"] = str(e)
            logger.error(f"Task failed: {e}")

        finally:
            self._is_running = False
            self._task_history.append(task_record)

        return task_record

    def _is_task_complete(
        self,
        task: str,
        screen_state: ScreenState,
        results: list[ActionResult],
    ) -> bool:
        """Check if task is complete."""
        # Simplified completion check
        # In real implementation, use vision model to verify
        return len(results) > 0 and all(r.is_success for r in results)

    async def click_element(self, text: str) -> ActionResult:
        """Helper: Click element by text."""
        state = await self.screen_analyzer.capture_and_analyze()
        element = state.find_element_by_text(text)

        if not element:
            return ActionResult(
                action_id=f"action_{uuid.uuid4().hex[:8]}",
                status=ActionStatus.FAILED,
                error=f"Element with text '{text}' not found",
            )

        action = MouseAction(
            action_type="click",
            x=element.center[0],
            y=element.center[1],
        )
        return await self.gui_controller.execute(action)

    async def type_text(self, text: str) -> ActionResult:
        """Helper: Type text."""
        action = KeyboardAction(action_type="type", text=text)
        return await self.gui_controller.execute(action)

    async def press_key(self, key: str, modifiers: list[str] | None = None) -> ActionResult:
        """Helper: Press a key combination."""
        action = KeyboardAction(
            action_type="press" if not modifiers else "hotkey",
            key=key,
            modifiers=modifiers or [],
        )
        return await self.gui_controller.execute(action)

    def get_task_history(self) -> list[dict[str, Any]]:
        return self._task_history.copy()

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "is_running": self._is_running,
            "simulate_mode": self.config.simulate,
            "tasks_completed": len(self._task_history),
        }

    def __repr__(self) -> str:
        return f"ComputerAgent(name='{self.config.name}', simulate={self.config.simulate})"


def create_computer_agent(
    name: str = "ComputerAgent",
    simulate: bool = True,
    vision_model: VisionModel | None = None,
    llm_func: Callable[[str], str] | None = None,
    **config_kwargs: Any,
) -> ComputerAgent:
    """
    Factory function to create a ComputerAgent.

    Args:
        name: Agent name
        simulate: Whether to simulate actions
        vision_model: Vision model for screen analysis
        llm_func: LLM function for planning
        **config_kwargs: Additional config parameters

    Returns:
        Configured ComputerAgent instance
    """
    config = ComputerAgentConfig(name=name, simulate=simulate, **config_kwargs)
    return ComputerAgent(config=config, vision_model=vision_model, llm_func=llm_func)
