"""
Browser Agent: Web Navigation and Information Extraction.

Core Idea:
    This module provides agents capable of navigating web pages,
    interacting with web elements, filling forms, and extracting
    information from web content.

Mathematical Foundation:
    Web navigation as MDP:
        S: Page states (URL, DOM)
        A: Actions (click, type, scroll, navigate)
        T: Page transitions
        R: Task completion reward

    Optimal navigation policy:
        π*(s) = argmax_a Σ P(s'|s,a) * [R(s,a,s') + γV*(s')]

Design Patterns:
    - State Pattern: Page state management
    - Command Pattern: Navigation actions
    - Strategy Pattern: Different extraction strategies

References:
    - WebArena: https://arxiv.org/abs/2307.13854
    - Mind2Web: https://arxiv.org/abs/2306.06070
    - WebAgent: https://arxiv.org/abs/2401.01614

Author: zhangfeng
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Union,
)
from urllib.parse import urlparse

__all__ = [
    "PageState",
    "DOMElement",
    "ElementType",
    "NavigationAction",
    "FormAction",
    "ClickAction",
    "ScrollAction",
    "WaitAction",
    "PageAnalyzer",
    "NavigationPlanner",
    "BrowserController",
    "BrowserAgent",
    "BrowserAgentConfig",
    "create_browser_agent",
]

logger = logging.getLogger(__name__)


class ElementType(str, Enum):
    """Types of DOM elements."""
    LINK = "link"
    BUTTON = "button"
    INPUT = "input"
    TEXTAREA = "textarea"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    IMAGE = "image"
    FORM = "form"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    DIV = "div"
    SPAN = "span"
    IFRAME = "iframe"
    VIDEO = "video"
    UNKNOWN = "unknown"


class ActionStatus(str, Enum):
    """Status of browser action execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class DOMElement:
    """
    Represents a DOM element on a web page.

    Attributes:
        element_id: Unique identifier
        element_type: Type of element
        tag_name: HTML tag name
        text: Text content
        attributes: Element attributes (id, class, href, etc.)
        xpath: XPath selector
        css_selector: CSS selector
        is_visible: Whether element is visible
        is_enabled: Whether element is interactive
        bounding_box: Element position and size
    """
    element_id: str = field(default_factory=lambda: f"elem_{uuid.uuid4().hex[:8]}")
    element_type: ElementType = ElementType.UNKNOWN
    tag_name: str = ""
    text: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    xpath: str = ""
    css_selector: str = ""
    is_visible: bool = True
    is_enabled: bool = True
    bounding_box: dict[str, float] | None = None
    children: list[str] = field(default_factory=list)

    @property
    def id_attr(self) -> str | None:
        return self.attributes.get("id")

    @property
    def class_attr(self) -> str | None:
        return self.attributes.get("class")

    @property
    def href(self) -> str | None:
        return self.attributes.get("href")

    @property
    def value(self) -> str | None:
        return self.attributes.get("value")

    def matches_text(self, query: str, case_sensitive: bool = False) -> bool:
        """Check if element text matches query."""
        if case_sensitive:
            return query in self.text
        return query.lower() in self.text.lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "tag_name": self.tag_name,
            "text": self.text[:100] if self.text else "",
            "attributes": self.attributes,
            "xpath": self.xpath,
            "is_visible": self.is_visible,
            "is_enabled": self.is_enabled,
        }


@dataclass
class PageState:
    """
    Represents the current state of a web page.

    Attributes:
        url: Current page URL
        title: Page title
        elements: Detected DOM elements
        html: Page HTML content
        screenshot: Base64 encoded screenshot
        load_time: Page load time in seconds
    """
    url: str = ""
    title: str = ""
    elements: list[DOMElement] = field(default_factory=list)
    html: str = ""
    screenshot: str | None = None
    load_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cookies: dict[str, str] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        """Get domain from URL."""
        parsed = urlparse(self.url)
        return parsed.netloc

    def find_element_by_text(
        self,
        text: str,
        element_type: ElementType | None = None,
    ) -> DOMElement | None:
        """Find element by text content."""
        for elem in self.elements:
            if elem.matches_text(text):
                if element_type is None or elem.element_type == element_type:
                    return elem
        return None

    def find_element_by_id(self, element_id: str) -> DOMElement | None:
        """Find element by HTML id attribute."""
        for elem in self.elements:
            if elem.id_attr == element_id:
                return elem
        return None

    def find_elements_by_type(self, element_type: ElementType) -> list[DOMElement]:
        """Find all elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]

    def get_links(self) -> list[DOMElement]:
        """Get all link elements."""
        return self.find_elements_by_type(ElementType.LINK)

    def get_forms(self) -> list[DOMElement]:
        """Get all form elements."""
        return self.find_elements_by_type(ElementType.FORM)

    def get_inputs(self) -> list[DOMElement]:
        """Get all input elements."""
        return self.find_elements_by_type(ElementType.INPUT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "element_count": len(self.elements),
            "load_time": self.load_time,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NavigationAction:
    """Navigate to a URL."""
    action_type: str = "navigate"
    url: str = ""
    wait_for: str = "load"  # load, networkidle, domcontentloaded
    timeout: float = 30.0


@dataclass
class ClickAction:
    """Click on an element."""
    action_type: str = "click"
    selector: str = ""  # CSS selector or XPath
    element_id: str | None = None
    text: str | None = None  # Click element with this text
    button: str = "left"  # left, right, middle
    click_count: int = 1


@dataclass
class FormAction:
    """Fill a form field."""
    action_type: str = "fill"
    selector: str = ""
    value: str = ""
    clear_first: bool = True


@dataclass
class ScrollAction:
    """Scroll the page."""
    action_type: str = "scroll"
    direction: str = "down"  # up, down, left, right
    amount: int = 300  # pixels
    to_element: str | None = None  # Scroll to element


@dataclass
class WaitAction:
    """Wait for a condition."""
    action_type: str = "wait"
    condition: str = "time"  # time, element, url, text
    value: float | str = 1.0
    timeout: float = 30.0


@dataclass
class ActionResult:
    """Result of a browser action."""
    action_id: str
    status: ActionStatus
    output: Any = None
    error: str | None = None
    execution_time: float = 0.0
    before_state: PageState | None = None
    after_state: PageState | None = None

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
        }


BrowserAction = Union[NavigationAction, ClickAction, FormAction, ScrollAction, WaitAction]


class PageAnalyzer:
    """
    Analyzes web page content and structure.

    Core Idea:
        Parses HTML to extract interactive elements and
        understand page structure for navigation.

    Example:
        >>> analyzer = PageAnalyzer()
        >>> state = analyzer.analyze(html, url)
        >>> forms = state.get_forms()
    """

    def __init__(self, llm_func: Callable[[str], str] | None = None) -> None:
        self.llm_func = llm_func or self._mock_llm

    def _mock_llm(self, prompt: str) -> str:
        return "Page contains navigation menu, search form, and content area."

    def analyze(self, html: str, url: str) -> PageState:
        """Analyze HTML and create PageState."""
        elements = self._extract_elements(html)
        title = self._extract_title(html)

        return PageState(
            url=url,
            title=title,
            elements=elements,
            html=html,
        )

    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML."""
        match = re.search(r"<title[^>]*>([^<]*)</title>", html, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_elements(self, html: str) -> list[DOMElement]:
        """Extract DOM elements from HTML."""
        elements: list[DOMElement] = []

        # Extract links
        for match in re.finditer(r'<a\s+([^>]*)>([^<]*)</a>', html, re.IGNORECASE):
            attrs = self._parse_attributes(match.group(1))
            elements.append(DOMElement(
                element_type=ElementType.LINK,
                tag_name="a",
                text=match.group(2).strip(),
                attributes=attrs,
            ))

        # Extract buttons
        for match in re.finditer(r'<button\s*([^>]*)>([^<]*)</button>', html, re.IGNORECASE):
            attrs = self._parse_attributes(match.group(1))
            elements.append(DOMElement(
                element_type=ElementType.BUTTON,
                tag_name="button",
                text=match.group(2).strip(),
                attributes=attrs,
            ))

        # Extract inputs
        for match in re.finditer(r'<input\s+([^>]*)/?>', html, re.IGNORECASE):
            attrs = self._parse_attributes(match.group(1))
            input_type = attrs.get("type", "text")
            elem_type = ElementType.INPUT
            if input_type == "checkbox":
                elem_type = ElementType.CHECKBOX
            elif input_type == "radio":
                elem_type = ElementType.RADIO
            elements.append(DOMElement(
                element_type=elem_type,
                tag_name="input",
                attributes=attrs,
            ))

        # Extract textareas
        for match in re.finditer(r'<textarea\s*([^>]*)>([^<]*)</textarea>', html, re.IGNORECASE):
            attrs = self._parse_attributes(match.group(1))
            elements.append(DOMElement(
                element_type=ElementType.TEXTAREA,
                tag_name="textarea",
                text=match.group(2).strip(),
                attributes=attrs,
            ))

        return elements

    def _parse_attributes(self, attr_string: str) -> dict[str, str]:
        """Parse HTML attributes from string."""
        attrs: dict[str, str] = {}
        for match in re.finditer(r'(\w+)=["\']([^"\']*)["\']', attr_string):
            attrs[match.group(1)] = match.group(2)
        return attrs

    def describe_page(self, state: PageState) -> str:
        """Generate natural language description of page."""
        prompt = f"""Describe this web page:
URL: {state.url}
Title: {state.title}
Elements: {len(state.elements)} ({len(state.get_links())} links, {len(state.get_forms())} forms)
"""
        return self.llm_func(prompt)


class NavigationPlanner:
    """
    Plans navigation sequences to achieve goals.

    Core Idea:
        Given a goal and current page state, plans a sequence
        of browser actions to achieve the goal.

    Example:
        >>> planner = NavigationPlanner()
        >>> actions = await planner.plan("Search for Python tutorials", page_state)
    """

    def __init__(self, llm_func: Callable[[str], str] | None = None) -> None:
        self.llm_func = llm_func or self._mock_llm
        self._plan_history: list[dict[str, Any]] = []

    def _mock_llm(self, prompt: str) -> str:
        if "search" in prompt.lower():
            return "FILL: search_input with 'query'\nCLICK: search_button"
        return "NAVIGATE: to target page"

    async def plan(
        self,
        goal: str,
        page_state: PageState,
        max_actions: int = 10,
    ) -> list[BrowserAction]:
        """Plan actions to achieve goal."""
        actions: list[BrowserAction] = []

        goal_lower = goal.lower()

        # Handle navigation goals
        if goal_lower.startswith("go to") or goal_lower.startswith("navigate"):
            url = self._extract_url(goal)
            if url:
                actions.append(NavigationAction(url=url))

        # Handle search goals
        elif "search" in goal_lower:
            query = self._extract_query(goal)
            search_input = self._find_search_input(page_state)
            if search_input:
                actions.append(FormAction(
                    selector=f"#{search_input.id_attr}" if search_input.id_attr else "input[type='search']",
                    value=query,
                ))
                actions.append(ClickAction(text="Search"))

        # Handle click goals
        elif "click" in goal_lower:
            target = self._extract_click_target(goal)
            # Try exact match first
            element = page_state.find_element_by_text(target)
            if not element:
                # Try partial match - element text in target or target in element text
                for elem in page_state.elements:
                    if elem.text:
                        if elem.text.lower() in target.lower() or target.lower() in elem.text.lower():
                            element = elem
                            break
            if element:
                actions.append(ClickAction(
                    selector=f"#{element.id_attr}" if element.id_attr else "",
                    text=element.text,
                ))

        # Handle form filling
        elif "fill" in goal_lower or "enter" in goal_lower:
            field_value = self._extract_field_value(goal)
            if field_value:
                field, value = field_value
                actions.append(FormAction(selector=f"input[name='{field}']", value=value))

        self._plan_history.append({
            "goal": goal,
            "actions": [self._action_to_dict(a) for a in actions],
            "timestamp": datetime.utcnow().isoformat(),
        })

        return actions

    def _extract_url(self, goal: str) -> str | None:
        """Extract URL from goal."""
        match = re.search(r'https?://\S+', goal)
        if match:
            return match.group(0)
        # Try to construct URL from domain mention
        match = re.search(r'(?:go to|navigate to)\s+(\S+)', goal, re.IGNORECASE)
        if match:
            domain = match.group(1)
            if not domain.startswith("http"):
                return f"https://{domain}"
            return domain
        return None

    def _extract_query(self, goal: str) -> str:
        """Extract search query from goal."""
        patterns = [
            r"search for ['\"]?([^'\"]+)['\"]?",
            r"search ['\"]?([^'\"]+)['\"]?",
            r"find ['\"]?([^'\"]+)['\"]?",
        ]
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return goal

    def _extract_click_target(self, goal: str) -> str:
        """Extract click target from goal."""
        patterns = [
            r"click (?:on )?(?:the )?['\"]?([^'\"]+)['\"]?(?:\s+button)?",
            r"click ['\"]?([^'\"]+)['\"]?",
        ]
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_field_value(self, goal: str) -> tuple[str, str] | None:
        """Extract field and value from goal."""
        patterns = [
            r"(?:fill|enter)\s+['\"]?([^'\"]+)['\"]?\s+(?:in|into)\s+['\"]?([^'\"]+)['\"]?",
            r"(?:fill|enter)\s+['\"]?([^'\"]+)['\"]?",
        ]
        for pattern in patterns:
            match = re.search(pattern, goal, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    return (match.group(2), match.group(1))
                return ("input", match.group(1))
        return None

    def _find_search_input(self, state: PageState) -> DOMElement | None:
        """Find search input field."""
        for elem in state.elements:
            if elem.element_type == ElementType.INPUT:
                if any(kw in str(elem.attributes).lower() for kw in ["search", "query", "q"]):
                    return elem
        # Return first input as fallback
        inputs = state.get_inputs()
        return inputs[0] if inputs else None

    def _action_to_dict(self, action: BrowserAction) -> dict[str, Any]:
        """Convert action to dictionary."""
        if hasattr(action, "__dict__"):
            return {k: v for k, v in action.__dict__.items() if not k.startswith("_")}
        return {"action_type": str(action)}

    def get_plan_history(self) -> list[dict[str, Any]]:
        return self._plan_history.copy()


class BrowserController:
    """
    Controls browser for action execution.

    Core Idea:
        Provides a safe interface to execute browser actions,
        with support for simulation mode for testing.

    Example:
        >>> controller = BrowserController(simulate=True)
        >>> result = await controller.execute(navigate_action)
    """

    def __init__(
        self,
        simulate: bool = True,
        default_timeout: float = 30.0,
        action_delay: float = 0.5,
    ) -> None:
        self.simulate = simulate
        self.default_timeout = default_timeout
        self.action_delay = action_delay
        self._current_state: PageState | None = None
        self._execution_history: list[ActionResult] = []

    async def execute(self, action: BrowserAction) -> ActionResult:
        """Execute a browser action."""
        action_id = f"action_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        try:
            if isinstance(action, NavigationAction):
                output = await self._execute_navigate(action)
            elif isinstance(action, ClickAction):
                output = await self._execute_click(action)
            elif isinstance(action, FormAction):
                output = await self._execute_fill(action)
            elif isinstance(action, ScrollAction):
                output = await self._execute_scroll(action)
            elif isinstance(action, WaitAction):
                output = await self._execute_wait(action)
            else:
                output = "Unknown action type"

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

    async def _execute_navigate(self, action: NavigationAction) -> str:
        """Execute navigation action."""
        await asyncio.sleep(self.action_delay)

        if self.simulate:
            logger.info(f"[SIMULATE] Navigate to: {action.url}")
            self._current_state = PageState(
                url=action.url,
                title=f"Page: {action.url}",
                elements=[
                    DOMElement(element_type=ElementType.LINK, text="Home", attributes={"href": "/"}),
                    DOMElement(element_type=ElementType.INPUT, attributes={"type": "search", "name": "q"}),
                    DOMElement(element_type=ElementType.BUTTON, text="Search"),
                ],
            )
            return f"Navigated to {action.url}"

        # Real implementation would use Playwright/Selenium
        return f"Navigated to {action.url}"

    async def _execute_click(self, action: ClickAction) -> str:
        """Execute click action."""
        await asyncio.sleep(self.action_delay)

        if self.simulate:
            target = action.selector or action.text or action.element_id
            logger.info(f"[SIMULATE] Click: {target}")
            return f"Clicked on {target}"

        return f"Clicked on {action.selector or action.text}"

    async def _execute_fill(self, action: FormAction) -> str:
        """Execute form fill action."""
        await asyncio.sleep(self.action_delay)

        if self.simulate:
            logger.info(f"[SIMULATE] Fill {action.selector} with '{action.value}'")
            return f"Filled {action.selector} with '{action.value}'"

        return f"Filled {action.selector}"

    async def _execute_scroll(self, action: ScrollAction) -> str:
        """Execute scroll action."""
        await asyncio.sleep(self.action_delay)

        if self.simulate:
            logger.info(f"[SIMULATE] Scroll {action.direction} {action.amount}px")
            return f"Scrolled {action.direction} {action.amount}px"

        return f"Scrolled {action.direction}"

    async def _execute_wait(self, action: WaitAction) -> str:
        """Execute wait action."""
        if action.condition == "time":
            await asyncio.sleep(float(action.value))
            return f"Waited {action.value} seconds"

        if self.simulate:
            await asyncio.sleep(0.1)
            return f"Waited for {action.condition}: {action.value}"

        return f"Waited for {action.condition}"

    async def execute_sequence(
        self,
        actions: list[BrowserAction],
    ) -> list[ActionResult]:
        """Execute a sequence of actions."""
        results = []
        for action in actions:
            result = await self.execute(action)
            results.append(result)
            if not result.is_success:
                break
        return results

    def get_current_state(self) -> PageState | None:
        return self._current_state

    def get_execution_history(self) -> list[ActionResult]:
        return self._execution_history.copy()

    def clear_history(self) -> None:
        self._execution_history.clear()


@dataclass
class BrowserAgentConfig:
    """Configuration for BrowserAgent."""
    name: str = "BrowserAgent"
    simulate: bool = True
    max_steps: int = 50
    default_timeout: float = 30.0
    action_delay: float = 0.5
    enable_screenshots: bool = True
    verbose: bool = False


class BrowserAgent:
    """
    Agent for automated web browsing and information extraction.

    Core Idea:
        Combines page analysis, navigation planning, and browser
        control to perform automated web tasks.

    Example:
        >>> agent = BrowserAgent()
        >>> result = await agent.execute_task("Search Google for Python tutorials")
    """

    def __init__(
        self,
        config: BrowserAgentConfig | None = None,
        llm_func: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config or BrowserAgentConfig()
        self.llm_func = llm_func or self._mock_llm

        self.page_analyzer = PageAnalyzer(llm_func=self.llm_func)
        self.navigation_planner = NavigationPlanner(llm_func=self.llm_func)
        self.browser_controller = BrowserController(
            simulate=self.config.simulate,
            default_timeout=self.config.default_timeout,
            action_delay=self.config.action_delay,
        )

        self._task_history: list[dict[str, Any]] = []
        self._is_running = False
        logger.info(f"BrowserAgent '{self.config.name}' initialized")

    def _mock_llm(self, prompt: str) -> str:
        return "Page analysis complete."

    async def execute_task(
        self,
        task: str,
        start_url: str | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute a web browsing task.

        Args:
            task: Natural language task description
            start_url: Initial URL to navigate to
            max_steps: Maximum steps to take

        Returns:
            Task execution result
        """
        max_steps = max_steps or self.config.max_steps
        self._is_running = True

        task_record = {
            "task": task,
            "start_url": start_url,
            "start_time": datetime.utcnow().isoformat(),
            "steps": [],
            "status": "running",
        }

        try:
            # Navigate to start URL if provided
            if start_url:
                nav_action = NavigationAction(url=start_url)
                await self.browser_controller.execute(nav_action)

            for step in range(max_steps):
                # Get current state
                page_state = self.browser_controller.get_current_state()
                if not page_state:
                    page_state = PageState(url=start_url or "about:blank")

                # Plan actions
                actions = await self.navigation_planner.plan(task, page_state)

                if not actions:
                    logger.info("No more actions to perform")
                    break

                # Execute actions
                results = await self.browser_controller.execute_sequence(actions)

                step_record = {
                    "step": step + 1,
                    "page_url": page_state.url,
                    "actions": len(actions),
                    "results": [r.to_dict() for r in results],
                }
                task_record["steps"].append(step_record)

                # Check if task is complete
                if self._is_task_complete(task, page_state, results):
                    break

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
        page_state: PageState,
        results: list[ActionResult],
    ) -> bool:
        """Check if task is complete."""
        return len(results) > 0 and all(r.is_success for r in results)

    async def navigate(self, url: str) -> ActionResult:
        """Navigate to a URL."""
        action = NavigationAction(url=url)
        return await self.browser_controller.execute(action)

    async def click(
        self,
        selector: str | None = None,
        text: str | None = None,
    ) -> ActionResult:
        """Click an element."""
        action = ClickAction(selector=selector or "", text=text)
        return await self.browser_controller.execute(action)

    async def fill(self, selector: str, value: str) -> ActionResult:
        """Fill a form field."""
        action = FormAction(selector=selector, value=value)
        return await self.browser_controller.execute(action)

    async def search(self, query: str, search_url: str = "https://www.google.com") -> dict[str, Any]:
        """Perform a search."""
        return await self.execute_task(
            f"Search for '{query}'",
            start_url=search_url,
            max_steps=5,
        )

    async def extract_links(self) -> list[dict[str, str]]:
        """Extract all links from current page."""
        state = self.browser_controller.get_current_state()
        if not state:
            return []

        links = state.get_links()
        return [{"text": l.text, "href": l.href or ""} for l in links]

    async def extract_text(self) -> str:
        """Extract text content from current page."""
        state = self.browser_controller.get_current_state()
        if not state:
            return ""

        texts = [e.text for e in state.elements if e.text]
        return "\n".join(texts)

    def get_current_url(self) -> str | None:
        """Get current page URL."""
        state = self.browser_controller.get_current_state()
        return state.url if state else None

    def get_task_history(self) -> list[dict[str, Any]]:
        return self._task_history.copy()

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "is_running": self._is_running,
            "simulate_mode": self.config.simulate,
            "current_url": self.get_current_url(),
            "tasks_completed": len(self._task_history),
        }

    def __repr__(self) -> str:
        return f"BrowserAgent(name='{self.config.name}', simulate={self.config.simulate})"


def create_browser_agent(
    name: str = "BrowserAgent",
    simulate: bool = True,
    llm_func: Callable[[str], str] | None = None,
    **config_kwargs: Any,
) -> BrowserAgent:
    """
    Factory function to create a BrowserAgent.

    Args:
        name: Agent name
        simulate: Whether to simulate browser actions
        llm_func: LLM function for planning
        **config_kwargs: Additional config parameters

    Returns:
        Configured BrowserAgent instance
    """
    config = BrowserAgentConfig(name=name, simulate=simulate, **config_kwargs)
    return BrowserAgent(config=config, llm_func=llm_func)
