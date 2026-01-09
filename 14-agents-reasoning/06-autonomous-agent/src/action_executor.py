"""
Action Executor: Unified Action Execution for Autonomous Agents.

Core Idea:
    This module provides a unified interface for executing various types of actions
    including tool calls, code execution, file operations, and web requests.
    It handles timeouts, retries, and error recovery.

Mathematical Foundation:
    Retry with exponential backoff:
        delay(n) = min(base_delay * 2^n + jitter, max_delay)
    
    Success probability after k retries:
        P(success) = 1 - (1 - p)^k where p is single attempt success rate

Design Patterns:
    - Command Pattern: Actions as executable commands
    - Strategy Pattern: Different execution strategies per action type
    - Chain of Responsibility: Action validation and execution pipeline

References:
    - AutoGPT Command System
    - LangChain Tool Execution
    - OpenAI Function Calling

Author: AI-Practices
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

__all__ = [
    "ActionType",
    "ActionStatus",
    "ActionResult",
    "Action",
    "ActionExecutor",
    "ToolAction",
    "CodeAction",
    "FileAction",
    "WebAction",
    "ThinkAction",
    "ActionRegistry",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ActionType(str, Enum):
    """Types of actions an agent can perform."""
    TOOL = "tool"
    CODE = "code"
    FILE = "file"
    WEB = "web"
    THINK = "think"
    COMMUNICATE = "communicate"
    WAIT = "wait"


class ActionStatus(str, Enum):
    """Action execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ActionResult:
    """Result of an action execution."""
    action_id: str
    status: ActionStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_success(self) -> bool:
        return self.status == ActionStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "output": str(self.output) if self.output else None,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Action:
    """Base action data structure."""
    action_type: ActionType
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    action_id: str = field(default_factory=lambda: f"action_{uuid.uuid4().hex[:8]}")
    description: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "name": self.name,
            "parameters": self.parameters,
            "description": self.description,
            "timeout": self.timeout,
        }


class ActionHandler(ABC):
    """Abstract base class for action handlers."""

    @abstractmethod
    def can_handle(self, action: Action) -> bool:
        """Check if this handler can execute the action."""
        pass

    @abstractmethod
    def execute(self, action: Action) -> ActionResult:
        """Execute the action and return result."""
        pass


class ToolAction(ActionHandler):
    """Handler for tool/function call actions."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        self._tools[name] = func

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.TOOL

    def execute(self, action: Action) -> ActionResult:
        start_time = time.time()
        tool_name = action.name
        
        if tool_name not in self._tools:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=f"Tool not found: {tool_name}",
                execution_time=time.time() - start_time,
            )
        
        try:
            tool_func = self._tools[tool_name]
            result = tool_func(**action.parameters)
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.SUCCESS,
                output=result,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
                metadata={"traceback": traceback.format_exc()},
            )

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())


class CodeAction(ActionHandler):
    """Handler for code execution actions."""

    def __init__(self, allowed_modules: Optional[List[str]] = None) -> None:
        self.allowed_modules = allowed_modules or ["math", "json", "re", "datetime"]
        self._globals: Dict[str, Any] = {"__builtins__": {}}
        self._setup_safe_builtins()

    def _setup_safe_builtins(self) -> None:
        safe_builtins = [
            "abs", "all", "any", "bool", "dict", "enumerate", "filter",
            "float", "int", "len", "list", "map", "max", "min", "print",
            "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
        ]
        import builtins
        for name in safe_builtins:
            self._globals["__builtins__"][name] = getattr(builtins, name)
        
        for module_name in self.allowed_modules:
            try:
                self._globals[module_name] = __import__(module_name)
            except ImportError:
                pass

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.CODE

    def execute(self, action: Action) -> ActionResult:
        start_time = time.time()
        code = action.parameters.get("code", "")
        
        if not code:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error="No code provided",
                execution_time=time.time() - start_time,
            )
        
        try:
            local_vars: Dict[str, Any] = {}
            exec(code, self._globals.copy(), local_vars)
            
            result = local_vars.get("result", local_vars.get("output", None))
            
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.SUCCESS,
                output=result,
                execution_time=time.time() - start_time,
                metadata={"local_vars": list(local_vars.keys())},
            )
        except SyntaxError as e:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=f"Syntax error: {e}",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
                metadata={"traceback": traceback.format_exc()},
            )


class FileAction(ActionHandler):
    """Handler for file system actions."""

    def __init__(self, workspace_dir: Optional[str] = None) -> None:
        self.workspace_dir = workspace_dir or "."
        self._allowed_operations = {"read", "write", "append", "list", "exists", "delete"}

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.FILE

    def execute(self, action: Action) -> ActionResult:
        import os
        start_time = time.time()
        operation = action.parameters.get("operation", "read")
        path = action.parameters.get("path", "")
        
        if operation not in self._allowed_operations:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=f"Unknown operation: {operation}",
                execution_time=time.time() - start_time,
            )
        
        full_path = os.path.join(self.workspace_dir, path) if path else self.workspace_dir
        
        try:
            if operation == "read":
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return ActionResult(
                    action_id=action.action_id,
                    status=ActionStatus.SUCCESS,
                    output=content,
                    execution_time=time.time() - start_time,
                )
            
            elif operation == "write":
                content = action.parameters.get("content", "")
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return ActionResult(
                    action_id=action.action_id,
                    status=ActionStatus.SUCCESS,
                    output=f"Written {len(content)} bytes to {path}",
                    execution_time=time.time() - start_time,
                )
            
            elif operation == "append":
                content = action.parameters.get("content", "")
                with open(full_path, "a", encoding="utf-8") as f:
                    f.write(content)
                return ActionResult(
                    action_id=action.action_id,
                    status=ActionStatus.SUCCESS,
                    output=f"Appended {len(content)} bytes to {path}",
                    execution_time=time.time() - start_time,
                )
            
            elif operation == "list":
                files = os.listdir(full_path)
                return ActionResult(
                    action_id=action.action_id,
                    status=ActionStatus.SUCCESS,
                    output=files,
                    execution_time=time.time() - start_time,
                )
            
            elif operation == "exists":
                exists = os.path.exists(full_path)
                return ActionResult(
                    action_id=action.action_id,
                    status=ActionStatus.SUCCESS,
                    output=exists,
                    execution_time=time.time() - start_time,
                )
            
            elif operation == "delete":
                if os.path.exists(full_path):
                    os.remove(full_path)
                return ActionResult(
                    action_id=action.action_id,
                    status=ActionStatus.SUCCESS,
                    output=f"Deleted {path}",
                    execution_time=time.time() - start_time,
                )
            
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=f"Unhandled operation: {operation}",
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )


class WebAction(ActionHandler):
    """Handler for web request actions."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self._allowed_methods = {"GET", "POST"}

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.WEB

    def execute(self, action: Action) -> ActionResult:
        import urllib.request
        import urllib.error
        import json as json_module
        
        start_time = time.time()
        url = action.parameters.get("url", "")
        method = action.parameters.get("method", "GET").upper()
        headers = action.parameters.get("headers", {})
        data = action.parameters.get("data")
        
        if not url:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error="No URL provided",
                execution_time=time.time() - start_time,
            )
        
        if method not in self._allowed_methods:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=f"Method not allowed: {method}",
                execution_time=time.time() - start_time,
            )
        
        try:
            if data and isinstance(data, dict):
                data = json_module.dumps(data).encode("utf-8")
                headers["Content-Type"] = "application/json"
            elif data and isinstance(data, str):
                data = data.encode("utf-8")
            
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read().decode("utf-8")
                status_code = response.status
            
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.SUCCESS,
                output=content,
                execution_time=time.time() - start_time,
                metadata={"status_code": status_code, "url": url},
            )
            
        except urllib.error.HTTPError as e:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=f"HTTP {e.code}: {e.reason}",
                execution_time=time.time() - start_time,
                metadata={"status_code": e.code},
            )
        except urllib.error.URLError as e:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=f"URL Error: {e.reason}",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )


class ThinkAction(ActionHandler):
    """Handler for thinking/reasoning actions (no-op, for logging)."""

    def can_handle(self, action: Action) -> bool:
        return action.action_type == ActionType.THINK

    def execute(self, action: Action) -> ActionResult:
        start_time = time.time()
        thought = action.parameters.get("thought", "")
        
        return ActionResult(
            action_id=action.action_id,
            status=ActionStatus.SUCCESS,
            output=thought,
            execution_time=time.time() - start_time,
            metadata={"type": "thought"},
        )


class ActionRegistry:
    """Registry for action handlers."""

    def __init__(self) -> None:
        self._handlers: List[ActionHandler] = []

    def register(self, handler: ActionHandler) -> None:
        self._handlers.append(handler)

    def get_handler(self, action: Action) -> Optional[ActionHandler]:
        for handler in self._handlers:
            if handler.can_handle(action):
                return handler
        return None

    def list_handlers(self) -> List[str]:
        return [type(h).__name__ for h in self._handlers]


class ActionExecutor:
    """
    Central executor for all agent actions.
    
    Features:
    - Unified execution interface
    - Timeout handling
    - Retry with exponential backoff
    - Execution history tracking
    - Concurrent execution support
    
    Example:
        >>> executor = ActionExecutor()
        >>> action = Action(ActionType.TOOL, "calculator", {"expression": "2+2"})
        >>> result = executor.execute(action)
        >>> print(result.output)
    """

    def __init__(
        self,
        registry: Optional[ActionRegistry] = None,
        default_timeout: float = 30.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self.registry = registry or self._create_default_registry()
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._history: List[ActionResult] = []
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _create_default_registry(self) -> ActionRegistry:
        registry = ActionRegistry()
        registry.register(ToolAction())
        registry.register(CodeAction())
        registry.register(FileAction())
        registry.register(WebAction())
        registry.register(ThinkAction())
        return registry

    def execute(self, action: Action) -> ActionResult:
        """Execute an action with timeout and retry support."""
        handler = self.registry.get_handler(action)
        
        if not handler:
            result = ActionResult(
                action_id=action.action_id,
                status=ActionStatus.FAILED,
                error=f"No handler for action type: {action.action_type}",
            )
            self._history.append(result)
            return result
        
        timeout = action.timeout or self.default_timeout
        max_retries = action.max_retries or self.max_retries
        
        for attempt in range(max_retries):
            try:
                future = self._executor.submit(handler.execute, action)
                result = future.result(timeout=timeout)
                
                if result.is_success:
                    self._history.append(result)
                    return result
                
                if attempt < max_retries - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Action {action.name} failed (attempt {attempt + 1}), "
                        f"retrying in {delay:.1f}s: {result.error}"
                    )
                    time.sleep(delay)
                else:
                    self._history.append(result)
                    return result
                    
            except FuturesTimeoutError:
                result = ActionResult(
                    action_id=action.action_id,
                    status=ActionStatus.TIMEOUT,
                    error=f"Action timed out after {timeout}s",
                    execution_time=timeout,
                )
                if attempt == max_retries - 1:
                    self._history.append(result)
                    return result
                    
            except Exception as e:
                result = ActionResult(
                    action_id=action.action_id,
                    status=ActionStatus.FAILED,
                    error=str(e),
                    metadata={"traceback": traceback.format_exc()},
                )
                if attempt == max_retries - 1:
                    self._history.append(result)
                    return result
        
        return result

    def _calculate_backoff(self, attempt: int) -> float:
        import random
        delay = self.retry_base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.1 * delay)
        return min(delay + jitter, 60.0)

    async def execute_async(self, action: Action) -> ActionResult:
        """Execute an action asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, action)

    async def execute_batch(self, actions: List[Action]) -> List[ActionResult]:
        """Execute multiple actions concurrently."""
        tasks = [self.execute_async(action) for action in actions]
        return await asyncio.gather(*tasks)

    def get_history(self) -> List[ActionResult]:
        return self._history.copy()

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._history)
        if total == 0:
            return {"total": 0, "success_rate": 0.0}
        
        success = sum(1 for r in self._history if r.is_success)
        failed = sum(1 for r in self._history if r.status == ActionStatus.FAILED)
        timeout = sum(1 for r in self._history if r.status == ActionStatus.TIMEOUT)
        avg_time = sum(r.execution_time for r in self._history) / total
        
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "timeout": timeout,
            "success_rate": success / total,
            "avg_execution_time": avg_time,
        }

    def clear_history(self) -> None:
        self._history.clear()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"ActionExecutor(total={stats['total']}, success_rate={stats['success_rate']:.1%})"


def create_action(
    action_type: Union[str, ActionType],
    name: str,
    **parameters: Any,
) -> Action:
    """Factory function to create actions."""
    if isinstance(action_type, str):
        action_type = ActionType(action_type)
    return Action(action_type=action_type, name=name, parameters=parameters)
