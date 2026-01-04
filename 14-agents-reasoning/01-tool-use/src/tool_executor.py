"""
Tool Executor: Safe Execution Engine for AI Agent Tools

Core Idea:
    The Tool Executor implements a controlled execution environment for AI agent
    tools, providing isolation, timeout management, retry logic, and comprehensive
    result tracking. It acts as the runtime layer between LLM-generated function
    calls and actual Python function execution.

Mathematical Theory:
    Tool execution can be modeled as a state machine with probabilistic transitions:

    $$M = (S, \Sigma, \delta, s_0, F)$$

    where:
    - $S = \{pending, running, success, error, timeout, cancelled\}$ is the state set
    - $\Sigma$ is the input alphabet (tool invocations)
    - $\delta: S \times \Sigma \rightarrow S$ is the transition function
    - $s_0 = pending$ is the initial state
    - $F = \{success, error, timeout, cancelled\}$ are final states

    The retry mechanism follows exponential backoff:

    $$t_n = t_0 \cdot b^n$$

    where $t_0$ is the initial delay, $b$ is the backoff factor, and $n$ is the attempt.

    Execution time distribution for timeout analysis:

    $$P(T > t) = e^{-\lambda t}$$ (exponential model)

Problem Statement:
    Direct function execution in AI agents poses several challenges:
    1. Unbounded execution time can block the entire system
    2. Exceptions may crash the agent without proper handling
    3. No visibility into execution history or performance metrics
    4. Lack of retry mechanisms for transient failures
    5. No hooks for logging, monitoring, or access control

    The executor solves these by wrapping all tool calls in a managed environment.

Algorithm Comparison:
    | Strategy           | Pros                        | Cons                      |
    |--------------------|-----------------------------|-----------------------------|
    | Thread Pool        | Timeout support, isolation  | GIL limitations             |
    | Process Pool       | True parallelism, isolation | IPC overhead, serialization |
    | Async/Await        | Efficient I/O, scalable     | Requires async tools        |
    | Direct Call        | Simple, fast                | No timeout, no isolation    |

Complexity:
    - Single execution: O(f) where f = tool function complexity
    - Batch sequential: O(n * f) where n = number of calls
    - Batch parallel: O(max(f_i)) with thread pool overhead
    - History lookup: O(h) where h = history size
    - Statistics: O(h) for aggregation

Summary:
    This module provides ExecutionStatus, ExecutionResult, ExecutionContext, and
    ToolExecutor classes for safe tool execution. Features include timeout control,
    retry with backoff, parallel batch execution, lifecycle hooks, and execution
    history with statistics.

References:
    - Thread Pool Executor: Python concurrent.futures documentation
    - Retry Patterns: AWS Architecture Blog - Exponential Backoff
    - Circuit Breaker: Fowler, Release It! (2nd ed.)
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .function_calling import FunctionCall
    from .tool_registry import Tool, ToolRegistry


class ExecutionStatus(str, Enum):
    """Enumeration of possible tool execution states.

    Core Idea:
        Represents the finite state machine states for tool execution lifecycle,
        enabling precise tracking and conditional logic based on execution outcome.

    State Transitions:
        PENDING -> RUNNING -> SUCCESS
                          -> ERROR
                          -> TIMEOUT
                          -> CANCELLED

    Attributes:
        SUCCESS: Tool completed without errors.
        ERROR: Tool raised an exception during execution.
        TIMEOUT: Execution exceeded the allowed time limit.
        CANCELLED: Execution was cancelled before completion.
        PENDING: Execution has not yet started.
        RUNNING: Execution is currently in progress.
    """

    SUCCESS: Final[str] = "success"
    ERROR: Final[str] = "error"
    TIMEOUT: Final[str] = "timeout"
    CANCELLED: Final[str] = "cancelled"
    PENDING: Final[str] = "pending"
    RUNNING: Final[str] = "running"


@dataclass(slots=True)
class ExecutionResult:
    """Immutable record of a single tool execution.

    Core Idea:
        Encapsulates all information about a tool execution, including outcome,
        timing, and context. Provides a complete audit trail for debugging,
        monitoring, and LLM feedback.

    Mathematical Model:
        An execution result $R$ is a tuple:
        $$R = (s, o, e, t, n, a, id, ts)$$
        where:
        - $s \in S$ is the final state
        - $o$ is the output value (if successful)
        - $e$ is the error message (if failed)
        - $t \in \mathbb{R}^+$ is the execution time
        - $n$ is the tool name
        - $a$ is the arguments dictionary
        - $id$ is the unique call identifier
        - $ts$ is the timestamp

    Attributes:
        status: Final execution state from ExecutionStatus enum.
        output: Return value from successful execution (None if failed).
        error: Error message string (None if successful).
        execution_time: Wall-clock time in seconds.
        tool_name: Name of the executed tool.
        arguments: Dictionary of arguments passed to the tool.
        call_id: UUID for tracking and correlation.
        timestamp: When execution completed.

    Example:
        >>> result = ExecutionResult(
        ...     status=ExecutionStatus.SUCCESS,
        ...     output=42,
        ...     tool_name="calculator",
        ...     arguments={"expression": "6 * 7"},
        ...     execution_time=0.001
        ... )
        >>> result.is_success
        True
        >>> result.to_message()
        "Tool 'calculator' executed successfully.\\nResult: 42"
    """

    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        """Check if execution completed successfully.

        Returns:
            True if status is SUCCESS, False otherwise.

        Complexity:
            Time: O(1)
        """
        return self.status == ExecutionStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        """Check if execution failed (error or timeout).

        Returns:
            True if status is ERROR or TIMEOUT, False otherwise.

        Complexity:
            Time: O(1)
        """
        return self.status in (ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export.

        Returns:
            Dictionary with all fields, timestamp as ISO string.

        Complexity:
            Time: O(a) where a = number of arguments
            Space: O(a)
        """
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_message(self) -> str:
        """Format as human-readable message for LLM feedback.

        Returns:
            Formatted string describing execution outcome.

        Complexity:
            Time: O(1)
        """
        if self.is_success:
            return f"Tool '{self.tool_name}' executed successfully.\nResult: {self.output}"
        return f"Tool '{self.tool_name}' failed.\nError: {self.error}"

    def __repr__(self) -> str:
        """Provide informative string representation."""
        return (
            f"ExecutionResult({self.tool_name}, {self.status.value}, "
            f"time={self.execution_time:.3f}s)"
        )


@dataclass(slots=True)
class ExecutionContext:
    """Execution environment configuration for tool invocation.

    Core Idea:
        Provides a container for execution-time configuration including user
        identity, permissions, timeout settings, and shared variables. Enables
        context-aware tool execution with proper access control.

    Security Model:
        Permission checking follows a whitelist approach:
        $$\text{allowed}(p) = (p \in P) \lor (* \in P)$$
        where $P$ is the permission set and $*$ is the wildcard permission.

    Attributes:
        user_id: Identifier for the requesting user (for audit trails).
        session_id: Identifier for the current session (for correlation).
        variables: Shared key-value store accessible to tools.
        permissions: List of granted permission strings.
        max_retries: Maximum retry attempts for failed executions.
        timeout: Maximum execution time in seconds.
        sandbox: Whether to execute in restricted sandbox mode.

    Example:
        >>> context = ExecutionContext(
        ...     user_id="user123",
        ...     permissions=["read", "write"],
        ...     timeout=10.0
        ... )
        >>> context.has_permission("read")
        True
        >>> context.has_permission("delete")
        False
    """

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout: float = 30.0
    sandbox: bool = False

    def has_permission(self, permission: str) -> bool:
        """Check if a specific permission is granted.

        Args:
            permission: Permission string to check.

        Returns:
            True if permission is granted or wildcard (*) is present.

        Complexity:
            Time: O(p) where p = number of permissions
        """
        return permission in self.permissions or "*" in self.permissions

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Retrieve a context variable by name.

        Args:
            name: Variable name to look up.
            default: Value to return if variable not found.

        Returns:
            Variable value or default.

        Complexity:
            Time: O(1) hash table lookup
        """
        return self.variables.get(name, default)

    def set_variable(self, name: str, value: Any) -> None:
        """Store a context variable.

        Args:
            name: Variable name.
            value: Value to store.

        Complexity:
            Time: O(1) hash table insertion
        """
        self.variables[name] = value

    def __repr__(self) -> str:
        """Provide informative string representation."""
        return (
            f"ExecutionContext(user={self.user_id}, "
            f"permissions={len(self.permissions)}, timeout={self.timeout}s)"
        )


class ToolExecutor:
    """Safe execution engine for AI agent tools.

    Core Idea:
        Provides a managed execution environment that wraps tool invocations
        with timeout control, error handling, retry logic, and comprehensive
        logging. Acts as the bridge between LLM-generated function calls and
        actual Python function execution.

    Design Patterns:
        - Command Pattern: Encapsulates tool invocations as executable objects
        - Observer Pattern: Lifecycle hooks for monitoring and logging
        - Retry Pattern: Automatic retry with configurable backoff

    Thread Safety:
        Uses ThreadPoolExecutor for isolation. Multiple concurrent executions
        are safe, but individual tool functions must be thread-safe.

    Attributes:
        registry: ToolRegistry containing available tools.
        default_timeout: Default execution timeout in seconds.
        max_retries: Default maximum retry attempts.
        retry_delay: Default delay between retries in seconds.

    Example:
        >>> registry = ToolRegistry()
        >>> @registry.register
        ... def add(a: int, b: int) -> int:
        ...     '''Add two numbers.'''
        ...     return a + b
        >>> executor = ToolExecutor(registry)
        >>> result = executor.execute("add", {"a": 1, "b": 2})
        >>> result.output
        3
    """

    __slots__ = (
        "registry",
        "default_timeout",
        "max_retries",
        "retry_delay",
        "_executor",
        "_execution_history",
        "_hooks",
    )

    def __init__(
        self,
        registry: "ToolRegistry",
        default_timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_workers: int = 10,
    ) -> None:
        """Initialize the tool executor.

        Args:
            registry: ToolRegistry containing available tools.
            default_timeout: Default execution timeout in seconds.
            max_retries: Default maximum retry attempts.
            retry_delay: Default delay between retries in seconds.
            max_workers: Maximum concurrent executions in thread pool.

        Complexity:
            Time: O(1)
            Space: O(w) where w = max_workers (thread pool allocation)
        """
        self.registry = registry
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._execution_history: List[ExecutionResult] = []
        self._hooks: Dict[str, List[Callable[..., None]]] = {
            "before_execute": [],
            "after_execute": [],
            "on_error": [],
        }

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a tool by name with given arguments.

        Core Algorithm:
            1. Validate tool existence and enabled status
            2. Run before_execute hooks
            3. Submit to thread pool with timeout
            4. Capture result or exception
            5. Run after_execute/on_error hooks
            6. Record in history and return

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments to pass to the tool.
            context: Optional execution context with permissions and config.
            timeout: Optional timeout override in seconds.

        Returns:
            ExecutionResult containing status, output/error, and metadata.

        Complexity:
            Time: O(f + h) where f = tool function time, h = hook time
            Space: O(a) where a = arguments size
        """
        context = context or ExecutionContext()
        timeout = timeout or context.timeout or self.default_timeout

        # Import here to avoid circular dependency
        from .tool_registry import Tool

        # Validate tool existence
        tool: Optional[Tool] = self.registry.get(tool_name)
        if tool is None:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
                arguments=arguments,
            )

        # Validate tool is enabled
        if not tool.enabled:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Tool '{tool_name}' is disabled",
                tool_name=tool_name,
                arguments=arguments,
            )

        # Run before_execute hooks
        self._run_hooks("before_execute", tool, arguments, context)

        # Execute with timeout
        start_time = time.time()
        result = self._execute_with_timeout(tool, arguments, timeout)
        result.execution_time = time.time() - start_time
        result.tool_name = tool_name
        result.arguments = arguments

        # Run post-execution hooks
        self._run_hooks("after_execute", tool, arguments, context, result)
        if result.is_error:
            self._run_hooks("on_error", tool, arguments, context, result)

        # Record in history
        self._execution_history.append(result)

        return result

    def _execute_with_timeout(
        self,
        tool: "Tool",
        arguments: Dict[str, Any],
        timeout: float,
    ) -> ExecutionResult:
        """Execute tool function with timeout protection.

        Uses ThreadPoolExecutor for timeout enforcement. The tool function
        runs in a separate thread, allowing the main thread to enforce
        the timeout deadline.

        Args:
            tool: Tool instance to execute.
            arguments: Arguments to pass to tool function.
            timeout: Maximum execution time in seconds.

        Returns:
            ExecutionResult with SUCCESS, ERROR, or TIMEOUT status.

        Complexity:
            Time: O(min(f, timeout)) where f = function execution time
        """
        try:
            future = self._executor.submit(tool.func, **arguments)
            output = future.result(timeout=timeout)
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output,
            )
        except FuturesTimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timed out after {timeout} seconds",
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            )

    def execute_call(
        self,
        call: "FunctionCall",
        context: Optional[ExecutionContext] = None,
    ) -> ExecutionResult:
        """Execute a FunctionCall object directly.

        Convenience method that extracts name and arguments from a FunctionCall
        and delegates to execute().

        Args:
            call: FunctionCall object containing name and arguments.
            context: Optional execution context.

        Returns:
            ExecutionResult with call_id from the FunctionCall if provided.

        Complexity:
            Time: O(f) where f = tool function complexity
        """
        result = self.execute(call.name, call.arguments, context)
        result.call_id = call.id or result.call_id
        return result

    def execute_batch(
        self,
        calls: List["FunctionCall"],
        context: Optional[ExecutionContext] = None,
        parallel: bool = False,
    ) -> List[ExecutionResult]:
        """Execute multiple function calls in batch.

        Args:
            calls: List of FunctionCall objects to execute.
            context: Shared execution context for all calls.
            parallel: If True, execute calls concurrently using thread pool.

        Returns:
            List of ExecutionResult objects in same order as input calls.

        Complexity:
            Sequential: O(n * f) where n = calls, f = avg function time
            Parallel: O(max(f_i)) with thread pool overhead
        """
        if parallel:
            return self._execute_parallel(calls, context)
        return [self.execute_call(call, context) for call in calls]

    def _execute_parallel(
        self,
        calls: List["FunctionCall"],
        context: Optional[ExecutionContext] = None,
    ) -> List[ExecutionResult]:
        """Execute multiple calls concurrently using thread pool.

        Maintains order correspondence between input calls and output results
        by using a dictionary to track futures.

        Args:
            calls: List of FunctionCall objects.
            context: Shared execution context.

        Returns:
            List of ExecutionResult objects preserving input order.

        Complexity:
            Time: O(max(f_i)) where f_i = individual function times
            Space: O(n) for futures dictionary
        """
        import concurrent.futures

        results: List[ExecutionResult] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.execute_call, call, context): i
                for i, call in enumerate(calls)
            }
            # Collect results maintaining order
            result_map: Dict[int, ExecutionResult] = {}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                result_map[idx] = future.result()

            results = [result_map[i] for i in range(len(calls))]

        return results

    def execute_with_retry(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute tool with automatic retry on failure.

        Implements simple retry with fixed delay. For production use,
        consider exponential backoff with jitter.

        Retry Formula:
            $$\text{total\_attempts} = 1 + \text{max\_retries}$$

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            context: Optional execution context.
            max_retries: Maximum retry attempts (default: self.max_retries).
            retry_delay: Delay between retries in seconds.

        Returns:
            ExecutionResult from first successful attempt or last failed attempt.

        Complexity:
            Time: O(r * f) where r = retries, f = function time
        """
        max_retries = max_retries if max_retries is not None else self.max_retries
        retry_delay = retry_delay if retry_delay is not None else self.retry_delay

        last_result: Optional[ExecutionResult] = None
        for attempt in range(max_retries + 1):
            result = self.execute(tool_name, arguments, context)
            if result.is_success:
                return result
            last_result = result
            if attempt < max_retries:
                time.sleep(retry_delay)

        return last_result  # type: ignore[return-value]

    async def execute_async(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute tool asynchronously.

        Wraps synchronous execute() in asyncio executor for non-blocking
        operation in async contexts.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            context: Optional execution context.
            timeout: Optional timeout override.

        Returns:
            ExecutionResult from the execution.

        Complexity:
            Time: O(f) where f = tool function complexity
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.execute(tool_name, arguments, context, timeout),
        )

    def add_hook(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:
        """Register a lifecycle hook callback.

        Available Events:
            - before_execute: Called before tool execution (tool, args, context)
            - after_execute: Called after execution (tool, args, context, result)
            - on_error: Called on failure (tool, args, context, result)

        Args:
            event: Event name to hook into.
            callback: Function to call when event fires.

        Raises:
            ValueError: If event name is not recognized.

        Complexity:
            Time: O(1) list append
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown event: {event}. Valid: {list(self._hooks.keys())}")
        self._hooks[event].append(callback)

    def remove_hook(self, event: str, callback: Callable[..., None]) -> bool:
        """Unregister a lifecycle hook callback.

        Args:
            event: Event name the callback was registered for.
            callback: The callback function to remove.

        Returns:
            True if callback was found and removed, False otherwise.

        Complexity:
            Time: O(h) where h = number of hooks for event
        """
        if event in self._hooks and callback in self._hooks[event]:
            self._hooks[event].remove(callback)
            return True
        return False

    def _run_hooks(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Execute all registered hooks for an event.

        Hook exceptions are caught and logged but do not propagate,
        ensuring hooks cannot break the main execution flow.

        Args:
            event: Event name to trigger.
            *args: Positional arguments to pass to callbacks.
            **kwargs: Keyword arguments to pass to callbacks.

        Complexity:
            Time: O(h * c) where h = hooks, c = callback complexity
        """
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Hook errors must not affect main execution

    def get_history(
        self,
        limit: Optional[int] = None,
        tool_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
    ) -> List[ExecutionResult]:
        """Retrieve execution history with optional filtering.

        Args:
            limit: Maximum number of results to return (most recent).
            tool_name: Filter by specific tool name.
            status: Filter by execution status.

        Returns:
            List of ExecutionResult objects matching criteria.

        Complexity:
            Time: O(h) where h = history size
            Space: O(min(h, limit))
        """
        history = self._execution_history

        if tool_name is not None:
            history = [r for r in history if r.tool_name == tool_name]
        if status is not None:
            history = [r for r in history if r.status == status]

        if limit is not None:
            history = history[-limit:]

        return history

    def clear_history(self) -> None:
        """Clear all execution history.

        Complexity:
            Time: O(1)
        """
        self._execution_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics from execution history.

        Statistics Computed:
            - total_executions: Total number of executions
            - success_count: Number of successful executions
            - error_count: Number of failed executions
            - success_rate: Ratio of successes to total
            - average_time: Mean execution time in seconds
            - total_time: Sum of all execution times

        Returns:
            Dictionary containing computed statistics.

        Complexity:
            Time: O(h) where h = history size
            Space: O(1)
        """
        total = len(self._execution_history)
        if total == 0:
            return {
                "total_executions": 0,
                "success_count": 0,
                "error_count": 0,
                "success_rate": 0.0,
                "average_time": 0.0,
                "total_time": 0.0,
            }

        success_count = sum(1 for r in self._execution_history if r.is_success)
        total_time = sum(r.execution_time for r in self._execution_history)

        return {
            "total_executions": total,
            "success_count": success_count,
            "error_count": total - success_count,
            "success_rate": success_count / total,
            "average_time": total_time / total,
            "total_time": total_time,
        }

    def shutdown(self) -> None:
        """Gracefully shutdown the executor thread pool.

        Waits for all pending executions to complete before returning.
        After shutdown, no new executions can be submitted.

        Complexity:
            Time: O(p) where p = pending executions
        """
        self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        """Provide informative string representation."""
        return (
            f"ToolExecutor(tools={len(self.registry)}, "
            f"history={len(self._execution_history)}, "
            f"timeout={self.default_timeout}s)"
        )


__all__ = [
    "ExecutionStatus",
    "ExecutionResult",
    "ExecutionContext",
    "ToolExecutor",
]


if __name__ == "__main__":
    # Self-test: Verify core functionality
    from .tool_registry import ToolRegistry

    registry = ToolRegistry()

    @registry.register(tags=["math"])
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @registry.register(tags=["math"])
    def divide(a: int, b: int) -> float:
        """Divide two numbers."""
        return a / b

    @registry.register(tags=["slow"])
    def slow_function(seconds: float) -> str:
        """Sleep for specified seconds."""
        import time
        time.sleep(seconds)
        return f"Slept for {seconds}s"

    executor = ToolExecutor(registry, default_timeout=5.0)

    # Test successful execution
    result = executor.execute("add", {"a": 2, "b": 3})
    assert result.is_success
    assert result.output == 5
    assert result.tool_name == "add"

    # Test error handling (division by zero)
    result = executor.execute("divide", {"a": 1, "b": 0})
    assert result.is_error
    assert result.status == ExecutionStatus.ERROR
    assert "ZeroDivisionError" in result.error

    # Test tool not found
    result = executor.execute("nonexistent", {})
    assert result.is_error
    assert "not found" in result.error

    # Test disabled tool
    registry.disable("add")
    result = executor.execute("add", {"a": 1, "b": 2})
    assert result.is_error
    assert "disabled" in result.error
    registry.enable("add")

    # Test timeout
    result = executor.execute("slow_function", {"seconds": 10.0}, timeout=0.1)
    assert result.status == ExecutionStatus.TIMEOUT

    # Test execution context
    context = ExecutionContext(
        user_id="test_user",
        permissions=["read", "write"],
        timeout=5.0,
    )
    assert context.has_permission("read")
    assert context.has_permission("write")
    assert not context.has_permission("delete")

    # Test wildcard permission
    admin_context = ExecutionContext(permissions=["*"])
    assert admin_context.has_permission("anything")

    # Test context variables
    context.set_variable("key", "value")
    assert context.get_variable("key") == "value"
    assert context.get_variable("missing", "default") == "default"

    # Test hooks
    hook_called = []

    def before_hook(tool, args, ctx):
        hook_called.append("before")

    def after_hook(tool, args, ctx, result):
        hook_called.append("after")

    executor.add_hook("before_execute", before_hook)
    executor.add_hook("after_execute", after_hook)

    executor.execute("add", {"a": 1, "b": 1})
    assert "before" in hook_called
    assert "after" in hook_called

    # Test hook removal
    assert executor.remove_hook("before_execute", before_hook)
    assert not executor.remove_hook("before_execute", before_hook)  # Already removed

    # Test history
    history = executor.get_history(limit=5)
    assert len(history) <= 5
    assert all(isinstance(r, ExecutionResult) for r in history)

    # Test history filtering
    success_history = executor.get_history(status=ExecutionStatus.SUCCESS)
    assert all(r.is_success for r in success_history)

    # Test statistics
    stats = executor.get_stats()
    assert "total_executions" in stats
    assert "success_rate" in stats
    assert stats["total_executions"] > 0

    # Test result serialization
    result = executor.execute("add", {"a": 5, "b": 5})
    result_dict = result.to_dict()
    assert result_dict["status"] == "success"
    assert result_dict["output"] == 10

    # Test result message
    message = result.to_message()
    assert "successfully" in message
    assert "10" in message

    # Test batch execution
    from .function_calling import FunctionCall

    calls = [
        FunctionCall(name="add", arguments={"a": 1, "b": 1}),
        FunctionCall(name="add", arguments={"a": 2, "b": 2}),
        FunctionCall(name="add", arguments={"a": 3, "b": 3}),
    ]

    # Sequential batch
    results = executor.execute_batch(calls, parallel=False)
    assert len(results) == 3
    assert results[0].output == 2
    assert results[1].output == 4
    assert results[2].output == 6

    # Parallel batch
    results = executor.execute_batch(calls, parallel=True)
    assert len(results) == 3
    assert set(r.output for r in results) == {2, 4, 6}

    # Test clear history
    executor.clear_history()
    assert len(executor.get_history()) == 0

    # Test shutdown
    executor.shutdown()

    print("All self-tests passed.")
