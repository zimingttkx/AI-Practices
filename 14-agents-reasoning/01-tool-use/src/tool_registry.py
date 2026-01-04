"""
Tool Registry: Centralized Management for Agent Capabilities

Core Idea:
    The Tool Registry implements the Service Locator pattern for AI agent tools,
    providing a centralized repository for registering, discovering, and managing
    callable functions that extend an LLM's capabilities.

Mathematical Theory:
    The registry can be modeled as a labeled directed graph:

    $$G = (V, E, L)$$

    where:
    - $V$ is the set of tool nodes
    - $E \subseteq V \times V$ represents tool dependencies
    - $L: V \rightarrow 2^T$ maps tools to their tag sets $T$

    Tool discovery is a set intersection operation:

    $$\text{discover}(t) = \{v \in V : t \in L(v)\}$$

Problem Statement:
    AI agents require access to multiple external tools, but:
    1. Tools must be discoverable by the LLM through schema descriptions
    2. Tools need lifecycle management (enable/disable)
    3. Tool access should be filterable by capability tags
    4. Multiple output formats (OpenAI, Anthropic) must be supported

    The registry solves these by providing a unified interface for tool management.

Algorithm Comparison:
    | Pattern              | Pros                        | Cons                      |
    |----------------------|-----------------------------|---------------------------|
    | Service Locator      | Centralized, discoverable   | Global state concerns     |
    | Dependency Injection | Testable, explicit deps     | Complex wiring            |
    | Plugin Architecture  | Extensible, isolated        | Runtime overhead          |
    | Direct References    | Simple, fast                | Tight coupling            |

Complexity:
    - Registration: O(t) where t = number of tags
    - Lookup by name: O(1) hash table access
    - Lookup by tag: O(n) where n = tools with tag
    - Schema export: O(n * p) where p = avg parameters per tool

Summary:
    This module provides Tool and ToolRegistry classes for managing agent
    capabilities. Tools wrap Python functions with metadata for LLM consumption.
    The registry supports decorator-based registration, tag-based filtering,
    and multi-format schema export.

References:
    - Service Locator Pattern: Fowler, PoEAA
    - OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
    - Anthropic Tool Use: https://docs.anthropic.com/claude/docs/tool-use
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
    overload,
)

from .function_calling import (
    FunctionDefinition,
    FunctionParameter,
    ParameterType,
    create_function_schema,
)

# Type variable for generic callable preservation
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class Tool:
    """Executable tool with metadata for LLM-based invocation.

    Core Idea:
        Wraps a Python callable with rich metadata enabling LLMs to understand
        when and how to invoke the tool, while providing runtime controls
        like enable/disable and confirmation requirements.

    Attributes:
        name: Unique identifier for tool invocation.
        description: Natural language explanation for LLM context.
        func: The underlying Python callable.
        parameters: List of parameter specifications.
        returns: Description of return value (optional).
        tags: Classification labels for filtering and organization.
        enabled: Runtime toggle for tool availability.
        requires_confirmation: Flag for sensitive operations requiring user approval.

    Example:
        >>> def search(query: str, limit: int = 10) -> str:
        ...     '''Search the web for information.'''
        ...     return f"Results for: {query}"
        >>> tool = Tool(
        ...     name="web_search",
        ...     description="Search the web for information",
        ...     func=search,
        ...     parameters=[...],
        ...     tags=["search", "web"]
        ... )
        >>> tool("python tutorials")
        'Results for: python tutorials'
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters: List[FunctionParameter] = field(default_factory=list)
    returns: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        """Validate tool configuration after initialization."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid tool name: '{self.name}'")
        if not callable(self.func):
            raise TypeError(f"Tool func must be callable, got {type(self.func)}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the underlying function with runtime checks.

        Raises:
            RuntimeError: If tool is disabled.

        Returns:
            Result of the underlying function call.
        """
        if not self.enabled:
            raise RuntimeError(f"Tool '{self.name}' is disabled")
        return self.func(*args, **kwargs)

    def to_function_definition(self) -> FunctionDefinition:
        """Convert to FunctionDefinition for schema generation.

        Returns:
            FunctionDefinition with this tool's metadata.
        """
        return FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            returns=self.returns,
        )

    def to_openai_schema(self) -> Dict[str, Any]:
        """Export as OpenAI function calling format.

        Returns:
            Dictionary conforming to OpenAI's tool specification.
        """
        return self.to_function_definition().to_openai_schema()

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Export as Anthropic tool use format.

        Returns:
            Dictionary conforming to Anthropic's tool specification.
        """
        return self.to_function_definition().to_anthropic_schema()

    def get_signature(self) -> str:
        """Generate human-readable function signature.

        Returns:
            String representation like "tool_name(param1: type, param2: type = default)"
        """
        param_strs: List[str] = []
        for param in self.parameters:
            sig = f"{param.name}: {param.type.value}"
            if not param.required:
                sig += f" = {param.default!r}"
            param_strs.append(sig)
        return f"{self.name}({', '.join(param_strs)})"

    def __repr__(self) -> str:
        """Provide informative string representation."""
        status = "enabled" if self.enabled else "disabled"
        tags_str = f", tags={self.tags}" if self.tags else ""
        return f"Tool({self.name}, {status}{tags_str})"


class ToolRegistry:
    """Centralized repository for tool registration and discovery.

    Core Idea:
        Implements the Service Locator pattern, providing a single point of
        access for all registered tools with support for filtering, lifecycle
        management, and multi-format schema export.

    Design Patterns:
        - Service Locator: Centralized tool discovery
        - Registry: Named object storage and retrieval
        - Decorator: Convenient registration syntax

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access,
        external synchronization is required.

    Example:
        >>> registry = ToolRegistry()
        >>> @registry.register(tags=["math"])
        ... def add(a: int, b: int) -> int:
        ...     '''Add two numbers.'''
        ...     return a + b
        >>> registry.get("add")(1, 2)
        3
        >>> registry.list_names()
        ['add']
    """

    __slots__ = ("_tools", "_tag_index")

    def __init__(self) -> None:
        """Initialize empty registry with tag index."""
        self._tools: Dict[str, Tool] = {}
        self._tag_index: Dict[str, List[str]] = {}

    @overload
    def register(self, func: F) -> F: ...

    @overload
    def register(
        self,
        func: None = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        requires_confirmation: bool = False,
    ) -> Callable[[F], F]: ...

    def register(
        self,
        func: Optional[F] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        requires_confirmation: bool = False,
    ) -> Union[F, Callable[[F], F]]:
        """Register a function as a tool via decorator.

        Supports both bare decorator (@registry.register) and parameterized
        decorator (@registry.register(name="custom")) syntax.

        Args:
            func: Function to register (provided automatically when used as bare decorator).
            name: Override tool name (defaults to function name).
            description: Override description (defaults to docstring).
            tags: Classification tags for filtering.
            requires_confirmation: Mark as requiring user confirmation.

        Returns:
            Decorated function (preserves original callable).

        Raises:
            ValueError: If tool with same name already registered.

        Example:
            >>> @registry.register
            ... def simple_tool(x: int) -> int:
            ...     '''A simple tool.'''
            ...     return x * 2

            >>> @registry.register(name="custom", tags=["utility"])
            ... def another_tool(x: int) -> int:
            ...     return x + 1
        """
        def decorator(f: F) -> F:
            tool = self._create_tool_from_function(
                f,
                name=name,
                description=description,
                tags=tags,
                requires_confirmation=requires_confirmation,
            )
            self._register_tool(tool)

            @wraps(f)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return f(*args, **kwargs)

            # Attach tool reference for introspection
            wrapper._tool = tool  # type: ignore[attr-defined]
            return wrapper  # type: ignore[return-value]

        if func is not None:
            return decorator(func)
        return decorator

    def register_tool(self, tool: Tool) -> None:
        """Directly register a pre-constructed Tool object.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If tool with same name already registered.
        """
        self._register_tool(tool)

    def _create_tool_from_function(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        requires_confirmation: bool = False,
    ) -> Tool:
        """Create Tool from function using introspection."""
        func_def = create_function_schema(func, description)

        return Tool(
            name=name or func_def.name,
            description=description or func_def.description,
            func=func,
            parameters=func_def.parameters,
            returns=func_def.returns,
            tags=tags or [],
            requires_confirmation=requires_confirmation,
        )

    def _register_tool(self, tool: Tool) -> None:
        """Internal registration with index maintenance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool

        # Update tag index
        for tag in tool.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            self._tag_index[tag].append(tool.name)

    def unregister(self, name: str) -> Optional[Tool]:
        """Remove tool from registry.

        Args:
            name: Tool name to unregister.

        Returns:
            Removed Tool if found, None otherwise.
        """
        tool = self._tools.pop(name, None)
        if tool is not None:
            # Clean up tag index
            for tag in tool.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].remove(name)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """Retrieve tool by name.

        Args:
            name: Tool identifier.

        Returns:
            Tool if found, None otherwise.

        Complexity:
            Time: O(1) hash table lookup
        """
        return self._tools.get(name)

    def __getitem__(self, name: str) -> Tool:
        """Dictionary-style access with KeyError on missing.

        Args:
            name: Tool identifier.

        Returns:
            Tool instance.

        Raises:
            KeyError: If tool not found.
        """
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found")
        return tool

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        """Iterate over all registered tools."""
        return iter(self._tools.values())

    def list_tools(
        self,
        tags: Optional[List[str]] = None,
        enabled_only: bool = True,
    ) -> List[Tool]:
        """List tools with optional filtering.

        Args:
            tags: Filter by tags (returns tools matching ANY tag).
            enabled_only: Only return enabled tools.

        Returns:
            List of matching Tool instances.

        Complexity:
            Time: O(n) where n = number of tools
        """
        tools = list(self._tools.values())

        if enabled_only:
            tools = [t for t in tools if t.enabled]

        if tags:
            tag_set = set(tags)
            tools = [t for t in tools if tag_set & set(t.tags)]

        return tools

    def list_names(self) -> List[str]:
        """Get all registered tool names.

        Returns:
            List of tool name strings.
        """
        return list(self._tools.keys())

    def list_tags(self) -> List[str]:
        """Get all unique tags across registered tools.

        Returns:
            List of unique tag strings.
        """
        return list(self._tag_index.keys())

    def get_by_tag(self, tag: str) -> List[Tool]:
        """Retrieve all tools with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of tools having the specified tag.

        Complexity:
            Time: O(k) where k = tools with tag
        """
        tool_names = self._tag_index.get(tag, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def enable(self, name: str) -> bool:
        """Enable a tool by name.

        Args:
            name: Tool identifier.

        Returns:
            True if tool was found and enabled, False otherwise.
        """
        tool = self._tools.get(name)
        if tool is not None:
            tool.enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a tool by name.

        Args:
            name: Tool identifier.

        Returns:
            True if tool was found and disabled, False otherwise.
        """
        tool = self._tools.get(name)
        if tool is not None:
            tool.enabled = False
            return True
        return False

    def to_openai_tools(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Export all tools in OpenAI function calling format.

        Args:
            enabled_only: Only export enabled tools.

        Returns:
            List of OpenAI-formatted tool schemas.

        Complexity:
            Time: O(n * p) where n = tools, p = avg parameters
        """
        tools = self.list_tools(enabled_only=enabled_only)
        return [t.to_openai_schema() for t in tools]

    def to_anthropic_tools(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Export all tools in Anthropic tool use format.

        Args:
            enabled_only: Only export enabled tools.

        Returns:
            List of Anthropic-formatted tool schemas.

        Complexity:
            Time: O(n * p) where n = tools, p = avg parameters
        """
        tools = self.list_tools(enabled_only=enabled_only)
        return [t.to_anthropic_schema() for t in tools]

    def get_tool_descriptions(self) -> str:
        """Generate human-readable tool documentation.

        Useful for constructing system prompts that describe available tools.

        Returns:
            Formatted string describing all enabled tools.
        """
        lines = ["Available tools:"]
        for tool in self.list_tools():
            lines.append(f"\n- {tool.get_signature()}")
            lines.append(f"  Description: {tool.description}")
            if tool.tags:
                lines.append(f"  Tags: {', '.join(tool.tags)}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()
        self._tag_index.clear()

    def __repr__(self) -> str:
        """Provide informative string representation."""
        enabled = sum(1 for t in self._tools.values() if t.enabled)
        return f"ToolRegistry({len(self._tools)} tools, {enabled} enabled)"


# Global default registry instance
_default_registry: ToolRegistry = ToolRegistry()


@overload
def tool(func: F) -> F: ...


@overload
def tool(
    func: None = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    requires_confirmation: bool = False,
    registry: Optional[ToolRegistry] = None,
) -> Callable[[F], F]: ...


def tool(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    requires_confirmation: bool = False,
    registry: Optional[ToolRegistry] = None,
) -> Union[F, Callable[[F], F]]:
    """Global decorator for tool registration.

    Convenience decorator that registers tools to the global default registry
    (or a specified registry). Supports both bare and parameterized usage.

    Args:
        func: Function to register (auto-provided for bare decorator).
        name: Override tool name.
        description: Override description.
        tags: Classification tags.
        requires_confirmation: Mark as requiring confirmation.
        registry: Target registry (defaults to global registry).

    Returns:
        Decorated function.

    Example:
        >>> @tool
        ... def calculator(expression: str) -> float:
        ...     '''Evaluate a math expression.'''
        ...     return eval(expression)

        >>> @tool(name="web_search", tags=["search"])
        ... def search(query: str) -> str:
        ...     '''Search the web.'''
        ...     return f"Results for: {query}"
    """
    target_registry = registry or _default_registry
    return target_registry.register(
        func,
        name=name,
        description=description,
        tags=tags,
        requires_confirmation=requires_confirmation,
    )


def get_default_registry() -> ToolRegistry:
    """Access the global default tool registry.

    Returns:
        The singleton default ToolRegistry instance.
    """
    return _default_registry


def reset_default_registry() -> None:
    """Reset the global default registry to empty state.

    Useful for testing or reinitializing the tool environment.
    """
    global _default_registry
    _default_registry = ToolRegistry()


__all__ = [
    "Tool",
    "ToolRegistry",
    "tool",
    "get_default_registry",
    "reset_default_registry",
]


if __name__ == "__main__":
    # Self-test: Verify core functionality
    registry = ToolRegistry()

    @registry.register(tags=["math", "utility"])
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @registry.register(name="multiply", tags=["math"])
    def mult(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    # Test registration
    assert "add" in registry
    assert "multiply" in registry
    assert len(registry) == 2

    # Test invocation
    assert registry["add"](2, 3) == 5
    assert registry["multiply"](4, 5) == 20

    # Test tag filtering
    math_tools = registry.get_by_tag("math")
    assert len(math_tools) == 2

    utility_tools = registry.get_by_tag("utility")
    assert len(utility_tools) == 1

    # Test enable/disable
    registry.disable("add")
    assert not registry["add"].enabled
    try:
        registry["add"](1, 2)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

    registry.enable("add")
    assert registry["add"](1, 2) == 3

    # Test schema export
    openai_tools = registry.to_openai_tools()
    assert len(openai_tools) == 2
    assert all(t["type"] == "function" for t in openai_tools)

    anthropic_tools = registry.to_anthropic_tools()
    assert len(anthropic_tools) == 2
    assert all("input_schema" in t for t in anthropic_tools)

    # Test unregister
    registry.unregister("add")
    assert "add" not in registry
    assert len(registry) == 1

    # Test global decorator
    reset_default_registry()

    @tool(tags=["test"])
    def global_tool(x: int) -> int:
        """A global tool."""
        return x * 2

    global_reg = get_default_registry()
    assert "global_tool" in global_reg
    assert global_reg["global_tool"](5) == 10

    print("All self-tests passed.")
