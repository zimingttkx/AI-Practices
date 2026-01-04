"""
Function Calling: LLM-to-External-World Interface Protocol

Core Idea:
    Function Calling enables Large Language Models to interact with external systems
    by generating structured JSON outputs that conform to predefined schemas, bridging
    the gap between natural language understanding and programmatic execution.

Mathematical Theory:
    The function calling mechanism can be formalized as a mapping:

    $$f: \mathcal{T} \times \mathcal{S} \rightarrow \mathcal{C}$$

    where:
    - $\mathcal{T}$ is the space of natural language text inputs
    - $\mathcal{S} = \{s_1, s_2, ..., s_n\}$ is the set of function schemas
    - $\mathcal{C}$ is the space of valid function calls

    The LLM learns a conditional distribution:

    $$P(c | t, S) = \prod_{i=1}^{|c|} P(c_i | c_{<i}, t, S)$$

    where $c$ is the generated function call token sequence.

Problem Statement:
    LLMs are fundamentally text generators with no native capability to:
    1. Execute code or system commands
    2. Access real-time information (weather, stock prices, etc.)
    3. Interact with databases or external APIs
    4. Perform precise mathematical computations

    Function Calling solves this by having the LLM output structured "intent"
    that external systems can parse and execute.

Algorithm Comparison:
    | Approach           | Pros                      | Cons                        |
    |--------------------|---------------------------|-----------------------------|
    | Function Calling   | Type-safe, validated      | Requires schema definition  |
    | ReAct Prompting    | Flexible, no schema       | Parsing errors, unreliable  |
    | Code Generation    | Powerful, expressive      | Security risks, sandboxing  |
    | Plugin Systems     | Extensible, modular       | Complex integration         |

Complexity:
    - Schema Generation: O(p) where p = number of parameters
    - JSON Parsing: O(n) where n = output length
    - Validation: O(p * v) where v = validation rules per parameter
    - Space: O(s) where s = schema size

Summary:
    This module implements the Function Calling protocol with support for both
    OpenAI and Anthropic formats. It provides schema generation from Python
    functions, multi-format parsing, and comprehensive validation.

References:
    - OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
    - Anthropic Tool Use: https://docs.anthropic.com/claude/docs/tool-use
    - JSON Schema Spec: https://json-schema.org/specification.html
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_type_hints,
)


class ParameterType(str, Enum):
    """JSON Schema primitive types for function parameters.

    Maps directly to JSON Schema type keywords as defined in the specification.
    Each type corresponds to a Python native type for validation purposes.

    Attributes:
        STRING: UTF-8 encoded text (Python str)
        INTEGER: Whole numbers without fractional component (Python int)
        NUMBER: IEEE 754 double-precision floating-point (Python float)
        BOOLEAN: Logical true/false values (Python bool)
        ARRAY: Ordered sequence of elements (Python list)
        OBJECT: Key-value mapping (Python dict)
    """

    STRING: Final[str] = "string"
    INTEGER: Final[str] = "integer"
    NUMBER: Final[str] = "number"
    BOOLEAN: Final[str] = "boolean"
    ARRAY: Final[str] = "array"
    OBJECT: Final[str] = "object"


# Type validation mapping: ParameterType -> (type_check_function, type_name)
_TYPE_VALIDATORS: Final[Dict[ParameterType, Tuple[Callable[[Any], bool], str]]] = {
    ParameterType.STRING: (lambda v: isinstance(v, str), "string"),
    ParameterType.INTEGER: (
        lambda v: isinstance(v, int) and not isinstance(v, bool),
        "integer",
    ),
    ParameterType.NUMBER: (
        lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "number",
    ),
    ParameterType.BOOLEAN: (lambda v: isinstance(v, bool), "boolean"),
    ParameterType.ARRAY: (lambda v: isinstance(v, list), "array"),
    ParameterType.OBJECT: (lambda v: isinstance(v, dict), "object"),
}

# Python type to ParameterType mapping for automatic schema generation
_PYTHON_TYPE_MAP: Final[Dict[Type, ParameterType]] = {
    str: ParameterType.STRING,
    int: ParameterType.INTEGER,
    float: ParameterType.NUMBER,
    bool: ParameterType.BOOLEAN,
    list: ParameterType.ARRAY,
    dict: ParameterType.OBJECT,
}


@dataclass(frozen=False, slots=True)
class FunctionParameter:
    """Immutable specification for a single function parameter.

    Core Idea:
        Encapsulates all metadata required to validate and document a function
        parameter, enabling automatic JSON Schema generation and type checking.

    Attributes:
        name: Parameter identifier (must be valid Python identifier).
        type: JSON Schema type classification.
        description: Human-readable explanation for LLM context.
        required: Whether the parameter must be provided (default: True).
        enum: Exhaustive list of valid values (optional constraint).
        default: Fallback value when parameter is omitted.
        items: Schema for array element types (when type is ARRAY).
        properties: Schema for object properties (when type is OBJECT).

    Example:
        >>> param = FunctionParameter(
        ...     name="temperature",
        ...     type=ParameterType.NUMBER,
        ...     description="Sampling temperature between 0 and 2",
        ...     required=False,
        ...     default=1.0
        ... )
        >>> param.to_schema()
        {'type': 'number', 'description': 'Sampling temperature...', 'default': 1.0}
    """

    name: str
    type: ParameterType
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    items: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate parameter configuration after initialization."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid parameter name: '{self.name}'")
        if self.enum is not None and len(self.enum) == 0:
            raise ValueError("Enum list cannot be empty")
        if self.type == ParameterType.ARRAY and self.items is None:
            self.items = {"type": "string"}  # Default to string array

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format.

        Returns:
            Dictionary conforming to JSON Schema specification for this parameter.

        Complexity:
            Time: O(1) - constant field access
            Space: O(k) where k = number of optional fields set
        """
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }

        if self.enum is not None:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        if self.items is not None:
            schema["items"] = self.items
        if self.properties is not None:
            schema["properties"] = self.properties

        return schema

    def validate_value(self, value: Any) -> Optional[str]:
        """Validate a value against this parameter's type specification.

        Args:
            value: The value to validate.

        Returns:
            Error message string if validation fails, None if valid.

        Complexity:
            Time: O(e) where e = enum size (if enum validation)
            Space: O(1)
        """
        validator, type_name = _TYPE_VALIDATORS.get(
            self.type, (lambda _: True, "unknown")
        )

        if not validator(value):
            return f"Expected {type_name}, got {type(value).__name__}"

        if self.enum is not None and value not in self.enum:
            return f"Value must be one of {self.enum}, got {value!r}"

        return None


@dataclass(frozen=False, slots=True)
class FunctionDefinition:
    """Complete specification of a callable function for LLM consumption.

    Core Idea:
        Provides a language-agnostic description of a function's interface,
        enabling LLMs to understand when and how to invoke external capabilities.

    Mathematical Formulation:
        A function definition $F$ is a tuple:
        $$F = (n, d, P, r)$$
        where:
        - $n$ is the function name (identifier)
        - $d$ is the natural language description
        - $P = \{p_1, p_2, ..., p_k\}$ is the parameter set
        - $r$ is the return type description

    Attributes:
        name: Unique function identifier (valid Python identifier).
        description: Natural language explanation of function purpose.
        parameters: Ordered list of parameter specifications.
        returns: Description of return value (optional).

    Example:
        >>> func_def = FunctionDefinition(
        ...     name="get_weather",
        ...     description="Retrieve current weather for a location",
        ...     parameters=[
        ...         FunctionParameter("city", ParameterType.STRING, "City name"),
        ...         FunctionParameter("unit", ParameterType.STRING, "Temperature unit",
        ...                          required=False, enum=["celsius", "fahrenheit"])
        ...     ]
        ... )
    """

    name: str
    description: str
    parameters: List[FunctionParameter] = field(default_factory=list)
    returns: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate function definition after initialization."""
        if not self.name.isidentifier():
            raise ValueError(f"Invalid function name: '{self.name}'")
        if not self.description.strip():
            raise ValueError("Function description cannot be empty")

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI Function Calling format.

        OpenAI format wraps the function in a type discriminator:
        {"type": "function", "function": {...}}

        Returns:
            Dictionary conforming to OpenAI's function calling specification.

        Complexity:
            Time: O(p) where p = number of parameters
            Space: O(p)
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Convert to Anthropic Tool Use format.

        Anthropic format uses 'input_schema' instead of 'parameters':
        {"name": "...", "description": "...", "input_schema": {...}}

        Returns:
            Dictionary conforming to Anthropic's tool use specification.

        Complexity:
            Time: O(p) where p = number of parameters
            Space: O(p)
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def get_required_params(self) -> List[str]:
        """Get list of required parameter names.

        Returns:
            List of parameter names where required=True.
        """
        return [p.name for p in self.parameters if p.required]

    def get_param_by_name(self, name: str) -> Optional[FunctionParameter]:
        """Retrieve parameter specification by name.

        Args:
            name: Parameter name to look up.

        Returns:
            FunctionParameter if found, None otherwise.

        Complexity:
            Time: O(p) where p = number of parameters
        """
        for param in self.parameters:
            if param.name == name:
                return param
        return None


@dataclass(frozen=False, slots=True)
class FunctionCall:
    """Represents a single function invocation request from an LLM.

    Core Idea:
        Captures the intent of an LLM to invoke a specific function with
        particular arguments, serving as the intermediate representation
        between LLM output and actual function execution.

    Attributes:
        name: Target function identifier.
        arguments: Mapping of parameter names to values.
        id: Optional unique identifier for tracking (used by some APIs).

    Example:
        >>> call = FunctionCall(
        ...     name="get_weather",
        ...     arguments={"city": "Tokyo", "unit": "celsius"},
        ...     id="call_abc123"
        ... )
        >>> call.to_dict()
        {'name': 'get_weather', 'arguments': {...}, 'id': 'call_abc123'}
    """

    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary format.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "arguments": self.arguments,
        }
        if self.id is not None:
            result["id"] = self.id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FunctionCall:
        """Deserialize from dictionary format.

        Args:
            data: Dictionary containing 'name' and optionally 'arguments', 'id'.

        Returns:
            New FunctionCall instance.

        Raises:
            KeyError: If required 'name' field is missing.
        """
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
            id=data.get("id"),
        )

    def __repr__(self) -> str:
        """Provide readable string representation."""
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"FunctionCall({self.name}({args_str}))"


class FunctionCallParser:
    """Multi-format parser for extracting function calls from LLM outputs.

    Core Idea:
        LLMs may output function calls in various formats depending on the
        provider and prompting strategy. This parser normalizes all formats
        into a unified FunctionCall representation.

    Supported Formats:
        1. OpenAI: {"name": "func", "arguments": {...}}
        2. Anthropic: {"type": "tool_use", "name": "func", "input": {...}}
        3. Markdown JSON blocks: ```json\n{...}\n```
        4. Raw JSON objects embedded in text

    Algorithm:
        1. Attempt to extract JSON from markdown code blocks
        2. Fall back to regex-based JSON object detection
        3. Parse extracted JSON into normalized FunctionCall objects
        4. Validate against registered function schemas (if provided)

    Complexity:
        - Parsing: O(n * m) where n = text length, m = number of patterns
        - Validation: O(p) where p = number of parameters

    Example:
        >>> parser = FunctionCallParser([weather_func_def])
        >>> calls = parser.parse('```json\\n{"name": "get_weather", ...}\\n```')
        >>> errors = parser.validate(calls[0])
    """

    # Compiled regex patterns for JSON extraction (ordered by specificity)
    _JSON_BLOCK_PATTERN: Final[re.Pattern] = re.compile(
        r"```(?:json)?\s*\n?(.*?)\n?```",
        re.DOTALL | re.IGNORECASE,
    )
    _JSON_OBJECT_PATTERN: Final[re.Pattern] = re.compile(
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        re.DOTALL,
    )

    def __init__(self, functions: Optional[List[FunctionDefinition]] = None) -> None:
        """Initialize parser with optional function schema registry.

        Args:
            functions: List of function definitions for validation.
                      If None, validation will only check basic structure.
        """
        self._functions: Dict[str, FunctionDefinition] = {
            f.name: f for f in (functions or [])
        }

    @property
    def functions(self) -> Dict[str, FunctionDefinition]:
        """Access registered function definitions."""
        return self._functions

    def register_function(self, func_def: FunctionDefinition) -> None:
        """Register a new function definition for validation.

        Args:
            func_def: Function definition to register.

        Raises:
            ValueError: If function with same name already registered.
        """
        if func_def.name in self._functions:
            raise ValueError(f"Function '{func_def.name}' already registered")
        self._functions[func_def.name] = func_def

    def parse(self, text: str) -> List[FunctionCall]:
        """Extract all function calls from text.

        Args:
            text: Raw LLM output potentially containing function calls.

        Returns:
            List of parsed FunctionCall objects (may be empty).

        Note:
            This method never raises exceptions for malformed input;
            it simply returns an empty list if no valid calls are found.
        """
        calls: List[FunctionCall] = []

        # Strategy 1: Extract from markdown JSON blocks
        json_blocks = self._JSON_BLOCK_PATTERN.findall(text)
        for block in json_blocks:
            call = self._try_parse_json(block.strip())
            if call is not None:
                calls.append(call)

        # Strategy 2: Fall back to raw JSON object detection
        if not calls:
            json_match = self._JSON_OBJECT_PATTERN.search(text)
            if json_match:
                call = self._try_parse_json(json_match.group())
                if call is not None:
                    calls.append(call)

        return calls

    def _try_parse_json(self, json_str: str) -> Optional[FunctionCall]:
        """Attempt to parse JSON string into FunctionCall.

        Args:
            json_str: Potential JSON string.

        Returns:
            FunctionCall if parsing succeeds, None otherwise.
        """
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                return None
            return self._normalize_call_format(data)
        except json.JSONDecodeError:
            return None

    def _normalize_call_format(self, data: Dict[str, Any]) -> Optional[FunctionCall]:
        """Normalize various API formats to unified FunctionCall.

        Handles:
            - OpenAI: {"name": "...", "arguments": {...}}
            - Anthropic: {"type": "tool_use", "name": "...", "input": {...}}
            - Generic: {"function": "...", "args": {...}}

        Args:
            data: Parsed JSON dictionary.

        Returns:
            Normalized FunctionCall or None if format unrecognized.
        """
        # OpenAI format
        if "name" in data and "arguments" in data:
            arguments = data["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return None
            return FunctionCall(
                name=data["name"],
                arguments=arguments,
                id=data.get("id"),
            )

        # Anthropic format
        if data.get("type") == "tool_use" and "name" in data:
            return FunctionCall(
                name=data["name"],
                arguments=data.get("input", {}),
                id=data.get("id"),
            )

        # Generic format
        if "function" in data:
            return FunctionCall(
                name=data["function"],
                arguments=data.get("args", data.get("arguments", {})),
            )

        return None

    def validate(self, call: FunctionCall) -> List[str]:
        """Validate function call against registered schema.

        Performs comprehensive validation:
            1. Function existence check
            2. Required parameter presence
            3. Parameter type conformance
            4. Unknown parameter detection

        Args:
            call: FunctionCall to validate.

        Returns:
            List of error messages (empty if valid).

        Complexity:
            Time: O(p) where p = number of parameters
        """
        errors: List[str] = []

        # Check function registration
        if call.name not in self._functions:
            errors.append(f"Unknown function: {call.name}")
            return errors

        func_def = self._functions[call.name]
        provided_params = set(call.arguments.keys())
        defined_params = {p.name for p in func_def.parameters}

        # Check required parameters
        for param in func_def.parameters:
            if param.required and param.name not in provided_params:
                errors.append(f"Missing required parameter: {param.name}")

        # Check parameter types
        for param in func_def.parameters:
            if param.name in call.arguments:
                value = call.arguments[param.name]
                type_error = param.validate_value(value)
                if type_error:
                    errors.append(f"Parameter '{param.name}': {type_error}")

        # Check for unknown parameters
        unknown_params = provided_params - defined_params
        for param_name in unknown_params:
            errors.append(f"Unknown parameter: {param_name}")

        return errors


def create_function_schema(
    func: Callable[..., Any],
    description: Optional[str] = None,
) -> FunctionDefinition:
    """Generate FunctionDefinition from Python function via introspection.

    Core Idea:
        Leverages Python's type hints and inspection capabilities to
        automatically generate JSON Schema-compatible function definitions,
        eliminating manual schema maintenance.

    Algorithm:
        1. Extract function name and docstring
        2. Parse type hints using typing.get_type_hints()
        3. Inspect signature for default values and required status
        4. Map Python types to JSON Schema types
        5. Construct FunctionDefinition with inferred metadata

    Args:
        func: Python function to generate schema from.
        description: Override description (defaults to first line of docstring).

    Returns:
        FunctionDefinition with inferred parameters.

    Raises:
        TypeError: If function has no type hints and cannot be introspected.

    Example:
        >>> def get_weather(city: str, unit: str = "celsius") -> str:
        ...     '''Get current weather for a city.'''
        ...     return f"Weather in {city}"
        >>> schema = create_function_schema(get_weather)
        >>> schema.name
        'get_weather'
        >>> len(schema.parameters)
        2

    Complexity:
        Time: O(p) where p = number of parameters
        Space: O(p)
    """
    func_name = func.__name__
    docstring = func.__doc__ or f"Function {func_name}"
    func_description = description or docstring.strip().split("\n")[0]

    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    sig = inspect.signature(func)
    parameters: List[FunctionParameter] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = ParameterType.STRING
        if param_name in hints:
            param_type = _python_type_to_param_type(hints[param_name])

        is_required = param.default is inspect.Parameter.empty
        default_value = None if is_required else param.default

        parameters.append(
            FunctionParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter {param_name}",
                required=is_required,
                default=default_value,
            )
        )

    return FunctionDefinition(
        name=func_name,
        description=func_description,
        parameters=parameters,
    )


def _python_type_to_param_type(python_type: Type) -> ParameterType:
    """Map Python type annotation to JSON Schema ParameterType.

    Handles:
        - Primitive types (str, int, float, bool)
        - Container types (list, dict)
        - Optional[T] (extracts inner type)
        - Union[T, None] (extracts non-None type)
        - Generic aliases (List[str], Dict[str, int])

    Args:
        python_type: Python type annotation.

    Returns:
        Corresponding ParameterType enum value.

    Note:
        Unknown types default to STRING for maximum compatibility.
    """
    if python_type in _PYTHON_TYPE_MAP:
        return _PYTHON_TYPE_MAP[python_type]

    origin = getattr(python_type, "__origin__", None)

    if origin is Union:
        args = getattr(python_type, "__args__", ())
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return _python_type_to_param_type(non_none_args[0])

    if origin is list:
        return ParameterType.ARRAY

    if origin is dict:
        return ParameterType.OBJECT

    return ParameterType.STRING


__all__ = [
    "ParameterType",
    "FunctionParameter",
    "FunctionDefinition",
    "FunctionCall",
    "FunctionCallParser",
    "create_function_schema",
]


if __name__ == "__main__":
    def example_function(
        query: str,
        max_results: int = 10,
        include_metadata: bool = False,
    ) -> str:
        """Search for information based on query."""
        return f"Results for: {query}"

    schema = create_function_schema(example_function)
    assert schema.name == "example_function"
    assert len(schema.parameters) == 3
    assert schema.parameters[0].required is True
    assert schema.parameters[1].required is False
    assert schema.parameters[1].default == 10

    openai_schema = schema.to_openai_schema()
    assert openai_schema["type"] == "function"
    assert "query" in openai_schema["function"]["parameters"]["properties"]

    anthropic_schema = schema.to_anthropic_schema()
    assert "input_schema" in anthropic_schema

    parser = FunctionCallParser([schema])
    test_input = '```json\n{"name": "example_function", "arguments": {"query": "test"}}\n```'
    calls = parser.parse(test_input)
    assert len(calls) == 1
    assert calls[0].name == "example_function"
    assert calls[0].arguments["query"] == "test"

    errors = parser.validate(calls[0])
    assert len(errors) == 0

    bad_call = FunctionCall(name="example_function", arguments={})
    errors = parser.validate(bad_call)
    assert any("Missing required" in e for e in errors)

    print("All self-tests passed.")
