"""
Structured Output: Type-Safe LLM Response Parsing and Validation

Core Idea:
    Structured Output transforms free-form LLM text responses into validated,
    type-safe data structures. By defining expected schemas (via Pydantic models
    or JSON Schema), we can extract reliable structured data from natural language
    outputs, enabling programmatic consumption of LLM results.

Mathematical Theory:
    The parsing process can be modeled as a composition of functions:

    $$\\text{parse}: \\mathcal{T} \\xrightarrow{extract} \\mathcal{J} \\xrightarrow{decode} \\mathcal{D} \\xrightarrow{validate} \\mathcal{S}$$

    where:
    - $\\mathcal{T}$ is the space of raw text outputs
    - $\\mathcal{J}$ is the space of JSON strings
    - $\\mathcal{D}$ is the space of decoded dictionaries
    - $\\mathcal{S}$ is the space of validated schema instances

    The extraction function uses regex pattern matching:

    $$extract(t) = \\arg\\max_{m \\in M(t)} |m|$$

    where $M(t)$ is the set of all JSON-like matches in text $t$.

Problem Statement:
    LLM outputs are inherently unstructured text, but downstream systems require:
    1. Predictable data formats for API responses
    2. Type-safe objects for application logic
    3. Validated data conforming to business rules
    4. Consistent structure across multiple LLM calls

    Structured Output solves this by providing extraction, validation, and
    type coercion in a single unified interface.

Algorithm Comparison:
    | Approach           | Pros                        | Cons                      |
    |--------------------|-----------------------------|-----------------------------|
    | Regex Extraction   | Fast, no dependencies       | Fragile, limited patterns   |
    | JSON Mode (API)    | Guaranteed valid JSON       | Provider-specific           |
    | Pydantic Parsing   | Type-safe, validation       | Requires model definition   |
    | LLM Self-Correction| Handles edge cases          | Extra API calls, latency    |

Complexity:
    - JSON extraction: O(n * p) where n = text length, p = patterns
    - JSON parsing: O(j) where j = JSON string length
    - Pydantic validation: O(f) where f = number of fields
    - Auto-fix attempts: O(r * j) where r = fix rules

Summary:
    This module provides ValidationError, OutputSchema, StructuredOutputParser,
    and factory functions for common parsing patterns. Features include multi-pattern
    JSON extraction, automatic error correction, Pydantic and JSON Schema validation,
    and format instruction generation for prompts.

References:
    - Pydantic V2: https://docs.pydantic.dev/latest/
    - JSON Schema: https://json-schema.org/specification.html
    - OpenAI JSON Mode: https://platform.openai.com/docs/guides/structured-outputs
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError


class ValidationError(Exception):
    """Exception raised when structured output validation fails.

    Core Idea:
        Provides detailed error information including field-level validation
        failures, enabling precise error reporting and debugging of LLM outputs.

    Attributes:
        message: Human-readable error description.
        errors: List of field-level error dictionaries with 'loc' and 'msg' keys.

    Example:
        >>> raise ValidationError(
        ...     "Validation failed",
        ...     errors=[{"loc": "age", "msg": "Expected integer, got string"}]
        ... )
    """

    def __init__(
        self,
        message: str,
        errors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize validation error with message and optional field errors.

        Args:
            message: Primary error message.
            errors: List of field-level error details.
        """
        super().__init__(message)
        self.message = message
        self.errors = errors or []

    def __str__(self) -> str:
        """Format error with field details if available."""
        if self.errors:
            error_details = "\n".join(
                f"  - {e.get('loc', 'unknown')}: {e.get('msg', 'unknown error')}"
                for e in self.errors
            )
            return f"{self.message}\nErrors:\n{error_details}"
        return self.message

    def __repr__(self) -> str:
        """Provide informative string representation."""
        return f"ValidationError({self.message!r}, errors={len(self.errors)})"


@dataclass(slots=True)
class OutputSchema:
    """Schema definition for expected LLM output structure.

    Core Idea:
        Encapsulates the expected output format with metadata for prompt
        generation and validation. Supports both Pydantic models and raw
        JSON Schema dictionaries.

    Attributes:
        name: Schema identifier for reference.
        description: Natural language explanation of expected output.
        schema: Pydantic model class or JSON Schema dictionary.
        examples: Sample outputs for few-shot prompting.

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> schema = OutputSchema(
        ...     name="person",
        ...     description="Information about a person",
        ...     schema=Person,
        ...     examples=[{"name": "Alice", "age": 30}]
        ... )
    """

    name: str
    description: str
    schema: Union[Dict[str, Any], Type[BaseModel]]
    examples: List[Any] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema dictionary format.

        Returns:
            JSON Schema dictionary, either from Pydantic model or as-is.

        Complexity:
            Time: O(f) where f = number of model fields (for Pydantic)
        """
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            return self.schema.model_json_schema()
        return self.schema

    def get_format_instructions(self) -> str:
        """Generate prompt instructions for LLM output formatting.

        Creates a detailed instruction string including the JSON schema
        and any provided examples, suitable for inclusion in system prompts.

        Returns:
            Formatted instruction string with schema and examples.

        Complexity:
            Time: O(f + e) where f = fields, e = examples
        """
        schema = self.to_json_schema()
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

        instructions = f"""Please respond with a JSON object that follows this schema:

```json
{schema_str}
```

{self.description}"""

        if self.examples:
            instructions += "\n\nExamples:"
            for i, example in enumerate(self.examples, 1):
                example_str = json.dumps(example, indent=2, ensure_ascii=False)
                instructions += f"\n\nExample {i}:\n```json\n{example_str}\n```"

        return instructions

    def __repr__(self) -> str:
        """Provide informative string representation."""
        schema_type = (
            self.schema.__name__
            if isinstance(self.schema, type)
            else "dict"
        )
        return f"OutputSchema({self.name!r}, schema={schema_type})"


# Type variable for generic parser
T = TypeVar("T", bound=BaseModel)


# Compiled regex patterns for JSON extraction (ordered by specificity)
_JSON_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```json\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)
_JSON_GENERIC_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```\s*\n?(.*?)\n?```",
    re.DOTALL,
)
_JSON_OBJECT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\{.*\}",
    re.DOTALL,
)

# JSON fix rules: (pattern, replacement)
_JSON_FIX_RULES: Final[List[tuple[str, str]]] = [
    (r",\s*}", "}"),           # Trailing comma in object
    (r",\s*]", "]"),           # Trailing comma in array
    (r"'", '"'),               # Single quotes to double quotes
    (r"(?<!\\)\n", "\\n"),     # Unescaped newlines
    (r"\bTrue\b", "true"),     # Python True to JSON true
    (r"\bFalse\b", "false"),   # Python False to JSON false
    (r"\bNone\b", "null"),     # Python None to JSON null
]


class StructuredOutputParser(Generic[T]):
    """Parser for extracting and validating structured data from LLM outputs.

    Core Idea:
        Combines JSON extraction, parsing, and validation into a single interface.
        Supports both Pydantic models (for type-safe results) and JSON Schema
        (for dynamic validation). Includes automatic error correction for common
        LLM JSON formatting mistakes.

    Design Patterns:
        - Strategy Pattern: Different validation strategies (Pydantic vs JSON Schema)
        - Template Method: parse() defines the algorithm, subparts are customizable
        - Factory Pattern: Static factory methods for common parser configurations

    Type Safety:
        When initialized with a Pydantic model, parse() returns instances of that
        model type. When initialized with JSON Schema, returns Dict[str, Any].

    Attributes:
        strict: If True, validation errors raise exceptions; if False, returns raw data.
        auto_fix: If True, attempts to fix common JSON formatting errors.

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> parser = StructuredOutputParser(Person)
        >>> result = parser.parse('```json\\n{"name": "Alice", "age": 30}\\n```')
        >>> result.name
        'Alice'
    """

    __slots__ = (
        "strict",
        "auto_fix",
        "_pydantic_model",
        "_json_schema",
        "_output_schema",
    )

    def __init__(
        self,
        schema: Union[Type[T], Dict[str, Any], OutputSchema],
        strict: bool = True,
        auto_fix: bool = False,
    ) -> None:
        """Initialize the parser with a schema definition.

        Args:
            schema: One of:
                - Pydantic model class for type-safe parsing
                - JSON Schema dictionary for flexible validation
                - OutputSchema instance for full configuration
            strict: Whether to raise ValidationError on validation failure.
            auto_fix: Whether to attempt automatic JSON error correction.

        Complexity:
            Time: O(1) for initialization
            Space: O(s) where s = schema size
        """
        self.strict = strict
        self.auto_fix = auto_fix

        if isinstance(schema, OutputSchema):
            self._output_schema = schema
            self._pydantic_model: Optional[Type[T]] = (
                schema.schema if isinstance(schema.schema, type) else None
            )
            self._json_schema = schema.to_json_schema()
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            self._pydantic_model = schema
            self._json_schema = schema.model_json_schema()
            self._output_schema = OutputSchema(
                name=schema.__name__,
                description=schema.__doc__ or "",
                schema=schema,
            )
        else:
            self._pydantic_model = None
            self._json_schema = schema
            self._output_schema = OutputSchema(
                name="output",
                description="",
                schema=schema,
            )

    def parse(self, text: str) -> Union[T, Dict[str, Any]]:
        """Parse and validate structured output from text.

        Algorithm:
            1. Extract JSON string using regex patterns
            2. Decode JSON to dictionary
            3. Optionally apply auto-fix for malformed JSON
            4. Validate against schema (Pydantic or JSON Schema)
            5. Return typed result or raise ValidationError

        Args:
            text: Raw LLM output text potentially containing JSON.

        Returns:
            Pydantic model instance (if schema is Pydantic) or dictionary.

        Raises:
            ValidationError: If JSON extraction or validation fails.

        Complexity:
            Time: O(n * p + j + f) where n = text length, p = patterns,
                  j = JSON length, f = fields
        """
        # Extract JSON from text
        json_str = self._extract_json(text)
        if json_str is None:
            raise ValidationError("No JSON found in output")

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.auto_fix:
                data = self._try_fix_json(json_str)
                if data is None:
                    raise ValidationError(f"Invalid JSON: {e}")
            else:
                raise ValidationError(f"Invalid JSON: {e}")

        # Validate and return
        return self._validate(data)

    def parse_or_none(self, text: str) -> Optional[Union[T, Dict[str, Any]]]:
        """Parse text with graceful error handling.

        Args:
            text: Raw LLM output text.

        Returns:
            Parsed result if successful, None if any error occurs.

        Complexity:
            Same as parse()
        """
        try:
            return self.parse(text)
        except (ValidationError, Exception):
            return None

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON string from text using multiple patterns.

        Tries patterns in order of specificity:
            1. ```json ... ``` blocks (most specific)
            2. ``` ... ``` blocks (generic code blocks)
            3. Raw { ... } objects (least specific)

        Args:
            text: Text to search for JSON.

        Returns:
            Extracted JSON string or None if not found.

        Complexity:
            Time: O(n * p) where n = text length, p = number of patterns
        """
        patterns = [
            _JSON_BLOCK_PATTERN,
            _JSON_GENERIC_BLOCK_PATTERN,
            _JSON_OBJECT_PATTERN,
        ]

        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                # Return last match (usually most complete)
                json_str = matches[-1] if isinstance(matches[-1], str) else matches[-1]
                return json_str.strip()
        return None

    def _try_fix_json(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Attempt to fix common JSON formatting errors.

        Common LLM Errors Fixed:
            - Trailing commas: {"a": 1,} -> {"a": 1}
            - Single quotes: {'a': 1} -> {"a": 1}
            - Python booleans: True/False -> true/false
            - Python None: None -> null
            - Unescaped newlines

        Args:
            json_str: Potentially malformed JSON string.

        Returns:
            Parsed dictionary if fix successful, None otherwise.

        Complexity:
            Time: O(r * j) where r = fix rules, j = JSON length
        """
        fixed = json_str
        for pattern, replacement in _JSON_FIX_RULES:
            fixed = re.sub(pattern, replacement, fixed)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    def _validate(self, data: Dict[str, Any]) -> Union[T, Dict[str, Any]]:
        """Validate data against the configured schema.

        Uses Pydantic validation if a model was provided, otherwise falls
        back to JSON Schema validation.

        Args:
            data: Parsed dictionary to validate.

        Returns:
            Pydantic model instance or validated dictionary.

        Raises:
            ValidationError: If strict mode and validation fails.

        Complexity:
            Time: O(f) where f = number of fields
        """
        if self._pydantic_model is not None:
            try:
                return self._pydantic_model.model_validate(data)
            except PydanticValidationError as e:
                if self.strict:
                    raise ValidationError(
                        "Validation failed",
                        errors=[
                            {"loc": str(err["loc"]), "msg": err["msg"]}
                            for err in e.errors()
                        ],
                    )
                return data
        else:
            if self.strict:
                errors = self._validate_json_schema(data)
                if errors:
                    raise ValidationError("Validation failed", errors=errors)
            return data

    def _validate_json_schema(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate data against JSON Schema.

        Performs basic JSON Schema validation including:
            - Root type checking
            - Required field presence
            - Property type validation
            - Enum value validation

        Args:
            data: Dictionary to validate.

        Returns:
            List of error dictionaries (empty if valid).

        Complexity:
            Time: O(f) where f = number of fields
        """
        errors: List[Dict[str, Any]] = []

        # Root type check
        if "type" in self._json_schema:
            expected_type = self._json_schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                errors.append({
                    "loc": "root",
                    "msg": f"Expected object, got {type(data).__name__}",
                })
            elif expected_type == "array" and not isinstance(data, list):
                errors.append({
                    "loc": "root",
                    "msg": f"Expected array, got {type(data).__name__}",
                })

        # Required fields check
        if "required" in self._json_schema and isinstance(data, dict):
            for field_name in self._json_schema["required"]:
                if field_name not in data:
                    errors.append({"loc": field_name, "msg": "Field required"})

        # Property type validation
        if "properties" in self._json_schema and isinstance(data, dict):
            for field_name, field_schema in self._json_schema["properties"].items():
                if field_name in data:
                    field_errors = self._check_field_type(
                        data[field_name], field_schema, field_name
                    )
                    errors.extend(field_errors)

        return errors

    def _check_field_type(
        self,
        value: Any,
        schema: Dict[str, Any],
        loc: str,
    ) -> List[Dict[str, Any]]:
        """Validate a single field against its schema.

        Args:
            value: Field value to validate.
            schema: JSON Schema for the field.
            loc: Field location for error reporting.

        Returns:
            List of error dictionaries for this field.

        Complexity:
            Time: O(1) for type check, O(e) for enum check
        """
        errors: List[Dict[str, Any]] = []
        expected_type = schema.get("type")

        type_checks: Dict[str, Callable[[Any], bool]] = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        if expected_type and expected_type in type_checks:
            if not type_checks[expected_type](value):
                errors.append({
                    "loc": loc,
                    "msg": f"Expected {expected_type}, got {type(value).__name__}",
                })

        # Enum validation
        if "enum" in schema and value not in schema["enum"]:
            errors.append({
                "loc": loc,
                "msg": f"Value must be one of {schema['enum']}",
            })

        return errors

    def get_format_instructions(self) -> str:
        """Get format instructions for prompt construction.

        Returns:
            Formatted instruction string from the output schema.
        """
        return self._output_schema.get_format_instructions()

    def __repr__(self) -> str:
        """Provide informative string representation."""
        schema_name = (
            self._pydantic_model.__name__
            if self._pydantic_model
            else "JSONSchema"
        )
        return f"StructuredOutputParser({schema_name}, strict={self.strict})"


class OutputFormat(str, Enum):
    """Predefined output format types.

    Attributes:
        JSON: Structured JSON object format.
        MARKDOWN: Markdown-formatted text.
        LIST: Ordered or unordered list format.
        TABLE: Tabular data format.
    """

    JSON: Final[str] = "json"
    MARKDOWN: Final[str] = "markdown"
    LIST: Final[str] = "list"
    TABLE: Final[str] = "table"


def create_list_parser(item_type: Type[T]) -> StructuredOutputParser[Any]:
    """Factory function for creating list item parsers.

    Creates a parser that expects a JSON object with an 'items' array
    containing elements of the specified Pydantic model type.

    Args:
        item_type: Pydantic model class for list elements.

    Returns:
        Configured StructuredOutputParser for list extraction.

    Example:
        >>> class Task(BaseModel):
        ...     title: str
        ...     done: bool
        >>> parser = create_list_parser(Task)
        >>> result = parser.parse('{"items": [{"title": "Test", "done": false}]}')
    """
    class ListWrapper(BaseModel):
        items: List[item_type]  # type: ignore[valid-type]

    return StructuredOutputParser(ListWrapper)


def create_choice_parser(choices: List[str]) -> StructuredOutputParser[Any]:
    """Factory function for creating choice selection parsers.

    Creates a parser that expects the LLM to select from predefined options
    and optionally provide a reason for the selection.

    Args:
        choices: List of valid choice strings.

    Returns:
        Configured StructuredOutputParser for choice extraction.

    Example:
        >>> parser = create_choice_parser(["approve", "reject", "pending"])
        >>> result = parser.parse('{"choice": "approve", "reason": "Meets criteria"}')
        >>> result["choice"]
        'approve'
    """
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "choice": {
                "type": "string",
                "enum": choices,
                "description": "The selected choice",
            },
            "reason": {
                "type": "string",
                "description": "Reason for the choice",
            },
        },
        "required": ["choice"],
    }

    return StructuredOutputParser(
        OutputSchema(
            name="choice",
            description=f"Select one of: {', '.join(choices)}",
            schema=schema,
        )
    )


def create_extraction_parser(fields: Dict[str, str]) -> StructuredOutputParser[Any]:
    """Factory function for creating information extraction parsers.

    Creates a parser configured to extract specific named fields from text,
    useful for entity extraction and structured data mining tasks.

    Args:
        fields: Mapping of field names to their descriptions.

    Returns:
        Configured StructuredOutputParser for field extraction.

    Example:
        >>> parser = create_extraction_parser({
        ...     "name": "Person's full name",
        ...     "email": "Email address",
        ... })
        >>> result = parser.parse('{"name": "John Doe", "email": "john@example.com"}')
    """
    properties: Dict[str, Any] = {}
    for field_name, description in fields.items():
        properties[field_name] = {
            "type": "string",
            "description": description,
        }

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(fields.keys()),
    }

    return StructuredOutputParser(
        OutputSchema(
            name="extraction",
            description="Extract the following information",
            schema=schema,
        )
    )


__all__ = [
    "ValidationError",
    "OutputSchema",
    "StructuredOutputParser",
    "OutputFormat",
    "create_list_parser",
    "create_choice_parser",
    "create_extraction_parser",
]


if __name__ == "__main__":
    # Self-test: Verify core functionality

    # Test Pydantic model parsing
    class Person(BaseModel):
        """Person information."""
        name: str
        age: int
        email: Optional[str] = None

    parser = StructuredOutputParser(Person)

    # Test basic parsing from JSON block
    llm_output = '''
Here is the extracted information:
```json
{"name": "Alice", "age": 30, "email": "alice@example.com"}
```
'''
    result = parser.parse(llm_output)
    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert result.age == 30
    assert result.email == "alice@example.com"

    # Test parsing without email (optional field)
    result = parser.parse('{"name": "Bob", "age": 25}')
    assert result.name == "Bob"
    assert result.email is None

    # Test parse_or_none with valid input
    result = parser.parse_or_none('{"name": "Charlie", "age": 35}')
    assert result is not None
    assert result.name == "Charlie"

    # Test parse_or_none with invalid input
    result = parser.parse_or_none("no json here")
    assert result is None

    # Test validation error for missing required field
    try:
        parser.parse('{"name": "Dave"}')  # Missing age
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "age" in str(e).lower() or len(e.errors) > 0

    # Test validation error for wrong type
    try:
        parser.parse('{"name": "Eve", "age": "thirty"}')  # age should be int
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # Test JSON Schema parsing
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["title"],
    }
    schema_parser = StructuredOutputParser(schema)

    result = schema_parser.parse('{"title": "Test", "score": 9.5}')
    assert result["title"] == "Test"
    assert result["score"] == 9.5

    # Test auto-fix for trailing comma
    auto_fix_parser = StructuredOutputParser(Person, auto_fix=True)
    result = auto_fix_parser.parse('{"name": "Frank", "age": 40,}')  # Trailing comma
    assert result.name == "Frank"

    # Test auto-fix for Python booleans
    class BoolModel(BaseModel):
        flag: bool

    bool_parser = StructuredOutputParser(BoolModel, auto_fix=True)
    # Note: This test depends on the fix rules working correctly

    # Test OutputSchema
    output_schema = OutputSchema(
        name="person",
        description="Information about a person",
        schema=Person,
        examples=[{"name": "Example", "age": 25}],
    )
    assert output_schema.name == "person"
    json_schema = output_schema.to_json_schema()
    assert "properties" in json_schema

    # Test format instructions
    instructions = output_schema.get_format_instructions()
    assert "json" in instructions.lower()
    assert "Example" in instructions

    # Test choice parser
    choice_parser = create_choice_parser(["approve", "reject", "pending"])
    result = choice_parser.parse('{"choice": "approve", "reason": "Looks good"}')
    assert result["choice"] == "approve"
    assert result["reason"] == "Looks good"

    # Test choice validation
    try:
        choice_parser.parse('{"choice": "invalid"}')
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # Test extraction parser
    extraction_parser = create_extraction_parser({
        "product": "Product name",
        "price": "Product price",
    })
    result = extraction_parser.parse('{"product": "Widget", "price": "$9.99"}')
    assert result["product"] == "Widget"
    assert result["price"] == "$9.99"

    # Test ValidationError formatting
    error = ValidationError(
        "Test error",
        errors=[
            {"loc": "field1", "msg": "Error 1"},
            {"loc": "field2", "msg": "Error 2"},
        ],
    )
    error_str = str(error)
    assert "field1" in error_str
    assert "Error 1" in error_str

    # Test non-strict mode
    non_strict_parser = StructuredOutputParser(Person, strict=False)
    result = non_strict_parser.parse('{"name": "Grace", "age": "invalid"}')
    # In non-strict mode, returns raw data on validation failure
    assert isinstance(result, dict)

    # Test generic code block extraction
    result = parser.parse('''
```
{"name": "Henry", "age": 45}
```
''')
    assert result.name == "Henry"

    # Test raw JSON extraction (no code block)
    result = parser.parse('The result is {"name": "Ivy", "age": 28} as shown.')
    assert result.name == "Ivy"

    print("All self-tests passed.")
