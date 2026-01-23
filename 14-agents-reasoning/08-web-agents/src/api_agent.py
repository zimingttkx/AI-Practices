"""
API Agent: API Discovery, Request Building, and Execution.

Core Idea:
    This module provides agents capable of discovering API endpoints,
    building requests, executing API chains, and handling errors
    with automatic recovery.

Mathematical Foundation:
    API chain execution as directed graph:
        G = (V, E) where V = endpoints, E = data dependencies

    Optimal request ordering via topological sort:
        order = TopologicalSort(G)

    Error recovery with retry:
        P(success_k) = 1 - (1 - p)^k

Design Patterns:
    - Builder Pattern: Request construction
    - Chain of Responsibility: Request/response processing
    - Strategy Pattern: Different API interaction strategies

References:
    - OpenAPI Specification: https://swagger.io/specification/
    - RestGPT: https://arxiv.org/abs/2306.06624
    - ToolLLM: https://arxiv.org/abs/2307.16789

Author: zhangfeng
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
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
)
from urllib.parse import urlencode, urljoin, urlparse

__all__ = [
    "HTTPMethod",
    "APIEndpoint",
    "APIRequest",
    "APIResponse",
    "APISchema",
    "APIDiscovery",
    "RequestBuilder",
    "ResponseParser",
    "APIChainExecutor",
    "APIAgent",
    "APIAgentConfig",
    "create_api_agent",
]

logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ResponseStatus(str, Enum):
    """API response status categories."""
    SUCCESS = "success"
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"


@dataclass
class APIEndpoint:
    """
    Represents an API endpoint.

    Attributes:
        path: URL path (e.g., /users/{id})
        method: HTTP method
        description: Endpoint description
        parameters: Path/query parameters
        request_body: Request body schema
        response_schema: Response schema
        requires_auth: Whether authentication is required
    """
    path: str
    method: HTTPMethod = HTTPMethod.GET
    description: str = ""
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    request_body: dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None
    requires_auth: bool = False
    tags: list[str] = field(default_factory=list)

    @property
    def path_params(self) -> list[str]:
        """Extract path parameters."""
        return re.findall(r'\{(\w+)\}', self.path)

    def build_url(self, base_url: str, path_values: dict[str, str] | None = None) -> str:
        """Build full URL with path parameters."""
        path = self.path
        if path_values:
            for param, value in path_values.items():
                path = path.replace(f"{{{param}}}", str(value))
        return urljoin(base_url, path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method.value,
            "description": self.description,
            "parameters": self.parameters,
            "requires_auth": self.requires_auth,
            "tags": self.tags,
        }


@dataclass
class APIRequest:
    """
    Represents an API request.

    Attributes:
        url: Full request URL
        method: HTTP method
        headers: Request headers
        query_params: Query parameters
        body: Request body
        timeout: Request timeout in seconds
    """
    url: str
    method: HTTPMethod = HTTPMethod.GET
    headers: dict[str, str] = field(default_factory=dict)
    query_params: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | str | None = None
    timeout: float = 30.0
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")

    @property
    def full_url(self) -> str:
        """Get URL with query parameters."""
        if self.query_params:
            return f"{self.url}?{urlencode(self.query_params)}"
        return self.url

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "url": self.url,
            "method": self.method.value,
            "headers": {k: v for k, v in self.headers.items() if k.lower() != "authorization"},
            "query_params": self.query_params,
            "has_body": self.body is not None,
        }


@dataclass
class APIResponse:
    """
    Represents an API response.

    Attributes:
        status_code: HTTP status code
        headers: Response headers
        body: Response body
        elapsed_time: Request duration in seconds
        request: Original request
    """
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: Any = None
    elapsed_time: float = 0.0
    request: APIRequest | None = None
    error: str | None = None
    response_id: str = field(default_factory=lambda: f"res_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def is_client_error(self) -> bool:
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        return 500 <= self.status_code < 600

    @property
    def status(self) -> ResponseStatus:
        if self.is_success:
            return ResponseStatus.SUCCESS
        elif self.is_client_error:
            return ResponseStatus.CLIENT_ERROR
        elif self.is_server_error:
            return ResponseStatus.SERVER_ERROR
        elif self.error and "timeout" in self.error.lower():
            return ResponseStatus.TIMEOUT
        return ResponseStatus.NETWORK_ERROR

    def json(self) -> Any:
        """Parse body as JSON."""
        if isinstance(self.body, str):
            return json.loads(self.body)
        return self.body

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "status_code": self.status_code,
            "is_success": self.is_success,
            "elapsed_time": self.elapsed_time,
            "body_preview": str(self.body)[:200] if self.body else None,
            "error": self.error,
        }


@dataclass
class APISchema:
    """
    Represents an API schema (OpenAPI-like).

    Attributes:
        base_url: API base URL
        title: API title
        version: API version
        endpoints: List of endpoints
    """
    base_url: str
    title: str = ""
    version: str = "1.0.0"
    description: str = ""
    endpoints: list[APIEndpoint] = field(default_factory=list)
    security_schemes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def find_endpoint(
        self,
        path: str | None = None,
        method: HTTPMethod | None = None,
        tag: str | None = None,
    ) -> list[APIEndpoint]:
        """Find endpoints matching criteria."""
        results = self.endpoints
        if path:
            results = [e for e in results if path in e.path]
        if method:
            results = [e for e in results if e.method == method]
        if tag:
            results = [e for e in results if tag in e.tags]
        return results

    def get_endpoint_by_path(self, path: str, method: HTTPMethod) -> APIEndpoint | None:
        """Get specific endpoint."""
        for ep in self.endpoints:
            if ep.path == path and ep.method == method:
                return ep
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "title": self.title,
            "version": self.version,
            "endpoint_count": len(self.endpoints),
            "endpoints": [e.to_dict() for e in self.endpoints],
        }


class APIDiscovery:
    """
    Discovers and parses API schemas.

    Core Idea:
        Parses OpenAPI/Swagger specifications to understand
        API structure and capabilities.

    Example:
        >>> discovery = APIDiscovery()
        >>> schema = discovery.from_openapi(openapi_spec)
    """

    def __init__(self, llm_func: Callable[[str], str] | None = None) -> None:
        self.llm_func = llm_func or self._mock_llm

    def _mock_llm(self, prompt: str) -> str:
        return "API endpoint discovered."

    def from_openapi(self, spec: dict[str, Any]) -> APISchema:
        """Parse OpenAPI specification."""
        base_url = ""
        if "servers" in spec and spec["servers"]:
            base_url = spec["servers"][0].get("url", "")

        endpoints: list[APIEndpoint] = []

        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in [m.value for m in HTTPMethod]:
                    # Parse parameters
                    parameters = {}
                    for param in details.get("parameters", []):
                        parameters[param.get("name", "")] = {
                            "in": param.get("in"),
                            "required": param.get("required", False),
                            "schema": param.get("schema", {}),
                        }

                    endpoints.append(APIEndpoint(
                        path=path,
                        method=HTTPMethod(method.upper()),
                        description=details.get("summary", "") or details.get("description", ""),
                        parameters=parameters,
                        request_body=details.get("requestBody"),
                        requires_auth="security" in details,
                        tags=details.get("tags", []),
                    ))

        return APISchema(
            base_url=base_url,
            title=spec.get("info", {}).get("title", ""),
            version=spec.get("info", {}).get("version", ""),
            description=spec.get("info", {}).get("description", ""),
            endpoints=endpoints,
        )

    def from_url(self, url: str) -> APISchema:
        """Discover API from base URL (mock implementation)."""
        # In real implementation, would try common paths like /openapi.json, /swagger.json
        return APISchema(
            base_url=url,
            title="Discovered API",
            endpoints=[
                APIEndpoint(path="/", method=HTTPMethod.GET, description="Root endpoint"),
            ],
        )

    def infer_from_response(self, response: APIResponse) -> APIEndpoint | None:
        """Infer endpoint schema from response."""
        if not response.is_success or not response.body:
            return None

        # Basic inference
        return APIEndpoint(
            path=urlparse(response.request.url).path if response.request else "/",
            method=response.request.method if response.request else HTTPMethod.GET,
            response_schema={"type": "object"} if isinstance(response.body, dict) else {"type": "array"},
        )


class RequestBuilder:
    """
    Builds API requests from natural language or structured input.

    Core Idea:
        Translates high-level intent into concrete API requests
        with proper parameters, headers, and authentication.

    Example:
        >>> builder = RequestBuilder(base_url="https://api.example.com")
        >>> request = builder.build("GET /users/123")
    """

    def __init__(
        self,
        base_url: str = "",
        default_headers: dict[str, str] | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_headers = default_headers or {"Content-Type": "application/json"}
        self.auth_token = auth_token

    def build(
        self,
        endpoint: str | APIEndpoint,
        method: HTTPMethod | None = None,
        path_params: dict[str, str] | None = None,
        query_params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> APIRequest:
        """Build an API request."""
        # Determine URL and method
        if isinstance(endpoint, str):
            # Parse string like "GET /users/{id}"
            parts = endpoint.split(maxsplit=1)
            if len(parts) == 2 and parts[0].upper() in [m.value for m in HTTPMethod]:
                method = HTTPMethod(parts[0].upper())
                path = parts[1]
            else:
                path = endpoint
                method = method or HTTPMethod.GET

            # Apply path parameters
            if path_params:
                for param, value in path_params.items():
                    path = path.replace(f"{{{param}}}", str(value))

            url = f"{self.base_url}{path}" if path.startswith("/") else path
        else:
            url = endpoint.build_url(self.base_url, path_params)
            method = method or endpoint.method

        # Build headers
        request_headers = self.default_headers.copy()
        if self.auth_token:
            request_headers["Authorization"] = f"Bearer {self.auth_token}"
        if headers:
            request_headers.update(headers)

        return APIRequest(
            url=url,
            method=method or HTTPMethod.GET,
            headers=request_headers,
            query_params=query_params or {},
            body=body,
        )

    def build_from_natural_language(
        self,
        description: str,
        schema: APISchema | None = None,
    ) -> APIRequest | None:
        """Build request from natural language description."""
        description_lower = description.lower()

        # Infer method from description
        method = HTTPMethod.GET
        if any(word in description_lower for word in ["create", "add", "post"]):
            method = HTTPMethod.POST
        elif any(word in description_lower for word in ["update", "modify", "put"]):
            method = HTTPMethod.PUT
        elif any(word in description_lower for word in ["delete", "remove"]):
            method = HTTPMethod.DELETE

        # Try to find matching endpoint in schema
        if schema:
            for endpoint in schema.endpoints:
                if endpoint.method == method:
                    # Check if description mentions endpoint
                    path_words = endpoint.path.replace("/", " ").replace("{", " ").replace("}", " ").split()
                    if any(word in description_lower for word in path_words if len(word) > 2):
                        return self.build(endpoint)

        # Default: extract path from description
        match = re.search(r'(/\w+(?:/\w+)*)', description)
        if match:
            return self.build(match.group(1), method=method)

        return None


class ResponseParser:
    """
    Parses and extracts data from API responses.

    Core Idea:
        Provides flexible data extraction from API responses
        using JSONPath-like queries and LLM-based extraction.

    Example:
        >>> parser = ResponseParser()
        >>> users = parser.extract(response, "$.data.users[*].name")
    """

    def __init__(self, llm_func: Callable[[str], str] | None = None) -> None:
        self.llm_func = llm_func or self._mock_llm

    def _mock_llm(self, prompt: str) -> str:
        return "Extracted data from response."

    def extract(self, response: APIResponse, path: str) -> Any:
        """Extract data using JSONPath-like syntax."""
        if not response.is_success or not response.body:
            return None

        data = response.body if isinstance(response.body, dict) else response.json()

        # Simple path extraction (e.g., "data.users[0].name")
        parts = path.replace("$.", "").split(".")
        result = data

        for part in parts:
            if not result:
                return None

            # Handle array index
            match = re.match(r'(\w+)\[(\d+|\*)\]', part)
            if match:
                key, index = match.groups()
                if key and key in result:
                    result = result[key]
                if index == "*":
                    continue  # Return all items
                elif isinstance(result, list):
                    idx = int(index)
                    result = result[idx] if idx < len(result) else None
            elif isinstance(result, dict) and part in result:
                result = result[part]
            else:
                return None

        return result

    def extract_all(self, response: APIResponse, paths: dict[str, str]) -> dict[str, Any]:
        """Extract multiple paths."""
        return {name: self.extract(response, path) for name, path in paths.items()}

    def summarize(self, response: APIResponse) -> str:
        """Generate summary of response."""
        if not response.is_success:
            return f"Error {response.status_code}: {response.error or 'Unknown error'}"

        body = response.body
        if isinstance(body, dict):
            keys = list(body.keys())[:5]
            return f"Response with keys: {', '.join(keys)}"
        elif isinstance(body, list):
            return f"Response with {len(body)} items"
        return f"Response: {str(body)[:100]}"


@dataclass
class APIChainStep:
    """A step in an API chain."""
    step_id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    endpoint: APIEndpoint | None = None
    request: APIRequest | None = None
    response: APIResponse | None = None
    extract_paths: dict[str, str] = field(default_factory=dict)
    extracted_data: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"


class APIChainExecutor:
    """
    Executes chains of API requests with data dependencies.

    Core Idea:
        Manages sequences of API calls where later calls
        depend on data from earlier responses.

    Example:
        >>> executor = APIChainExecutor()
        >>> executor.add_step(get_user, extract={"user_id": "$.id"})
        >>> executor.add_step(get_orders, depends_on=["step1"], params={"user_id": "${step1.user_id}"})
        >>> results = await executor.execute()
    """

    def __init__(
        self,
        request_builder: RequestBuilder | None = None,
        response_parser: ResponseParser | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.request_builder = request_builder or RequestBuilder()
        self.response_parser = response_parser or ResponseParser()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.steps: list[APIChainStep] = []
        self._context: dict[str, Any] = {}

    def add_step(
        self,
        endpoint: str | APIEndpoint,
        extract_paths: dict[str, str] | None = None,
        depends_on: list[str] | None = None,
        **request_kwargs: Any,
    ) -> str:
        """Add a step to the chain."""
        step = APIChainStep(
            extract_paths=extract_paths or {},
            depends_on=depends_on or [],
        )

        # Build request
        step.request = self.request_builder.build(endpoint, **request_kwargs)

        if isinstance(endpoint, APIEndpoint):
            step.endpoint = endpoint

        self.steps.append(step)
        return step.step_id

    async def execute(self) -> list[APIChainStep]:
        """Execute all steps in dependency order."""
        # Topological sort for dependency order
        executed = set()
        results: list[APIChainStep] = []

        while len(executed) < len(self.steps):
            for step in self.steps:
                if step.step_id in executed:
                    continue

                # Check if dependencies are met
                deps_met = all(dep in executed for dep in step.depends_on)
                if not deps_met:
                    continue

                # Resolve template variables
                if step.request:
                    step.request = self._resolve_templates(step.request)

                # Execute with retry
                step.response = await self._execute_with_retry(step.request)
                step.status = "success" if step.response.is_success else "failed"

                # Extract data
                if step.response.is_success and step.extract_paths:
                    step.extracted_data = self.response_parser.extract_all(
                        step.response, step.extract_paths
                    )
                    self._context[step.step_id] = step.extracted_data

                executed.add(step.step_id)
                results.append(step)

                # Stop chain on failure if configured
                if not step.response.is_success:
                    break

        return results

    def _resolve_templates(self, request: APIRequest) -> APIRequest:
        """Resolve template variables like ${step1.user_id}."""
        def resolve_value(value: str) -> str:
            if not isinstance(value, str):
                return value

            pattern = r'\$\{(\w+)\.(\w+)\}'
            matches = re.findall(pattern, value)

            for step_id, key in matches:
                if step_id in self._context and key in self._context[step_id]:
                    value = value.replace(f"${{{step_id}.{key}}}", str(self._context[step_id][key]))

            return value

        # Resolve in URL
        new_url = resolve_value(request.url)

        # Resolve in query params
        new_params = {k: resolve_value(v) for k, v in request.query_params.items()}

        # Resolve in body
        new_body = request.body
        if isinstance(new_body, dict):
            new_body = {k: resolve_value(str(v)) if isinstance(v, str) else v for k, v in new_body.items()}

        return APIRequest(
            url=new_url,
            method=request.method,
            headers=request.headers,
            query_params=new_params,
            body=new_body,
            timeout=request.timeout,
        )

    async def _execute_with_retry(self, request: APIRequest) -> APIResponse:
        """Execute request with retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await self._execute_request(request)

                if response.is_success or response.is_client_error:
                    return response

                # Retry on server error
                if response.is_server_error and attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue

                return response

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        return APIResponse(
            status_code=0,
            error=last_error or "Max retries exceeded",
            request=request,
        )

    async def _execute_request(self, request: APIRequest) -> APIResponse:
        """Execute a single request (mock implementation)."""
        start_time = time.time()

        # Mock response
        await asyncio.sleep(0.1)

        return APIResponse(
            status_code=200,
            body={"success": True, "data": {"id": 123, "name": "Test"}},
            elapsed_time=time.time() - start_time,
            request=request,
        )

    def clear(self) -> None:
        """Clear all steps and context."""
        self.steps.clear()
        self._context.clear()


@dataclass
class APIAgentConfig:
    """Configuration for APIAgent."""
    name: str = "APIAgent"
    base_url: str = ""
    auth_token: str | None = None
    max_retries: int = 3
    timeout: float = 30.0
    verbose: bool = False


class APIAgent:
    """
    Agent for API interaction and orchestration.

    Core Idea:
        Provides a unified interface for API discovery, request
        building, execution, and response parsing.

    Example:
        >>> agent = APIAgent(base_url="https://api.example.com")
        >>> response = await agent.get("/users/123")
        >>> data = agent.extract(response, "$.name")
    """

    def __init__(
        self,
        config: APIAgentConfig | None = None,
        llm_func: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config or APIAgentConfig()
        self.llm_func = llm_func or self._mock_llm

        self.discovery = APIDiscovery(llm_func=self.llm_func)
        self.request_builder = RequestBuilder(
            base_url=self.config.base_url,
            auth_token=self.config.auth_token,
        )
        self.response_parser = ResponseParser(llm_func=self.llm_func)
        self.chain_executor = APIChainExecutor(
            request_builder=self.request_builder,
            response_parser=self.response_parser,
            max_retries=self.config.max_retries,
        )

        self._schema: APISchema | None = None
        self._request_history: list[APIRequest] = []
        self._response_history: list[APIResponse] = []

        logger.info(f"APIAgent '{self.config.name}' initialized")

    def _mock_llm(self, prompt: str) -> str:
        return "API operation complete."

    def set_schema(self, schema: dict[str, Any] | APISchema) -> None:
        """Set API schema from OpenAPI spec or APISchema."""
        if isinstance(schema, dict):
            self._schema = self.discovery.from_openapi(schema)
        else:
            self._schema = schema

    async def request(
        self,
        method: HTTPMethod,
        path: str,
        **kwargs: Any,
    ) -> APIResponse:
        """Make an API request."""
        request = self.request_builder.build(path, method=method, **kwargs)
        self._request_history.append(request)

        response = await self.chain_executor._execute_with_retry(request)
        self._response_history.append(response)

        return response

    async def get(self, path: str, **kwargs: Any) -> APIResponse:
        """Make a GET request."""
        return await self.request(HTTPMethod.GET, path, **kwargs)

    async def post(self, path: str, body: dict[str, Any] | None = None, **kwargs: Any) -> APIResponse:
        """Make a POST request."""
        return await self.request(HTTPMethod.POST, path, body=body, **kwargs)

    async def put(self, path: str, body: dict[str, Any] | None = None, **kwargs: Any) -> APIResponse:
        """Make a PUT request."""
        return await self.request(HTTPMethod.PUT, path, body=body, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> APIResponse:
        """Make a DELETE request."""
        return await self.request(HTTPMethod.DELETE, path, **kwargs)

    def extract(self, response: APIResponse, path: str) -> Any:
        """Extract data from response."""
        return self.response_parser.extract(response, path)

    async def execute_chain(self, steps: list[dict[str, Any]]) -> list[APIChainStep]:
        """Execute a chain of API requests."""
        self.chain_executor.clear()

        for step in steps:
            self.chain_executor.add_step(**step)

        return await self.chain_executor.execute()

    def find_endpoint(self, query: str) -> APIEndpoint | None:
        """Find endpoint matching query."""
        if not self._schema:
            return None

        query_lower = query.lower()
        for endpoint in self._schema.endpoints:
            if query_lower in endpoint.description.lower():
                return endpoint
            if query_lower in endpoint.path.lower():
                return endpoint

        return None

    async def execute_natural_language(self, description: str) -> APIResponse | None:
        """Execute API call from natural language description."""
        request = self.request_builder.build_from_natural_language(description, self._schema)
        if request:
            self._request_history.append(request)
            response = await self.chain_executor._execute_with_retry(request)
            self._response_history.append(response)
            return response
        return None

    def get_history(self) -> dict[str, list[dict[str, Any]]]:
        """Get request/response history."""
        return {
            "requests": [r.to_dict() for r in self._request_history],
            "responses": [r.to_dict() for r in self._response_history],
        }

    def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        return {
            "name": self.config.name,
            "base_url": self.config.base_url,
            "has_schema": self._schema is not None,
            "endpoint_count": len(self._schema.endpoints) if self._schema else 0,
            "requests_made": len(self._request_history),
        }

    def __repr__(self) -> str:
        return f"APIAgent(name='{self.config.name}', base_url='{self.config.base_url}')"


def create_api_agent(
    name: str = "APIAgent",
    base_url: str = "",
    auth_token: str | None = None,
    llm_func: Callable[[str], str] | None = None,
    **config_kwargs: Any,
) -> APIAgent:
    """
    Factory function to create an APIAgent.

    Args:
        name: Agent name
        base_url: API base URL
        auth_token: Authentication token
        llm_func: LLM function for natural language processing
        **config_kwargs: Additional config parameters

    Returns:
        Configured APIAgent instance
    """
    config = APIAgentConfig(name=name, base_url=base_url, auth_token=auth_token, **config_kwargs)
    return APIAgent(config=config, llm_func=llm_func)
