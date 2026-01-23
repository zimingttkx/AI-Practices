"""
08-web-agents: Web Navigation and API Interaction Module.

This module provides agents for web browsing automation and API interaction,
enabling automated information extraction and web-based task completion.

Components:
    - BrowserAgent: Web navigation, form filling, information extraction
    - APIAgent: API discovery, request chaining, error recovery

Author: zhangfeng
Version: 1.0.0
"""

from .api_agent import (
    APIAgent,
    APIAgentConfig,
    APIChainExecutor,
    APIDiscovery,
    APIEndpoint,
    APIRequest,
    APIResponse,
    APISchema,
    HTTPMethod,
    RequestBuilder,
    ResponseParser,
    create_api_agent,
)
from .browser_agent import (
    BrowserAgent,
    BrowserAgentConfig,
    BrowserController,
    ClickAction,
    DOMElement,
    ElementType,
    FormAction,
    NavigationAction,
    NavigationPlanner,
    PageAnalyzer,
    PageState,
    ScrollAction,
    WaitAction,
    create_browser_agent,
)

__all__ = [
    # Browser Agent
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
    # API Agent
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
