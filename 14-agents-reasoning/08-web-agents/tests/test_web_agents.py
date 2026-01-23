"""
Unit tests for the 08-web-agents module.

Tests cover:
- Browser Agent: Page analysis, navigation planning, browser control
- API Agent: API discovery, request building, chain execution

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

from browser_agent import (
    PageState,
    DOMElement,
    ElementType,
    NavigationAction,
    FormAction,
    ClickAction,
    ScrollAction,
    WaitAction,
    ActionStatus,
    ActionResult,
    PageAnalyzer,
    NavigationPlanner,
    BrowserController,
    BrowserAgent,
    BrowserAgentConfig,
    create_browser_agent,
)

from api_agent import (
    HTTPMethod,
    APIEndpoint,
    APIRequest,
    APIResponse,
    APISchema,
    ResponseStatus,
    APIDiscovery,
    RequestBuilder,
    ResponseParser,
    APIChainExecutor,
    APIAgent,
    APIAgentConfig,
    create_api_agent,
)


# =============================================================================
# DOM Element Tests
# =============================================================================

class TestDOMElement:
    """Tests for DOMElement."""

    def test_creation(self):
        elem = DOMElement(
            element_type=ElementType.BUTTON,
            tag_name="button",
            text="Click Me",
        )
        assert elem.element_type == ElementType.BUTTON
        assert elem.text == "Click Me"
        assert elem.element_id.startswith("elem_")

    def test_attributes(self):
        elem = DOMElement(
            attributes={"id": "submit-btn", "class": "btn primary", "href": "/submit"},
        )
        assert elem.id_attr == "submit-btn"
        assert elem.class_attr == "btn primary"
        assert elem.href == "/submit"

    def test_matches_text(self):
        elem = DOMElement(text="Submit Form")
        assert elem.matches_text("submit") is True
        assert elem.matches_text("SUBMIT") is True
        assert elem.matches_text("submit", case_sensitive=True) is False
        assert elem.matches_text("cancel") is False

    def test_to_dict(self):
        elem = DOMElement(
            element_type=ElementType.LINK,
            text="Home",
        )
        d = elem.to_dict()
        assert d["element_type"] == "link"
        assert d["text"] == "Home"


# =============================================================================
# Page State Tests
# =============================================================================

class TestPageState:
    """Tests for PageState."""

    def test_creation(self):
        state = PageState(
            url="https://example.com",
            title="Example Page",
        )
        assert state.url == "https://example.com"
        assert state.title == "Example Page"

    def test_domain(self):
        state = PageState(url="https://www.example.com/path/page")
        assert state.domain == "www.example.com"

    def test_find_element_by_text(self):
        elements = [
            DOMElement(element_type=ElementType.BUTTON, text="Submit"),
            DOMElement(element_type=ElementType.LINK, text="Home"),
        ]
        state = PageState(elements=elements)
        
        found = state.find_element_by_text("Submit")
        assert found is not None
        assert found.element_type == ElementType.BUTTON

    def test_find_element_by_id(self):
        elem = DOMElement(attributes={"id": "main-btn"})
        state = PageState(elements=[elem])
        
        found = state.find_element_by_id("main-btn")
        assert found is not None

    def test_find_elements_by_type(self):
        elements = [
            DOMElement(element_type=ElementType.LINK, text="Link 1"),
            DOMElement(element_type=ElementType.LINK, text="Link 2"),
            DOMElement(element_type=ElementType.BUTTON, text="Button"),
        ]
        state = PageState(elements=elements)
        
        links = state.find_elements_by_type(ElementType.LINK)
        assert len(links) == 2

    def test_get_links(self):
        elements = [
            DOMElement(element_type=ElementType.LINK, text="Link"),
            DOMElement(element_type=ElementType.BUTTON, text="Button"),
        ]
        state = PageState(elements=elements)
        
        links = state.get_links()
        assert len(links) == 1


# =============================================================================
# Browser Action Tests
# =============================================================================

class TestBrowserActions:
    """Tests for browser actions."""

    def test_navigation_action(self):
        action = NavigationAction(url="https://example.com")
        assert action.action_type == "navigate"
        assert action.url == "https://example.com"

    def test_click_action(self):
        action = ClickAction(selector="#submit-btn", text="Submit")
        assert action.action_type == "click"
        assert action.selector == "#submit-btn"

    def test_form_action(self):
        action = FormAction(selector="#email", value="test@example.com")
        assert action.action_type == "fill"
        assert action.value == "test@example.com"

    def test_scroll_action(self):
        action = ScrollAction(direction="down", amount=500)
        assert action.direction == "down"
        assert action.amount == 500

    def test_wait_action(self):
        action = WaitAction(condition="time", value=2.0)
        assert action.condition == "time"


# =============================================================================
# Page Analyzer Tests
# =============================================================================

class TestPageAnalyzer:
    """Tests for PageAnalyzer."""

    def test_creation(self):
        analyzer = PageAnalyzer()
        assert analyzer.llm_func is not None

    def test_extract_title(self):
        analyzer = PageAnalyzer()
        html = "<html><head><title>Test Page</title></head><body></body></html>"
        title = analyzer._extract_title(html)
        assert title == "Test Page"

    def test_extract_elements(self):
        analyzer = PageAnalyzer()
        html = """
        <html>
            <body>
                <a href="/home">Home</a>
                <button>Submit</button>
                <input type="text" id="name">
            </body>
        </html>
        """
        elements = analyzer._extract_elements(html)
        assert len(elements) >= 3

    def test_analyze(self):
        analyzer = PageAnalyzer()
        html = "<html><head><title>Test</title></head><body><a href='/'>Link</a></body></html>"
        state = analyzer.analyze(html, "https://example.com")
        assert state.url == "https://example.com"
        assert state.title == "Test"
        assert len(state.elements) > 0


# =============================================================================
# Navigation Planner Tests
# =============================================================================

class TestNavigationPlanner:
    """Tests for NavigationPlanner."""

    def test_creation(self):
        planner = NavigationPlanner()
        assert planner.llm_func is not None

    @pytest.mark.asyncio
    async def test_plan_navigation(self):
        planner = NavigationPlanner()
        state = PageState()
        
        actions = await planner.plan("Go to https://google.com", state)
        assert len(actions) >= 1
        assert isinstance(actions[0], NavigationAction)

    @pytest.mark.asyncio
    async def test_plan_search(self):
        planner = NavigationPlanner()
        state = PageState(elements=[
            DOMElement(element_type=ElementType.INPUT, attributes={"type": "search", "id": "q"}),
        ])
        
        actions = await planner.plan("Search for Python tutorials", state)
        assert len(actions) >= 1

    @pytest.mark.asyncio
    async def test_plan_click(self):
        planner = NavigationPlanner()
        state = PageState(elements=[
            DOMElement(element_type=ElementType.BUTTON, text="Submit"),
        ])
        
        actions = await planner.plan("Click the Submit button", state)
        assert len(actions) >= 1

    def test_extract_url(self):
        planner = NavigationPlanner()
        url = planner._extract_url("Go to https://example.com")
        assert url == "https://example.com"

    def test_extract_query(self):
        planner = NavigationPlanner()
        query = planner._extract_query("Search for 'Python tutorials'")
        assert "Python" in query


# =============================================================================
# Browser Controller Tests
# =============================================================================

class TestBrowserController:
    """Tests for BrowserController."""

    def test_creation(self):
        controller = BrowserController(simulate=True)
        assert controller.simulate is True

    @pytest.mark.asyncio
    async def test_execute_navigate(self):
        controller = BrowserController(simulate=True, action_delay=0.01)
        action = NavigationAction(url="https://example.com")
        
        result = await controller.execute(action)
        assert result.is_success
        assert controller.get_current_state() is not None

    @pytest.mark.asyncio
    async def test_execute_click(self):
        controller = BrowserController(simulate=True, action_delay=0.01)
        action = ClickAction(selector="#btn")
        
        result = await controller.execute(action)
        assert result.is_success

    @pytest.mark.asyncio
    async def test_execute_fill(self):
        controller = BrowserController(simulate=True, action_delay=0.01)
        action = FormAction(selector="#email", value="test@test.com")
        
        result = await controller.execute(action)
        assert result.is_success

    @pytest.mark.asyncio
    async def test_execute_sequence(self):
        controller = BrowserController(simulate=True, action_delay=0.01)
        actions = [
            NavigationAction(url="https://example.com"),
            ClickAction(text="Button"),
        ]
        
        results = await controller.execute_sequence(actions)
        assert len(results) == 2
        assert all(r.is_success for r in results)


# =============================================================================
# Browser Agent Tests
# =============================================================================

class TestBrowserAgent:
    """Tests for BrowserAgent."""

    def test_creation(self):
        agent = BrowserAgent()
        assert agent.config.name == "BrowserAgent"

    def test_with_config(self):
        config = BrowserAgentConfig(name="TestBrowser", simulate=True)
        agent = BrowserAgent(config=config)
        assert agent.config.name == "TestBrowser"

    @pytest.mark.asyncio
    async def test_execute_task(self):
        agent = BrowserAgent()
        result = await agent.execute_task(
            "Go to example.com",
            start_url="https://example.com",
            max_steps=2,
        )
        assert result["status"] in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_navigate(self):
        agent = BrowserAgent()
        result = await agent.navigate("https://example.com")
        assert result.is_success

    @pytest.mark.asyncio
    async def test_click(self):
        agent = BrowserAgent()
        result = await agent.click(text="Button")
        assert result.is_success

    @pytest.mark.asyncio
    async def test_fill(self):
        agent = BrowserAgent()
        result = await agent.fill("#input", "value")
        assert result.is_success

    @pytest.mark.asyncio
    async def test_search(self):
        agent = BrowserAgent()
        result = await agent.search("Python")
        assert "status" in result

    def test_get_status(self):
        agent = BrowserAgent()
        status = agent.get_status()
        assert "name" in status
        assert "is_running" in status

    def test_factory_function(self):
        agent = create_browser_agent(name="MyBrowser", simulate=True)
        assert agent.config.name == "MyBrowser"


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestAPIEndpoint:
    """Tests for APIEndpoint."""

    def test_creation(self):
        endpoint = APIEndpoint(
            path="/users/{id}",
            method=HTTPMethod.GET,
            description="Get user by ID",
        )
        assert endpoint.path == "/users/{id}"
        assert endpoint.method == HTTPMethod.GET

    def test_path_params(self):
        endpoint = APIEndpoint(path="/users/{user_id}/posts/{post_id}")
        params = endpoint.path_params
        assert "user_id" in params
        assert "post_id" in params

    def test_build_url(self):
        endpoint = APIEndpoint(path="/users/{id}")
        url = endpoint.build_url("https://api.example.com", {"id": "123"})
        assert url == "https://api.example.com/users/123"


# =============================================================================
# API Request Tests
# =============================================================================

class TestAPIRequest:
    """Tests for APIRequest."""

    def test_creation(self):
        request = APIRequest(
            url="https://api.example.com/users",
            method=HTTPMethod.GET,
        )
        assert request.url == "https://api.example.com/users"
        assert request.method == HTTPMethod.GET

    def test_full_url_with_params(self):
        request = APIRequest(
            url="https://api.example.com/users",
            query_params={"page": "1", "limit": "10"},
        )
        assert "page=1" in request.full_url
        assert "limit=10" in request.full_url

    def test_to_dict(self):
        request = APIRequest(url="https://api.example.com/users")
        d = request.to_dict()
        assert d["url"] == "https://api.example.com/users"


# =============================================================================
# API Response Tests
# =============================================================================

class TestAPIResponse:
    """Tests for APIResponse."""

    def test_success_response(self):
        response = APIResponse(status_code=200, body={"data": "test"})
        assert response.is_success is True
        assert response.status == ResponseStatus.SUCCESS

    def test_client_error(self):
        response = APIResponse(status_code=404)
        assert response.is_client_error is True
        assert response.status == ResponseStatus.CLIENT_ERROR

    def test_server_error(self):
        response = APIResponse(status_code=500)
        assert response.is_server_error is True
        assert response.status == ResponseStatus.SERVER_ERROR

    def test_json_parsing(self):
        response = APIResponse(status_code=200, body='{"key": "value"}')
        data = response.json()
        assert data["key"] == "value"


# =============================================================================
# API Schema Tests
# =============================================================================

class TestAPISchema:
    """Tests for APISchema."""

    def test_creation(self):
        schema = APISchema(
            base_url="https://api.example.com",
            title="Example API",
        )
        assert schema.base_url == "https://api.example.com"

    def test_find_endpoint(self):
        endpoints = [
            APIEndpoint(path="/users", method=HTTPMethod.GET, tags=["users"]),
            APIEndpoint(path="/posts", method=HTTPMethod.GET, tags=["posts"]),
        ]
        schema = APISchema(base_url="https://api.example.com", endpoints=endpoints)
        
        found = schema.find_endpoint(path="/users")
        assert len(found) == 1
        
        found = schema.find_endpoint(tag="posts")
        assert len(found) == 1


# =============================================================================
# API Discovery Tests
# =============================================================================

class TestAPIDiscovery:
    """Tests for APIDiscovery."""

    def test_creation(self):
        discovery = APIDiscovery()
        assert discovery.llm_func is not None

    def test_from_openapi(self):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {"summary": "List users", "tags": ["users"]},
                    "post": {"summary": "Create user", "tags": ["users"]},
                },
            },
        }
        
        discovery = APIDiscovery()
        schema = discovery.from_openapi(spec)
        
        assert schema.title == "Test API"
        assert schema.base_url == "https://api.example.com"
        assert len(schema.endpoints) == 2


# =============================================================================
# Request Builder Tests
# =============================================================================

class TestRequestBuilder:
    """Tests for RequestBuilder."""

    def test_creation(self):
        builder = RequestBuilder(base_url="https://api.example.com")
        assert builder.base_url == "https://api.example.com"

    def test_build_simple(self):
        builder = RequestBuilder(base_url="https://api.example.com")
        request = builder.build("/users")
        assert request.url == "https://api.example.com/users"
        assert request.method == HTTPMethod.GET

    def test_build_with_method(self):
        builder = RequestBuilder(base_url="https://api.example.com")
        request = builder.build("POST /users", body={"name": "Test"})
        assert request.method == HTTPMethod.POST

    def test_build_with_path_params(self):
        builder = RequestBuilder(base_url="https://api.example.com")
        request = builder.build("/users/{id}", path_params={"id": "123"})
        assert "123" in request.url

    def test_build_with_auth(self):
        builder = RequestBuilder(
            base_url="https://api.example.com",
            auth_token="test-token",
        )
        request = builder.build("/users")
        assert "Authorization" in request.headers


# =============================================================================
# Response Parser Tests
# =============================================================================

class TestResponseParser:
    """Tests for ResponseParser."""

    def test_creation(self):
        parser = ResponseParser()
        assert parser.llm_func is not None

    def test_extract_simple(self):
        parser = ResponseParser()
        response = APIResponse(
            status_code=200,
            body={"data": {"user": {"name": "John"}}},
        )
        
        name = parser.extract(response, "data.user.name")
        assert name == "John"

    def test_extract_array(self):
        parser = ResponseParser()
        response = APIResponse(
            status_code=200,
            body={"users": [{"name": "A"}, {"name": "B"}]},
        )
        
        first = parser.extract(response, "users[0].name")
        assert first == "A"

    def test_extract_all(self):
        parser = ResponseParser()
        response = APIResponse(
            status_code=200,
            body={"id": 1, "name": "Test"},
        )
        
        result = parser.extract_all(response, {"id": "id", "name": "name"})
        assert result["id"] == 1
        assert result["name"] == "Test"


# =============================================================================
# API Chain Executor Tests
# =============================================================================

class TestAPIChainExecutor:
    """Tests for APIChainExecutor."""

    def test_creation(self):
        executor = APIChainExecutor()
        assert executor.max_retries == 3

    def test_add_step(self):
        executor = APIChainExecutor()
        step_id = executor.add_step("/users")
        assert step_id.startswith("step_")
        assert len(executor.steps) == 1

    @pytest.mark.asyncio
    async def test_execute_single(self):
        executor = APIChainExecutor()
        executor.add_step("/users")
        
        results = await executor.execute()
        assert len(results) == 1
        assert results[0].status == "success"

    @pytest.mark.asyncio
    async def test_execute_chain(self):
        executor = APIChainExecutor()
        step1 = executor.add_step("/users", extract_paths={"user_id": "data.id"})
        executor.add_step("/users/${step1.user_id}/posts", depends_on=[step1])
        
        results = await executor.execute()
        assert len(results) == 2

    def test_clear(self):
        executor = APIChainExecutor()
        executor.add_step("/users")
        executor.clear()
        assert len(executor.steps) == 0


# =============================================================================
# API Agent Tests
# =============================================================================

class TestAPIAgent:
    """Tests for APIAgent."""

    def test_creation(self):
        agent = APIAgent()
        assert agent.config.name == "APIAgent"

    def test_with_config(self):
        config = APIAgentConfig(
            name="TestAPI",
            base_url="https://api.example.com",
        )
        agent = APIAgent(config=config)
        assert agent.config.name == "TestAPI"

    @pytest.mark.asyncio
    async def test_get(self):
        agent = create_api_agent(base_url="https://api.example.com")
        response = await agent.get("/users")
        assert response.is_success

    @pytest.mark.asyncio
    async def test_post(self):
        agent = create_api_agent(base_url="https://api.example.com")
        response = await agent.post("/users", body={"name": "Test"})
        assert response.is_success

    def test_set_schema(self):
        agent = APIAgent()
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {"/users": {"get": {"summary": "List users"}}},
        }
        agent.set_schema(spec)
        assert agent._schema is not None

    def test_extract(self):
        agent = APIAgent()
        response = APIResponse(status_code=200, body={"name": "Test"})
        name = agent.extract(response, "name")
        assert name == "Test"

    @pytest.mark.asyncio
    async def test_execute_chain(self):
        agent = create_api_agent(base_url="https://api.example.com")
        steps = [
            {"endpoint": "/users"},
            {"endpoint": "/posts"},
        ]
        results = await agent.execute_chain(steps)
        assert len(results) == 2

    def test_get_history(self):
        agent = APIAgent()
        history = agent.get_history()
        assert "requests" in history
        assert "responses" in history

    def test_get_status(self):
        agent = APIAgent()
        status = agent.get_status()
        assert "name" in status
        assert "requests_made" in status

    def test_factory_function(self):
        agent = create_api_agent(
            name="MyAPI",
            base_url="https://api.example.com",
            auth_token="secret",
        )
        assert agent.config.name == "MyAPI"
        assert agent.request_builder.auth_token == "secret"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_browser_agent_workflow(self):
        """Test complete browser agent workflow."""
        agent = create_browser_agent(name="IntegrationTest", simulate=True)
        
        # Navigate to a page
        result = await agent.navigate("https://example.com")
        assert result.is_success
        
        # Click an element
        result = await agent.click(text="Button")
        assert result.is_success
        
        # Fill a form
        result = await agent.fill("#input", "test value")
        assert result.is_success

    @pytest.mark.asyncio
    async def test_api_agent_workflow(self):
        """Test complete API agent workflow."""
        agent = create_api_agent(
            name="IntegrationTest",
            base_url="https://api.example.com",
        )
        
        # Make GET request
        response = await agent.get("/users")
        assert response.is_success
        
        # Extract data
        data = agent.extract(response, "data")
        
        # Check history
        history = agent.get_history()
        assert len(history["requests"]) == 1

    @pytest.mark.asyncio
    async def test_api_chain_workflow(self):
        """Test API chain execution."""
        agent = create_api_agent(base_url="https://api.example.com")
        
        # Define chain
        steps = [
            {"endpoint": "/auth/login", "extract_paths": {"token": "data.token"}},
            {"endpoint": "/users/me"},
        ]
        
        # Execute chain
        results = await agent.execute_chain(steps)
        assert len(results) >= 1
