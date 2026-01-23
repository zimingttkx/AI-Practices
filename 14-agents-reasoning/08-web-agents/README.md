# 08-web-agents

Web Agents 模块：网页自动化与 API 交互。

## 模块结构

```
08-web-agents/
├── src/
│   ├── __init__.py
│   ├── browser_agent.py    # 网页导航与信息提取
│   └── api_agent.py        # API发现与链式调用
├── tests/
│   ├── __init__.py
│   └── test_web_agents.py  # 80 tests
├── notebooks/              # Jupyter 教程
├── 知识点.md               # 技术文档
└── README.md
```

## 核心组件

### BrowserAgent
- **PageAnalyzer**: HTML解析，DOM元素提取
- **NavigationPlanner**: 从目标规划导航动作
- **BrowserController**: 执行浏览器操作

### APIAgent
- **APIDiscovery**: OpenAPI规范解析
- **RequestBuilder**: 请求构建
- **ResponseParser**: 响应解析，数据提取
- **APIChainExecutor**: 链式API调用

## 快速开始

```python
# Browser Agent
from browser_agent import create_browser_agent

agent = create_browser_agent(simulate=True)
result = await agent.execute_task(
    "Search for Python tutorials",
    start_url="https://www.google.com"
)

# API Agent
from api_agent import create_api_agent

agent = create_api_agent(base_url="https://api.example.com")
response = await agent.get("/users")
users = agent.extract(response, "data.users")
```

## 测试

```bash
pytest 14-agents-reasoning/08-web-agents/tests -v
```
