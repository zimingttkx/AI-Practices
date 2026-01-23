# 07-computer-use

Computer Use Agent 模块：屏幕理解与 GUI 自动化、代码生成与分析。

## 模块结构

```
07-computer-use/
├── src/
│   ├── __init__.py
│   ├── computer_agent.py   # 屏幕理解与GUI自动化
│   └── code_agent.py       # 代码生成、调试、重构
├── tests/
│   ├── __init__.py
│   └── test_computer_use.py  # 80 tests
├── notebooks/              # Jupyter 教程
├── 知识点.md               # 技术文档
└── README.md
```

## 核心组件

### ComputerAgent
- **ScreenAnalyzer**: 屏幕截图分析，UI元素检测
- **ActionPlanner**: 从自然语言目标规划动作序列
- **GUIController**: 执行鼠标/键盘操作

### CodeAgent
- **CodeAnalyzer**: 静态分析、问题检测、代码指标
- **CodeGenerator**: 从描述生成代码
- **CodeDebugger**: 错误分析、修复建议
- **CodeRefactorer**: 重构建议

## 快速开始

```python
# Computer Agent
from computer_agent import create_computer_agent

agent = create_computer_agent(simulate=True)
result = await agent.execute_task("Click the Submit button")

# Code Agent
from code_agent import create_code_agent

agent = create_code_agent()
code = agent.generate("Create a function to sort a list")
analysis = agent.analyze(code)
```

## 测试

```bash
pytest 14-agents-reasoning/07-computer-use/tests -v
```
