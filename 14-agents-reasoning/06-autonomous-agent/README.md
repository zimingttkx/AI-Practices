# 06-autonomous-agent: AutoGPT 风格自主智能体

> 实现完整的自主智能体系统，包含目标管理、动作执行、自我反思和 OODA 执行循环。

## 快速开始

```python
from autonomous_agent import AutonomousAgent, AgentBuilder

# 方式1: 直接创建
agent = AutonomousAgent()
agent.set_objective("完成数据分析任务")
agent.register_tool("analyze", my_analyze_func)

# 方式2: Builder 模式
agent = (
    AgentBuilder("ResearchBot")
    .with_config(max_iterations=20)
    .with_tool("search", search_func)
    .with_constraint("引用来源")
    .with_objective("研究 LLM 最新进展")
    .build()
)
```

## 模块结构

```
06-autonomous-agent/
├── src/
│   ├── goal_manager.py      # 目标管理 (HTN分解、优先级队列)
│   ├── action_executor.py   # 动作执行 (工具/代码/文件)
│   ├── self_reflection.py   # 自我反思 (UCB1策略调整)
│   ├── agent_loop.py        # OODA执行循环
│   └── autonomous_agent.py  # 主类集成
├── tests/                   # 60 单元测试
├── notebooks/               # 3 个教程
└── 知识点.md                # 完整文档
```

## 核心组件

| 组件 | 功能 | 关键算法 |
|:-----|:-----|:---------|
| GoalManager | 目标分解与调度 | HTN, 优先级堆 |
| ActionExecutor | 工具调用、代码执行 | 指数退避重试 |
| SelfReflector | 经验学习 | UCB1 策略选择 |
| AgentLoop | 决策循环 | OODA 模型 |

## 测试

```bash
pytest tests/test_autonomous_agent.py -v
# 60 passed
```

## 参考

- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [BabyAGI](https://github.com/yoheinakajima/babyagi)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
