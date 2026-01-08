# 05-multi-agent: 多智能体系统

多智能体系统模块，实现智能体协作、辩论和编排功能。

## 模块结构

```
05-multi-agent/
├── src/
│   ├── __init__.py              # 模块导出
│   ├── agent_base.py            # Agent 基类与工厂
│   ├── agent_communication.py   # 通信协议
│   ├── agent_orchestrator.py    # 编排器
│   ├── debate_agents.py         # 辩论智能体
│   └── collaborative_agents.py  # 协作智能体
├── notebooks/
│   ├── 01_AgentBase_tutorial.ipynb
│   ├── 02_MultiAgentDebate_tutorial.ipynb
│   └── 03_CollaborativeAgents_tutorial.ipynb
├── tests/
│   └── test_multi_agent.py
├── 知识点.md
└── README.md
```

## 核心功能

| 模块 | 功能 | 关键类 |
|-----|------|-------|
| agent_base | Agent 基础架构 | `BaseAgent`, `SimpleAgent`, `ReActAgent` |
| agent_communication | 消息传递 | `MessageBus`, `DirectChannel`, `BroadcastChannel` |
| agent_orchestrator | 任务编排 | `RoundRobinOrchestrator`, `CapabilityBasedOrchestrator` |
| debate_agents | 辩论系统 | `DebaterAgent`, `JudgeAgent`, `DebateArena` |
| collaborative_agents | 协作系统 | `CollaborativeTeam`, `ConsensusBuilder`, `VotingSystem` |

## 快速开始

```python
from src.agent_base import create_agent, MockLLM
from src.collaborative_agents import CollaborativeTeam, TeamConfig

# 创建智能体
llm = MockLLM(responses=["Response 1", "Response 2"])
agent = create_agent("Assistant", system_prompt="You are helpful.", llm=llm)

# 运行交互
import asyncio
response = asyncio.run(agent.step("Hello!"))
print(response.content)
```

## 运行测试

```bash
cd 14-agents-reasoning/05-multi-agent
pytest tests/ -v
```

## 参考文献

- [AutoGen](https://github.com/microsoft/autogen)
- [AI Safety via Debate](https://arxiv.org/abs/1805.00899)
