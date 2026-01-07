# 04-planning: AI Agent 规划系统

## 概述

本模块实现了 AI Agent 的完整规划系统，包括任务分解、计划生成、计划执行和计划优化四个核心组件。

## 模块结构

```
04-planning/
├── src/
│   ├── __init__.py              # 模块导出
│   ├── task_decomposition.py    # 任务分解
│   ├── plan_generation.py       # 计划生成
│   ├── plan_execution.py        # 计划执行
│   └── plan_refinement.py       # 计划优化
├── tests/                       # 单元测试 (139 tests)
├── notebooks/                   # Jupyter 教程
│   ├── 01_TaskDecomposition_tutorial.ipynb
│   ├── 02_PlanGeneration_tutorial.ipynb
│   └── 03_PlanExecution_tutorial.ipynb
├── 知识点.md                    # 详细知识点文档
└── README.md
```

## 快速开始

```python
from src import (
    Task, Plan, create_task, create_planner,
    create_executor, execute_plan, create_refinement
)

# 1. 创建任务
task = create_task("构建Web应用", "开发完整的Web应用程序", priority="high")

# 2. 生成计划
planner = create_planner("hierarchical")
plan = planner.generate("构建Web应用")

# 3. 执行计划
context = execute_plan(plan)

# 4. 查看结果
print(f"完成进度: {plan.progress:.0%}")
```

## 核心组件

| 组件 | 说明 | 主要类 |
|------|------|--------|
| 任务分解 | 将复杂任务分解为子任务 | `Task`, `TaskDecomposer`, `HierarchicalDecomposer` |
| 计划生成 | 生成可执行的计划 | `Plan`, `ForwardPlanner`, `HierarchicalPlanner` |
| 计划执行 | 执行计划中的任务 | `PlanExecutor`, `ExecutionPolicy`, `ExecutionContext` |
| 计划优化 | 失败恢复和计划优化 | `PlanRefinement`, `FailureRecovery`, `PlanOptimizer` |

## 运行测试

```bash
cd 14-agents-reasoning/04-planning
python -m pytest tests/ -v
```

## 学习路径

1. 阅读 `知识点.md` 了解理论基础
2. 运行 `notebooks/` 中的教程
3. 查看 `tests/` 中的测试用例了解 API 用法

## 依赖

- Python 3.8+
- 无外部依赖（纯 Python 实现）
