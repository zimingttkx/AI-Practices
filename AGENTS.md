# AI-Practices 项目 Agent 配置

这是一个 AI/ML 学习与实践项目，包含 14 个核心模块，涵盖从基础到前沿的机器学习技术。

## 核心命令

- 类型检查和 Lint: `ruff check . --fix && mypy .`
- 格式化: `black . --line-length=100`
- 运行测试: `pytest -v --tb=short`
- 运行单个测试: `pytest <path> -v`
- 依赖检查: `pip check`
- 安装开发依赖: `pip install -e ".[dev]"`

## 项目结构

```
AI-Practices/
├── 01-foundations/          # 数学基础、Python、NumPy
├── 02-neural-networks/      # 感知机、MLP、反向传播
├── 03-computer-vision/      # CNN、目标检测、图像分割
├── 04-sequence-models/      # RNN、LSTM、Transformer
├── 05-advanced-topics/      # 优化、正则化、迁移学习
├── 06-generative-models/    # GAN、VAE、扩散模型
├── 07-reinforcement-learning/ # Q-Learning、PPO、A3C
├── 08-theory-notes/         # 理论笔记
├── 09-practical-projects/   # 实战项目
├── 10-large-language-models/ # LLM、提示工程、微调
├── 11-multimodal-learning/  # 多模态、CLIP、视觉语言
├── 12-deployment-optimization/ # 量化、剪枝、部署
├── 13-distributed-training/ # 分布式训练、DeepSpeed
├── 14-agents-reasoning/     # AI Agents、推理、规划
├── utils/                   # 共享工具函数
├── scripts/                 # 自动化脚本
└── docs/                    # 文档
```

## 开发规范

### 代码风格
- Python 严格类型注解
- Black 格式化，行宽 100
- Ruff 作为 linter
- 使用 pathlib 而非 os.path
- 优先使用 f-string
- 函数和类必须有 docstring

### 命名规范
- 文件名: snake_case.py
- 类名: PascalCase
- 函数/变量: snake_case
- 常量: UPPER_SNAKE_CASE
- 私有: _leading_underscore

### 模块结构标准
每个模块应包含:
```
XX-module-name/
├── README.md           # 模块说明
├── src/               # 源代码
│   ├── __init__.py
│   ├── model.py       # 模型定义
│   ├── train.py       # 训练逻辑
│   └── utils.py       # 工具函数
├── tests/             # 测试
│   └── test_model.py
├── notebooks/         # Jupyter notebooks
└── docs/              # 文档
```

## Git 工作流

- 分支命名: `feature/<slug>` 或 `fix/<slug>`
- 提交信息格式: `<type>(<scope>): <description>`
  - type: feat, fix, docs, test, refactor, chore
  - scope: 模块名或组件名
- 提交前必须通过 lint 和测试
- PR 必须有清晰的描述

## 技术栈

### 核心框架
- PyTorch 2.5+ (主要框架)
- TensorFlow/Keras 2.18+ (部分示例)
- Transformers 4.47+ (NLP/LLM)
- JAX/Flax (高级优化)

### 常用库
- NumPy, Pandas, Matplotlib
- scikit-learn, OpenCV
- Weights & Biases (实验跟踪)
- Hydra (配置管理)

### 开发工具
- pytest (测试)
- black, ruff, mypy (代码质量)
- pre-commit (Git hooks)

## 常见模式

### 模型定义
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化层...
    
    def forward(self, x):
        # 前向传播...
        return output
```

### 训练循环
```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        # 验证...
```

### 配置管理
```python
from dataclasses import dataclass

@dataclass
class Config:
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
```

## 注意事项

### 性能优化
- 使用 `torch.compile()` 加速 (PyTorch 2.0+)
- 启用混合精度训练 `torch.cuda.amp`
- 使用 DataLoader 的 `num_workers` 和 `pin_memory`
- 避免在训练循环中创建新张量

### 常见陷阱
- 忘记 `model.eval()` 和 `torch.no_grad()`
- GPU 内存泄漏 (detach tensors)
- 随机种子未固定导致不可复现
- 学习率未正确调度

### 测试要求
- 新功能必须有对应测试
- 测试覆盖率目标: 80%+
- 使用 pytest fixtures 共享设置
- Mock 外部依赖

## 外部资源

- 项目文档: `docs/` 目录
- 开发指南: `DEVELOPMENT.md`
- 贡献指南: `CONTRIBUTING.md`
- 路线图: `ROADMAP.md`

## Agent 行为优化

### 代码生成优先级
1. 首先理解现有代码结构和模式
2. 遵循项目既有风格
3. 添加类型注解
4. 编写必要的测试
5. 更新相关文档

### 搜索策略
- 使用 ripgrep (rg) 搜索代码
- 使用 fd 查找文件
- 优先搜索 src/ 目录
- 检查 tests/ 了解预期行为

### 回复风格
- 简洁直接，不要冗余
- 代码注释只添加必要的
- 不主动创建或更新文档（除非明确要求）
- 完成任务后简短总结 (1-4 句)

### 自动验证
- 修改代码后自动运行相关测试
- 检查类型错误
- 验证导入是否正确
- 确保不破坏现有功能
