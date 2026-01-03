# 开发准则

本文档定义了 AI-Practices 项目的代码风格、文件命名和目录结构规范。

> **最后更新**: 2026-01-03

---

## 目录结构

### 模块命名

```
XX-module-name/
├── YY-submodule-name/
│   ├── src/                    # 源代码
│   │   ├── __init__.py
│   │   └── module.py
│   ├── tests/                  # 单元测试
│   │   └── test_module.py
│   ├── notebooks/              # Jupyter 教程
│   │   └── 01_tutorial.ipynb
│   ├── README.md               # 模块说明
│   └── 知识点.md                # 知识点文档
└── README.md                   # 模块总览
```

### 命名规范

| 类型 | 格式 | 示例 |
|:-----|:-----|:-----|
| 模块目录 | `XX-kebab-case` | `01-foundations`, `10-large-language-models` |
| 子模块目录 | `YY-kebab-case` | `01-training-models`, `02-classification` |
| Python 文件 | `snake_case.py` | `transformer_architecture.py`, `clip.py` |
| 测试文件 | `test_*.py` | `test_clip.py`, `test_transformer.py` |
| Notebook | `NN_PascalCase_tutorial.ipynb` | `01_CLIP_tutorial.ipynb` |
| 类名 | `PascalCase` | `TransformerEncoder`, `CLIPModel` |
| 函数名 | `snake_case` | `compute_loss`, `forward_pass` |
| 常量 | `UPPER_SNAKE_CASE` | `MAX_LENGTH`, `DEFAULT_LR` |

---

## 代码风格

### Python 代码规范

遵循 PEP 8 风格指南，使用 Black 格式化，行宽 100 字符。

```python
"""
模块文档字符串

简要描述模块功能和主要内容。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ModelConfig:
    """模型配置类
    
    Attributes:
        hidden_size: 隐藏层维度
        num_layers: 层数
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


class Model(nn.Module):
    """模型实现
    
    核心思想:
        简要描述模型的核心原理和创新点。
    
    数学原理:
        相关的数学公式和推导。
    
    Args:
        config: 模型配置
    
    Example:
        >>> config = ModelConfig(hidden_size=512)
        >>> model = Model(config)
        >>> output = model(input_tensor)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layer = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
        
        Returns:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        return self.layer(x)


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """计算损失函数
    
    Args:
        predictions: 预测值
        targets: 目标值
        reduction: 归约方式 ("mean", "sum", "none")
    
    Returns:
        损失值
    """
    return nn.functional.mse_loss(predictions, targets, reduction=reduction)
```

### 类型注解

所有公开函数和方法必须包含类型注解：

```python
from typing import Dict, List, Optional, Tuple, Union

def process_data(
    data: List[Dict[str, torch.Tensor]],
    batch_size: int = 32,
    shuffle: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```

### 文档字符串

使用 Google 风格的文档字符串：

```python
def function(arg1: int, arg2: str) -> bool:
    """函数简要描述
    
    详细描述（可选）。
    
    Args:
        arg1: 参数1说明
        arg2: 参数2说明
    
    Returns:
        返回值说明
    
    Raises:
        ValueError: 异常说明
    
    Example:
        >>> result = function(1, "test")
        >>> print(result)
        True
    """
```

## 测试规范

### 测试文件结构

```python
"""模块测试"""

import pytest
import torch

from module import Model, ModelConfig


class TestModelConfig:
    """ModelConfig 测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ModelConfig()
        assert config.hidden_size == 768
        assert config.num_layers == 12
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = ModelConfig(hidden_size=512, num_layers=6)
        assert config.hidden_size == 512


class TestModel:
    """Model 测试类"""
    
    @pytest.fixture
    def model(self):
        """创建测试模型"""
        config = ModelConfig(hidden_size=64, num_layers=2)
        return Model(config)
    
    def test_forward_shape(self, model):
        """测试前向传播输出形状"""
        x = torch.randn(2, 10, 64)
        output = model(x)
        assert output.shape == (2, 10, 64)
    
    def test_forward_dtype(self, model):
        """测试输出数据类型"""
        x = torch.randn(2, 10, 64)
        output = model(x)
        assert output.dtype == torch.float32
```

### 测试命名

- 测试类: `Test{ClassName}`
- 测试方法: `test_{method_name}_{scenario}`

## Notebook 规范

### 结构

1. **标题和介绍** - Markdown 单元格
2. **导入** - 代码单元格
3. **数据准备** - 代码单元格
4. **实现** - 代码和 Markdown 交替
5. **结果展示** - 代码单元格
6. **总结** - Markdown 单元格

### 示例

```markdown
# 模型名称 教程

本教程介绍...

## 1. 环境准备
```

```python
import torch
import torch.nn as nn
```

```markdown
## 2. 模型实现

### 2.1 核心组件
```

## Git 提交规范

### 提交信息格式

```
type(scope): description

[optional body]
```

### 类型

| 类型 | 说明 |
|:-----|:-----|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `style` | 代码格式 |
| `refactor` | 重构 |
| `test` | 测试 |
| `chore` | 构建/工具 |

### 示例

```
feat(llm): add LoRA fine-tuning implementation
fix(cv): correct batch normalization in ResNet
docs(readme): update module overview
test(rl): add DQN unit tests
```

## 依赖管理

### 添加依赖

1. 核心依赖添加到 `pyproject.toml` 的 `dependencies`
2. 可选依赖添加到 `[project.optional-dependencies]`
3. 同步更新 `requirements.txt`

### 依赖分组

```toml
[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]
llm = ["transformers", "peft", "accelerate"]
full = ["tensorflow", "keras", "xgboost"]
```

## 代码审查清单

- [ ] 代码遵循 PEP 8 规范
- [ ] 所有公开函数有类型注解
- [ ] 所有公开函数有文档字符串
- [ ] 单元测试覆盖核心功能
- [ ] 无硬编码的敏感信息
- [ ] 无冗余的调试代码
- [ ] 提交信息格式正确

---

## 环境配置

### 本地开发

```bash
# 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 创建虚拟环境
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# 安装依赖
pip install -r requirements.txt
pip install -e ".[dev]"

# 运行测试
pytest -v
```

### Docker 开发

```bash
# 构建镜像
docker-compose build

# 启动开发环境
docker-compose up dev

# 启动 Jupyter Lab
docker-compose up jupyter
```

---

## 质量保证

### 代码检查

```bash
# 格式化代码
black .

# 代码检查
ruff check .

# 类型检查
mypy .
```

### 测试覆盖

```bash
# 运行所有测试
pytest

# 带覆盖率报告
pytest --cov=. --cov-report=html

# 运行特定模块测试
pytest 13-distributed-training/
```

---

## 常见问题

### Q: 如何添加新模块？

1. 创建目录结构 `XX-module-name/YY-submodule/`
2. 添加 `src/`、`tests/`、`notebooks/` 目录
3. 更新 `pyproject.toml` 的 `testpaths`
4. 更新 `.github/CODEOWNERS`
5. 更新 README.md 模块列表

### Q: 如何运行特定测试？

```bash
# 运行单个测试文件
pytest path/to/test_file.py

# 运行特定测试函数
pytest path/to/test_file.py::test_function_name

# 运行特定测试类
pytest path/to/test_file.py::TestClassName
```
