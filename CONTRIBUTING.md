# Contributing to AI-Practices

[English](#english) | [中文](#中文)

---

## English

Thank you for your interest in contributing to AI-Practices!

### Ways to Contribute

- **Report Bugs**: Use [GitHub Issues](https://github.com/zimingttkx/AI-Practices/issues)
- **Suggest Features**: Open a feature request or start a [Discussion](https://github.com/zimingttkx/AI-Practices/discussions)
- **Improve Documentation**: Fix typos, clarify explanations, or add examples
- **Submit Code**: Add new notebooks, fix bugs, or enhance existing content
- **Share Knowledge**: Help answer questions in Discussions

### Getting Started

1. **Fork and clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Practices.git
   cd AI-Practices
   ```

2. **Setup environment**
   ```bash
   conda env create -f environment.yml
   conda activate ai-practices
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Code Standards

#### Python Code
- Follow [PEP 8](https://pep8.org/)
- Use meaningful names
- Add **bilingual docstrings** (Chinese first line, then English; see below)
- Keep functions modular
- Add `from __future__ import annotations` at the top of every `.py` file
- Use `logging` instead of `print()` in library code

#### Docstring Convention (Bilingual)

All docstrings should follow this bilingual format:

```python
def train_epoch(model, dataloader, criterion, optimizer):
    """训练一个 epoch。/ Train for one epoch.

    执行单轮前向传播、损失计算、反向传播和参数更新。

    Args:
        model: 待训练的 PyTorch 模型 / PyTorch model to train
        dataloader: 训练数据加载器 / training data loader
        criterion: 损失函数 / loss function
        optimizer: 优化器 / optimizer

    Returns:
        float: 平均损失 / average loss
    """
```

#### Jupyter Notebooks
- Clear outputs before committing
- Include markdown explanations
- Use consistent structure
- For TF→PyTorch dual versions, use `_pytorch` suffix (e.g., `01-LinearRegression_pytorch.ipynb`)

#### Commit Messages
- Format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(cv): add ResNet implementation
fix(nlp): correct tokenization in BERT
docs(readme): update installation
```

### CI/CD Requirements

All pull requests must pass the CI pipeline:

1. **Lint checks**: `ruff check .` and `black --check .` must pass
2. **Tests**: `pytest` on Python 3.9, 3.10, 3.11 must pass
3. **Pre-commit hooks**: Install and run before pushing:
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```

### Pull Request Process

1. **Before submitting**
   - Follow style guidelines
   - Run pre-commit hooks locally
   - Run tests locally: `pytest`
   - Update documentation
   - Rebase on latest `main`

2. **Submit PR**
   - Fill out template
   - Link related issues
   - Request review

3. **Review process**
   - Address feedback
   - Keep PR focused
   - Be patient

### Questions?

- Open a [Discussion](https://github.com/zimingttkx/AI-Practices/discussions)
- Check existing issues first

---

## 中文

感谢您有兴趣为 AI-Practices 做出贡献！

### 贡献方式

- **报告 Bug**：使用 [GitHub Issues](https://github.com/zimingttkx/AI-Practices/issues)
- **建议功能**：开启功能请求或在 [Discussions](https://github.com/zimingttkx/AI-Practices/discussions) 中讨论
- **改进文档**：修正错别字、澄清说明或添加示例
- **提交代码**：添加新 notebook、修复 bug 或增强现有内容
- **分享知识**：在 Discussions 中帮助回答问题

### 开始贡献

1. **Fork 并克隆**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Practices.git
   cd AI-Practices
   ```

2. **设置环境**
   ```bash
   conda env create -f environment.yml
   conda activate ai-practices
   ```

3. **创建分支**
   ```bash
   git checkout -b feature/功能名称
   ```

### 代码规范

#### Python 代码
- 遵循 [PEP 8](https://pep8.org/)
- 使用有意义的命名
- 添加**中英双语文档字符串**（中文首行，英文次行；见英文部分示例）
- 保持函数模块化
- 每个 `.py` 文件顶部添加 `from __future__ import annotations`
- 库代码中使用 `logging` 替代 `print()`

#### Jupyter Notebooks
- 提交前清除输出
- 包含 markdown 说明
- 使用一致的结构
- TF→PyTorch 双版本使用 `_pytorch` 后缀（如 `01-LinearRegression_pytorch.ipynb`）

#### 提交信息
- 格式：`type(scope): description`
- 类型：`feat`、`fix`、`docs`、`style`、`refactor`、`test`、`chore`

示例：
```
feat(cv): 添加 ResNet 实现
fix(nlp): 修正 BERT 分词问题
docs(readme): 更新安装说明
```

### CI/CD 要求

所有 Pull Request 必须通过 CI 流水线：

1. **Lint 检查**：`ruff check .` 和 `black --check .` 必须通过
2. **测试**：Python 3.9、3.10、3.11 上的 `pytest` 必须通过
3. **Pre-commit 钩子**：推送前安装并运行：
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```

### Pull Request 流程

1. **提交前**
   - 遵循风格指南
   - 本地运行 pre-commit 钩子
   - 本地运行测试：`pytest`
   - 更新文档
   - 在最新 `main` 上 rebase

2. **提交 PR**
   - 填写模板
   - 链接相关 issue
   - 请求审查

3. **审查流程**
   - 处理反馈
   - 保持 PR 专注
   - 保持耐心

### 有问题？

- 开启 [Discussion](https://github.com/zimingttkx/AI-Practices/discussions)
- 先查看现有 issues

---

Thank you for contributing!
