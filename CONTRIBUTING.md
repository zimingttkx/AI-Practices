# Contributing to AI-Practices

[English](#english) | [中文](#中文)

---

## English

Thank you for your interest in contributing to AI-Practices! This document provides guidelines and steps for contributing.

### Ways to Contribute

- **Report Bugs**: Use [GitHub Issues](https://github.com/zimingttkx/AI-Practices/issues) with the bug report template
- **Suggest Features**: Open a feature request issue or start a [Discussion](https://github.com/zimingttkx/AI-Practices/discussions)
- **Improve Documentation**: Fix typos, clarify explanations, or add examples
- **Submit Code**: Add new notebooks, fix bugs, or enhance existing content
- **Share Knowledge**: Help answer questions in Discussions

### Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Practices.git
   cd AI-Practices
   ```

2. **Set up the environment**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate ai-practices

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

### Code Standards

#### Python Code
- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

#### Jupyter Notebooks
- Clear all outputs before committing (or keep essential outputs)
- Include markdown cells explaining the code
- Use consistent cell structure:
  1. Title and introduction
  2. Imports
  3. Data loading
  4. Implementation
  5. Results and visualization
  6. Summary

#### Commit Messages
- Use clear, descriptive commit messages
- Follow the format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(cv): add ResNet implementation notebook
fix(nlp): correct tokenization in BERT example
docs(readme): update installation instructions
```

### Pull Request Process

1. **Before submitting**
   - Ensure your code follows the style guidelines
   - Test your changes locally
   - Update documentation if needed
   - Rebase on the latest `main` branch

2. **Submit PR**
   - Fill out the PR template completely
   - Link related issues
   - Request review from maintainers

3. **Review process**
   - Address reviewer feedback
   - Keep the PR focused on a single change
   - Be patient and respectful

### Project Structure

```
AI-Practices/
├── 01-foundations/              # ML basics
├── 02-neural-networks/          # Deep learning fundamentals
├── 03-computer-vision/          # CNN and image processing
├── 04-sequence-models/          # RNN, LSTM, Transformers
├── 05-advanced-topics/          # Specialized topics
├── 06-generative-models/        # GANs, VAE, Diffusion
├── 07-reinforcement-learning/   # RL algorithms
├── 08-theory-notes/             # Mathematical foundations
├── 09-practical-projects/       # End-to-end projects
├── 10-large-language-models/    # LLM, RAG, Agents
├── 11-multimodal-learning/      # CLIP, Diffusion, Audio
├── 12-deployment-optimization/  # Quantization, Serving, MLOps
├── utils/                       # Shared utilities
└── docs/                        # Documentation
```

### Questions?

- Open a [Discussion](https://github.com/zimingttkx/AI-Practices/discussions)
- Check existing issues and discussions first

---

## 中文

感谢您有兴趣为 AI-Practices 做出贡献！本文档提供了贡献的指南和步骤。

### 贡献方式

- **报告 Bug**：使用 [GitHub Issues](https://github.com/zimingttkx/AI-Practices/issues) 的 bug 报告模板
- **建议功能**：开启功能请求 issue 或在 [Discussions](https://github.com/zimingttkx/AI-Practices/discussions) 中讨论
- **改进文档**：修正错别字、澄清说明或添加示例
- **提交代码**：添加新 notebook、修复 bug 或增强现有内容
- **分享知识**：在 Discussions 中帮助回答问题

### 开始贡献

1. **Fork 仓库**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Practices.git
   cd AI-Practices
   ```

2. **设置环境**
   ```bash
   # 使用 conda（推荐）
   conda env create -f environment.yml
   conda activate ai-practices

   # 或使用 pip
   pip install -r requirements.txt
   ```

3. **创建分支**
   ```bash
   git checkout -b feature/功能名称
   # 或
   git checkout -b fix/bug修复描述
   ```

### 代码规范

#### Python 代码
- 遵循 [PEP 8](https://pep8.org/) 风格指南
- 使用有意义的变量和函数名
- 为函数和类添加文档字符串
- 保持函数专注和模块化

#### Jupyter Notebooks
- 提交前清除所有输出（或保留必要输出）
- 包含解释代码的 markdown 单元格
- 使用一致的单元格结构：
  1. 标题和介绍
  2. 导入
  3. 数据加载
  4. 实现
  5. 结果和可视化
  6. 总结

#### 提交信息
- 使用清晰、描述性的提交信息
- 遵循格式：`type(scope): description`
- 类型：`feat`、`fix`、`docs`、`style`、`refactor`、`test`、`chore`

示例：
```
feat(cv): 添加 ResNet 实现 notebook
fix(nlp): 修正 BERT 示例中的分词问题
docs(readme): 更新安装说明
```

### Pull Request 流程

1. **提交前**
   - 确保代码遵循风格指南
   - 在本地测试更改
   - 如需要则更新文档
   - 在最新的 `main` 分支上 rebase

2. **提交 PR**
   - 完整填写 PR 模板
   - 链接相关 issue
   - 请求维护者审查

3. **审查流程**
   - 处理审查反馈
   - 保持 PR 专注于单一更改
   - 保持耐心和尊重

### 有问题？

- 开启 [Discussion](https://github.com/zimingttkx/AI-Practices/discussions)
- 先查看现有的 issues 和 discussions

---

Thank you for contributing! 感谢您的贡献！
