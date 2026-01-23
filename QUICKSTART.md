# 快速开始指南

## 环境设置

### 方式 1: Conda (推荐)

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate ai-practices

# 验证安装
python --version
pip list
```

### 方式 2: pip + venv

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装开发工具
pip install -e ".[dev]"
```

### 方式 3: Docker

```bash
# 开发环境
docker-compose up dev

# Jupyter Lab
docker-compose up jupyter

# 运行测试
docker-compose up test
```

## 开发工具设置

### Pre-commit Hooks

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 hooks
pre-commit install

# 手动运行
pre-commit run --all-files
```

### 代码格式化

```bash
# Black 格式化
black .

# Ruff 检查和修复
ruff check . --fix

# 类型检查
mypy .
```

## 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块
pytest 07-reinforcement-learning/

# 带覆盖率
pytest --cov=. --cov-report=html

# 并行运行
pytest -n auto
```

## 常用命令

### 依赖管理

```bash
# 检查依赖冲突
pip check

# 查看过期包
pip list --outdated

# 更新包
pip install --upgrade package-name
```

### Git 工作流

```bash
# 创建功能分支
git checkout -b feature/your-feature

# 提交代码
git add .
git commit -m "feat(module): description"

# 推送到远程
git push origin feature/your-feature
```

### Jupyter

```bash
# 启动 Jupyter Lab
jupyter lab

# 启动 Notebook
jupyter notebook

# 清除输出
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

## 项目结构

```
AI-Practices/
├── 01-foundations/              # 机器学习基础
├── 02-neural-networks/          # 神经网络
├── 03-computer-vision/          # 计算机视觉
├── 04-sequence-models/          # 序列模型
├── 05-advanced-topics/          # 高级专题
├── 06-generative-models/        # 生成模型
├── 07-reinforcement-learning/   # 强化学习
├── 08-theory-notes/             # 理论笔记
├── 09-practical-projects/       # 实战项目
├── 10-large-language-models/    # 大语言模型
├── 11-multimodal-learning/      # 多模态学习
├── 12-deployment-optimization/  # 部署优化
├── 13-distributed-training/     # 分布式训练
└── 14-agents-reasoning/         # 智能体推理
```

## 常见问题

### Q: 依赖安装失败？
```bash
# 升级 pip
pip install --upgrade pip setuptools wheel

# 清除缓存
pip cache purge

# 重新安装
pip install -r requirements.txt
```

### Q: PyTorch 安装问题？
```bash
# CPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Q: Jupyter 内核问题？
```bash
# 安装内核
python -m ipykernel install --user --name=ai-practices

# 列出内核
jupyter kernelspec list

# 删除内核
jupyter kernelspec uninstall ai-practices
```

## 资源链接

- [项目文档](./README.md)
- [开发指南](./DEVELOPMENT.md)
- [贡献指南](./CONTRIBUTING.md)
- [更新日志](./CHANGELOG.md)
- [路线图](./ROADMAP.md)

## 获取帮助

- [GitHub Issues](https://github.com/zimingttkx/AI-Practices/issues)
- [GitHub Discussions](https://github.com/zimingttkx/AI-Practices/discussions)
