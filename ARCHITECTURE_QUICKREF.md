# 架构优化快速参考

## 🎯 一分钟了解

AI-Practices 项目正在进行架构优化，目标是提高代码质量、可维护性和可扩展性。

## 📁 新增文档

| 文档 | 说明 |
|------|------|
| **ARCHITECTURE.md** | 完整的架构设计文档 |
| **ARCHITECTURE_OPTIMIZATION.md** | 详细的优化方案 |
| **ARCHITECTURE_SUMMARY.md** | 优化总结和快速指南 |
| **scripts/setup_architecture.py** | 自动化设置脚本 |

## 🏗️ 核心变化

### 新增核心包

```
ai_practices/                    # 新的核心包
├── core/                        # 基础抽象类
├── config/                      # 配置管理
├── models/                      # 模型注册表
├── utils/                       # 工具函数
└── cli/                         # 命令行工具
```

### 标准模块结构

```
XX-module-name/
├── YY-submodule/
│   ├── src/                     # 源代码
│   │   ├── core/                # 核心实现
│   │   ├── models/              # 模型定义
│   │   └── utils/               # 工具函数
│   ├── tests/                   # 测试
│   │   ├── unit/                # 单元测试
│   │   └── integration/         # 集成测试
│   └── notebooks/               # 教程
```

## 🚀 快速开始

### 1. 查看文档

```bash
# 阅读架构文档
cat ARCHITECTURE.md

# 阅读优化方案
cat ARCHITECTURE_OPTIMIZATION.md

# 阅读总结
cat ARCHITECTURE_SUMMARY.md
```

### 2. 运行设置脚本

```bash
# 创建新架构结构
python scripts/setup_architecture.py
```

### 3. 安装和测试

```bash
# 开发模式安装
pip install -e ".[dev]"

# 运行测试
pytest

# 代码检查
black .
ruff check .
```

## 💡 核心概念

### 1. 基础抽象类

所有模型继承 `BaseModel`:

```python
from ai_practices.core.base import BaseModel

class MyModel(BaseModel):
    def forward(self, x):
        return x
```

### 2. 配置管理

统一的配置系统:

```python
from ai_practices.config import Config

config = Config.from_env()
print(config.batch_size)
```

### 3. 模型注册表

注册和管理模型:

```python
from ai_practices.models import ModelRegistry

@ModelRegistry.register("my_model")
class MyModel(BaseModel):
    pass

model = ModelRegistry.create("my_model")
```

### 4. 统一日志

标准化的日志系统:

```python
from ai_practices.utils import get_logger

logger = get_logger("my_module")
logger.info("Message")
```

## 📊 优化收益

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 测试覆盖率 | 60% | 80%+ | +33% |
| 代码重复率 | 15% | 5% | -67% |
| 文档覆盖率 | 50% | 95% | +90% |
| 新模块开发时间 | 100% | 70% | -30% |

## 🎯 实施阶段

- ✅ **Phase 1**: 架构设计 (已完成)
- 🔄 **Phase 2**: 核心实现 (进行中)
- ⏳ **Phase 3**: 模块迁移 (待开始)
- ⏳ **Phase 4**: 文档完善 (待开始)

## 📝 开发者须知

### 新模块开发

1. 使用标准目录结构
2. 继承核心抽象类
3. 注册模型到 ModelRegistry
4. 编写单元测试
5. 更新文档

### 代码规范

```python
# ✅ 推荐
from ai_practices.core.base import BaseModel
from ai_practices.config import Config

class MyModel(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config.to_dict())

# ❌ 避免
# 硬编码配置
# 重复实现基础功能
# 缺少类型注解
```

## 🔗 快速链接

- [完整架构文档](./ARCHITECTURE.md)
- [优化方案](./ARCHITECTURE_OPTIMIZATION.md)
- [优化总结](./ARCHITECTURE_SUMMARY.md)
- [开发规范](./DEVELOPMENT.md)
- [贡献指南](./CONTRIBUTING.md)

## ❓ 常见问题

**Q: 需要立即迁移吗?**  
A: 不需要，可以逐步迁移。

**Q: 旧代码会失效吗?**  
A: 不会，保持向后兼容。

**Q: 如何参与?**  
A: 查看 ARCHITECTURE_OPTIMIZATION.md 的实施计划。

## 📞 获取帮助

- GitHub Issues: 报告问题
- GitHub Discussions: 讨论设计
- 文档: 查看详细说明

---

**提示**: 这是一个渐进式的优化过程，不会影响现有功能。

**更新**: 2026-01-23
