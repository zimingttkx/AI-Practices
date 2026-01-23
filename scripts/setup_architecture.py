#!/usr/bin/env python3
"""
项目架构优化脚本

此脚本用于自动化创建新的项目架构结构。
"""

import os
from pathlib import Path
from typing import List


def create_directory(path: Path, description: str = ""):
    """创建目录"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {path}")
        if description:
            print(f"  说明: {description}")
    else:
        print(f"- 目录已存在: {path}")


def create_file(path: Path, content: str = "", description: str = ""):
    """创建文件"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 创建文件: {path}")
        if description:
            print(f"  说明: {description}")
    else:
        print(f"- 文件已存在: {path}")


def create_core_package(root: Path):
    """创建核心包结构"""
    print("\n=== 创建核心包结构 ===")
    
    core_dir = root / "ai_practices"
    
    # 创建主包目录
    create_directory(core_dir, "核心包根目录")
    
    # __init__.py
    init_content = '''"""
AI-Practices: 系统化人工智能学习与实践平台

这是一个模块化的 AI/ML 学习平台，提供从基础到高级的完整学习路径。
"""

__version__ = "1.0.0"
__author__ = "zimingttkx"

from ai_practices.config import Config
from ai_practices.core.base import BaseModel, BaseTrainer

__all__ = [
    "Config",
    "BaseModel",
    "BaseTrainer",
]
'''
    create_file(core_dir / "__init__.py", init_content, "包初始化文件")
    
    # __version__.py
    version_content = '''"""版本信息"""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)
'''
    create_file(core_dir / "__version__.py", version_content, "版本信息")


def create_core_module(root: Path):
    """创建 core 模块"""
    print("\n=== 创建 core 模块 ===")
    
    core_dir = root / "ai_practices" / "core"
    create_directory(core_dir, "核心抽象类模块")
    
    # __init__.py
    init_content = '''"""核心抽象类"""

from ai_practices.core.base import BaseModel, BaseTrainer, BaseEvaluator

__all__ = ["BaseModel", "BaseTrainer", "BaseEvaluator"]
'''
    create_file(core_dir / "__init__.py", init_content)
    
    # base.py
    base_content = '''"""基础抽象类"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """模型基类
    
    所有模型都应该继承此类并实现必要的方法。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化模型
        
        Args:
            config: 模型配置字典
        """
        self.config = config or {}
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """前向传播
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            模型输出
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载模型
        
        Args:
            path: 模型路径
        """
        pass


class BaseTrainer(ABC):
    """训练器基类"""
    
    def __init__(self, model: BaseModel, config: Optional[Dict] = None):
        """初始化训练器
        
        Args:
            model: 要训练的模型
            config: 训练配置
        """
        self.model = model
        self.config = config or {}
    
    @abstractmethod
    def train_step(self, batch) -> Dict[str, float]:
        """单步训练
        
        Args:
            batch: 训练批次数据
        
        Returns:
            训练指标字典
        """
        pass
    
    @abstractmethod
    def validate_step(self, batch) -> Dict[str, float]:
        """单步验证
        
        Args:
            batch: 验证批次数据
        
        Returns:
            验证指标字典
        """
        pass
    
    def fit(self, train_loader, val_loader=None, epochs=10):
        """完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
        """
        for epoch in range(epochs):
            # 训练
            train_metrics = []
            for batch in train_loader:
                metrics = self.train_step(batch)
                train_metrics.append(metrics)
            
            # 验证
            if val_loader:
                val_metrics = []
                for batch in val_loader:
                    metrics = self.validate_step(batch)
                    val_metrics.append(metrics)
                
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train: {train_metrics[-1]} - "
                      f"Val: {val_metrics[-1]}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train: {train_metrics[-1]}")


class BaseEvaluator(ABC):
    """评估器基类"""
    
    def __init__(self, model: BaseModel):
        """初始化评估器
        
        Args:
            model: 要评估的模型
        """
        self.model = model
    
    @abstractmethod
    def evaluate(self, data_loader) -> Dict[str, float]:
        """评估模型
        
        Args:
            data_loader: 数据加载器
        
        Returns:
            评估指标字典
        """
        pass
'''
    create_file(core_dir / "base.py", base_content, "基础抽象类")


def create_config_module(root: Path):
    """创建配置模块"""
    print("\n=== 创建配置模块 ===")
    
    config_dir = root / "ai_practices" / "config"
    create_directory(config_dir, "配置管理模块")
    
    # __init__.py
    init_content = '''"""配置管理"""

from ai_practices.config.settings import Config

__all__ = ["Config"]
'''
    create_file(config_dir / "__init__.py", init_content)
    
    # settings.py
    settings_content = '''"""全局配置"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class Config:
    """全局配置类"""
    
    # 路径配置
    root_dir: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    models_dir: Path = field(default_factory=lambda: Path.cwd() / "models")
    logs_dir: Path = field(default_factory=lambda: Path.cwd() / "logs")
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    device: str = "cuda"
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置
        
        Returns:
            配置实例
        """
        return cls(
            root_dir=Path(os.getenv("AI_PRACTICES_ROOT", Path.cwd())),
            data_dir=Path(os.getenv("AI_PRACTICES_DATA", Path.cwd() / "data")),
            models_dir=Path(os.getenv("AI_PRACTICES_MODELS", Path.cwd() / "models")),
            logs_dir=Path(os.getenv("AI_PRACTICES_LOGS", Path.cwd() / "logs")),
            device=os.getenv("AI_PRACTICES_DEVICE", "cuda"),
        )
    
    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """从 YAML 文件加载配置
        
        Args:
            path: 配置文件路径
        
        Returns:
            配置实例
        """
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_dict(self) -> dict:
        """转换为字典
        
        Returns:
            配置字典
        """
        return {
            "root_dir": str(self.root_dir),
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "logs_dir": str(self.logs_dir),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "device": self.device,
            "log_level": self.log_level,
        }


# 全局配置实例
config = Config.from_env()
'''
    create_file(config_dir / "settings.py", settings_content, "配置类")


def create_models_module(root: Path):
    """创建模型模块"""
    print("\n=== 创建模型模块 ===")
    
    models_dir = root / "ai_practices" / "models"
    create_directory(models_dir, "模型管理模块")
    
    # __init__.py
    init_content = '''"""模型管理"""

from ai_practices.models.registry import ModelRegistry

__all__ = ["ModelRegistry"]
'''
    create_file(models_dir / "__init__.py", init_content)
    
    # registry.py
    registry_content = '''"""模型注册表"""

from typing import Dict, Type, Callable
from ai_practices.core.base import BaseModel


class ModelRegistry:
    """模型注册表
    
    用于注册和管理所有模型类。
    
    Example:
        >>> @ModelRegistry.register("my_model")
        >>> class MyModel(BaseModel):
        >>>     def forward(self, x):
        >>>         return x
        >>>
        >>> model = ModelRegistry.create("my_model")
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """注册模型装饰器
        
        Args:
            name: 模型名称
        
        Returns:
            装饰器函数
        
        Raises:
            ValueError: 如果模型名称已存在
        """
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            if name in cls._registry:
                raise ValueError(f"Model '{name}' already registered")
            cls._registry[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """获取模型类
        
        Args:
            name: 模型名称
        
        Returns:
            模型类
        
        Raises:
            ValueError: 如果模型不存在
        """
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found in registry")
        return cls._registry[name]
    
    @classmethod
    def list_models(cls) -> list:
        """列出所有注册的模型
        
        Returns:
            模型名称列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """创建模型实例
        
        Args:
            name: 模型名称
            **kwargs: 模型参数
        
        Returns:
            模型实例
        """
        model_class = cls.get(name)
        return model_class(**kwargs)
'''
    create_file(models_dir / "registry.py", registry_content, "模型注册表")


def create_utils_structure(root: Path):
    """创建 utils 结构"""
    print("\n=== 重构 utils 目录 ===")
    
    utils_dir = root / "ai_practices" / "utils"
    create_directory(utils_dir, "工具函数模块")
    
    # __init__.py
    init_content = '''"""工具函数"""

from ai_practices.utils.logging import get_logger

__all__ = ["get_logger"]
'''
    create_file(utils_dir / "__init__.py", init_content)
    
    # logging.py
    logging_content = '''"""日志工具"""

import logging
from pathlib import Path
from typing import Optional


_loggers = {}


def get_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO"
) -> logging.Logger:
    """获取日志器
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别
    
    Returns:
        日志器实例
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger
'''
    create_file(utils_dir / "logging.py", logging_content, "日志工具")


def create_tests_structure(root: Path):
    """创建测试结构"""
    print("\n=== 创建测试结构 ===")
    
    tests_dir = root / "tests"
    create_directory(tests_dir, "测试根目录")
    
    # conftest.py
    conftest_content = '''"""Pytest 配置和 fixtures"""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data():
    """示例数据"""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def sample_tensor():
    """示例张量"""
    return torch.randn(32, 3, 224, 224)


@pytest.fixture
def temp_dir(tmp_path):
    """临时目录"""
    return tmp_path


@pytest.fixture
def config():
    """测试配置"""
    from ai_practices.config import Config
    return Config(
        batch_size=16,
        epochs=2,
        device="cpu"
    )


@pytest.fixture
def mock_model():
    """模拟模型"""
    from ai_practices.core.base import BaseModel
    
    class MockModel(BaseModel):
        def forward(self, x):
            return x
        
        def save(self, path):
            pass
        
        def load(self, path):
            pass
    
    return MockModel()
'''
    create_file(tests_dir / "conftest.py", conftest_content, "Pytest 配置")
    
    # 创建测试子目录
    create_directory(tests_dir / "unit", "单元测试")
    create_directory(tests_dir / "integration", "集成测试")
    create_directory(tests_dir / "fixtures", "测试数据")


def create_setup_files(root: Path):
    """创建安装文件"""
    print("\n=== 创建安装文件 ===")
    
    # setup.py
    setup_content = '''"""安装脚本"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# 读取版本
version_file = Path(__file__).parent / "ai_practices" / "__version__.py"
version = {}
if version_file.exists():
    exec(version_file.read_text(), version)

setup(
    name="ai-practices",
    version=version.get("__version__", "1.0.0"),
    author="zimingttkx",
    description="Systematic AI/ML learning platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zimingttkx/AI-Practices",
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.0",
        "torch>=2.5.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "black>=24.10.0",
            "ruff>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-practices=ai_practices.cli.commands:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
'''
    create_file(root / "setup.py", setup_content, "安装脚本")


def main():
    """主函数"""
    print("=" * 60)
    print("AI-Practices 项目架构优化脚本")
    print("=" * 60)
    
    # 获取项目根目录
    root = Path.cwd()
    print(f"\n项目根目录: {root}")
    
    # 确认执行
    response = input("\n是否继续创建新的架构结构? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 创建结构
    try:
        create_core_package(root)
        create_core_module(root)
        create_config_module(root)
        create_models_module(root)
        create_utils_structure(root)
        create_tests_structure(root)
        create_setup_files(root)
        
        print("\n" + "=" * 60)
        print("✓ 架构结构创建完成！")
        print("=" * 60)
        
        print("\n下一步:")
        print("1. 安装开发依赖: pip install -e \".[dev]\"")
        print("2. 运行测试: pytest")
        print("3. 查看文档: ARCHITECTURE.md")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
