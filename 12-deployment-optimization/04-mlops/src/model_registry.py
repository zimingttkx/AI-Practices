"""
模型注册模块 (Model Registry)

提供模型版本管理和注册功能。

主要功能:
- 模型注册 (Model Registration)
- 版本管理 (Version Management)
- 阶段转换 (Stage Transition)
- 模型加载 (Model Loading)

支持的后端:
- 本地文件系统
- MLflow Model Registry
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import json
import time
import hashlib
import shutil
import threading
from datetime import datetime

# 检查可选依赖
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ModelStage(Enum):
    """模型阶段"""
    NONE = "none"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """模型版本信息"""
    name: str
    version: str
    stage: ModelStage = ModelStage.NONE
    model_path: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    run_id: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage.value,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "params": self.params,
            "tags": self.tags,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "run_id": self.run_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """从字典创建"""
        data = data.copy()
        data["stage"] = ModelStage(data.get("stage", "none"))
        return cls(**data)


@dataclass
class RegisteredModel:
    """注册模型"""
    name: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def latest_version(self) -> Optional[ModelVersion]:
        """获取最新版本"""
        if not self.versions:
            return None
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.created_at,
            reverse=True
        )
        return sorted_versions[0]

    @property
    def production_version(self) -> Optional[ModelVersion]:
        """获取生产版本"""
        for version in self.versions.values():
            if version.stage == ModelStage.PRODUCTION:
                return version
        return None

    @property
    def staging_version(self) -> Optional[ModelVersion]:
        """获取预发布版本"""
        for version in self.versions.values():
            if version.stage == ModelStage.STAGING:
                return version
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegisteredModel":
        """从字典创建"""
        data = data.copy()
        versions = {}
        for k, v in data.get("versions", {}).items():
            versions[k] = ModelVersion.from_dict(v)
        data["versions"] = versions
        return cls(**data)


class BaseRegistry(ABC):
    """注册中心基类"""

    @abstractmethod
    def register_model(
        self,
        name: str,
        model_path: str,
        version: Optional[str] = None,
        **kwargs
    ) -> ModelVersion:
        """注册模型"""
        pass

    @abstractmethod
    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersion]:
        """获取模型"""
        pass

    @abstractmethod
    def transition_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage
    ) -> None:
        """转换阶段"""
        pass

    @abstractmethod
    def list_models(self) -> List[RegisteredModel]:
        """列出所有模型"""
        pass

    @abstractmethod
    def delete_model(self, name: str, version: Optional[str] = None) -> None:
        """删除模型"""
        pass


class ModelRegistry(BaseRegistry):
    """
    本地模型注册中心

    将模型保存到本地文件系统。

    Example:
        >>> registry = ModelRegistry("./model_registry")
        >>> version = registry.register_model("my_model", "model.pt", version="1.0.0")
        >>> model = registry.get_model("my_model", stage=ModelStage.PRODUCTION)
        >>> registry.transition_stage("my_model", "1.0.0", ModelStage.PRODUCTION)
    """

    def __init__(self, registry_path: str = "./model_registry"):
        """
        初始化注册中心

        Args:
            registry_path: 注册中心路径
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self._models: Dict[str, RegisteredModel] = {}
        self._lock = threading.Lock()

        # 加载已有模型
        self._load_registry()

    def _load_registry(self) -> None:
        """加载注册表"""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            with open(registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for name, model_data in data.items():
                    self._models[name] = RegisteredModel.from_dict(model_data)

    def _save_registry(self) -> None:
        """保存注册表"""
        registry_file = self.registry_path / "registry.json"
        data = {name: model.to_dict() for name, model in self._models.items()}
        with open(registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _generate_version(self, name: str) -> str:
        """生成版本号"""
        if name not in self._models or not self._models[name].versions:
            return "1"

        versions = list(self._models[name].versions.keys())
        try:
            max_version = max(int(v) for v in versions if v.isdigit())
            return str(max_version + 1)
        except ValueError:
            return str(len(versions) + 1)

    def register_model(
        self,
        name: str,
        model_path: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
        run_id: str = "",
    ) -> ModelVersion:
        """
        注册模型

        Args:
            name: 模型名称
            model_path: 模型文件路径
            version: 版本号(可选，自动生成)
            metrics: 模型指标
            params: 模型参数
            tags: 标签
            description: 描述
            run_id: 关联的运行ID

        Returns:
            模型版本信息
        """
        src_path = Path(model_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with self._lock:
            # 创建或获取注册模型
            if name not in self._models:
                self._models[name] = RegisteredModel(name=name)

            # 生成版本号
            if version is None:
                version = self._generate_version(name)

            # 创建版本目录
            version_dir = self.registry_path / name / version
            version_dir.mkdir(parents=True, exist_ok=True)

            # 复制模型文件
            if src_path.is_dir():
                dest_path = version_dir / "model"
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
            else:
                dest_path = version_dir / src_path.name
                shutil.copy2(src_path, dest_path)

            # 创建版本记录
            model_version = ModelVersion(
                name=name,
                version=version,
                model_path=str(dest_path),
                metrics=metrics or {},
                params=params or {},
                tags=tags or {},
                description=description,
                run_id=run_id,
                source=str(src_path.absolute()),
            )

            # 保存到注册表
            self._models[name].versions[version] = model_version
            self._models[name].updated_at = time.time()
            self._save_registry()

            return model_version

    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersion]:
        """
        获取模型

        Args:
            name: 模型名称
            version: 版本号
            stage: 阶段

        Returns:
            模型版本信息
        """
        if name not in self._models:
            return None

        model = self._models[name]

        if stage is not None:
            # 按阶段获取
            for v in model.versions.values():
                if v.stage == stage:
                    return v
            return None

        if version is not None:
            # 按版本获取
            return model.versions.get(version)

        # 返回最新版本
        return model.latest_version

    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        loader: Optional[Callable[[str], Any]] = None
    ) -> Any:
        """
        加载模型

        Args:
            name: 模型名称
            version: 版本号
            stage: 阶段
            loader: 自定义加载函数

        Returns:
            加载的模型对象
        """
        model_version = self.get_model(name, version, stage)
        if model_version is None:
            raise ValueError(f"Model not found: {name}")

        model_path = model_version.model_path

        if loader is not None:
            return loader(model_path)

        # 尝试自动检测并加载
        path = Path(model_path)

        if path.suffix == ".pt" or path.suffix == ".pth":
            try:
                import torch
                return torch.load(model_path)
            except ImportError:
                raise ImportError("PyTorch is required to load .pt/.pth files")

        elif path.suffix == ".pkl" or path.suffix == ".pickle":
            import pickle
            with open(model_path, "rb") as f:
                return pickle.load(f)

        elif path.suffix == ".joblib":
            try:
                import joblib
                return joblib.load(model_path)
            except ImportError:
                raise ImportError("joblib is required to load .joblib files")

        elif path.suffix == ".onnx":
            try:
                import onnx
                return onnx.load(model_path)
            except ImportError:
                raise ImportError("onnx is required to load .onnx files")

        else:
            raise ValueError(f"Unknown model format: {path.suffix}. Please provide a loader function.")

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage
    ) -> None:
        """
        转换模型阶段

        Args:
            name: 模型名称
            version: 版本号
            stage: 目标阶段
        """
        with self._lock:
            if name not in self._models:
                raise ValueError(f"Model not found: {name}")

            if version not in self._models[name].versions:
                raise ValueError(f"Version not found: {version}")

            # 如果转换到 Production 或 Staging，先将其他版本降级
            if stage in (ModelStage.PRODUCTION, ModelStage.STAGING):
                for v in self._models[name].versions.values():
                    if v.stage == stage:
                        v.stage = ModelStage.ARCHIVED
                        v.updated_at = time.time()

            # 更新目标版本
            self._models[name].versions[version].stage = stage
            self._models[name].versions[version].updated_at = time.time()
            self._models[name].updated_at = time.time()
            self._save_registry()

    def list_models(self) -> List[RegisteredModel]:
        """
        列出所有模型

        Returns:
            注册模型列表
        """
        return list(self._models.values())

    def list_versions(self, name: str) -> List[ModelVersion]:
        """
        列出模型的所有版本

        Args:
            name: 模型名称

        Returns:
            版本列表
        """
        if name not in self._models:
            return []
        return list(self._models[name].versions.values())

    def delete_model(self, name: str, version: Optional[str] = None) -> None:
        """
        删除模型

        Args:
            name: 模型名称
            version: 版本号(可选，不指定则删除整个模型)
        """
        with self._lock:
            if name not in self._models:
                raise ValueError(f"Model not found: {name}")

            if version is not None:
                # 删除特定版本
                if version not in self._models[name].versions:
                    raise ValueError(f"Version not found: {version}")

                # 删除版本目录
                version_dir = self.registry_path / name / version
                if version_dir.exists():
                    shutil.rmtree(version_dir)

                # 从注册表删除
                del self._models[name].versions[version]
                self._models[name].updated_at = time.time()

                # 如果没有版本了，删除整个模型
                if not self._models[name].versions:
                    del self._models[name]
                    model_dir = self.registry_path / name
                    if model_dir.exists():
                        shutil.rmtree(model_dir)
            else:
                # 删除整个模型
                model_dir = self.registry_path / name
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                del self._models[name]

            self._save_registry()

    def update_model_description(
        self,
        name: str,
        description: str,
        version: Optional[str] = None
    ) -> None:
        """
        更新模型描述

        Args:
            name: 模型名称
            description: 新描述
            version: 版本号(可选)
        """
        with self._lock:
            if name not in self._models:
                raise ValueError(f"Model not found: {name}")

            if version is not None:
                if version not in self._models[name].versions:
                    raise ValueError(f"Version not found: {version}")
                self._models[name].versions[version].description = description
                self._models[name].versions[version].updated_at = time.time()
            else:
                self._models[name].description = description

            self._models[name].updated_at = time.time()
            self._save_registry()

    def set_model_tags(
        self,
        name: str,
        tags: Dict[str, str],
        version: Optional[str] = None
    ) -> None:
        """
        设置模型标签

        Args:
            name: 模型名称
            tags: 标签字典
            version: 版本号(可选)
        """
        with self._lock:
            if name not in self._models:
                raise ValueError(f"Model not found: {name}")

            if version is not None:
                if version not in self._models[name].versions:
                    raise ValueError(f"Version not found: {version}")
                self._models[name].versions[version].tags.update(tags)
                self._models[name].versions[version].updated_at = time.time()
            else:
                self._models[name].tags.update(tags)

            self._models[name].updated_at = time.time()
            self._save_registry()

    def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersion]:
        """
        搜索模型

        Args:
            query: 搜索关键词(匹配名称或描述)
            tags: 标签过滤
            stage: 阶段过滤

        Returns:
            匹配的模型版本列表
        """
        results = []

        for model in self._models.values():
            for version in model.versions.values():
                # 名称/描述匹配
                if query:
                    if query.lower() not in version.name.lower() and \
                       query.lower() not in version.description.lower():
                        continue

                # 标签匹配
                if tags:
                    match = all(
                        version.tags.get(k) == v
                        for k, v in tags.items()
                    )
                    if not match:
                        continue

                # 阶段匹配
                if stage and version.stage != stage:
                    continue

                results.append(version)

        return results

    def compare_versions(
        self,
        name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        比较两个版本

        Args:
            name: 模型名称
            version1: 版本1
            version2: 版本2

        Returns:
            比较结果
        """
        v1 = self.get_model(name, version1)
        v2 = self.get_model(name, version2)

        if v1 is None or v2 is None:
            raise ValueError("One or both versions not found")

        # 比较指标
        metrics_diff = {}
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for metric in all_metrics:
            val1 = v1.metrics.get(metric)
            val2 = v2.metrics.get(metric)
            if val1 is not None and val2 is not None:
                metrics_diff[metric] = {
                    "v1": val1,
                    "v2": val2,
                    "diff": val2 - val1,
                    "pct_change": ((val2 - val1) / val1 * 100) if val1 != 0 else None
                }
            else:
                metrics_diff[metric] = {"v1": val1, "v2": val2}

        # 比较参数
        params_diff = {}
        all_params = set(v1.params.keys()) | set(v2.params.keys())
        for param in all_params:
            val1 = v1.params.get(param)
            val2 = v2.params.get(param)
            if val1 != val2:
                params_diff[param] = {"v1": val1, "v2": val2}

        return {
            "version1": version1,
            "version2": version2,
            "metrics_diff": metrics_diff,
            "params_diff": params_diff,
            "stage_v1": v1.stage.value,
            "stage_v2": v2.stage.value,
        }


class MLflowRegistry(BaseRegistry):
    """
    MLflow 模型注册中心

    使用 MLflow Model Registry 作为后端。

    Example:
        >>> registry = MLflowRegistry(tracking_uri="http://localhost:5000")
        >>> version = registry.register_model("my_model", "runs:/run_id/model")
        >>> registry.transition_stage("my_model", "1", ModelStage.PRODUCTION)
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        初始化 MLflow 注册中心

        Args:
            tracking_uri: MLflow 服务器地址
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()

    def register_model(
        self,
        name: str,
        model_path: str,
        version: Optional[str] = None,
        **kwargs
    ) -> ModelVersion:
        """
        注册模型

        Args:
            name: 模型名称
            model_path: 模型URI (如 "runs:/run_id/model")
            version: 忽略(MLflow自动生成)
            **kwargs: 额外参数

        Returns:
            模型版本信息
        """
        # 注册模型
        result = mlflow.register_model(model_path, name)

        # 转换为本地格式
        return ModelVersion(
            name=result.name,
            version=result.version,
            stage=self._convert_stage(result.current_stage),
            model_path=model_path,
            description=kwargs.get("description", ""),
            run_id=result.run_id or "",
            source=result.source or "",
        )

    def _convert_stage(self, mlflow_stage: str) -> ModelStage:
        """转换 MLflow 阶段到本地阶段"""
        stage_map = {
            "None": ModelStage.NONE,
            "Staging": ModelStage.STAGING,
            "Production": ModelStage.PRODUCTION,
            "Archived": ModelStage.ARCHIVED,
        }
        return stage_map.get(mlflow_stage, ModelStage.NONE)

    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersion]:
        """获取模型"""
        try:
            if stage is not None:
                stage_map = {
                    ModelStage.STAGING: "Staging",
                    ModelStage.PRODUCTION: "Production",
                    ModelStage.ARCHIVED: "Archived",
                }
                mlflow_stage = stage_map.get(stage)
                if mlflow_stage:
                    versions = self.client.get_latest_versions(name, stages=[mlflow_stage])
                    if versions:
                        v = versions[0]
                        return ModelVersion(
                            name=v.name,
                            version=v.version,
                            stage=self._convert_stage(v.current_stage),
                            model_path=v.source,
                            run_id=v.run_id or "",
                        )
                return None

            if version is not None:
                v = self.client.get_model_version(name, version)
                return ModelVersion(
                    name=v.name,
                    version=v.version,
                    stage=self._convert_stage(v.current_stage),
                    model_path=v.source,
                    run_id=v.run_id or "",
                )

            # 获取最新版本
            versions = self.client.get_latest_versions(name)
            if versions:
                v = versions[0]
                return ModelVersion(
                    name=v.name,
                    version=v.version,
                    stage=self._convert_stage(v.current_stage),
                    model_path=v.source,
                    run_id=v.run_id or "",
                )
            return None

        except Exception:
            return None

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage
    ) -> None:
        """转换模型阶段"""
        stage_map = {
            ModelStage.NONE: "None",
            ModelStage.STAGING: "Staging",
            ModelStage.PRODUCTION: "Production",
            ModelStage.ARCHIVED: "Archived",
        }
        mlflow_stage = stage_map.get(stage, "None")
        self.client.transition_model_version_stage(name, version, mlflow_stage)

    def list_models(self) -> List[RegisteredModel]:
        """列出所有模型"""
        models = []
        for rm in self.client.search_registered_models():
            models.append(RegisteredModel(
                name=rm.name,
                description=rm.description or "",
                tags=rm.tags or {},
            ))
        return models

    def delete_model(self, name: str, version: Optional[str] = None) -> None:
        """删除模型"""
        if version is not None:
            self.client.delete_model_version(name, version)
        else:
            self.client.delete_registered_model(name)

    def load_model(self, name: str, version: Optional[str] = None, stage: Optional[ModelStage] = None):
        """加载模型"""
        if stage is not None:
            stage_map = {
                ModelStage.STAGING: "Staging",
                ModelStage.PRODUCTION: "Production",
            }
            mlflow_stage = stage_map.get(stage, "Production")
            model_uri = f"models:/{name}/{mlflow_stage}"
        elif version is not None:
            model_uri = f"models:/{name}/{version}"
        else:
            model_uri = f"models:/{name}/latest"

        return mlflow.pyfunc.load_model(model_uri)


def create_registry(
    backend: str = "local",
    **kwargs
) -> BaseRegistry:
    """
    创建模型注册中心

    Args:
        backend: 后端类型 ("local", "mlflow")
        **kwargs: 额外参数

    Returns:
        注册中心实例

    Example:
        >>> # 本地注册中心
        >>> registry = create_registry("local", registry_path="./models")

        >>> # MLflow 注册中心
        >>> registry = create_registry("mlflow", tracking_uri="http://localhost:5000")
    """
    if backend == "local":
        return ModelRegistry(**kwargs)
    elif backend == "mlflow":
        return MLflowRegistry(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: local, mlflow")
