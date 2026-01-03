"""
模型注册模块单元测试
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_registry import (
    ModelStage,
    ModelVersion,
    RegisteredModel,
    ModelRegistry,
    create_registry,
    MLFLOW_AVAILABLE,
)


class TestModelVersion:
    """测试 ModelVersion 数据类"""

    def test_model_version_creation(self):
        """测试创建模型版本"""
        version = ModelVersion(name="test_model", version="1.0")
        assert version.name == "test_model"
        assert version.version == "1.0"
        assert version.stage == ModelStage.NONE

    def test_model_version_to_dict(self):
        """测试转换为字典"""
        version = ModelVersion(
            name="test_model",
            version="1.0",
            stage=ModelStage.PRODUCTION,
            metrics={"accuracy": 0.95}
        )
        data = version.to_dict()
        assert data["name"] == "test_model"
        assert data["stage"] == "production"
        assert data["metrics"]["accuracy"] == 0.95

    def test_model_version_from_dict(self):
        """测试从字典创建"""
        data = {
            "name": "test_model",
            "version": "1.0",
            "stage": "staging",
            "model_path": "/path/to/model",
            "metrics": {"accuracy": 0.95},
            "params": {},
            "tags": {},
            "description": "",
            "created_at": time.time(),
            "updated_at": time.time(),
            "run_id": "",
            "source": "",
        }
        version = ModelVersion.from_dict(data)
        assert version.name == "test_model"
        assert version.stage == ModelStage.STAGING


class TestRegisteredModel:
    """测试 RegisteredModel 数据类"""

    def test_registered_model_creation(self):
        """测试创建注册模型"""
        model = RegisteredModel(name="test_model")
        assert model.name == "test_model"
        assert model.versions == {}

    def test_latest_version(self):
        """测试获取最新版本"""
        model = RegisteredModel(name="test_model")
        model.versions["1"] = ModelVersion(
            name="test_model",
            version="1",
            created_at=time.time() - 100
        )
        model.versions["2"] = ModelVersion(
            name="test_model",
            version="2",
            created_at=time.time()
        )

        latest = model.latest_version
        assert latest.version == "2"

    def test_production_version(self):
        """测试获取生产版本"""
        model = RegisteredModel(name="test_model")
        model.versions["1"] = ModelVersion(
            name="test_model",
            version="1",
            stage=ModelStage.ARCHIVED
        )
        model.versions["2"] = ModelVersion(
            name="test_model",
            version="2",
            stage=ModelStage.PRODUCTION
        )

        prod = model.production_version
        assert prod.version == "2"

    def test_staging_version(self):
        """测试获取预发布版本"""
        model = RegisteredModel(name="test_model")
        model.versions["1"] = ModelVersion(
            name="test_model",
            version="1",
            stage=ModelStage.STAGING
        )

        staging = model.staging_version
        assert staging.version == "1"


class TestModelRegistry:
    """测试 ModelRegistry"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def model_file(self, temp_dir):
        """创建测试模型文件"""
        model_path = Path(temp_dir) / "test_model.pkl"
        model_path.write_bytes(b"fake model content")
        return str(model_path)

    def test_registry_creation(self, temp_dir):
        """测试创建注册中心"""
        registry = ModelRegistry(registry_path=temp_dir)
        assert registry.registry_path.exists()

    def test_register_model(self, temp_dir, model_file):
        """测试注册模型"""
        registry = ModelRegistry(registry_path=temp_dir)
        version = registry.register_model(
            name="test_model",
            model_path=model_file,
            version="1.0",
            metrics={"accuracy": 0.95}
        )

        assert version.name == "test_model"
        assert version.version == "1.0"
        assert version.metrics["accuracy"] == 0.95

    def test_register_model_auto_version(self, temp_dir, model_file):
        """测试自动版本号"""
        registry = ModelRegistry(registry_path=temp_dir)

        v1 = registry.register_model("test_model", model_file)
        v2 = registry.register_model("test_model", model_file)

        assert v1.version == "1"
        assert v2.version == "2"

    def test_get_model_by_version(self, temp_dir, model_file):
        """测试按版本获取模型"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model("test_model", model_file, version="1.0")

        model = registry.get_model("test_model", version="1.0")
        assert model is not None
        assert model.version == "1.0"

    def test_get_model_by_stage(self, temp_dir, model_file):
        """测试按阶段获取模型"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model("test_model", model_file, version="1.0")
        registry.transition_stage("test_model", "1.0", ModelStage.PRODUCTION)

        model = registry.get_model("test_model", stage=ModelStage.PRODUCTION)
        assert model is not None
        assert model.version == "1.0"

    def test_get_model_latest(self, temp_dir, model_file):
        """测试获取最新版本"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model("test_model", model_file, version="1.0")
        registry.register_model("test_model", model_file, version="2.0")

        model = registry.get_model("test_model")
        assert model.version == "2.0"

    def test_transition_stage(self, temp_dir, model_file):
        """测试转换阶段"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model("test_model", model_file, version="1.0")

        registry.transition_stage("test_model", "1.0", ModelStage.STAGING)
        model = registry.get_model("test_model", version="1.0")
        assert model.stage == ModelStage.STAGING

        registry.transition_stage("test_model", "1.0", ModelStage.PRODUCTION)
        model = registry.get_model("test_model", version="1.0")
        assert model.stage == ModelStage.PRODUCTION

    def test_transition_stage_demotes_previous(self, temp_dir, model_file):
        """测试转换阶段时降级之前的版本"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model("test_model", model_file, version="1.0")
        registry.register_model("test_model", model_file, version="2.0")

        registry.transition_stage("test_model", "1.0", ModelStage.PRODUCTION)
        registry.transition_stage("test_model", "2.0", ModelStage.PRODUCTION)

        v1 = registry.get_model("test_model", version="1.0")
        v2 = registry.get_model("test_model", version="2.0")

        assert v1.stage == ModelStage.ARCHIVED
        assert v2.stage == ModelStage.PRODUCTION

    def test_list_models(self, temp_dir, model_file):
        """测试列出所有模型"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model("model_a", model_file)
        registry.register_model("model_b", model_file)

        models = registry.list_models()
        names = [m.name for m in models]
        assert "model_a" in names
        assert "model_b" in names

    def test_list_versions(self, temp_dir, model_file):
        """测试列出模型版本"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model("test_model", model_file, version="1.0")
        registry.register_model("test_model", model_file, version="2.0")

        versions = registry.list_versions("test_model")
        version_nums = [v.version for v in versions]
        assert "1.0" in version_nums
        assert "2.0" in version_nums

    def test_delete_model_version(self, temp_dir, model_file):
        """测试删除模型版本"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model("test_model", model_file, version="1.0")
        registry.register_model("test_model", model_file, version="2.0")

        registry.delete_model("test_model", version="1.0")

        versions = registry.list_versions("test_model")
        assert len(versions) == 1
        assert versions[0].version == "2.0"

    def test_delete_model(self, temp_dir, model_file):
        """测试删除整个模型"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model("test_model", model_file)

        registry.delete_model("test_model")

        model = registry.get_model("test_model")
        assert model is None

    def test_search_models(self, temp_dir, model_file):
        """测试搜索模型"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model(
            "image_classifier",
            model_file,
            tags={"type": "classification"}
        )
        registry.register_model(
            "text_encoder",
            model_file,
            tags={"type": "encoding"}
        )

        # 按名称搜索
        results = registry.search_models(query="image")
        assert len(results) == 1
        assert results[0].name == "image_classifier"

        # 按标签搜索
        results = registry.search_models(tags={"type": "classification"})
        assert len(results) == 1

    def test_compare_versions(self, temp_dir, model_file):
        """测试比较版本"""
        registry = ModelRegistry(registry_path=temp_dir)
        registry.register_model(
            "test_model",
            model_file,
            version="1.0",
            metrics={"accuracy": 0.90},
            params={"lr": 0.001}
        )
        registry.register_model(
            "test_model",
            model_file,
            version="2.0",
            metrics={"accuracy": 0.95},
            params={"lr": 0.0001}
        )

        comparison = registry.compare_versions("test_model", "1.0", "2.0")
        assert comparison["metrics_diff"]["accuracy"]["v1"] == 0.90
        assert comparison["metrics_diff"]["accuracy"]["v2"] == 0.95
        assert comparison["params_diff"]["lr"]["v1"] == 0.001
        assert comparison["params_diff"]["lr"]["v2"] == 0.0001

    def test_persistence(self, temp_dir, model_file):
        """测试持久化"""
        # 创建并注册模型
        registry1 = ModelRegistry(registry_path=temp_dir)
        registry1.register_model("test_model", model_file, version="1.0")

        # 重新加载
        registry2 = ModelRegistry(registry_path=temp_dir)
        model = registry2.get_model("test_model", version="1.0")

        assert model is not None
        assert model.version == "1.0"


class TestCreateRegistry:
    """测试 create_registry 工厂函数"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_create_local_registry(self, temp_dir):
        """测试创建本地注册中心"""
        registry = create_registry("local", registry_path=temp_dir)
        assert isinstance(registry, ModelRegistry)

    def test_create_invalid_backend(self):
        """测试无效后端"""
        with pytest.raises(ValueError):
            create_registry("invalid")

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_create_mlflow_registry(self):
        """测试创建 MLflow 注册中心"""
        from model_registry import MLflowRegistry
        registry = create_registry("mlflow")
        assert isinstance(registry, MLflowRegistry)
