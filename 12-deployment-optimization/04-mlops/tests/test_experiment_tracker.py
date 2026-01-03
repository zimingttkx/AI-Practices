"""
实验追踪模块单元测试
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiment_tracker import (
    Experiment,
    ExperimentStatus,
    ExperimentTracker,
    ExperimentComparator,
    create_tracker,
    MLFLOW_AVAILABLE,
    WANDB_AVAILABLE,
)


class TestExperiment:
    """测试 Experiment 数据类"""

    def test_experiment_creation(self):
        """测试创建实验"""
        exp = Experiment(name="test_exp")
        assert exp.name == "test_exp"
        assert exp.experiment_id != ""
        assert exp.run_id != ""
        assert exp.status == ExperimentStatus.RUNNING

    def test_experiment_duration(self):
        """测试实验持续时间"""
        exp = Experiment(name="test_exp")
        time.sleep(0.1)
        assert exp.duration >= 0.1

    def test_experiment_to_dict(self):
        """测试转换为字典"""
        exp = Experiment(
            name="test_exp",
            params={"lr": 0.001},
            description="Test experiment"
        )
        data = exp.to_dict()
        assert data["name"] == "test_exp"
        assert data["params"] == {"lr": 0.001}
        assert data["description"] == "Test experiment"

    def test_experiment_from_dict(self):
        """测试从字典创建"""
        data = {
            "name": "test_exp",
            "experiment_id": "abc123",
            "run_id": "def456",
            "params": {"lr": 0.001},
            "metrics": {},
            "artifacts": [],
            "tags": {},
            "start_time": time.time(),
            "end_time": None,
            "status": "running",
            "description": "Test",
        }
        exp = Experiment.from_dict(data)
        assert exp.name == "test_exp"
        assert exp.experiment_id == "abc123"
        assert exp.status == ExperimentStatus.RUNNING


class TestExperimentTracker:
    """测试 ExperimentTracker"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_tracker_creation(self, temp_dir):
        """测试创建追踪器"""
        tracker = ExperimentTracker(
            experiment_name="test_exp",
            save_dir=temp_dir
        )
        assert tracker.experiment_name == "test_exp"
        assert tracker.run_dir.exists()

    def test_log_params(self, temp_dir):
        """测试记录参数"""
        tracker = ExperimentTracker("test_exp", save_dir=temp_dir)
        tracker.log_params({"lr": 0.001, "batch_size": 32})

        exp = tracker.get_experiment()
        assert exp.params["lr"] == 0.001
        assert exp.params["batch_size"] == 32

    def test_log_param(self, temp_dir):
        """测试记录单个参数"""
        tracker = ExperimentTracker("test_exp", save_dir=temp_dir)
        tracker.log_param("lr", 0.001)

        exp = tracker.get_experiment()
        assert exp.params["lr"] == 0.001

    def test_log_metrics(self, temp_dir):
        """测试记录指标"""
        tracker = ExperimentTracker("test_exp", save_dir=temp_dir)
        tracker.log_metrics({"loss": 0.5, "acc": 0.9}, step=0)
        tracker.log_metrics({"loss": 0.3, "acc": 0.95}, step=1)

        exp = tracker.get_experiment()
        assert len(exp.metrics["loss"]) == 2
        assert exp.metrics["loss"][0]["value"] == 0.5
        assert exp.metrics["loss"][1]["value"] == 0.3

    def test_log_metric(self, temp_dir):
        """测试记录单个指标"""
        tracker = ExperimentTracker("test_exp", save_dir=temp_dir)
        tracker.log_metric("loss", 0.5, step=0)

        exp = tracker.get_experiment()
        assert exp.metrics["loss"][0]["value"] == 0.5

    def test_log_artifact(self, temp_dir):
        """测试记录文件"""
        tracker = ExperimentTracker("test_exp", save_dir=temp_dir)

        # 创建测试文件
        test_file = Path(temp_dir) / "test_artifact.txt"
        test_file.write_text("test content")

        tracker.log_artifact(str(test_file))

        exp = tracker.get_experiment()
        assert len(exp.artifacts) == 1

    def test_set_tags(self, temp_dir):
        """测试设置标签"""
        tracker = ExperimentTracker("test_exp", save_dir=temp_dir)
        tracker.set_tags({"env": "test", "version": "1.0"})

        exp = tracker.get_experiment()
        assert exp.tags["env"] == "test"
        assert exp.tags["version"] == "1.0"

    def test_end_run(self, temp_dir):
        """测试结束运行"""
        tracker = ExperimentTracker("test_exp", save_dir=temp_dir)
        tracker.end_run(ExperimentStatus.COMPLETED)

        exp = tracker.get_experiment()
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.end_time is not None

    def test_get_best_metric(self, temp_dir):
        """测试获取最佳指标"""
        tracker = ExperimentTracker("test_exp", save_dir=temp_dir)
        tracker.log_metric("loss", 0.5, step=0)
        tracker.log_metric("loss", 0.3, step=1)
        tracker.log_metric("loss", 0.4, step=2)

        best = tracker.get_best_metric("loss", mode="min")
        assert best["value"] == 0.3

        best = tracker.get_best_metric("loss", mode="max")
        assert best["value"] == 0.5

    def test_load_experiment(self, temp_dir):
        """测试加载实验"""
        tracker = ExperimentTracker("test_exp", save_dir=temp_dir)
        tracker.log_params({"lr": 0.001})
        tracker.end_run()

        # 加载实验
        loaded = ExperimentTracker.load_experiment(str(tracker.run_dir))
        assert loaded.name == "test_exp"
        assert loaded.params["lr"] == 0.001

    def test_list_experiments(self, temp_dir):
        """测试列出实验"""
        # 创建多个实验
        tracker1 = ExperimentTracker("exp1", save_dir=temp_dir, run_name="run1")
        tracker1.end_run()

        tracker2 = ExperimentTracker("exp2", save_dir=temp_dir, run_name="run2")
        tracker2.end_run()

        experiments = ExperimentTracker.list_experiments(temp_dir)
        assert len(experiments) == 2


class TestExperimentComparator:
    """测试 ExperimentComparator"""

    def test_add_experiment(self):
        """测试添加实验"""
        comparator = ExperimentComparator()
        comparator.add_experiment(
            "exp1",
            {"lr": 0.001},
            {"acc": 0.95}
        )

        assert "exp1" in comparator.experiments

    def test_compare(self):
        """测试比较实验"""
        comparator = ExperimentComparator()
        comparator.add_experiment("exp1", {"lr": 0.001}, {"acc": 0.95})
        comparator.add_experiment("exp2", {"lr": 0.01}, {"acc": 0.92})

        results = comparator.compare("acc", ascending=False)
        assert results[0]["name"] == "exp1"
        assert results[1]["name"] == "exp2"

    def test_get_best(self):
        """测试获取最佳实验"""
        comparator = ExperimentComparator()
        comparator.add_experiment("exp1", {"lr": 0.001}, {"loss": 0.5})
        comparator.add_experiment("exp2", {"lr": 0.01}, {"loss": 0.3})

        best = comparator.get_best("loss", mode="min")
        assert best["name"] == "exp2"


class TestCreateTracker:
    """测试 create_tracker 工厂函数"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_create_local_tracker(self, temp_dir):
        """测试创建本地追踪器"""
        tracker = create_tracker("test_exp", backend="local", save_dir=temp_dir)
        assert isinstance(tracker, ExperimentTracker)

    def test_create_invalid_backend(self, temp_dir):
        """测试无效后端"""
        with pytest.raises(ValueError):
            create_tracker("test_exp", backend="invalid")

    @pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="MLflow not installed")
    def test_create_mlflow_tracker(self, temp_dir):
        """测试创建 MLflow 追踪器"""
        from experiment_tracker import MLflowTracker
        tracker = create_tracker("test_exp", backend="mlflow")
        assert isinstance(tracker, MLflowTracker)
        tracker.end_run()

    @pytest.mark.skipif(not WANDB_AVAILABLE, reason="W&B not installed")
    def test_create_wandb_tracker(self):
        """测试创建 W&B 追踪器"""
        from experiment_tracker import WandbTracker
        tracker = create_tracker("test_project", backend="wandb")
        assert isinstance(tracker, WandbTracker)
        tracker.end_run()
