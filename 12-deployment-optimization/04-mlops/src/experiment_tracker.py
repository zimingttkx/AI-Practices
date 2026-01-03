"""
实验追踪模块 (Experiment Tracking)

提供实验追踪功能，记录训练过程中的参数、指标和模型文件。

主要功能:
- 参数记录 (Parameters Logging)
- 指标追踪 (Metrics Tracking)
- 模型保存 (Artifact Logging)
- 实验对比 (Experiment Comparison)

支持的后端:
- 本地文件系统
- MLflow
- Weights & Biases (可选)
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentStatus(Enum):
    """实验状态"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class Experiment:
    """实验记录数据类"""
    name: str
    experiment_id: str = ""
    run_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: ExperimentStatus = ExperimentStatus.RUNNING
    description: str = ""

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = self._generate_id()
        if not self.run_id:
            self.run_id = self._generate_id()

    def _generate_id(self) -> str:
        """生成唯一ID"""
        content = f"{self.name}_{time.time()}_{id(self)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @property
    def duration(self) -> float:
        """实验持续时间(秒)"""
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "params": self.params,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status.value,
            "description": self.description,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """从字典创建"""
        data = data.copy()
        data["status"] = ExperimentStatus(data.get("status", "running"))
        data.pop("duration", None)
        return cls(**data)


class BaseTracker(ABC):
    """追踪器基类"""

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """记录参数"""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录指标"""
        pass

    @abstractmethod
    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """记录文件"""
        pass

    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """设置标签"""
        pass

    @abstractmethod
    def end_run(self, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> None:
        """结束运行"""
        pass


class ExperimentTracker(BaseTracker):
    """
    本地实验追踪器

    将实验数据保存到本地文件系统。

    Example:
        >>> tracker = ExperimentTracker("my_experiment")
        >>> tracker.log_params({"lr": 0.001, "batch_size": 32})
        >>> for epoch in range(10):
        ...     tracker.log_metrics({"loss": 0.5, "acc": 0.9}, step=epoch)
        >>> tracker.log_artifact("model.pt")
        >>> tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str,
        save_dir: str = "./experiments",
        run_name: Optional[str] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        初始化追踪器

        Args:
            experiment_name: 实验名称
            save_dir: 保存目录
            run_name: 运行名称
            description: 实验描述
            tags: 标签
        """
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建实验记录
        self.experiment = Experiment(
            name=experiment_name,
            description=description,
            tags=tags or {},
        )

        # 创建目录结构
        self.run_dir = self.save_dir / experiment_name / self.run_name
        self.artifacts_dir = self.run_dir / "artifacts"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

        # 步数计数器
        self._step = 0
        self._lock = threading.Lock()

        # 保存初始状态
        self._save_experiment()

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        记录参数

        Args:
            params: 参数字典
        """
        with self._lock:
            self.experiment.params.update(params)
            self._save_experiment()

    def log_param(self, key: str, value: Any) -> None:
        """
        记录单个参数

        Args:
            key: 参数名
            value: 参数值
        """
        self.log_params({key: value})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        记录指标

        Args:
            metrics: 指标字典
            step: 步数
        """
        with self._lock:
            if step is None:
                step = self._step
                self._step += 1

            for key, value in metrics.items():
                if key not in self.experiment.metrics:
                    self.experiment.metrics[key] = []
                self.experiment.metrics[key].append({
                    "value": value,
                    "step": step,
                    "timestamp": time.time(),
                })

            self._save_experiment()

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        记录单个指标

        Args:
            key: 指标名
            value: 指标值
            step: 步数
        """
        self.log_metrics({key: value}, step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """
        记录文件

        Args:
            path: 源文件路径
            artifact_path: 目标路径(相对于artifacts目录)
        """
        src_path = Path(path)
        if not src_path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        if artifact_path:
            dest_path = self.artifacts_dir / artifact_path
        else:
            dest_path = self.artifacts_dir / src_path.name

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if src_path.is_dir():
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dest_path)

        with self._lock:
            self.experiment.artifacts.append(str(dest_path.relative_to(self.run_dir)))
            self._save_experiment()

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        设置标签

        Args:
            tags: 标签字典
        """
        with self._lock:
            self.experiment.tags.update(tags)
            self._save_experiment()

    def set_tag(self, key: str, value: str) -> None:
        """
        设置单个标签

        Args:
            key: 标签名
            value: 标签值
        """
        self.set_tags({key: value})

    def end_run(self, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> None:
        """
        结束运行

        Args:
            status: 结束状态
        """
        with self._lock:
            self.experiment.end_time = time.time()
            self.experiment.status = status
            self._save_experiment()

    def _save_experiment(self) -> None:
        """保存实验数据"""
        experiment_file = self.run_dir / "experiment.json"
        with open(experiment_file, "w", encoding="utf-8") as f:
            json.dump(self.experiment.to_dict(), f, indent=2, ensure_ascii=False)

    def get_experiment(self) -> Experiment:
        """获取实验记录"""
        return self.experiment

    def get_best_metric(self, metric_name: str, mode: str = "min") -> Optional[Dict[str, Any]]:
        """
        获取最佳指标

        Args:
            metric_name: 指标名
            mode: "min" 或 "max"

        Returns:
            最佳指标记录
        """
        if metric_name not in self.experiment.metrics:
            return None

        metrics = self.experiment.metrics[metric_name]
        if not metrics:
            return None

        if mode == "min":
            return min(metrics, key=lambda x: x["value"])
        else:
            return max(metrics, key=lambda x: x["value"])

    @classmethod
    def load_experiment(cls, run_dir: str) -> Experiment:
        """
        加载实验记录

        Args:
            run_dir: 运行目录

        Returns:
            实验记录
        """
        experiment_file = Path(run_dir) / "experiment.json"
        if not experiment_file.exists():
            raise FileNotFoundError(f"Experiment file not found: {experiment_file}")

        with open(experiment_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Experiment.from_dict(data)

    @classmethod
    def list_experiments(cls, save_dir: str = "./experiments") -> List[Dict[str, Any]]:
        """
        列出所有实验

        Args:
            save_dir: 保存目录

        Returns:
            实验列表
        """
        save_path = Path(save_dir)
        if not save_path.exists():
            return []

        experiments = []
        for exp_dir in save_path.iterdir():
            if exp_dir.is_dir():
                for run_dir in exp_dir.iterdir():
                    if run_dir.is_dir():
                        try:
                            exp = cls.load_experiment(str(run_dir))
                            experiments.append({
                                "name": exp.name,
                                "run_name": run_dir.name,
                                "status": exp.status.value,
                                "start_time": exp.start_time,
                                "duration": exp.duration,
                                "path": str(run_dir),
                            })
                        except Exception:
                            pass

        return sorted(experiments, key=lambda x: x["start_time"], reverse=True)


class MLflowTracker(BaseTracker):
    """
    MLflow 实验追踪器

    使用 MLflow 作为后端进行实验追踪。

    Example:
        >>> tracker = MLflowTracker("my_experiment", tracking_uri="http://localhost:5000")
        >>> tracker.log_params({"lr": 0.001})
        >>> tracker.log_metrics({"loss": 0.5}, step=0)
        >>> tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        初始化 MLflow 追踪器

        Args:
            experiment_name: 实验名称
            tracking_uri: MLflow 服务器地址
            run_name: 运行名称
            tags: 标签
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")

        self.experiment_name = experiment_name
        self.run_name = run_name

        # 设置 tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # 设置实验
        mlflow.set_experiment(experiment_name)

        # 开始运行
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id

        # 设置标签
        if tags:
            self.set_tags(tags)

    def log_params(self, params: Dict[str, Any]) -> None:
        """记录参数"""
        mlflow.log_params(params)

    def log_param(self, key: str, value: Any) -> None:
        """记录单个参数"""
        mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录指标"""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """记录单个指标"""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """记录文件"""
        mlflow.log_artifact(path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """记录目录"""
        mlflow.log_artifacts(local_dir, artifact_path)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """设置标签"""
        mlflow.set_tags(tags)

    def set_tag(self, key: str, value: str) -> None:
        """设置单个标签"""
        mlflow.set_tag(key, value)

    def end_run(self, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> None:
        """结束运行"""
        status_map = {
            ExperimentStatus.COMPLETED: "FINISHED",
            ExperimentStatus.FAILED: "FAILED",
            ExperimentStatus.KILLED: "KILLED",
            ExperimentStatus.RUNNING: "RUNNING",
        }
        mlflow.end_run(status=status_map.get(status, "FINISHED"))

    def log_model(self, model: Any, artifact_path: str, **kwargs) -> None:
        """
        记录模型

        Args:
            model: 模型对象
            artifact_path: 保存路径
            **kwargs: 额外参数
        """
        # 尝试检测模型类型并使用对应的 log_model
        model_type = type(model).__module__.split(".")[0]

        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        elif model_type == "torch":
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
        elif model_type == "tensorflow" or model_type == "keras":
            mlflow.tensorflow.log_model(model, artifact_path, **kwargs)
        else:
            # 使用通用方法
            mlflow.pyfunc.log_model(artifact_path, python_model=model, **kwargs)

    @property
    def artifact_uri(self) -> str:
        """获取 artifact URI"""
        return mlflow.get_artifact_uri()


class WandbTracker(BaseTracker):
    """
    Weights & Biases 实验追踪器

    使用 W&B 作为后端进行实验追踪。

    Example:
        >>> tracker = WandbTracker("my_project", run_name="experiment_1")
        >>> tracker.log_params({"lr": 0.001})
        >>> tracker.log_metrics({"loss": 0.5})
        >>> tracker.end_run()
    """

    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ):
        """
        初始化 W&B 追踪器

        Args:
            project: 项目名称
            run_name: 运行名称
            config: 配置参数
            tags: 标签列表
            notes: 备注
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")

        self.project = project
        self.run_name = run_name

        # 初始化 wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=tags,
            notes=notes,
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        """记录参数"""
        wandb.config.update(params)

    def log_param(self, key: str, value: Any) -> None:
        """记录单个参数"""
        wandb.config.update({key: value})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录指标"""
        if step is not None:
            metrics["step"] = step
        wandb.log(metrics)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """记录单个指标"""
        self.log_metrics({key: value}, step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """记录文件"""
        artifact = wandb.Artifact(
            name=artifact_path or Path(path).stem,
            type="file"
        )
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """设置标签"""
        # W&B 使用列表形式的标签
        tag_list = [f"{k}:{v}" for k, v in tags.items()]
        self.run.tags = self.run.tags + tuple(tag_list)

    def set_tag(self, key: str, value: str) -> None:
        """设置单个标签"""
        self.set_tags({key: value})

    def end_run(self, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> None:
        """结束运行"""
        exit_code = 0 if status == ExperimentStatus.COMPLETED else 1
        wandb.finish(exit_code=exit_code)

    def log_image(self, key: str, image: Any) -> None:
        """记录图像"""
        wandb.log({key: wandb.Image(image)})

    def log_table(self, key: str, data: List[List[Any]], columns: List[str]) -> None:
        """记录表格"""
        table = wandb.Table(data=data, columns=columns)
        wandb.log({key: table})


def create_tracker(
    experiment_name: str,
    backend: str = "local",
    **kwargs
) -> BaseTracker:
    """
    创建实验追踪器

    Args:
        experiment_name: 实验名称
        backend: 后端类型 ("local", "mlflow", "wandb")
        **kwargs: 额外参数

    Returns:
        追踪器实例

    Example:
        >>> # 本地追踪器
        >>> tracker = create_tracker("my_exp", backend="local")

        >>> # MLflow 追踪器
        >>> tracker = create_tracker("my_exp", backend="mlflow",
        ...                          tracking_uri="http://localhost:5000")

        >>> # W&B 追踪器
        >>> tracker = create_tracker("my_project", backend="wandb")
    """
    if backend == "local":
        return ExperimentTracker(experiment_name, **kwargs)
    elif backend == "mlflow":
        return MLflowTracker(experiment_name, **kwargs)
    elif backend == "wandb":
        return WandbTracker(experiment_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: local, mlflow, wandb")


class ExperimentComparator:
    """
    实验比较器

    用于比较多个实验的结果。

    Example:
        >>> comparator = ExperimentComparator()
        >>> comparator.add_experiment("exp1", {"lr": 0.001}, {"acc": 0.95})
        >>> comparator.add_experiment("exp2", {"lr": 0.01}, {"acc": 0.92})
        >>> comparator.compare()
    """

    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def add_experiment(
        self,
        name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> None:
        """
        添加实验

        Args:
            name: 实验名称
            params: 参数
            metrics: 指标
        """
        self.experiments[name] = {
            "params": params,
            "metrics": metrics,
        }

    def add_from_tracker(self, tracker: ExperimentTracker) -> None:
        """
        从追踪器添加实验

        Args:
            tracker: 实验追踪器
        """
        exp = tracker.get_experiment()
        # 获取最后一个指标值
        metrics = {}
        for key, values in exp.metrics.items():
            if values:
                metrics[key] = values[-1]["value"] if isinstance(values[-1], dict) else values[-1]

        self.add_experiment(exp.name, exp.params, metrics)

    def compare(self, metric_name: Optional[str] = None, ascending: bool = True) -> List[Dict[str, Any]]:
        """
        比较实验

        Args:
            metric_name: 排序指标名
            ascending: 是否升序

        Returns:
            排序后的实验列表
        """
        results = []
        for name, data in self.experiments.items():
            results.append({
                "name": name,
                **data["params"],
                **data["metrics"],
            })

        if metric_name and results:
            results.sort(
                key=lambda x: x.get(metric_name, float("inf")),
                reverse=not ascending
            )

        return results

    def get_best(self, metric_name: str, mode: str = "min") -> Optional[Dict[str, Any]]:
        """
        获取最佳实验

        Args:
            metric_name: 指标名
            mode: "min" 或 "max"

        Returns:
            最佳实验
        """
        results = self.compare(metric_name, ascending=(mode == "min"))
        return results[0] if results else None

    def to_dataframe(self):
        """
        转换为 DataFrame

        Returns:
            pandas DataFrame
        """
        try:
            import pandas as pd
            results = self.compare()
            return pd.DataFrame(results)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
