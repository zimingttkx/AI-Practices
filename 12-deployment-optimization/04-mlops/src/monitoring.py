"""
生产监控模块 (Production Monitoring)

提供模型生产环境监控功能。

主要功能:
- 指标收集 (Metrics Collection)
- 数据漂移检测 (Data Drift Detection)
- 模型性能监控 (Model Performance Monitoring)
- 告警管理 (Alert Management)

支持的后端:
- 本地存储
- Prometheus (可选)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from collections import deque
import json
import time
import threading
import statistics
from datetime import datetime

# 检查可选依赖
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"      # 计数器(只增不减)
    GAUGE = "gauge"          # 仪表盘(可增可减)
    HISTOGRAM = "histogram"  # 直方图(分布)
    SUMMARY = "summary"      # 摘要(百分位)


class AlertSeverity(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警信息"""
    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
        }


@dataclass
class MetricValue:
    """指标值"""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class DriftDetector:
    """
    数据漂移检测器

    支持多种漂移检测方法:
    - KS检验 (Kolmogorov-Smirnov)
    - PSI (Population Stability Index)
    - 均值/方差检测

    Example:
        >>> detector = DriftDetector(reference_data)
        >>> result = detector.detect_drift(current_data)
        >>> if result["drift_detected"]:
        ...     print("检测到数据漂移!")
    """

    def __init__(
        self,
        reference_data: Optional[List[float]] = None,
        window_size: int = 1000,
    ):
        """
        初始化漂移检测器

        Args:
            reference_data: 参考数据
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self._reference_data: List[float] = []
        self._current_window: deque = deque(maxlen=window_size)

        if reference_data:
            self.set_reference(reference_data)

    def set_reference(self, data: List[float]) -> None:
        """
        设置参考数据

        Args:
            data: 参考数据列表
        """
        self._reference_data = list(data)
        self._reference_mean = statistics.mean(data)
        self._reference_std = statistics.stdev(data) if len(data) > 1 else 0

    def add_sample(self, value: float) -> None:
        """
        添加样本

        Args:
            value: 样本值
        """
        self._current_window.append(value)

    def detect_drift(
        self,
        current_data: Optional[List[float]] = None,
        method: str = "ks",
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        检测数据漂移

        Args:
            current_data: 当前数据(可选，使用滑动窗口)
            method: 检测方法 ("ks", "psi", "mean", "std")
            threshold: 阈值

        Returns:
            检测结果字典
        """
        if current_data is None:
            current_data = list(self._current_window)

        if not self._reference_data or not current_data:
            return {
                "drift_detected": False,
                "method": method,
                "error": "Insufficient data",
            }

        if method == "ks":
            return self._ks_test(current_data, threshold)
        elif method == "psi":
            return self._psi_test(current_data, threshold)
        elif method == "mean":
            return self._mean_test(current_data, threshold)
        elif method == "std":
            return self._std_test(current_data, threshold)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _ks_test(self, current_data: List[float], threshold: float) -> Dict[str, Any]:
        """KS检验"""
        if SCIPY_AVAILABLE:
            statistic, p_value = stats.ks_2samp(self._reference_data, current_data)
            drift_detected = p_value < threshold
        else:
            # 简化版本：比较分布
            ref_sorted = sorted(self._reference_data)
            cur_sorted = sorted(current_data)
            n1, n2 = len(ref_sorted), len(cur_sorted)

            # 计算最大差异
            max_diff = 0
            i, j = 0, 0
            while i < n1 and j < n2:
                f1 = (i + 1) / n1
                f2 = (j + 1) / n2
                max_diff = max(max_diff, abs(f1 - f2))
                if ref_sorted[i] <= cur_sorted[j]:
                    i += 1
                else:
                    j += 1

            statistic = max_diff
            # 简化的p值估计
            p_value = 1.0 - statistic
            drift_detected = statistic > 0.1

        return {
            "drift_detected": drift_detected,
            "method": "ks",
            "statistic": statistic,
            "p_value": p_value,
            "threshold": threshold,
        }

    def _psi_test(self, current_data: List[float], threshold: float = 0.2) -> Dict[str, Any]:
        """PSI检验"""
        bins = 10

        # 计算分箱
        all_data = self._reference_data + current_data
        min_val, max_val = min(all_data), max(all_data)
        bin_width = (max_val - min_val) / bins if max_val > min_val else 1

        def get_bin_counts(data: List[float]) -> List[float]:
            counts = [0] * bins
            for v in data:
                idx = min(int((v - min_val) / bin_width), bins - 1)
                counts[idx] += 1
            # 转换为比例，避免除零
            total = len(data)
            return [(c / total) if total > 0 else 0.001 for c in counts]

        ref_pcts = get_bin_counts(self._reference_data)
        cur_pcts = get_bin_counts(current_data)

        # 计算PSI
        psi = 0
        for r, c in zip(ref_pcts, cur_pcts):
            r = max(r, 0.001)
            c = max(c, 0.001)
            psi += (c - r) * (c / r if r > 0 else 0)

        # PSI > 0.2 通常表示显著漂移
        drift_detected = psi > threshold

        return {
            "drift_detected": drift_detected,
            "method": "psi",
            "psi": psi,
            "threshold": threshold,
        }

    def _mean_test(self, current_data: List[float], threshold: float) -> Dict[str, Any]:
        """均值检测"""
        current_mean = statistics.mean(current_data)
        diff = abs(current_mean - self._reference_mean)
        relative_diff = diff / abs(self._reference_mean) if self._reference_mean != 0 else diff

        drift_detected = relative_diff > threshold

        return {
            "drift_detected": drift_detected,
            "method": "mean",
            "reference_mean": self._reference_mean,
            "current_mean": current_mean,
            "relative_diff": relative_diff,
            "threshold": threshold,
        }

    def _std_test(self, current_data: List[float], threshold: float) -> Dict[str, Any]:
        """方差检测"""
        current_std = statistics.stdev(current_data) if len(current_data) > 1 else 0
        diff = abs(current_std - self._reference_std)
        relative_diff = diff / self._reference_std if self._reference_std != 0 else diff

        drift_detected = relative_diff > threshold

        return {
            "drift_detected": drift_detected,
            "method": "std",
            "reference_std": self._reference_std,
            "current_std": current_std,
            "relative_diff": relative_diff,
            "threshold": threshold,
        }


class MetricsMonitor:
    """
    指标监控器

    收集和管理服务指标。

    Example:
        >>> monitor = MetricsMonitor()
        >>> monitor.record_latency(0.05)
        >>> monitor.record_prediction(pred, label)
        >>> stats = monitor.get_stats()
    """

    def __init__(self, window_size: int = 10000, save_path: Optional[str] = None):
        """
        初始化监控器

        Args:
            window_size: 滑动窗口大小
            save_path: 保存路径
        """
        self.window_size = window_size
        self.save_path = Path(save_path) if save_path else None

        # 指标存储
        self._latencies: deque = deque(maxlen=window_size)
        self._predictions: deque = deque(maxlen=window_size)
        self._labels: deque = deque(maxlen=window_size)
        self._errors: deque = deque(maxlen=window_size)

        # 计数器
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0

        # 告警
        self._alerts: List[Alert] = []
        self._alert_rules: Dict[str, Dict[str, Any]] = {}

        # 线程锁
        self._lock = threading.Lock()

        # 开始时间
        self._start_time = time.time()

    def record_latency(self, latency_ms: float) -> None:
        """
        记录延迟

        Args:
            latency_ms: 延迟(毫秒)
        """
        with self._lock:
            self._latencies.append(MetricValue(value=latency_ms))
            self._total_requests += 1
            self._successful_requests += 1

    def record_error(self, error: str) -> None:
        """
        记录错误

        Args:
            error: 错误信息
        """
        with self._lock:
            self._errors.append({
                "error": error,
                "timestamp": time.time(),
            })
            self._total_requests += 1
            self._failed_requests += 1

    def record_prediction(
        self,
        prediction: Any,
        label: Optional[Any] = None,
        latency_ms: Optional[float] = None
    ) -> None:
        """
        记录预测

        Args:
            prediction: 预测值
            label: 真实标签
            latency_ms: 延迟
        """
        with self._lock:
            self._predictions.append({
                "prediction": prediction,
                "timestamp": time.time(),
            })
            if label is not None:
                self._labels.append({
                    "label": label,
                    "timestamp": time.time(),
                })
            if latency_ms is not None:
                self._latencies.append(MetricValue(value=latency_ms))

            self._total_requests += 1
            self._successful_requests += 1

    def add_alert_rule(
        self,
        name: str,
        metric: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        message: Optional[str] = None,
    ) -> None:
        """
        添加告警规则

        Args:
            name: 规则名称
            metric: 指标名称 ("latency_p99", "error_rate", "throughput")
            condition: 条件 ("gt", "lt", "gte", "lte")
            threshold: 阈值
            severity: 告警级别
            message: 告警消息
        """
        self._alert_rules[name] = {
            "metric": metric,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "message": message or f"{metric} {condition} {threshold}",
        }

    def check_alerts(self) -> List[Alert]:
        """
        检查告警

        Returns:
            触发的告警列表
        """
        triggered = []
        stats = self.get_stats()

        for name, rule in self._alert_rules.items():
            metric_value = stats.get(rule["metric"])
            if metric_value is None:
                continue

            threshold = rule["threshold"]
            condition = rule["condition"]

            # 检查条件
            triggered_alert = False
            if condition == "gt" and metric_value > threshold:
                triggered_alert = True
            elif condition == "lt" and metric_value < threshold:
                triggered_alert = True
            elif condition == "gte" and metric_value >= threshold:
                triggered_alert = True
            elif condition == "lte" and metric_value <= threshold:
                triggered_alert = True

            if triggered_alert:
                alert = Alert(
                    name=name,
                    severity=rule["severity"],
                    message=rule["message"],
                    metric_name=rule["metric"],
                    metric_value=metric_value,
                    threshold=threshold,
                )
                triggered.append(alert)
                self._alerts.append(alert)

        return triggered

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计字典
        """
        with self._lock:
            latencies = [m.value for m in self._latencies]

            stats = {
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "error_rate": self._failed_requests / self._total_requests if self._total_requests > 0 else 0,
                "uptime_seconds": time.time() - self._start_time,
            }

            if latencies:
                sorted_latencies = sorted(latencies)
                n = len(sorted_latencies)
                stats.update({
                    "latency_avg": statistics.mean(latencies),
                    "latency_min": min(latencies),
                    "latency_max": max(latencies),
                    "latency_p50": sorted_latencies[int(n * 0.5)],
                    "latency_p90": sorted_latencies[int(n * 0.9)],
                    "latency_p99": sorted_latencies[min(int(n * 0.99), n - 1)],
                })

                # 计算吞吐量
                if len(self._latencies) >= 2:
                    time_span = self._latencies[-1].timestamp - self._latencies[0].timestamp
                    if time_span > 0:
                        stats["throughput"] = len(self._latencies) / time_span

            return stats

    def get_alerts(self, resolved: Optional[bool] = None) -> List[Alert]:
        """
        获取告警

        Args:
            resolved: 过滤已解决/未解决

        Returns:
            告警列表
        """
        if resolved is None:
            return self._alerts.copy()
        return [a for a in self._alerts if a.resolved == resolved]

    def resolve_alert(self, alert_name: str) -> None:
        """
        解决告警

        Args:
            alert_name: 告警名称
        """
        for alert in self._alerts:
            if alert.name == alert_name and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()

    def reset(self) -> None:
        """重置监控器"""
        with self._lock:
            self._latencies.clear()
            self._predictions.clear()
            self._labels.clear()
            self._errors.clear()
            self._total_requests = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._alerts.clear()
            self._start_time = time.time()


class ModelMonitor:
    """
    模型监控器

    综合监控模型性能，包括:
    - 推理延迟
    - 预测分布
    - 数据漂移
    - 模型性能

    Example:
        >>> monitor = ModelMonitor("my_model")
        >>> monitor.record_inference(input_data, prediction, latency_ms=50)
        >>> if monitor.check_drift():
        ...     print("检测到漂移!")
    """

    def __init__(
        self,
        model_name: str,
        model_version: str = "1.0",
        reference_data: Optional[List[float]] = None,
        window_size: int = 10000,
    ):
        """
        初始化模型监控器

        Args:
            model_name: 模型名称
            model_version: 模型版本
            reference_data: 参考数据(用于漂移检测)
            window_size: 滑动窗口大小
        """
        self.model_name = model_name
        self.model_version = model_version

        # 指标监控器
        self.metrics_monitor = MetricsMonitor(window_size=window_size)

        # 漂移检测器
        self.drift_detector = DriftDetector(
            reference_data=reference_data,
            window_size=window_size
        )

        # 预测分布
        self._prediction_values: deque = deque(maxlen=window_size)
        self._input_features: deque = deque(maxlen=window_size)

        # 性能指标
        self._accuracy_window: deque = deque(maxlen=1000)

        self._lock = threading.Lock()

    def record_inference(
        self,
        input_data: Any,
        prediction: Any,
        label: Optional[Any] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        记录推理

        Args:
            input_data: 输入数据
            prediction: 预测结果
            label: 真实标签
            latency_ms: 延迟
        """
        with self._lock:
            # 记录预测
            self.metrics_monitor.record_prediction(prediction, label, latency_ms)

            # 记录预测值(用于分布监控)
            if isinstance(prediction, (int, float)):
                self._prediction_values.append(prediction)
                self.drift_detector.add_sample(prediction)

            # 记录输入特征(用于输入漂移检测)
            if isinstance(input_data, (list, tuple)) and input_data:
                if isinstance(input_data[0], (int, float)):
                    self._input_features.append(input_data)

            # 记录准确率
            if label is not None:
                correct = prediction == label
                self._accuracy_window.append(1 if correct else 0)

    def record_error(self, error: str) -> None:
        """记录错误"""
        self.metrics_monitor.record_error(error)

    def check_drift(self, method: str = "psi", threshold: float = 0.2) -> Dict[str, Any]:
        """
        检查数据漂移

        Args:
            method: 检测方法
            threshold: 阈值

        Returns:
            漂移检测结果
        """
        return self.drift_detector.detect_drift(method=method, threshold=threshold)

    def get_accuracy(self) -> Optional[float]:
        """
        获取当前准确率

        Returns:
            准确率(0-1)
        """
        if not self._accuracy_window:
            return None
        return sum(self._accuracy_window) / len(self._accuracy_window)

    def get_prediction_distribution(self) -> Dict[str, Any]:
        """
        获取预测分布统计

        Returns:
            分布统计
        """
        if not self._prediction_values:
            return {}

        values = list(self._prediction_values)
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        获取综合统计

        Returns:
            统计字典
        """
        stats = self.metrics_monitor.get_stats()
        stats.update({
            "model_name": self.model_name,
            "model_version": self.model_version,
            "accuracy": self.get_accuracy(),
            "prediction_distribution": self.get_prediction_distribution(),
        })
        return stats

    def add_alert_rule(self, *args, **kwargs) -> None:
        """添加告警规则"""
        self.metrics_monitor.add_alert_rule(*args, **kwargs)

    def check_alerts(self) -> List[Alert]:
        """检查告警"""
        return self.metrics_monitor.check_alerts()

    def export_metrics(self, format: str = "json") -> Union[str, Dict]:
        """
        导出指标

        Args:
            format: 导出格式 ("json", "dict")

        Returns:
            导出的指标
        """
        stats = self.get_stats()
        if format == "json":
            return json.dumps(stats, indent=2, default=str)
        return stats


class PrometheusExporter:
    """
    Prometheus 指标导出器

    将指标导出到 Prometheus 格式。

    Example:
        >>> exporter = PrometheusExporter(port=8000)
        >>> exporter.record_latency("my_model", 0.05)
        >>> exporter.start()
    """

    def __init__(self, port: int = 8000, prefix: str = "ml_model"):
        """
        初始化导出器

        Args:
            port: HTTP 端口
            prefix: 指标前缀
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client is not installed. "
                "Install with: pip install prometheus-client"
            )

        self.port = port
        self.prefix = prefix
        self._started = False

        # 创建指标
        self._prediction_counter = Counter(
            f"{prefix}_predictions_total",
            "Total number of predictions",
            ["model_name", "model_version"]
        )

        self._prediction_latency = Histogram(
            f"{prefix}_prediction_latency_seconds",
            "Prediction latency in seconds",
            ["model_name"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )

        self._error_counter = Counter(
            f"{prefix}_errors_total",
            "Total number of errors",
            ["model_name", "error_type"]
        )

        self._model_accuracy = Gauge(
            f"{prefix}_accuracy",
            "Current model accuracy",
            ["model_name", "model_version"]
        )

        self._drift_score = Gauge(
            f"{prefix}_drift_score",
            "Data drift score",
            ["model_name", "method"]
        )

    def start(self) -> None:
        """启动 HTTP 服务器"""
        if not self._started:
            start_http_server(self.port)
            self._started = True

    def record_prediction(
        self,
        model_name: str,
        model_version: str = "1.0",
        latency_seconds: Optional[float] = None
    ) -> None:
        """记录预测"""
        self._prediction_counter.labels(
            model_name=model_name,
            model_version=model_version
        ).inc()

        if latency_seconds is not None:
            self._prediction_latency.labels(
                model_name=model_name
            ).observe(latency_seconds)

    def record_error(
        self,
        model_name: str,
        error_type: str = "unknown"
    ) -> None:
        """记录错误"""
        self._error_counter.labels(
            model_name=model_name,
            error_type=error_type
        ).inc()

    def set_accuracy(
        self,
        model_name: str,
        accuracy: float,
        model_version: str = "1.0"
    ) -> None:
        """设置准确率"""
        self._model_accuracy.labels(
            model_name=model_name,
            model_version=model_version
        ).set(accuracy)

    def set_drift_score(
        self,
        model_name: str,
        score: float,
        method: str = "psi"
    ) -> None:
        """设置漂移分数"""
        self._drift_score.labels(
            model_name=model_name,
            method=method
        ).set(score)


def create_monitor(
    model_name: str,
    model_version: str = "1.0",
    reference_data: Optional[List[float]] = None,
    enable_prometheus: bool = False,
    prometheus_port: int = 8000,
    **kwargs
) -> ModelMonitor:
    """
    创建模型监控器

    Args:
        model_name: 模型名称
        model_version: 模型版本
        reference_data: 参考数据
        enable_prometheus: 是否启用 Prometheus
        prometheus_port: Prometheus 端口
        **kwargs: 额外参数

    Returns:
        模型监控器实例

    Example:
        >>> monitor = create_monitor("my_model", reference_data=train_predictions)
        >>> monitor.record_inference(input_data, prediction, latency_ms=50)
    """
    monitor = ModelMonitor(
        model_name=model_name,
        model_version=model_version,
        reference_data=reference_data,
        **kwargs
    )

    if enable_prometheus:
        try:
            exporter = PrometheusExporter(port=prometheus_port)
            exporter.start()
            monitor._prometheus_exporter = exporter
        except ImportError:
            pass  # Prometheus not available

    return monitor
