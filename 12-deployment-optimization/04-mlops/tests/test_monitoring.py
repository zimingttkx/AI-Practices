"""
监控模块单元测试
"""

import pytest
import time
import statistics
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monitoring import (
    MetricType,
    AlertSeverity,
    Alert,
    MetricValue,
    DriftDetector,
    MetricsMonitor,
    ModelMonitor,
    create_monitor,
    PROMETHEUS_AVAILABLE,
)


class TestAlert:
    """测试 Alert 数据类"""

    def test_alert_creation(self):
        """测试创建告警"""
        alert = Alert(
            name="high_latency",
            severity=AlertSeverity.WARNING,
            message="Latency too high",
            metric_name="latency_p99",
            metric_value=150.0,
            threshold=100.0
        )
        assert alert.name == "high_latency"
        assert alert.severity == AlertSeverity.WARNING
        assert not alert.resolved

    def test_alert_to_dict(self):
        """测试转换为字典"""
        alert = Alert(
            name="high_latency",
            severity=AlertSeverity.ERROR,
            message="Latency too high",
            metric_name="latency_p99",
            metric_value=150.0,
            threshold=100.0
        )
        data = alert.to_dict()
        assert data["name"] == "high_latency"
        assert data["severity"] == "error"
        assert data["metric_value"] == 150.0


class TestDriftDetector:
    """测试 DriftDetector"""

    def test_detector_creation(self):
        """测试创建检测器"""
        reference = [1.0, 2.0, 3.0, 4.0, 5.0]
        detector = DriftDetector(reference_data=reference)
        assert detector._reference_data == reference

    def test_set_reference(self):
        """测试设置参考数据"""
        detector = DriftDetector()
        reference = [1.0, 2.0, 3.0, 4.0, 5.0]
        detector.set_reference(reference)
        assert detector._reference_mean == 3.0

    def test_add_sample(self):
        """测试添加样本"""
        detector = DriftDetector(window_size=100)
        for i in range(50):
            detector.add_sample(float(i))
        assert len(detector._current_window) == 50

    def test_detect_drift_no_data(self):
        """测试无数据时的漂移检测"""
        detector = DriftDetector()
        result = detector.detect_drift()
        assert not result["drift_detected"]
        assert "error" in result

    def test_detect_drift_ks(self):
        """测试 KS 检验"""
        reference = [float(i) for i in range(100)]
        detector = DriftDetector(reference_data=reference)

        # 相似数据 - 无漂移
        similar_data = [float(i) for i in range(100)]
        result = detector.detect_drift(similar_data, method="ks")
        assert not result["drift_detected"]

        # 不同数据 - 有漂移
        different_data = [float(i + 100) for i in range(100)]
        result = detector.detect_drift(different_data, method="ks")
        assert result["drift_detected"]

    def test_detect_drift_psi(self):
        """测试 PSI 检验"""
        reference = [float(i) for i in range(100)]
        detector = DriftDetector(reference_data=reference)

        # 相似数据 - 无漂移
        similar_data = [float(i) for i in range(100)]
        result = detector.detect_drift(similar_data, method="psi", threshold=0.2)
        assert not result["drift_detected"]

    def test_detect_drift_mean(self):
        """测试均值检测"""
        reference = [10.0] * 100
        detector = DriftDetector(reference_data=reference)

        # 相似均值 - 无漂移
        similar_data = [10.5] * 100
        result = detector.detect_drift(similar_data, method="mean", threshold=0.1)
        assert not result["drift_detected"]

        # 不同均值 - 有漂移
        different_data = [20.0] * 100
        result = detector.detect_drift(different_data, method="mean", threshold=0.1)
        assert result["drift_detected"]

    def test_detect_drift_std(self):
        """测试方差检测"""
        reference = [float(i) for i in range(100)]
        detector = DriftDetector(reference_data=reference)

        # 相似方差 - 无漂移
        similar_data = [float(i) for i in range(100)]
        result = detector.detect_drift(similar_data, method="std", threshold=0.5)
        assert not result["drift_detected"]


class TestMetricsMonitor:
    """测试 MetricsMonitor"""

    def test_monitor_creation(self):
        """测试创建监控器"""
        monitor = MetricsMonitor(window_size=1000)
        assert monitor.window_size == 1000
        assert monitor._total_requests == 0

    def test_record_latency(self):
        """测试记录延迟"""
        monitor = MetricsMonitor()
        monitor.record_latency(50.0)
        monitor.record_latency(100.0)

        assert monitor._total_requests == 2
        assert monitor._successful_requests == 2

    def test_record_error(self):
        """测试记录错误"""
        monitor = MetricsMonitor()
        monitor.record_error("Connection timeout")

        assert monitor._total_requests == 1
        assert monitor._failed_requests == 1

    def test_record_prediction(self):
        """测试记录预测"""
        monitor = MetricsMonitor()
        monitor.record_prediction(prediction=1, label=1, latency_ms=50.0)

        assert monitor._total_requests == 1
        assert len(monitor._predictions) == 1

    def test_get_stats(self):
        """测试获取统计信息"""
        monitor = MetricsMonitor()

        # 记录一些数据
        for i in range(100):
            monitor.record_latency(float(i))

        stats = monitor.get_stats()
        assert stats["total_requests"] == 100
        assert stats["successful_requests"] == 100
        assert stats["failed_requests"] == 0
        assert "latency_avg" in stats
        assert "latency_p50" in stats
        assert "latency_p99" in stats

    def test_get_stats_with_errors(self):
        """测试带错误的统计"""
        monitor = MetricsMonitor()

        for i in range(90):
            monitor.record_latency(50.0)
        for i in range(10):
            monitor.record_error("Error")

        stats = monitor.get_stats()
        assert stats["total_requests"] == 100
        assert stats["error_rate"] == 0.1

    def test_add_alert_rule(self):
        """测试添加告警规则"""
        monitor = MetricsMonitor()
        monitor.add_alert_rule(
            name="high_latency",
            metric="latency_p99",
            condition="gt",
            threshold=100.0,
            severity=AlertSeverity.WARNING
        )

        assert "high_latency" in monitor._alert_rules

    def test_check_alerts(self):
        """测试检查告警"""
        monitor = MetricsMonitor()
        monitor.add_alert_rule(
            name="high_latency",
            metric="latency_p99",
            condition="gt",
            threshold=50.0,
            severity=AlertSeverity.WARNING
        )

        # 记录高延迟数据
        for i in range(100):
            monitor.record_latency(100.0)

        alerts = monitor.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].name == "high_latency"

    def test_get_alerts(self):
        """测试获取告警"""
        monitor = MetricsMonitor()
        monitor.add_alert_rule(
            name="high_error_rate",
            metric="error_rate",
            condition="gt",
            threshold=0.05,
            severity=AlertSeverity.ERROR
        )

        # 触发告警
        for i in range(90):
            monitor.record_latency(50.0)
        for i in range(10):
            monitor.record_error("Error")

        monitor.check_alerts()

        all_alerts = monitor.get_alerts()
        assert len(all_alerts) == 1

        unresolved = monitor.get_alerts(resolved=False)
        assert len(unresolved) == 1

    def test_resolve_alert(self):
        """测试解决告警"""
        monitor = MetricsMonitor()
        monitor.add_alert_rule(
            name="test_alert",
            metric="error_rate",
            condition="gt",
            threshold=0.0,
            severity=AlertSeverity.WARNING
        )

        monitor.record_error("Error")
        monitor.check_alerts()

        monitor.resolve_alert("test_alert")

        resolved = monitor.get_alerts(resolved=True)
        assert len(resolved) == 1
        assert resolved[0].resolved_at is not None

    def test_reset(self):
        """测试重置"""
        monitor = MetricsMonitor()
        monitor.record_latency(50.0)
        monitor.record_error("Error")

        monitor.reset()

        assert monitor._total_requests == 0
        assert len(monitor._latencies) == 0
        assert len(monitor._errors) == 0


class TestModelMonitor:
    """测试 ModelMonitor"""

    def test_monitor_creation(self):
        """测试创建模型监控器"""
        monitor = ModelMonitor(
            model_name="test_model",
            model_version="1.0"
        )
        assert monitor.model_name == "test_model"
        assert monitor.model_version == "1.0"

    def test_record_inference(self):
        """测试记录推理"""
        monitor = ModelMonitor("test_model")
        monitor.record_inference(
            input_data=[1.0, 2.0, 3.0],
            prediction=0.5,
            label=1,
            latency_ms=50.0
        )

        stats = monitor.get_stats()
        assert stats["total_requests"] == 1

    def test_record_error(self):
        """测试记录错误"""
        monitor = ModelMonitor("test_model")
        monitor.record_error("Model inference failed")

        stats = monitor.get_stats()
        assert stats["failed_requests"] == 1

    def test_check_drift(self):
        """测试检查漂移"""
        reference = [float(i) for i in range(100)]
        monitor = ModelMonitor(
            "test_model",
            reference_data=reference
        )

        # 添加相似数据
        for i in range(100):
            monitor.record_inference(
                input_data=[1.0],
                prediction=float(i)
            )

        result = monitor.check_drift(method="psi")
        assert "drift_detected" in result

    def test_get_accuracy(self):
        """测试获取准确率"""
        monitor = ModelMonitor("test_model")

        # 记录预测
        for i in range(80):
            monitor.record_inference([1.0], prediction=1, label=1)
        for i in range(20):
            monitor.record_inference([1.0], prediction=0, label=1)

        accuracy = monitor.get_accuracy()
        assert accuracy == 0.8

    def test_get_prediction_distribution(self):
        """测试获取预测分布"""
        monitor = ModelMonitor("test_model")

        for i in range(100):
            monitor.record_inference([1.0], prediction=float(i))

        dist = monitor.get_prediction_distribution()
        assert dist["count"] == 100
        assert "mean" in dist
        assert "std" in dist

    def test_get_stats(self):
        """测试获取综合统计"""
        monitor = ModelMonitor("test_model", model_version="2.0")

        for i in range(50):
            monitor.record_inference([1.0], prediction=1, label=1, latency_ms=50.0)

        stats = monitor.get_stats()
        assert stats["model_name"] == "test_model"
        assert stats["model_version"] == "2.0"
        assert stats["total_requests"] == 50
        assert stats["accuracy"] == 1.0

    def test_export_metrics(self):
        """测试导出指标"""
        monitor = ModelMonitor("test_model")
        monitor.record_inference([1.0], prediction=1, latency_ms=50.0)

        # JSON 格式
        json_output = monitor.export_metrics(format="json")
        assert isinstance(json_output, str)
        assert "test_model" in json_output

        # 字典格式
        dict_output = monitor.export_metrics(format="dict")
        assert isinstance(dict_output, dict)


class TestCreateMonitor:
    """测试 create_monitor 工厂函数"""

    def test_create_basic_monitor(self):
        """测试创建基本监控器"""
        monitor = create_monitor("test_model")
        assert isinstance(monitor, ModelMonitor)
        assert monitor.model_name == "test_model"

    def test_create_monitor_with_reference(self):
        """测试创建带参考数据的监控器"""
        reference = [float(i) for i in range(100)]
        monitor = create_monitor(
            "test_model",
            reference_data=reference
        )
        assert monitor.drift_detector._reference_data == reference

    def test_create_monitor_with_version(self):
        """测试创建带版本的监控器"""
        monitor = create_monitor(
            "test_model",
            model_version="2.0"
        )
        assert monitor.model_version == "2.0"


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus not installed")
class TestPrometheusExporter:
    """测试 PrometheusExporter"""

    def test_exporter_creation(self):
        """测试创建导出器"""
        from monitoring import PrometheusExporter
        exporter = PrometheusExporter(port=9999, prefix="test")
        assert exporter.port == 9999
        assert exporter.prefix == "test"
