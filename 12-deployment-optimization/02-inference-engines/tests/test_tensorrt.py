"""
TensorRT 模块单元测试
"""

import pytest
import numpy as np
import tempfile
import os
import sys

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorrt_engine import (
    EngineConfig,
    Precision,
    CalibrationAlgorithm,
    TENSORRT_AVAILABLE,
    PYCUDA_AVAILABLE,
)

# 检查 TensorRT 是否可用
skip_if_no_tensorrt = pytest.mark.skipif(
    not TENSORRT_AVAILABLE or not PYCUDA_AVAILABLE,
    reason="TensorRT or PyCUDA not available"
)


# ==================== EngineConfig 测试 ====================

class TestEngineConfig:
    """测试引擎配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = EngineConfig()

        assert config.precision == Precision.FP16
        assert config.workspace_size == 1 << 30  # 1GB
        assert config.min_batch_size == 1
        assert config.opt_batch_size == 8
        assert config.max_batch_size == 32
        assert config.calibration_algorithm == CalibrationAlgorithm.ENTROPY

    def test_custom_config(self):
        """测试自定义配置"""
        config = EngineConfig(
            precision=Precision.INT8,
            workspace_size=2 << 30,  # 2GB
            min_batch_size=1,
            opt_batch_size=16,
            max_batch_size=64,
            calibration_algorithm=CalibrationAlgorithm.MINMAX
        )

        assert config.precision == Precision.INT8
        assert config.workspace_size == 2 << 30
        assert config.opt_batch_size == 16
        assert config.max_batch_size == 64
        assert config.calibration_algorithm == CalibrationAlgorithm.MINMAX


# ==================== 枚举测试 ====================

class TestEnums:
    """测试枚举类型"""

    def test_precision(self):
        """测试精度枚举"""
        assert Precision.FP32.value == "fp32"
        assert Precision.FP16.value == "fp16"
        assert Precision.INT8.value == "int8"

    def test_calibration_algorithm(self):
        """测试校准算法枚举"""
        assert CalibrationAlgorithm.ENTROPY.value == "entropy"
        assert CalibrationAlgorithm.MINMAX.value == "minmax"
        assert CalibrationAlgorithm.PERCENTILE.value == "percentile"


# ==================== TRTLogger 测试 ====================

class TestTRTLogger:
    """测试 TensorRT 日志记录器"""

    @skip_if_no_tensorrt
    def test_create_logger(self):
        """测试创建日志记录器"""
        from tensorrt_engine import TRTLogger

        logger = TRTLogger(log_level=2)
        assert logger is not None
        assert logger.get_logger() is not None

    @skip_if_no_tensorrt
    def test_different_log_levels(self):
        """测试不同日志级别"""
        from tensorrt_engine import TRTLogger

        for level in [0, 1, 2, 3]:
            logger = TRTLogger(log_level=level)
            assert logger is not None


# ==================== EngineBuilder 测试 ====================

class TestEngineBuilder:
    """测试引擎构建器"""

    @skip_if_no_tensorrt
    def test_create_builder(self):
        """测试创建构建器"""
        from tensorrt_engine import EngineBuilder

        config = EngineConfig()
        builder = EngineBuilder(config)
        assert builder is not None

    @skip_if_no_tensorrt
    def test_builder_with_custom_config(self):
        """测试使用自定义配置创建构建器"""
        from tensorrt_engine import EngineBuilder

        config = EngineConfig(
            precision=Precision.FP32,
            workspace_size=512 * 1024 * 1024  # 512MB
        )
        builder = EngineBuilder(config)
        assert builder is not None
        assert builder.config.precision == Precision.FP32


# ==================== Int8Calibrator 测试 ====================

class TestInt8Calibrator:
    """测试 INT8 校准器"""

    @skip_if_no_tensorrt
    def test_create_calibrator(self):
        """测试创建校准器"""
        from tensorrt_engine import Int8Calibrator

        # 创建校准数据
        calibration_data = [
            np.random.randn(1, 3, 224, 224).astype(np.float32)
            for _ in range(10)
        ]

        calibrator = Int8Calibrator(calibration_data)
        assert calibrator is not None
        assert calibrator.get_batch_size() == 1

    @skip_if_no_tensorrt
    def test_calibrator_with_cache(self):
        """测试带缓存的校准器"""
        from tensorrt_engine import Int8Calibrator

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "calibration.cache")

            calibration_data = [
                np.random.randn(1, 3, 224, 224).astype(np.float32)
                for _ in range(5)
            ]

            calibrator = Int8Calibrator(
                calibration_data,
                cache_file=cache_file
            )
            assert calibrator is not None
            assert calibrator.cache_file == cache_file


# ==================== TensorRTEngine 测试 ====================

class TestTensorRTEngine:
    """测试 TensorRT 推理引擎"""

    @skip_if_no_tensorrt
    def test_engine_not_available_without_file(self):
        """测试没有引擎文件时的错误"""
        from tensorrt_engine import TensorRTEngine

        with pytest.raises(ValueError):
            TensorRTEngine()  # 没有提供引擎路径或序列化引擎


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试 (需要 TensorRT 和 ONNX 模型)"""

    @skip_if_no_tensorrt
    @pytest.mark.skip(reason="Requires ONNX model and GPU")
    def test_build_and_run_engine(self):
        """测试构建和运行引擎"""
        from tensorrt_engine import EngineBuilder, TensorRTEngine, EngineConfig

        # 这个测试需要实际的 ONNX 模型和 GPU
        # 在实际环境中取消 skip 装饰器
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
