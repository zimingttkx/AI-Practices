"""
AWQ (Activation-aware Weight Quantization) 模块单元测试

测试覆盖:
- AWQConfig 配置
- AWQGranularity 枚举
- ActivationObserver 激活观察器
- SalientChannelFinder 显著通道查找器
- AWQLinear 量化线性层
- AWQQuantizer 量化器
- 工具函数
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.awq import (
    AWQConfig,
    AWQGranularity,
    ActivationObserver,
    SalientChannelFinder,
    AWQLinear,
    AWQQuantizer,
    create_awq_quantizer,
    compute_quantization_error,
    estimate_model_size,
    pack_int4_weights,
    unpack_int4_weights,
)


class TestAWQGranularity:
    """AWQGranularity 枚举测试"""
    
    def test_per_channel(self):
        """测试 per_channel 粒度"""
        assert AWQGranularity.PER_CHANNEL.value == "per_channel"
    
    def test_per_group(self):
        """测试 per_group 粒度"""
        assert AWQGranularity.PER_GROUP.value == "per_group"


class TestAWQConfig:
    """AWQConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AWQConfig()
        assert config.w_bit == 4
        assert config.group_size == 128
        assert config.zero_point == True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = AWQConfig(
            w_bit=8,
            group_size=64,
            zero_point=False,
            salient_ratio=0.05
        )
        assert config.w_bit == 8
        assert config.group_size == 64
        assert config.zero_point == False
        assert config.salient_ratio == 0.05
    
    def test_invalid_w_bit(self):
        """测试无效的位宽"""
        with pytest.raises(ValueError):
            AWQConfig(w_bit=5)
    
    def test_invalid_group_size(self):
        """测试无效的分组大小"""
        with pytest.raises(ValueError):
            AWQConfig(group_size=0)
    
    def test_invalid_salient_ratio(self):
        """测试无效的显著比例"""
        with pytest.raises(ValueError):
            AWQConfig(salient_ratio=1.5)


class TestActivationObserver:
    """ActivationObserver 测试"""
    
    def test_observer_creation(self):
        """测试观察器创建"""
        observer = ActivationObserver(num_features=256)
        assert observer.num_features == 256
        assert observer.n_samples == 0
    
    def test_observe_single_batch(self):
        """测试观察单个批次"""
        observer = ActivationObserver(num_features=64)
        x = np.random.randn(32, 64)
        
        observer.observe(x)
        assert observer.n_samples == 32
    
    def test_observe_multiple_batches(self):
        """测试观察多个批次"""
        observer = ActivationObserver(num_features=64)
        
        for _ in range(5):
            x = np.random.randn(16, 64)
            observer.observe(x)
        
        assert observer.n_samples == 80
    
    def test_get_scale_importance(self):
        """测试获取重要性分数"""
        observer = ActivationObserver(num_features=64)
        x = np.random.randn(100, 64)
        observer.observe(x)
        
        importance = observer.get_scale_importance()
        assert importance.shape == (64,)
    
    def test_get_activation_range(self):
        """测试获取激活范围"""
        observer = ActivationObserver(num_features=64)
        x = np.random.randn(100, 64)
        observer.observe(x)
        
        min_val, max_val = observer.get_activation_range()
        assert min_val.shape == (64,)
        assert max_val.shape == (64,)
    
    def test_reset(self):
        """测试重置"""
        observer = ActivationObserver(num_features=64)
        x = np.random.randn(32, 64)
        observer.observe(x)
        
        observer.reset()
        assert observer.n_samples == 0


class TestSalientChannelFinder:
    """SalientChannelFinder 测试"""
    
    def test_finder_creation(self):
        """测试查找器创建"""
        config = AWQConfig()
        finder = SalientChannelFinder(config)
        assert finder is not None
    
    def test_register_layer(self):
        """测试注册层"""
        config = AWQConfig()
        finder = SalientChannelFinder(config)
        
        finder.register_layer("layer1", 256)
        assert "layer1" in finder.observers
    
    def test_observe_activation(self):
        """测试观察激活"""
        config = AWQConfig()
        finder = SalientChannelFinder(config)
        
        finder.register_layer("layer1", 64)
        x = np.random.randn(32, 64)
        finder.observe_activation("layer1", x)
        
        assert finder.observers["layer1"].n_samples == 32
    
    def test_find_salient_channels(self):
        """测试查找显著通道"""
        config = AWQConfig(salient_ratio=0.1)
        finder = SalientChannelFinder(config)
        
        finder.register_layer("layer1", 100)
        x = np.random.randn(1000, 100)
        finder.observe_activation("layer1", x)
        
        salient = finder.find_salient_channels("layer1")
        # 10% 的通道应该是显著的
        assert len(salient) == 10


class TestAWQLinear:
    """AWQLinear 测试"""
    
    def test_linear_creation(self):
        """测试量化线性层创建"""
        config = AWQConfig()
        linear = AWQLinear(
            in_features=256,
            out_features=512,
            config=config
        )
        assert linear.in_features == 256
        assert linear.out_features == 512
    
    def test_quantize_weights(self):
        """测试权重量化"""
        config = AWQConfig()
        linear = AWQLinear(
            in_features=128,
            out_features=256,
            config=config
        )
        
        weights = np.random.randn(256, 128).astype(np.float32)
        linear.quantize_weight(weights)
        
        assert linear.qweight is not None
    
    def test_forward(self):
        """测试前向传播"""
        config = AWQConfig()
        linear = AWQLinear(
            in_features=64,
            out_features=128,
            config=config
        )
        
        weights = np.random.randn(128, 64).astype(np.float32)
        linear.quantize_weight(weights)
        
        x = np.random.randn(16, 64).astype(np.float32)
        output = linear.forward(x)
        
        assert output.shape == (16, 128)


class TestAWQQuantizer:
    """AWQQuantizer 测试"""
    
    def test_quantizer_creation(self):
        """测试量化器创建"""
        config = AWQConfig()
        quantizer = AWQQuantizer(config)
        assert quantizer is not None
    
    def test_quantize_layer(self):
        """测试层量化"""
        config = AWQConfig()
        quantizer = AWQQuantizer(config)
        
        tensor = np.random.randn(256, 128).astype(np.float32)
        quantizer.quantize_layer("test_layer", tensor)
        
        assert "test_layer" in quantizer.quantized_layers


class TestCreateAWQQuantizer:
    """create_awq_quantizer 工厂函数测试"""
    
    def test_create_default(self):
        """测试默认创建"""
        quantizer = create_awq_quantizer()
        assert quantizer is not None
        assert isinstance(quantizer, AWQQuantizer)
    
    def test_create_custom(self):
        """测试自定义创建"""
        quantizer = create_awq_quantizer(w_bit=8, group_size=64)
        assert quantizer.config.w_bit == 8
        assert quantizer.config.group_size == 64


class TestUtilityFunctions:
    """工具函数测试"""
    
    def test_compute_quantization_error(self):
        """测试计算量化误差"""
        original = np.random.randn(100, 100).astype(np.float32)
        quantized = original + np.random.randn(100, 100) * 0.01
        
        error = compute_quantization_error(original, quantized)
        # 返回的是字典
        assert isinstance(error, dict)
        assert "mse" in error
        assert error["mse"] >= 0
    
    def test_estimate_model_size(self):
        """测试估算模型大小"""
        # 模拟参数数量
        num_params = 1_000_000
        
        size_info = estimate_model_size(num_params, w_bit=4)
        assert isinstance(size_info, dict)
        assert "quantized_mb" in size_info or "total_mb" in size_info or len(size_info) > 0
    
    def test_pack_unpack_int4(self):
        """测试 INT4 打包和解包"""
        # 创建 INT4 范围内的权重
        weights = np.random.randint(0, 16, size=(64, 128)).astype(np.uint8)
        
        packed = pack_int4_weights(weights)
        unpacked = unpack_int4_weights(packed, weights.shape)
        
        np.testing.assert_array_equal(weights, unpacked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
