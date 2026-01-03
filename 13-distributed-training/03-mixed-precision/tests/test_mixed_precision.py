"""
03-mixed-precision 模块测试
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from amp import (
    AMPConfig,
    AMPTrainer,
    get_autocast_dtype,
    autocast_context,
)
from bf16_training import (
    BF16Config,
    BF16Trainer,
    convert_to_bf16,
    is_bf16_supported,
)
from gradient_scaling import (
    GradScalerConfig,
    SmartGradScaler,
    DynamicLossScaler,
)


# ============== AMP 测试 ==============

class TestAMPConfig:
    """AMPConfig 测试"""
    
    def test_default_config(self):
        config = AMPConfig()
        assert config.enabled is True
        assert config.dtype == torch.float16
        assert config.use_grad_scaler is True
    
    def test_custom_config(self):
        config = AMPConfig(
            dtype=torch.bfloat16,
            init_scale=32768.0,
            growth_interval=1000,
        )
        assert config.dtype == torch.bfloat16
        assert config.init_scale == 32768.0
        assert config.growth_interval == 1000


class TestGetAutocastDtype:
    """get_autocast_dtype 测试"""
    
    def test_float16(self):
        assert get_autocast_dtype("float16") == torch.float16
        assert get_autocast_dtype("fp16") == torch.float16
    
    def test_bfloat16(self):
        assert get_autocast_dtype("bfloat16") == torch.bfloat16
        assert get_autocast_dtype("bf16") == torch.bfloat16
    
    def test_default(self):
        assert get_autocast_dtype("unknown") == torch.float16


class TestAMPTrainer:
    """AMPTrainer 测试"""
    
    @pytest.fixture
    def simple_model(self):
        return nn.Linear(10, 5)
    
    def test_trainer_init(self, simple_model):
        config = AMPConfig(enabled=False)
        trainer = AMPTrainer(simple_model, config)
        assert trainer.config.enabled is False
        assert trainer.scaler is None
    
    def test_trainer_with_scaler(self, simple_model):
        config = AMPConfig(enabled=True, use_grad_scaler=True)
        trainer = AMPTrainer(simple_model, config, device=torch.device("cpu"))
        assert trainer.scaler is not None
    
    def test_get_scale_no_scaler(self, simple_model):
        config = AMPConfig(enabled=False)
        trainer = AMPTrainer(simple_model, config)
        assert trainer.get_scale() == 1.0
    
    def test_state_dict(self, simple_model):
        config = AMPConfig(enabled=False)
        trainer = AMPTrainer(simple_model, config)
        state = trainer.state_dict()
        assert "model" in state


# ============== BF16 测试 ==============

class TestBF16Config:
    """BF16Config 测试"""
    
    def test_default_config(self):
        config = BF16Config()
        assert config.enabled is True
        assert config.convert_weights is True
        assert config.keep_batchnorm_fp32 is True
    
    def test_custom_config(self):
        config = BF16Config(
            keep_layernorm_fp32=False,
            master_weights=False,
        )
        assert config.keep_layernorm_fp32 is False
        assert config.master_weights is False


class TestConvertToBF16:
    """convert_to_bf16 测试"""
    
    def test_convert_linear(self):
        model = nn.Linear(10, 5)
        model_bf16 = convert_to_bf16(model)
        assert model_bf16.weight.dtype == torch.bfloat16
    
    def test_keep_batchnorm_fp32(self):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
        )
        model_bf16 = convert_to_bf16(model, keep_batchnorm_fp32=True)
        assert model_bf16[0].weight.dtype == torch.bfloat16
        assert model_bf16[1].weight.dtype == torch.float32
    
    def test_keep_layernorm_fp32(self):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.LayerNorm(10),
        )
        model_bf16 = convert_to_bf16(model, keep_layernorm_fp32=True)
        assert model_bf16[0].weight.dtype == torch.bfloat16
        assert model_bf16[1].weight.dtype == torch.float32


class TestBF16Trainer:
    """BF16Trainer 测试"""
    
    @pytest.fixture
    def simple_model(self):
        return nn.Linear(10, 5)
    
    def test_trainer_disabled(self, simple_model):
        config = BF16Config(enabled=False)
        trainer = BF16Trainer(simple_model, config, device=torch.device("cpu"))
        assert trainer.config.enabled is False
    
    def test_state_dict(self, simple_model):
        config = BF16Config(enabled=False)
        trainer = BF16Trainer(simple_model, config, device=torch.device("cpu"))
        state = trainer.state_dict()
        assert "model" in state


# ============== Gradient Scaling 测试 ==============

class TestGradScalerConfig:
    """GradScalerConfig 测试"""
    
    def test_default_config(self):
        config = GradScalerConfig()
        assert config.init_scale == 65536.0
        assert config.growth_factor == 2.0
        assert config.backoff_factor == 0.5
    
    def test_custom_config(self):
        config = GradScalerConfig(
            init_scale=32768.0,
            growth_interval=1000,
            max_scale=2.0 ** 20,
        )
        assert config.init_scale == 32768.0
        assert config.growth_interval == 1000


class TestSmartGradScaler:
    """SmartGradScaler 测试"""
    
    def test_init(self):
        scaler = SmartGradScaler()
        assert scaler.get_scale() == 65536.0
    
    def test_scale_loss(self):
        scaler = SmartGradScaler()
        loss = torch.tensor(1.0)
        scaled = scaler.scale(loss)
        assert scaled.item() == 65536.0
    
    def test_scale_disabled(self):
        config = GradScalerConfig(enabled=False)
        scaler = SmartGradScaler(config)
        loss = torch.tensor(1.0)
        scaled = scaler.scale(loss)
        assert scaled.item() == 1.0
    
    def test_state_dict(self):
        scaler = SmartGradScaler()
        state = scaler.state_dict()
        assert "scale" in state
        assert "growth_tracker" in state
    
    def test_load_state_dict(self):
        scaler = SmartGradScaler()
        state = {"scale": 1024.0, "growth_tracker": 100, "overflow_count": 5, "total_steps": 1000}
        scaler.load_state_dict(state)
        assert scaler.get_scale() == 1024.0
        assert scaler._growth_tracker == 100
    
    def test_overflow_ratio(self):
        scaler = SmartGradScaler()
        assert scaler.get_overflow_ratio() == 0.0


class TestDynamicLossScaler:
    """DynamicLossScaler 测试"""
    
    def test_init(self):
        scaler = DynamicLossScaler()
        assert scaler.loss_scale() == 2.0 ** 16
    
    def test_update_scale_overflow(self):
        scaler = DynamicLossScaler(init_scale=1024.0)
        scaler.update_scale(overflow=True)
        assert scaler.scale == 512.0
    
    def test_update_scale_no_overflow(self):
        scaler = DynamicLossScaler(init_scale=1024.0, scale_window=10)
        for _ in range(10):
            scaler.update_scale(overflow=False)
        assert scaler.scale == 2048.0
    
    def test_min_scale(self):
        scaler = DynamicLossScaler(init_scale=2.0, min_scale=1.0)
        scaler.update_scale(overflow=True)
        assert scaler.scale == 1.0
        scaler.update_scale(overflow=True)
        assert scaler.scale == 1.0
    
    def test_has_overflow(self):
        scaler = DynamicLossScaler()
        param = nn.Parameter(torch.tensor([1.0, 2.0]))
        param.grad = torch.tensor([1.0, float("inf")])
        assert scaler.has_overflow([param]) is True
    
    def test_no_overflow(self):
        scaler = DynamicLossScaler()
        param = nn.Parameter(torch.tensor([1.0, 2.0]))
        param.grad = torch.tensor([1.0, 2.0])
        assert scaler.has_overflow([param]) is False
    
    def test_state_dict(self):
        scaler = DynamicLossScaler()
        state = scaler.state_dict()
        assert "scale" in state
        assert "step_count" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
