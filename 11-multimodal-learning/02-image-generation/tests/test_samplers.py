"""
高级采样器单元测试

测试 DPM++, UniPC, Euler, Heun, LMS 等采样器的正确性
"""

import pytest
import torch
import math
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from samplers import (
    # 枚举
    SamplerType,
    PredictionType,
    # 配置
    SamplerConfig,
    DPMPPConfig,
    UniPCConfig,
    EulerConfig,
    HeunConfig,
    LMSConfig,
    # 采样器
    BaseSampler,
    NoiseSchedule,
    DPMPPSampler,
    UniPCSampler,
    EulerSampler,
    EulerAncestralSampler,
    HeunSampler,
    LMSSampler,
    # 工厂函数
    create_sampler,
)


# ============================================================================
# 测试配置
# ============================================================================

class TestSamplerConfig:
    """测试采样器配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SamplerConfig()
        assert config.num_train_timesteps == 1000
        assert config.num_inference_steps == 50
        assert config.beta_start == 0.00085
        assert config.beta_end == 0.012
        assert config.beta_schedule == "scaled_linear"
        assert config.prediction_type == PredictionType.EPSILON
        assert config.clip_sample is True
        
    def test_custom_config(self):
        """测试自定义配置"""
        config = SamplerConfig(
            num_train_timesteps=500,
            num_inference_steps=25,
            beta_schedule="linear",
            prediction_type=PredictionType.V_PREDICTION
        )
        assert config.num_train_timesteps == 500
        assert config.num_inference_steps == 25
        assert config.beta_schedule == "linear"
        assert config.prediction_type == PredictionType.V_PREDICTION
        
    def test_prediction_type_string_conversion(self):
        """测试预测类型字符串转换"""
        config = SamplerConfig(prediction_type="epsilon")
        assert config.prediction_type == PredictionType.EPSILON
        
        config = SamplerConfig(prediction_type="v")
        assert config.prediction_type == PredictionType.V_PREDICTION


class TestDPMPPConfig:
    """测试 DPM++ 配置"""
    
    def test_default_dpmpp_config(self):
        """测试默认 DPM++ 配置"""
        config = DPMPPConfig()
        assert config.solver_order == 2
        assert config.solver_type == "midpoint"
        assert config.lower_order_final is True
        assert config.use_karras_sigmas is False
        assert config.algorithm_type == "dpmsolver++"
        
    def test_custom_dpmpp_config(self):
        """测试自定义 DPM++ 配置"""
        config = DPMPPConfig(
            solver_order=3,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver"
        )
        assert config.solver_order == 3
        assert config.use_karras_sigmas is True
        assert config.algorithm_type == "dpmsolver"


class TestUniPCConfig:
    """测试 UniPC 配置"""
    
    def test_default_unipc_config(self):
        """测试默认 UniPC 配置"""
        config = UniPCConfig()
        assert config.solver_order == 2
        assert config.predict_x0 is True
        assert config.thresholding is False
        assert config.variant == "bh1"


class TestEulerConfig:
    """测试 Euler 配置"""
    
    def test_default_euler_config(self):
        """测试默认 Euler 配置"""
        config = EulerConfig()
        assert config.use_karras_sigmas is False
        assert config.s_churn == 0.0
        assert config.s_noise == 1.0


# ============================================================================
# 测试噪声调度
# ============================================================================

class TestNoiseSchedule:
    """测试噪声调度"""
    
    def test_linear_schedule(self):
        """测试线性调度"""
        config = SamplerConfig(beta_schedule="linear")
        schedule = NoiseSchedule(config)
        
        assert len(schedule.betas) == config.num_train_timesteps
        assert schedule.betas[0] == pytest.approx(config.beta_start, rel=1e-5)
        assert schedule.betas[-1] == pytest.approx(config.beta_end, rel=1e-5)
        
    def test_scaled_linear_schedule(self):
        """测试缩放线性调度"""
        config = SamplerConfig(beta_schedule="scaled_linear")
        schedule = NoiseSchedule(config)
        
        assert len(schedule.betas) == config.num_train_timesteps
        # 缩放线性: beta = (sqrt(beta_start) + t * (sqrt(beta_end) - sqrt(beta_start)))^2
        assert schedule.betas[0] == pytest.approx(config.beta_start, rel=1e-5)
        assert schedule.betas[-1] == pytest.approx(config.beta_end, rel=1e-5)
        
    def test_cosine_schedule(self):
        """测试余弦调度"""
        config = SamplerConfig(beta_schedule="cosine")
        schedule = NoiseSchedule(config)
        
        assert len(schedule.betas) == config.num_train_timesteps
        # 余弦调度的 beta 应该在合理范围内
        assert torch.all(schedule.betas >= 0.0001)
        assert torch.all(schedule.betas <= 0.9999)
        
    def test_alphas_cumprod(self):
        """测试累积 alpha"""
        config = SamplerConfig()
        schedule = NoiseSchedule(config)
        
        # alphas_cumprod 应该单调递减
        for i in range(len(schedule.alphas_cumprod) - 1):
            assert schedule.alphas_cumprod[i] > schedule.alphas_cumprod[i + 1]
            
        # 第一个值应该接近 1，最后一个值应该接近 0
        assert schedule.alphas_cumprod[0] > 0.99
        assert schedule.alphas_cumprod[-1] < 0.01
        
    def test_sqrt_values(self):
        """测试平方根值"""
        config = SamplerConfig()
        schedule = NoiseSchedule(config)
        
        # 验证平方根计算
        expected_sqrt = torch.sqrt(schedule.alphas_cumprod)
        assert torch.allclose(schedule.sqrt_alphas_cumprod, expected_sqrt)
        
        expected_sqrt_one_minus = torch.sqrt(1.0 - schedule.alphas_cumprod)
        assert torch.allclose(schedule.sqrt_one_minus_alphas_cumprod, expected_sqrt_one_minus)
        
    def test_sigmas(self):
        """测试 sigma 值"""
        config = SamplerConfig()
        schedule = NoiseSchedule(config)
        
        # sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
        expected_sigmas = ((1 - schedule.alphas_cumprod) / schedule.alphas_cumprod) ** 0.5
        assert torch.allclose(schedule.sigmas, expected_sigmas)


# ============================================================================
# 测试 DPM++ 采样器
# ============================================================================

class TestDPMPPSampler:
    """测试 DPM++ 采样器"""
    
    @pytest.fixture
    def sampler(self):
        """创建 DPM++ 采样器"""
        config = DPMPPConfig(num_inference_steps=20)
        return DPMPPSampler(config)
    
    def test_initialization(self, sampler):
        """测试初始化"""
        assert sampler.config.solver_order == 2
        assert sampler.model_outputs == []
        assert sampler.sample_history == []
        
    def test_set_timesteps(self, sampler):
        """测试设置时间步"""
        sampler.set_timesteps(20)
        
        assert sampler.timesteps is not None
        assert sampler.sigmas is not None
        assert len(sampler.timesteps) == 20
        assert len(sampler.sigmas) == 21  # 包含最终的 0
        
    def test_set_timesteps_with_device(self, sampler):
        """测试设置时间步到指定设备"""
        device = torch.device("cpu")
        sampler.set_timesteps(20, device=device)
        
        assert sampler.timesteps.device == device
        assert sampler.sigmas.device == device
        
    def test_karras_sigmas(self):
        """测试 Karras sigma 调度"""
        config = DPMPPConfig(use_karras_sigmas=True, num_inference_steps=20)
        sampler = DPMPPSampler(config)
        sampler.set_timesteps(20)
        
        # Karras sigmas 应该单调递减
        for i in range(len(sampler.sigmas) - 1):
            assert sampler.sigmas[i] >= sampler.sigmas[i + 1]
            
    def test_step_output_shape(self, sampler):
        """测试单步输出形状"""
        sampler.set_timesteps(20)
        
        batch_size, channels, height, width = 2, 4, 32, 32
        sample = torch.randn(batch_size, channels, height, width)
        model_output = torch.randn(batch_size, channels, height, width)
        timestep = sampler.timesteps[0].item()
        
        output = sampler.step(model_output, timestep, sample)
        
        assert output.shape == sample.shape
        
    def test_step_reduces_noise(self, sampler):
        """测试单步是否减少噪声"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        
        # 执行多步
        for i, t in enumerate(sampler.timesteps[:5]):
            sample = sampler.step(model_output, t.item(), sample)
            
        # 输出应该是有限的
        assert torch.isfinite(sample).all()
        
    def test_first_order_update(self, sampler):
        """测试一阶更新"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        
        # 第一步应该使用一阶方法
        output = sampler.step(model_output, sampler.timesteps[0].item(), sample)
        
        assert output.shape == sample.shape
        assert torch.isfinite(output).all()
        
    def test_second_order_update(self, sampler):
        """测试二阶更新"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        
        # 执行两步以触发二阶更新
        sample = sampler.step(model_output, sampler.timesteps[0].item(), sample)
        sample = sampler.step(model_output, sampler.timesteps[1].item(), sample)
        
        assert torch.isfinite(sample).all()


# ============================================================================
# 测试 UniPC 采样器
# ============================================================================

class TestUniPCSampler:
    """测试 UniPC 采样器"""
    
    @pytest.fixture
    def sampler(self):
        """创建 UniPC 采样器"""
        config = UniPCConfig(num_inference_steps=20)
        return UniPCSampler(config)
    
    def test_initialization(self, sampler):
        """测试初始化"""
        assert sampler.config.solver_order == 2
        assert sampler.model_outputs == []
        assert sampler.timestep_list == []
        
    def test_set_timesteps(self, sampler):
        """测试设置时间步"""
        sampler.set_timesteps(20)
        
        assert sampler.timesteps is not None
        assert len(sampler.timesteps) == 20
        # 时间步应该从大到小
        for i in range(len(sampler.timesteps) - 1):
            assert sampler.timesteps[i] > sampler.timesteps[i + 1]
            
    def test_step_output_shape(self, sampler):
        """测试单步输出形状"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        timestep = sampler.timesteps[0].item()
        
        output = sampler.step(model_output, timestep, sample)
        
        assert output.shape == sample.shape
        
    def test_threshold_sample(self):
        """测试动态阈值"""
        config = UniPCConfig(thresholding=True, sample_max_value=1.0)
        sampler = UniPCSampler(config)
        
        # 创建超出范围的样本
        sample = torch.randn(2, 4, 32, 32) * 10
        
        thresholded = sampler._threshold_sample(sample)
        
        # 阈值处理后应该在范围内
        assert torch.all(thresholded >= -1.0)
        assert torch.all(thresholded <= 1.0)
        
    def test_multistep_update(self, sampler):
        """测试多步更新"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        
        # 执行多步
        for t in sampler.timesteps[:5]:
            sample = sampler.step(model_output, t.item(), sample)
            
        assert torch.isfinite(sample).all()


# ============================================================================
# 测试 Euler 采样器
# ============================================================================

class TestEulerSampler:
    """测试 Euler 采样器"""
    
    @pytest.fixture
    def sampler(self):
        """创建 Euler 采样器"""
        config = EulerConfig(num_inference_steps=20)
        return EulerSampler(config)
    
    def test_initialization(self, sampler):
        """测试初始化"""
        assert sampler.sigmas is None
        assert sampler.timesteps is None
        
    def test_set_timesteps(self, sampler):
        """测试设置时间步"""
        sampler.set_timesteps(20)
        
        assert sampler.timesteps is not None
        assert sampler.sigmas is not None
        assert len(sampler.timesteps) == 20
        assert len(sampler.sigmas) == 21
        
    def test_karras_sigmas(self):
        """测试 Karras sigma"""
        config = EulerConfig(use_karras_sigmas=True, num_inference_steps=20)
        sampler = EulerSampler(config)
        sampler.set_timesteps(20)
        
        # sigmas 应该单调递减
        for i in range(len(sampler.sigmas) - 1):
            assert sampler.sigmas[i] >= sampler.sigmas[i + 1]
            
    def test_step_output_shape(self, sampler):
        """测试单步输出形状"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        timestep = sampler.timesteps[0].item()
        
        output = sampler.step(model_output, timestep, sample)
        
        assert output.shape == sample.shape
        
    def test_full_sampling_loop(self, sampler):
        """测试完整采样循环"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        
        for t in sampler.timesteps:
            model_output = torch.randn_like(sample)
            sample = sampler.step(model_output, t.item(), sample)
            
        assert torch.isfinite(sample).all()


class TestEulerAncestralSampler:
    """测试 Euler Ancestral 采样器"""
    
    @pytest.fixture
    def sampler(self):
        """创建 Euler Ancestral 采样器"""
        config = EulerConfig(num_inference_steps=20)
        return EulerAncestralSampler(config)
    
    def test_step_with_noise(self, sampler):
        """测试带噪声的单步"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        timestep = sampler.timesteps[0].item()
        
        # 使用固定种子
        generator = torch.Generator().manual_seed(42)
        output = sampler.step(model_output, timestep, sample, generator=generator)
        
        assert output.shape == sample.shape
        assert torch.isfinite(output).all()
        
    def test_reproducibility(self, sampler):
        """测试可重复性"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        timestep = sampler.timesteps[0].item()
        
        # 相同种子应该产生相同结果
        gen1 = torch.Generator().manual_seed(42)
        output1 = sampler.step(model_output, timestep, sample.clone(), generator=gen1)
        
        gen2 = torch.Generator().manual_seed(42)
        output2 = sampler.step(model_output, timestep, sample.clone(), generator=gen2)
        
        assert torch.allclose(output1, output2)


# ============================================================================
# 测试 Heun 采样器
# ============================================================================

class TestHeunSampler:
    """测试 Heun 采样器"""
    
    @pytest.fixture
    def sampler(self):
        """创建 Heun 采样器"""
        config = HeunConfig(num_inference_steps=20)
        return HeunSampler(config)
    
    def test_initialization(self, sampler):
        """测试初始化"""
        assert sampler._prev_derivative is None
        assert sampler._dt is None
        assert sampler._sample is None
        
    def test_set_timesteps(self, sampler):
        """测试设置时间步"""
        sampler.set_timesteps(20)
        
        assert sampler.timesteps is not None
        assert sampler.sigmas is not None
        
    def test_step_output_shape(self, sampler):
        """测试单步输出形状"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        timestep = sampler.timesteps[0].item()
        
        output = sampler.step(model_output, timestep, sample)
        
        assert output.shape == sample.shape
        
    def test_two_step_correction(self, sampler):
        """测试两步校正"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        
        # 第一步: Euler 预测
        output1 = sampler.step(model_output, sampler.timesteps[0].item(), sample)
        assert sampler._prev_derivative is not None
        
        # 第二步: Heun 校正
        output2 = sampler.step(model_output, sampler.timesteps[0].item(), output1)
        assert sampler._prev_derivative is None  # 重置


# ============================================================================
# 测试 LMS 采样器
# ============================================================================

class TestLMSSampler:
    """测试 LMS 采样器"""
    
    @pytest.fixture
    def sampler(self):
        """创建 LMS 采样器"""
        config = LMSConfig(num_inference_steps=20, order=4)
        return LMSSampler(config)
    
    def test_initialization(self, sampler):
        """测试初始化"""
        assert sampler.config.order == 4
        assert sampler.derivatives == []
        
    def test_set_timesteps(self, sampler):
        """测试设置时间步"""
        sampler.set_timesteps(20)
        
        assert sampler.timesteps is not None
        assert sampler.sigmas is not None
        assert sampler.derivatives == []
        
    def test_lms_coefficients(self, sampler):
        """测试 LMS 系数计算"""
        # 一阶系数
        coeffs1 = sampler._get_lms_coefficients(1, 0, 1)
        assert len(coeffs1) == 1
        assert coeffs1[0] == pytest.approx(1.0)
        
        # 二阶系数
        coeffs2 = sampler._get_lms_coefficients(2, 1, 2)
        assert len(coeffs2) == 2
        
    def test_step_accumulates_derivatives(self, sampler):
        """测试步骤累积导数"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn(2, 4, 32, 32)
        
        # 执行多步
        for i, t in enumerate(sampler.timesteps[:5]):
            sample = sampler.step(model_output, t.item(), sample)
            expected_len = min(i + 1, sampler.config.order)
            assert len(sampler.derivatives) == expected_len
            
    def test_full_sampling_loop(self, sampler):
        """测试完整采样循环"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        
        for t in sampler.timesteps:
            model_output = torch.randn_like(sample)
            sample = sampler.step(model_output, t.item(), sample)
            
        assert torch.isfinite(sample).all()


# ============================================================================
# 测试工厂函数
# ============================================================================

class TestCreateSampler:
    """测试采样器工厂函数"""
    
    def test_create_dpmpp_2m(self):
        """测试创建 DPM++ 2M"""
        sampler = create_sampler("dpm++_2m")
        assert isinstance(sampler, DPMPPSampler)
        assert sampler.config.solver_order == 2
        
    def test_create_dpmpp_2s(self):
        """测试创建 DPM++ 2S"""
        sampler = create_sampler("dpm++_2s")
        assert isinstance(sampler, DPMPPSampler)
        assert sampler.config.solver_type == "heun"
        
    def test_create_unipc(self):
        """测试创建 UniPC"""
        sampler = create_sampler("unipc")
        assert isinstance(sampler, UniPCSampler)
        
    def test_create_euler(self):
        """测试创建 Euler"""
        sampler = create_sampler("euler")
        assert isinstance(sampler, EulerSampler)
        
    def test_create_euler_ancestral(self):
        """测试创建 Euler Ancestral"""
        sampler = create_sampler("euler_a")
        assert isinstance(sampler, EulerAncestralSampler)
        
    def test_create_heun(self):
        """测试创建 Heun"""
        sampler = create_sampler("heun")
        assert isinstance(sampler, HeunSampler)
        
    def test_create_lms(self):
        """测试创建 LMS"""
        sampler = create_sampler("lms")
        assert isinstance(sampler, LMSSampler)
        
    def test_create_with_enum(self):
        """测试使用枚举创建"""
        sampler = create_sampler(SamplerType.EULER)
        assert isinstance(sampler, EulerSampler)
        
    def test_create_with_custom_params(self):
        """测试使用自定义参数创建"""
        sampler = create_sampler(
            "euler",
            num_train_timesteps=500,
            num_inference_steps=25
        )
        assert sampler.config.num_train_timesteps == 500
        assert sampler.config.num_inference_steps == 25
        
    def test_create_unknown_type(self):
        """测试创建未知类型"""
        with pytest.raises(ValueError):
            create_sampler("unknown_sampler")


# ============================================================================
# 测试枚举
# ============================================================================

class TestEnums:
    """测试枚举类型"""
    
    def test_sampler_type_values(self):
        """测试采样器类型值"""
        assert SamplerType.DDPM.value == "ddpm"
        assert SamplerType.DDIM.value == "ddim"
        assert SamplerType.DPM_PP_2M.value == "dpm++_2m"
        assert SamplerType.EULER.value == "euler"
        assert SamplerType.EULER_ANCESTRAL.value == "euler_a"
        assert SamplerType.HEUN.value == "heun"
        assert SamplerType.LMS.value == "lms"
        
    def test_prediction_type_values(self):
        """测试预测类型值"""
        assert PredictionType.EPSILON.value == "epsilon"
        assert PredictionType.V_PREDICTION.value == "v"
        assert PredictionType.SAMPLE.value == "sample"


# ============================================================================
# 测试 BaseSampler 方法
# ============================================================================

class TestBaseSamplerMethods:
    """测试 BaseSampler 基类方法"""
    
    @pytest.fixture
    def sampler(self):
        """创建采样器"""
        config = EulerConfig(num_inference_steps=20)
        return EulerSampler(config)
    
    def test_add_noise(self, sampler):
        """测试添加噪声"""
        original = torch.randn(2, 4, 32, 32)
        noise = torch.randn_like(original)
        timesteps = torch.tensor([500, 500])
        
        noisy = sampler.add_noise(original, noise, timesteps)
        
        assert noisy.shape == original.shape
        assert not torch.allclose(noisy, original)
        
    def test_get_prev_timestep(self, sampler):
        """测试获取上一个时间步"""
        sampler.set_timesteps(20)
        
        t = sampler.timesteps[0].item()
        prev_t = sampler._get_prev_timestep(t)
        assert prev_t == sampler.timesteps[1].item()
        
    def test_convert_model_output_epsilon(self, sampler):
        """测试转换模型输出 (epsilon)"""
        sampler.set_timesteps(20)
        
        sample = torch.randn(2, 4, 32, 32)
        model_output = torch.randn_like(sample)
        timestep = 500
        
        pred_x0 = sampler._convert_model_output(model_output, timestep, sample)
        
        assert pred_x0.shape == sample.shape
        assert torch.isfinite(pred_x0).all()


# ============================================================================
# 集成测试
# ============================================================================

class TestSamplerIntegration:
    """采样器集成测试"""
    
    @pytest.mark.parametrize("sampler_type", [
        "dpm++_2m", "unipc", "euler", "euler_a", "heun", "lms"
    ])
    def test_all_samplers_produce_finite_output(self, sampler_type):
        """测试所有采样器产生有限输出"""
        sampler = create_sampler(sampler_type, num_inference_steps=10)
        sampler.set_timesteps(10)
        
        sample = torch.randn(1, 4, 16, 16)
        
        for t in sampler.timesteps[:5]:
            model_output = torch.randn_like(sample)
            sample = sampler.step(model_output, t.item(), sample)
            
        assert torch.isfinite(sample).all()
        
    @pytest.mark.parametrize("sampler_type", [
        "dpm++_2m", "unipc", "euler", "heun", "lms"
    ])
    def test_samplers_deterministic(self, sampler_type):
        """测试采样器确定性"""
        torch.manual_seed(42)
        
        sampler1 = create_sampler(sampler_type, num_inference_steps=10)
        sampler1.set_timesteps(10)
        
        sample1 = torch.randn(1, 4, 16, 16)
        model_output = torch.randn_like(sample1)
        
        output1 = sampler1.step(model_output, sampler1.timesteps[0].item(), sample1.clone())
        
        sampler2 = create_sampler(sampler_type, num_inference_steps=10)
        sampler2.set_timesteps(10)
        
        output2 = sampler2.step(model_output, sampler2.timesteps[0].item(), sample1.clone())
        
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
