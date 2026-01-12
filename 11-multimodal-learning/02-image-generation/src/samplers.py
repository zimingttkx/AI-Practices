"""
高级采样器实现

本模块实现了多种先进的扩散模型采样算法，包括 DPM++、UniPC、Euler 等。
这些采样器可以显著减少采样步数，同时保持或提升生成质量。

=== 核心思想 ===

1. DPM++ (Diffusion Probabilistic Model Solver++)
   - 基于 ODE 求解器的高阶方法
   - 支持 2M (多步) 和 2S (单步) 变体
   - 通常 20-25 步即可获得高质量结果

2. UniPC (Unified Predictor-Corrector)
   - 统一的预测-校正框架
   - 结合多种求解器的优点
   - 支持任意阶数的求解

3. Euler 采样器
   - 简单高效的一阶方法
   - Euler Ancestral 添加随机性
   - 适合快速原型和测试

4. Heun 采样器
   - 二阶 Runge-Kutta 方法
   - 比 Euler 更准确
   - 每步需要两次模型评估

=== 数学基础 ===

扩散 ODE:
    dx/dt = f(x, t) = -0.5 * β(t) * [x + s(x, t)]
    
    其中 s(x, t) 是 score function

DPM++ 2M 更新:
    x_{t-1} = (σ_{t-1}/σ_t) * x_t + (α_{t-1} - α_t * σ_{t-1}/σ_t) * D_t
    
    其中 D_t 是去噪方向的线性组合

UniPC 更新:
    x_{t-1} = x_t + h * Σ(b_i * k_i)
    
    其中 k_i 是多步预测值

=== 参考文献 ===

1. DPM-Solver++:
   Lu et al. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
   https://arxiv.org/abs/2211.01095

2. UniPC:
   Zhao et al. "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models"
   https://arxiv.org/abs/2302.04867

3. Euler Methods:
   Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models"
   https://arxiv.org/abs/2206.00364

=== 核心组件 ===

    - SamplerConfig: 采样器配置基类
    - BaseSampler: 采样器基类
    - DPMPPConfig/DPMPPSampler: DPM++ 采样器
    - UniPCConfig/UniPCSampler: UniPC 采样器
    - EulerConfig/EulerSampler: Euler 采样器
    - EulerAncestralSampler: Euler Ancestral 采样器
    - HeunSampler: Heun 采样器
    - create_sampler: 采样器工厂函数
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Callable, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerType(Enum):
    """采样器类型枚举"""
    DDPM = "ddpm"
    DDIM = "ddim"
    DPM_PP_2M = "dpm++_2m"
    DPM_PP_2S = "dpm++_2s"
    DPM_PP_SDE = "dpm++_sde"
    UNIPC = "unipc"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_a"
    HEUN = "heun"
    LMS = "lms"


class PredictionType(Enum):
    """模型预测类型"""
    EPSILON = "epsilon"      # 预测噪声
    V_PREDICTION = "v"       # 预测 v = α*ε - σ*x
    SAMPLE = "sample"        # 预测原始样本


@dataclass
class SamplerConfig:
    """采样器配置基类"""
    num_train_timesteps: int = 1000
    num_inference_steps: int = 50
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    prediction_type: PredictionType = PredictionType.EPSILON
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    set_alpha_to_one: bool = False
    
    def __post_init__(self):
        if isinstance(self.prediction_type, str):
            self.prediction_type = PredictionType(self.prediction_type)


class NoiseSchedule:
    """噪声调度 - 计算扩散过程中的各种系数"""
    
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        
        # 计算 beta 调度
        if config.beta_schedule == "linear":
            betas = torch.linspace(config.beta_start, config.beta_end, config.num_train_timesteps)
        elif config.beta_schedule == "scaled_linear":
            betas = torch.linspace(
                config.beta_start ** 0.5, 
                config.beta_end ** 0.5, 
                config.num_train_timesteps
            ) ** 2
        elif config.beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(config.num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 设置最终 alpha
        if config.set_alpha_to_one:
            self.final_alpha_cumprod = torch.tensor(1.0)
        else:
            self.final_alpha_cumprod = self.alphas_cumprod[0]
        
        # 预计算常用值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # sigma 和 lambda (用于 DPM++)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.log_sigmas = torch.log(self.sigmas)
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """余弦 beta 调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def get_sigmas(self, timesteps: torch.Tensor) -> torch.Tensor:
        """获取指定时间步的 sigma 值"""
        return self.sigmas[timesteps]
    
    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """将 sigma 转换为时间步"""
        log_sigma = torch.log(sigma)
        dists = log_sigma - self.log_sigmas[:, None]
        low_idx = torch.clamp(
            torch.searchsorted(self.log_sigmas.flip(0), log_sigma).flip(0) - 1,
            0, len(self.log_sigmas) - 2
        )
        high_idx = low_idx + 1
        
        low = self.log_sigmas[low_idx]
        high = self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = torch.clamp(w, 0, 1)
        
        t = (1 - w) * low_idx + w * high_idx
        return t


class BaseSampler(ABC):
    """采样器基类"""
    
    def __init__(self, config: SamplerConfig):
        self.config = config
        self.schedule = NoiseSchedule(config)
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        
    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """设置推理时间步"""
        self.num_inference_steps = num_inference_steps
        
        # 均匀分布的时间步
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0)  # 从大到小
        
        if device is not None:
            timesteps = timesteps.to(device)
        
        self.timesteps = timesteps.long()
        
    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """单步采样"""
        pass
    
    def _get_prev_timestep(self, timestep: int) -> int:
        """获取上一个时间步"""
        if self.timesteps is None:
            raise ValueError("Timesteps not set. Call set_timesteps first.")
        
        index = (self.timesteps == timestep).nonzero()
        if len(index) == 0:
            return 0
        
        index = index[0].item()
        if index + 1 < len(self.timesteps):
            return self.timesteps[index + 1].item()
        return 0
    
    def _convert_model_output(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor
    ) -> torch.Tensor:
        """将模型输出转换为预测的原始样本"""
        alpha_t = self.schedule.alphas_cumprod[timestep]
        sigma_t = self.schedule.sqrt_one_minus_alphas_cumprod[timestep]
        
        if self.config.prediction_type == PredictionType.EPSILON:
            # 从噪声预测转换为样本预测
            pred_x0 = (sample - sigma_t * model_output) / torch.sqrt(alpha_t)
        elif self.config.prediction_type == PredictionType.V_PREDICTION:
            # v = alpha * epsilon - sigma * x
            pred_x0 = torch.sqrt(alpha_t) * sample - sigma_t * model_output
        elif self.config.prediction_type == PredictionType.SAMPLE:
            pred_x0 = model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        # 裁剪样本
        if self.config.clip_sample:
            pred_x0 = torch.clamp(pred_x0, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        return pred_x0
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """向样本添加噪声"""
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        while len(sqrt_alpha.shape) < len(original_samples.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise


# ============================================================================
# DPM++ 采样器
# ============================================================================

@dataclass
class DPMPPConfig(SamplerConfig):
    """DPM++ 采样器配置"""
    solver_order: int = 2           # 求解器阶数 (1, 2, 3)
    solver_type: str = "midpoint"   # "midpoint" 或 "heun"
    lower_order_final: bool = True  # 最后几步使用低阶方法
    use_karras_sigmas: bool = False # 使用 Karras sigma 调度
    algorithm_type: str = "dpmsolver++"  # "dpmsolver" 或 "dpmsolver++"


class DPMPPSampler(BaseSampler):
    """DPM++ 采样器 - 高效的高阶 ODE 求解器"""
    
    def __init__(self, config: DPMPPConfig):
        super().__init__(config)
        self.config: DPMPPConfig = config
        
        # 存储历史模型输出 (用于多步方法)
        self.model_outputs: List[torch.Tensor] = []
        self.sample_history: List[torch.Tensor] = []
        
    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """设置推理时间步"""
        self.num_inference_steps = num_inference_steps
        
        if self.config.use_karras_sigmas:
            sigmas = self._get_karras_sigmas(num_inference_steps)
        else:
            # 均匀分布在 log-sigma 空间
            sigmas = torch.linspace(
                self.schedule.log_sigmas[0],
                self.schedule.log_sigmas[-1],
                num_inference_steps + 1
            ).exp()
        
        # 转换 sigma 到时间步
        timesteps = []
        for sigma in sigmas[:-1]:
            t = self.schedule.sigma_to_t(sigma.unsqueeze(0))
            timesteps.append(t.item())
        
        self.timesteps = torch.tensor(timesteps, dtype=torch.long)
        self.sigmas = sigmas
        
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)
        
        # 重置历史
        self.model_outputs = []
        self.sample_history = []
        
    def _get_karras_sigmas(self, num_steps: int) -> torch.Tensor:
        """Karras et al. 的 sigma 调度"""
        sigma_min = self.schedule.sigmas[-1].item()
        sigma_max = self.schedule.sigmas[0].item()
        
        rho = 7.0  # Karras 论文中的默认值
        ramp = torch.linspace(0, 1, num_steps + 1)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        
        # 从大到小排列
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas.flip(0)
    
    def _sigma_to_alpha_sigma(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将 sigma 转换为 alpha 和 sigma"""
        alpha = 1 / torch.sqrt(1 + sigma ** 2)
        sigma_out = sigma * alpha
        return alpha, sigma_out
    
    def _get_lambda(self, sigma: torch.Tensor) -> torch.Tensor:
        """计算 lambda = log(alpha/sigma)"""
        alpha, sigma_out = self._sigma_to_alpha_sigma(sigma)
        return torch.log(alpha) - torch.log(sigma_out)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """DPM++ 单步采样"""
        if self.timesteps is None or self.sigmas is None:
            raise ValueError("Timesteps not set. Call set_timesteps first.")
        
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index[0].item()
        
        sigma_t = self.sigmas[step_index]
        sigma_s = self.sigmas[step_index + 1]
        
        # 转换模型输出为去噪方向
        alpha_t, sigma_t_out = self._sigma_to_alpha_sigma(sigma_t)
        
        if self.config.prediction_type == PredictionType.EPSILON:
            denoised = (sample - sigma_t_out * model_output) / alpha_t
        elif self.config.prediction_type == PredictionType.V_PREDICTION:
            denoised = alpha_t * sample - sigma_t_out * model_output
        else:
            denoised = model_output
        
        if self.config.clip_sample:
            denoised = torch.clamp(denoised, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        # 存储历史
        self.model_outputs.append(denoised)
        if len(self.model_outputs) > self.config.solver_order:
            self.model_outputs.pop(0)
        
        # 计算 lambda
        lambda_t = self._get_lambda(sigma_t)
        lambda_s = self._get_lambda(sigma_s)
        h = lambda_s - lambda_t
        
        # 根据阶数选择更新方法
        order = min(step_index + 1, self.config.solver_order)
        if self.config.lower_order_final and step_index >= len(self.timesteps) - 2:
            order = 1
        
        if order == 1:
            # 一阶 (Euler)
            x_s = self._dpm_solver_first_order_update(sample, sigma_t, sigma_s, denoised)
        elif order == 2:
            # 二阶
            x_s = self._dpm_solver_second_order_update(
                sample, sigma_t, sigma_s, self.model_outputs[-2], denoised
            )
        else:
            # 三阶
            x_s = self._dpm_solver_third_order_update(
                sample, sigma_t, sigma_s,
                self.model_outputs[-3], self.model_outputs[-2], denoised
            )
        
        return x_s
    
    def _dpm_solver_first_order_update(
        self,
        sample: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_s: torch.Tensor,
        denoised: torch.Tensor
    ) -> torch.Tensor:
        """DPM++ 一阶更新"""
        alpha_t, sigma_t_out = self._sigma_to_alpha_sigma(sigma_t)
        alpha_s, sigma_s_out = self._sigma_to_alpha_sigma(sigma_s)
        
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t_out)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s_out)
        h = lambda_s - lambda_t
        
        if self.config.algorithm_type == "dpmsolver++":
            x_s = (sigma_s_out / sigma_t_out) * sample - alpha_s * (torch.exp(-h) - 1) * denoised
        else:
            x_s = (alpha_s / alpha_t) * sample - sigma_s_out * (torch.exp(h) - 1) * denoised
        
        return x_s
    
    def _dpm_solver_second_order_update(
        self,
        sample: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_s: torch.Tensor,
        denoised_t: torch.Tensor,
        denoised_s: torch.Tensor
    ) -> torch.Tensor:
        """DPM++ 二阶更新 (2M)"""
        alpha_t, sigma_t_out = self._sigma_to_alpha_sigma(sigma_t)
        alpha_s, sigma_s_out = self._sigma_to_alpha_sigma(sigma_s)
        
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t_out)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s_out)
        h = lambda_s - lambda_t
        
        if self.config.algorithm_type == "dpmsolver++":
            # DPM-Solver++ 2M
            r = 0.5
            D0 = denoised_s
            D1 = (denoised_s - denoised_t) / (2 * r)
            
            x_s = (
                (sigma_s_out / sigma_t_out) * sample
                - alpha_s * (torch.exp(-h) - 1) * D0
                - alpha_s * (torch.exp(-h) - 1 + h) * D1
            )
        else:
            # DPM-Solver 2M
            r = 0.5
            D0 = denoised_s
            D1 = (denoised_s - denoised_t) / (2 * r)
            
            x_s = (
                (alpha_s / alpha_t) * sample
                - sigma_s_out * (torch.exp(h) - 1) * D0
                - sigma_s_out * (torch.exp(h) - 1 - h) * D1
            )
        
        return x_s
    
    def _dpm_solver_third_order_update(
        self,
        sample: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_s: torch.Tensor,
        denoised_0: torch.Tensor,
        denoised_1: torch.Tensor,
        denoised_2: torch.Tensor
    ) -> torch.Tensor:
        """DPM++ 三阶更新"""
        alpha_t, sigma_t_out = self._sigma_to_alpha_sigma(sigma_t)
        alpha_s, sigma_s_out = self._sigma_to_alpha_sigma(sigma_s)
        
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t_out)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s_out)
        h = lambda_s - lambda_t
        
        r1, r2 = 1/3, 2/3
        D0 = denoised_2
        D1 = (denoised_2 - denoised_1) / r2
        D2 = ((denoised_2 - denoised_1) / r2 - (denoised_1 - denoised_0) / r1) / (r2 - r1)
        
        if self.config.algorithm_type == "dpmsolver++":
            x_s = (
                (sigma_s_out / sigma_t_out) * sample
                - alpha_s * (torch.exp(-h) - 1) * D0
                - alpha_s * (torch.exp(-h) - 1 + h) * D1
                - alpha_s * (torch.exp(-h) - 1 + h - 0.5 * h ** 2) * D2
            )
        else:
            x_s = (
                (alpha_s / alpha_t) * sample
                - sigma_s_out * (torch.exp(h) - 1) * D0
                - sigma_s_out * (torch.exp(h) - 1 - h) * D1
                - sigma_s_out * (torch.exp(h) - 1 - h - 0.5 * h ** 2) * D2
            )
        
        return x_s


# ============================================================================
# UniPC 采样器
# ============================================================================

@dataclass
class UniPCConfig(SamplerConfig):
    """UniPC 采样器配置"""
    solver_order: int = 2
    predict_x0: bool = True
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    sample_max_value: float = 1.0
    variant: str = "bh1"  # "bh1" 或 "bh2"


class UniPCSampler(BaseSampler):
    """UniPC 采样器 - 统一的预测-校正框架"""
    
    def __init__(self, config: UniPCConfig):
        super().__init__(config)
        self.config: UniPCConfig = config
        self.model_outputs: List[torch.Tensor] = []
        self.timestep_list: List[int] = []
        
    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """设置推理时间步"""
        self.num_inference_steps = num_inference_steps
        
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0)
        
        if device is not None:
            timesteps = timesteps.to(device)
        
        self.timesteps = timesteps.long()
        self.model_outputs = []
        self.timestep_list = []
        
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """动态阈值处理"""
        if not self.config.thresholding:
            return sample
        
        batch_size = sample.shape[0]
        s = torch.quantile(
            sample.reshape(batch_size, -1).abs(),
            self.config.dynamic_thresholding_ratio,
            dim=1
        )
        s = torch.clamp(s, min=1.0, max=self.config.sample_max_value)
        s = s.view(batch_size, *([1] * (sample.ndim - 1)))
        sample = torch.clamp(sample, -s, s) / s
        return sample
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """UniPC 单步采样"""
        if self.timesteps is None:
            raise ValueError("Timesteps not set. Call set_timesteps first.")
        
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index[0].item()
        
        prev_timestep = self._get_prev_timestep(timestep)
        
        # 转换模型输出
        alpha_t = self.schedule.alphas_cumprod[timestep]
        sigma_t = self.schedule.sqrt_one_minus_alphas_cumprod[timestep]
        
        if self.config.prediction_type == PredictionType.EPSILON:
            x0_pred = (sample - sigma_t * model_output) / torch.sqrt(alpha_t)
        elif self.config.prediction_type == PredictionType.V_PREDICTION:
            x0_pred = torch.sqrt(alpha_t) * sample - sigma_t * model_output
        else:
            x0_pred = model_output
        
        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)
        elif self.config.clip_sample:
            x0_pred = torch.clamp(x0_pred, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        # 存储历史
        self.model_outputs.append(x0_pred)
        self.timestep_list.append(timestep)
        
        if len(self.model_outputs) > self.config.solver_order:
            self.model_outputs.pop(0)
            self.timestep_list.pop(0)
        
        # 计算更新
        order = min(len(self.model_outputs), self.config.solver_order)
        
        if order == 1:
            x_prev = self._first_order_update(sample, timestep, prev_timestep, x0_pred)
        else:
            x_prev = self._multistep_update(sample, timestep, prev_timestep)
        
        return x_prev
    
    def _first_order_update(
        self,
        sample: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        x0_pred: torch.Tensor
    ) -> torch.Tensor:
        """一阶更新"""
        alpha_t = self.schedule.alphas_cumprod[timestep]
        alpha_s = self.schedule.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        sigma_t = torch.sqrt(1 - alpha_t)
        sigma_s = torch.sqrt(1 - alpha_s)
        
        lambda_t = torch.log(torch.sqrt(alpha_t)) - torch.log(sigma_t)
        lambda_s = torch.log(torch.sqrt(alpha_s)) - torch.log(sigma_s)
        h = lambda_s - lambda_t
        
        x_s = (sigma_s / sigma_t) * sample - torch.sqrt(alpha_s) * (torch.exp(-h) - 1) * x0_pred
        return x_s
    
    def _multistep_update(
        self,
        sample: torch.Tensor,
        timestep: int,
        prev_timestep: int
    ) -> torch.Tensor:
        """多步更新"""
        order = len(self.model_outputs)
        
        lambdas = []
        for t in self.timestep_list:
            alpha = self.schedule.alphas_cumprod[t]
            sigma = torch.sqrt(1 - alpha)
            lambdas.append(torch.log(torch.sqrt(alpha)) - torch.log(sigma))
        
        alpha_s = self.schedule.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        sigma_s = torch.sqrt(1 - alpha_s)
        lambda_s = torch.log(torch.sqrt(alpha_s)) - torch.log(sigma_s)
        
        alpha_t = self.schedule.alphas_cumprod[timestep]
        sigma_t = torch.sqrt(1 - alpha_t)
        
        h = lambda_s - lambdas[-1]
        
        if self.config.variant == "bh1":
            rks = [(lambda_s - lambdas[i]) / h for i in range(order)]
        else:
            rks = [(lambdas[-1] - lambdas[i]) / h for i in range(order - 1)] + [1.0]
        
        D = self.model_outputs[-1]
        for i in range(1, order):
            D = D + self._get_coefficients(rks, i) * (self.model_outputs[-1] - self.model_outputs[-(i+1)])
        
        x_s = (sigma_s / sigma_t) * sample - torch.sqrt(alpha_s) * (torch.exp(-h) - 1) * D
        return x_s
    
    def _get_coefficients(self, rks: List[float], order: int) -> float:
        """计算多步系数"""
        if order == 1:
            return 0.5
        elif order == 2:
            return 1/6
        return 0.0


# ============================================================================
# Euler 采样器
# ============================================================================

@dataclass
class EulerConfig(SamplerConfig):
    """Euler 采样器配置"""
    use_karras_sigmas: bool = False
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float('inf')
    s_noise: float = 1.0


class EulerSampler(BaseSampler):
    """Euler 采样器 - 简单高效的一阶方法"""
    
    def __init__(self, config: EulerConfig):
        super().__init__(config)
        self.config: EulerConfig = config
        self.sigmas: Optional[torch.Tensor] = None
        
    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """设置推理时间步"""
        self.num_inference_steps = num_inference_steps
        
        if self.config.use_karras_sigmas:
            sigmas = self._get_karras_sigmas(num_inference_steps)
        else:
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = torch.arange(0, num_inference_steps) * step_ratio
            sigmas = self.schedule.sigmas[timesteps.long()]
            sigmas = torch.cat([sigmas, torch.zeros(1)])
        
        self.sigmas = sigmas
        
        timesteps_list = []
        for sigma in sigmas[:-1]:
            t = self.schedule.sigma_to_t(sigma.unsqueeze(0))
            timesteps_list.append(int(t.item()))
        
        self.timesteps = torch.tensor(timesteps_list, dtype=torch.long)
        
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)
    
    def _get_karras_sigmas(self, num_steps: int) -> torch.Tensor:
        """Karras sigma 调度"""
        sigma_min = self.schedule.sigmas[-1].item()
        sigma_max = self.schedule.sigmas[0].item()
        
        rho = 7.0
        ramp = torch.linspace(0, 1, num_steps + 1)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        
        # 从大到小排列
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas.flip(0)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Euler 单步采样"""
        if self.sigmas is None:
            raise ValueError("Timesteps not set. Call set_timesteps first.")
        
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index[0].item()
        
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        
        # 转换模型输出为去噪方向
        if self.config.prediction_type == PredictionType.EPSILON:
            pred_x0 = sample - sigma * model_output
        elif self.config.prediction_type == PredictionType.V_PREDICTION:
            pred_x0 = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))
        else:
            pred_x0 = model_output
        
        # 计算导数
        derivative = (sample - pred_x0) / sigma
        
        # Euler 步进
        dt = sigma_next - sigma
        x_next = sample + derivative * dt
        
        return x_next


class EulerAncestralSampler(EulerSampler):
    """Euler Ancestral 采样器 - 带随机性的 Euler 方法"""
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> torch.Tensor:
        """Euler Ancestral 单步采样"""
        if self.sigmas is None:
            raise ValueError("Timesteps not set. Call set_timesteps first.")
        
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index[0].item()
        
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        
        # 转换模型输出
        if self.config.prediction_type == PredictionType.EPSILON:
            pred_x0 = sample - sigma * model_output
        elif self.config.prediction_type == PredictionType.V_PREDICTION:
            pred_x0 = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))
        else:
            pred_x0 = model_output
        
        # 计算 sigma_up 和 sigma_down (避免负数开方)
        sigma_ratio = sigma_next ** 2 / sigma ** 2
        sigma_up_sq = sigma_next ** 2 * (1 - sigma_ratio)
        sigma_up = torch.sqrt(torch.clamp(sigma_up_sq, min=0))
        sigma_down = torch.sqrt(torch.clamp(sigma_next ** 2 - sigma_up ** 2, min=0))
        
        # 计算导数
        derivative = (sample - pred_x0) / sigma
        
        # Euler 步进
        dt = sigma_down - sigma
        x_next = sample + derivative * dt
        
        # 添加噪声
        if sigma_next > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            x_next = x_next + noise * sigma_up
        
        return x_next


# ============================================================================
# Heun 采样器
# ============================================================================

@dataclass
class HeunConfig(SamplerConfig):
    """Heun 采样器配置"""
    use_karras_sigmas: bool = False


class HeunSampler(BaseSampler):
    """Heun 采样器 - 二阶 Runge-Kutta 方法"""
    
    def __init__(self, config: HeunConfig):
        super().__init__(config)
        self.config: HeunConfig = config
        self.sigmas: Optional[torch.Tensor] = None
        self._prev_derivative: Optional[torch.Tensor] = None
        self._dt: Optional[torch.Tensor] = None
        self._sample: Optional[torch.Tensor] = None
        
    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """设置推理时间步"""
        self.num_inference_steps = num_inference_steps
        
        if self.config.use_karras_sigmas:
            sigmas = self._get_karras_sigmas(num_inference_steps)
        else:
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = torch.arange(0, num_inference_steps) * step_ratio
            sigmas = self.schedule.sigmas[timesteps.long()]
            sigmas = torch.cat([sigmas, torch.zeros(1)])
        
        self.sigmas = sigmas
        
        timesteps_list = []
        for sigma in sigmas[:-1]:
            t = self.schedule.sigma_to_t(sigma.unsqueeze(0))
            timesteps_list.append(int(t.item()))
        
        self.timesteps = torch.tensor(timesteps_list, dtype=torch.long)
        
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)
        
        self._prev_derivative = None
        self._dt = None
        self._sample = None
    
    def _get_karras_sigmas(self, num_steps: int) -> torch.Tensor:
        """Karras sigma 调度"""
        sigma_min = self.schedule.sigmas[-1].item()
        sigma_max = self.schedule.sigmas[0].item()
        
        rho = 7.0
        ramp = torch.linspace(0, 1, num_steps + 1)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        
        # 从大到小排列
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas.flip(0)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Heun 单步采样"""
        if self.sigmas is None:
            raise ValueError("Timesteps not set. Call set_timesteps first.")
        
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index[0].item()
        
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        
        # 转换模型输出
        if self.config.prediction_type == PredictionType.EPSILON:
            pred_x0 = sample - sigma * model_output
        elif self.config.prediction_type == PredictionType.V_PREDICTION:
            pred_x0 = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5) + (sample / (sigma ** 2 + 1))
        else:
            pred_x0 = model_output
        
        derivative = (sample - pred_x0) / sigma
        dt = sigma_next - sigma
        
        if self._prev_derivative is None:
            # 第一次调用: Euler 预测
            x_next = sample + derivative * dt
            self._prev_derivative = derivative
            self._dt = dt
            self._sample = sample
        else:
            # 第二次调用: Heun 校正
            derivative_avg = (self._prev_derivative + derivative) / 2
            x_next = self._sample + derivative_avg * self._dt
            self._prev_derivative = None
            self._dt = None
            self._sample = None
        
        return x_next


# ============================================================================
# LMS 采样器 (Linear Multi-Step)
# ============================================================================

@dataclass
class LMSConfig(SamplerConfig):
    """LMS 采样器配置"""
    order: int = 4


class LMSSampler(BaseSampler):
    """LMS 采样器 - 线性多步方法"""
    
    def __init__(self, config: LMSConfig):
        super().__init__(config)
        self.config: LMSConfig = config
        self.derivatives: List[torch.Tensor] = []
        self.sigmas: Optional[torch.Tensor] = None
        
    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """设置推理时间步"""
        self.num_inference_steps = num_inference_steps
        
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        sigmas = self.schedule.sigmas[timesteps.long()]
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        
        self.sigmas = sigmas
        
        timesteps_list = []
        for sigma in sigmas[:-1]:
            t = self.schedule.sigma_to_t(sigma.unsqueeze(0))
            timesteps_list.append(int(t.item()))
        
        self.timesteps = torch.tensor(timesteps_list, dtype=torch.long)
        
        if device is not None:
            self.timesteps = self.timesteps.to(device)
            self.sigmas = self.sigmas.to(device)
        
        self.derivatives = []
    
    def _get_lms_coefficients(self, order: int, t: float, current_order: int) -> List[float]:
        """计算 LMS 系数 (Lagrange 插值)"""
        coeffs = []
        for i in range(current_order):
            coeff = 1.0
            for j in range(current_order):
                if i != j:
                    coeff *= (t - j) / (i - j)
            coeffs.append(coeff)
        return coeffs
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """LMS 单步采样"""
        if self.sigmas is None:
            raise ValueError("Timesteps not set. Call set_timesteps first.")
        
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = 0
        else:
            step_index = step_index[0].item()
        
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        
        # 转换模型输出
        if self.config.prediction_type == PredictionType.EPSILON:
            pred_x0 = sample - sigma * model_output
        else:
            pred_x0 = model_output
        
        derivative = (sample - pred_x0) / sigma
        
        # 存储导数历史
        self.derivatives.append(derivative)
        if len(self.derivatives) > self.config.order:
            self.derivatives.pop(0)
        
        # 计算 LMS 更新
        order = min(len(self.derivatives), self.config.order)
        coeffs = self._get_lms_coefficients(order, order - 1, order)
        
        # 加权求和
        derivative_sum = sum(c * d for c, d in zip(coeffs, self.derivatives))
        
        dt = sigma_next - sigma
        x_next = sample + derivative_sum * dt
        
        return x_next


# ============================================================================
# 工厂函数
# ============================================================================

def create_sampler(
    sampler_type: Union[str, SamplerType],
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 50,
    **kwargs
) -> BaseSampler:
    """
    创建采样器
    
    Args:
        sampler_type: 采样器类型
        num_train_timesteps: 训练时间步数
        num_inference_steps: 推理时间步数
        **kwargs: 额外配置参数
    
    Returns:
        采样器实例
    """
    if isinstance(sampler_type, str):
        sampler_type = SamplerType(sampler_type)
    
    base_kwargs = {
        "num_train_timesteps": num_train_timesteps,
        "num_inference_steps": num_inference_steps,
        **kwargs
    }
    
    if sampler_type == SamplerType.DPM_PP_2M:
        config = DPMPPConfig(solver_order=2, **base_kwargs)
        return DPMPPSampler(config)
    
    elif sampler_type == SamplerType.DPM_PP_2S:
        config = DPMPPConfig(solver_order=2, solver_type="heun", **base_kwargs)
        return DPMPPSampler(config)
    
    elif sampler_type == SamplerType.UNIPC:
        config = UniPCConfig(**base_kwargs)
        return UniPCSampler(config)
    
    elif sampler_type == SamplerType.EULER:
        config = EulerConfig(**base_kwargs)
        return EulerSampler(config)
    
    elif sampler_type == SamplerType.EULER_ANCESTRAL:
        config = EulerConfig(**base_kwargs)
        return EulerAncestralSampler(config)
    
    elif sampler_type == SamplerType.HEUN:
        config = HeunConfig(**base_kwargs)
        return HeunSampler(config)
    
    elif sampler_type == SamplerType.LMS:
        config = LMSConfig(**base_kwargs)
        return LMSSampler(config)
    
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


# 导出
__all__ = [
    # 枚举
    "SamplerType",
    "PredictionType",
    # 配置
    "SamplerConfig",
    "DPMPPConfig",
    "UniPCConfig", 
    "EulerConfig",
    "HeunConfig",
    "LMSConfig",
    # 采样器
    "BaseSampler",
    "NoiseSchedule",
    "DPMPPSampler",
    "UniPCSampler",
    "EulerSampler",
    "EulerAncestralSampler",
    "HeunSampler",
    "LMSSampler",
    # 工厂函数
    "create_sampler",
]
