"""
扩散模型 (Diffusion Models) 实现

扩散模型通过逐步添加噪声（前向过程）和学习去噪（反向过程）来生成高质量图像。

=== 核心思想 ===

扩散模型包含两个过程：

1. 前向过程 (Forward Process / Diffusion)
   - 逐步向数据添加高斯噪声
   - 经过 T 步后，数据变成纯噪声
   - 这是一个固定的马尔可夫链

2. 反向过程 (Reverse Process / Denoising)
   - 学习从噪声逐步恢复数据
   - 使用神经网络预测每一步的噪声
   - 生成时从纯噪声开始逐步去噪

=== 数学基础 ===

前向过程 (添加噪声):
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
    
    直接采样:
    x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε,  ε ~ N(0, I)
    
    其中:
    - β_t: 噪声调度 (noise schedule)
    - α_t = 1 - β_t
    - ᾱ_t = ∏_{s=1}^t α_s (累积乘积)

反向过程 (去噪):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)
    
    μ_θ(x_t, t) = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t))

训练目标 (简化):
    L = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||²]

=== 算法流程 ===

训练阶段:
    输入: 干净图像 x_0
      ↓
    采样时间步: t ~ Uniform(1, T)
    采样噪声: ε ~ N(0, I)
      ↓
    加噪: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
      ↓
    预测噪声: ε_θ = UNet(x_t, t)
      ↓
    计算损失: L = ||ε - ε_θ||²

采样阶段 (DDPM):
    输入: 纯噪声 x_T ~ N(0, I)
      ↓
    for t = T, T-1, ..., 1:
        预测噪声: ε_θ = UNet(x_t, t)
        计算均值: μ = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ)
        采样: x_{t-1} = μ + σ_t z,  z ~ N(0, I)
      ↓
    输出: 生成图像 x_0

采样阶段 (DDIM - 加速):
    - 使用子序列 τ ⊂ {1,...,T}
    - 确定性采样，无需添加噪声
    - 可将 1000 步减少到 50 步

=== 参考文献 ===

1. DDPM:
   Ho et al. "Denoising Diffusion Probabilistic Models" NeurIPS 2020
   https://arxiv.org/abs/2006.11239

2. DDIM:
   Song et al. "Denoising Diffusion Implicit Models" ICLR 2021
   https://arxiv.org/abs/2010.02502

3. Improved DDPM:
   Nichol & Dhariwal "Improved Denoising Diffusion Probabilistic Models" 2021

=== 核心组件 ===

    - DiffusionConfig: 扩散模型配置
    - SinusoidalPositionEmbedding: 正弦位置编码
    - TimeEmbedding: 时间步嵌入
    - ResBlock: 残差块 (带时间条件)
    - AttentionBlock: 自注意力块
    - DownBlock/UpBlock: 下采样/上采样块
    - UNet: 去噪网络
    - NoiseScheduler: 噪声调度器
    - DDPM: 去噪扩散概率模型
    - DDIMSampler: DDIM 加速采样器
    - create_diffusion_model: 创建预定义模型
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    """扩散模型配置"""

    # 图像配置
    image_size: int = 64
    in_channels: int = 3

    # UNet 配置
    model_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    num_heads: int = 4
    dropout: float = 0.0

    # 扩散配置
    num_timesteps: int = 1000
    beta_schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # 条件配置
    num_classes: Optional[int] = None  # 类别条件生成


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码 - 用于时间步嵌入"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: 时间步 [batch_size]
        Returns:
            嵌入向量 [batch_size, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """时间步嵌入模块"""

    def __init__(self, model_channels: int, time_embed_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(model_channels)
        self.linear1 = nn.Linear(model_channels, time_embed_dim)
        self.linear2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        x = self.sinusoidal(timesteps)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x


class ResBlock(nn.Module):
    """残差块 - 带时间嵌入条件"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        up: bool = False,
        down: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down

        # 第一个卷积块
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 时间嵌入投影
        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        # 第二个卷积块
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 跳跃连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # 上/下采样
        if up:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        elif down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # 上/下采样
        if self.up:
            x = self.upsample(x)
        elif self.down:
            x = self.downsample(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # 添加时间嵌入
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # 跳跃连接
        if self.up:
            x = self.upsample(x)
        elif self.down:
            x = self.downsample(x)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """自注意力块"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        h = self.norm(x)
        h = h.view(batch_size, channels, -1).transpose(1, 2)  # [B, H*W, C]
        h, _ = self.attention(h, h, h)
        h = h.transpose(1, 2).view(batch_size, channels, height, width)
        h = self.proj(h)

        return x + h


class DownBlock(nn.Module):
    """下采样块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 2,
        has_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        downsample: bool = True
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    time_embed_dim,
                    dropout
                )
            )
            if has_attention:
                self.attn_blocks.append(AttentionBlock(out_channels, num_heads))
            else:
                self.attn_blocks.append(nn.Identity())

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, stride=2, padding=1
            )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []

        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, time_emb)
            x = attn_block(x)
            skips.append(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x, skips


class UpBlock(nn.Module):
    """上采样块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 2,
        has_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0,
        upsample: bool = True
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        for i in range(num_res_blocks):
            res_in_channels = (in_channels if i == 0 else out_channels) + skip_channels
            self.res_blocks.append(
                ResBlock(res_in_channels, out_channels, time_embed_dim, dropout)
            )
            if has_attention:
                self.attn_blocks.append(AttentionBlock(out_channels, num_heads))
            else:
                self.attn_blocks.append(nn.Identity())

        self.upsample = None
        if upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        skips: List[torch.Tensor]
    ) -> torch.Tensor:
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = res_block(x, time_emb)
            x = attn_block(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class UNet(nn.Module):
    """UNet 去噪网络"""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        time_embed_dim = config.model_channels * 4

        # 时间嵌入
        self.time_embed = TimeEmbedding(config.model_channels, time_embed_dim)

        # 类别嵌入 (可选)
        if config.num_classes is not None:
            self.class_embed = nn.Embedding(config.num_classes, time_embed_dim)
        else:
            self.class_embed = None

        # 输入卷积
        self.conv_in = nn.Conv2d(
            config.in_channels, config.model_channels,
            kernel_size=3, padding=1
        )

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        channels = [config.model_channels]
        ch = config.model_channels
        resolution = config.image_size

        for i, mult in enumerate(config.channel_mult):
            out_ch = config.model_channels * mult
            has_attn = resolution in config.attention_resolutions

            self.down_blocks.append(
                DownBlock(
                    ch, out_ch, time_embed_dim,
                    num_res_blocks=config.num_res_blocks,
                    has_attention=has_attn,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    downsample=(i < len(config.channel_mult) - 1)
                )
            )

            ch = out_ch
            channels.append(ch)
            if i < len(config.channel_mult) - 1:
                resolution //= 2

        # 中间块
        self.mid_block1 = ResBlock(ch, ch, time_embed_dim, config.dropout)
        self.mid_attn = AttentionBlock(ch, config.num_heads)
        self.mid_block2 = ResBlock(ch, ch, time_embed_dim, config.dropout)

        # 上采样路径
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(config.channel_mult)):
            out_ch = config.model_channels * mult
            skip_ch = channels.pop()
            has_attn = resolution in config.attention_resolutions

            self.up_blocks.append(
                UpBlock(
                    ch, out_ch, skip_ch, time_embed_dim,
                    num_res_blocks=config.num_res_blocks,
                    has_attention=has_attn,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    upsample=(i < len(config.channel_mult) - 1)
                )
            )

            ch = out_ch
            if i < len(config.channel_mult) - 1:
                resolution *= 2

        # 输出层
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, config.in_channels, kernel_size=3, padding=1)

        # 初始化输出层为零
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 噪声图像 [batch_size, in_channels, H, W]
            timesteps: 时间步 [batch_size]
            class_labels: 类别标签 [batch_size] (可选)

        Returns:
            预测的噪声 [batch_size, in_channels, H, W]
        """
        # 时间嵌入
        t_emb = self.time_embed(timesteps.float())

        # 添加类别嵌入
        if self.class_embed is not None and class_labels is not None:
            t_emb = t_emb + self.class_embed(class_labels)

        # 输入卷积
        h = self.conv_in(x)

        # 下采样
        all_skips = []
        for down_block in self.down_blocks:
            h, skips = down_block(h, t_emb)
            all_skips.extend(skips)

        # 中间块
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # 上采样
        for up_block in self.up_blocks:
            # 获取对应数量的 skip connections
            num_skips = len(up_block.res_blocks)
            skips = [all_skips.pop() for _ in range(num_skips)]
            h = up_block(h, t_emb, skips)

        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class NoiseScheduler:
    """噪声调度器 - 管理扩散过程中的噪声水平"""

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps

        # 计算 beta 调度
        if config.beta_schedule == "linear":
            self.betas = torch.linspace(
                config.beta_start, config.beta_end, config.num_timesteps
            )
        elif config.beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(config.num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")

        # 预计算扩散参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 用于采样的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # 后验分布参数
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """余弦 beta 调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """从预计算的张量中提取对应时间步的值"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散过程 - 给图像添加噪声

        q(x_t | x_0) = N(x_t; sqrt(α̅_t) * x_0, (1 - α̅_t) * I)

        Args:
            x_start: 原始图像 [batch_size, C, H, W]
            t: 时间步 [batch_size]
            noise: 噪声 (可选)

        Returns:
            噪声图像 x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """从噪声预测原始图像 x_0"""
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape
        )
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算后验分布 q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1_t = self._extract(
            self.posterior_mean_coef1.to(x_t.device), t, x_t.shape
        )
        posterior_mean_coef2_t = self._extract(
            self.posterior_mean_coef2.to(x_t.device), t, x_t.shape
        )
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t

        posterior_log_variance_t = self._extract(
            self.posterior_log_variance_clipped.to(x_t.device), t, x_t.shape
        )

        return posterior_mean, posterior_log_variance_t


class DDPM(nn.Module):
    """去噪扩散概率模型 (Denoising Diffusion Probabilistic Model)"""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        self.unet = UNet(config)
        self.scheduler = NoiseScheduler(config)

    def forward(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        训练前向传播 - 计算去噪损失

        Args:
            x: 原始图像 [batch_size, C, H, W]
            class_labels: 类别标签 (可选)

        Returns:
            损失值
        """
        batch_size = x.shape[0]
        device = x.device

        # 随机采样时间步
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=device)

        # 采样噪声
        noise = torch.randn_like(x)

        # 前向扩散
        x_t = self.scheduler.q_sample(x, t, noise)

        # 预测噪声
        noise_pred = self.unet(x_t, t, class_labels)

        # 计算损失 (简单 MSE)
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def training_step(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """训练步骤"""
        return self.forward(x, class_labels)

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        单步反向采样 p(x_{t-1} | x_t)
        """
        # 预测噪声
        noise_pred = self.unet(x_t, t, class_labels)

        # 预测 x_0
        x_start = self.scheduler.predict_start_from_noise(x_t, t, noise_pred)
        x_start = torch.clamp(x_start, -1, 1)

        # 计算后验均值和方差
        posterior_mean, posterior_log_variance = self.scheduler.q_posterior(x_start, x_t, t)

        # 采样
        noise = torch.randn_like(x_t)
        # t=0 时不添加噪声
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        x_prev = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        DDPM 采样 - 从噪声生成图像

        Args:
            batch_size: 批次大小
            class_labels: 类别标签 (可选)
            device: 设备
            return_intermediates: 是否返回中间结果

        Returns:
            生成的图像 [batch_size, C, H, W]
        """
        if device is None:
            device = next(self.parameters()).device

        # 从纯噪声开始
        x = torch.randn(
            batch_size,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
            device=device
        )

        intermediates = [x] if return_intermediates else None

        # 逐步去噪
        for t in reversed(range(self.config.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, class_labels)

            if return_intermediates and t % 100 == 0:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x


class DDIMSampler:
    """DDIM 加速采样器"""

    def __init__(self, model: DDPM, num_inference_steps: int = 50):
        self.model = model
        self.scheduler = model.scheduler
        self.num_inference_steps = num_inference_steps

        # 计算采样时间步
        self.timesteps = self._get_timesteps(num_inference_steps)

    def _get_timesteps(self, num_steps: int) -> torch.Tensor:
        """获取均匀分布的时间步"""
        step_ratio = self.scheduler.num_timesteps // num_steps
        timesteps = torch.arange(0, num_steps) * step_ratio
        timesteps = timesteps.flip(0)  # 从大到小
        return timesteps.long()

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        noise_pred: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM 单步采样

        Args:
            x_t: 当前噪声图像
            t: 当前时间步
            t_prev: 上一个时间步
            noise_pred: 预测的噪声
            eta: 随机性参数 (0=确定性, 1=DDPM)

        Returns:
            x_{t-1}
        """
        device = x_t.device

        # 获取 alpha 值
        alpha_t = self.scheduler.alphas_cumprod[t].to(device)
        alpha_t_prev = self.scheduler.alphas_cumprod[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0, device=device)

        # 预测 x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        # 计算方差
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)

        # 计算 x_{t-1}
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * noise_pred
        x_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt

        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        eta: float = 0.0,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        DDIM 采样

        Args:
            batch_size: 批次大小
            class_labels: 类别标签
            device: 设备
            eta: 随机性参数
            return_intermediates: 是否返回中间结果

        Returns:
            生成的图像
        """
        if device is None:
            device = next(self.model.parameters()).device

        # 从纯噪声开始
        x = torch.randn(
            batch_size,
            self.model.config.in_channels,
            self.model.config.image_size,
            self.model.config.image_size,
            device=device
        )

        intermediates = [x] if return_intermediates else None
        timesteps = self.timesteps.to(device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # 预测噪声
            noise_pred = self.model.unet(x, t_batch, class_labels)

            # 获取上一个时间步
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else -1

            # DDIM 步骤
            x = self.step(x, t.item(), t_prev.item() if isinstance(t_prev, torch.Tensor) else t_prev, noise_pred, eta)

            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x


def create_diffusion_model(model_size: str = "base") -> DDPM:
    """
    创建预定义大小的扩散模型

    Args:
        model_size: 模型大小 ("small", "base", "large")

    Returns:
        DDPM 模型实例
    """
    configs = {
        "small": DiffusionConfig(
            image_size=32,
            model_channels=64,
            channel_mult=(1, 2, 2),
            num_res_blocks=1,
            attention_resolutions=(8,),
            num_heads=4,
            num_timesteps=1000
        ),
        "base": DiffusionConfig(
            image_size=64,
            model_channels=128,
            channel_mult=(1, 2, 2, 4),
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            num_heads=4,
            num_timesteps=1000
        ),
        "large": DiffusionConfig(
            image_size=128,
            model_channels=192,
            channel_mult=(1, 2, 3, 4),
            num_res_blocks=2,
            attention_resolutions=(32, 16, 8),
            num_heads=8,
            num_timesteps=1000
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return DDPM(configs[model_size])
