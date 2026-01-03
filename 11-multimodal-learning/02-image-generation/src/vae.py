"""
变分自编码器 (Variational Autoencoder, VAE) 实现

VAE 是一种生成模型，通过学习数据的潜在表示来生成新样本。
核心思想是将编码器输出建模为概率分布，并通过重参数化技巧实现端到端训练。

=== 核心思想 ===

VAE 的核心创新：

1. 概率潜在空间
   - 编码器输出分布参数 (μ, σ) 而非确定性向量
   - 潜在空间具有连续性和平滑性
   - 支持有意义的插值和采样

2. 重参数化技巧 (Reparameterization Trick)
   - z = μ + σ * ε,  ε ~ N(0, I)
   - 将随机性从计算图中分离
   - 使得梯度可以通过采样操作反向传播

3. 变分推断
   - 用变分分布 q(z|x) 近似真实后验 p(z|x)
   - 最大化证据下界 (ELBO)

=== 数学基础 ===

证据下界 (ELBO):
    log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
             = -L_recon - L_KL

重建损失:
    L_recon = ||x - x̂||² 或 BCE(x, x̂)

KL 散度 (高斯分布):
    KL(q||p) = -1/2 * Σ(1 + log(σ²) - μ² - σ²)

总损失:
    L = L_recon + β * L_KL
    
    其中 β 控制重建质量与潜在空间正则化的平衡

=== 算法流程 ===

训练阶段:
    输入: 图像 x
      ↓
    编码: μ, log_var = Encoder(x)
      ↓
    重参数化: z = μ + exp(0.5*log_var) * ε
      ↓
    解码: x̂ = Decoder(z)
      ↓
    计算损失: L = L_recon + β * L_KL
      ↓
    反向传播更新参数

生成阶段:
    采样: z ~ N(0, I)
      ↓
    解码: x = Decoder(z)
      ↓
    输出: 生成图像 x

=== 参考文献 ===

1. VAE 原始论文:
   Kingma & Welling "Auto-Encoding Variational Bayes" ICLR 2014
   https://arxiv.org/abs/1312.6114

2. β-VAE:
   Higgins et al. "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" 2017

3. VQ-VAE:
   van den Oord et al. "Neural Discrete Representation Learning" NeurIPS 2017

=== 核心组件 ===

    - VAEConfig: VAE 模型配置
    - ResidualBlock: 残差块
    - AttentionBlock: 自注意力块
    - Encoder: 编码器 (图像 → 潜在分布)
    - Decoder: 解码器 (潜在向量 → 图像)
    - VAE: 完整的变分自编码器
    - vae_loss: ELBO 损失函数
    - create_vae_model: 创建预定义模型
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    """VAE 模型配置"""

    # 输入配置
    image_size: int = 256
    in_channels: int = 3

    # 潜在空间配置
    latent_dim: int = 256
    latent_channels: int = 4  # 潜在空间通道数

    # 编码器配置
    encoder_channels: Tuple[int, ...] = (64, 128, 256, 512)

    # 解码器配置
    decoder_channels: Tuple[int, ...] = (512, 256, 128, 64)

    # 训练配置
    kl_weight: float = 1.0  # β-VAE 的 β 参数

    def __post_init__(self):
        # 验证图像尺寸可以被下采样
        min_size = 2 ** len(self.encoder_channels)
        assert self.image_size >= min_size, \
            f"image_size ({self.image_size}) must be >= {min_size}"


class ResidualBlock(nn.Module):
    """残差块 - 用于编码器和解码器"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
        upsample: bool = False
    ):
        super().__init__()

        self.downsample = downsample
        self.upsample = upsample

        # 主路径
        if upsample:
            self.conv1 = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            )
        elif downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1
            )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接
        if in_channels != out_channels or downsample or upsample:
            if upsample:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels, out_channels,
                        kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(out_channels)
                )
            elif downsample:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = F.silu(out)

        return out


class AttentionBlock(nn.Module):
    """自注意力块 - 用于捕获全局依赖"""

    def __init__(self, channels: int, num_heads: int = 8):
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

        # 归一化
        h = self.norm(x)

        # 重塑为序列
        h = h.view(batch_size, channels, -1).transpose(1, 2)  # [B, H*W, C]

        # 自注意力
        h, _ = self.attention(h, h, h)

        # 重塑回图像
        h = h.transpose(1, 2).view(batch_size, channels, height, width)
        h = self.proj(h)

        return x + h


class Encoder(nn.Module):
    """VAE 编码器 - 将图像编码为潜在分布参数"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # 初始卷积
        self.conv_in = nn.Conv2d(
            config.in_channels, config.encoder_channels[0],
            kernel_size=3, stride=1, padding=1
        )

        # 下采样块
        self.down_blocks = nn.ModuleList()
        in_ch = config.encoder_channels[0]

        for i, out_ch in enumerate(config.encoder_channels):
            # 残差块
            self.down_blocks.append(ResidualBlock(in_ch, out_ch, downsample=(i > 0)))
            self.down_blocks.append(ResidualBlock(out_ch, out_ch))

            # 在较深层添加注意力
            if i >= len(config.encoder_channels) - 2:
                self.down_blocks.append(AttentionBlock(out_ch))

            in_ch = out_ch

        # 最终层
        final_ch = config.encoder_channels[-1]
        self.conv_out = nn.Conv2d(final_ch, final_ch, kernel_size=3, padding=1)
        self.norm_out = nn.GroupNorm(32, final_ch)

        # 计算潜在空间的空间尺寸
        self.latent_h = config.image_size // (2 ** (len(config.encoder_channels) - 1))
        self.latent_w = self.latent_h

        # 输出均值和对数方差
        self.conv_mu = nn.Conv2d(final_ch, config.latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(final_ch, config.latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码图像为潜在分布参数

        Args:
            x: 输入图像 [batch_size, in_channels, H, W]

        Returns:
            mu: 均值 [batch_size, latent_channels, h, w]
            logvar: 对数方差 [batch_size, latent_channels, h, w]
        """
        h = self.conv_in(x)

        for block in self.down_blocks:
            h = block(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """VAE 解码器 - 从潜在向量重建图像"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # 初始卷积
        self.conv_in = nn.Conv2d(
            config.latent_channels, config.decoder_channels[0],
            kernel_size=3, stride=1, padding=1
        )

        # 上采样块
        self.up_blocks = nn.ModuleList()
        in_ch = config.decoder_channels[0]

        for i, out_ch in enumerate(config.decoder_channels):
            # 在较浅层添加注意力
            if i < 2:
                self.up_blocks.append(AttentionBlock(in_ch))

            # 残差块
            self.up_blocks.append(ResidualBlock(in_ch, out_ch))
            self.up_blocks.append(ResidualBlock(out_ch, out_ch, upsample=(i < len(config.decoder_channels) - 1)))

            in_ch = out_ch

        # 最终层
        final_ch = config.decoder_channels[-1]
        self.norm_out = nn.GroupNorm(32, final_ch)
        self.conv_out = nn.Conv2d(final_ch, config.in_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在向量解码图像

        Args:
            z: 潜在向量 [batch_size, latent_channels, h, w]

        Returns:
            重建图像 [batch_size, in_channels, H, W]
        """
        h = self.conv_in(z)

        for block in self.up_blocks:
            h = block(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class VAE(nn.Module):
    """变分自编码器 (Variational Autoencoder)"""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧 - 使采样过程可微分

        z = μ + σ * ε, 其中 ε ~ N(0, I)

        Args:
            mu: 均值 [batch_size, latent_channels, h, w]
            logvar: 对数方差 [batch_size, latent_channels, h, w]

        Returns:
            z: 采样的潜在向量 [batch_size, latent_channels, h, w]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码图像为潜在分布参数"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜在向量解码图像"""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [batch_size, in_channels, H, W]
            return_latent: 是否返回潜在向量

        Returns:
            recon: 重建图像 [batch_size, in_channels, H, W]
            mu: 均值
            logvar: 对数方差
            z: (可选) 潜在向量
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        if return_latent:
            return recon, mu, logvar, z
        return recon, mu, logvar

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 1,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        从先验分布采样生成图像

        Args:
            num_samples: 采样数量
            device: 设备

        Returns:
            生成的图像 [num_samples, in_channels, H, W]
        """
        if device is None:
            device = next(self.parameters()).device

        # 计算潜在空间尺寸
        latent_h = self.config.image_size // (2 ** (len(self.config.encoder_channels) - 1))
        latent_w = latent_h

        # 从标准正态分布采样
        z = torch.randn(
            num_samples,
            self.config.latent_channels,
            latent_h,
            latent_w,
            device=device
        )

        return self.decode(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """重建输入图像"""
        recon, _, _ = self.forward(x)
        return recon

    @torch.no_grad()
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        在两个图像的潜在表示之间进行插值

        Args:
            x1: 第一个图像 [1, C, H, W]
            x2: 第二个图像 [1, C, H, W]
            num_steps: 插值步数

        Returns:
            插值结果 [num_steps, C, H, W]
        """
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)

        # 线性插值
        alphas = torch.linspace(0, 1, num_steps, device=x1.device)
        interpolated = []

        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            img = self.decode(z)
            interpolated.append(img)

        return torch.cat(interpolated, dim=0)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE 损失函数 (ELBO)

    L = 重建损失 + β * KL散度

    Args:
        recon: 重建图像
        target: 目标图像
        mu: 潜在分布均值
        logvar: 潜在分布对数方差
        kl_weight: KL散度权重 (β-VAE)
        reduction: 归约方式

    Returns:
        total_loss: 总损失
        recon_loss: 重建损失
        kl_loss: KL散度
    """
    # 重建损失 (MSE)
    recon_loss = F.mse_loss(recon, target, reduction=reduction)

    # KL 散度: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if reduction == "mean":
        kl_loss = kl_loss / mu.numel()

    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def create_vae_model(model_size: str = "base") -> VAE:
    """
    创建预定义大小的 VAE 模型

    Args:
        model_size: 模型大小 ("small", "base", "large")

    Returns:
        VAE 模型实例
    """
    configs = {
        "small": VAEConfig(
            image_size=64,
            latent_channels=4,
            encoder_channels=(32, 64, 128, 256),
            decoder_channels=(256, 128, 64, 32),
            kl_weight=0.001
        ),
        "base": VAEConfig(
            image_size=256,
            latent_channels=4,
            encoder_channels=(64, 128, 256, 512),
            decoder_channels=(512, 256, 128, 64),
            kl_weight=0.0001
        ),
        "large": VAEConfig(
            image_size=512,
            latent_channels=8,
            encoder_channels=(64, 128, 256, 512, 512),
            decoder_channels=(512, 512, 256, 128, 64),
            kl_weight=0.00001
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return VAE(configs[model_size])
