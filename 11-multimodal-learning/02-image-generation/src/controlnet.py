"""
ControlNet 实现

ControlNet 为 Stable Diffusion 添加精确的条件控制，
通过复制 UNet 编码器并添加零卷积层来实现可控生成。

=== 核心思想 ===

ControlNet 的关键创新：

1. 零卷积 (Zero Convolution)
   - 权重和偏置初始化为零
   - 训练初期不影响原模型输出
   - 逐渐学习条件控制信号

2. 锁定复制 (Locked Copy)
   - 复制 SD UNet 编码器作为可训练副本
   - 原始 SD 参数保持冻结
   - 仅训练 ControlNet 分支

3. 多种控制类型
   - Canny Edge: 边缘检测控制
   - Pose: 人体姿态控制
   - Depth: 深度图控制
   - Segmentation: 语义分割控制

=== 数学基础 ===

零卷积初始化:
    W = 0, b = 0
    y = ZeroConv(x) = 0 (初始时)

控制信号注入:
    y = F(x) + ZeroConv(ControlNet(c))
    
    其中:
    - F(x): 原始 SD UNet 特征
    - c: 条件图像 (边缘/姿态/深度)
    - ZeroConv: 零初始化卷积

=== 参考文献 ===

1. ControlNet:
   Zhang et al. "Adding Conditional Control to Text-to-Image Diffusion Models" ICCV 2023
   https://arxiv.org/abs/2302.05543

=== 核心组件 ===

    - ControlNetConfig: ControlNet 配置
    - ZeroConv: 零初始化卷积层
    - ControlNetConditioningEmbedding: 条件图像编码
    - ControlNetBlock: ControlNet 块
    - ControlNet: 完整的控制网络
    - create_controlnet: 创建预定义 ControlNet
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_diffusion import (
    SDConfig,
    SDResBlock,
    SpatialTransformer,
    Downsample,
    SinusoidalPositionEmbedding
)


@dataclass
class ControlNetConfig:
    """ControlNet 配置"""

    # 基础配置 (与 SD 对齐)
    image_size: int = 512
    latent_channels: int = 4
    latent_scale_factor: int = 8

    # UNet 配置
    model_channels: int = 320
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 2, 1)
    num_heads: int = 8
    transformer_depth: int = 1
    dropout: float = 0.0

    # 条件配置
    context_dim: int = 768
    conditioning_channels: int = 3  # 条件图像通道数
    conditioning_embedding_channels: int = 256

    # 扩散配置
    num_timesteps: int = 1000


class ZeroConv(nn.Module):
    """零初始化卷积层 - ControlNet 的关键组件"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ControlNetConditioningEmbedding(nn.Module):
    """条件图像编码器 - 将控制图像编码为特征"""

    def __init__(self, config: ControlNetConfig):
        super().__init__()

        channels = config.conditioning_embedding_channels
        self.conv_in = nn.Conv2d(
            config.conditioning_channels, channels // 4,
            kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([
            nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(channels // 4, channels // 2, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, config.model_channels, kernel_size=3, padding=1, stride=2),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = F.silu(x)

        for block in self.blocks:
            x = block(x)
            x = F.silu(x)

        return x


class ControlNetBlock(nn.Module):
    """ControlNet 块 - 残差块 + 可选的空间 Transformer"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        config: ControlNetConfig,
        has_attention: bool = False
    ):
        super().__init__()

        self.res_block = SDResBlock(in_channels, out_channels, time_embed_dim, config.dropout)

        if has_attention:
            self.attn = SpatialTransformer(
                out_channels,
                config.num_heads,
                out_channels // config.num_heads,
                config.transformer_depth,
                config.context_dim,
                config.dropout
            )
        else:
            self.attn = None

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.res_block(x, time_emb)
        if self.attn is not None:
            x = self.attn(x, context)
        return x


class ControlNet(nn.Module):
    """ControlNet - 为 Stable Diffusion 添加条件控制"""

    def __init__(self, config: ControlNetConfig):
        super().__init__()
        self.config = config

        time_embed_dim = config.model_channels * 4
        latent_size = config.image_size // config.latent_scale_factor

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(config.model_channels),
            nn.Linear(config.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # 条件图像编码
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(config)

        # 输入卷积
        self.conv_in = nn.Conv2d(
            config.latent_channels, config.model_channels,
            kernel_size=3, padding=1
        )

        # 输入零卷积
        self.controlnet_down_blocks = nn.ModuleList()
        self.zero_convs = nn.ModuleList()

        # 初始零卷积
        self.zero_convs.append(ZeroConv(config.model_channels, config.model_channels))

        ch = config.model_channels
        resolution = latent_size

        # 下采样块
        for i, mult in enumerate(config.channel_mult):
            out_ch = config.model_channels * mult
            has_attn = resolution in config.attention_resolutions

            for _ in range(config.num_res_blocks):
                self.controlnet_down_blocks.append(
                    ControlNetBlock(ch, out_ch, time_embed_dim, config, has_attn)
                )
                self.zero_convs.append(ZeroConv(out_ch, out_ch))
                ch = out_ch

            if i < len(config.channel_mult) - 1:
                self.controlnet_down_blocks.append(Downsample(ch))
                self.zero_convs.append(ZeroConv(ch, ch))
                resolution //= 2

        # 中间块
        self.controlnet_mid_block = ControlNetBlock(ch, ch, time_embed_dim, config, has_attention=True)
        self.middle_block_out = ZeroConv(ch, ch)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        conditioning_scale: float = 1.0
    ) -> Dict[str, List[torch.Tensor]]:
        """
        前向传播

        Args:
            x: 潜在表示 [batch_size, latent_channels, H, W]
            timesteps: 时间步 [batch_size]
            controlnet_cond: 控制条件图像 [batch_size, 3, H*8, W*8]
            context: 文本条件 [batch_size, seq_len, context_dim]
            conditioning_scale: 控制强度

        Returns:
            包含下采样和中间块输出的字典
        """
        # 时间嵌入
        t_emb = self.time_embed(timesteps.float())

        # 条件图像编码
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

        # 输入
        h = self.conv_in(x)
        h = h + controlnet_cond

        down_block_res_samples = [self.zero_convs[0](h)]

        # 下采样
        zero_idx = 1
        block_idx = 0
        for i, mult in enumerate(self.config.channel_mult):
            for _ in range(self.config.num_res_blocks):
                h = self.controlnet_down_blocks[block_idx](h, t_emb, context)
                down_block_res_samples.append(self.zero_convs[zero_idx](h))
                block_idx += 1
                zero_idx += 1

            if i < len(self.config.channel_mult) - 1:
                h = self.controlnet_down_blocks[block_idx](h)
                down_block_res_samples.append(self.zero_convs[zero_idx](h))
                block_idx += 1
                zero_idx += 1

        # 中间块
        h = self.controlnet_mid_block(h, t_emb, context)
        mid_block_res_sample = self.middle_block_out(h)

        # 应用控制强度
        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        return {
            "down_block_res_samples": down_block_res_samples,
            "mid_block_res_sample": mid_block_res_sample
        }


def create_controlnet(control_type: str = "canny", model_size: str = "base") -> ControlNet:
    """
    创建 ControlNet 模型

    Args:
        control_type: 控制类型 ("canny", "pose", "depth")
        model_size: 模型大小 ("tiny", "base")

    Returns:
        ControlNet 模型实例
    """
    # 不同控制类型的通道数
    conditioning_channels = {
        "canny": 1,      # 边缘图 (灰度)
        "pose": 3,       # 姿态图 (RGB)
        "depth": 1,      # 深度图 (灰度)
        "segmentation": 3,  # 分割图 (RGB)
    }

    if control_type not in conditioning_channels:
        raise ValueError(f"Unknown control type: {control_type}. Choose from {list(conditioning_channels.keys())}")

    configs = {
        "tiny": ControlNetConfig(
            image_size=256,
            model_channels=128,
            channel_mult=(1, 2, 4),
            num_res_blocks=1,
            attention_resolutions=(2, 1),
            num_heads=4,
            context_dim=256,
            conditioning_channels=conditioning_channels[control_type]
        ),
        "base": ControlNetConfig(
            image_size=512,
            model_channels=320,
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            attention_resolutions=(4, 2, 1),
            num_heads=8,
            context_dim=768,
            conditioning_channels=conditioning_channels[control_type]
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return ControlNet(configs[model_size])
