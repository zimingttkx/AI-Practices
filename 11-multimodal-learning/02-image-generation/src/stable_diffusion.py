"""
Stable Diffusion 实现

Stable Diffusion 是一种潜在扩散模型 (Latent Diffusion Model)，
在压缩的潜在空间中进行扩散过程，大大提高了效率。

=== 核心思想 ===

Stable Diffusion 的关键创新：

1. 潜在空间扩散 (Latent Diffusion)
   - 使用 VAE 将图像压缩到低维潜在空间
   - 在潜在空间而非像素空间进行扩散
   - 计算效率提升 4-8 倍

2. 条件生成 (Conditional Generation)
   - 使用 CLIP 文本编码器处理文本提示
   - 通过交叉注意力将文本条件注入 UNet
   - 支持 Classifier-Free Guidance

3. 高效架构
   - UNet 带有交叉注意力层
   - 支持多种分辨率 (512x512, 768x768 等)

=== 数学基础 ===

潜在空间编码:
    z = E(x) / scale_factor
    x = D(z * scale_factor)

条件扩散:
    ε_θ(z_t, t, c) = UNet(z_t, t, TextEncoder(prompt))

Classifier-Free Guidance:
    ε̃ = ε_uncond + w * (ε_cond - ε_uncond)
    
    其中 w 是 guidance scale (通常 7.5)

=== 参考文献 ===

1. Latent Diffusion:
   Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" CVPR 2022
   https://arxiv.org/abs/2112.10752

2. Classifier-Free Guidance:
   Ho & Salimans "Classifier-Free Diffusion Guidance" 2022

=== 核心组件 ===

    - SDConfig: Stable Diffusion 配置
    - CLIPTextEncoder: CLIP 文本编码器
    - SpatialTransformer: 空间 Transformer (交叉注意力)
    - SDUNet: Stable Diffusion UNet
    - SDNoiseScheduler: 噪声调度器
    - StableDiffusion: 完整的文本到图像模型
    - create_sd_model: 创建预定义模型
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae import VAE, VAEConfig
from diffusion import (
    DiffusionConfig,
    UNet,
    NoiseScheduler,
    TimeEmbedding,
    ResBlock,
    AttentionBlock,
    SinusoidalPositionEmbedding
)


@dataclass
class SDConfig:
    """Stable Diffusion 配置"""

    # 图像配置
    image_size: int = 512
    in_channels: int = 3

    # VAE 配置
    latent_channels: int = 4
    latent_scale_factor: int = 8  # 图像到潜在空间的缩放因子

    # UNet 配置
    model_channels: int = 320
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 2, 1)  # 相对于潜在空间尺寸
    num_heads: int = 8
    transformer_depth: int = 1
    dropout: float = 0.0

    # 文本编码器配置
    vocab_size: int = 49408
    max_text_length: int = 77
    text_embed_dim: int = 768
    text_num_layers: int = 12
    text_num_heads: int = 12

    # 扩散配置
    num_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012

    # 条件配置
    context_dim: int = 768  # 文本条件维度


class CLIPTextEmbedding(nn.Module):
    """CLIP 文本嵌入层"""

    def __init__(self, config: SDConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_text_length, config.text_embed_dim)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :x.shape[1], :]
        return x


class CLIPAttention(nn.Module):
    """CLIP 注意力层"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(output)


class CLIPMLP(nn.Module):
    """CLIP MLP 层"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate='tanh')
        x = self.fc2(x)
        return x


class CLIPEncoderLayer(nn.Module):
    """CLIP 编码器层"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = CLIPAttention(embed_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = CLIPMLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPTextEncoder(nn.Module):
    """CLIP 文本编码器"""

    def __init__(self, config: SDConfig):
        super().__init__()
        self.config = config

        self.embeddings = CLIPTextEmbedding(config)
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(config.text_embed_dim, config.text_num_heads)
            for _ in range(config.text_num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.text_embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码文本

        Args:
            input_ids: token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 (可选)

        Returns:
            文本嵌入 [batch_size, seq_len, embed_dim]
        """
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_layer_norm(x)
        return x


class CrossAttention(nn.Module):
    """交叉注意力 - 用于文本条件"""

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 查询 [batch_size, seq_len, query_dim]
            context: 上下文 [batch_size, context_len, context_dim]
        """
        if context is None:
            context = x

        batch_size = x.shape[0]

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        return self.to_out(output)


class GEGLU(nn.Module):
    """GEGLU 激活函数"""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """基础 Transformer 块 - 自注意力 + 交叉注意力 + FFN"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()

        # 自注意力
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, dim, num_heads, head_dim, dropout)

        # 交叉注意力
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, num_heads, head_dim, dropout)

        # 前馈网络
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    """空间 Transformer - 将 Transformer 应用于空间特征"""

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        depth: int,
        context_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = num_heads * head_dim

        self.norm = nn.GroupNorm(32, in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, num_heads, head_dim, context_dim, dropout)
            for _ in range(depth)
        ])

        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        # 重塑为序列
        x = x.view(batch_size, -1, height * width).transpose(1, 2)

        for block in self.transformer_blocks:
            x = block(x, context)

        # 重塑回空间
        x = x.transpose(1, 2).view(batch_size, -1, height, width)
        x = self.proj_out(x)

        return x + residual


class SDResBlock(nn.Module):
    """Stable Diffusion 残差块"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample(nn.Module):
    """下采样层"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """上采样层"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SDUNet(nn.Module):
    """Stable Diffusion UNet - 带交叉注意力的条件 UNet"""

    def __init__(self, config: SDConfig):
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

        # 输入卷积
        self.conv_in = nn.Conv2d(
            config.latent_channels, config.model_channels,
            kernel_size=3, padding=1
        )

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        ch = config.model_channels
        resolution = latent_size
        channels_list = [ch]

        for i, mult in enumerate(config.channel_mult):
            out_ch = config.model_channels * mult
            has_attn = resolution in config.attention_resolutions

            # 残差块
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(SDResBlock(ch, out_ch, time_embed_dim, config.dropout))
                if has_attn:
                    self.down_blocks.append(
                        SpatialTransformer(
                            out_ch, config.num_heads, out_ch // config.num_heads,
                            config.transformer_depth, config.context_dim, config.dropout
                        )
                    )
                ch = out_ch
                channels_list.append(ch)

            # 下采样
            if i < len(config.channel_mult) - 1:
                self.down_samples.append(Downsample(ch))
                channels_list.append(ch)
                resolution //= 2

        # 中间块
        self.mid_block1 = SDResBlock(ch, ch, time_embed_dim, config.dropout)
        self.mid_attn = SpatialTransformer(
            ch, config.num_heads, ch // config.num_heads,
            config.transformer_depth, config.context_dim, config.dropout
        )
        self.mid_block2 = SDResBlock(ch, ch, time_embed_dim, config.dropout)

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(config.channel_mult)):
            out_ch = config.model_channels * mult
            has_attn = resolution in config.attention_resolutions

            for j in range(config.num_res_blocks + 1):
                skip_ch = channels_list.pop()
                self.up_blocks.append(SDResBlock(ch + skip_ch, out_ch, time_embed_dim, config.dropout))
                if has_attn:
                    self.up_blocks.append(
                        SpatialTransformer(
                            out_ch, config.num_heads, out_ch // config.num_heads,
                            config.transformer_depth, config.context_dim, config.dropout
                        )
                    )
                ch = out_ch

            if i < len(config.channel_mult) - 1:
                self.up_samples.append(Upsample(ch))
                resolution *= 2

        # 输出层
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, config.latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        # 保存通道列表用于 forward
        self._channels_list = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 潜在表示 [batch_size, latent_channels, H, W]
            timesteps: 时间步 [batch_size]
            context: 文本条件 [batch_size, seq_len, context_dim]

        Returns:
            预测的噪声 [batch_size, latent_channels, H, W]
        """
        # 时间嵌入
        t_emb = self.time_embed(timesteps.float())

        # 输入卷积
        h = self.conv_in(x)
        skips = [h]

        # 下采样
        down_idx = 0
        sample_idx = 0
        for i, mult in enumerate(self.config.channel_mult):
            for _ in range(self.config.num_res_blocks):
                block = self.down_blocks[down_idx]
                h = block(h, t_emb)
                down_idx += 1

                # 检查是否有注意力块
                if down_idx < len(self.down_blocks) and isinstance(self.down_blocks[down_idx], SpatialTransformer):
                    h = self.down_blocks[down_idx](h, context)
                    down_idx += 1

                skips.append(h)

            if i < len(self.config.channel_mult) - 1:
                h = self.down_samples[sample_idx](h)
                sample_idx += 1
                skips.append(h)

        # 中间块
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)

        # 上采样
        up_idx = 0
        sample_idx = 0
        for i, mult in enumerate(reversed(self.config.channel_mult)):
            for _ in range(self.config.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)

                block = self.up_blocks[up_idx]
                h = block(h, t_emb)
                up_idx += 1

                if up_idx < len(self.up_blocks) and isinstance(self.up_blocks[up_idx], SpatialTransformer):
                    h = self.up_blocks[up_idx](h, context)
                    up_idx += 1

            if i < len(self.config.channel_mult) - 1:
                h = self.up_samples[sample_idx](h)
                sample_idx += 1

        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class SDNoiseScheduler:
    """Stable Diffusion 噪声调度器"""

    def __init__(self, config: SDConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps

        # Scaled linear beta schedule
        betas = torch.linspace(
            config.beta_start ** 0.5,
            config.beta_end ** 0.5,
            config.num_timesteps
        ) ** 2

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """添加噪声到潜在表示"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(latents.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(latents.device)

        while len(sqrt_alpha.shape) < len(latents.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * latents + sqrt_one_minus_alpha * noise


class StableDiffusion(nn.Module):
    """Stable Diffusion 完整模型"""

    def __init__(self, config: SDConfig):
        super().__init__()
        self.config = config

        # 文本编码器
        self.text_encoder = CLIPTextEncoder(config)

        # UNet
        self.unet = SDUNet(config)

        # 噪声调度器
        self.scheduler = SDNoiseScheduler(config)

        # VAE (简化版)
        vae_config = VAEConfig(
            image_size=config.image_size,
            latent_channels=config.latent_channels,
            encoder_channels=(64, 128, 256, 512),
            decoder_channels=(512, 256, 128, 64)
        )
        self.vae = VAE(vae_config)
        self.vae_scale_factor = 0.18215

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """编码文本为条件嵌入"""
        return self.text_encoder(input_ids)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像到潜在空间"""
        mu, _ = self.vae.encode(images)
        return mu * self.vae_scale_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码图像"""
        latents = latents / self.vae_scale_factor
        return self.vae.decode(latents)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """训练前向传播"""
        batch_size = images.shape[0]
        device = images.device

        latents = self.encode_image(images)
        text_embeds = self.encode_text(input_ids)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.config.num_timesteps, (batch_size,), device=device)

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.unet(noisy_latents, timesteps, text_embeds)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """文本到图像生成"""
        batch_size = prompt_embeds.shape[0]
        device = prompt_embeds.device
        latent_size = self.config.image_size // self.config.latent_scale_factor

        latents = torch.randn(batch_size, self.config.latent_channels, latent_size, latent_size, device=device)

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        step_ratio = self.config.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0).to(device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t_batch] * 2)
                noise_pred = self.unet(latent_input, t_input, prompt_embeds)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.unet(latents, t_batch, prompt_embeds)

            alpha_t = self.scheduler.alphas_cumprod[t].to(device)
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0)
            alpha_t_prev = self.scheduler.alphas_cumprod[t_prev].to(device)

            pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred
            latents = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return self.decode_latents(latents)


def create_sd_model(model_size: str = "tiny") -> StableDiffusion:
    """创建预定义大小的 Stable Diffusion 模型"""
    configs = {
        "tiny": SDConfig(
            image_size=256,
            model_channels=128,
            channel_mult=(1, 2, 4),
            num_res_blocks=1,
            attention_resolutions=(2, 1),
            num_heads=4,
            text_embed_dim=256,
            text_num_layers=4,
            context_dim=256,
            num_timesteps=1000
        ),
        "base": SDConfig(
            image_size=512,
            model_channels=320,
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            attention_resolutions=(4, 2, 1),
            num_heads=8,
            text_embed_dim=768,
            text_num_layers=12,
            context_dim=768,
            num_timesteps=1000
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    return StableDiffusion(configs[model_size])
