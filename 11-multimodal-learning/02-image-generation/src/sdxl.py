"""
SDXL (Stable Diffusion XL) 实现

SDXL 是 Stable Diffusion 的升级版本，主要改进：
1. 双文本编码器 (CLIP ViT-L/14 + OpenCLIP ViT-bigG/14)
2. 更大的 UNet 架构 (2.6B 参数)
3. 支持 1024x1024 原生分辨率
4. 尺寸和裁剪条件嵌入
5. Refiner 模型支持

参考文献:
- SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
  https://arxiv.org/abs/2307.01952
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDXLModelType(Enum):
    """SDXL 模型类型"""
    BASE = "base"
    REFINER = "refiner"


@dataclass
class SDXLConfig:
    """SDXL 配置"""
    
    # 图像配置
    image_size: int = 1024
    in_channels: int = 3
    
    # VAE 配置
    latent_channels: int = 4
    latent_scale_factor: int = 8
    
    # UNet 配置
    model_channels: int = 320
    channel_mult: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 2)
    num_heads: int = -1  # 使用 head_dim 代替
    head_dim: int = 64
    transformer_depth: Tuple[int, ...] = (1, 2, 10)  # 每个阶段的 transformer 深度
    use_linear_projection: bool = True
    dropout: float = 0.0
    
    # 文本编码器配置 (双编码器)
    # CLIP ViT-L/14
    clip_embed_dim: int = 768
    clip_num_layers: int = 12
    clip_num_heads: int = 12
    clip_max_length: int = 77
    clip_vocab_size: int = 49408
    
    # OpenCLIP ViT-bigG/14
    openclip_embed_dim: int = 1280
    openclip_num_layers: int = 32
    openclip_num_heads: int = 20
    openclip_max_length: int = 77
    openclip_vocab_size: int = 49408
    
    # 组合文本嵌入维度
    context_dim: int = 2048  # clip_embed_dim + openclip_embed_dim
    pooled_embed_dim: int = 1280  # OpenCLIP pooled output
    
    # 条件嵌入
    addition_embed_type: str = "text_time"  # 额外条件类型
    addition_time_embed_dim: int = 256
    
    # 扩散配置
    num_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    
    # 模型类型
    model_type: SDXLModelType = SDXLModelType.BASE


@dataclass
class SDXLRefinerConfig(SDXLConfig):
    """SDXL Refiner 配置"""
    model_type: SDXLModelType = SDXLModelType.REFINER
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    transformer_depth: Tuple[int, ...] = (1, 1, 1, 1)
    # Refiner 只使用 OpenCLIP
    context_dim: int = 1280


# ============================================================================
# 文本编码器
# ============================================================================

class CLIPTextEmbedding(nn.Module):
    """CLIP 文本嵌入"""
    
    def __init__(self, vocab_size: int, embed_dim: int, max_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_length, embed_dim)
        )
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :x.shape[1], :]
        return x


class CLIPAttention(nn.Module):
    """CLIP 自注意力"""
    
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
        
    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    """CLIP MLP"""
    
    def __init__(self, embed_dim: int, intermediate_size: Optional[int] = None):
        super().__init__()
        intermediate_size = intermediate_size or embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate='tanh')
        return self.fc2(x)


class CLIPEncoderLayer(nn.Module):
    """CLIP 编码器层"""
    
    def __init__(self, embed_dim: int, num_heads: int, intermediate_size: Optional[int] = None):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = CLIPAttention(embed_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = CLIPMLP(embed_dim, intermediate_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPTextEncoderWithPooling(nn.Module):
    """带池化输出的 CLIP 文本编码器"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_length: int,
        intermediate_size: Optional[int] = None
    ):
        super().__init__()
        self.embeddings = CLIPTextEmbedding(vocab_size, embed_dim, max_length)
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(embed_dim, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        return_pooled: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embeddings(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_layer_norm(x)
        
        pooled = None
        if return_pooled:
            # 使用 EOS token 的嵌入作为池化输出
            eos_indices = input_ids.argmax(dim=-1)
            pooled = x[torch.arange(x.shape[0], device=x.device), eos_indices]
            pooled = self.text_projection(pooled)
            
        return x, pooled


class SDXLTextEncoder(nn.Module):
    """SDXL 双文本编码器"""
    
    def __init__(self, config: SDXLConfig):
        super().__init__()
        self.config = config
        
        # CLIP ViT-L/14
        self.clip_encoder = CLIPTextEncoderWithPooling(
            vocab_size=config.clip_vocab_size,
            embed_dim=config.clip_embed_dim,
            num_layers=config.clip_num_layers,
            num_heads=config.clip_num_heads,
            max_length=config.clip_max_length
        )
        
        # OpenCLIP ViT-bigG/14
        self.openclip_encoder = CLIPTextEncoderWithPooling(
            vocab_size=config.openclip_vocab_size,
            embed_dim=config.openclip_embed_dim,
            num_layers=config.openclip_num_layers,
            num_heads=config.openclip_num_heads,
            max_length=config.openclip_max_length,
            intermediate_size=config.openclip_embed_dim * 4
        )
        
    def forward(
        self,
        clip_input_ids: torch.Tensor,
        openclip_input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本
        
        Returns:
            text_embeds: 拼接的文本嵌入 [B, seq_len, context_dim]
            pooled_embeds: OpenCLIP 池化嵌入 [B, pooled_embed_dim]
        """
        clip_embeds, _ = self.clip_encoder(clip_input_ids, return_pooled=False)
        openclip_embeds, pooled_embeds = self.openclip_encoder(openclip_input_ids, return_pooled=True)
        
        # 拼接两个编码器的输出
        text_embeds = torch.cat([clip_embeds, openclip_embeds], dim=-1)
        
        return text_embeds, pooled_embeds


# ============================================================================
# 条件嵌入
# ============================================================================

class TimestepEmbedding(nn.Module):
    """时间步嵌入"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x


class SinusoidalTimestepEmbedding(nn.Module):
    """正弦时间步嵌入"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=timesteps.device) / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class SDXLAdditionEmbedding(nn.Module):
    """SDXL 额外条件嵌入 (尺寸、裁剪、目标尺寸)"""
    
    def __init__(self, config: SDXLConfig):
        super().__init__()
        self.config = config
        
        # 时间嵌入维度
        time_embed_dim = config.model_channels * 4
        
        # 池化文本嵌入投影
        self.text_embedder = nn.Sequential(
            nn.Linear(config.pooled_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 尺寸条件嵌入 (original_size, crop_coords, target_size)
        # 每个条件有 2 个值 (height, width)，共 6 个值
        self.add_time_proj = SinusoidalTimestepEmbedding(config.addition_time_embed_dim)
        self.add_embedding = nn.Sequential(
            nn.Linear(config.addition_time_embed_dim * 6, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
    def forward(
        self,
        pooled_embeds: torch.Tensor,
        original_size: torch.Tensor,
        crop_coords: torch.Tensor,
        target_size: torch.Tensor
    ) -> torch.Tensor:
        # 池化文本嵌入
        text_emb = self.text_embedder(pooled_embeds)
        
        # 尺寸条件嵌入
        time_embeds = []
        for size_tensor in [original_size, crop_coords, target_size]:
            for i in range(size_tensor.shape[1]):
                time_embeds.append(self.add_time_proj(size_tensor[:, i]))
        
        time_emb = torch.cat(time_embeds, dim=-1)
        add_emb = self.add_embedding(time_emb)
        
        return text_emb + add_emb


# ============================================================================
# 注意力模块
# ============================================================================

class SDXLCrossAttention(nn.Module):
    """SDXL 交叉注意力"""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_linear_projection: bool = True
    ):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = head_dim * num_heads
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_linear_projection = use_linear_projection
        
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
        if context is None:
            context = x
            
        batch_size = x.shape[0]
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.to_out(out)


class GEGLU(nn.Module):
    """GEGLU 激活"""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class SDXLFeedForward(nn.Module):
    """SDXL 前馈网络"""
    
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


class SDXLTransformerBlock(nn.Module):
    """SDXL Transformer 块"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        dropout: float = 0.0,
        use_linear_projection: bool = True
    ):
        super().__init__()
        
        # 自注意力
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = SDXLCrossAttention(
            dim, dim, num_heads, head_dim, dropout, use_linear_projection
        )
        
        # 交叉注意力
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = SDXLCrossAttention(
            dim, context_dim, num_heads, head_dim, dropout, use_linear_projection
        )
        
        # 前馈网络
        self.norm3 = nn.LayerNorm(dim)
        self.ff = SDXLFeedForward(dim, dropout=dropout)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        # 交叉注意力：如果没有 context，使用 x 作为 context
        if context is not None:
            x = x + self.attn2(self.norm2(x), context)
        else:
            x = x + self.attn2(self.norm2(x), x)
        x = x + self.ff(self.norm3(x))
        return x


class SDXLSpatialTransformer(nn.Module):
    """SDXL 空间 Transformer"""
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        depth: int,
        context_dim: int,
        dropout: float = 0.0,
        use_linear_projection: bool = True
    ):
        super().__init__()
        inner_dim = num_heads * head_dim
        
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
            
        self.transformer_blocks = nn.ModuleList([
            SDXLTransformerBlock(inner_dim, num_heads, head_dim, context_dim, dropout, use_linear_projection)
            for _ in range(depth)
        ])
        
        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1)
            
        self.use_linear_projection = use_linear_projection
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        
        if self.use_linear_projection:
            x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
            x = self.proj_in(x)
        else:
            x = self.proj_in(x)
            x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)
            
        for block in self.transformer_blocks:
            x = block(x, context)
            
        if self.use_linear_projection:
            x = self.proj_out(x)
            x = x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        else:
            x = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
            x = self.proj_out(x)
            
        return x + residual


# ============================================================================
# UNet 组件
# ============================================================================

class SDXLResBlock(nn.Module):
    """SDXL 残差块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
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


class SDXLDownsample(nn.Module):
    """SDXL 下采样"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SDXLUpsample(nn.Module):
    """SDXL 上采样"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SDXLDownBlock(nn.Module):
    """SDXL 下采样块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_res_blocks: int,
        transformer_depth: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        dropout: float = 0.0,
        add_downsample: bool = True,
        use_linear_projection: bool = True
    ):
        super().__init__()
        
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(SDXLResBlock(in_ch, out_channels, time_embed_dim, dropout))
            
            if transformer_depth > 0:
                self.attentions.append(
                    SDXLSpatialTransformer(
                        out_channels, num_heads, head_dim, transformer_depth,
                        context_dim, dropout, use_linear_projection
                    )
                )
            else:
                self.attentions.append(None)
                
        self.downsample = SDXLDownsample(out_channels) if add_downsample else None
        
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        outputs = []
        
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, time_emb)
            if attn is not None:
                x = attn(x, context)
            outputs.append(x)
            
        if self.downsample is not None:
            x = self.downsample(x)
            outputs.append(x)
            
        return x, outputs


class SDXLUpBlock(nn.Module):
    """SDXL 上采样块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_channels: int,
        time_embed_dim: int,
        num_res_blocks: int,
        transformer_depth: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        dropout: float = 0.0,
        add_upsample: bool = True,
        use_linear_projection: bool = True
    ):
        super().__init__()
        
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for i in range(num_res_blocks):
            skip_channels = prev_channels if i == 0 else out_channels
            in_ch = in_channels if i == 0 else out_channels
            
            self.resnets.append(
                SDXLResBlock(in_ch + skip_channels, out_channels, time_embed_dim, dropout)
            )
            
            if transformer_depth > 0:
                self.attentions.append(
                    SDXLSpatialTransformer(
                        out_channels, num_heads, head_dim, transformer_depth,
                        context_dim, dropout, use_linear_projection
                    )
                )
            else:
                self.attentions.append(None)
                
        self.upsample = SDXLUpsample(out_channels) if add_upsample else None
        
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        skips: List[torch.Tensor],
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = resnet(x, time_emb)
            if attn is not None:
                x = attn(x, context)
                
        if self.upsample is not None:
            x = self.upsample(x)
            
        return x


class SDXLUpBlockV2(nn.Module):
    """SDXL 上采样块 V2 - 支持不同的 skip 通道数"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: List[int],
        time_embed_dim: int,
        num_res_blocks: int,
        transformer_depth: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        dropout: float = 0.0,
        add_upsample: bool = True,
        use_linear_projection: bool = True
    ):
        super().__init__()
        
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for i in range(num_res_blocks):
            skip_ch = skip_channels[i] if i < len(skip_channels) else out_channels
            in_ch = in_channels if i == 0 else out_channels
            
            self.resnets.append(
                SDXLResBlock(in_ch + skip_ch, out_channels, time_embed_dim, dropout)
            )
            
            if transformer_depth > 0:
                self.attentions.append(
                    SDXLSpatialTransformer(
                        out_channels, num_heads, head_dim, transformer_depth,
                        context_dim, dropout, use_linear_projection
                    )
                )
            else:
                self.attentions.append(None)
                
        self.upsample = SDXLUpsample(out_channels) if add_upsample else None
        
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        skips: List[torch.Tensor],
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = resnet(x, time_emb)
            if attn is not None:
                x = attn(x, context)
                
        if self.upsample is not None:
            x = self.upsample(x)
            
        return x


class SDXLMidBlock(nn.Module):
    """SDXL 中间块"""
    
    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        transformer_depth: int,
        num_heads: int,
        head_dim: int,
        context_dim: int,
        dropout: float = 0.0,
        use_linear_projection: bool = True
    ):
        super().__init__()
        
        self.resnet1 = SDXLResBlock(channels, channels, time_embed_dim, dropout)
        self.attention = SDXLSpatialTransformer(
            channels, num_heads, head_dim, transformer_depth,
            context_dim, dropout, use_linear_projection
        )
        self.resnet2 = SDXLResBlock(channels, channels, time_embed_dim, dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.resnet1(x, time_emb)
        x = self.attention(x, context)
        x = self.resnet2(x, time_emb)
        return x


# ============================================================================
# SDXL UNet
# ============================================================================

class SDXLUNet(nn.Module):
    """SDXL UNet"""
    
    def __init__(self, config: SDXLConfig):
        super().__init__()
        self.config = config
        
        time_embed_dim = config.model_channels * 4
        num_heads = config.model_channels // config.head_dim
        
        # 时间嵌入
        self.time_proj = SinusoidalTimestepEmbedding(config.model_channels)
        self.time_embedding = TimestepEmbedding(config.model_channels, time_embed_dim)
        
        # 额外条件嵌入
        self.add_embedding = SDXLAdditionEmbedding(config)
        
        # 输入卷积
        self.conv_in = nn.Conv2d(config.latent_channels, config.model_channels, kernel_size=3, padding=1)
        
        # 计算每层的通道数
        block_out_channels = [config.model_channels * m for m in config.channel_mult]
        
        # 跟踪 skip 连接的通道数
        self.skip_channels = [config.model_channels]  # conv_in 输出
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        output_channel = config.model_channels
        
        for i, out_ch in enumerate(block_out_channels):
            input_channel = output_channel
            transformer_depth = config.transformer_depth[i] if i < len(config.transformer_depth) else 0
            is_final = (i == len(block_out_channels) - 1)
            
            self.down_blocks.append(
                SDXLDownBlock(
                    input_channel, out_ch, time_embed_dim, config.num_res_blocks,
                    transformer_depth, num_heads, config.head_dim, config.context_dim,
                    config.dropout, add_downsample=not is_final,
                    use_linear_projection=config.use_linear_projection
                )
            )
            
            # 记录每个 res block 的输出通道
            for _ in range(config.num_res_blocks):
                self.skip_channels.append(out_ch)
            # 如果有下采样，也记录
            if not is_final:
                self.skip_channels.append(out_ch)
                
            output_channel = out_ch
                
        # 中间块
        mid_transformer_depth = config.transformer_depth[-1] if config.transformer_depth else 1
        self.mid_block = SDXLMidBlock(
            output_channel, time_embed_dim, mid_transformer_depth, num_heads, config.head_dim,
            config.context_dim, config.dropout, config.use_linear_projection
        )
        
        # 上采样块 - 反向遍历
        self.up_blocks = nn.ModuleList()
        reversed_block_out_channels = list(reversed(block_out_channels))
        
        for i, out_ch in enumerate(reversed_block_out_channels):
            is_final = (i == len(reversed_block_out_channels) - 1)
            
            # 对应的下采样块索引
            down_idx = len(block_out_channels) - 1 - i
            transformer_depth = config.transformer_depth[down_idx] if down_idx < len(config.transformer_depth) else 0
            
            # 计算需要的 skip 通道数 (不需要 reverse，因为 pop 是从末尾取)
            skip_chs = []
            for _ in range(config.num_res_blocks + 1):
                skip_chs.append(self.skip_channels.pop())
            # skip_chs 现在的顺序就是 pop 的顺序，与 forward 中使用的顺序一致
            
            self.up_blocks.append(
                SDXLUpBlockV2(
                    output_channel, out_ch, skip_chs, time_embed_dim, config.num_res_blocks + 1,
                    transformer_depth, num_heads, config.head_dim, config.context_dim,
                    config.dropout, add_upsample=not is_final,
                    use_linear_projection=config.use_linear_projection
                )
            )
            output_channel = out_ch
            
        # 输出层
        self.norm_out = nn.GroupNorm(32, output_channel, eps=1e-6)
        self.conv_out = nn.Conv2d(output_channel, config.latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        pooled_embeds: torch.Tensor,
        original_size: torch.Tensor,
        crop_coords: torch.Tensor,
        target_size: torch.Tensor
    ) -> torch.Tensor:
        # 时间嵌入
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        
        # 额外条件嵌入
        add_emb = self.add_embedding(pooled_embeds, original_size, crop_coords, target_size)
        t_emb = t_emb + add_emb
        
        # 输入
        x = self.conv_in(x)
        
        # 下采样
        skips = [x]
        for block in self.down_blocks:
            x, block_skips = block(x, t_emb, context)
            skips.extend(block_skips)
            
        # 中间
        x = self.mid_block(x, t_emb, context)
        
        # 上采样
        for block in self.up_blocks:
            x = block(x, t_emb, skips, context)
            
        # 输出
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x


# ============================================================================
# SDXL 噪声调度器
# ============================================================================

class SDXLNoiseScheduler:
    """SDXL 噪声调度器"""
    
    def __init__(self, config: SDXLConfig):
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
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(latents.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(latents.device)
        
        while len(sqrt_alpha.shape) < len(latents.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
            
        return sqrt_alpha * latents + sqrt_one_minus_alpha * noise


# ============================================================================
# SDXL 完整模型
# ============================================================================

class SDXL(nn.Module):
    """SDXL 完整模型"""
    
    def __init__(self, config: SDXLConfig):
        super().__init__()
        self.config = config
        
        # 文本编码器
        self.text_encoder = SDXLTextEncoder(config)
        
        # UNet
        self.unet = SDXLUNet(config)
        
        # 噪声调度器
        self.scheduler = SDXLNoiseScheduler(config)
        
        # VAE 缩放因子
        self.vae_scale_factor = 0.13025  # SDXL 使用的缩放因子
        
    def encode_text(
        self,
        clip_input_ids: torch.Tensor,
        openclip_input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.text_encoder(clip_input_ids, openclip_input_ids)
    
    def forward(
        self,
        latents: torch.Tensor,
        clip_input_ids: torch.Tensor,
        openclip_input_ids: torch.Tensor,
        original_size: Optional[torch.Tensor] = None,
        crop_coords: Optional[torch.Tensor] = None,
        target_size: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """训练前向传播"""
        batch_size = latents.shape[0]
        device = latents.device
        
        # 默认尺寸条件
        if original_size is None:
            original_size = torch.tensor([[self.config.image_size, self.config.image_size]], device=device).expand(batch_size, -1)
        if crop_coords is None:
            crop_coords = torch.zeros(batch_size, 2, device=device)
        if target_size is None:
            target_size = torch.tensor([[self.config.image_size, self.config.image_size]], device=device).expand(batch_size, -1)
            
        # 文本编码
        text_embeds, pooled_embeds = self.encode_text(clip_input_ids, openclip_input_ids)
        
        # 添加噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.config.num_timesteps, (batch_size,), device=device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.unet(
            noisy_latents, timesteps, text_embeds, pooled_embeds,
            original_size, crop_coords, target_size
        )
        
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def generate(
        self,
        text_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_text_embeds: Optional[torch.Tensor] = None,
        negative_pooled_embeds: Optional[torch.Tensor] = None,
        original_size: Optional[torch.Tensor] = None,
        crop_coords: Optional[torch.Tensor] = None,
        target_size: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """SDXL 图像生成"""
        batch_size = text_embeds.shape[0]
        device = text_embeds.device
        latent_size = self.config.image_size // self.config.latent_scale_factor
        
        # 初始化潜在表示
        if latents is None:
            latents = torch.randn(
                batch_size, self.config.latent_channels, latent_size, latent_size,
                device=device
            )
            
        # 默认尺寸条件
        if original_size is None:
            original_size = torch.tensor(
                [[self.config.image_size, self.config.image_size]], device=device
            ).expand(batch_size, -1)
        if crop_coords is None:
            crop_coords = torch.zeros(batch_size, 2, device=device)
        if target_size is None:
            target_size = original_size.clone()
            
        # Classifier-Free Guidance
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            if negative_text_embeds is None:
                negative_text_embeds = torch.zeros_like(text_embeds)
            if negative_pooled_embeds is None:
                negative_pooled_embeds = torch.zeros_like(pooled_embeds)
            text_embeds = torch.cat([negative_text_embeds, text_embeds])
            pooled_embeds = torch.cat([negative_pooled_embeds, pooled_embeds])
            original_size = torch.cat([original_size, original_size])
            crop_coords = torch.cat([crop_coords, crop_coords])
            target_size = torch.cat([target_size, target_size])
            
        # 时间步
        step_ratio = self.config.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0).to(device)
        
        # 采样循环
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            if do_cfg:
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t_batch] * 2)
            else:
                latent_input = latents
                t_input = t_batch
                
            noise_pred = self.unet(
                latent_input, t_input, text_embeds, pooled_embeds,
                original_size, crop_coords, target_size
            )
            
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            # DDIM 步骤
            alpha_t = self.scheduler.alphas_cumprod[t].to(device)
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(0)
            alpha_t_prev = self.scheduler.alphas_cumprod[t_prev].to(device)
            
            pred_x0 = (latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred
            latents = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt
            
        return latents


# ============================================================================
# 工厂函数
# ============================================================================

def create_sdxl_model(model_type: str = "tiny") -> SDXL:
    """创建 SDXL 模型"""
    configs = {
        "tiny": SDXLConfig(
            image_size=256,
            model_channels=128,
            channel_mult=(1, 2, 4),
            num_res_blocks=1,
            transformer_depth=(1, 1, 1),
            head_dim=32,
            clip_embed_dim=256,
            clip_num_layers=4,
            clip_num_heads=4,
            openclip_embed_dim=512,
            openclip_num_layers=8,
            openclip_num_heads=8,
            context_dim=768,
            pooled_embed_dim=512,
            addition_time_embed_dim=128
        ),
        "small": SDXLConfig(
            image_size=512,
            model_channels=256,
            channel_mult=(1, 2, 4),
            num_res_blocks=2,
            transformer_depth=(1, 2, 4),
            head_dim=64,
            clip_embed_dim=512,
            clip_num_layers=8,
            clip_num_heads=8,
            openclip_embed_dim=768,
            openclip_num_layers=16,
            openclip_num_heads=12,
            context_dim=1280,
            pooled_embed_dim=768,
            addition_time_embed_dim=256
        ),
        "base": SDXLConfig(
            image_size=1024,
            model_channels=320,
            channel_mult=(1, 2, 4),
            num_res_blocks=2,
            transformer_depth=(1, 2, 10),
            head_dim=64,
            clip_embed_dim=768,
            clip_num_layers=12,
            clip_num_heads=12,
            openclip_embed_dim=1280,
            openclip_num_layers=32,
            openclip_num_heads=20,
            context_dim=2048,
            pooled_embed_dim=1280,
            addition_time_embed_dim=256
        )
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(configs.keys())}")
        
    return SDXL(configs[model_type])
