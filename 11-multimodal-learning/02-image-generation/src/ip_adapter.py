"""
IP-Adapter (Image Prompt Adapter) 实现

IP-Adapter 允许使用图像作为提示来引导图像生成，
通过将图像特征注入到扩散模型的交叉注意力层中实现。

支持的功能:
- 图像编码器 (CLIP Vision)
- 图像投影层
- 解耦交叉注意力
- 多图像融合
- 风格/内容分离
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class IPAdapterType(Enum):
    """IP-Adapter 类型"""
    STANDARD = "standard"       # 标准 IP-Adapter
    PLUS = "plus"               # IP-Adapter Plus (更多特征)
    PLUS_FACE = "plus_face"     # IP-Adapter Plus Face
    FULL_FACE = "full_face"     # 完整人脸特征


@dataclass
class IPAdapterConfig:
    """IP-Adapter 配置"""
    adapter_type: IPAdapterType = IPAdapterType.STANDARD
    
    # 图像编码器配置
    image_encoder_dim: int = 1024           # CLIP 图像编码器维度
    num_image_tokens: int = 4               # 图像 token 数量
    
    # 投影层配置
    cross_attention_dim: int = 768          # 交叉注意力维度
    num_projection_layers: int = 4          # 投影层数量
    
    # 注意力配置
    num_heads: int = 12                     # 注意力头数
    scale: float = 1.0                      # 图像特征缩放因子
    
    # Plus 版本配置
    use_fine_grained_features: bool = False # 使用细粒度特征
    num_queries: int = 16                   # 查询数量 (Plus 版本)
    
    # 人脸特定配置
    use_face_id: bool = False               # 使用人脸 ID 特征
    face_embed_dim: int = 512               # 人脸嵌入维度


class ImageProjection(nn.Module):
    """图像投影层 - 将 CLIP 图像特征投影到交叉注意力空间"""
    
    def __init__(
        self,
        image_embed_dim: int = 1024,
        cross_attention_dim: int = 768,
        num_image_tokens: int = 4
    ):
        super().__init__()
        self.image_embed_dim = image_embed_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_image_tokens = num_image_tokens
        
        self.proj = nn.Linear(image_embed_dim, cross_attention_dim * num_image_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: [batch, image_embed_dim]
        Returns:
            [batch, num_image_tokens, cross_attention_dim]
        """
        batch_size = image_embeds.shape[0]
        
        # 投影并重塑
        x = self.proj(image_embeds)
        x = x.view(batch_size, self.num_image_tokens, self.cross_attention_dim)
        x = self.norm(x)
        
        return x


class ImageProjectionPlus(nn.Module):
    """IP-Adapter Plus 图像投影层 - 使用可学习查询提取细粒度特征"""
    
    def __init__(
        self,
        image_embed_dim: int = 1024,
        cross_attention_dim: int = 768,
        num_queries: int = 16,
        num_heads: int = 12,
        num_layers: int = 4
    ):
        super().__init__()
        self.num_queries = num_queries
        self.cross_attention_dim = cross_attention_dim
        
        # 可学习查询
        self.queries = nn.Parameter(torch.randn(1, num_queries, cross_attention_dim) * 0.02)
        
        # 输入投影
        self.input_proj = nn.Linear(image_embed_dim, cross_attention_dim)
        
        # Transformer 解码器层
        self.layers = nn.ModuleList([
            PerceiverAttentionBlock(cross_attention_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(
        self,
        image_embeds: torch.Tensor,
        image_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            image_embeds: [batch, image_embed_dim] 或 [batch, seq_len, image_embed_dim]
            image_hidden_states: 可选的细粒度特征 [batch, seq_len, dim]
        Returns:
            [batch, num_queries, cross_attention_dim]
        """
        batch_size = image_embeds.shape[0]
        
        # 处理输入
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(1)
            
        # 投影图像特征
        x = self.input_proj(image_embeds)
        
        # 如果有细粒度特征，拼接
        if image_hidden_states is not None:
            hidden_proj = self.input_proj(image_hidden_states)
            x = torch.cat([x, hidden_proj], dim=1)
            
        # 扩展查询到 batch
        queries = self.queries.expand(batch_size, -1, -1)
        
        # 通过 Perceiver 层
        for layer in self.layers:
            queries = layer(queries, x)
            
        return self.norm(queries)


class PerceiverAttentionBlock(nn.Module):
    """Perceiver 风格的注意力块"""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # 交叉注意力
        x = self.norm1(queries)
        context_norm = self.norm2(context)
        x = queries + self.cross_attn(x, context_norm, context_norm)[0]
        
        # 自注意力
        x_norm = self.norm3(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]
        
        # FFN
        x = x + self.ffn(x)
        
        return x


class IPAdapterCrossAttention(nn.Module):
    """IP-Adapter 解耦交叉注意力"""
    
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        scale: float = 1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = scale
        self.attention_scale = head_dim ** -0.5
        
        # 文本注意力的 K, V
        self.to_k_text = nn.Linear(cross_attention_dim, self.inner_dim, bias=False)
        self.to_v_text = nn.Linear(cross_attention_dim, self.inner_dim, bias=False)
        
        # 图像注意力的 K, V (IP-Adapter 新增)
        self.to_k_image = nn.Linear(cross_attention_dim, self.inner_dim, bias=False)
        self.to_v_image = nn.Linear(cross_attention_dim, self.inner_dim, bias=False)
        
        # 共享的 Q 和输出投影
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, query_dim]
            text_embeds: [batch, text_len, cross_attention_dim]
            image_embeds: [batch, image_tokens, cross_attention_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Query
        q = self.to_q(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 文本注意力
        k_text = self.to_k_text(text_embeds)
        v_text = self.to_v_text(text_embeds)
        k_text = k_text.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_text = v_text.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_text = torch.matmul(q, k_text.transpose(-2, -1)) * self.attention_scale
        attn_text = F.softmax(attn_text, dim=-1)
        out_text = torch.matmul(attn_text, v_text)
        
        # 图像注意力
        k_image = self.to_k_image(image_embeds)
        v_image = self.to_v_image(image_embeds)
        k_image = k_image.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_image = v_image.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_image = torch.matmul(q, k_image.transpose(-2, -1)) * self.attention_scale
        attn_image = F.softmax(attn_image, dim=-1)
        out_image = torch.matmul(attn_image, v_image)
        
        # 合并输出
        out = out_text + self.scale * out_image
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim)
        out = self.to_out(out)
        out = self.dropout(out)
        
        return out


class CLIPVisionEncoder(nn.Module):
    """简化的 CLIP 视觉编码器"""
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        output_dim: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim)
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        batch_size = pixel_values.shape[0]
        
        # Patch embedding
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        hidden_states = []
        for block in self.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_states.append(x)
                
        x = self.norm(x)
        
        # 提取 CLS token 并投影
        cls_output = self.proj(x[:, 0])
        
        if output_hidden_states:
            return cls_output, hidden_states
        return cls_output


class VisionTransformerBlock(nn.Module):
    """Vision Transformer 块"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class IPAdapter(nn.Module):
    """IP-Adapter 主模块"""
    
    def __init__(self, config: IPAdapterConfig):
        super().__init__()
        self.config = config
        
        # 图像投影层
        if config.adapter_type == IPAdapterType.STANDARD:
            self.image_proj = ImageProjection(
                config.image_encoder_dim,
                config.cross_attention_dim,
                config.num_image_tokens
            )
        else:
            self.image_proj = ImageProjectionPlus(
                config.image_encoder_dim,
                config.cross_attention_dim,
                config.num_queries,
                config.num_heads,
                config.num_projection_layers
            )
            
        # 人脸 ID 投影（可选）
        if config.use_face_id:
            self.face_proj = nn.Linear(config.face_embed_dim, config.cross_attention_dim)
        else:
            self.face_proj = None
            
        self.scale = config.scale
        
    def encode_image(
        self,
        image_embeds: torch.Tensor,
        image_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if isinstance(self.image_proj, ImageProjectionPlus):
            return self.image_proj(image_embeds, image_hidden_states)
        return self.image_proj(image_embeds)
    
    def encode_face(self, face_embeds: torch.Tensor) -> Optional[torch.Tensor]:
        if self.face_proj is not None:
            return self.face_proj(face_embeds).unsqueeze(1)
        return None
        
    def forward(
        self,
        image_embeds: torch.Tensor,
        image_hidden_states: Optional[torch.Tensor] = None,
        face_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 编码图像
        ip_tokens = self.encode_image(image_embeds, image_hidden_states)
        
        # 添加人脸特征
        if face_embeds is not None and self.face_proj is not None:
            face_tokens = self.encode_face(face_embeds)
            ip_tokens = torch.cat([ip_tokens, face_tokens], dim=1)
            
        return ip_tokens


class IPAdapterManager:
    """IP-Adapter 管理器 - 用于注入和管理 IP-Adapter"""
    
    def __init__(self, config: IPAdapterConfig):
        self.config = config
        self.ip_adapter = IPAdapter(config)
        self.injected_attentions: Dict[str, IPAdapterCrossAttention] = {}
        
    def inject_to_unet(
        self,
        unet: nn.Module,
        target_modules: Optional[List[str]] = None
    ) -> nn.Module:
        if target_modules is None:
            target_modules = ["attn2", "cross_attn"]
            
        for name, module in unet.named_modules():
            if any(target in name for target in target_modules):
                if hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                    self._inject_attention(unet, name, module)
                    
        return unet
    
    def _inject_attention(self, model: nn.Module, name: str, module: nn.Module):
        # 获取维度信息
        query_dim = module.to_q.in_features if hasattr(module, 'to_q') else 768
        cross_dim = module.to_k.in_features if hasattr(module, 'to_k') else 768
        
        # 创建 IP-Adapter 注意力
        ip_attn = IPAdapterCrossAttention(
            query_dim=query_dim,
            cross_attention_dim=cross_dim,
            num_heads=self.config.num_heads,
            scale=self.config.scale
        )
        
        self.injected_attentions[name] = ip_attn
        
    def get_image_embeds(
        self,
        image_embeds: torch.Tensor,
        image_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.ip_adapter(image_embeds, image_hidden_states)
    
    def save_weights(self, path: str):
        state_dict = {
            'ip_adapter': self.ip_adapter.state_dict(),
            'injected_attentions': {
                name: attn.state_dict()
                for name, attn in self.injected_attentions.items()
            }
        }
        torch.save(state_dict, path)
        
    def load_weights(self, path: str):
        state_dict = torch.load(path, map_location='cpu')
        self.ip_adapter.load_state_dict(state_dict['ip_adapter'])
        for name, attn_state in state_dict['injected_attentions'].items():
            if name in self.injected_attentions:
                self.injected_attentions[name].load_state_dict(attn_state)


def create_ip_adapter(
    adapter_type: str = "standard",
    image_encoder_dim: int = 1024,
    cross_attention_dim: int = 768,
    num_tokens: int = 4,
    scale: float = 1.0
) -> Tuple[IPAdapter, IPAdapterConfig]:
    type_map = {
        "standard": IPAdapterType.STANDARD,
        "plus": IPAdapterType.PLUS,
        "plus_face": IPAdapterType.PLUS_FACE,
        "full_face": IPAdapterType.FULL_FACE
    }
    
    if adapter_type not in type_map:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
        
    config = IPAdapterConfig(
        adapter_type=type_map[adapter_type],
        image_encoder_dim=image_encoder_dim,
        cross_attention_dim=cross_attention_dim,
        num_image_tokens=num_tokens,
        scale=scale,
        use_fine_grained_features=(adapter_type != "standard"),
        use_face_id=("face" in adapter_type)
    )
    
    return IPAdapter(config), config
