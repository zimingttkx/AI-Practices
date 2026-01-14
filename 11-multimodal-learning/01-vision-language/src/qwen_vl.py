"""
Qwen-VL (Qwen Vision-Language) 实现

Qwen-VL 是阿里巴巴通义千问团队提出的视觉语言模型。

=== 核心思想 ===

Qwen-VL 的核心创新：

1. 位置感知视觉-语言适配器:
   - 使用单层交叉注意力压缩视觉特征
   - 保留位置信息用于视觉定位任务
   - 将任意分辨率图像压缩为固定数量的 token

2. 多任务预训练:
   - 图像描述生成
   - 视觉问答 (VQA)
   - 视觉定位 (Grounding)
   - 文本识别 (OCR)

3. 高分辨率支持:
   - 支持动态分辨率输入
   - 使用 2D 绝对位置编码

=== 数学基础 ===

视觉-语言适配器 (Resampler):
    给定视觉特征 V ∈ R^(N×D_v) 和可学习查询 Q ∈ R^(M×D):
    
    Output = CrossAttention(Q, V, V)
           = softmax(Q * K^T / √d) * V
    
    其中 M << N，实现特征压缩

位置编码:
    对于图像位置 (x, y):
    PE(x, y) = [sin(x/10000^(2i/d)), cos(x/10000^(2i/d)),
                sin(y/10000^(2i/d)), cos(y/10000^(2i/d))]

=== 架构组件 ===

1. 视觉编码器:
   - ViT-G/14 (OpenCLIP)
   - 输出 256 个视觉 token

2. 视觉-语言适配器:
   - 单层交叉注意力
   - 256 个可学习查询
   - 2D 位置编码

3. 语言模型:
   - Qwen-7B/14B
   - 支持中英双语

=== 参考文献 ===

1. Qwen-VL 原始论文:
   Bai et al. "Qwen-VL: A Versatile Vision-Language Model for Understanding,
   Localization, Text Reading, and Beyond" 2023
   https://arxiv.org/abs/2308.12966

2. Qwen2-VL:
   Wang et al. "Qwen2-VL: Enhancing Vision-Language Model's Perception of
   the World at Any Resolution" 2024
   https://arxiv.org/abs/2409.12191

=== 核心组件 ===

    - QwenVLConfig: 模型配置
    - VisualResampler: 视觉-语言适配器
    - QwenVLVisionEncoder: 视觉编码器
    - QwenVLAttention: 注意力模块
    - QwenVLMLP: MLP 模块
    - QwenVLBlock: Transformer 块
    - QwenVL: 完整模型
    - create_qwen_vl_model: 模型工厂函数
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class QwenVLModelSize(Enum):
    """Qwen-VL 模型尺寸"""
    QWEN_VL_7B = "qwen-vl-7b"
    QWEN_VL_14B = "qwen-vl-14b"
    QWEN2_VL_2B = "qwen2-vl-2b"
    QWEN2_VL_7B = "qwen2-vl-7b"


@dataclass
class QwenVLConfig:
    """Qwen-VL 模型配置。

    参数：
        # 视觉编码器配置
        image_size: 输入图像大小
        patch_size: 图像分块大小
        vision_layers: 视觉编码器层数
        vision_width: 视觉编码器隐藏层维度
        vision_heads: 视觉编码器注意力头数
        
        # 视觉适配器配置
        num_query_tokens: 查询 token 数量
        resampler_heads: Resampler 注意力头数
        
        # 语言模型配置
        vocab_size: 词汇表大小
        hidden_size: LLM 隐藏层维度
        num_layers: LLM 层数
        num_heads: LLM 注意力头数
        num_kv_heads: KV 头数 (GQA)
        intermediate_size: FFN 中间层维度
        max_position_embeddings: 最大位置编码
        
        # 通用配置
        dropout: Dropout 概率
        use_gradient_checkpointing: 是否使用梯度检查点
        rope_theta: RoPE 基础频率
    """

    # 视觉编码器配置
    image_size: int = 448
    patch_size: int = 14
    vision_layers: int = 48
    vision_width: int = 1664
    vision_heads: int = 16

    # 视觉适配器配置
    num_query_tokens: int = 256
    resampler_heads: int = 16

    # 语言模型配置
    vocab_size: int = 151936
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    intermediate_size: int = 11008
    max_position_embeddings: int = 8192

    # 通用配置
    dropout: float = 0.0
    use_gradient_checkpointing: bool = False
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self):
        assert self.image_size % self.patch_size == 0
        assert self.hidden_size % self.num_heads == 0


class RMSNorm(nn.Module):
    """RMS Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, 
    cos: torch.Tensor, sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class QwenVLPatchEmbedding(nn.Module):
    """Qwen-VL 图像分块嵌入"""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.num_patches = (config.image_size // config.patch_size) ** 2

        self.projection = nn.Conv2d(
            3, config.vision_width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_width))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.vision_width)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.position_embedding
        return x


class VisionAttention(nn.Module):
    """视觉编码器注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class VisionMLP(nn.Module):
    """视觉编码器 MLP"""
    
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class VisionEncoderBlock(nn.Module):
    """视觉编码器 Transformer 块"""
    
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = VisionAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = VisionMLP(d_model, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class QwenVLVisionEncoder(nn.Module):
    """Qwen-VL 视觉编码器"""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.config = config
        self.patch_embed = QwenVLPatchEmbedding(config)
        
        self.blocks = nn.ModuleList([
            VisionEncoderBlock(config.vision_width, config.vision_heads, 4.0, config.dropout)
            for _ in range(config.vision_layers)
        ])
        
        self.ln_post = nn.LayerNorm(config.vision_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)


class VisualResampler(nn.Module):
    """视觉-语言适配器 (Resampler)
    
    使用交叉注意力将视觉特征压缩为固定数量的 token。
    """
    
    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.num_queries = config.num_query_tokens
        self.hidden_size = config.hidden_size
        
        # 可学习查询
        self.query = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))
        
        # 视觉特征投影
        self.visual_proj = nn.Linear(config.vision_width, config.hidden_size, bias=False)
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_size, config.resampler_heads, dropout=config.dropout, batch_first=True
        )
        
        self.ln_q = nn.LayerNorm(config.hidden_size)
        self.ln_kv = nn.LayerNorm(config.hidden_size)
        self.ln_post = nn.LayerNorm(config.hidden_size)
        
        # 2D 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, config.hidden_size))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.query, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        batch_size = visual_features.shape[0]
        
        # 投影视觉特征
        visual_features = self.visual_proj(visual_features)
        
        # 添加位置编码
        seq_len = visual_features.shape[1]
        visual_features = visual_features + self.pos_embed[:, :seq_len, :]
        
        # 扩展查询
        query = self.query.expand(batch_size, -1, -1)
        
        # 交叉注意力
        query = self.ln_q(query)
        kv = self.ln_kv(visual_features)
        output, _ = self.cross_attn(query, kv, kv)
        output = self.ln_post(output)
        
        return output


class QwenVLAttention(nn.Module):
    """Qwen-VL 注意力模块"""
    
    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_key_value_groups = config.num_heads // config.num_kv_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)


class QwenVLMLP(nn.Module):
    """Qwen-VL MLP 模块"""
    
    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenVLBlock(nn.Module):
    """Qwen-VL Transformer 块"""
    
    def __init__(self, config: QwenVLConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = QwenVLAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = QwenVLMLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class QwenVL(nn.Module):
    """Qwen-VL 完整模型"""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.config = config
        
        self.vision_encoder = QwenVLVisionEncoder(config)
        self.visual_resampler = VisualResampler(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([
            QwenVLBlock(config, i) for i in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        vision_features = self.vision_encoder(images)
        return self.visual_resampler(vision_features)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        
        if images is not None:
            vision_features = self.get_vision_features(images)
            num_vision_tokens = vision_features.shape[1]
            
            if image_positions is not None:
                for i in range(batch_size):
                    start_pos = image_positions[i].item()
                    hidden_states[i, start_pos:start_pos + num_vision_tokens] = vision_features[i]
            else:
                hidden_states[:, :num_vision_tokens] = vision_features
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        causal_mask = self._make_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask + causal_mask
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1), ignore_index=-100
            )
        
        return {"logits": logits, "loss": loss, "hidden_states": hidden_states}

    def _make_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128, temperature: float = 1.0, top_p: float = 0.9, do_sample: bool = True,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids, images)
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.gather(sorted_indices, -1, torch.multinomial(sorted_probs, 1))
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if (next_token == 151643).all():  # Qwen EOS token
                break
        return input_ids


def create_qwen_vl_model(model_size: str = "7b") -> QwenVL:
    """创建 Qwen-VL 模型"""
    configs = {
        "7b": QwenVLConfig(
            image_size=448, patch_size=14, vision_layers=48, vision_width=1664, vision_heads=16,
            num_query_tokens=256, hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=32,
            intermediate_size=11008
        ),
        "14b": QwenVLConfig(
            image_size=448, patch_size=14, vision_layers=48, vision_width=1664, vision_heads=16,
            num_query_tokens=256, hidden_size=5120, num_layers=40, num_heads=40, num_kv_heads=40,
            intermediate_size=13696
        ),
    }
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    return QwenVL(configs[model_size])
