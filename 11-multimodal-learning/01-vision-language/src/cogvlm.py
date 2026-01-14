"""
CogVLM (Cognitive Visual Language Model) 实现

CogVLM 是清华大学和智谱 AI 提出的视觉语言模型，通过视觉专家模块实现深度视觉-语言融合。

=== 核心思想 ===

CogVLM 的核心创新是 Visual Expert 模块：

1. 视觉专家 (Visual Expert):
   - 在每个 Transformer 层添加独立的视觉专家 QKV 和 FFN
   - 视觉 token 使用视觉专家处理，文本 token 使用原始 LLM 参数
   - 保持 LLM 原有能力的同时增强视觉理解

2. 架构特点:
   - 冻结原始 LLM 参数
   - 仅训练视觉编码器和视觉专家模块
   - 支持高分辨率图像输入 (1344x1344)

=== 数学基础 ===

视觉专家注意力:
    对于视觉 token v 和文本 token t:
    
    Q_v = W_q^v * v,  K_v = W_k^v * v,  V_v = W_v^v * v  (视觉专家)
    Q_t = W_q * t,    K_t = W_k * t,    V_t = W_v * t    (原始 LLM)
    
    Attention([v; t]) = softmax([Q_v; Q_t] * [K_v; K_t]^T / √d) * [V_v; V_t]

视觉专家 FFN:
    FFN_v(x) = W_2^v * GELU(W_1^v * x)  (视觉 token)
    FFN_t(x) = W_2 * GELU(W_1 * x)      (文本 token)

=== 架构组件 ===

1. 视觉编码器:
   - EVA-CLIP ViT-E (4.4B 参数)
   - 输出 1225 个视觉 token (35x35)

2. 视觉专家模块:
   - 每层独立的 QKV 投影
   - 每层独立的 FFN
   - 与 LLM 层数相同

3. 语言模型:
   - 基于 Vicuna-7B/13B
   - 冻结原始参数

=== 参考文献 ===

1. CogVLM 原始论文:
   Wang et al. "CogVLM: Visual Expert for Pretrained Language Models" 2023
   https://arxiv.org/abs/2311.03079

2. CogVLM2:
   Hong et al. "CogVLM2: Visual Language Models for Image and Video Understanding" 2024
   https://arxiv.org/abs/2408.16500

3. EVA-CLIP:
   Sun et al. "EVA-CLIP: Improved Training Techniques for CLIP at Scale" 2023
   https://arxiv.org/abs/2303.15389

=== 核心组件 ===

    - CogVLMConfig: 模型配置
    - VisualExpertAttention: 视觉专家注意力
    - VisualExpertMLP: 视觉专家 FFN
    - CogVLMBlock: CogVLM Transformer 块
    - CogVLMVisionEncoder: 视觉编码器
    - CogVLM: 完整模型
    - create_cogvlm_model: 模型工厂函数
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


class CogVLMModelSize(Enum):
    """CogVLM 模型尺寸"""
    COGVLM_7B = "cogvlm-7b"
    COGVLM_17B = "cogvlm-17b"
    COGVLM2_19B = "cogvlm2-19b"


@dataclass
class CogVLMConfig:
    """CogVLM 模型配置。

    参数：
        # 视觉编码器配置
        image_size: 输入图像大小
        patch_size: 图像分块大小
        vision_layers: 视觉编码器层数
        vision_width: 视觉编码器隐藏层维度
        vision_heads: 视觉编码器注意力头数
        
        # 语言模型配置
        vocab_size: 词汇表大小
        hidden_size: LLM 隐藏层维度
        num_layers: LLM 层数
        num_heads: LLM 注意力头数
        num_kv_heads: KV 头数 (GQA)
        intermediate_size: FFN 中间层维度
        max_position_embeddings: 最大位置编码
        
        # 视觉专家配置
        visual_expert_intermediate_size: 视觉专家 FFN 中间层维度
        
        # 通用配置
        dropout: Dropout 概率
        use_gradient_checkpointing: 是否使用梯度检查点
        rope_theta: RoPE 基础频率
    """

    # 视觉编码器配置
    image_size: int = 1344
    patch_size: int = 14
    vision_layers: int = 63
    vision_width: int = 1792
    vision_heads: int = 16

    # 语言模型配置
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32  # GQA
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096

    # 视觉专家配置
    visual_expert_intermediate_size: int = 11008

    # 通用配置
    dropout: float = 0.0
    use_gradient_checkpointing: bool = False
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        assert self.image_size % self.patch_size == 0
        assert self.hidden_size % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0


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
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 预计算频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 预计算 cos/sin 缓存
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
    """旋转一半的维度"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用 RoPE"""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class CogVLMPatchEmbedding(nn.Module):
    """CogVLM 图像分块嵌入"""

    def __init__(self, config: CogVLMConfig):
        super().__init__()
        self.config = config
        self.num_patches = (config.image_size // config.patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=config.vision_width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_width))
        
        # 位置嵌入
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.vision_width)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        
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
        x = self.proj(x)
        
        return x


class VisionMLP(nn.Module):
    """视觉编码器 MLP"""
    
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


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


class CogVLMVisionEncoder(nn.Module):
    """CogVLM 视觉编码器 (EVA-CLIP 风格)"""

    def __init__(self, config: CogVLMConfig):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

        self.patch_embed = CogVLMPatchEmbedding(config)
        
        self.blocks = nn.ModuleList([
            VisionEncoderBlock(
                d_model=config.vision_width,
                num_heads=config.vision_heads,
                mlp_ratio=4.0,
                dropout=config.dropout
            )
            for _ in range(config.vision_layers)
        ])

        self.ln_post = nn.LayerNorm(config.vision_width)
        
        # 投影到 LLM 隐藏维度
        self.projection = nn.Linear(config.vision_width, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_post(x)
        x = self.projection(x)

        return x


class VisualExpertAttention(nn.Module):
    """视觉专家注意力模块
    
    核心创新：为视觉 token 和文本 token 使用不同的 QKV 投影。
    """
    
    def __init__(self, config: CogVLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_key_value_groups = config.num_heads // config.num_kv_heads
        
        # 文本 QKV (原始 LLM 参数)
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        
        # 视觉专家 QKV
        self.vision_q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.vision_k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.vision_v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, 
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # 分别计算视觉和文本的 QKV
        if vision_mask is not None:
            # 视觉 token
            vision_hidden = hidden_states * vision_mask.unsqueeze(-1)
            text_hidden = hidden_states * (~vision_mask).unsqueeze(-1)
            
            q = self.q_proj(text_hidden) + self.vision_q_proj(vision_hidden)
            k = self.k_proj(text_hidden) + self.vision_k_proj(vision_hidden)
            v = self.v_proj(text_hidden) + self.vision_v_proj(vision_hidden)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        
        # 重塑
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # GQA: 扩展 KV
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # 注意力计算
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output


class VisualExpertMLP(nn.Module):
    """视觉专家 MLP 模块"""
    
    def __init__(self, config: CogVLMConfig):
        super().__init__()
        # 文本 FFN (原始 LLM)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        # 视觉专家 FFN
        self.vision_gate_proj = nn.Linear(config.hidden_size, config.visual_expert_intermediate_size, bias=False)
        self.vision_up_proj = nn.Linear(config.hidden_size, config.visual_expert_intermediate_size, bias=False)
        self.vision_down_proj = nn.Linear(config.visual_expert_intermediate_size, config.hidden_size, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        vision_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if vision_mask is not None:
            vision_x = x * vision_mask.unsqueeze(-1)
            text_x = x * (~vision_mask).unsqueeze(-1)
            
            # 文本 FFN
            text_out = self.down_proj(F.silu(self.gate_proj(text_x)) * self.up_proj(text_x))
            # 视觉专家 FFN
            vision_out = self.vision_down_proj(
                F.silu(self.vision_gate_proj(vision_x)) * self.vision_up_proj(vision_x)
            )
            return text_out + vision_out
        else:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CogVLMBlock(nn.Module):
    """CogVLM Transformer 块 (带视觉专家)"""
    
    def __init__(self, config: CogVLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VisualExpertAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VisualExpertMLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask, position_ids, vision_mask
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, vision_mask)
        hidden_states = residual + hidden_states
        
        return hidden_states


class CogVLM(nn.Module):
    """CogVLM 完整模型"""

    def __init__(self, config: CogVLMConfig):
        super().__init__()
        self.config = config

        # 视觉编码器
        self.vision_encoder = CogVLMVisionEncoder(config)
        
        # 文本嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            CogVLMBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """获取视觉特征"""
        return self.vision_encoder(images)

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # 文本嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 如果有图像，插入视觉特征
        if images is not None:
            vision_features = self.get_vision_features(images)
            num_vision_tokens = vision_features.shape[1]
            
            # 创建视觉掩码
            if vision_mask is None:
                vision_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
                vision_mask[:, :num_vision_tokens] = True
            
            # 替换视觉位置的嵌入
            hidden_states = hidden_states.clone()
            for i in range(batch_size):
                vision_positions = vision_mask[i].nonzero(as_tuple=True)[0]
                if len(vision_positions) > 0:
                    hidden_states[i, vision_positions[:num_vision_tokens]] = vision_features[i]
        
        # 创建因果注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        causal_mask = self._make_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask + causal_mask
        
        # 位置 ID
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Transformer 层
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer, hidden_states, attention_mask, position_ids, vision_mask,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, position_ids, vision_mask)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"logits": logits, "loss": loss, "hidden_states": hidden_states}

    def _make_causal_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """自回归生成"""
        batch_size = input_ids.shape[0]
        
        # 获取视觉特征
        vision_features = None
        if images is not None:
            vision_features = self.get_vision_features(images)
        
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids, images)
            next_token_logits = outputs["logits"][:, -1, :]
            
            if do_sample:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Top-p 采样
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = torch.gather(sorted_indices, -1, next_token)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # EOS 检查
            if (next_token == 2).all():  # 假设 EOS token ID = 2
                break
        
        return input_ids


def create_cogvlm_model(model_size: str = "7b") -> CogVLM:
    """创建 CogVLM 模型"""
    configs = {
        "7b": CogVLMConfig(
            image_size=490, patch_size=14, vision_layers=24, vision_width=1024, vision_heads=16,
            hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=32,
            intermediate_size=11008, visual_expert_intermediate_size=11008
        ),
        "17b": CogVLMConfig(
            image_size=490, patch_size=14, vision_layers=48, vision_width=1792, vision_heads=16,
            hidden_size=6144, num_layers=48, num_heads=48, num_kv_heads=48,
            intermediate_size=16384, visual_expert_intermediate_size=16384
        ),
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    return CogVLM(configs[model_size])
