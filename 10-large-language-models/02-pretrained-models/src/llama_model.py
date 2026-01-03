"""
LLaMA 模型架构实现

本模块提供 LLaMA (Large Language Model Meta AI) 的生产级实现，
基于论文 "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)。

核心组件：
    - RMSNorm (Root Mean Square Layer Normalization)
    - RoPE (Rotary Position Embedding)
    - SwiGLU 激活函数
    - 分组查询注意力 (GQA)

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "LLaMAConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "LLaMAAttention",
    "LLaMAMLP",
    "LLaMABlock",
    "LLaMA",
]


@dataclass
class LLaMAConfig:
    """LLaMA 模型配置。

    参数：
        vocab_size: 词汇表大小
        max_seq_len: 最大序列长度
        n_layers: Transformer 层数
        n_heads: 查询注意力头数
        n_kv_heads: KV注意力头数 (GQA)，None表示等于n_heads (MHA)
        d_model: 模型隐藏维度
        d_ff: FFN中间维度，默认为 8/3 * d_model (SwiGLU调整)
        norm_eps: RMSNorm epsilon
        rope_theta: RoPE基础频率
        
    预设配置：
        LLaMA-7B:  n_layers=32, n_heads=32, d_model=4096
        LLaMA-13B: n_layers=40, n_heads=40, d_model=5120
        LLaMA-70B: n_layers=80, n_heads=64, n_kv_heads=8, d_model=8192
    """
    vocab_size: int = 32000
    max_seq_len: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    d_model: int = 4096
    d_ff: Optional[int] = None
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_ff is None:
            # SwiGLU 调整: 2/3 * 4 * d_model = 8/3 * d_model
            self.d_ff = int(8 * self.d_model / 3)
            # 对齐到256的倍数
            self.d_ff = ((self.d_ff + 255) // 256) * 256
        
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model必须能被n_heads整除")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads必须能被n_kv_heads整除")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。

    相比LayerNorm，移除了均值中心化，计算更高效。
    
    数学公式：
        RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)。

    将位置信息编码为旋转矩阵，使得注意力分数自然包含相对位置信息。
    
    数学原理：
        f(x, m) = x * cos(m*theta) + rotate(x) * sin(m*theta)
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 预计算频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 预计算cos和sin
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: Tensor) -> Tensor:
    """将张量的后半部分旋转到前半部分。"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """应用旋转位置编码到Q和K。"""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LLaMAAttention(nn.Module):
    """LLaMA 注意力层，支持分组查询注意力 (GQA)。"""

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # KV重复次数
        self.head_dim = config.d_model // config.n_heads

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, 
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T, _ = x.size()

        # 计算Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 应用RoPE
        cos, sin = self.rotary_emb(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV Cache (推理优化)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # GQA: 重复KV头
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 因果掩码
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(T, k.size(2), dtype=torch.bool, device=x.device), 
                diagonal=k.size(2) - T + 1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        else:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.wo(output)

        return output, new_kv_cache


class LLaMAMLP(nn.Module):
    """LLaMA MLP层，使用SwiGLU激活。
    
    SwiGLU(x) = Swish(x * W1) * (x * W3)
    """

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class LLaMABlock(nn.Module):
    """LLaMA Transformer Block (Pre-RMSNorm)。"""

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attention = LLaMAAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = LLaMAMLP(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        h, new_kv_cache = self.attention(
            self.attention_norm(x), attention_mask, kv_cache
        )
        x = x + h
        x = x + self.mlp(self.ffn_norm(x))
        return x, new_kv_cache


class LLaMA(nn.Module):
    """LLaMA 语言模型。

    完整的 LLaMA 架构，包含：
        - Token Embedding
        - N × LLaMA Block (RMSNorm + RoPE + SwiGLU + GQA)
        - RMSNorm
        - LM Head
    """

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([LLaMABlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定
        self.tok_embeddings.weight = self.lm_head.weight

        # 初始化
        self.apply(self._init_weights)
        print(f"LLaMA 模型参数量: {self.get_num_params() / 1e9:.2f}B")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = False,
        past_kv_cache: Optional[list] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[list]]:
        """
        参数：
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len]
            use_cache: 是否使用KV缓存
            past_kv_cache: 历史KV缓存
        """
        h = self.tok_embeddings(input_ids)
        
        new_kv_cache = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_kv_cache[i] if past_kv_cache else None
            h, kv = layer(h, attention_mask, past_kv)
            if use_cache:
                new_kv_cache.append(kv)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss, new_kv_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
    ) -> Tensor:
        """自回归文本生成，支持KV Cache加速。"""
        past_kv_cache = None
        
        for _ in range(max_new_tokens):
            if use_cache and past_kv_cache is not None:
                idx_input = input_ids[:, -1:]
            else:
                idx_input = input_ids
            
            logits, _, past_kv_cache = self(
                idx_input, use_cache=use_cache, past_kv_cache=past_kv_cache
            )
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids
