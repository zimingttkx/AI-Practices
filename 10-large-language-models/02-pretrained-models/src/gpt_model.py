"""
GPT 模型架构实现

本模块提供 GPT (Generative Pre-trained Transformer) 的生产级实现，
基于论文 "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)。

核心组件：
    - 因果自注意力 (Causal Self-Attention)
    - GPT Block (Pre-LN Transformer Decoder)
    - GPT 语言模型

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
    "GPTConfig",
    "CausalSelfAttention",
    "GPTBlock",
    "GPT",
]


@dataclass
class GPTConfig:
    """GPT 模型配置。

    参数：
        vocab_size: 词汇表大小
        max_seq_len: 最大序列长度
        n_layers: Transformer 层数
        n_heads: 注意力头数
        d_model: 模型隐藏维度
        d_ff: 前馈网络中间维度，默认为 4 * d_model
        dropout: Dropout 概率
        bias: 是否使用偏置项
        
    预设配置：
        GPT-2 Small:  n_layers=12, n_heads=12, d_model=768
        GPT-2 Medium: n_layers=24, n_heads=16, d_model=1024
        GPT-2 Large:  n_layers=36, n_heads=20, d_model=1280
        GPT-2 XL:     n_layers=48, n_heads=25, d_model=1600
    """
    vocab_size: int = 50257
    max_seq_len: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: Optional[int] = None
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除"
            )


class CausalSelfAttention(nn.Module):
    """因果自注意力机制。

    实现带因果掩码的多头自注意力，确保每个位置只能关注
    当前及之前的位置，用于自回归语言建模。

    数学原理：
        Attention(Q, K, V) = softmax(QK^T / √d_k + M) V
        
        其中 M 是因果掩码：
        M[i,j] = 0 if j <= i else -∞
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.n_heads
        self.dropout = config.dropout

        # 合并的 QKV 投影
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码 (下三角矩阵)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(
        self, 
        x: Tensor, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        参数：
            x: 输入张量 [batch, seq_len, d_model]
            attention_mask: 可选的注意力掩码 [batch, seq_len]
            
        返回：
            输出张量 [batch, seq_len, d_model]
        """
        B, T, C = x.size()

        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # 重塑为多头格式: [B, n_heads, T, d_k]
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        
        # 应用因果掩码
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # 应用额外的注意力掩码（如padding mask）
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, 1, 1, T)
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 加权求和
        y = att @ v
        
        # 合并多头
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """GPT 前馈网络。

    结构：Linear → GELU → Linear → Dropout
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):
    """GPT Transformer Block (Pre-LN)。

    结构：
        x → LN → Attention → + → LN → MLP → +
            └─────────────────┘   └──────────┘
                残差连接              残差连接
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self, 
        x: Tensor, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT 语言模型。

    完整的 GPT 架构，包含：
        - Token Embedding
        - Position Embedding  
        - N × GPT Block
        - Layer Norm
        - LM Head (与 Token Embedding 权重绑定)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.max_seq_len, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        
        # 语言模型头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重绑定
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)
        
        # 特殊初始化：残差投影层
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        print(f"GPT 模型参数量: {self.get_num_params() / 1e6:.2f}M")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        参数：
            input_ids: 输入token ID [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            labels: 标签，用于计算损失 [batch, seq_len]
            
        返回：
            logits: 预测logits [batch, seq_len, vocab_size]
            loss: 如果提供labels，返回交叉熵损失
        """
        device = input_ids.device
        B, T = input_ids.size()
        
        assert T <= self.config.max_seq_len, \
            f"序列长度 {T} 超过最大长度 {self.config.max_seq_len}"

        # 位置索引
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # Embedding
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, attention_mask)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tensor:
        """自回归文本生成。

        参数：
            input_ids: 输入token ID [batch, seq_len]
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
            
        返回：
            生成的token ID [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]
            
            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-P 采样
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

            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids
