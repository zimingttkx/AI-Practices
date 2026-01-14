"""
Mixture of Experts (MoE) 稀疏专家混合模型

============================================================
核心思想 (Core Idea)
============================================================
MoE 通过稀疏激活实现条件计算：每个 token 只激活部分专家，
从而在增加模型容量的同时保持计算量不变。

============================================================
数学基础 (Mathematical Foundation)
============================================================
门控函数 (Router):
    G(x) = Softmax(W_g · x)  # 专家权重
    
Top-K 选择:
    TopK(G(x)) = {(i, g_i) | g_i 在前 K 大}

输出计算:
    y = Σ_{i ∈ TopK} g_i · E_i(x)

负载均衡损失:
    L_aux = α · N · Σ_i f_i · P_i
    其中 f_i = 分配给专家 i 的 token 比例
         P_i = 路由到专家 i 的概率均值

============================================================
参考文献 (References)
============================================================
[1] Shazeer, N., et al. (2017). Outrageously Large Neural Networks:
    The Sparsely-Gated Mixture-of-Experts Layer.
[2] Fedus, W., et al. (2022). Switch Transformers: Scaling to
    Trillion Parameter Models with Simple and Efficient Sparsity.
[3] Lepikhin, D., et al. (2021). GShard: Scaling Giant Models with
    Conditional Computation and Automatic Sharding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np

__all__ = [
    "MoEConfig",
    "RouterType",
    "Expert",
    "Router",
    "TopKRouter",
    "SwitchRouter", 
    "ExpertChoiceRouter",
    "MoELayer",
    "MoEBlock",
    "MoETransformerBlock",
    "MoEModel",
    "create_moe_model",
    "compute_load_balancing_loss",
]


class RouterType(Enum):
    """路由器类型。"""
    TOP_K = "top_k"           # Top-K 路由 (标准 MoE)
    SWITCH = "switch"         # Switch 路由 (Top-1)
    EXPERT_CHOICE = "expert_choice"  # 专家选择路由


@dataclass
class MoEConfig:
    """MoE 模型配置。
    
    参数：
        d_model: 模型隐藏维度
        n_layers: 层数
        n_experts: 专家数量
        n_experts_per_tok: 每个 token 激活的专家数 (Top-K)
        expert_capacity_factor: 专家容量因子
        router_type: 路由器类型
        router_jitter: 路由器噪声 (训练时)
        router_z_loss_coef: Z-loss 系数
        router_aux_loss_coef: 辅助损失系数
        d_ff: FFN 中间维度 (默认 4 * d_model)
        n_heads: 注意力头数
        vocab_size: 词表大小
        max_seq_len: 最大序列长度
        dropout: Dropout 率
        bias: 是否使用偏置
    """
    d_model: int = 768
    n_layers: int = 12
    n_experts: int = 8
    n_experts_per_tok: int = 2
    expert_capacity_factor: float = 1.25
    router_type: RouterType = RouterType.TOP_K
    router_jitter: float = 0.0
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.01
    d_ff: Optional[int] = None
    n_heads: int = 12
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dropout: float = 0.1
    bias: bool = True
    
    def __post_init__(self) -> None:
        """验证配置。"""
        if self.d_model <= 0:
            raise ValueError(f"d_model 必须为正数，得到 {self.d_model}")
        if self.n_experts <= 0:
            raise ValueError(f"n_experts 必须为正数，得到 {self.n_experts}")
        if self.n_experts_per_tok <= 0:
            raise ValueError(f"n_experts_per_tok 必须为正数")
        if self.n_experts_per_tok > self.n_experts:
            raise ValueError(f"n_experts_per_tok 不能大于 n_experts")
        
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 Softmax。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU 激活函数。"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class Expert:
    """单个专家 (FFN)。
    
    结构: Linear -> GELU -> Linear
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = True,
        expert_id: int = 0,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.expert_id = expert_id
        
        # 参数初始化
        scale = 1.0 / np.sqrt(d_model)
        self.w1 = np.random.randn(d_ff, d_model) * scale
        self.w2 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff) if bias else None
        self.b2 = np.zeros(d_model) if bias else None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """前向传播。
        
        Args:
            x: [batch, seq_len, d_model] 或 [n_tokens, d_model]
        """
        # Up projection
        h = x @ self.w1.T
        if self.b1 is not None:
            h = h + self.b1
        
        # Activation
        h = gelu(h)
        
        # Down projection
        out = h @ self.w2.T
        if self.b2 is not None:
            out = out + self.b2
        
        return out


class Router:
    """路由器基类。"""
    
    def __init__(self, d_model: int, n_experts: int, jitter: float = 0.0):
        self.d_model = d_model
        self.n_experts = n_experts
        self.jitter = jitter
        
        # 路由权重
        scale = 1.0 / np.sqrt(d_model)
        self.weight = np.random.randn(n_experts, d_model) * scale
    
    def _compute_router_logits(
        self,
        x: np.ndarray,
        training: bool = False,
    ) -> np.ndarray:
        """计算路由 logits。"""
        # [batch * seq_len, d_model] @ [d_model, n_experts]
        logits = x @ self.weight.T
        
        # 训练时添加噪声
        if training and self.jitter > 0:
            noise = np.random.uniform(
                1.0 - self.jitter,
                1.0 + self.jitter,
                size=logits.shape
            )
            logits = logits * noise
        
        return logits
    
    def __call__(
        self,
        x: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """路由计算。
        
        Returns:
            dispatch_mask: 分发掩码
            combine_weights: 组合权重
            aux_info: 辅助信息 (损失等)
        """
        raise NotImplementedError


class TopKRouter(Router):
    """Top-K 路由器。
    
    每个 token 选择 K 个专家。
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        n_experts_per_tok: int = 2,
        capacity_factor: float = 1.25,
        jitter: float = 0.0,
    ):
        super().__init__(d_model, n_experts, jitter)
        self.n_experts_per_tok = n_experts_per_tok
        self.capacity_factor = capacity_factor
    
    def __call__(
        self,
        x: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Top-K 路由。
        
        Args:
            x: [n_tokens, d_model]
        
        Returns:
            expert_indices: [n_tokens, k] 选中的专家索引
            expert_weights: [n_tokens, k] 专家权重
            aux_info: 辅助信息
        """
        n_tokens = x.shape[0]
        
        # 计算路由 logits
        logits = self._compute_router_logits(x, training)  # [n_tokens, n_experts]
        
        # Softmax 得到概率
        probs = softmax(logits, axis=-1)
        
        # Top-K 选择
        k = self.n_experts_per_tok
        top_k_indices = np.argsort(probs, axis=-1)[:, -k:]  # [n_tokens, k]
        
        # 获取 Top-K 权重并重新归一化
        top_k_probs = np.take_along_axis(probs, top_k_indices, axis=-1)
        top_k_weights = top_k_probs / np.sum(top_k_probs, axis=-1, keepdims=True)
        
        # 计算辅助损失
        aux_info = self._compute_aux_loss(probs, top_k_indices)
        
        return top_k_indices, top_k_weights, aux_info
    
    def _compute_aux_loss(
        self,
        probs: np.ndarray,
        indices: np.ndarray,
    ) -> Dict[str, Any]:
        """计算负载均衡辅助损失。"""
        n_tokens = probs.shape[0]
        
        # f_i: 分配给专家 i 的 token 比例
        expert_counts = np.zeros(self.n_experts)
        for idx in indices.flatten():
            expert_counts[idx] += 1
        f = expert_counts / (n_tokens * self.n_experts_per_tok)
        
        # P_i: 路由到专家 i 的平均概率
        P = np.mean(probs, axis=0)
        
        # 负载均衡损失: N * sum(f_i * P_i)
        aux_loss = self.n_experts * np.sum(f * P)
        
        # Z-loss: 鼓励 logits 不要太大
        z_loss = np.mean(np.log(np.sum(np.exp(probs), axis=-1)) ** 2)
        
        return {
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "expert_counts": expert_counts,
            "load_balance": f,
        }


class SwitchRouter(Router):
    """Switch 路由器 (Top-1)。
    
    每个 token 只选择 1 个专家，更高效但可能损失精度。
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        capacity_factor: float = 1.0,
        jitter: float = 0.0,
    ):
        super().__init__(d_model, n_experts, jitter)
        self.capacity_factor = capacity_factor
    
    def __call__(
        self,
        x: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Switch 路由 (Top-1)。"""
        n_tokens = x.shape[0]
        
        logits = self._compute_router_logits(x, training)
        probs = softmax(logits, axis=-1)
        
        # Top-1 选择
        expert_indices = np.argmax(probs, axis=-1, keepdims=True)  # [n_tokens, 1]
        expert_weights = np.take_along_axis(probs, expert_indices, axis=-1)
        
        # 计算辅助损失
        expert_counts = np.zeros(self.n_experts)
        for idx in expert_indices.flatten():
            expert_counts[idx] += 1
        f = expert_counts / n_tokens
        P = np.mean(probs, axis=0)
        aux_loss = self.n_experts * np.sum(f * P)
        
        aux_info = {
            "aux_loss": aux_loss,
            "expert_counts": expert_counts,
            "load_balance": f,
        }
        
        return expert_indices, expert_weights, aux_info


class ExpertChoiceRouter(Router):
    """专家选择路由器。
    
    让专家选择 token，而不是 token 选择专家。
    保证完美的负载均衡。
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        capacity_factor: float = 1.0,
        jitter: float = 0.0,
    ):
        super().__init__(d_model, n_experts, jitter)
        self.capacity_factor = capacity_factor
    
    def __call__(
        self,
        x: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """专家选择路由。"""
        n_tokens = x.shape[0]
        
        logits = self._compute_router_logits(x, training)  # [n_tokens, n_experts]
        
        # 转置: 让专家选择 token
        logits_t = logits.T  # [n_experts, n_tokens]
        
        # 每个专家的容量
        capacity = int(self.capacity_factor * n_tokens / self.n_experts)
        capacity = max(1, capacity)
        
        # 每个专家选择 top-capacity 个 token
        expert_indices = []
        expert_weights = []
        
        for e in range(self.n_experts):
            expert_logits = logits_t[e]  # [n_tokens]
            probs = softmax(expert_logits)
            
            # 选择 top-capacity 个 token
            top_indices = np.argsort(probs)[-capacity:]
            top_weights = probs[top_indices]
            
            expert_indices.append(top_indices)
            expert_weights.append(top_weights)
        
        aux_info = {
            "aux_loss": 0.0,  # 专家选择天然负载均衡
            "capacity": capacity,
        }
        
        return expert_indices, expert_weights, aux_info


class MoELayer:
    """MoE 层。
    
    包含多个专家和一个路由器。
    """
    
    def __init__(self, config: MoEConfig):
        self.config = config
        
        # 创建专家
        self.experts = [
            Expert(
                d_model=config.d_model,
                d_ff=config.d_ff,
                bias=config.bias,
                expert_id=i,
            )
            for i in range(config.n_experts)
        ]
        
        # 创建路由器
        if config.router_type == RouterType.TOP_K:
            self.router = TopKRouter(
                d_model=config.d_model,
                n_experts=config.n_experts,
                n_experts_per_tok=config.n_experts_per_tok,
                capacity_factor=config.expert_capacity_factor,
                jitter=config.router_jitter,
            )
        elif config.router_type == RouterType.SWITCH:
            self.router = SwitchRouter(
                d_model=config.d_model,
                n_experts=config.n_experts,
                capacity_factor=config.expert_capacity_factor,
                jitter=config.router_jitter,
            )
        else:
            self.router = ExpertChoiceRouter(
                d_model=config.d_model,
                n_experts=config.n_experts,
                capacity_factor=config.expert_capacity_factor,
                jitter=config.router_jitter,
            )
    
    def __call__(
        self,
        x: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """前向传播。
        
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model]
            aux_info: 辅助信息
        """
        batch_size, seq_len, d_model = x.shape
        
        # 展平为 [n_tokens, d_model]
        x_flat = x.reshape(-1, d_model)
        n_tokens = x_flat.shape[0]
        
        # 路由
        expert_indices, expert_weights, aux_info = self.router(x_flat, training)
        
        # 初始化输出
        output = np.zeros_like(x_flat)
        
        if isinstance(self.router, ExpertChoiceRouter):
            # 专家选择模式
            for e, (indices, weights) in enumerate(zip(expert_indices, expert_weights)):
                if len(indices) > 0:
                    expert_input = x_flat[indices]
                    expert_output = self.experts[e](expert_input)
                    output[indices] += expert_output * weights[:, np.newaxis]
        else:
            # Token 选择模式 (Top-K / Switch)
            k = expert_indices.shape[1]
            for i in range(n_tokens):
                for j in range(k):
                    expert_idx = expert_indices[i, j]
                    weight = expert_weights[i, j]
                    expert_output = self.experts[expert_idx](x_flat[i:i+1])
                    output[i] += weight * expert_output[0]
        
        # 恢复形状
        output = output.reshape(batch_size, seq_len, d_model)
        
        return output, aux_info


class LayerNorm:
    """Layer Normalization。"""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.weight = np.ones(d_model)
        self.bias = np.zeros(d_model)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps) * self.weight + self.bias


class MultiHeadAttention:
    """多头注意力。"""
    
    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        scale = 1.0 / np.sqrt(d_model)
        self.w_q = np.random.randn(d_model, d_model) * scale
        self.w_k = np.random.randn(d_model, d_model) * scale
        self.w_v = np.random.randn(d_model, d_model) * scale
        self.w_o = np.random.randn(d_model, d_model) * scale
        
        self.b_q = np.zeros(d_model) if bias else None
        self.b_k = np.zeros(d_model) if bias else None
        self.b_v = np.zeros(d_model) if bias else None
        self.b_o = np.zeros(d_model) if bias else None
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        
        # QKV 投影
        q = x @ self.w_q.T + (self.b_q if self.b_q is not None else 0)
        k = x @ self.w_k.T + (self.b_k if self.b_k is not None else 0)
        v = x @ self.w_v.T + (self.b_v if self.b_v is not None else 0)
        
        # 重塑为多头
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 注意力分数
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # 因果掩码
        if mask is None:
            mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        scores = scores + mask
        
        # Softmax
        attn = softmax(scores, axis=-1)
        
        # 输出
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        out = out @ self.w_o.T + (self.b_o if self.b_o is not None else 0)
        
        return out


class MoETransformerBlock:
    """MoE Transformer 块。
    
    结构: Attention -> MoE FFN
    """
    
    def __init__(self, config: MoEConfig, layer_idx: int = 0):
        self.config = config
        self.layer_idx = layer_idx
        
        # 注意力
        self.attn_norm = LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.bias)
        
        # MoE FFN
        self.ffn_norm = LayerNorm(config.d_model)
        self.moe = MoELayer(config)
    
    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # 注意力
        h = self.attn_norm(x)
        h = self.attn(h, mask)
        x = x + h
        
        # MoE FFN
        h = self.ffn_norm(x)
        h, aux_info = self.moe(h, training)
        x = x + h
        
        return x, aux_info


class MoEModel:
    """MoE 语言模型。"""
    
    def __init__(self, config: MoEConfig):
        self.config = config
        
        # 嵌入
        self.embedding = np.random.randn(config.vocab_size, config.d_model) * 0.02
        
        # Transformer 层
        self.layers = [
            MoETransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ]
        
        # 最终归一化
        self.norm = LayerNorm(config.d_model)
        
        # LM Head
        self.lm_head = self.embedding  # 权重共享
    
    def __call__(
        self,
        input_ids: np.ndarray,
        labels: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        x = self.embedding[input_ids]
        
        # Transformer 层
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_info = layer(x, training=training)
            total_aux_loss += aux_info.get("aux_loss", 0.0)
        
        # 归一化
        x = self.norm(x)
        
        # LM Head
        logits = x @ self.lm_head.T
        
        result = {
            "logits": logits,
            "aux_loss": total_aux_loss * self.config.router_aux_loss_coef,
        }
        
        # 计算损失
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            
            # 交叉熵
            probs = softmax(shift_logits, axis=-1)
            log_probs = np.log(probs + 1e-10)
            
            loss = 0.0
            count = 0
            for b in range(batch_size):
                for s in range(seq_len - 1):
                    if shift_labels[b, s] >= 0:
                        loss -= log_probs[b, s, shift_labels[b, s]]
                        count += 1
            
            lm_loss = loss / max(count, 1)
            result["lm_loss"] = lm_loss
            result["loss"] = lm_loss + result["aux_loss"]
        
        return result


def create_moe_model(
    model_size: str = "base",
    n_experts: int = 8,
    **kwargs,
) -> MoEModel:
    """创建预设配置的 MoE 模型。"""
    configs = {
        "small": MoEConfig(d_model=512, n_layers=6, n_heads=8, n_experts=n_experts),
        "base": MoEConfig(d_model=768, n_layers=12, n_heads=12, n_experts=n_experts),
        "large": MoEConfig(d_model=1024, n_layers=24, n_heads=16, n_experts=n_experts),
    }
    
    if model_size not in configs:
        raise ValueError(f"未知模型大小: {model_size}")
    
    config = configs[model_size]
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return MoEModel(config)


def compute_load_balancing_loss(
    router_probs: np.ndarray,
    expert_indices: np.ndarray,
    n_experts: int,
) -> float:
    """计算负载均衡损失。"""
    n_tokens = router_probs.shape[0]
    
    expert_counts = np.zeros(n_experts)
    for idx in expert_indices.flatten():
        expert_counts[idx] += 1
    f = expert_counts / n_tokens
    P = np.mean(router_probs, axis=0)
    
    return n_experts * np.sum(f * P)
