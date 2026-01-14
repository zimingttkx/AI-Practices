"""
Mamba: 选择性状态空间模型 (Selective State Space Model)

============================================================
核心思想 (Core Idea)
============================================================
Mamba 是一种基于状态空间模型(SSM)的序列建模架构，通过选择性机制
实现输入依赖的参数化，在保持线性时间复杂度的同时获得与 Transformer
相当的建模能力。

============================================================
数学基础 (Mathematical Foundation)
============================================================
连续时间状态空间模型：
    h'(t) = A h(t) + B x(t)
    y(t) = C h(t) + D x(t)

离散化 (Zero-Order Hold):
    A_bar = exp(Δ A)
    B_bar = (Δ A)^{-1} (exp(Δ A) - I) · Δ B ≈ Δ B

选择性机制 (Mamba 核心创新):
    Δ, B, C = f(x)  # 输入依赖的参数
    
    使得模型能够：
    1. 选择性地记住或遗忘信息
    2. 根据内容调整时间尺度
    3. 实现类似注意力的上下文感知

============================================================
算法复杂度 (Complexity Analysis)
============================================================
- 时间复杂度: O(L) - 线性于序列长度
- 空间复杂度: O(L·D·N) - L序列长度, D模型维度, N状态维度
- 对比 Transformer: O(L²) → O(L)

============================================================
参考文献 (References)
============================================================
[1] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling 
    with Selective State Spaces. arXiv:2312.00752
[2] Gu, A., et al. (2022). Efficiently Modeling Long Sequences with 
    Structured State Spaces. ICLR 2022.
[3] Dao, T., et al. (2024). Transformers are SSMs: Generalized Models 
    and Efficient Algorithms Through Structured State Space Duality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "MambaConfig",
    "SelectiveSSM",
    "MambaBlock",
    "MambaLayer",
    "MambaModel",
    "MambaForCausalLM",
    "create_mamba_model",
    "discretize_ssm",
    "selective_scan",
    "causal_conv1d",
]


@dataclass
class MambaConfig:
    """Mamba 模型配置。
    
    参数：
        d_model: 模型隐藏维度
        n_layers: Mamba 层数
        d_state: SSM 状态维度 (N)
        d_conv: 卷积核大小
        expand: 内部维度扩展因子
        dt_rank: Δ 投影的秩 (默认 "auto" = d_model // 16)
        dt_min: Δ 最小值
        dt_max: Δ 最大值
        dt_init: Δ 初始化方式 ("constant", "random")
        dt_scale: Δ 缩放因子
        dt_init_floor: Δ 初始化下限
        bias: 是否使用偏置
        conv_bias: 卷积是否使用偏置
        vocab_size: 词表大小
        pad_vocab_size_multiple: 词表大小对齐
        initializer_range: 参数初始化范围
        residual_in_fp32: 残差连接是否使用 FP32
        rms_norm_eps: RMSNorm epsilon
    """
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    vocab_size: int = 50257
    pad_vocab_size_multiple: int = 8
    initializer_range: float = 0.02
    residual_in_fp32: bool = True
    rms_norm_eps: float = 1e-5
    
    def __post_init__(self) -> None:
        """验证配置参数。"""
        if self.d_model <= 0:
            raise ValueError(f"d_model 必须为正数，得到 {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers 必须为正数，得到 {self.n_layers}")
        if self.d_state <= 0:
            raise ValueError(f"d_state 必须为正数，得到 {self.d_state}")
        if self.d_conv <= 0:
            raise ValueError(f"d_conv 必须为正数，得到 {self.d_conv}")
        if self.expand <= 0:
            raise ValueError(f"expand 必须为正数，得到 {self.expand}")
        
        # 自动计算 dt_rank
        if self.dt_rank == "auto":
            self.dt_rank = max(1, self.d_model // 16)
        
        # 对齐词表大小
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple 
                - self.vocab_size % self.pad_vocab_size_multiple
            )
    
    @property
    def d_inner(self) -> int:
        """内部维度 = d_model * expand。"""
        return self.d_model * self.expand


def discretize_ssm(
    A: np.ndarray,
    B: np.ndarray,
    delta: np.ndarray,
    method: str = "zoh"
) -> Tuple[np.ndarray, np.ndarray]:
    """离散化连续时间 SSM 参数。
    
    将连续时间状态空间模型离散化为离散时间形式。
    
    连续时间: h'(t) = A h(t) + B x(t)
    离散时间: h[k] = A_bar h[k-1] + B_bar x[k]
    
    Args:
        A: 状态转移矩阵 [d_inner, d_state]
        B: 输入矩阵 [batch, seq_len, d_state]
        delta: 时间步长 [batch, seq_len, d_inner]
        method: 离散化方法 ("zoh", "bilinear", "euler")
    
    Returns:
        A_bar: 离散化状态矩阵
        B_bar: 离散化输入矩阵
    
    数学推导 (Zero-Order Hold):
        A_bar = exp(Δ · A)
        B_bar = (Δ · A)^{-1} (exp(Δ · A) - I) · Δ · B
              ≈ Δ · B  (当 Δ 较小时)
    """
    if method == "zoh":
        # Zero-Order Hold 离散化
        # A_bar = exp(delta * A)
        # 对于对角矩阵 A，exp(delta * A) = diag(exp(delta * a_i))
        delta_A = np.einsum("bld,dn->bldn", delta, A)
        A_bar = np.exp(delta_A)
        
        # B_bar ≈ delta * B (简化近似)
        # 完整形式: B_bar = (exp(delta*A) - I) / A * B
        delta_B = np.einsum("bld,bln->bldn", delta, B)
        B_bar = delta_B
        
    elif method == "bilinear":
        # 双线性变换 (Tustin)
        # A_bar = (I + Δ/2 · A) / (I - Δ/2 · A)
        delta_A = np.einsum("bld,dn->bldn", delta, A) / 2
        A_bar = (1 + delta_A) / (1 - delta_A)
        delta_B = np.einsum("bld,bln->bldn", delta, B)
        B_bar = delta_B / (1 - delta_A)
        
    elif method == "euler":
        # 前向欧拉
        # A_bar = I + Δ · A
        delta_A = np.einsum("bld,dn->bldn", delta, A)
        A_bar = 1 + delta_A
        delta_B = np.einsum("bld,bln->bldn", delta, B)
        B_bar = delta_B
        
    else:
        raise ValueError(f"未知离散化方法: {method}")
    
    return A_bar, B_bar


def selective_scan(
    x: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: Optional[np.ndarray] = None,
) -> np.ndarray:
    """选择性扫描算法 (Selective Scan)。
    
    这是 Mamba 的核心算法，实现输入依赖的状态空间模型。
    
    Args:
        x: 输入序列 [batch, seq_len, d_inner]
        delta: 时间步长 [batch, seq_len, d_inner]
        A: 状态矩阵 [d_inner, d_state]
        B: 输入矩阵 [batch, seq_len, d_state]
        C: 输出矩阵 [batch, seq_len, d_state]
        D: 跳跃连接 [d_inner] (可选)
    
    Returns:
        y: 输出序列 [batch, seq_len, d_inner]
    
    算法流程:
        for k in range(seq_len):
            h[k] = A_bar[k] * h[k-1] + B_bar[k] * x[k]
            y[k] = C[k] @ h[k] + D * x[k]
    
    复杂度: O(L·D·N) 其中 L=seq_len, D=d_inner, N=d_state
    """
    batch_size, seq_len, d_inner = x.shape
    d_state = A.shape[1]
    
    # 离散化
    A_bar, B_bar = discretize_ssm(A, B, delta)
    
    # 初始化状态
    h = np.zeros((batch_size, d_inner, d_state))
    
    # 输出序列
    y = np.zeros_like(x)
    
    # 顺序扫描 (可并行化为 parallel scan)
    for k in range(seq_len):
        # 状态更新: h[k] = A_bar[k] * h[k-1] + B_bar[k] * x[k]
        h = A_bar[:, k, :, :] * h + B_bar[:, k, :, :] * x[:, k, :, np.newaxis]
        
        # 输出计算: y[k] = C[k] @ h[k]
        y[:, k, :] = np.einsum("bdn,bn->bd", h, C[:, k, :])
    
    # 跳跃连接
    if D is not None:
        y = y + x * D
    
    return y


def causal_conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """因果一维卷积。
    
    Args:
        x: 输入 [batch, d_inner, seq_len]
        weight: 卷积核 [d_inner, 1, d_conv]
        bias: 偏置 [d_inner]
    
    Returns:
        y: 输出 [batch, d_inner, seq_len]
    """
    batch_size, d_inner, seq_len = x.shape
    d_conv = weight.shape[2]
    
    # 左填充以实现因果卷积
    x_padded = np.pad(x, ((0, 0), (0, 0), (d_conv - 1, 0)), mode='constant')
    
    # 深度可分离卷积
    y = np.zeros_like(x)
    for i in range(d_inner):
        for j in range(seq_len):
            y[:, i, j] = np.sum(
                x_padded[:, i, j:j + d_conv] * weight[i, 0, :],
                axis=-1
            )
    
    if bias is not None:
        y = y + bias[:, np.newaxis]
    
    return y


class RMSNorm:
    """Root Mean Square Layer Normalization。
    
    RMSNorm 相比 LayerNorm 去掉了均值中心化，计算更高效。
    
    公式: y = x / RMS(x) * g
    其中: RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.weight = np.ones(d_model)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class SelectiveSSM:
    """选择性状态空间模型 (Selective SSM)。
    
    Mamba 的核心创新：使参数 Δ, B, C 依赖于输入，
    从而实现内容感知的序列建模。
    
    Args:
        d_model: 模型维度
        d_state: 状态维度 N
        d_conv: 卷积核大小
        expand: 扩展因子
        dt_rank: Δ 投影秩
        dt_min: Δ 最小值
        dt_max: Δ 最大值
        dt_init: Δ 初始化方式
        dt_scale: Δ 缩放因子
        bias: 是否使用偏置
        conv_bias: 卷积是否使用偏置
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 48,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank
        
        # 输入投影: x -> (z, x_proj)
        # z 用于门控，x_proj 用于 SSM
        self.in_proj_weight = np.random.randn(
            self.d_inner * 2, d_model
        ) * 0.02
        self.in_proj_bias = np.zeros(self.d_inner * 2) if bias else None
        
        # 因果卷积
        self.conv1d_weight = np.random.randn(
            self.d_inner, 1, d_conv
        ) * 0.02
        self.conv1d_bias = np.zeros(self.d_inner) if conv_bias else None
        
        # SSM 参数投影: x -> (Δ, B, C)
        self.x_proj_weight = np.random.randn(
            dt_rank + d_state * 2, self.d_inner
        ) * 0.02
        
        # Δ 投影
        self.dt_proj_weight = self._init_dt_proj(
            dt_rank, self.d_inner, dt_init, dt_min, dt_max, dt_scale
        )
        self.dt_proj_bias = self._init_dt_bias(
            self.d_inner, dt_min, dt_max, dt_init
        )
        
        # A 参数 (对角矩阵，使用 log 参数化)
        # A = -exp(A_log) 保证稳定性
        self.A_log = self._init_A(self.d_inner, d_state)
        
        # D 参数 (跳跃连接)
        self.D = np.ones(self.d_inner)
        
        # 输出投影
        self.out_proj_weight = np.random.randn(
            d_model, self.d_inner
        ) * 0.02
        self.out_proj_bias = np.zeros(d_model) if bias else None
    
    def _init_dt_proj(
        self,
        dt_rank: int,
        d_inner: int,
        dt_init: str,
        dt_min: float,
        dt_max: float,
        dt_scale: float,
    ) -> np.ndarray:
        """初始化 Δ 投影权重。"""
        if dt_init == "constant":
            dt_init_std = dt_rank ** -0.5 * dt_scale
        elif dt_init == "random":
            dt_init_std = (dt_rank ** -0.5) * dt_scale
        else:
            raise ValueError(f"未知 dt_init: {dt_init}")
        
        return np.random.randn(d_inner, dt_rank) * dt_init_std
    
    def _init_dt_bias(
        self,
        d_inner: int,
        dt_min: float,
        dt_max: float,
        dt_init: str,
    ) -> np.ndarray:
        """初始化 Δ 偏置。"""
        # 使 softplus(bias) 在 [dt_min, dt_max] 范围内
        dt = np.exp(
            np.random.uniform(
                np.log(dt_min), np.log(dt_max), size=d_inner
            )
        )
        # softplus 逆函数
        inv_softplus = lambda x: np.log(np.exp(x) - 1)
        return inv_softplus(dt)
    
    def _init_A(self, d_inner: int, d_state: int) -> np.ndarray:
        """初始化 A 矩阵 (HiPPO 初始化)。"""
        # 使用简化的 HiPPO 初始化
        A = np.repeat(np.arange(1, d_state + 1)[np.newaxis, :], d_inner, axis=0)
        return np.log(A)
    
    def __call__(
        self,
        x: np.ndarray,
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """前向传播。
        
        Args:
            x: 输入 [batch, seq_len, d_model]
            cache: 推理缓存 (用于自回归生成)
        
        Returns:
            y: 输出 [batch, seq_len, d_model]
            new_cache: 更新后的缓存
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        xz = x @ self.in_proj_weight.T
        if self.in_proj_bias is not None:
            xz = xz + self.in_proj_bias
        
        # 分割为 x 和 z
        x_proj, z = np.split(xz, 2, axis=-1)
        
        # 转置用于卷积: [B, L, D] -> [B, D, L]
        x_conv = x_proj.transpose(0, 2, 1)
        
        # 因果卷积
        x_conv = causal_conv1d(x_conv, self.conv1d_weight, self.conv1d_bias)
        
        # 转回: [B, D, L] -> [B, L, D]
        x_conv = x_conv.transpose(0, 2, 1)
        
        # SiLU 激活
        x_conv = x_conv * (1 / (1 + np.exp(-x_conv)))  # silu
        
        # SSM 参数投影
        x_dbl = x_conv @ self.x_proj_weight.T
        
        # 分割 Δ, B, C
        dt, B, C = np.split(
            x_dbl, 
            [self.dt_rank, self.dt_rank + self.d_state], 
            axis=-1
        )
        
        # Δ 投影 + softplus
        dt = dt @ self.dt_proj_weight.T + self.dt_proj_bias
        dt = np.log(1 + np.exp(dt))  # softplus
        
        # A 参数
        A = -np.exp(self.A_log)
        
        # 选择性扫描
        y = selective_scan(x_conv, dt, A, B, C, self.D)
        
        # 门控
        z_act = z * (1 / (1 + np.exp(-z)))  # silu
        y = y * z_act
        
        # 输出投影
        y = y @ self.out_proj_weight.T
        if self.out_proj_bias is not None:
            y = y + self.out_proj_bias
        
        return y, None


class MambaBlock:
    """Mamba 块 (残差连接 + 归一化)。
    
    结构:
        y = x + SSM(Norm(x))
    """
    
    def __init__(self, config: MambaConfig, layer_idx: int = 0):
        self.config = config
        self.layer_idx = layer_idx
        
        # 归一化
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        
        # 选择性 SSM
        self.ssm = SelectiveSSM(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dt_rank=config.dt_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            dt_scale=config.dt_scale,
            bias=config.bias,
            conv_bias=config.conv_bias,
        )
    
    def __call__(
        self,
        x: np.ndarray,
        cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """前向传播。"""
        # 残差连接
        residual = x
        
        # 归一化
        x = self.norm(x)
        
        # SSM
        x, new_cache = self.ssm(x, cache)
        
        # 残差
        if self.config.residual_in_fp32:
            x = x.astype(np.float32) + residual.astype(np.float32)
        else:
            x = x + residual
        
        return x, new_cache


class MambaLayer:
    """Mamba 层 (多个 MambaBlock 堆叠)。"""
    
    def __init__(self, config: MambaConfig):
        self.config = config
        self.blocks = [
            MambaBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ]
    
    def __call__(
        self,
        x: np.ndarray,
        cache: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> Tuple[np.ndarray, Optional[List[Dict[str, np.ndarray]]]]:
        """前向传播。"""
        new_caches = []
        
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache else None
            x, new_cache = block(x, layer_cache)
            new_caches.append(new_cache)
        
        return x, new_caches if cache else None


class MambaModel:
    """Mamba 基础模型。
    
    结构:
        Embedding -> N x MambaBlock -> RMSNorm
    """
    
    def __init__(self, config: MambaConfig):
        self.config = config
        
        # 词嵌入
        self.embedding = np.random.randn(
            config.vocab_size, config.d_model
        ) * config.initializer_range
        
        # Mamba 层
        self.layers = MambaLayer(config)
        
        # 最终归一化
        self.norm_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)
    
    def __call__(
        self,
        input_ids: np.ndarray,
        cache: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> Tuple[np.ndarray, Optional[List[Dict[str, np.ndarray]]]]:
        """前向传播。
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            cache: 推理缓存
        
        Returns:
            hidden_states: 隐藏状态 [batch, seq_len, d_model]
            new_cache: 更新后的缓存
        """
        # 词嵌入
        x = self.embedding[input_ids]
        
        # Mamba 层
        x, new_cache = self.layers(x, cache)
        
        # 最终归一化
        x = self.norm_f(x)
        
        return x, new_cache


class MambaForCausalLM:
    """Mamba 因果语言模型。
    
    结构:
        MambaModel -> LM Head
    """
    
    def __init__(self, config: MambaConfig):
        self.config = config
        
        # 基础模型
        self.model = MambaModel(config)
        
        # LM Head (与 embedding 共享权重)
        self.lm_head_weight = self.model.embedding  # 权重共享
    
    def __call__(
        self,
        input_ids: np.ndarray,
        labels: Optional[np.ndarray] = None,
        cache: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """前向传播。
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            labels: 标签 (用于计算损失) [batch, seq_len]
            cache: 推理缓存
        
        Returns:
            dict: 包含 logits, loss (可选), cache
        """
        # 获取隐藏状态
        hidden_states, new_cache = self.model(input_ids, cache)
        
        # LM Head
        logits = hidden_states @ self.lm_head_weight.T
        
        result = {"logits": logits, "cache": new_cache}
        
        # 计算损失
        if labels is not None:
            # 移位: 预测下一个 token
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            
            # 交叉熵损失
            loss = self._cross_entropy_loss(shift_logits, shift_labels)
            result["loss"] = loss
        
        return result
    
    def _cross_entropy_loss(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """计算交叉熵损失。"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        logits_exp = np.exp(logits - logits_max)
        probs = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)
        
        # 负对数似然
        log_probs = np.log(probs + 1e-10)
        
        # 收集标签对应的 log_probs
        loss = 0.0
        count = 0
        for b in range(batch_size):
            for s in range(seq_len):
                if labels[b, s] >= 0:  # 忽略 padding
                    loss -= log_probs[b, s, labels[b, s]]
                    count += 1
        
        return loss / max(count, 1)
    
    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> np.ndarray:
        """自回归生成。
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P (nucleus) 采样
        
        Returns:
            generated_ids: 生成的 token IDs [batch, seq_len + max_new_tokens]
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.copy()
        cache = None
        
        for _ in range(max_new_tokens):
            # 前向传播
            result = self(generated, cache=cache)
            logits = result["logits"]
            cache = result["cache"]
            
            # 获取最后一个位置的 logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-K 采样
            if top_k > 0:
                top_k_indices = np.argsort(next_logits, axis=-1)[:, -top_k:]
                mask = np.ones_like(next_logits) * (-1e10)
                for b in range(batch_size):
                    mask[b, top_k_indices[b]] = next_logits[b, top_k_indices[b]]
                next_logits = mask
            
            # Softmax
            probs = np.exp(next_logits - np.max(next_logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            # Top-P 采样
            if top_p < 1.0:
                sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
                sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
                cumsum_probs = np.cumsum(sorted_probs, axis=-1)
                
                for b in range(batch_size):
                    cutoff_idx = np.searchsorted(cumsum_probs[b], top_p)
                    probs[b, sorted_indices[b, cutoff_idx + 1:]] = 0
                
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            # 采样
            next_tokens = np.array([
                np.random.choice(self.config.vocab_size, p=probs[b])
                for b in range(batch_size)
            ])
            
            # 拼接
            generated = np.concatenate([
                generated,
                next_tokens[:, np.newaxis]
            ], axis=1)
        
        return generated


class Mamba2Block:
    """Mamba-2 块 (SSD - State Space Duality)。
    
    Mamba-2 的核心改进：
    1. 将 SSM 与线性注意力统一 (State Space Duality)
    2. 更高效的并行扫描算法
    3. 支持张量并行
    
    参考: Dao, T., et al. (2024). Transformers are SSMs.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        bias: bool = False,
        conv_bias: bool = True,
        layer_idx: int = 0,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // headdim
        self.layer_idx = layer_idx
        
        # 归一化
        self.norm = RMSNorm(d_model)
        
        # 输入投影 (合并 x, z, B, C, dt)
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj_weight = np.random.randn(d_in_proj, d_model) * 0.02
        self.in_proj_bias = np.zeros(d_in_proj) if bias else None
        
        # 因果卷积
        self.conv1d_weight = np.random.randn(self.d_inner, 1, d_conv) * 0.02
        self.conv1d_bias = np.zeros(self.d_inner) if conv_bias else None
        
        # A 参数 (每个头一个标量)
        self.A_log = np.log(np.arange(1, self.nheads + 1))
        
        # D 参数
        self.D = np.ones(self.nheads)
        
        # 输出投影
        self.out_proj_weight = np.random.randn(d_model, self.d_inner) * 0.02
        self.out_proj_bias = np.zeros(d_model) if bias else None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """前向传播。"""
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # 归一化
        x = self.norm(x)
        
        # 输入投影
        xz = x @ self.in_proj_weight.T
        if self.in_proj_bias is not None:
            xz = xz + self.in_proj_bias
        
        # 分割
        d_mlp = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        x_proj = xz[:, :, :self.d_inner]
        z = xz[:, :, self.d_inner:2*self.d_inner]
        B = xz[:, :, 2*self.d_inner:2*self.d_inner + self.ngroups * self.d_state]
        C = xz[:, :, 2*self.d_inner + self.ngroups * self.d_state:
               2*self.d_inner + 2 * self.ngroups * self.d_state]
        dt = xz[:, :, -self.nheads:]
        
        # 卷积
        x_conv = x_proj.transpose(0, 2, 1)
        x_conv = causal_conv1d(x_conv, self.conv1d_weight, self.conv1d_bias)
        x_conv = x_conv.transpose(0, 2, 1)
        
        # SiLU
        x_conv = x_conv * (1 / (1 + np.exp(-x_conv)))
        
        # Softplus for dt
        dt = np.log(1 + np.exp(dt))
        
        # A 参数
        A = -np.exp(self.A_log)
        
        # 简化的 SSD 计算 (实际实现使用分块并行)
        # 这里使用顺序扫描作为参考实现
        y = self._ssd_sequential(x_conv, dt, A, B, C)
        
        # 门控
        z_act = z * (1 / (1 + np.exp(-z)))
        y = y * z_act
        
        # 输出投影
        y = y @ self.out_proj_weight.T
        if self.out_proj_bias is not None:
            y = y + self.out_proj_bias
        
        return y + residual
    
    def _ssd_sequential(
        self,
        x: np.ndarray,
        dt: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
    ) -> np.ndarray:
        """顺序 SSD 计算 (参考实现)。"""
        batch_size, seq_len, d_inner = x.shape
        
        # 重塑为多头格式
        x = x.reshape(batch_size, seq_len, self.nheads, self.headdim)
        B = B.reshape(batch_size, seq_len, self.ngroups, self.d_state)
        C = C.reshape(batch_size, seq_len, self.ngroups, self.d_state)
        
        # 初始化状态
        h = np.zeros((batch_size, self.nheads, self.headdim, self.d_state))
        y = np.zeros((batch_size, seq_len, self.nheads, self.headdim))
        
        for t in range(seq_len):
            # 离散化
            dt_t = dt[:, t, :, np.newaxis, np.newaxis]  # [B, H, 1, 1]
            A_bar = np.exp(dt_t * A[:, np.newaxis, np.newaxis])  # [B, H, 1, 1]
            
            # 状态更新
            x_t = x[:, t, :, :, np.newaxis]  # [B, H, D, 1]
            B_t = B[:, t, 0, np.newaxis, np.newaxis, :]  # [B, 1, 1, N]
            h = A_bar * h + x_t * B_t * dt_t
            
            # 输出
            C_t = C[:, t, 0, :]  # [B, N]
            y[:, t, :, :] = np.einsum("bhdn,bn->bhd", h, C_t)
        
        # 重塑回原始格式
        y = y.reshape(batch_size, seq_len, d_inner)
        
        # D 跳跃连接
        x_flat = x.reshape(batch_size, seq_len, d_inner)
        y = y + x_flat * np.repeat(self.D, self.headdim)
        
        return y


def create_mamba_model(
    model_size: str = "base",
    vocab_size: int = 50257,
    **kwargs,
) -> MambaForCausalLM:
    """创建预设配置的 Mamba 模型。
    
    Args:
        model_size: 模型大小 ("small", "base", "large", "xlarge")
        vocab_size: 词表大小
        **kwargs: 覆盖默认配置的参数
    
    Returns:
        MambaForCausalLM 实例
    
    预设配置:
        - small:  130M 参数 (d=768,  n=24)
        - base:   370M 参数 (d=1024, n=48)
        - large:  790M 参数 (d=1536, n=48)
        - xlarge: 1.4B 参数 (d=2048, n=48)
    """
    configs = {
        "small": MambaConfig(
            d_model=768,
            n_layers=24,
            d_state=16,
            expand=2,
            vocab_size=vocab_size,
        ),
        "base": MambaConfig(
            d_model=1024,
            n_layers=48,
            d_state=16,
            expand=2,
            vocab_size=vocab_size,
        ),
        "large": MambaConfig(
            d_model=1536,
            n_layers=48,
            d_state=16,
            expand=2,
            vocab_size=vocab_size,
        ),
        "xlarge": MambaConfig(
            d_model=2048,
            n_layers=48,
            d_state=16,
            expand=2,
            vocab_size=vocab_size,
        ),
    }
    
    if model_size not in configs:
        raise ValueError(
            f"未知模型大小: {model_size}，可选: {list(configs.keys())}"
        )
    
    config = configs[model_size]
    
    # 覆盖配置
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return MambaForCausalLM(config)


# 模型参数量计算
def count_parameters(config: MambaConfig) -> Dict[str, int]:
    """计算 Mamba 模型参数量。
    
    Args:
        config: 模型配置
    
    Returns:
        各部分参数量统计
    """
    d_model = config.d_model
    d_inner = config.d_inner
    d_state = config.d_state
    d_conv = config.d_conv
    dt_rank = config.dt_rank
    n_layers = config.n_layers
    vocab_size = config.vocab_size
    
    # 每层参数
    in_proj = d_inner * 2 * d_model  # 输入投影
    conv1d = d_inner * d_conv  # 卷积
    x_proj = (dt_rank + d_state * 2) * d_inner  # SSM 参数投影
    dt_proj = d_inner * dt_rank + d_inner  # Δ 投影 + 偏置
    A_log = d_inner * d_state  # A 参数
    D = d_inner  # D 参数
    out_proj = d_model * d_inner  # 输出投影
    norm = d_model  # RMSNorm
    
    per_layer = in_proj + conv1d + x_proj + dt_proj + A_log + D + out_proj + norm
    
    # 总参数
    embedding = vocab_size * d_model
    final_norm = d_model
    
    total = embedding + n_layers * per_layer + final_norm
    
    return {
        "embedding": embedding,
        "per_layer": per_layer,
        "all_layers": n_layers * per_layer,
        "final_norm": final_norm,
        "total": total,
        "total_millions": total / 1e6,
    }
