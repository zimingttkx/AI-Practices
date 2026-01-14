"""
Ring Attention: 分布式长序列注意力

============================================================
核心思想 (Core Idea)
============================================================
Ring Attention 通过环形通信模式在多个设备间分布式计算注意力，
实现近乎无限的序列长度，同时保持内存效率。

============================================================
数学基础 (Mathematical Foundation)
============================================================
标准注意力:
    Attention(Q, K, V) = softmax(QK^T / √d) V

Ring Attention 分块计算:
    将 K, V 分成 P 块 (P = 设备数)
    每个设备持有 Q 的一部分，K/V 通过环形传递
    
    对于设备 i:
    1. 计算本地 QK^T 块
    2. 接收下一个 K/V 块
    3. 累积 softmax 分母和输出
    4. 重复直到处理完所有 K/V 块

============================================================
参考文献 (References)
============================================================
[1] Liu, H., et al. (2023). Ring Attention with Blockwise 
    Transformers for Near-Infinite Context. arXiv:2310.01889
[2] Li, S., et al. (2023). Sequence Parallelism: Long Sequence 
    Training from System Perspective. ACL 2023.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np

__all__ = [
    "RingAttentionConfig",
    "BlockwiseAttention",
    "RingCommunicator",
    "RingAttention",
    "RingAttentionLayer",
    "SequenceParallel",
    "create_ring_attention",
    "compute_blockwise_attention",
]


@dataclass
class RingAttentionConfig:
    """Ring Attention 配置。
    
    参数：
        d_model: 模型维度
        n_heads: 注意力头数
        block_size: 块大小 (每个设备处理的序列长度)
        n_devices: 设备数量 (环大小)
        causal: 是否使用因果掩码
        dropout: Dropout 率
        use_flash: 是否使用 Flash Attention 风格计算
    """
    d_model: int = 768
    n_heads: int = 12
    block_size: int = 1024
    n_devices: int = 8
    causal: bool = True
    dropout: float = 0.0
    use_flash: bool = True
    
    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model 必须为正数")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads 必须为正数")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model 必须能被 n_heads 整除")
        if self.block_size <= 0:
            raise ValueError(f"block_size 必须为正数")
        if self.n_devices <= 0:
            raise ValueError(f"n_devices 必须为正数")
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
    
    @property
    def max_seq_len(self) -> int:
        """最大支持序列长度。"""
        return self.block_size * self.n_devices


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 Softmax。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class RingCommunicator:
    """环形通信模拟器。
    
    模拟分布式环境中的环形通信模式。
    """
    
    def __init__(self, n_devices: int):
        self.n_devices = n_devices
        self.buffers: Dict[int, Dict[str, np.ndarray]] = {
            i: {} for i in range(n_devices)
        }
    
    def send(self, src: int, dst: int, key: str, data: np.ndarray) -> None:
        """发送数据。"""
        self.buffers[dst][key] = data.copy()
    
    def recv(self, device: int, key: str) -> np.ndarray:
        """接收数据。"""
        return self.buffers[device].pop(key)
    
    def ring_send_recv(
        self,
        device: int,
        send_data: np.ndarray,
        key: str = "kv",
    ) -> np.ndarray:
        """环形发送接收 (同时发送和接收)。"""
        next_device = (device + 1) % self.n_devices
        prev_device = (device - 1) % self.n_devices
        
        # 发送到下一个设备
        self.send(device, next_device, key, send_data)
        
        # 从上一个设备接收 (模拟)
        # 实际实现中这是同步的
        return send_data  # 简化: 返回原数据用于测试


class BlockwiseAttention:
    """分块注意力计算。
    
    使用在线 softmax 算法，支持分块累积计算。
    """
    
    def __init__(self, head_dim: int, causal: bool = True):
        self.head_dim = head_dim
        self.scale = 1.0 / np.sqrt(head_dim)
        self.causal = causal
    
    def __call__(
        self,
        q_block: np.ndarray,
        k_block: np.ndarray,
        v_block: np.ndarray,
        q_offset: int = 0,
        k_offset: int = 0,
        prev_max: Optional[np.ndarray] = None,
        prev_sum: Optional[np.ndarray] = None,
        prev_out: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """分块注意力计算 (在线 softmax)。
        
        Args:
            q_block: Query 块 [batch, n_heads, q_len, head_dim]
            k_block: Key 块 [batch, n_heads, k_len, head_dim]
            v_block: Value 块 [batch, n_heads, k_len, head_dim]
            q_offset: Query 在全局序列中的偏移
            k_offset: Key 在全局序列中的偏移
            prev_max: 之前块的最大值
            prev_sum: 之前块的 exp 和
            prev_out: 之前块的输出累积
        
        Returns:
            new_max: 更新后的最大值
            new_sum: 更新后的 exp 和
            new_out: 更新后的输出
        """
        batch_size, n_heads, q_len, _ = q_block.shape
        k_len = k_block.shape[2]
        
        # 计算注意力分数
        scores = np.einsum("bhqd,bhkd->bhqk", q_block, k_block) * self.scale
        
        # 因果掩码
        if self.causal:
            q_indices = np.arange(q_len) + q_offset
            k_indices = np.arange(k_len) + k_offset
            mask = q_indices[:, None] < k_indices[None, :]
            scores = np.where(mask, -1e9, scores)
        
        # 在线 softmax 更新
        block_max = np.max(scores, axis=-1, keepdims=True)
        
        if prev_max is None:
            new_max = block_max
            exp_scores = np.exp(scores - new_max)
            new_sum = np.sum(exp_scores, axis=-1, keepdims=True)
            new_out = np.einsum("bhqk,bhkd->bhqd", exp_scores, v_block)
        else:
            # 更新最大值
            new_max = np.maximum(prev_max, block_max)
            
            # 重新缩放之前的累积
            prev_scale = np.exp(prev_max - new_max)
            curr_scale = np.exp(block_max - new_max)
            
            exp_scores = np.exp(scores - new_max)
            block_sum = np.sum(exp_scores, axis=-1, keepdims=True)
            
            new_sum = prev_sum * prev_scale + block_sum
            
            # 更新输出
            block_out = np.einsum("bhqk,bhkd->bhqd", exp_scores, v_block)
            new_out = prev_out * prev_scale + block_out
        
        return new_max, new_sum, new_out
    
    def finalize(
        self,
        out: np.ndarray,
        sum_exp: np.ndarray,
    ) -> np.ndarray:
        """最终化输出 (除以 softmax 分母)。"""
        return out / sum_exp


class RingAttention:
    """Ring Attention 实现。
    
    通过环形通信在多设备间分布式计算注意力。
    """
    
    def __init__(self, config: RingAttentionConfig):
        self.config = config
        self.head_dim = config.head_dim
        self.scale = 1.0 / np.sqrt(self.head_dim)
        
        # 投影权重
        scale = 1.0 / np.sqrt(config.d_model)
        self.w_q = np.random.randn(config.d_model, config.d_model) * scale
        self.w_k = np.random.randn(config.d_model, config.d_model) * scale
        self.w_v = np.random.randn(config.d_model, config.d_model) * scale
        self.w_o = np.random.randn(config.d_model, config.d_model) * scale
        
        # 分块注意力计算器
        self.blockwise_attn = BlockwiseAttention(self.head_dim, config.causal)
        
        # 通信器
        self.comm = RingCommunicator(config.n_devices)
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """分割为多头: [B, L, D] -> [B, H, L, D/H]"""
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.config.n_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)
    
    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """合并多头: [B, H, L, D/H] -> [B, L, D]"""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq_len, self.config.d_model)
    
    def __call__(
        self,
        x: np.ndarray,
        device_id: int = 0,
    ) -> np.ndarray:
        """Ring Attention 前向传播。
        
        Args:
            x: 输入 [batch, seq_len, d_model] (本设备的序列块)
            device_id: 当前设备 ID
        
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV 投影
        q = x @ self.w_q.T
        k = x @ self.w_k.T
        v = x @ self.w_v.T
        
        # 分割多头
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # 本设备的 Q 偏移
        q_offset = device_id * seq_len
        
        # 初始化累积变量
        max_score = None
        sum_exp = None
        output = None
        
        # 当前 KV 块
        current_k = k
        current_v = v
        current_k_offset = device_id * seq_len
        
        # 环形迭代
        for step in range(self.config.n_devices):
            # 计算当前块的注意力
            max_score, sum_exp, output = self.blockwise_attn(
                q, current_k, current_v,
                q_offset=q_offset,
                k_offset=current_k_offset,
                prev_max=max_score,
                prev_sum=sum_exp,
                prev_out=output,
            )
            
            # 环形传递 KV (除了最后一步)
            if step < self.config.n_devices - 1:
                # 模拟环形通信
                next_device = (device_id + step + 1) % self.config.n_devices
                current_k_offset = next_device * seq_len
                # 实际实现中这里会进行真正的通信
        
        # 最终化输出
        output = self.blockwise_attn.finalize(output, sum_exp)
        
        # 合并多头
        output = self._merge_heads(output)
        
        # 输出投影
        output = output @ self.w_o.T
        
        return output


class RingAttentionLayer:
    """Ring Attention 层 (带残差和归一化)。"""
    
    def __init__(self, config: RingAttentionConfig):
        self.config = config
        self.attn = RingAttention(config)
        
        # Layer Norm
        self.norm_weight = np.ones(config.d_model)
        self.norm_bias = np.zeros(config.d_model)
        self.eps = 1e-5
    
    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps) * self.norm_weight + self.norm_bias
    
    def __call__(self, x: np.ndarray, device_id: int = 0) -> np.ndarray:
        # Pre-norm
        h = self._layer_norm(x)
        # Attention
        h = self.attn(h, device_id)
        # Residual
        return x + h


class SequenceParallel:
    """序列并行管理器。
    
    管理长序列在多设备间的分布。
    """
    
    def __init__(self, config: RingAttentionConfig):
        self.config = config
    
    def split_sequence(
        self,
        x: np.ndarray,
    ) -> List[np.ndarray]:
        """将序列分割到多个设备。"""
        batch_size, seq_len, d_model = x.shape
        block_size = self.config.block_size
        n_devices = self.config.n_devices
        
        # 填充到 block_size * n_devices
        target_len = block_size * n_devices
        if seq_len < target_len:
            pad_len = target_len - seq_len
            x = np.pad(x, ((0, 0), (0, pad_len), (0, 0)))
        
        # 分割
        blocks = []
        for i in range(n_devices):
            start = i * block_size
            end = start + block_size
            blocks.append(x[:, start:end, :])
        
        return blocks
    
    def gather_sequence(
        self,
        blocks: List[np.ndarray],
        original_len: int,
    ) -> np.ndarray:
        """收集分布式序列。"""
        x = np.concatenate(blocks, axis=1)
        return x[:, :original_len, :]


def compute_blockwise_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    block_size: int = 1024,
    causal: bool = True,
) -> np.ndarray:
    """分块计算注意力 (单设备版本)。
    
    用于长序列的内存高效注意力计算。
    """
    batch, n_heads, seq_len, head_dim = q.shape
    scale = 1.0 / np.sqrt(head_dim)
    
    n_blocks = (seq_len + block_size - 1) // block_size
    
    output = np.zeros_like(q)
    
    for q_block_idx in range(n_blocks):
        q_start = q_block_idx * block_size
        q_end = min(q_start + block_size, seq_len)
        q_block = q[:, :, q_start:q_end, :]
        
        max_score = None
        sum_exp = None
        block_out = None
        
        for k_block_idx in range(n_blocks):
            k_start = k_block_idx * block_size
            k_end = min(k_start + block_size, seq_len)
            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]
            
            # 计算分数
            scores = np.einsum("bhqd,bhkd->bhqk", q_block, k_block) * scale
            
            # 因果掩码
            if causal:
                q_indices = np.arange(q_end - q_start) + q_start
                k_indices = np.arange(k_end - k_start) + k_start
                mask = q_indices[:, None] < k_indices[None, :]
                scores = np.where(mask, -1e9, scores)
            
            # 在线 softmax
            block_max = np.max(scores, axis=-1, keepdims=True)
            
            if max_score is None:
                max_score = block_max
                exp_scores = np.exp(scores - max_score)
                sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
                block_out = np.einsum("bhqk,bhkd->bhqd", exp_scores, v_block)
            else:
                new_max = np.maximum(max_score, block_max)
                prev_scale = np.exp(max_score - new_max)
                exp_scores = np.exp(scores - new_max)
                curr_sum = np.sum(exp_scores, axis=-1, keepdims=True)
                sum_exp = sum_exp * prev_scale + curr_sum
                block_out = block_out * prev_scale + np.einsum("bhqk,bhkd->bhqd", exp_scores, v_block)
                max_score = new_max
        
        output[:, :, q_start:q_end, :] = block_out / sum_exp
    
    return output


def create_ring_attention(
    d_model: int = 768,
    n_heads: int = 12,
    block_size: int = 1024,
    n_devices: int = 8,
    **kwargs,
) -> RingAttention:
    """创建 Ring Attention 实例。"""
    config = RingAttentionConfig(
        d_model=d_model,
        n_heads=n_heads,
        block_size=block_size,
        n_devices=n_devices,
        **kwargs,
    )
    return RingAttention(config)
