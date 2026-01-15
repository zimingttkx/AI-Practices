"""
Flash Attention: 高效注意力机制实现

============================================================
核心思想 (Core Idea)
============================================================
Flash Attention 通过分块计算和内存优化，将注意力机制的内存复杂度
从 O(N²) 降低到 O(N)，同时利用 GPU 硬件特性实现高效计算。

Flash Attention 3 在此基础上引入：
1. Warp-Specialization: Producer-Consumer 异步流水线
2. GEMM-Softmax 重叠: 2-stage pingpong 调度隐藏 softmax 延迟
3. FP8 低精度: Block Quantization + Incoherent Processing

============================================================
算法演进 (Algorithm Evolution)
============================================================
V1 (2022): 分块计算 + 重计算 (避免存储 N×N 矩阵)
V2 (2023): 优化并行策略 + 减少非矩阵乘法操作
V3 (2024): 异步流水线 + FP8 + Hopper GPU 优化

============================================================
复杂度分析 (Complexity Analysis)
============================================================
标准 Attention:
    - 时间: O(N² · d)
    - 空间: O(N² + N·d)  # 需要存储 S, P 矩阵

Flash Attention:
    - 时间: O(N² · d)    # 相同
    - 空间: O(N·d)       # 不存储中间矩阵
    - IO: O(N² · d / M)  # M = SRAM 大小

============================================================
参考文献 (References)
============================================================
[1] Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient 
    Exact Attention with IO-Awareness. NeurIPS 2022.
[2] Dao, T. (2023). FlashAttention-2: Faster Attention with Better 
    Parallelism and Work Partitioning. arXiv:2307.08691
[3] Shah, J., et al. (2024). FlashAttention-3: Fast and Accurate 
    Attention with Asynchrony and Low-precision. arXiv:2407.08608
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from abc import ABC, abstractmethod

# =============================================================================
# 常量定义
# =============================================================================

class Constants:
    """全局常量"""
    EPS = 1e-6                    # 数值稳定性 epsilon
    MIN_SCALE = 1e-12             # 最小缩放因子
    SOFTMAX_OPS_PER_ELEM = 5      # softmax 每元素操作数 (exp, sub, sum, div, max)
    FP8_E4M3_MAX = 448.0          # FP8 E4M3 最大值
    FP8_E5M2_MAX = 57344.0        # FP8 E5M2 最大值


__all__ = [
    "FlashAttentionConfig",
    "OnlineSoftmax",
    "BlockwiseAttention",
    "WarpScheduler",
    "FP8Quantizer",
    "IncoherentProcessor",
    "FlashAttentionV1",
    "FlashAttentionV2",
    "FlashAttentionV3",
    "create_flash_attention",
    "standard_attention",
    "compute_attention_flops",
    "validate_attention_inputs",
]


# =============================================================================
# 工具函数 (消除重复代码)
# =============================================================================

def validate_attention_inputs(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> None:
    """验证注意力输入的有效性。
    
    Raises:
        ValueError: 输入维度或形状不匹配
    """
    if query.ndim not in (3, 4):
        raise ValueError(f"Query must be 3D or 4D, got {query.ndim}D")
    if key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("Q, K, V must have same number of dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(f"Q head_dim ({query.shape[-1]}) != K head_dim ({key.shape[-1]})")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(f"K seq_len ({key.shape[-2]}) != V seq_len ({value.shape[-2]})")
    if np.isnan(query).any() or np.isnan(key).any() or np.isnan(value).any():
        raise ValueError("Input contains NaN values")


def create_causal_mask(
    q_start: int,
    q_end: int,
    kv_start: int,
    kv_end: int,
    batch_size: int
) -> np.ndarray:
    """创建因果掩码。
    
    Args:
        q_start, q_end: Query 范围
        kv_start, kv_end: Key/Value 范围
        batch_size: 批次大小
        
    Returns:
        因果掩码 [batch, q_len, kv_len]
    """
    q_indices = np.arange(q_start, q_end)[:, np.newaxis]
    k_indices = np.arange(kv_start, kv_end)[np.newaxis, :]
    causal_mask = k_indices <= q_indices
    return np.broadcast_to(causal_mask, (batch_size, q_end - q_start, kv_end - kv_start))


def compute_scale(head_dim: int, softmax_scale: Optional[float] = None) -> float:
    """计算注意力缩放因子。"""
    return softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)


class PrecisionMode(Enum):
    """计算精度模式"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


class SchedulingMode(Enum):
    """调度模式"""
    SYNCHRONOUS = "sync"      # 同步执行
    PINGPONG = "pingpong"     # 2-stage 交替
    PERSISTENT = "persistent"  # 持久化内核


@dataclass
class FlashAttentionConfig:
    """Flash Attention 配置类。
    
    参数：
        block_size_q: Query 分块大小 (Br)
        block_size_kv: Key/Value 分块大小 (Bc)
        num_stages: 流水线阶段数 (用于异步调度)
        precision: 计算精度 (fp32/fp16/bf16/fp8)
        causal: 是否使用因果掩码
        dropout_p: Dropout 概率
        softmax_scale: Softmax 缩放因子 (默认 1/sqrt(d))
        deterministic: 是否确定性计算
        return_softmax: 是否返回 softmax 权重 (用于调试)
        
    Flash Attention 3 特有参数：
        use_warp_specialization: 是否使用 warp 专门化
        scheduling_mode: 调度模式 (sync/pingpong/persistent)
        use_fp8_quantization: 是否使用 FP8 量化
        use_incoherent_processing: 是否使用 incoherent processing
    """
    # 基础参数
    block_size_q: int = 128
    block_size_kv: int = 128
    num_stages: int = 2
    precision: PrecisionMode = PrecisionMode.FP32
    causal: bool = False
    dropout_p: float = 0.0
    softmax_scale: Optional[float] = None
    deterministic: bool = False
    return_softmax: bool = False
    
    # Flash Attention 3 参数
    use_warp_specialization: bool = True
    scheduling_mode: SchedulingMode = SchedulingMode.PINGPONG
    use_fp8_quantization: bool = False
    use_incoherent_processing: bool = False
    fp8_block_size: int = 256
    
    def __post_init__(self):
        """验证配置参数"""
        if self.block_size_q <= 0 or self.block_size_kv <= 0:
            raise ValueError("Block sizes must be positive")
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1)")
        if isinstance(self.precision, str):
            self.precision = PrecisionMode(self.precision)
        if isinstance(self.scheduling_mode, str):
            self.scheduling_mode = SchedulingMode(self.scheduling_mode)


class OnlineSoftmax:
    """在线 Softmax 算法实现。
    
    核心思想：分块计算 softmax，无需预先知道全局 max。
    
    算法：
        对于每个新块 S_j:
        1. m_new = max(m_old, rowmax(S_j))
        2. l_new = l_old * exp(m_old - m_new) + rowsum(exp(S_j - m_new))
        3. O_new = O_old * exp(m_old - m_new) + exp(S_j - m_new) @ V_j
        4. 最终: O = O / l
    """
    
    def __init__(self, eps: float = Constants.EPS):
        self.eps = eps
    
    def init_state(self, batch_size: int, seq_len: int, head_dim: int) -> Dict[str, np.ndarray]:
        """初始化在线 softmax 状态。
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度 (query)
            head_dim: 头维度
            
        Returns:
            状态字典: {m, l, o}
        """
        return {
            "m": np.full((batch_size, seq_len), -np.inf, dtype=np.float32),  # 当前最大值
            "l": np.zeros((batch_size, seq_len), dtype=np.float32),          # 当前指数和
            "o": np.zeros((batch_size, seq_len, head_dim), dtype=np.float32) # 当前输出
        }
    
    def update(
        self,
        state: Dict[str, np.ndarray],
        scores: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """更新在线 softmax 状态。
        
        Args:
            state: 当前状态 {m, l, o}
            scores: 当前块的注意力分数 [batch, seq_q, block_kv]
            values: 当前块的 values [batch, block_kv, head_dim]
            mask: 可选的掩码 [batch, seq_q, block_kv]
            
        Returns:
            更新后的状态
        """
        m_old, l_old, o_old = state["m"], state["l"], state["o"]
        
        # 应用掩码
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        
        # 计算当前块的 rowmax
        m_block = np.max(scores, axis=-1)  # [batch, seq_q]
        
        # 更新全局 max
        m_new = np.maximum(m_old, m_block)
        
        # 计算缩放因子
        scale_old = np.exp(m_old - m_new)  # [batch, seq_q]
        scale_block = np.exp(scores - m_new[..., np.newaxis])  # [batch, seq_q, block_kv]
        
        # 更新 l (指数和)
        l_block = np.sum(scale_block, axis=-1)  # [batch, seq_q]
        l_new = l_old * scale_old + l_block
        
        # 更新 o (输出累积)
        # o_new = o_old * scale_old + scale_block @ values
        o_scaled = o_old * scale_old[..., np.newaxis]  # [batch, seq_q, head_dim]
        o_block = np.einsum("bqk,bkd->bqd", scale_block, values)  # [batch, seq_q, head_dim]
        o_new = o_scaled + o_block
        
        return {"m": m_new, "l": l_new, "o": o_new}
    
    def finalize(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """完成在线 softmax 计算。
        
        Args:
            state: 最终状态
            
        Returns:
            归一化后的输出 [batch, seq_q, head_dim]
        """
        l = state["l"]
        o = state["o"]
        # 归一化
        return o / (l[..., np.newaxis] + self.eps)


class BlockwiseAttention:
    """分块注意力计算。
    
    将 Q, K, V 分成小块，逐块计算注意力，使用 OnlineSoftmax 累积结果。
    
    内存优化：
        标准: O(N²) - 需要存储完整的 S = QK^T 和 P = softmax(S)
        分块: O(Br × Bc) - 只需存储当前块
    
    IO 优化：
        标准: 从 HBM 读取 Q,K,V，写入 S,P，再读取 P,V 写入 O
        分块: 从 HBM 读取 Q,K,V 到 SRAM，在 SRAM 中完成所有计算，只写回 O
    """
    
    def __init__(self, config: FlashAttentionConfig):
        """初始化分块注意力。
        
        Args:
            config: Flash Attention 配置
        """
        self.config = config
        self.online_softmax = OnlineSoftmax()
    
    def compute(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """计算分块注意力。
        
        Args:
            query: [batch, seq_q, head_dim]
            key: [batch, seq_kv, head_dim]
            value: [batch, seq_kv, head_dim]
            mask: 可选掩码 [batch, seq_q, seq_kv]
            
        Returns:
            output: [batch, seq_q, head_dim]
            stats: 可选的统计信息
        """
        batch_size, seq_q, head_dim = query.shape
        _, seq_kv, _ = key.shape
        
        Br = self.config.block_size_q
        Bc = self.config.block_size_kv
        
        # 计算缩放因子
        scale = self.config.softmax_scale
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        # 初始化在线 softmax 状态
        state = self.online_softmax.init_state(batch_size, seq_q, head_dim)
        
        # 分块数量
        num_blocks_kv = (seq_kv + Bc - 1) // Bc
        
        # 遍历 KV 块
        for j in range(num_blocks_kv):
            kv_start = j * Bc
            kv_end = min((j + 1) * Bc, seq_kv)
            
            # 提取当前 KV 块
            k_block = key[:, kv_start:kv_end, :]      # [batch, block_kv, head_dim]
            v_block = value[:, kv_start:kv_end, :]    # [batch, block_kv, head_dim]
            
            # 计算注意力分数: S = Q @ K^T * scale
            scores = np.einsum("bqd,bkd->bqk", query, k_block) * scale
            
            # 提取掩码块
            mask_block = None
            if mask is not None:
                mask_block = mask[:, :, kv_start:kv_end]
            
            # 因果掩码
            if self.config.causal:
                # 创建因果掩码: query[i] 只能看到 key[j] where j <= i
                q_indices = np.arange(seq_q)[:, np.newaxis]
                k_indices = np.arange(kv_start, kv_end)[np.newaxis, :]
                causal_mask = k_indices <= q_indices
                
                if mask_block is not None:
                    mask_block = mask_block & causal_mask
                else:
                    mask_block = np.broadcast_to(causal_mask, (batch_size, seq_q, kv_end - kv_start))
            
            # 更新在线 softmax
            state = self.online_softmax.update(state, scores, v_block, mask_block)
        
        # 完成计算
        output = self.online_softmax.finalize(state)
        
        stats = None
        if self.config.return_softmax:
            stats = {"m": state["m"], "l": state["l"]}
        
        return output, stats


class WarpScheduler:
    """Warp 调度器 - 模拟 Flash Attention 3 的异步流水线。
    
    Flash Attention 3 核心创新：
    1. Producer-Consumer 分离: Producer warps 负责数据加载，Consumer warps 负责计算
    2. Pingpong 调度: 2-stage 交替执行，隐藏 softmax 延迟
    3. 异步 WGMMA: 利用 Hopper GPU 的异步 Tensor Core
    
    调度模式：
    - SYNCHRONOUS: 传统同步执行
    - PINGPONG: 2-stage 交替 (GEMM 和 softmax 重叠)
    - PERSISTENT: 持久化内核 (减少内核启动开销)
    
    Pingpong 调度示意：
        Stage 0: GEMM(S_0) ──overlap──→ Softmax(S_prev)
        Stage 1: GEMM(S_1) ──overlap──→ Softmax(S_0)
        Stage 0: GEMM(S_2) ──overlap──→ Softmax(S_1)
        ...
    """
    
    def __init__(self, config: FlashAttentionConfig):
        """初始化调度器。
        
        Args:
            config: Flash Attention 配置
        """
        self.config = config
        self.num_stages = config.num_stages
        self.mode = config.scheduling_mode
        
        # 模拟 SMEM 缓冲区
        self.smem_buffer: List[Dict[str, np.ndarray]] = []
        
        # 统计信息
        self.stats = {
            "gemm_ops": 0,
            "softmax_ops": 0,
            "overlapped_ops": 0,
            "memory_transfers": 0
        }
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0
    
    def simulate_producer(
        self,
        key: np.ndarray,
        value: np.ndarray,
        block_idx: int
    ) -> Dict[str, np.ndarray]:
        """模拟 Producer Warp 的数据加载。
        
        在真实 GPU 上，这由 TMA (Tensor Memory Accelerator) 执行。
        
        Args:
            key: 完整的 key 张量
            value: 完整的 value 张量
            block_idx: 当前块索引
            
        Returns:
            加载到 SMEM 的数据块
        """
        Bc = self.config.block_size_kv
        seq_kv = key.shape[1]
        
        start = block_idx * Bc
        end = min((block_idx + 1) * Bc, seq_kv)
        
        block_data = {
            "k": key[:, start:end, :].copy(),
            "v": value[:, start:end, :].copy(),
            "block_idx": block_idx
        }
        
        self.stats["memory_transfers"] += 1
        return block_data
    
    def simulate_consumer_gemm(
        self,
        query: np.ndarray,
        k_block: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """模拟 Consumer Warp 的 GEMM 计算。
        
        在真实 GPU 上，这由 WGMMA (Warpgroup Matrix Multiply-Accumulate) 执行。
        
        Args:
            query: Query 张量 [batch, seq_q, head_dim]
            k_block: Key 块 [batch, block_kv, head_dim]
            scale: 缩放因子
            
        Returns:
            注意力分数 [batch, seq_q, block_kv]
        """
        scores = np.einsum("bqd,bkd->bqk", query, k_block) * scale
        self.stats["gemm_ops"] += 1
        return scores
    
    def simulate_consumer_softmax(
        self,
        scores: np.ndarray,
        state: Dict[str, np.ndarray],
        v_block: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """模拟 Consumer Warp 的 Softmax 计算。
        
        Args:
            scores: 注意力分数
            state: 在线 softmax 状态
            v_block: Value 块
            mask: 可选掩码
            
        Returns:
            更新后的状态
        """
        online_softmax = OnlineSoftmax()
        new_state = online_softmax.update(state, scores, v_block, mask)
        self.stats["softmax_ops"] += 1
        return new_state
    
    def schedule_pingpong(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        scale: float,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """执行 Pingpong 调度。
        
        2-stage 流水线：
        - Stage A: 执行 GEMM(block_i)
        - Stage B: 执行 Softmax(block_{i-1}) + GEMM(block_i) 的 PV 累积
        
        Args:
            query: [batch, seq_q, head_dim]
            key: [batch, seq_kv, head_dim]
            value: [batch, seq_kv, head_dim]
            scale: 缩放因子
            mask: 可选掩码
            
        Returns:
            输出 [batch, seq_q, head_dim]
        """
        batch_size, seq_q, head_dim = query.shape
        seq_kv = key.shape[1]
        Bc = self.config.block_size_kv
        num_blocks = (seq_kv + Bc - 1) // Bc
        
        # 初始化状态
        online_softmax = OnlineSoftmax()
        state = online_softmax.init_state(batch_size, seq_q, head_dim)
        
        # 预取第一个块
        prev_scores = None
        prev_v_block = None
        prev_mask = None
        
        for j in range(num_blocks):
            # Producer: 加载当前块
            block_data = self.simulate_producer(key, value, j)
            k_block = block_data["k"]
            v_block = block_data["v"]
            
            # Consumer: GEMM (当前块)
            scores = self.simulate_consumer_gemm(query, k_block, scale)
            
            # 提取掩码
            kv_start = j * Bc
            kv_end = min((j + 1) * Bc, seq_kv)
            current_mask = None
            if mask is not None:
                current_mask = mask[:, :, kv_start:kv_end]
            
            # Pingpong: Softmax (前一个块) 与 GEMM (当前块) 重叠
            if prev_scores is not None:
                # 这里模拟重叠执行
                state = self.simulate_consumer_softmax(
                    prev_scores, state, prev_v_block, prev_mask
                )
                self.stats["overlapped_ops"] += 1
            
            # 保存当前块用于下一次迭代
            prev_scores = scores
            prev_v_block = v_block
            prev_mask = current_mask
        
        # 处理最后一个块
        if prev_scores is not None:
            state = self.simulate_consumer_softmax(
                prev_scores, state, prev_v_block, prev_mask
            )
        
        # 完成计算
        output = online_softmax.finalize(state)
        return output
    
    def get_stats(self) -> Dict[str, int]:
        """获取调度统计信息"""
        return self.stats.copy()


class FP8Quantizer:
    """FP8 量化器 - 实现 Flash Attention 3 的低精度优化。
    
    FP8 格式：
    - E4M3: 4位指数 + 3位尾数，范围大，精度低 (用于前向传播)
    - E5M2: 5位指数 + 2位尾数，范围更大，精度更低 (用于梯度)
    """
    
    def __init__(self, block_size: int = 256, format: str = "e4m3"):
        self.block_size = block_size
        self.format = format
        self.max_value = Constants.FP8_E4M3_MAX if format == "e4m3" else Constants.FP8_E5M2_MAX
    
    def quantize(
        self,
        x: np.ndarray,
        return_scale: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """量化张量到 FP8。
        
        Args:
            x: 输入张量
            return_scale: 是否返回缩放因子
            
        Returns:
            量化后的张量 (和可选的缩放因子)
        """
        original_shape = x.shape
        x_flat = x.flatten()
        
        # 计算块数量
        num_elements = len(x_flat)
        num_blocks = (num_elements + self.block_size - 1) // self.block_size
        
        # 填充到块大小的整数倍
        padded_size = num_blocks * self.block_size
        x_padded = np.zeros(padded_size, dtype=x.dtype)
        x_padded[:num_elements] = x_flat
        
        # 重塑为块
        x_blocks = x_padded.reshape(num_blocks, self.block_size)
        
        # 计算每块的缩放因子
        block_max = np.max(np.abs(x_blocks), axis=1, keepdims=True)
        block_max = np.maximum(block_max, Constants.MIN_SCALE)  # 避免除零
        scales = block_max / self.max_value
        
        # 量化
        x_scaled = x_blocks / scales
        x_quantized = np.clip(np.round(x_scaled), -self.max_value, self.max_value)
        
        # 反量化 (模拟 FP8 精度损失)
        x_dequantized = x_quantized * scales
        
        # 恢复形状
        x_out = x_dequantized.flatten()[:num_elements].reshape(original_shape)
        
        if return_scale:
            return x_out, scales.flatten()[:num_blocks]
        return x_out
    
    def dequantize(
        self,
        x_quantized: np.ndarray,
        scales: np.ndarray
    ) -> np.ndarray:
        """反量化 FP8 张量。
        
        Args:
            x_quantized: 量化后的张量
            scales: 缩放因子
            
        Returns:
            反量化后的张量
        """
        original_shape = x_quantized.shape
        x_flat = x_quantized.flatten()
        
        num_elements = len(x_flat)
        num_blocks = len(scales)
        padded_size = num_blocks * self.block_size
        
        x_padded = np.zeros(padded_size, dtype=x_quantized.dtype)
        x_padded[:num_elements] = x_flat
        
        x_blocks = x_padded.reshape(num_blocks, self.block_size)
        scales_expanded = scales.reshape(-1, 1)
        
        x_dequantized = x_blocks * scales_expanded
        
        return x_dequantized.flatten()[:num_elements].reshape(original_shape)
    
    def compute_quantization_error(
        self,
        x: np.ndarray
    ) -> Dict[str, float]:
        """计算量化误差。
        
        Args:
            x: 原始张量
            
        Returns:
            误差统计 {mse, max_error, relative_error}
        """
        x_quantized, _ = self.quantize(x)
        
        mse = np.mean((x - x_quantized) ** 2)
        max_error = np.max(np.abs(x - x_quantized))
        relative_error = np.mean(np.abs(x - x_quantized) / (np.abs(x) + 1e-12))
        
        return {
            "mse": float(mse),
            "max_error": float(max_error),
            "relative_error": float(relative_error)
        }


class IncoherentProcessor:
    """Incoherent Processing - 降低 FP8 量化误差的技术。
    
    核心思想：
    在量化前对数据应用随机正交变换，使数据分布更均匀，
    减少 outlier 对量化精度的影响。
    
    算法：
    1. 生成随机正交矩阵 R (Hadamard 或随机旋转)
    2. 量化前: x' = x @ R
    3. 量化: x'_q = quantize(x')
    4. 反量化后: x_out = x'_q @ R^T
    
    优势：
    - 将 outlier 的影响分散到多个元素
    - 减少 2.6x 的数值误差 (相比标准 FP8)
    
    参考：
    [1] Ashkboos et al. (2024). QuaRot: Outlier-Free 4-Bit Inference
    """
    
    def __init__(self, dim: int, seed: int = 42):
        """初始化 Incoherent Processor。
        
        Args:
            dim: 变换维度
            seed: 随机种子
        """
        self.dim = dim
        self.seed = seed
        self.rotation_matrix = self._generate_hadamard_matrix(dim)
    
    def _generate_hadamard_matrix(self, n: int) -> np.ndarray:
        """生成 Hadamard 矩阵 (或近似)。
        
        Hadamard 矩阵是正交矩阵，满足 H @ H^T = n * I
        
        Args:
            n: 矩阵维度
            
        Returns:
            归一化的 Hadamard 矩阵
        """
        # 找到最近的 2 的幂次
        k = int(np.ceil(np.log2(max(n, 1))))
        size = 2 ** k
        
        # 递归构造 Hadamard 矩阵
        H = np.array([[1.0]])
        for _ in range(k):
            H = np.block([[H, H], [H, -H]])
        
        # 归一化
        H = H / np.sqrt(size)
        
        # 截取到所需大小
        return H[:n, :n]
    
    def _generate_random_rotation(self, n: int) -> np.ndarray:
        """生成随机正交矩阵 (QR 分解方法)。
        
        Args:
            n: 矩阵维度
            
        Returns:
            随机正交矩阵
        """
        np.random.seed(self.seed)
        A = np.random.randn(n, n)
        Q, R = np.linalg.qr(A)
        # 确保行列式为 1 (proper rotation)
        D = np.diag(np.sign(np.diag(R)))
        return Q @ D
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """应用正交变换。
        
        Args:
            x: 输入张量 [..., dim]
            
        Returns:
            变换后的张量
        """
        return x @ self.rotation_matrix
    
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """应用逆变换。
        
        Args:
            x: 变换后的张量
            
        Returns:
            恢复的张量
        """
        return x @ self.rotation_matrix.T
    
    def process(
        self,
        x: np.ndarray,
        quantizer: FP8Quantizer
    ) -> np.ndarray:
        """完整的 incoherent processing 流程。
        
        Args:
            x: 输入张量
            quantizer: FP8 量化器
            
        Returns:
            处理后的张量
        """
        # 变换
        x_transformed = self.transform(x)
        
        # 量化
        x_quantized, _ = quantizer.quantize(x_transformed)
        
        # 逆变换
        x_output = self.inverse_transform(x_quantized)
        
        return x_output


# =============================================================================
# Flash Attention 主类实现
# =============================================================================

class FlashAttentionBase(ABC):
    """Flash Attention 抽象基类 - 消除重复代码。"""
    
    def __init__(self, config: Optional[FlashAttentionConfig] = None):
        self.config = config or FlashAttentionConfig()
        self.online_softmax = OnlineSoftmax()
    
    @abstractmethod
    def _forward_single_head(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """单头前向传播 - 子类实现。"""
        pass
    
    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """前向传播 - 统一的多头处理。"""
        validate_attention_inputs(query, key, value, mask)
        
        if query.ndim == 4:
            # 多头: [batch, heads, seq, dim]
            batch, heads, seq_q, head_dim = query.shape
            outputs = []
            for h in range(heads):
                m_h = mask[:, h] if mask is not None and mask.ndim == 4 else mask
                out_h = self._forward_single_head(
                    query[:, h], key[:, h], value[:, h], m_h
                )
                outputs.append(out_h)
            return np.stack(outputs, axis=1)
        else:
            return self._forward_single_head(query, key, value, mask)
    
    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)


class FlashAttentionV1(FlashAttentionBase):
    """Flash Attention V1 实现 - 分块计算 + 重计算。"""
    
    def __init__(self, config: Optional[FlashAttentionConfig] = None):
        super().__init__(config)
        self.blockwise_attention = BlockwiseAttention(self.config)
    
    def _forward_single_head(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        output, _ = self.blockwise_attention.compute(query, key, value, mask)
        return output


class FlashAttentionV2(FlashAttentionBase):
    """Flash Attention V2 实现 - 优化并行 + 减少非 GEMM 操作。"""
    
    def _forward_single_head(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """单头前向传播 - V2 外层遍历 Q 块优化。"""
        batch_size, seq_q, head_dim = query.shape
        seq_kv = key.shape[1]
        
        Br = self.config.block_size_q
        Bc = self.config.block_size_kv
        scale = compute_scale(head_dim, self.config.softmax_scale)
        
        output = np.zeros((batch_size, seq_q, head_dim), dtype=np.float32)
        num_blocks_q = (seq_q + Br - 1) // Br
        num_blocks_kv = (seq_kv + Bc - 1) // Bc
        
        for i in range(num_blocks_q):
            q_start = i * Br
            q_end = min((i + 1) * Br, seq_q)
            q_block = query[:, q_start:q_end, :]
            
            # 初始化当前 Q 块的状态
            block_len = q_end - q_start
            state = self.online_softmax.init_state(batch_size, block_len, head_dim)
            
            # 内层循环遍历 KV 块
            for j in range(num_blocks_kv):
                kv_start = j * Bc
                kv_end = min((j + 1) * Bc, seq_kv)
                
                k_block = key[:, kv_start:kv_end, :]
                v_block = value[:, kv_start:kv_end, :]
                
                # 计算分数
                scores = np.einsum("bqd,bkd->bqk", q_block, k_block) * scale
                
                # 掩码处理
                block_mask = None
                if mask is not None:
                    block_mask = mask[:, q_start:q_end, kv_start:kv_end]
                
                # 因果掩码
                if self.config.causal:
                    causal = create_causal_mask(q_start, q_end, kv_start, kv_end, batch_size)
                    block_mask = block_mask & causal if block_mask is not None else causal
                
                # 更新状态
                state = self.online_softmax.update(state, scores, v_block, block_mask)
            
            # 完成当前 Q 块
            output[:, q_start:q_end, :] = self.online_softmax.finalize(state)
        
        return output


class FlashAttentionV3(FlashAttentionBase):
    """Flash Attention V3 实现 - Warp-Specialization + FP8。"""
    
    def __init__(self, config: Optional[FlashAttentionConfig] = None):
        super().__init__(config or FlashAttentionConfig(
            use_warp_specialization=True,
            scheduling_mode=SchedulingMode.PINGPONG
        ))
        self.scheduler = WarpScheduler(self.config)
        self.quantizer = FP8Quantizer(block_size=self.config.fp8_block_size) if self.config.use_fp8_quantization else None
        self.incoherent_processor = None
    
    def _forward_single_head(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """单头前向传播 - V3 Pingpong 调度。"""
        batch_size, seq_q, head_dim = query.shape
        scale = compute_scale(head_dim, self.config.softmax_scale)
        
        # FP8 量化
        if self.config.use_fp8_quantization and self.quantizer is not None:
            query, key, value = self._apply_fp8(query, key, value, head_dim)
        
        # 根据调度模式选择执行路径
        if self.config.scheduling_mode == SchedulingMode.PINGPONG:
            return self._forward_pingpong(query, key, value, scale, mask)
        return self._forward_sync(query, key, value, scale, mask)
    
    def _apply_fp8(self, query, key, value, head_dim):
        """应用 FP8 量化。"""
        if self.config.use_incoherent_processing:
            if self.incoherent_processor is None or self.incoherent_processor.dim != head_dim:
                self.incoherent_processor = IncoherentProcessor(head_dim)
            return (
                self.incoherent_processor.process(query, self.quantizer),
                self.incoherent_processor.process(key, self.quantizer),
                self.incoherent_processor.process(value, self.quantizer)
            )
        return self.quantizer.quantize(query)[0], self.quantizer.quantize(key)[0], self.quantizer.quantize(value)[0]
    
    def _forward_pingpong(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        scale: float,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Pingpong 调度的前向传播。"""
        batch_size, seq_q, head_dim = query.shape
        seq_kv = key.shape[1]
        Br = self.config.block_size_q
        
        output = np.zeros((batch_size, seq_q, head_dim), dtype=np.float32)
        num_blocks_q = (seq_q + Br - 1) // Br
        self.scheduler.reset_stats()
        
        for i in range(num_blocks_q):
            q_start = i * Br
            q_end = min((i + 1) * Br, seq_q)
            q_block = query[:, q_start:q_end, :]
            
            block_mask = mask[:, q_start:q_end, :] if mask is not None else None
            
            if self.config.causal:
                causal = create_causal_mask(q_start, q_end, 0, seq_kv, batch_size)
                block_mask = block_mask & causal if block_mask is not None else causal
            
            block_output = self.scheduler.schedule_pingpong(q_block, key, value, scale, block_mask)
            output[:, q_start:q_end, :] = block_output
        
        return output
    
    def _forward_sync(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        scale: float,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """同步执行的前向传播 (回退模式)。"""
        blockwise = BlockwiseAttention(self.config)
        output, _ = blockwise.compute(query, key, value, mask)
        return output
    
    def get_scheduler_stats(self) -> Dict[str, int]:
        """获取调度器统计信息。"""
        return self.scheduler.get_stats()


# =============================================================================
# 工具函数
# =============================================================================

def standard_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    scale: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    causal: bool = False
) -> np.ndarray:
    """标准注意力实现 (用于对比验证)。"""
    batch_size, seq_q, head_dim = query.shape
    seq_kv = key.shape[1]
    scale = compute_scale(head_dim, scale)
    
    scores = np.einsum("bqd,bkd->bqk", query, key) * scale
    
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    
    if causal:
        causal_mask = create_causal_mask(0, seq_q, 0, seq_kv, batch_size)[0]  # 取单个 batch
        scores = np.where(causal_mask, scores, -np.inf)
    
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attention_weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + Constants.MIN_SCALE)
    
    return np.einsum("bqk,bkd->bqd", attention_weights, value)


def compute_attention_flops(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    head_dim: int,
    causal: bool = False
) -> Dict[str, int]:
    """计算注意力机制的 FLOPs。"""
    base = batch_size * num_heads * seq_len_q * seq_len_kv
    qk_flops = 2 * base * head_dim
    softmax_flops = Constants.SOFTMAX_OPS_PER_ELEM * base
    pv_flops = 2 * base * head_dim
    
    if causal:
        qk_flops, softmax_flops, pv_flops = qk_flops // 2, softmax_flops // 2, pv_flops // 2
    
    return {"qk_gemm": qk_flops, "softmax": softmax_flops, "pv_gemm": pv_flops, "total": qk_flops + softmax_flops + pv_flops}


def create_flash_attention(
    version: str = "v3",
    block_size: int = 128,
    causal: bool = False,
    use_fp8: bool = False,
    **kwargs
) -> Union[FlashAttentionV1, FlashAttentionV2, FlashAttentionV3]:
    """创建 Flash Attention 实例的工厂函数。
    
    Args:
        version: 版本 ("v1", "v2", "v3")
        block_size: 分块大小
        causal: 是否因果掩码
        use_fp8: 是否使用 FP8 量化 (仅 V3)
        **kwargs: 其他配置参数
        
    Returns:
        Flash Attention 实例
        
    Example:
        >>> attn = create_flash_attention("v3", block_size=64, causal=True)
        >>> output = attn(query, key, value)
    """
    config = FlashAttentionConfig(
        block_size_q=block_size,
        block_size_kv=block_size,
        causal=causal,
        use_fp8_quantization=use_fp8,
        **kwargs
    )
    
    version = version.lower()
    if version == "v1":
        return FlashAttentionV1(config)
    elif version == "v2":
        return FlashAttentionV2(config)
    elif version == "v3":
        return FlashAttentionV3(config)
    else:
        raise ValueError(f"Unknown version: {version}. Choose from 'v1', 'v2', 'v3'")
