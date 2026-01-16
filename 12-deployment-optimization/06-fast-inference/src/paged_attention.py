"""
PagedAttention: 分页注意力机制实现

============================================================
核心思想 (Core Idea)
============================================================
PagedAttention 借鉴操作系统虚拟内存的分页机制，将 KV Cache 分割成
固定大小的块 (Block)，实现非连续内存存储和按需分配，解决 LLM 推理
中 KV Cache 内存碎片化和浪费问题。

============================================================
关键创新 (Key Innovations)
============================================================
1. 块级内存管理: KV Cache 按块分配，消除内存碎片
2. 按需分配: 仅在生成新 token 时分配内存
3. Copy-on-Write: Beam Search 等场景共享前缀块
4. 内存换入换出: 支持 GPU-CPU 内存交换

============================================================
内存优化效果 (Memory Optimization)
============================================================
传统方法: 预分配 max_seq_len 的连续内存
    - 内存浪费: 平均 60-80% (短序列)
    - 无法动态调整

PagedAttention:
    - 内存利用率: >95%
    - 支持更大批次和更长序列
    - 内存碎片: <4% (块大小内部碎片)

============================================================
参考文献 (References)
============================================================
[1] Kwon, W., et al. (2023). Efficient Memory Management for Large 
    Language Model Serving with PagedAttention. SOSP 2023.
[2] vLLM: https://github.com/vllm-project/vllm
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

__all__ = [
    "PagedAttentionConfig",
    "BlockStatus",
    "KVBlock",
    "BlockTable",
    "BlockAllocator",
    "PagedKVCache",
    "PagedAttention",
    "create_paged_attention",
    "compute_num_blocks_needed",
]


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class PagedAttentionConfig:
    """PagedAttention 配置类。
    
    Args:
        block_size: 每个块存储的 token 数量 (默认 16)
        num_blocks: 总块数量 (物理块池大小)
        num_layers: Transformer 层数
        num_heads: 注意力头数
        head_dim: 每个头的维度
        dtype: 数据类型 (float16/float32/bfloat16)
        device: 设备 (cpu/cuda)
        enable_swap: 是否启用 GPU-CPU 换入换出
        swap_space_gb: CPU 交换空间大小 (GB)
        enable_prefix_caching: 是否启用前缀缓存
    """
    block_size: int = 16
    num_blocks: int = 1024
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype: str = "float16"
    device: str = "cpu"
    enable_swap: bool = False
    swap_space_gb: float = 4.0
    enable_prefix_caching: bool = False
    
    def __post_init__(self):
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
    
    @property
    def block_memory_bytes(self) -> int:
        """计算单个块的内存占用 (字节)"""
        dtype_size = 2 if self.dtype in ("float16", "bfloat16") else 4
        # K 和 V 各占一份
        return 2 * self.block_size * self.num_heads * self.head_dim * dtype_size
    
    @property
    def total_memory_bytes(self) -> int:
        """计算总内存占用 (字节)"""
        return self.num_blocks * self.num_layers * self.block_memory_bytes


class BlockStatus(Enum):
    """块状态枚举"""
    FREE = auto()       # 空闲，可分配
    ALLOCATED = auto()  # 已分配，正在使用
    SWAPPED = auto()    # 已换出到 CPU
    CACHED = auto()     # 缓存中 (前缀缓存)


# =============================================================================
# KV Block 实现
# =============================================================================

@dataclass
class KVBlock:
    """KV Cache 块。
    
    每个块存储固定数量 token 的 Key 和 Value 向量。
    
    Attributes:
        block_id: 物理块 ID
        block_size: 块大小 (token 数)
        num_heads: 注意力头数
        head_dim: 头维度
        status: 块状态
        ref_count: 引用计数 (用于 Copy-on-Write)
        num_tokens: 当前存储的 token 数量
        key_cache: Key 缓存 [block_size, num_heads, head_dim]
        value_cache: Value 缓存 [block_size, num_heads, head_dim]
    """
    block_id: int
    block_size: int
    num_heads: int
    head_dim: int
    status: BlockStatus = BlockStatus.FREE
    ref_count: int = 0
    num_tokens: int = 0
    key_cache: Optional[np.ndarray] = None
    value_cache: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.key_cache is None:
            self.key_cache = np.zeros(
                (self.block_size, self.num_heads, self.head_dim),
                dtype=np.float32
            )
        if self.value_cache is None:
            self.value_cache = np.zeros(
                (self.block_size, self.num_heads, self.head_dim),
                dtype=np.float32
            )
    
    def is_full(self) -> bool:
        """检查块是否已满"""
        return self.num_tokens >= self.block_size
    
    def is_empty(self) -> bool:
        """检查块是否为空"""
        return self.num_tokens == 0
    
    def remaining_slots(self) -> int:
        """返回剩余可用槽位数"""
        return self.block_size - self.num_tokens
    
    def allocate(self) -> None:
        """分配块"""
        self.status = BlockStatus.ALLOCATED
        self.ref_count = 1
    
    def free(self) -> None:
        """释放块"""
        self.status = BlockStatus.FREE
        self.ref_count = 0
        self.num_tokens = 0
        self.key_cache.fill(0)
        self.value_cache.fill(0)
    
    def add_reference(self) -> None:
        """增加引用计数 (Copy-on-Write)"""
        self.ref_count += 1
    
    def remove_reference(self) -> bool:
        """减少引用计数，返回是否应该释放"""
        self.ref_count -= 1
        return self.ref_count <= 0
    
    def append_kv(
        self,
        key: np.ndarray,
        value: np.ndarray
    ) -> int:
        """追加 KV 对到块中。
        
        Args:
            key: Key 向量 [num_tokens, num_heads, head_dim]
            value: Value 向量 [num_tokens, num_heads, head_dim]
            
        Returns:
            实际写入的 token 数量
        """
        num_new_tokens = key.shape[0]
        available = self.remaining_slots()
        num_to_write = min(num_new_tokens, available)
        
        if num_to_write > 0:
            start = self.num_tokens
            end = start + num_to_write
            self.key_cache[start:end] = key[:num_to_write]
            self.value_cache[start:end] = value[:num_to_write]
            self.num_tokens = end
        
        return num_to_write
    
    def get_kv(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取有效的 KV 缓存。
        
        Returns:
            (key_cache, value_cache) 仅包含有效 token
        """
        return (
            self.key_cache[:self.num_tokens],
            self.value_cache[:self.num_tokens]
        )
    
    def copy_from(self, other: "KVBlock") -> None:
        """从另一个块复制数据 (Copy-on-Write)"""
        np.copyto(self.key_cache, other.key_cache)
        np.copyto(self.value_cache, other.value_cache)
        self.num_tokens = other.num_tokens


# =============================================================================
# Block Table 实现
# =============================================================================

class BlockTable:
    """块表：管理逻辑块到物理块的映射。
    
    每个序列维护一个块表，记录其使用的物理块列表。
    类似于操作系统的页表。
    
    Attributes:
        sequence_id: 序列 ID
        block_ids: 物理块 ID 列表 (按逻辑顺序)
        num_tokens: 序列当前的 token 总数
    """
    
    def __init__(self, sequence_id: int, block_size: int):
        self.sequence_id = sequence_id
        self.block_size = block_size
        self.block_ids: List[int] = []
        self.num_tokens: int = 0
    
    def __len__(self) -> int:
        return len(self.block_ids)
    
    def __getitem__(self, idx: int) -> int:
        return self.block_ids[idx]
    
    def append_block(self, block_id: int) -> None:
        """追加物理块"""
        self.block_ids.append(block_id)
    
    def get_last_block_id(self) -> Optional[int]:
        """获取最后一个块的 ID"""
        return self.block_ids[-1] if self.block_ids else None
    
    def get_num_blocks(self) -> int:
        """获取块数量"""
        return len(self.block_ids)
    
    def get_logical_block_index(self, token_position: int) -> int:
        """根据 token 位置获取逻辑块索引"""
        return token_position // self.block_size
    
    def get_block_offset(self, token_position: int) -> int:
        """根据 token 位置获取块内偏移"""
        return token_position % self.block_size
    
    def copy(self) -> "BlockTable":
        """复制块表 (用于 fork)"""
        new_table = BlockTable(self.sequence_id, self.block_size)
        new_table.block_ids = self.block_ids.copy()
        new_table.num_tokens = self.num_tokens
        return new_table


# =============================================================================
# Block Allocator 实现
# =============================================================================

class BlockAllocator:
    """块分配器：管理物理块的分配和释放。
    
    维护空闲块列表，支持分配、释放、Copy-on-Write 等操作。
    
    Attributes:
        config: PagedAttention 配置
        blocks: 所有物理块的列表
        free_blocks: 空闲块 ID 集合
        num_allocated: 已分配块数量
    """
    
    def __init__(self, config: PagedAttentionConfig):
        self.config = config
        self.blocks: List[KVBlock] = []
        self.free_blocks: Set[int] = set()
        self.num_allocated: int = 0
        
        # 初始化物理块池
        self._init_blocks()
    
    def _init_blocks(self) -> None:
        """初始化物理块池"""
        for i in range(self.config.num_blocks):
            block = KVBlock(
                block_id=i,
                block_size=self.config.block_size,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                status=BlockStatus.FREE
            )
            self.blocks.append(block)
            self.free_blocks.add(i)
    
    def get_num_free_blocks(self) -> int:
        """获取空闲块数量"""
        return len(self.free_blocks)
    
    def get_num_allocated_blocks(self) -> int:
        """获取已分配块数量"""
        return self.num_allocated
    
    def can_allocate(self, num_blocks: int = 1) -> bool:
        """检查是否可以分配指定数量的块"""
        return len(self.free_blocks) >= num_blocks
    
    def allocate(self) -> Optional[KVBlock]:
        """分配一个空闲块。
        
        Returns:
            分配的块，如果没有空闲块则返回 None
        """
        if not self.free_blocks:
            return None
        
        block_id = self.free_blocks.pop()
        block = self.blocks[block_id]
        block.allocate()
        self.num_allocated += 1
        
        return block
    
    def allocate_n(self, n: int) -> List[KVBlock]:
        """分配 n 个空闲块。
        
        Args:
            n: 需要分配的块数量
            
        Returns:
            分配的块列表，如果空闲块不足则返回空列表
        """
        if not self.can_allocate(n):
            return []
        
        allocated = []
        for _ in range(n):
            block = self.allocate()
            if block:
                allocated.append(block)
        
        return allocated
    
    def free(self, block_id: int) -> None:
        """释放一个块。
        
        Args:
            block_id: 要释放的块 ID
        """
        if block_id < 0 or block_id >= len(self.blocks):
            raise ValueError(f"Invalid block_id: {block_id}")
        
        block = self.blocks[block_id]
        if block.status == BlockStatus.FREE:
            return  # 已经是空闲状态
        
        if block.remove_reference():
            block.free()
            self.free_blocks.add(block_id)
            self.num_allocated -= 1
    
    def free_blocks_for_sequence(self, block_table: BlockTable) -> None:
        """释放序列使用的所有块。
        
        Args:
            block_table: 序列的块表
        """
        for block_id in block_table.block_ids:
            self.free(block_id)
    
    def get_block(self, block_id: int) -> KVBlock:
        """获取指定 ID 的块"""
        if block_id < 0 or block_id >= len(self.blocks):
            raise ValueError(f"Invalid block_id: {block_id}")
        return self.blocks[block_id]
    
    def fork_block(self, block_id: int) -> Optional[int]:
        """Fork 一个块 (Copy-on-Write)。
        
        增加原块的引用计数，不实际复制数据。
        
        Args:
            block_id: 要 fork 的块 ID
            
        Returns:
            原块 ID (引用计数已增加)
        """
        if block_id < 0 or block_id >= len(self.blocks):
            return None
        
        block = self.blocks[block_id]
        block.add_reference()
        return block_id
    
    def copy_on_write(self, block_id: int) -> Optional[KVBlock]:
        """执行 Copy-on-Write。
        
        当需要修改一个共享块时，复制数据到新块。
        
        Args:
            block_id: 原块 ID
            
        Returns:
            新分配的块 (包含复制的数据)
        """
        if block_id < 0 or block_id >= len(self.blocks):
            return None
        
        old_block = self.blocks[block_id]
        
        # 如果只有一个引用，不需要复制
        if old_block.ref_count <= 1:
            return old_block
        
        # 分配新块
        new_block = self.allocate()
        if new_block is None:
            return None
        
        # 复制数据
        new_block.copy_from(old_block)
        
        # 减少原块引用
        old_block.remove_reference()
        
        return new_block
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用统计"""
        block_memory = self.config.block_memory_bytes
        return {
            "total_blocks": self.config.num_blocks,
            "allocated_blocks": self.num_allocated,
            "free_blocks": len(self.free_blocks),
            "utilization": self.num_allocated / self.config.num_blocks if self.config.num_blocks > 0 else 0,
            "total_memory_mb": self.config.num_blocks * block_memory / (1024 * 1024),
            "used_memory_mb": self.num_allocated * block_memory / (1024 * 1024),
        }


# =============================================================================
# Paged KV Cache 实现
# =============================================================================

class PagedKVCache:
    """分页 KV Cache：管理多个序列的 KV 缓存。
    
    为每个序列维护独立的块表，支持动态分配、释放和共享。
    
    Attributes:
        config: PagedAttention 配置
        allocator: 块分配器
        block_tables: 序列 ID -> 块表的映射
        layer_caches: 每层的块分配器 (多层共享物理块池)
    """
    
    def __init__(self, config: PagedAttentionConfig):
        self.config = config
        self.allocator = BlockAllocator(config)
        self.block_tables: Dict[int, BlockTable] = {}
        self._next_sequence_id = 0
    
    def allocate_sequence(self, sequence_id: Optional[int] = None) -> int:
        """为新序列分配块表。
        
        Args:
            sequence_id: 可选的序列 ID，如果不提供则自动生成
            
        Returns:
            序列 ID
        """
        if sequence_id is None:
            sequence_id = self._next_sequence_id
            self._next_sequence_id += 1
        
        if sequence_id in self.block_tables:
            raise ValueError(f"Sequence {sequence_id} already exists")
        
        self.block_tables[sequence_id] = BlockTable(
            sequence_id=sequence_id,
            block_size=self.config.block_size
        )
        return sequence_id
    
    def free_sequence(self, sequence_id: int) -> None:
        """释放序列的所有资源。
        
        Args:
            sequence_id: 序列 ID
        """
        if sequence_id not in self.block_tables:
            return
        
        block_table = self.block_tables[sequence_id]
        self.allocator.free_blocks_for_sequence(block_table)
        del self.block_tables[sequence_id]
    
    def append_tokens(
        self,
        sequence_id: int,
        keys: np.ndarray,
        values: np.ndarray
    ) -> bool:
        """向序列追加 KV 缓存。
        
        Args:
            sequence_id: 序列 ID
            keys: Key 张量 [num_tokens, num_heads, head_dim]
            values: Value 张量 [num_tokens, num_heads, head_dim]
            
        Returns:
            是否成功追加
        """
        if sequence_id not in self.block_tables:
            raise ValueError(f"Sequence {sequence_id} not found")
        
        block_table = self.block_tables[sequence_id]
        num_tokens = keys.shape[0]
        tokens_written = 0
        
        while tokens_written < num_tokens:
            # 获取或分配当前块
            current_block = self._get_or_allocate_block(block_table)
            if current_block is None:
                return False  # 内存不足
            
            # 写入数据
            remaining_keys = keys[tokens_written:]
            remaining_values = values[tokens_written:]
            written = current_block.append_kv(remaining_keys, remaining_values)
            tokens_written += written
            block_table.num_tokens += written
        
        return True
    
    def _get_or_allocate_block(self, block_table: BlockTable) -> Optional[KVBlock]:
        """获取当前块或分配新块。"""
        if block_table.block_ids:
            last_block_id = block_table.get_last_block_id()
            last_block = self.allocator.get_block(last_block_id)
            
            # 检查是否需要 Copy-on-Write
            if last_block.ref_count > 1 and not last_block.is_full():
                new_block = self.allocator.copy_on_write(last_block_id)
                if new_block is None:
                    return None
                block_table.block_ids[-1] = new_block.block_id
                return new_block
            
            if not last_block.is_full():
                return last_block
        
        # 分配新块
        new_block = self.allocator.allocate()
        if new_block is None:
            return None
        
        block_table.append_block(new_block.block_id)
        return new_block
    
    def get_kv_cache(
        self,
        sequence_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取序列的完整 KV 缓存。
        
        Args:
            sequence_id: 序列 ID
            
        Returns:
            (keys, values) 完整的 KV 缓存
        """
        if sequence_id not in self.block_tables:
            raise ValueError(f"Sequence {sequence_id} not found")
        
        block_table = self.block_tables[sequence_id]
        
        if not block_table.block_ids:
            return (
                np.zeros((0, self.config.num_heads, self.config.head_dim)),
                np.zeros((0, self.config.num_heads, self.config.head_dim))
            )
        
        all_keys = []
        all_values = []
        
        for block_id in block_table.block_ids:
            block = self.allocator.get_block(block_id)
            k, v = block.get_kv()
            all_keys.append(k)
            all_values.append(v)
        
        return np.concatenate(all_keys, axis=0), np.concatenate(all_values, axis=0)
    
    def fork_sequence(self, source_id: int, target_id: Optional[int] = None) -> int:
        """Fork 序列 (用于 Beam Search)。
        
        使用 Copy-on-Write，共享前缀块。
        
        Args:
            source_id: 源序列 ID
            target_id: 目标序列 ID (可选)
            
        Returns:
            新序列 ID
        """
        if source_id not in self.block_tables:
            raise ValueError(f"Source sequence {source_id} not found")
        
        if target_id is None:
            target_id = self._next_sequence_id
            self._next_sequence_id += 1
        
        source_table = self.block_tables[source_id]
        target_table = source_table.copy()
        target_table.sequence_id = target_id
        
        # 增加所有块的引用计数
        for block_id in target_table.block_ids:
            self.allocator.fork_block(block_id)
        
        self.block_tables[target_id] = target_table
        return target_id
    
    def get_num_sequences(self) -> int:
        """获取当前序列数量"""
        return len(self.block_tables)
    
    def get_sequence_length(self, sequence_id: int) -> int:
        """获取序列长度"""
        if sequence_id not in self.block_tables:
            return 0
        return self.block_tables[sequence_id].num_tokens
    
    def can_append(self, sequence_id: int, num_tokens: int) -> bool:
        """检查是否可以追加指定数量的 token。"""
        if sequence_id not in self.block_tables:
            return False
        
        block_table = self.block_tables[sequence_id]
        current_tokens = block_table.num_tokens
        
        # 计算需要的块数
        blocks_needed = compute_num_blocks_needed(
            current_tokens + num_tokens,
            self.config.block_size
        ) - len(block_table.block_ids)
        
        return self.allocator.can_allocate(max(0, blocks_needed))


# =============================================================================
# PagedAttention 主类
# =============================================================================

class PagedAttention:
    """分页注意力计算。
    
    结合 PagedKVCache 实现高效的注意力计算，支持：
    1. 分块读取 KV Cache
    2. 多序列批量计算
    3. 因果掩码
    
    Attributes:
        config: PagedAttention 配置
        kv_cache: 分页 KV Cache
        scale: 注意力缩放因子
    """
    
    def __init__(self, config: PagedAttentionConfig):
        self.config = config
        self.kv_cache = PagedKVCache(config)
        self.scale = 1.0 / math.sqrt(config.head_dim)
    
    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        sequence_ids: List[int],
        is_prefill: bool = False,
        causal: bool = True
    ) -> np.ndarray:
        """前向传播。
        
        Args:
            query: Query 张量 [batch, num_heads, seq_len, head_dim]
            key: Key 张量 [batch, num_heads, seq_len, head_dim]
            value: Value 张量 [batch, num_heads, seq_len, head_dim]
            sequence_ids: 每个 batch 元素对应的序列 ID
            is_prefill: 是否是 prefill 阶段
            causal: 是否使用因果掩码
            
        Returns:
            输出张量 [batch, num_heads, seq_len, head_dim]
        """
        batch_size = query.shape[0]
        outputs = []
        
        for b in range(batch_size):
            seq_id = sequence_ids[b]
            q = query[b]  # [num_heads, seq_len, head_dim]
            k = key[b]
            v = value[b]
            
            # 更新 KV Cache
            # 转换为 [seq_len, num_heads, head_dim] 格式
            k_transposed = np.transpose(k, (1, 0, 2))
            v_transposed = np.transpose(v, (1, 0, 2))
            
            if seq_id not in self.kv_cache.block_tables:
                self.kv_cache.allocate_sequence(seq_id)
            
            self.kv_cache.append_tokens(seq_id, k_transposed, v_transposed)
            
            # 获取完整 KV Cache
            cached_k, cached_v = self.kv_cache.get_kv_cache(seq_id)
            # 转换回 [num_heads, seq_len, head_dim]
            cached_k = np.transpose(cached_k, (1, 0, 2))
            cached_v = np.transpose(cached_v, (1, 0, 2))
            
            # 计算注意力
            out = self._compute_attention(q, cached_k, cached_v, causal)
            outputs.append(out)
        
        return np.stack(outputs, axis=0)
    
    def _compute_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        causal: bool = True
    ) -> np.ndarray:
        """计算单个序列的注意力。
        
        Args:
            query: [num_heads, q_len, head_dim]
            key: [num_heads, kv_len, head_dim]
            value: [num_heads, kv_len, head_dim]
            causal: 是否因果掩码
            
        Returns:
            [num_heads, q_len, head_dim]
        """
        # 计算注意力分数: [num_heads, q_len, kv_len]
        scores = np.einsum("hqd,hkd->hqk", query, key) * self.scale
        
        # 应用因果掩码
        if causal:
            q_len = query.shape[1]
            kv_len = key.shape[1]
            # 因果掩码: query 位置 i 只能看到 key 位置 <= kv_len - q_len + i
            mask = np.triu(
                np.ones((q_len, kv_len), dtype=bool),
                k=kv_len - q_len + 1
            )
            scores = np.where(mask, -np.inf, scores)
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attention_weights = scores_exp / (np.sum(scores_exp, axis=-1, keepdims=True) + 1e-9)
        
        # 加权求和
        output = np.einsum("hqk,hkd->hqd", attention_weights, value)
        return output
    
    def decode_step(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        sequence_id: int
    ) -> np.ndarray:
        """单步解码 (生成单个 token)。
        
        Args:
            query: [num_heads, 1, head_dim]
            key: [num_heads, 1, head_dim]
            value: [num_heads, 1, head_dim]
            sequence_id: 序列 ID
            
        Returns:
            [num_heads, 1, head_dim]
        """
        # 追加新的 KV
        k_transposed = np.transpose(key, (1, 0, 2))
        v_transposed = np.transpose(value, (1, 0, 2))
        self.kv_cache.append_tokens(sequence_id, k_transposed, v_transposed)
        
        # 获取完整 KV Cache
        cached_k, cached_v = self.kv_cache.get_kv_cache(sequence_id)
        cached_k = np.transpose(cached_k, (1, 0, 2))
        cached_v = np.transpose(cached_v, (1, 0, 2))
        
        # 计算注意力 (decode 阶段不需要因果掩码，因为只有一个 query)
        return self._compute_attention(query, cached_k, cached_v, causal=False)
    
    def clear_sequence(self, sequence_id: int) -> None:
        """清除序列的 KV Cache"""
        self.kv_cache.free_sequence(sequence_id)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用统计"""
        return self.kv_cache.allocator.get_memory_usage()


# =============================================================================
# 工具函数
# =============================================================================

def compute_num_blocks_needed(num_tokens: int, block_size: int) -> int:
    """计算存储指定数量 token 需要的块数。
    
    Args:
        num_tokens: token 数量
        block_size: 块大小
        
    Returns:
        需要的块数
    """
    if num_tokens <= 0:
        return 0
    return (num_tokens + block_size - 1) // block_size


def compute_memory_for_sequence(
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "float16"
) -> int:
    """计算序列的 KV Cache 内存占用 (字节)。
    
    Args:
        seq_len: 序列长度
        num_layers: 层数
        num_heads: 头数
        head_dim: 头维度
        dtype: 数据类型
        
    Returns:
        内存占用 (字节)
    """
    dtype_size = 2 if dtype in ("float16", "bfloat16") else 4
    # K 和 V 各占一份，每层都有
    return 2 * seq_len * num_layers * num_heads * head_dim * dtype_size


def estimate_max_batch_size(
    available_memory_gb: float,
    max_seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "float16"
) -> int:
    """估算给定内存下的最大批次大小。
    
    Args:
        available_memory_gb: 可用内存 (GB)
        max_seq_len: 最大序列长度
        num_layers: 层数
        num_heads: 头数
        head_dim: 头维度
        dtype: 数据类型
        
    Returns:
        最大批次大小
    """
    available_bytes = available_memory_gb * 1024 * 1024 * 1024
    memory_per_seq = compute_memory_for_sequence(
        max_seq_len, num_layers, num_heads, head_dim, dtype
    )
    return max(1, int(available_bytes / memory_per_seq))


def create_paged_attention(
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    block_size: int = 16,
    num_blocks: int = 1024,
    dtype: str = "float16",
    **kwargs
) -> PagedAttention:
    """创建 PagedAttention 实例的工厂函数。
    
    Args:
        num_layers: Transformer 层数
        num_heads: 注意力头数
        head_dim: 头维度
        block_size: 块大小
        num_blocks: 总块数
        dtype: 数据类型
        **kwargs: 其他配置参数
        
    Returns:
        PagedAttention 实例
        
    Example:
        >>> attn = create_paged_attention(num_layers=12, num_heads=12, head_dim=64)
        >>> seq_id = attn.kv_cache.allocate_sequence()
        >>> output = attn.forward(query, key, value, [seq_id])
    """
    config = PagedAttentionConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
        dtype=dtype,
        **kwargs
    )
    return PagedAttention(config)


class MultiLayerPagedKVCache:
    """多层分页 KV Cache。
    
    为 Transformer 的每一层维护独立的 KV Cache。
    
    Attributes:
        config: 配置
        layer_caches: 每层的 PagedKVCache
    """
    
    def __init__(self, config: PagedAttentionConfig):
        self.config = config
        self.layer_caches: List[PagedKVCache] = [
            PagedKVCache(config) for _ in range(config.num_layers)
        ]
        self._sequence_ids: Set[int] = set()
    
    def allocate_sequence(self, sequence_id: Optional[int] = None) -> int:
        """为所有层分配序列。"""
        # 使用第一层生成 ID
        seq_id = self.layer_caches[0].allocate_sequence(sequence_id)
        
        # 为其他层分配相同 ID
        for layer_cache in self.layer_caches[1:]:
            layer_cache.allocate_sequence(seq_id)
        
        self._sequence_ids.add(seq_id)
        return seq_id
    
    def free_sequence(self, sequence_id: int) -> None:
        """释放所有层的序列。"""
        for layer_cache in self.layer_caches:
            layer_cache.free_sequence(sequence_id)
        self._sequence_ids.discard(sequence_id)
    
    def append_tokens(
        self,
        layer_idx: int,
        sequence_id: int,
        keys: np.ndarray,
        values: np.ndarray
    ) -> bool:
        """向指定层追加 KV。"""
        if layer_idx < 0 or layer_idx >= len(self.layer_caches):
            raise ValueError(f"Invalid layer_idx: {layer_idx}")
        return self.layer_caches[layer_idx].append_tokens(sequence_id, keys, values)
    
    def get_kv_cache(
        self,
        layer_idx: int,
        sequence_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定层的 KV Cache。"""
        if layer_idx < 0 or layer_idx >= len(self.layer_caches):
            raise ValueError(f"Invalid layer_idx: {layer_idx}")
        return self.layer_caches[layer_idx].get_kv_cache(sequence_id)
    
    def fork_sequence(self, source_id: int, target_id: Optional[int] = None) -> int:
        """Fork 序列到所有层。"""
        # 使用第一层生成目标 ID
        new_id = self.layer_caches[0].fork_sequence(source_id, target_id)
        
        # Fork 到其他层
        for layer_cache in self.layer_caches[1:]:
            layer_cache.fork_sequence(source_id, new_id)
        
        self._sequence_ids.add(new_id)
        return new_id
    
    def get_total_memory_usage(self) -> Dict[str, Any]:
        """获取所有层的总内存使用。"""
        total_allocated = sum(
            cache.allocator.get_num_allocated_blocks()
            for cache in self.layer_caches
        )
        total_free = sum(
            cache.allocator.get_num_free_blocks()
            for cache in self.layer_caches
        )
        block_memory = self.config.block_memory_bytes
        
        return {
            "num_layers": self.config.num_layers,
            "total_allocated_blocks": total_allocated,
            "total_free_blocks": total_free,
            "total_memory_mb": (total_allocated + total_free) * block_memory / (1024 * 1024),
            "used_memory_mb": total_allocated * block_memory / (1024 * 1024),
            "num_sequences": len(self._sequence_ids),
        }
