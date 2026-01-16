"""
PagedAttention 模块单元测试

测试覆盖:
- PagedAttentionConfig 配置验证
- BlockStatus 枚举
- KVBlock 块操作
- BlockTable 块表管理
- BlockAllocator 块分配器
- PagedKVCache KV缓存
- PagedAttention 注意力计算
- MultiLayerPagedKVCache 多层缓存
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paged_attention import (
    PagedAttentionConfig,
    BlockStatus,
    KVBlock,
    BlockTable,
    BlockAllocator,
    PagedKVCache,
    PagedAttention,
    MultiLayerPagedKVCache,
    create_paged_attention,
)


class TestPagedAttentionConfig:
    """PagedAttentionConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PagedAttentionConfig()
        assert config.block_size == 16
        assert config.num_blocks == 1024
        assert config.num_heads == 32
        assert config.head_dim == 128
        assert config.num_layers == 32
        assert config.dtype == "float16"
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = PagedAttentionConfig(
            block_size=32,
            num_blocks=512,
            num_heads=16,
            head_dim=64,
            num_layers=24
        )
        assert config.block_size == 32
        assert config.num_blocks == 512
        assert config.num_heads == 16
        assert config.head_dim == 64
        assert config.num_layers == 24
    
    def test_invalid_block_size(self):
        """测试无效的块大小"""
        with pytest.raises(ValueError):
            PagedAttentionConfig(block_size=0)
        with pytest.raises(ValueError):
            PagedAttentionConfig(block_size=-1)
    
    def test_invalid_num_blocks(self):
        """测试无效的块数量"""
        with pytest.raises(ValueError):
            PagedAttentionConfig(num_blocks=0)
    
    def test_invalid_num_heads(self):
        """测试无效的头数"""
        with pytest.raises(ValueError):
            PagedAttentionConfig(num_heads=0)
    
    def test_invalid_head_dim(self):
        """测试无效的头维度"""
        with pytest.raises(ValueError):
            PagedAttentionConfig(head_dim=0)
    
    def test_block_memory_bytes(self):
        """测试块内存计算"""
        config = PagedAttentionConfig(
            block_size=16, num_heads=8, head_dim=64
        )
        # block_size * num_heads * head_dim * 2 (K+V) * 2 (float16)
        expected = 16 * 8 * 64 * 2 * 2
        assert config.block_memory_bytes == expected
    
    def test_total_memory_bytes(self):
        """测试总内存计算"""
        config = PagedAttentionConfig(
            block_size=16, num_blocks=100, num_heads=8, head_dim=64, num_layers=1
        )
        expected = config.block_memory_bytes * 100
        assert config.total_memory_bytes == expected


class TestBlockStatus:
    """BlockStatus 枚举测试"""
    
    def test_status_values(self):
        """测试状态值"""
        assert BlockStatus.FREE is not None
        assert BlockStatus.ALLOCATED is not None
        assert BlockStatus.CACHED is not None


class TestKVBlock:
    """KVBlock 测试"""
    
    def test_block_creation(self):
        """测试块创建"""
        block = KVBlock(block_id=0, block_size=16, num_heads=8, head_dim=64)
        assert block.block_id == 0
        assert block.block_size == 16
        assert block.num_heads == 8
        assert block.head_dim == 64
        assert block.num_tokens == 0
        assert block.ref_count == 0
        assert block.status == BlockStatus.FREE
    
    def test_block_allocation(self):
        """测试块分配"""
        block = KVBlock(block_id=1, block_size=16, num_heads=8, head_dim=64)
        block.allocate()
        assert block.status == BlockStatus.ALLOCATED
        assert block.ref_count == 1
    
    def test_block_free(self):
        """测试块释放"""
        block = KVBlock(block_id=2, block_size=16, num_heads=8, head_dim=64)
        block.allocate()
        block.free()
        assert block.status == BlockStatus.FREE
        assert block.ref_count == 0
        assert block.num_tokens == 0
    
    def test_is_full(self):
        """测试块是否已满"""
        block = KVBlock(block_id=3, block_size=4, num_heads=8, head_dim=64)
        block.allocate()
        assert not block.is_full()
        block.num_tokens = 4
        assert block.is_full()
    
    def test_is_empty(self):
        """测试块是否为空"""
        block = KVBlock(block_id=4, block_size=16, num_heads=8, head_dim=64)
        assert block.is_empty()
        block.num_tokens = 1
        assert not block.is_empty()
    
    def test_remaining_slots(self):
        """测试剩余槽位"""
        block = KVBlock(block_id=5, block_size=16, num_heads=8, head_dim=64)
        block.allocate()
        assert block.remaining_slots() == 16
        block.num_tokens = 10
        assert block.remaining_slots() == 6
    
    def test_add_reference(self):
        """测试增加引用"""
        block = KVBlock(block_id=6, block_size=16, num_heads=8, head_dim=64)
        block.allocate()
        block.add_reference()
        assert block.ref_count == 2
    
    def test_remove_reference(self):
        """测试减少引用"""
        block = KVBlock(block_id=7, block_size=16, num_heads=8, head_dim=64)
        block.allocate()
        block.add_reference()
        freed = block.remove_reference()
        assert not freed
        assert block.ref_count == 1
    
    def test_remove_reference_to_zero(self):
        """测试引用减到零"""
        block = KVBlock(block_id=8, block_size=16, num_heads=8, head_dim=64)
        block.allocate()
        freed = block.remove_reference()
        assert freed  # 返回 True 表示应该释放
        assert block.ref_count == 0
    
    def test_append_kv(self):
        """测试追加 KV"""
        block = KVBlock(block_id=9, block_size=16, num_heads=8, head_dim=64)
        block.allocate()
        
        # 形状: [num_tokens, num_heads, head_dim]
        k = np.random.randn(1, 8, 64).astype(np.float32)
        v = np.random.randn(1, 8, 64).astype(np.float32)
        
        num_written = block.append_kv(k, v)
        assert num_written == 1
        assert block.num_tokens == 1
    
    def test_append_kv_multiple(self):
        """测试追加多个 KV"""
        block = KVBlock(block_id=10, block_size=4, num_heads=8, head_dim=64)
        block.allocate()
        
        # 一次追加 4 个 token
        k = np.random.randn(4, 8, 64).astype(np.float32)
        v = np.random.randn(4, 8, 64).astype(np.float32)
        num_written = block.append_kv(k, v)
        
        assert num_written == 4
        assert block.is_full()
    
    def test_get_kv(self):
        """测试获取 KV"""
        block = KVBlock(block_id=11, block_size=16, num_heads=8, head_dim=64)
        block.allocate()
        
        # 形状: [num_tokens, num_heads, head_dim]
        k = np.random.randn(3, 8, 64).astype(np.float32)
        v = np.random.randn(3, 8, 64).astype(np.float32)
        
        block.append_kv(k, v)
        
        k_out, v_out = block.get_kv()
        assert k_out.shape[0] == block.num_tokens
        assert v_out.shape[0] == block.num_tokens
    
    def test_copy_from(self):
        """测试从另一个块复制"""
        block1 = KVBlock(block_id=12, block_size=16, num_heads=8, head_dim=64)
        block1.allocate()
        
        k = np.random.randn(2, 8, 64).astype(np.float32)
        v = np.random.randn(2, 8, 64).astype(np.float32)
        block1.append_kv(k, v)
        
        block2 = KVBlock(block_id=13, block_size=16, num_heads=8, head_dim=64)
        block2.allocate()
        block2.copy_from(block1)
        
        assert block2.num_tokens == block1.num_tokens
        np.testing.assert_array_equal(block2.key_cache, block1.key_cache)
        np.testing.assert_array_equal(block2.value_cache, block1.value_cache)


class TestBlockTable:
    """BlockTable 测试"""
    
    def test_table_creation(self):
        """测试表创建"""
        table = BlockTable(sequence_id=0, block_size=16)
        assert table.sequence_id == 0
        assert len(table) == 0
    
    def test_append_block(self):
        """测试追加块"""
        table = BlockTable(sequence_id=1, block_size=16)
        table.append_block(0)
        assert len(table) == 1
        assert table[0] == 0
    
    def test_get_last_block_id(self):
        """测试获取最后一个块 ID"""
        table = BlockTable(sequence_id=2, block_size=16)
        table.append_block(5)
        table.append_block(10)
        table.append_block(15)
        
        assert table.get_last_block_id() == 15
    
    def test_get_last_block_id_empty(self):
        """测试空表获取最后一个块 ID"""
        table = BlockTable(sequence_id=3, block_size=16)
        assert table.get_last_block_id() is None
    
    def test_get_num_blocks(self):
        """测试获取块数量"""
        table = BlockTable(sequence_id=4, block_size=16)
        assert table.get_num_blocks() == 0
        
        table.append_block(0)
        table.append_block(1)
        assert table.get_num_blocks() == 2
    
    def test_get_logical_block_index(self):
        """测试获取逻辑块索引"""
        table = BlockTable(sequence_id=5, block_size=16)
        assert table.get_logical_block_index(0) == 0
        assert table.get_logical_block_index(15) == 0
        assert table.get_logical_block_index(16) == 1
        assert table.get_logical_block_index(32) == 2
    
    def test_get_block_offset(self):
        """测试获取块内偏移"""
        table = BlockTable(sequence_id=6, block_size=16)
        assert table.get_block_offset(0) == 0
        assert table.get_block_offset(5) == 5
        assert table.get_block_offset(16) == 0
        assert table.get_block_offset(20) == 4
    
    def test_copy(self):
        """测试复制表"""
        table = BlockTable(sequence_id=7, block_size=16)
        table.append_block(1)
        table.append_block(2)
        table.num_tokens = 20
        
        copied = table.copy()
        assert copied.sequence_id == table.sequence_id
        assert len(copied) == len(table)
        assert copied.num_tokens == table.num_tokens


class TestBlockAllocator:
    """BlockAllocator 测试"""
    
    def test_allocator_creation(self):
        """测试分配器创建"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        assert allocator.get_num_free_blocks() == 100
        assert allocator.get_num_allocated_blocks() == 0
    
    def test_allocate_block(self):
        """测试分配块"""
        config = PagedAttentionConfig(num_blocks=10, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        
        block = allocator.allocate()
        assert block is not None
        assert block.status == BlockStatus.ALLOCATED
        assert allocator.get_num_free_blocks() == 9
        assert allocator.get_num_allocated_blocks() == 1
    
    def test_allocate_multiple_blocks(self):
        """测试分配多个块"""
        config = PagedAttentionConfig(num_blocks=10, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        
        blocks = allocator.allocate_n(5)
        assert len(blocks) == 5
        assert allocator.get_num_free_blocks() == 5
        assert allocator.get_num_allocated_blocks() == 5
    
    def test_free_block(self):
        """测试释放块"""
        config = PagedAttentionConfig(num_blocks=10, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        
        block = allocator.allocate()
        allocator.free(block.block_id)
        
        assert block.status == BlockStatus.FREE
        assert allocator.get_num_free_blocks() == 10
        assert allocator.get_num_allocated_blocks() == 0
    
    def test_allocate_when_empty(self):
        """测试无可用块时分配"""
        config = PagedAttentionConfig(num_blocks=2, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        
        allocator.allocate()
        allocator.allocate()
        
        block = allocator.allocate()
        assert block is None
    
    def test_can_allocate(self):
        """测试是否可以分配"""
        config = PagedAttentionConfig(num_blocks=5, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        
        assert allocator.can_allocate(5)
        assert not allocator.can_allocate(6)
        
        allocator.allocate_n(3)
        assert allocator.can_allocate(2)
        assert not allocator.can_allocate(3)
    
    def test_get_block(self):
        """测试获取块"""
        config = PagedAttentionConfig(num_blocks=10, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        
        block = allocator.allocate()
        retrieved = allocator.get_block(block.block_id)
        assert retrieved is block
    
    def test_fork_block(self):
        """测试 fork 块"""
        config = PagedAttentionConfig(num_blocks=10, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        
        block = allocator.allocate()
        initial_ref = block.ref_count
        
        forked_id = allocator.fork_block(block.block_id)
        assert forked_id == block.block_id
        assert block.ref_count == initial_ref + 1
    
    def test_copy_on_write(self):
        """测试 Copy-on-Write"""
        config = PagedAttentionConfig(num_blocks=10, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        
        block = allocator.allocate()
        block.add_reference()  # ref_count = 2
        
        new_block = allocator.copy_on_write(block.block_id)
        assert new_block is not None
        assert new_block.block_id != block.block_id
        assert block.ref_count == 1
    
    def test_get_memory_usage(self):
        """测试内存使用统计"""
        config = PagedAttentionConfig(num_blocks=10, block_size=16, num_heads=8, head_dim=64)
        allocator = BlockAllocator(config)
        
        allocator.allocate_n(5)
        
        usage = allocator.get_memory_usage()
        assert "total_blocks" in usage
        assert "free_blocks" in usage
        assert "allocated_blocks" in usage
        assert usage["allocated_blocks"] == 5


class TestPagedKVCache:
    """PagedKVCache 测试"""
    
    def test_cache_creation(self):
        """测试缓存创建"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        assert cache is not None
    
    def test_allocate_sequence(self):
        """测试分配序列"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        seq_id = cache.allocate_sequence()
        assert seq_id is not None
        assert cache.get_sequence_length(seq_id) == 0
    
    def test_allocate_sequence_with_id(self):
        """测试指定 ID 分配序列"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        seq_id = cache.allocate_sequence(sequence_id=42)
        assert seq_id == 42
    
    def test_append_tokens(self):
        """测试追加 tokens"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        seq_id = cache.allocate_sequence()
        
        k = np.random.randn(5, 8, 64).astype(np.float32)
        v = np.random.randn(5, 8, 64).astype(np.float32)
        
        cache.append_tokens(seq_id, k, v)
        assert cache.get_sequence_length(seq_id) == 5
    
    def test_free_sequence(self):
        """测试释放序列"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        initial_free = cache.allocator.get_num_free_blocks()
        seq_id = cache.allocate_sequence()
        
        k = np.random.randn(20, 8, 64).astype(np.float32)
        v = np.random.randn(20, 8, 64).astype(np.float32)
        cache.append_tokens(seq_id, k, v)
        
        cache.free_sequence(seq_id)
        assert cache.allocator.get_num_free_blocks() == initial_free
    
    def test_get_kv_cache(self):
        """测试获取 KV 缓存"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        seq_id = cache.allocate_sequence()
        
        k = np.random.randn(5, 8, 64).astype(np.float32)
        v = np.random.randn(5, 8, 64).astype(np.float32)
        cache.append_tokens(seq_id, k, v)
        
        k_cache, v_cache = cache.get_kv_cache(seq_id)
        assert k_cache.shape == (5, 8, 64)
        assert v_cache.shape == (5, 8, 64)
    
    def test_fork_sequence(self):
        """测试 fork 序列"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        seq_id = cache.allocate_sequence()
        k = np.random.randn(10, 8, 64).astype(np.float32)
        v = np.random.randn(10, 8, 64).astype(np.float32)
        cache.append_tokens(seq_id, k, v)
        
        forked_id = cache.fork_sequence(seq_id)
        assert forked_id != seq_id
        assert cache.get_sequence_length(forked_id) == cache.get_sequence_length(seq_id)
    
    def test_get_num_sequences(self):
        """测试获取序列数量"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        assert cache.get_num_sequences() == 0
        
        cache.allocate_sequence()
        cache.allocate_sequence()
        assert cache.get_num_sequences() == 2
    
    def test_can_append(self):
        """测试是否可以追加"""
        config = PagedAttentionConfig(num_blocks=10, block_size=16, num_heads=8, head_dim=64)
        cache = PagedKVCache(config)
        
        seq_id = cache.allocate_sequence()
        assert cache.can_append(seq_id, 16)


class TestPagedAttention:
    """PagedAttention 测试"""
    
    def test_attention_creation(self):
        """测试注意力模块创建"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        attn = PagedAttention(config)
        assert attn is not None
        assert attn.kv_cache is not None
    
    def test_forward_prefill(self):
        """测试 prefill 前向传播"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        attn = PagedAttention(config)
        
        seq_id = attn.kv_cache.allocate_sequence()
        
        # 形状: [batch, num_heads, seq_len, head_dim]
        q = np.random.randn(1, 8, 10, 64).astype(np.float32)
        k = np.random.randn(1, 8, 10, 64).astype(np.float32)
        v = np.random.randn(1, 8, 10, 64).astype(np.float32)
        
        output = attn.forward(q, k, v, [seq_id], is_prefill=True)
        assert output.shape == (1, 8, 10, 64)
    
    def test_decode_step(self):
        """测试解码步骤"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        attn = PagedAttention(config)
        
        seq_id = attn.kv_cache.allocate_sequence()
        
        # Prefill: [batch, num_heads, seq_len, head_dim]
        q = np.random.randn(1, 8, 5, 64).astype(np.float32)
        k = np.random.randn(1, 8, 5, 64).astype(np.float32)
        v = np.random.randn(1, 8, 5, 64).astype(np.float32)
        attn.forward(q, k, v, [seq_id], is_prefill=True)
        
        # Decode: [num_heads, 1, head_dim]
        q = np.random.randn(8, 1, 64).astype(np.float32)
        k = np.random.randn(8, 1, 64).astype(np.float32)
        v = np.random.randn(8, 1, 64).astype(np.float32)
        
        output = attn.decode_step(q, k, v, seq_id)
        assert output.shape == (8, 1, 64)
    
    def test_clear_sequence(self):
        """测试清除序列"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        attn = PagedAttention(config)
        
        seq_id = attn.kv_cache.allocate_sequence()
        q = np.random.randn(1, 8, 5, 64).astype(np.float32)
        k = np.random.randn(1, 8, 5, 64).astype(np.float32)
        v = np.random.randn(1, 8, 5, 64).astype(np.float32)
        attn.forward(q, k, v, [seq_id], is_prefill=True)
        
        attn.clear_sequence(seq_id)
    
    def test_get_memory_usage(self):
        """测试内存使用统计"""
        config = PagedAttentionConfig(num_blocks=100, block_size=16, num_heads=8, head_dim=64)
        attn = PagedAttention(config)
        
        usage = attn.get_memory_usage()
        assert "total_blocks" in usage


class TestCreatePagedAttention:
    """create_paged_attention 工厂函数测试"""
    
    def test_create_default(self):
        """测试默认创建"""
        attn = create_paged_attention()
        assert attn is not None
        assert isinstance(attn, PagedAttention)
    
    def test_create_custom(self):
        """测试自定义创建"""
        attn = create_paged_attention(
            block_size=32,
            num_blocks=50,
            num_heads=16,
            head_dim=128
        )
        assert attn.config.block_size == 32
        assert attn.config.num_blocks == 50
        assert attn.config.num_heads == 16
        assert attn.config.head_dim == 128


class TestMultiLayerPagedKVCache:
    """MultiLayerPagedKVCache 测试"""
    
    def test_multi_layer_creation(self):
        """测试多层缓存创建"""
        config = PagedAttentionConfig(
            num_blocks=100, block_size=16, 
            num_heads=8, head_dim=64, num_layers=4
        )
        cache = MultiLayerPagedKVCache(config)
        assert len(cache.layer_caches) == 4
    
    def test_multi_layer_allocate(self):
        """测试多层分配"""
        config = PagedAttentionConfig(
            num_blocks=100, block_size=16,
            num_heads=8, head_dim=64, num_layers=4
        )
        cache = MultiLayerPagedKVCache(config)
        
        seq_id = cache.allocate_sequence()
        assert seq_id is not None
    
    def test_multi_layer_append(self):
        """测试多层追加"""
        config = PagedAttentionConfig(
            num_blocks=100, block_size=16,
            num_heads=8, head_dim=64, num_layers=4
        )
        cache = MultiLayerPagedKVCache(config)
        
        seq_id = cache.allocate_sequence()
        
        # 每层单独追加
        for layer_idx in range(4):
            k = np.random.randn(5, 8, 64).astype(np.float32)
            v = np.random.randn(5, 8, 64).astype(np.float32)
            cache.append_tokens(layer_idx, seq_id, k, v)
        
        # 检查每层的长度 - 注意参数顺序是 (layer_idx, sequence_id)
        for layer_idx in range(4):
            k, v = cache.get_kv_cache(layer_idx, seq_id)
            assert k.shape[0] == 5
    
    def test_multi_layer_free(self):
        """测试多层释放"""
        config = PagedAttentionConfig(
            num_blocks=100, block_size=16,
            num_heads=8, head_dim=64, num_layers=4
        )
        cache = MultiLayerPagedKVCache(config)
        
        seq_id = cache.allocate_sequence()
        for layer_idx in range(4):
            k = np.random.randn(5, 8, 64).astype(np.float32)
            v = np.random.randn(5, 8, 64).astype(np.float32)
            cache.append_tokens(layer_idx, seq_id, k, v)
        
        cache.free_sequence(seq_id)
    
    def test_multi_layer_fork(self):
        """测试多层 fork"""
        config = PagedAttentionConfig(
            num_blocks=100, block_size=16,
            num_heads=8, head_dim=64, num_layers=4
        )
        cache = MultiLayerPagedKVCache(config)
        
        seq_id = cache.allocate_sequence()
        for layer_idx in range(4):
            k = np.random.randn(5, 8, 64).astype(np.float32)
            v = np.random.randn(5, 8, 64).astype(np.float32)
            cache.append_tokens(layer_idx, seq_id, k, v)
        
        forked_id = cache.fork_sequence(seq_id)
        assert forked_id != seq_id
    
    def test_get_total_memory_usage(self):
        """测试总内存使用"""
        config = PagedAttentionConfig(
            num_blocks=100, block_size=16,
            num_heads=8, head_dim=64, num_layers=4
        )
        cache = MultiLayerPagedKVCache(config)
        
        usage = cache.get_total_memory_usage()
        assert "num_layers" in usage
        assert usage["num_layers"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
