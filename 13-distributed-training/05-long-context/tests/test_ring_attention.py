"""
Ring Attention 模块单元测试

测试覆盖:
- RingAttentionConfig 配置验证
- BlockwiseAttention 分块注意力
- RingCommunicator 环形通信
- RingAttention 主类
- SequenceParallel 序列并行
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ring_attention import (
    RingAttentionConfig,
    BlockwiseAttention,
    RingCommunicator,
    RingAttention,
    RingAttentionLayer,
    SequenceParallel,
    create_ring_attention,
    compute_blockwise_attention,
    softmax,
)


class TestRingAttentionConfig:
    """RingAttentionConfig 测试。"""
    
    def test_default_config(self):
        config = RingAttentionConfig()
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.block_size == 1024
        assert config.n_devices == 8
    
    def test_head_dim(self):
        config = RingAttentionConfig(d_model=768, n_heads=12)
        assert config.head_dim == 64
    
    def test_max_seq_len(self):
        config = RingAttentionConfig(block_size=1024, n_devices=8)
        assert config.max_seq_len == 8192
    
    def test_invalid_d_model(self):
        with pytest.raises(ValueError):
            RingAttentionConfig(d_model=0)
    
    def test_invalid_n_heads(self):
        with pytest.raises(ValueError):
            RingAttentionConfig(d_model=768, n_heads=0)
    
    def test_d_model_not_divisible(self):
        with pytest.raises(ValueError):
            RingAttentionConfig(d_model=100, n_heads=12)


class TestBlockwiseAttention:
    """BlockwiseAttention 测试。"""
    
    @pytest.fixture
    def attn(self):
        return BlockwiseAttention(head_dim=64, causal=True)
    
    def test_single_block(self, attn):
        batch, n_heads, seq_len, head_dim = 2, 4, 16, 64
        q = np.random.randn(batch, n_heads, seq_len, head_dim)
        k = np.random.randn(batch, n_heads, seq_len, head_dim)
        v = np.random.randn(batch, n_heads, seq_len, head_dim)
        
        max_score, sum_exp, out = attn(q, k, v)
        
        assert max_score.shape == (batch, n_heads, seq_len, 1)
        assert sum_exp.shape == (batch, n_heads, seq_len, 1)
        assert out.shape == (batch, n_heads, seq_len, head_dim)
    
    def test_accumulation(self, attn):
        batch, n_heads, seq_len, head_dim = 2, 4, 16, 64
        q = np.random.randn(batch, n_heads, seq_len, head_dim)
        k1 = np.random.randn(batch, n_heads, seq_len, head_dim)
        v1 = np.random.randn(batch, n_heads, seq_len, head_dim)
        k2 = np.random.randn(batch, n_heads, seq_len, head_dim)
        v2 = np.random.randn(batch, n_heads, seq_len, head_dim)
        
        # 第一块
        max1, sum1, out1 = attn(q, k1, v1, q_offset=0, k_offset=0)
        
        # 第二块 (累积)
        max2, sum2, out2 = attn(
            q, k2, v2, q_offset=0, k_offset=seq_len,
            prev_max=max1, prev_sum=sum1, prev_out=out1
        )
        
        assert max2.shape == max1.shape
        assert sum2.shape == sum1.shape
        assert out2.shape == out1.shape
    
    def test_finalize(self, attn):
        batch, n_heads, seq_len, head_dim = 2, 4, 16, 64
        out = np.random.randn(batch, n_heads, seq_len, head_dim)
        sum_exp = np.abs(np.random.randn(batch, n_heads, seq_len, 1)) + 1
        
        result = attn.finalize(out, sum_exp)
        assert result.shape == out.shape


class TestRingCommunicator:
    """RingCommunicator 测试。"""
    
    def test_send_recv(self):
        comm = RingCommunicator(n_devices=4)
        data = np.array([1, 2, 3])
        
        comm.send(0, 1, "test", data)
        received = comm.recv(1, "test")
        
        assert np.array_equal(received, data)
    
    def test_ring_send_recv(self):
        comm = RingCommunicator(n_devices=4)
        data = np.array([1, 2, 3])
        
        result = comm.ring_send_recv(0, data, "kv")
        assert result.shape == data.shape


class TestRingAttention:
    """RingAttention 测试。"""
    
    @pytest.fixture
    def config(self):
        return RingAttentionConfig(
            d_model=64, n_heads=4, block_size=16, n_devices=4
        )
    
    def test_output_shape(self, config):
        attn = RingAttention(config)
        x = np.random.randn(2, 16, 64)
        y = attn(x, device_id=0)
        assert y.shape == x.shape
    
    def test_different_devices(self, config):
        attn = RingAttention(config)
        x = np.random.randn(2, 16, 64)
        
        y0 = attn(x, device_id=0)
        y1 = attn(x, device_id=1)
        
        assert y0.shape == y1.shape


class TestRingAttentionLayer:
    """RingAttentionLayer 测试。"""
    
    @pytest.fixture
    def config(self):
        return RingAttentionConfig(
            d_model=64, n_heads=4, block_size=16, n_devices=4
        )
    
    def test_output_shape(self, config):
        layer = RingAttentionLayer(config)
        x = np.random.randn(2, 16, 64)
        y = layer(x, device_id=0)
        assert y.shape == x.shape
    
    def test_residual_connection(self, config):
        layer = RingAttentionLayer(config)
        x = np.random.randn(2, 16, 64)
        y = layer(x, device_id=0)
        # 输出应该与输入不同 (有残差)
        assert not np.allclose(y, x)


class TestSequenceParallel:
    """SequenceParallel 测试。"""
    
    @pytest.fixture
    def config(self):
        return RingAttentionConfig(
            d_model=64, n_heads=4, block_size=16, n_devices=4
        )
    
    def test_split_sequence(self, config):
        sp = SequenceParallel(config)
        x = np.random.randn(2, 64, 64)  # 64 = 16 * 4
        
        blocks = sp.split_sequence(x)
        
        assert len(blocks) == 4
        for block in blocks:
            assert block.shape == (2, 16, 64)
    
    def test_gather_sequence(self, config):
        sp = SequenceParallel(config)
        blocks = [np.random.randn(2, 16, 64) for _ in range(4)]
        
        x = sp.gather_sequence(blocks, original_len=60)
        
        assert x.shape == (2, 60, 64)
    
    def test_split_and_gather(self, config):
        sp = SequenceParallel(config)
        original = np.random.randn(2, 60, 64)
        
        blocks = sp.split_sequence(original)
        recovered = sp.gather_sequence(blocks, original_len=60)
        
        assert np.allclose(recovered, original)


class TestComputeBlockwiseAttention:
    """compute_blockwise_attention 测试。"""
    
    def test_output_shape(self):
        batch, n_heads, seq_len, head_dim = 2, 4, 64, 32
        q = np.random.randn(batch, n_heads, seq_len, head_dim)
        k = np.random.randn(batch, n_heads, seq_len, head_dim)
        v = np.random.randn(batch, n_heads, seq_len, head_dim)
        
        out = compute_blockwise_attention(q, k, v, block_size=16)
        
        assert out.shape == (batch, n_heads, seq_len, head_dim)
    
    def test_causal_mask(self):
        batch, n_heads, seq_len, head_dim = 1, 1, 8, 4
        q = np.ones((batch, n_heads, seq_len, head_dim))
        k = np.ones((batch, n_heads, seq_len, head_dim))
        v = np.arange(seq_len).reshape(1, 1, seq_len, 1).repeat(head_dim, axis=-1).astype(float)
        
        out = compute_blockwise_attention(q, k, v, block_size=4, causal=True)
        
        # 第一个位置只能看到自己
        # 最后一个位置可以看到所有
        assert out[0, 0, 0, 0] < out[0, 0, -1, 0]


class TestCreateRingAttention:
    """工厂函数测试。"""
    
    def test_create_default(self):
        attn = create_ring_attention()
        assert attn.config.d_model == 768
        assert attn.config.n_heads == 12
    
    def test_create_custom(self):
        attn = create_ring_attention(
            d_model=512, n_heads=8, block_size=512, n_devices=4
        )
        assert attn.config.d_model == 512
        assert attn.config.n_devices == 4


class TestHelperFunctions:
    """辅助函数测试。"""
    
    def test_softmax(self):
        x = np.array([[1, 2, 3], [1, 1, 1]])
        y = softmax(x)
        assert np.allclose(np.sum(y, axis=-1), 1.0)
    
    def test_softmax_numerical_stability(self):
        x = np.array([[1000, 1001, 1002]])
        y = softmax(x)
        assert np.allclose(np.sum(y, axis=-1), 1.0)
        assert not np.any(np.isnan(y))
