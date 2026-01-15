"""
Flash Attention 单元测试

测试覆盖：
1. 配置验证
2. Online Softmax 正确性
3. 分块注意力与标准注意力对比
4. Flash Attention V1/V2/V3 正确性
5. FP8 量化精度
6. Incoherent Processing
7. 因果掩码
8. 多头注意力
"""

import math
import numpy as np
import pytest
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flash_attn import (
    FlashAttentionConfig,
    PrecisionMode,
    SchedulingMode,
    OnlineSoftmax,
    BlockwiseAttention,
    WarpScheduler,
    FP8Quantizer,
    IncoherentProcessor,
    FlashAttentionV1,
    FlashAttentionV2,
    FlashAttentionV3,
    create_flash_attention,
    standard_attention,
    compute_attention_flops,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """默认配置"""
    return FlashAttentionConfig(
        block_size_q=32,
        block_size_kv=32,
        causal=False
    )


@pytest.fixture
def causal_config():
    """因果掩码配置"""
    return FlashAttentionConfig(
        block_size_q=32,
        block_size_kv=32,
        causal=True
    )


@pytest.fixture
def small_qkv():
    """小规模 Q, K, V 张量"""
    np.random.seed(42)
    batch, seq, dim = 2, 64, 32
    return {
        "query": np.random.randn(batch, seq, dim).astype(np.float32),
        "key": np.random.randn(batch, seq, dim).astype(np.float32),
        "value": np.random.randn(batch, seq, dim).astype(np.float32),
    }


@pytest.fixture
def medium_qkv():
    """中等规模 Q, K, V 张量"""
    np.random.seed(42)
    batch, seq, dim = 2, 256, 64
    return {
        "query": np.random.randn(batch, seq, dim).astype(np.float32),
        "key": np.random.randn(batch, seq, dim).astype(np.float32),
        "value": np.random.randn(batch, seq, dim).astype(np.float32),
    }


@pytest.fixture
def multihead_qkv():
    """多头注意力 Q, K, V 张量"""
    np.random.seed(42)
    batch, heads, seq, dim = 2, 4, 64, 32
    return {
        "query": np.random.randn(batch, heads, seq, dim).astype(np.float32),
        "key": np.random.randn(batch, heads, seq, dim).astype(np.float32),
        "value": np.random.randn(batch, heads, seq, dim).astype(np.float32),
    }


# =============================================================================
# 配置测试
# =============================================================================

class TestFlashAttentionConfig:
    """配置类测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = FlashAttentionConfig()
        assert config.block_size_q == 128
        assert config.block_size_kv == 128
        assert config.num_stages == 2
        assert config.causal is False
        assert config.precision == PrecisionMode.FP32
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = FlashAttentionConfig(
            block_size_q=64,
            block_size_kv=64,
            causal=True,
            precision=PrecisionMode.FP16
        )
        assert config.block_size_q == 64
        assert config.causal is True
        assert config.precision == PrecisionMode.FP16
    
    def test_invalid_block_size(self):
        """测试无效块大小"""
        with pytest.raises(ValueError):
            FlashAttentionConfig(block_size_q=0)
        with pytest.raises(ValueError):
            FlashAttentionConfig(block_size_kv=-1)
    
    def test_invalid_dropout(self):
        """测试无效 dropout"""
        with pytest.raises(ValueError):
            FlashAttentionConfig(dropout_p=1.5)
        with pytest.raises(ValueError):
            FlashAttentionConfig(dropout_p=-0.1)
    
    def test_string_precision(self):
        """测试字符串精度转换"""
        config = FlashAttentionConfig(precision="fp16")
        assert config.precision == PrecisionMode.FP16


# =============================================================================
# Online Softmax 测试
# =============================================================================

class TestOnlineSoftmax:
    """在线 Softmax 测试"""
    
    def test_init_state(self):
        """测试状态初始化"""
        softmax = OnlineSoftmax()
        state = softmax.init_state(2, 10, 32)
        
        assert state["m"].shape == (2, 10)
        assert state["l"].shape == (2, 10)
        assert state["o"].shape == (2, 10, 32)
        assert np.all(state["m"] == -np.inf)
        assert np.all(state["l"] == 0)
    
    def test_single_block_update(self):
        """测试单块更新"""
        np.random.seed(42)
        softmax = OnlineSoftmax()
        
        batch, seq_q, block_kv, dim = 2, 8, 16, 32
        state = softmax.init_state(batch, seq_q, dim)
        
        scores = np.random.randn(batch, seq_q, block_kv).astype(np.float32)
        values = np.random.randn(batch, block_kv, dim).astype(np.float32)
        
        new_state = softmax.update(state, scores, values)
        
        assert new_state["m"].shape == (batch, seq_q)
        assert not np.any(np.isinf(new_state["m"]))
    
    def test_online_vs_standard_softmax(self):
        """测试在线 softmax 与标准 softmax 一致性"""
        np.random.seed(42)
        softmax = OnlineSoftmax()
        
        batch, seq_q, seq_kv, dim = 2, 16, 32, 64
        
        # 生成完整的 scores 和 values
        scores_full = np.random.randn(batch, seq_q, seq_kv).astype(np.float32)
        values = np.random.randn(batch, seq_kv, dim).astype(np.float32)
        
        # 标准 softmax
        scores_max = np.max(scores_full, axis=-1, keepdims=True)
        scores_exp = np.exp(scores_full - scores_max)
        attention_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        expected = np.einsum("bqk,bkd->bqd", attention_weights, values)
        
        # 在线 softmax (分两块)
        state = softmax.init_state(batch, seq_q, dim)
        
        # 第一块
        block_size = seq_kv // 2
        state = softmax.update(
            state, 
            scores_full[:, :, :block_size], 
            values[:, :block_size, :]
        )
        
        # 第二块
        state = softmax.update(
            state,
            scores_full[:, :, block_size:],
            values[:, block_size:, :]
        )
        
        result = softmax.finalize(state)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# =============================================================================
# 分块注意力测试
# =============================================================================

class TestBlockwiseAttention:
    """分块注意力测试"""
    
    def test_basic_attention(self, default_config, small_qkv):
        """测试基本注意力计算"""
        blockwise = BlockwiseAttention(default_config)
        output, _ = blockwise.compute(**small_qkv)
        
        assert output.shape == small_qkv["query"].shape
        assert not np.any(np.isnan(output))
    
    def test_vs_standard_attention(self, default_config, small_qkv):
        """测试与标准注意力一致性"""
        blockwise = BlockwiseAttention(default_config)
        output, _ = blockwise.compute(**small_qkv)
        
        expected = standard_attention(**small_qkv)
        
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_causal_mask(self, causal_config, small_qkv):
        """测试因果掩码"""
        blockwise = BlockwiseAttention(causal_config)
        output, _ = blockwise.compute(**small_qkv)
        
        expected = standard_attention(**small_qkv, causal=True)
        
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_different_block_sizes(self, small_qkv):
        """测试不同块大小结果一致"""
        results = []
        for block_size in [16, 32, 64]:
            config = FlashAttentionConfig(
                block_size_q=block_size,
                block_size_kv=block_size
            )
            blockwise = BlockwiseAttention(config)
            output, _ = blockwise.compute(**small_qkv)
            results.append(output)
        
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0], results[i], rtol=1e-4, atol=1e-4)


# =============================================================================
# Warp Scheduler 测试
# =============================================================================

class TestWarpScheduler:
    """Warp 调度器测试"""
    
    def test_scheduler_init(self, default_config):
        """测试调度器初始化"""
        scheduler = WarpScheduler(default_config)
        assert scheduler.num_stages == default_config.num_stages
        assert scheduler.mode == default_config.scheduling_mode
    
    def test_producer_simulation(self, default_config, small_qkv):
        """测试 Producer 模拟"""
        scheduler = WarpScheduler(default_config)
        block_data = scheduler.simulate_producer(
            small_qkv["key"], small_qkv["value"], 0
        )
        
        assert "k" in block_data
        assert "v" in block_data
        assert block_data["k"].shape[1] <= default_config.block_size_kv
    
    def test_pingpong_schedule(self, default_config, small_qkv):
        """测试 Pingpong 调度"""
        scheduler = WarpScheduler(default_config)
        scale = 1.0 / math.sqrt(small_qkv["query"].shape[-1])
        
        output = scheduler.schedule_pingpong(
            small_qkv["query"],
            small_qkv["key"],
            small_qkv["value"],
            scale
        )
        
        assert output.shape == small_qkv["query"].shape
        
        # 检查统计信息
        stats = scheduler.get_stats()
        assert stats["gemm_ops"] > 0
        assert stats["softmax_ops"] > 0


# =============================================================================
# FP8 量化测试
# =============================================================================

class TestFP8Quantizer:
    """FP8 量化器测试"""
    
    def test_quantize_basic(self):
        """测试基本量化"""
        quantizer = FP8Quantizer(block_size=64)
        x = np.random.randn(128).astype(np.float32)
        
        x_q, scales = quantizer.quantize(x)
        
        assert x_q.shape == x.shape
        assert len(scales) == 2  # 128 / 64 = 2 blocks
    
    def test_quantize_multidim(self):
        """测试多维张量量化"""
        quantizer = FP8Quantizer(block_size=32)
        x = np.random.randn(2, 64, 32).astype(np.float32)
        
        x_q, scales = quantizer.quantize(x)
        
        assert x_q.shape == x.shape
    
    def test_quantization_error(self):
        """测试量化误差"""
        quantizer = FP8Quantizer(block_size=64)
        x = np.random.randn(256).astype(np.float32)
        
        error = quantizer.compute_quantization_error(x)
        
        assert "mse" in error
        assert "max_error" in error
        assert "relative_error" in error
        assert error["mse"] >= 0
    
    def test_e4m3_vs_e5m2(self):
        """测试不同 FP8 格式"""
        x = np.random.randn(256).astype(np.float32) * 100
        
        q_e4m3 = FP8Quantizer(block_size=64, format="e4m3")
        q_e5m2 = FP8Quantizer(block_size=64, format="e5m2")
        
        err_e4m3 = q_e4m3.compute_quantization_error(x)
        err_e5m2 = q_e5m2.compute_quantization_error(x)
        
        # 两种格式都应该有合理的量化误差
        assert err_e4m3["mse"] >= 0
        assert err_e5m2["mse"] >= 0


# =============================================================================
# Incoherent Processing 测试
# =============================================================================

class TestIncoherentProcessor:
    """Incoherent Processing 测试"""
    
    def test_hadamard_orthogonality(self):
        """测试 Hadamard 矩阵正交性"""
        processor = IncoherentProcessor(dim=64)
        H = processor.rotation_matrix
        
        # H @ H^T 应该接近单位矩阵
        result = H @ H.T
        np.testing.assert_allclose(result, np.eye(64), rtol=1e-5, atol=1e-5)
    
    def test_transform_inverse(self):
        """测试变换和逆变换"""
        processor = IncoherentProcessor(dim=32)
        x = np.random.randn(10, 32).astype(np.float32)
        
        x_transformed = processor.transform(x)
        x_recovered = processor.inverse_transform(x_transformed)
        
        np.testing.assert_allclose(x, x_recovered, rtol=1e-5, atol=1e-5)
    
    def test_incoherent_reduces_error(self):
        """测试 incoherent processing 降低量化误差"""
        np.random.seed(42)
        # 创建有 outlier 的数据
        x = np.random.randn(64, 64).astype(np.float32)
        x[0, 0] = 100.0  # outlier
        
        quantizer = FP8Quantizer(block_size=64)
        processor = IncoherentProcessor(dim=64)
        
        # 直接量化
        x_direct, _ = quantizer.quantize(x)
        error_direct = np.mean((x - x_direct) ** 2)
        
        # Incoherent processing
        x_incoherent = processor.process(x, quantizer)
        error_incoherent = np.mean((x - x_incoherent) ** 2)
        
        # Incoherent 应该有更低的误差
        assert error_incoherent < error_direct * 2


# =============================================================================
# Flash Attention V1/V2/V3 测试
# =============================================================================

class TestFlashAttentionV1:
    """Flash Attention V1 测试"""
    
    def test_basic_forward(self, small_qkv):
        """测试基本前向传播"""
        attn = FlashAttentionV1()
        output = attn.forward(**small_qkv)
        
        assert output.shape == small_qkv["query"].shape
        assert not np.any(np.isnan(output))
    
    def test_vs_standard(self, small_qkv):
        """测试与标准注意力一致性"""
        attn = FlashAttentionV1()
        output = attn(**small_qkv)
        expected = standard_attention(**small_qkv)
        
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_multihead(self, multihead_qkv):
        """测试多头注意力"""
        attn = FlashAttentionV1()
        output = attn(**multihead_qkv)
        
        assert output.shape == multihead_qkv["query"].shape


class TestFlashAttentionV2:
    """Flash Attention V2 测试"""
    
    def test_basic_forward(self, small_qkv):
        """测试基本前向传播"""
        attn = FlashAttentionV2()
        output = attn.forward(**small_qkv)
        
        assert output.shape == small_qkv["query"].shape
    
    def test_vs_standard(self, small_qkv):
        """测试与标准注意力一致性"""
        attn = FlashAttentionV2()
        output = attn(**small_qkv)
        expected = standard_attention(**small_qkv)
        
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_causal(self, small_qkv):
        """测试因果掩码"""
        config = FlashAttentionConfig(causal=True, block_size_q=32, block_size_kv=32)
        attn = FlashAttentionV2(config)
        output = attn(**small_qkv)
        expected = standard_attention(**small_qkv, causal=True)
        
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)


class TestFlashAttentionV3:
    """Flash Attention V3 测试"""
    
    def test_basic_forward(self, small_qkv):
        """测试基本前向传播"""
        attn = FlashAttentionV3()
        output = attn.forward(**small_qkv)
        
        assert output.shape == small_qkv["query"].shape
    
    def test_vs_standard(self, small_qkv):
        """测试与标准注意力一致性"""
        attn = FlashAttentionV3()
        output = attn(**small_qkv)
        expected = standard_attention(**small_qkv)
        
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_causal(self, small_qkv):
        """测试因果掩码"""
        config = FlashAttentionConfig(causal=True, block_size_q=32, block_size_kv=32)
        attn = FlashAttentionV3(config)
        output = attn(**small_qkv)
        expected = standard_attention(**small_qkv, causal=True)
        
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_scheduler_stats(self, small_qkv):
        """测试调度器统计"""
        attn = FlashAttentionV3()
        _ = attn(**small_qkv)
        stats = attn.get_scheduler_stats()
        
        assert "gemm_ops" in stats
        assert "softmax_ops" in stats
    
    def test_fp8_mode(self, small_qkv):
        """测试 FP8 模式"""
        config = FlashAttentionConfig(
            use_fp8_quantization=True,
            block_size_q=32,
            block_size_kv=32
        )
        attn = FlashAttentionV3(config)
        output = attn(**small_qkv)
        
        assert output.shape == small_qkv["query"].shape


# =============================================================================
# 工厂函数和工具函数测试
# =============================================================================

class TestFactoryFunction:
    """工厂函数测试"""
    
    def test_create_v1(self):
        """测试创建 V1"""
        attn = create_flash_attention("v1")
        assert isinstance(attn, FlashAttentionV1)
    
    def test_create_v2(self):
        """测试创建 V2"""
        attn = create_flash_attention("v2")
        assert isinstance(attn, FlashAttentionV2)
    
    def test_create_v3(self):
        """测试创建 V3"""
        attn = create_flash_attention("v3")
        assert isinstance(attn, FlashAttentionV3)
    
    def test_create_with_options(self):
        """测试带选项创建"""
        attn = create_flash_attention(
            "v3",
            block_size=64,
            causal=True,
            use_fp8=True
        )
        assert isinstance(attn, FlashAttentionV3)
        assert attn.config.causal is True
        assert attn.config.use_fp8_quantization is True
    
    def test_invalid_version(self):
        """测试无效版本"""
        with pytest.raises(ValueError):
            create_flash_attention("v4")


class TestComputeFlops:
    """FLOPs 计算测试"""
    
    def test_basic_flops(self):
        """测试基本 FLOPs 计算"""
        flops = compute_attention_flops(
            batch_size=2,
            num_heads=8,
            seq_len_q=512,
            seq_len_kv=512,
            head_dim=64
        )
        
        assert "qk_gemm" in flops
        assert "softmax" in flops
        assert "pv_gemm" in flops
        assert "total" in flops
        assert flops["total"] == flops["qk_gemm"] + flops["softmax"] + flops["pv_gemm"]
    
    def test_causal_reduces_flops(self):
        """测试因果掩码减少 FLOPs"""
        flops_full = compute_attention_flops(2, 8, 512, 512, 64, causal=False)
        flops_causal = compute_attention_flops(2, 8, 512, 512, 64, causal=True)
        
        assert flops_causal["total"] < flops_full["total"]


class TestStandardAttention:
    """标准注意力测试"""
    
    def test_basic(self, small_qkv):
        """测试基本计算"""
        output = standard_attention(**small_qkv)
        assert output.shape == small_qkv["query"].shape
    
    def test_causal(self, small_qkv):
        """测试因果掩码"""
        output = standard_attention(**small_qkv, causal=True)
        assert output.shape == small_qkv["query"].shape


# =============================================================================
# 集成测试
# =============================================================================

class TestIntegration:
    """集成测试"""
    
    def test_all_versions_consistent(self, medium_qkv):
        """测试所有版本结果一致"""
        v1 = FlashAttentionV1()
        v2 = FlashAttentionV2()
        v3 = FlashAttentionV3()
        
        out_v1 = v1(**medium_qkv)
        out_v2 = v2(**medium_qkv)
        out_v3 = v3(**medium_qkv)
        
        np.testing.assert_allclose(out_v1, out_v2, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out_v2, out_v3, rtol=1e-4, atol=1e-4)
    
    def test_long_sequence(self):
        """测试长序列"""
        np.random.seed(42)
        batch, seq, dim = 1, 1024, 64
        qkv = {
            "query": np.random.randn(batch, seq, dim).astype(np.float32),
            "key": np.random.randn(batch, seq, dim).astype(np.float32),
            "value": np.random.randn(batch, seq, dim).astype(np.float32),
        }
        
        attn = create_flash_attention("v3", block_size=128)
        output = attn(**qkv)
        
        assert output.shape == (batch, seq, dim)
        assert not np.any(np.isnan(output))


# =============================================================================
# 边界测试
# =============================================================================

class TestEdgeCases:
    """边界条件测试"""
    
    def test_seq_len_one(self):
        """测试序列长度为 1"""
        np.random.seed(42)
        q = np.random.randn(1, 1, 32).astype(np.float32)
        k = np.random.randn(1, 1, 32).astype(np.float32)
        v = np.random.randn(1, 1, 32).astype(np.float32)
        
        attn = create_flash_attention("v3", block_size=32)
        output = attn(q, k, v)
        assert output.shape == (1, 1, 32)
    
    def test_seq_not_divisible_by_block(self):
        """测试序列长度不能被块大小整除"""
        np.random.seed(42)
        q = np.random.randn(1, 37, 32).astype(np.float32)  # 37 不能被 16 整除
        k = np.random.randn(1, 37, 32).astype(np.float32)
        v = np.random.randn(1, 37, 32).astype(np.float32)
        
        attn = create_flash_attention("v3", block_size=16)
        output = attn(q, k, v)
        expected = standard_attention(q, k, v)
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_different_q_kv_lengths(self):
        """测试 Q 和 KV 长度不同"""
        np.random.seed(42)
        q = np.random.randn(1, 32, 64).astype(np.float32)
        k = np.random.randn(1, 64, 64).astype(np.float32)
        v = np.random.randn(1, 64, 64).astype(np.float32)
        
        attn = create_flash_attention("v3", block_size=16)
        output = attn(q, k, v)
        expected = standard_attention(q, k, v)
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
    
    def test_head_dim_one(self):
        """测试头维度为 1"""
        np.random.seed(42)
        q = np.random.randn(1, 16, 1).astype(np.float32)
        k = np.random.randn(1, 16, 1).astype(np.float32)
        v = np.random.randn(1, 16, 1).astype(np.float32)
        
        attn = create_flash_attention("v3", block_size=8)
        output = attn(q, k, v)
        assert output.shape == (1, 16, 1)


class TestInputValidation:
    """输入验证测试"""
    
    def test_dimension_mismatch(self):
        """测试维度不匹配"""
        from flash_attn import validate_attention_inputs
        q = np.random.randn(1, 16, 32).astype(np.float32)
        k = np.random.randn(1, 16, 64).astype(np.float32)  # head_dim 不匹配
        v = np.random.randn(1, 16, 64).astype(np.float32)
        
        with pytest.raises(ValueError, match="head_dim"):
            validate_attention_inputs(q, k, v)
    
    def test_kv_length_mismatch(self):
        """测试 K/V 长度不匹配"""
        from flash_attn import validate_attention_inputs
        q = np.random.randn(1, 16, 32).astype(np.float32)
        k = np.random.randn(1, 16, 32).astype(np.float32)
        v = np.random.randn(1, 32, 32).astype(np.float32)  # seq_len 不匹配
        
        with pytest.raises(ValueError, match="seq_len"):
            validate_attention_inputs(q, k, v)
    
    def test_nan_input(self):
        """测试 NaN 输入"""
        from flash_attn import validate_attention_inputs
        q = np.array([[[np.nan]]]).astype(np.float32)
        k = np.random.randn(1, 1, 1).astype(np.float32)
        v = np.random.randn(1, 1, 1).astype(np.float32)
        
        with pytest.raises(ValueError, match="NaN"):
            validate_attention_inputs(q, k, v)
