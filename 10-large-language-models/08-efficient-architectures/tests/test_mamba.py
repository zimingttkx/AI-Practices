"""
Mamba 模块单元测试

测试覆盖:
- MambaConfig 配置验证
- 离散化函数
- 选择性扫描算法
- SelectiveSSM 模块
- MambaBlock/MambaLayer
- MambaModel/MambaForCausalLM
- 工厂函数
"""

import pytest
import numpy as np
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mamba import (
    MambaConfig,
    SelectiveSSM,
    MambaBlock,
    MambaLayer,
    MambaModel,
    MambaForCausalLM,
    Mamba2Block,
    RMSNorm,
    create_mamba_model,
    count_parameters,
    discretize_ssm,
    selective_scan,
    causal_conv1d,
)


class TestMambaConfig:
    """MambaConfig 测试。"""
    
    def test_default_config(self):
        """测试默认配置。"""
        config = MambaConfig()
        assert config.d_model == 768
        assert config.n_layers == 24
        assert config.d_state == 16
        assert config.d_conv == 4
        assert config.expand == 2
    
    def test_auto_dt_rank(self):
        """测试自动 dt_rank 计算。"""
        config = MambaConfig(d_model=768)
        assert config.dt_rank == 48  # 768 // 16
        
        config = MambaConfig(d_model=1024)
        assert config.dt_rank == 64  # 1024 // 16
    
    def test_vocab_size_padding(self):
        """测试词表大小对齐。"""
        config = MambaConfig(vocab_size=50257)
        assert config.vocab_size % 8 == 0
        assert config.vocab_size == 50264
    
    def test_d_inner_property(self):
        """测试 d_inner 属性。"""
        config = MambaConfig(d_model=768, expand=2)
        assert config.d_inner == 1536
    
    def test_invalid_d_model(self):
        """测试无效 d_model。"""
        with pytest.raises(ValueError):
            MambaConfig(d_model=-1)
    
    def test_invalid_n_layers(self):
        """测试无效 n_layers。"""
        with pytest.raises(ValueError):
            MambaConfig(n_layers=0)
    
    def test_invalid_d_state(self):
        """测试无效 d_state。"""
        with pytest.raises(ValueError):
            MambaConfig(d_state=-5)


class TestRMSNorm:
    """RMSNorm 测试。"""
    
    def test_output_shape(self):
        """测试输出形状。"""
        norm = RMSNorm(d_model=64)
        x = np.random.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape
    
    def test_normalization(self):
        """测试归一化效果。"""
        norm = RMSNorm(d_model=64)
        x = np.random.randn(2, 10, 64) * 10  # 大数值
        y = norm(x)
        # RMS 应该接近 1
        rms = np.sqrt(np.mean(y ** 2, axis=-1))
        assert np.allclose(rms, 1.0, atol=0.1)


class TestDiscretizeSSM:
    """离散化函数测试。"""
    
    def test_zoh_discretization(self):
        """测试 ZOH 离散化。"""
        A = np.random.randn(8, 4)
        B = np.random.randn(2, 10, 4)
        delta = np.abs(np.random.randn(2, 10, 8)) * 0.1
        
        A_bar, B_bar = discretize_ssm(A, B, delta, method="zoh")
        
        assert A_bar.shape == (2, 10, 8, 4)
        assert B_bar.shape == (2, 10, 8, 4)
    
    def test_bilinear_discretization(self):
        """测试双线性离散化。"""
        A = np.random.randn(8, 4)
        B = np.random.randn(2, 10, 4)
        delta = np.abs(np.random.randn(2, 10, 8)) * 0.1
        
        A_bar, B_bar = discretize_ssm(A, B, delta, method="bilinear")
        
        assert A_bar.shape == (2, 10, 8, 4)
        assert B_bar.shape == (2, 10, 8, 4)
    
    def test_euler_discretization(self):
        """测试欧拉离散化。"""
        A = np.random.randn(8, 4)
        B = np.random.randn(2, 10, 4)
        delta = np.abs(np.random.randn(2, 10, 8)) * 0.1
        
        A_bar, B_bar = discretize_ssm(A, B, delta, method="euler")
        
        assert A_bar.shape == (2, 10, 8, 4)
        assert B_bar.shape == (2, 10, 8, 4)
    
    def test_invalid_method(self):
        """测试无效方法。"""
        A = np.random.randn(8, 4)
        B = np.random.randn(2, 10, 4)
        delta = np.abs(np.random.randn(2, 10, 8)) * 0.1
        
        with pytest.raises(ValueError):
            discretize_ssm(A, B, delta, method="invalid")


class TestSelectiveScan:
    """选择性扫描测试。"""
    
    def test_output_shape(self):
        """测试输出形状。"""
        batch_size, seq_len, d_inner, d_state = 2, 10, 8, 4
        
        x = np.random.randn(batch_size, seq_len, d_inner)
        delta = np.abs(np.random.randn(batch_size, seq_len, d_inner)) * 0.1
        A = -np.abs(np.random.randn(d_inner, d_state))
        B = np.random.randn(batch_size, seq_len, d_state)
        C = np.random.randn(batch_size, seq_len, d_state)
        D = np.ones(d_inner)
        
        y = selective_scan(x, delta, A, B, C, D)
        
        assert y.shape == x.shape
    
    def test_without_skip_connection(self):
        """测试无跳跃连接。"""
        batch_size, seq_len, d_inner, d_state = 2, 10, 8, 4
        
        x = np.random.randn(batch_size, seq_len, d_inner)
        delta = np.abs(np.random.randn(batch_size, seq_len, d_inner)) * 0.1
        A = -np.abs(np.random.randn(d_inner, d_state))
        B = np.random.randn(batch_size, seq_len, d_state)
        C = np.random.randn(batch_size, seq_len, d_state)
        
        y = selective_scan(x, delta, A, B, C, D=None)
        
        assert y.shape == x.shape


class TestCausalConv1d:
    """因果卷积测试。"""
    
    def test_output_shape(self):
        """测试输出形状。"""
        x = np.random.randn(2, 8, 10)  # [B, D, L]
        weight = np.random.randn(8, 1, 4)  # [D, 1, K]
        bias = np.random.randn(8)
        
        y = causal_conv1d(x, weight, bias)
        
        assert y.shape == x.shape
    
    def test_causality(self):
        """测试因果性 (输出只依赖于过去)。"""
        x = np.zeros((1, 4, 10))
        x[0, :, 5] = 1.0  # 在位置 5 设置脉冲
        
        weight = np.ones((4, 1, 3))
        
        y = causal_conv1d(x, weight, None)
        
        # 位置 5 之前应该为 0
        assert np.allclose(y[0, :, :5], 0)
        # 位置 5 及之后应该非零
        assert not np.allclose(y[0, :, 5:8], 0)


class TestSelectiveSSM:
    """SelectiveSSM 测试。"""
    
    @pytest.fixture
    def ssm(self):
        """创建 SSM 实例。"""
        return SelectiveSSM(
            d_model=64,
            d_state=8,
            d_conv=4,
            expand=2,
            dt_rank=4,
        )
    
    def test_output_shape(self, ssm):
        """测试输出形状。"""
        x = np.random.randn(2, 10, 64)
        y, _ = ssm(x)
        assert y.shape == x.shape
    
    def test_different_seq_lengths(self, ssm):
        """测试不同序列长度。"""
        for seq_len in [1, 5, 20, 100]:
            x = np.random.randn(2, seq_len, 64)
            y, _ = ssm(x)
            assert y.shape == x.shape


class TestMambaBlock:
    """MambaBlock 测试。"""
    
    @pytest.fixture
    def config(self):
        """创建配置。"""
        return MambaConfig(
            d_model=64,
            n_layers=2,
            d_state=8,
            d_conv=4,
            expand=2,
        )
    
    def test_output_shape(self, config):
        """测试输出形状。"""
        block = MambaBlock(config, layer_idx=0)
        x = np.random.randn(2, 10, 64)
        y, _ = block(x)
        assert y.shape == x.shape
    
    def test_residual_connection(self, config):
        """测试残差连接。"""
        block = MambaBlock(config, layer_idx=0)
        x = np.random.randn(2, 10, 64)
        y, _ = block(x)
        # 输出应该与输入有相关性 (残差)
        assert not np.allclose(y, 0)


class TestMambaLayer:
    """MambaLayer 测试。"""
    
    @pytest.fixture
    def config(self):
        """创建配置。"""
        return MambaConfig(
            d_model=64,
            n_layers=3,
            d_state=8,
            d_conv=4,
            expand=2,
        )
    
    def test_output_shape(self, config):
        """测试输出形状。"""
        layer = MambaLayer(config)
        x = np.random.randn(2, 10, 64)
        y, _ = layer(x)
        assert y.shape == x.shape
    
    def test_num_blocks(self, config):
        """测试块数量。"""
        layer = MambaLayer(config)
        assert len(layer.blocks) == config.n_layers


class TestMambaModel:
    """MambaModel 测试。"""
    
    @pytest.fixture
    def config(self):
        """创建配置。"""
        return MambaConfig(
            d_model=64,
            n_layers=2,
            d_state=8,
            d_conv=4,
            expand=2,
            vocab_size=1000,
        )
    
    def test_output_shape(self, config):
        """测试输出形状。"""
        model = MambaModel(config)
        input_ids = np.random.randint(0, 1000, size=(2, 10))
        hidden_states, _ = model(input_ids)
        assert hidden_states.shape == (2, 10, 64)
    
    def test_embedding(self, config):
        """测试嵌入层。"""
        model = MambaModel(config)
        assert model.embedding.shape == (config.vocab_size, config.d_model)


class TestMambaForCausalLM:
    """MambaForCausalLM 测试。"""
    
    @pytest.fixture
    def config(self):
        """创建配置。"""
        return MambaConfig(
            d_model=64,
            n_layers=2,
            d_state=8,
            d_conv=4,
            expand=2,
            vocab_size=1000,
        )
    
    def test_logits_shape(self, config):
        """测试 logits 形状。"""
        model = MambaForCausalLM(config)
        input_ids = np.random.randint(0, 1000, size=(2, 10))
        result = model(input_ids)
        assert result["logits"].shape == (2, 10, config.vocab_size)
    
    def test_loss_computation(self, config):
        """测试损失计算。"""
        model = MambaForCausalLM(config)
        input_ids = np.random.randint(0, 1000, size=(2, 10))
        labels = np.random.randint(0, 1000, size=(2, 10))
        result = model(input_ids, labels=labels)
        assert "loss" in result
        assert result["loss"] > 0
    
    def test_generate(self, config):
        """测试生成。"""
        model = MambaForCausalLM(config)
        input_ids = np.random.randint(0, 1000, size=(1, 5))
        generated = model.generate(input_ids, max_new_tokens=3)
        assert generated.shape == (1, 8)  # 5 + 3


class TestMamba2Block:
    """Mamba2Block 测试。"""
    
    def test_output_shape(self):
        """测试输出形状。"""
        block = Mamba2Block(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=32,
        )
        x = np.random.randn(2, 10, 64)
        y = block(x)
        assert y.shape == x.shape


class TestCreateMambaModel:
    """工厂函数测试。"""
    
    def test_create_small(self):
        """测试创建 small 模型。"""
        model = create_mamba_model("small")
        assert model.config.d_model == 768
        assert model.config.n_layers == 24
    
    def test_create_base(self):
        """测试创建 base 模型。"""
        model = create_mamba_model("base")
        assert model.config.d_model == 1024
        assert model.config.n_layers == 48
    
    def test_create_with_custom_vocab(self):
        """测试自定义词表大小。"""
        model = create_mamba_model("small", vocab_size=32000)
        assert model.config.vocab_size == 32000
    
    def test_invalid_size(self):
        """测试无效模型大小。"""
        with pytest.raises(ValueError):
            create_mamba_model("invalid")


class TestCountParameters:
    """参数量计算测试。"""
    
    def test_count_small(self):
        """测试 small 模型参数量。"""
        config = MambaConfig(d_model=768, n_layers=24, vocab_size=50264)
        params = count_parameters(config)
        assert params["total"] > 0
        assert params["total_millions"] > 100  # ~130M
    
    def test_count_components(self):
        """测试各组件参数量。"""
        config = MambaConfig(d_model=64, n_layers=2, vocab_size=1000)
        params = count_parameters(config)
        assert "embedding" in params
        assert "per_layer" in params
        assert "all_layers" in params
        assert params["all_layers"] == params["per_layer"] * 2
