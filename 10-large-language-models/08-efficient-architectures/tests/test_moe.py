"""
MoE 模块单元测试

测试覆盖:
- MoEConfig 配置验证
- Expert 专家模块
- Router 路由器 (TopK/Switch/ExpertChoice)
- MoELayer
- MoETransformerBlock
- MoEModel
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from moe import (
    MoEConfig,
    RouterType,
    Expert,
    TopKRouter,
    SwitchRouter,
    ExpertChoiceRouter,
    MoELayer,
    MoETransformerBlock,
    MoEModel,
    LayerNorm,
    MultiHeadAttention,
    create_moe_model,
    compute_load_balancing_loss,
    softmax,
    gelu,
)


class TestMoEConfig:
    """MoEConfig 测试。"""
    
    def test_default_config(self):
        config = MoEConfig()
        assert config.d_model == 768
        assert config.n_experts == 8
        assert config.n_experts_per_tok == 2
    
    def test_auto_d_ff(self):
        config = MoEConfig(d_model=512)
        assert config.d_ff == 2048
    
    def test_invalid_n_experts(self):
        with pytest.raises(ValueError):
            MoEConfig(n_experts=0)
    
    def test_invalid_n_experts_per_tok(self):
        with pytest.raises(ValueError):
            MoEConfig(n_experts=4, n_experts_per_tok=5)


class TestExpert:
    """Expert 测试。"""
    
    def test_output_shape(self):
        expert = Expert(d_model=64, d_ff=256)
        x = np.random.randn(10, 64)
        y = expert(x)
        assert y.shape == (10, 64)
    
    def test_batch_input(self):
        expert = Expert(d_model=64, d_ff=256)
        x = np.random.randn(2, 10, 64)
        y = expert(x)
        assert y.shape == (2, 10, 64)


class TestTopKRouter:
    """TopKRouter 测试。"""
    
    def test_output_shape(self):
        router = TopKRouter(d_model=64, n_experts=8, n_experts_per_tok=2)
        x = np.random.randn(20, 64)
        indices, weights, aux = router(x)
        assert indices.shape == (20, 2)
        assert weights.shape == (20, 2)
    
    def test_weights_sum_to_one(self):
        router = TopKRouter(d_model=64, n_experts=8, n_experts_per_tok=2)
        x = np.random.randn(20, 64)
        _, weights, _ = router(x)
        sums = np.sum(weights, axis=-1)
        assert np.allclose(sums, 1.0)
    
    def test_aux_loss(self):
        router = TopKRouter(d_model=64, n_experts=8, n_experts_per_tok=2)
        x = np.random.randn(20, 64)
        _, _, aux = router(x)
        assert "aux_loss" in aux
        assert aux["aux_loss"] >= 0


class TestSwitchRouter:
    """SwitchRouter 测试。"""
    
    def test_output_shape(self):
        router = SwitchRouter(d_model=64, n_experts=8)
        x = np.random.randn(20, 64)
        indices, weights, aux = router(x)
        assert indices.shape == (20, 1)
        assert weights.shape == (20, 1)
    
    def test_single_expert_selection(self):
        router = SwitchRouter(d_model=64, n_experts=8)
        x = np.random.randn(20, 64)
        indices, _, _ = router(x)
        assert np.all(indices >= 0)
        assert np.all(indices < 8)


class TestExpertChoiceRouter:
    """ExpertChoiceRouter 测试。"""
    
    def test_output_structure(self):
        router = ExpertChoiceRouter(d_model=64, n_experts=4)
        x = np.random.randn(20, 64)
        indices, weights, aux = router(x)
        assert len(indices) == 4
        assert len(weights) == 4


class TestMoELayer:
    """MoELayer 测试。"""
    
    @pytest.fixture
    def config(self):
        return MoEConfig(d_model=64, n_experts=4, n_experts_per_tok=2, d_ff=128)
    
    def test_output_shape(self, config):
        layer = MoELayer(config)
        x = np.random.randn(2, 10, 64)
        y, aux = layer(x)
        assert y.shape == x.shape
    
    def test_switch_router(self):
        config = MoEConfig(d_model=64, n_experts=4, router_type=RouterType.SWITCH, d_ff=128)
        layer = MoELayer(config)
        x = np.random.randn(2, 10, 64)
        y, aux = layer(x)
        assert y.shape == x.shape


class TestLayerNorm:
    """LayerNorm 测试。"""
    
    def test_output_shape(self):
        norm = LayerNorm(64)
        x = np.random.randn(2, 10, 64)
        y = norm(x)
        assert y.shape == x.shape
    
    def test_normalization(self):
        norm = LayerNorm(64)
        x = np.random.randn(2, 10, 64) * 10
        y = norm(x)
        mean = np.mean(y, axis=-1)
        assert np.allclose(mean, 0, atol=1e-5)


class TestMultiHeadAttention:
    """MultiHeadAttention 测试。"""
    
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=64, n_heads=4)
        x = np.random.randn(2, 10, 64)
        y = attn(x)
        assert y.shape == x.shape


class TestMoETransformerBlock:
    """MoETransformerBlock 测试。"""
    
    @pytest.fixture
    def config(self):
        return MoEConfig(d_model=64, n_experts=4, n_heads=4, d_ff=128)
    
    def test_output_shape(self, config):
        block = MoETransformerBlock(config)
        x = np.random.randn(2, 10, 64)
        y, aux = block(x)
        assert y.shape == x.shape


class TestMoEModel:
    """MoEModel 测试。"""
    
    @pytest.fixture
    def config(self):
        return MoEConfig(
            d_model=64, n_layers=2, n_experts=4, n_heads=4,
            d_ff=128, vocab_size=1000
        )
    
    def test_logits_shape(self, config):
        model = MoEModel(config)
        input_ids = np.random.randint(0, 1000, size=(2, 10))
        result = model(input_ids)
        assert result["logits"].shape == (2, 10, 1000)
    
    def test_loss_computation(self, config):
        model = MoEModel(config)
        input_ids = np.random.randint(0, 1000, size=(2, 10))
        labels = np.random.randint(0, 1000, size=(2, 10))
        result = model(input_ids, labels=labels)
        assert "loss" in result
        assert "lm_loss" in result
        assert "aux_loss" in result


class TestCreateMoEModel:
    """工厂函数测试。"""
    
    def test_create_small(self):
        model = create_moe_model("small", n_experts=4)
        assert model.config.d_model == 512
        assert model.config.n_experts == 4
    
    def test_create_base(self):
        model = create_moe_model("base", n_experts=8)
        assert model.config.d_model == 768
        assert model.config.n_experts == 8
    
    def test_invalid_size(self):
        with pytest.raises(ValueError):
            create_moe_model("invalid")


class TestHelperFunctions:
    """辅助函数测试。"""
    
    def test_softmax(self):
        x = np.array([[1, 2, 3], [1, 1, 1]])
        y = softmax(x)
        assert np.allclose(np.sum(y, axis=-1), 1.0)
    
    def test_gelu(self):
        x = np.array([0, 1, -1])
        y = gelu(x)
        assert y[0] == pytest.approx(0, abs=0.01)
        assert y[1] > 0
        assert y[2] < 0
    
    def test_load_balancing_loss(self):
        probs = np.random.rand(20, 8)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        indices = np.random.randint(0, 8, size=(20, 2))
        loss = compute_load_balancing_loss(probs, indices, n_experts=8)
        assert loss >= 0
