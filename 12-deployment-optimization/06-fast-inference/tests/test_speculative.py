"""
Speculative Decoding 模块单元测试

测试覆盖:
- SpeculativeConfig 配置
- DraftModel/TargetModel 接口
- MockDraftModel/MockTargetModel 模拟实现
- TokenVerifier 验证器
- SpeculativeDecoder 解码器
- TreeSpeculation 树形推测
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.speculative import (
    SpeculativeConfig,
    DraftModel,
    TargetModel,
    MockDraftModel,
    MockTargetModel,
    TokenVerifier,
    SpeculativeOutput,
    SpeculativeDecoder,
    TreeNode,
    TreeSpeculation,
    create_speculative_decoder,
    create_tree_speculation,
    compute_acceptance_rate,
    estimate_speedup,
)


class TestSpeculativeConfig:
    """SpeculativeConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SpeculativeConfig()
        assert config.num_speculative_tokens > 0
        assert config.max_sequence_length > 0
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SpeculativeConfig(
            num_speculative_tokens=8,
            temperature=0.7,
            top_p=0.9
        )
        assert config.num_speculative_tokens == 8
        assert config.temperature == 0.7
        assert config.top_p == 0.9
    
    def test_invalid_num_speculative_tokens(self):
        """测试无效的推测 token 数"""
        with pytest.raises(ValueError):
            SpeculativeConfig(num_speculative_tokens=0)


class TestMockDraftModel:
    """MockDraftModel 测试"""
    
    def test_model_creation(self):
        """测试模型创建"""
        model = MockDraftModel(vocab_size=1000)
        assert model is not None
        assert model.vocab_size == 1000
    
    def test_generate(self):
        """测试生成"""
        model = MockDraftModel(vocab_size=1000)
        input_ids = [1, 2, 3, 4, 5]
        
        tokens, probs = model.generate(input_ids, num_tokens=5)
        assert len(tokens) == 5
        assert len(probs) == 5
        for token in tokens:
            assert 0 <= token < 1000
    
    def test_get_logits(self):
        """测试获取 logits"""
        model = MockDraftModel(vocab_size=1000)
        input_ids = [1, 2, 3, 4, 5]
        
        logits = model.get_logits(input_ids)
        assert logits.shape == (1000,)


class TestMockTargetModel:
    """MockTargetModel 测试"""
    
    def test_model_creation(self):
        """测试模型创建"""
        model = MockTargetModel(vocab_size=1000)
        assert model is not None
    
    def test_verify(self):
        """测试验证"""
        model = MockTargetModel(vocab_size=1000)
        input_ids = [1, 2, 3, 4, 5]
        candidate_ids = [10, 20, 30]
        
        probs, next_probs = model.verify(input_ids, candidate_ids)
        assert len(probs) == 3
        assert next_probs.shape == (1000,)


class TestTokenVerifier:
    """TokenVerifier 测试"""
    
    def test_verifier_creation(self):
        """测试验证器创建"""
        verifier = TokenVerifier()
        assert verifier is not None
    
    def test_verify_tokens(self):
        """测试验证 tokens"""
        verifier = TokenVerifier()
        
        draft_tokens = [10, 20, 30]
        draft_probs = [
            np.ones(1000) / 1000,
            np.ones(1000) / 1000,
            np.ones(1000) / 1000
        ]
        target_probs = [
            np.ones(1000) / 1000,
            np.ones(1000) / 1000,
            np.ones(1000) / 1000
        ]
        next_token_probs = np.ones(1000) / 1000
        
        accepted, bonus_token = verifier.verify_tokens(draft_tokens, draft_probs, target_probs, next_token_probs)
        assert isinstance(accepted, list)


class TestSpeculativeDecoder:
    """SpeculativeDecoder 测试"""
    
    def test_decoder_creation(self):
        """测试解码器创建"""
        config = SpeculativeConfig(num_speculative_tokens=4)
        draft_model = MockDraftModel(vocab_size=1000)
        target_model = MockTargetModel(vocab_size=1000)
        
        decoder = SpeculativeDecoder(config, draft_model, target_model)
        assert decoder is not None
    
    def test_generate(self):
        """测试生成"""
        config = SpeculativeConfig(num_speculative_tokens=4)
        draft_model = MockDraftModel(vocab_size=1000)
        target_model = MockTargetModel(vocab_size=1000)
        
        decoder = SpeculativeDecoder(config, draft_model, target_model)
        input_ids = [1, 2, 3, 4, 5]
        
        output = decoder.generate(input_ids, max_tokens=10)
        # output 是 SpeculativeOutput
        assert output is not None
    
    def test_get_stats(self):
        """测试获取统计信息"""
        config = SpeculativeConfig(num_speculative_tokens=4)
        draft_model = MockDraftModel(vocab_size=1000)
        target_model = MockTargetModel(vocab_size=1000)
        
        decoder = SpeculativeDecoder(config, draft_model, target_model)
        
        stats = decoder.get_stats()
        assert isinstance(stats, dict)


class TestTreeSpeculation:
    """TreeSpeculation 测试"""
    
    def test_tree_creation(self):
        """测试树形推测创建"""
        config = SpeculativeConfig(num_speculative_tokens=4)
        draft_model = MockDraftModel(vocab_size=1000)
        
        tree = TreeSpeculation(config, draft_model)
        assert tree is not None
    
    def test_generate_tree(self):
        """测试生成树"""
        config = SpeculativeConfig(num_speculative_tokens=4)
        draft_model = MockDraftModel(vocab_size=1000)
        
        tree = TreeSpeculation(config, draft_model)
        input_ids = [1, 2, 3, 4, 5]
        
        root = tree.generate_tree(input_ids)
        assert root is not None
    
    def test_get_all_paths(self):
        """测试获取所有路径"""
        config = SpeculativeConfig(num_speculative_tokens=4)
        draft_model = MockDraftModel(vocab_size=1000)
        
        tree = TreeSpeculation(config, draft_model)
        input_ids = [1, 2, 3, 4, 5]
        
        root = tree.generate_tree(input_ids)
        paths = tree.get_all_paths(root)
        assert isinstance(paths, list)


class TestCreateSpeculativeDecoder:
    """create_speculative_decoder 工厂函数测试"""
    
    def test_create_default(self):
        """测试默认创建"""
        decoder = create_speculative_decoder()
        assert decoder is not None
        assert isinstance(decoder, SpeculativeDecoder)
    
    def test_create_custom(self):
        """测试自定义创建"""
        decoder = create_speculative_decoder(
            num_speculative_tokens=8
        )
        assert decoder.config.num_speculative_tokens == 8


class TestCreateTreeSpeculation:
    """create_tree_speculation 工厂函数测试"""
    
    def test_create_default(self):
        """测试默认创建"""
        tree = create_tree_speculation()
        assert tree is not None
        assert isinstance(tree, TreeSpeculation)


class TestUtilityFunctions:
    """工具函数测试"""
    
    def test_estimate_speedup(self):
        """测试估算加速比"""
        acceptance_rate = 0.8
        num_speculative = 4
        speedup = estimate_speedup(acceptance_rate, num_speculative)
        assert speedup > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
