"""
LoRA 模块单元测试
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lora import (
    LoRAType, LoRAConfig, LoRALinear, LoRAConv2d,
    LoHALinear, LoKrLinear, DyLoRALinear,
    LoRAInjectedLinear, LoRAInjectedConv2d,
    LoRAManager, create_lora_config, inject_lora_to_model
)


class TestLoRAConfig:
    """测试 LoRA 配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = LoRAConfig()
        
        assert config.rank == 4
        assert config.alpha == 1.0
        assert config.dropout == 0.0
        assert config.lora_type == LoRAType.LORA
        assert "q_proj" in config.target_modules
        
    def test_custom_config(self):
        """测试自定义配置"""
        config = LoRAConfig(
            rank=8,
            alpha=2.0,
            dropout=0.1,
            lora_type=LoRAType.LOHA,
            target_modules=["fc1", "fc2"]
        )
        
        assert config.rank == 8
        assert config.alpha == 2.0
        assert config.dropout == 0.1
        assert config.lora_type == LoRAType.LOHA
        assert config.target_modules == ["fc1", "fc2"]


class TestLoRALinear:
    """测试 LoRA 线性层"""
    
    def test_initialization(self):
        """测试初始化"""
        lora = LoRALinear(64, 128, rank=4, alpha=1.0)
        
        assert lora.in_features == 64
        assert lora.out_features == 128
        assert lora.rank == 4
        assert lora.scaling == 0.25  # alpha / rank
        
    def test_forward_shape(self):
        """测试前向传播形状"""
        lora = LoRALinear(64, 128, rank=4)
        x = torch.randn(2, 10, 64)
        
        output = lora(x)
        
        assert output.shape == (2, 10, 128)
        
    def test_forward_is_finite(self):
        """测试输出是有限的"""
        lora = LoRALinear(64, 128, rank=4)
        x = torch.randn(2, 10, 64)
        
        output = lora(x)
        
        assert torch.isfinite(output).all()
        
    def test_initial_output_near_zero(self):
        """测试初始输出接近零（B 矩阵初始化为零）"""
        lora = LoRALinear(64, 128, rank=4)
        x = torch.randn(2, 10, 64)
        
        output = lora(x)
        
        # 由于 B 初始化为零，输出应该接近零
        assert output.abs().max() < 1e-6
        
    def test_get_merged_weight(self):
        """测试权重合并"""
        lora = LoRALinear(64, 128, rank=4)
        original_weight = torch.randn(128, 64)
        
        merged = lora.get_merged_weight(original_weight)
        
        assert merged.shape == original_weight.shape
        
    def test_dropout(self):
        """测试 dropout"""
        lora = LoRALinear(64, 128, rank=4, dropout=0.5)
        x = torch.randn(2, 10, 64)
        
        # 训练模式下 dropout 应该生效
        lora.train()
        output1 = lora(x)
        output2 = lora(x)
        
        # 由于 B 初始化为零，即使有 dropout 输出也是零
        assert output1.abs().max() < 1e-6


class TestLoRAConv2d:
    """测试 LoRA 卷积层"""
    
    def test_initialization(self):
        """测试初始化"""
        lora = LoRAConv2d(64, 128, kernel_size=3, rank=4, padding=1)
        
        assert lora.in_channels == 64
        assert lora.out_channels == 128
        assert lora.rank == 4
        
    def test_forward_shape(self):
        """测试前向传播形状"""
        lora = LoRAConv2d(64, 128, kernel_size=3, rank=4, padding=1)
        x = torch.randn(2, 64, 16, 16)
        
        output = lora(x)
        
        assert output.shape == (2, 128, 16, 16)
        
    def test_initial_output_near_zero(self):
        """测试初始输出接近零"""
        lora = LoRAConv2d(64, 128, kernel_size=3, rank=4, padding=1)
        x = torch.randn(2, 64, 16, 16)
        
        output = lora(x)
        
        assert output.abs().max() < 1e-6


class TestLoHALinear:
    """测试 LoHA 线性层"""
    
    def test_initialization(self):
        """测试初始化"""
        loha = LoHALinear(64, 128, rank=4)
        
        assert loha.in_features == 64
        assert loha.out_features == 128
        assert loha.rank == 4
        
    def test_forward_shape(self):
        """测试前向传播形状"""
        loha = LoHALinear(64, 128, rank=4)
        x = torch.randn(2, 10, 64)
        
        output = loha(x)
        
        assert output.shape == (2, 10, 128)
        
    def test_get_delta_weight(self):
        """测试权重增量计算"""
        loha = LoHALinear(64, 128, rank=4)
        
        delta = loha.get_delta_weight()
        
        assert delta.shape == (128, 64)
        
    def test_hadamard_product(self):
        """测试 Hadamard 乘积"""
        loha = LoHALinear(64, 128, rank=4)
        
        # 手动计算
        w1 = loha.hada_w1_b @ loha.hada_w1_a
        w2 = loha.hada_w2_b @ loha.hada_w2_a
        expected = (w1 * w2) * loha.scaling
        
        delta = loha.get_delta_weight()
        
        assert torch.allclose(delta, expected)


class TestLoKrLinear:
    """测试 LoKr 线性层"""
    
    def test_initialization(self):
        """测试初始化"""
        lokr = LoKrLinear(64, 128, rank=4)
        
        assert lokr.in_features == 64
        assert lokr.out_features == 128
        assert lokr.rank == 4
        
    def test_forward_shape(self):
        """测试前向传播形状"""
        lokr = LoKrLinear(64, 128, rank=4)
        x = torch.randn(2, 10, 64)
        
        output = lokr(x)
        
        assert output.shape == (2, 10, 128)
        
    def test_get_delta_weight(self):
        """测试权重增量计算"""
        lokr = LoKrLinear(64, 128, rank=4)
        
        delta = lokr.get_delta_weight()
        
        assert delta.shape == (128, 64)
        
    def test_factor_finding(self):
        """测试因子查找"""
        lokr = LoKrLinear(64, 128, rank=4)
        
        # 64 和 128 都能被 8 整除
        assert lokr.factor == 8


class TestDyLoRALinear:
    """测试 DyLoRA 线性层"""
    
    def test_initialization(self):
        """测试初始化"""
        dylora = DyLoRALinear(64, 128, max_rank=8, min_rank=2)
        
        assert dylora.in_features == 64
        assert dylora.out_features == 128
        assert dylora.max_rank == 8
        assert dylora.min_rank == 2
        assert dylora.current_rank == 8
        
    def test_forward_shape(self):
        """测试前向传播形状"""
        dylora = DyLoRALinear(64, 128, max_rank=8)
        x = torch.randn(2, 10, 64)
        
        output = dylora(x)
        
        assert output.shape == (2, 10, 128)
        
    def test_set_rank(self):
        """测试设置秩"""
        dylora = DyLoRALinear(64, 128, max_rank=8, min_rank=2)
        
        dylora.set_rank(4)
        assert dylora.current_rank == 4
        
        # 测试边界
        dylora.set_rank(1)  # 小于 min_rank
        assert dylora.current_rank == 2
        
        dylora.set_rank(10)  # 大于 max_rank
        assert dylora.current_rank == 8
        
    def test_different_ranks_different_outputs(self):
        """测试不同秩产生不同输出"""
        dylora = DyLoRALinear(64, 128, max_rank=8, min_rank=2)
        # 设置非零权重
        nn.init.normal_(dylora.lora_A.weight)
        nn.init.normal_(dylora.lora_B.weight)
        
        x = torch.randn(2, 10, 64)
        
        dylora.set_rank(4)
        output_rank4 = dylora(x).clone()
        
        dylora.set_rank(8)
        output_rank8 = dylora(x).clone()
        
        assert not torch.allclose(output_rank4, output_rank8)


class TestLoRAInjectedLinear:
    """测试 LoRA 注入线性层"""
    
    def test_initialization(self):
        """测试初始化"""
        original = nn.Linear(64, 128)
        injected = LoRAInjectedLinear(original, rank=4)
        
        assert injected.in_features == 64
        assert injected.out_features == 128
        assert injected.enabled == True
        
    def test_forward_with_lora(self):
        """测试带 LoRA 的前向传播"""
        original = nn.Linear(64, 128)
        injected = LoRAInjectedLinear(original, rank=4)
        x = torch.randn(2, 10, 64)
        
        output = injected(x)
        
        assert output.shape == (2, 10, 128)
        
    def test_forward_without_lora(self):
        """测试禁用 LoRA 的前向传播"""
        original = nn.Linear(64, 128)
        injected = LoRAInjectedLinear(original, rank=4)
        x = torch.randn(2, 10, 64)
        
        # 禁用 LoRA
        injected.enabled = False
        output_disabled = injected(x)
        
        # 原始输出
        expected = original(x)
        
        assert torch.allclose(output_disabled, expected)
        
    def test_lora_adds_to_output(self):
        """测试 LoRA 添加到输出"""
        original = nn.Linear(64, 128)
        injected = LoRAInjectedLinear(original, rank=4)
        
        # 设置非零 LoRA 权重
        nn.init.normal_(injected.lora.lora_B.weight)
        
        x = torch.randn(2, 10, 64)
        
        output_with_lora = injected(x)
        injected.enabled = False
        output_without_lora = injected(x)
        
        # 输出应该不同
        assert not torch.allclose(output_with_lora, output_without_lora)
        
    def test_merge_weights(self):
        """测试权重合并"""
        original = nn.Linear(64, 128)
        original_weight = original.weight.data.clone()
        injected = LoRAInjectedLinear(original, rank=4)
        
        # 设置非零 LoRA 权重
        nn.init.normal_(injected.lora.lora_B.weight)
        
        x = torch.randn(2, 10, 64)
        output_before = injected(x).clone()
        
        # 合并权重
        injected.merge_weights()
        
        # 权重应该改变
        assert not torch.allclose(original.weight.data, original_weight)
        
        # 输出应该相同（因为 LoRA 被禁用但权重已合并）
        output_after = injected(x)
        assert torch.allclose(output_before, output_after, atol=1e-5)
        
    def test_different_lora_types(self):
        """测试不同 LoRA 类型"""
        original = nn.Linear(64, 128)
        x = torch.randn(2, 10, 64)
        
        for lora_type in [LoRAType.LORA, LoRAType.LOHA, LoRAType.LOKR, LoRAType.DYLORA]:
            injected = LoRAInjectedLinear(original, rank=4, lora_type=lora_type)
            output = injected(x)
            assert output.shape == (2, 10, 128)


class TestLoRAInjectedConv2d:
    """测试 LoRA 注入卷积层"""
    
    def test_initialization(self):
        """测试初始化"""
        original = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        injected = LoRAInjectedConv2d(original, rank=4)
        
        assert injected.enabled == True
        
    def test_forward(self):
        """测试前向传播"""
        original = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        injected = LoRAInjectedConv2d(original, rank=4)
        x = torch.randn(2, 64, 16, 16)
        
        output = injected(x)
        
        assert output.shape == (2, 128, 16, 16)


class TestLoRAManager:
    """测试 LoRA 管理器"""
    
    @pytest.fixture
    def simple_model(self):
        """创建简单测试模型"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.out_proj = nn.Linear(64, 64)
                self.fc = nn.Linear(64, 128)  # 不应该被注入
                
            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                out = self.out_proj(q + k + v)
                return self.fc(out)
                
        return SimpleModel()
        
    def test_inject_lora(self, simple_model):
        """测试 LoRA 注入"""
        config = LoRAConfig(
            rank=4,
            target_modules=["q_proj", "v_proj"]
        )
        manager = LoRAManager(config)
        
        manager.inject_lora(simple_model)
        
        # 检查注入的模块
        assert len(manager.injected_modules) == 2
        assert "q_proj" in manager.injected_modules
        assert "v_proj" in manager.injected_modules
        assert "k_proj" not in manager.injected_modules
        
    def test_get_lora_parameters(self, simple_model):
        """测试获取 LoRA 参数"""
        config = LoRAConfig(rank=4, target_modules=["q_proj", "v_proj"])
        manager = LoRAManager(config)
        manager.inject_lora(simple_model)
        
        params = manager.get_lora_parameters()
        
        # 每个 LoRA 层有 2 个参数（A 和 B）
        assert len(params) == 4
        
    def test_enable_disable_lora(self, simple_model):
        """测试启用/禁用 LoRA"""
        config = LoRAConfig(rank=4, target_modules=["q_proj"])
        manager = LoRAManager(config)
        manager.inject_lora(simple_model)
        
        # 禁用
        manager.disable_lora()
        for module in manager.injected_modules.values():
            assert module.enabled == False
            
        # 启用
        manager.enable_lora()
        for module in manager.injected_modules.values():
            assert module.enabled == True
            
    def test_save_load_weights(self, simple_model):
        """测试保存和加载权重"""
        config = LoRAConfig(rank=4, target_modules=["q_proj"])
        manager = LoRAManager(config)
        manager.inject_lora(simple_model)
        
        # 设置非零权重
        for module in manager.injected_modules.values():
            nn.init.normal_(module.lora.lora_A.weight)
            nn.init.normal_(module.lora.lora_B.weight)
            
        # 保存权重
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            
        try:
            manager.save_lora_weights(temp_path)
            
            # 创建新模型和管理器
            new_model = simple_model.__class__()
            new_manager = LoRAManager(config)
            new_manager.inject_lora(new_model)
            
            # 加载权重
            new_manager.load_lora_weights(temp_path)
            
            # 验证权重相同
            for name in manager.injected_modules:
                old_A = manager.injected_modules[name].lora.lora_A.weight
                new_A = new_manager.injected_modules[name].lora.lora_A.weight
                assert torch.allclose(old_A, new_A)
        finally:
            os.unlink(temp_path)


class TestCreateLoRAConfig:
    """测试 LoRA 配置工厂函数"""
    
    def test_create_default(self):
        """测试创建默认配置"""
        config = create_lora_config()
        
        assert config.rank == 4
        assert config.alpha == 1.0
        assert config.lora_type == LoRAType.LORA
        
    def test_create_loha(self):
        """测试创建 LoHA 配置"""
        config = create_lora_config(lora_type="loha")
        
        assert config.lora_type == LoRAType.LOHA
        
    def test_create_lokr(self):
        """测试创建 LoKr 配置"""
        config = create_lora_config(lora_type="lokr")
        
        assert config.lora_type == LoRAType.LOKR
        
    def test_create_dylora(self):
        """测试创建 DyLoRA 配置"""
        config = create_lora_config(lora_type="dylora")
        
        assert config.lora_type == LoRAType.DYLORA
        
    def test_create_unknown_raises(self):
        """测试未知类型抛出异常"""
        with pytest.raises(ValueError):
            create_lora_config(lora_type="unknown")
            
    def test_custom_target_modules(self):
        """测试自定义目标模块"""
        config = create_lora_config(target_modules=["fc1", "fc2"])
        
        assert config.target_modules == ["fc1", "fc2"]


class TestInjectLoRAToModel:
    """测试便捷注入函数"""
    
    def test_inject_returns_model_and_manager(self):
        """测试返回模型和管理器"""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        model, manager = inject_lora_to_model(
            model, rank=4, target_modules=["0", "2"]
        )
        
        assert isinstance(manager, LoRAManager)
        assert len(manager.injected_modules) == 2


class TestLoRAIntegration:
    """LoRA 集成测试"""
    
    def test_training_loop_simulation(self):
        """模拟训练循环"""
        # 创建模型
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 注入 LoRA
        model, manager = inject_lora_to_model(
            model, rank=4, target_modules=["0", "2"]
        )
        
        # 冻结原始参数
        for name, param in model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
                
        # 只有 LoRA 参数可训练
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) == 4  # 2 layers * 2 params (A, B)
        
        # 模拟前向传播
        x = torch.randn(2, 64)
        output = model(x)
        
        assert output.shape == (2, 64)
        
    def test_lora_reduces_trainable_params(self):
        """测试 LoRA 减少可训练参数"""
        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1024, 1024)
            def forward(self, x):
                return self.fc(x)
                
        model = SimpleLinear()
        original_params = sum(p.numel() for p in model.parameters())
        
        model, manager = inject_lora_to_model(
            model, rank=4, target_modules=["fc"]
        )
        
        # 冻结原始参数
        for name, param in model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
                
        lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # LoRA 参数应该远少于原始参数
        # rank=4: A(1024*4) + B(4*1024) = 8192 vs 1024*1024 = 1048576
        assert lora_params < original_params * 0.01
