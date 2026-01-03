"""
剪枝模块单元测试
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pruning import (
    PruningConfig,
    PruningType,
    ImportanceMetric,
    PruningMask,
    MagnitudePruner,
    StructuredPruner,
    GradientPruner,
    IterativePruner,
    prune_model,
    compute_model_sparsity,
    compute_magnitude_importance,
    compute_gradient_importance,
    compute_taylor_importance,
    create_pruning_mask,
)


# ==================== 测试模型 ====================

class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvModel(nn.Module):
    """卷积测试模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==================== 重要性计算测试 ====================

class TestImportanceComputation:
    """测试重要性计算"""

    def test_magnitude_importance_unstructured(self):
        """测试非结构化幅度重要性"""
        weight = torch.randn(32, 64)
        importance = compute_magnitude_importance(weight, dim=None)

        assert importance.shape == weight.shape
        assert (importance >= 0).all()

    def test_magnitude_importance_structured(self):
        """测试结构化幅度重要性"""
        weight = torch.randn(32, 64)
        importance = compute_magnitude_importance(weight, dim=0)

        assert importance.shape == (32,)
        assert (importance >= 0).all()

    def test_gradient_importance(self):
        """测试梯度重要性"""
        weight = torch.randn(32, 64)
        gradient = torch.randn(32, 64)

        importance = compute_gradient_importance(weight, gradient, dim=None)

        assert importance.shape == weight.shape
        assert (importance >= 0).all()

    def test_taylor_importance(self):
        """测试 Taylor 重要性"""
        weight = torch.randn(32, 64)
        gradient = torch.randn(32, 64)

        importance = compute_taylor_importance(weight, gradient, dim=None)

        assert importance.shape == weight.shape
        assert (importance >= 0).all()


# ==================== 掩码创建测试 ====================

class TestPruningMask:
    """测试剪枝掩码"""

    def test_create_unstructured_mask(self):
        """测试创建非结构化掩码"""
        importance = torch.rand(32, 64)
        mask = create_pruning_mask(importance, sparsity=0.5, structured=False)

        assert mask.shape == importance.shape
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

        # 检查稀疏度
        actual_sparsity = (mask == 0).float().mean().item()
        assert abs(actual_sparsity - 0.5) < 0.1

    def test_create_structured_mask(self):
        """测试创建结构化掩码"""
        importance = torch.rand(32)
        mask = create_pruning_mask(importance, sparsity=0.5, structured=True)

        assert mask.shape == importance.shape
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_zero_sparsity(self):
        """测试零稀疏度"""
        importance = torch.rand(32, 64)
        mask = create_pruning_mask(importance, sparsity=0.0, structured=False)

        assert (mask == 1).all()


class TestPruningMaskManager:
    """测试掩码管理器"""

    def test_register_and_get_mask(self):
        """测试注册和获取掩码"""
        manager = PruningMask()
        mask = torch.ones(32, 64)

        manager.register_mask("layer1", mask)
        retrieved = manager.get_mask("layer1")

        assert retrieved is not None
        assert torch.equal(retrieved, mask)

    def test_get_nonexistent_mask(self):
        """测试获取不存在的掩码"""
        manager = PruningMask()
        retrieved = manager.get_mask("nonexistent")

        assert retrieved is None

    def test_get_sparsity(self):
        """测试计算稀疏度"""
        manager = PruningMask()
        mask = torch.tensor([[1, 0, 1, 0], [0, 0, 1, 1]], dtype=torch.float)

        manager.register_mask("layer1", mask)
        sparsity = manager.get_sparsity()

        assert sparsity == 0.5  # 4/8 = 0.5


# ==================== MagnitudePruner 测试 ====================

class TestMagnitudePruner:
    """测试幅度剪枝器"""

    def test_local_pruning(self):
        """测试局部剪枝"""
        model = SimpleModel()
        config = PruningConfig(sparsity=0.3, global_pruning=False)
        pruner = MagnitudePruner(config)

        pruned_model = pruner.prune(model)

        # 检查模型仍然可以前向传播
        x = torch.randn(8, 64)
        y = pruned_model(x)
        assert y.shape == (8, 10)

        # 检查稀疏度
        sparsity = pruner.get_sparsity()
        assert sparsity > 0

    def test_global_pruning(self):
        """测试全局剪枝"""
        model = SimpleModel()
        config = PruningConfig(sparsity=0.3, global_pruning=True)
        pruner = MagnitudePruner(config)

        pruned_model = pruner.prune(model)

        x = torch.randn(8, 64)
        y = pruned_model(x)
        assert y.shape == (8, 10)

    def test_structured_pruning(self):
        """测试结构化剪枝"""
        model = SimpleModel()
        config = PruningConfig(
            pruning_type=PruningType.STRUCTURED,
            sparsity=0.3
        )
        pruner = MagnitudePruner(config)

        pruned_model = pruner.prune(model)

        x = torch.randn(8, 64)
        y = pruned_model(x)
        assert y.shape == (8, 10)


# ==================== StructuredPruner 测试 ====================

class TestStructuredPruner:
    """测试结构化剪枝器"""

    def test_prune_linear(self):
        """测试剪枝线性层"""
        model = SimpleModel()
        pruner = StructuredPruner()

        pruned_model = pruner.prune(model, sparsity=0.3)

        x = torch.randn(8, 64)
        y = pruned_model(x)
        assert y.shape == (8, 10)

    def test_prune_conv(self):
        """测试剪枝卷积层"""
        model = ConvModel()
        pruner = StructuredPruner()

        pruned_model = pruner.prune(model, sparsity=0.3)

        x = torch.randn(4, 3, 32, 32)
        y = pruned_model(x)
        assert y.shape == (4, 10)

    def test_get_pruned_indices(self):
        """测试获取剪枝索引"""
        model = SimpleModel()
        pruner = StructuredPruner()

        pruner.prune(model, sparsity=0.3)
        indices = pruner.get_pruned_indices(model)

        assert isinstance(indices, dict)

    def test_channel_importance(self):
        """测试通道重要性计算"""
        pruner = StructuredPruner()
        conv_weight = torch.randn(16, 3, 3, 3)

        importance = pruner.compute_channel_importance(conv_weight)

        assert importance.shape == (16,)
        assert (importance >= 0).all()

    def test_channel_importance_with_bn(self):
        """测试带 BN 的通道重要性"""
        pruner = StructuredPruner()
        conv_weight = torch.randn(16, 3, 3, 3)
        bn_weight = torch.randn(16)

        importance = pruner.compute_channel_importance(conv_weight, bn_weight)

        assert importance.shape == (16,)


# ==================== GradientPruner 测试 ====================

class TestGradientPruner:
    """测试梯度剪枝器"""

    def test_collect_gradients(self):
        """测试收集梯度"""
        model = SimpleModel()
        pruner = GradientPruner()

        # 创建数据加载器
        data = [(torch.randn(8, 64), torch.randint(0, 10, (8,))) for _ in range(5)]
        criterion = nn.CrossEntropyLoss()

        pruner.collect_gradients(model, data, criterion, num_batches=3)

        assert len(pruner.gradients) > 0

    def test_prune_after_gradient_collection(self):
        """测试收集梯度后剪枝"""
        model = SimpleModel()
        pruner = GradientPruner()

        data = [(torch.randn(8, 64), torch.randint(0, 10, (8,))) for _ in range(5)]
        criterion = nn.CrossEntropyLoss()

        pruner.collect_gradients(model, data, criterion, num_batches=3)
        pruned_model = pruner.prune(model, sparsity=0.3)

        x = torch.randn(8, 64)
        y = pruned_model(x)
        assert y.shape == (8, 10)

    def test_prune_without_gradients_raises(self):
        """测试未收集梯度时剪枝抛出异常"""
        model = SimpleModel()
        pruner = GradientPruner()

        with pytest.raises(RuntimeError):
            pruner.prune(model)


# ==================== 便捷函数测试 ====================

class TestPruneModel:
    """测试 prune_model 便捷函数"""

    def test_unstructured_pruning(self):
        """测试非结构化剪枝"""
        model = SimpleModel()
        pruned = prune_model(model, sparsity=0.3, pruning_type="unstructured")

        x = torch.randn(8, 64)
        y = pruned(x)
        assert y.shape == (8, 10)

    def test_structured_pruning(self):
        """测试结构化剪枝"""
        model = SimpleModel()
        pruned = prune_model(model, sparsity=0.3, pruning_type="structured")

        x = torch.randn(8, 64)
        y = pruned(x)
        assert y.shape == (8, 10)


class TestComputeModelSparsity:
    """测试模型稀疏度计算"""

    def test_compute_sparsity(self):
        """测试计算稀疏度"""
        model = SimpleModel()

        # 手动设置一些权重为零
        with torch.no_grad():
            model.fc1.weight[0, :] = 0

        stats = compute_model_sparsity(model)

        assert "overall" in stats
        assert "layers" in stats
        assert "total_params" in stats
        assert "zero_params" in stats
        assert stats["overall"] > 0

    def test_dense_model_sparsity(self):
        """测试稠密模型稀疏度"""
        model = SimpleModel()

        # 确保没有零权重
        with torch.no_grad():
            for param in model.parameters():
                param.data = torch.randn_like(param.data) + 0.1

        stats = compute_model_sparsity(model)

        assert stats["overall"] == 0.0


# ==================== 配置测试 ====================

class TestPruningConfig:
    """测试剪枝配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = PruningConfig()

        assert config.pruning_type == PruningType.UNSTRUCTURED
        assert config.sparsity == 0.5
        assert config.importance_metric == ImportanceMetric.MAGNITUDE

    def test_custom_config(self):
        """测试自定义配置"""
        config = PruningConfig(
            pruning_type=PruningType.STRUCTURED,
            sparsity=0.7,
            importance_metric=ImportanceMetric.GRADIENT,
            global_pruning=True
        )

        assert config.pruning_type == PruningType.STRUCTURED
        assert config.sparsity == 0.7
        assert config.importance_metric == ImportanceMetric.GRADIENT
        assert config.global_pruning is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
