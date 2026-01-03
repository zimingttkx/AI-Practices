"""
知识蒸馏模块单元测试
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distillation import (
    DistillationConfig,
    DistillationType,
    DistillationLoss,
    FeatureDistillation,
    RelationDistillation,
    AttentionTransfer,
    DistanceWiseDistillation,
    AngleWiseDistillation,
    FeatureExtractor,
    KnowledgeDistiller,
    distill_model,
    soft_cross_entropy,
)


# ==================== 测试模型 ====================

class TeacherModel(nn.Module):
    """教师模型 (较大)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class StudentModel(nn.Module):
    """学生模型 (较小)"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvTeacher(nn.Module):
    """卷积教师模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvStudent(nn.Module):
    """卷积学生模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==================== 软标签测试 ====================

class TestSoftCrossEntropy:
    """测试软标签交叉熵"""

    def test_basic_computation(self):
        """测试基本计算"""
        student_logits = torch.randn(8, 10)
        teacher_logits = torch.randn(8, 10)

        loss = soft_cross_entropy(student_logits, teacher_logits, temperature=4.0)

        assert loss.dim() == 0  # 标量
        assert loss >= 0

    def test_temperature_effect(self):
        """测试温度参数效果"""
        student_logits = torch.randn(8, 10)
        teacher_logits = torch.randn(8, 10)

        loss_t1 = soft_cross_entropy(student_logits, teacher_logits, temperature=1.0)
        loss_t4 = soft_cross_entropy(student_logits, teacher_logits, temperature=4.0)

        # 不同温度应该产生不同的损失
        assert loss_t1 != loss_t4

    def test_identical_logits(self):
        """测试相同 logits"""
        logits = torch.randn(8, 10)

        loss = soft_cross_entropy(logits, logits, temperature=4.0)

        # 相同分布的 KL 散度应该接近 0
        assert loss < 0.1


# ==================== DistillationLoss 测试 ====================

class TestDistillationLoss:
    """测试蒸馏损失"""

    def test_forward(self):
        """测试前向传播"""
        loss_fn = DistillationLoss(temperature=4.0, alpha=0.7)

        student_logits = torch.randn(8, 10)
        teacher_logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))

        total_loss, loss_dict = loss_fn(student_logits, teacher_logits, labels)

        assert total_loss.dim() == 0
        assert "soft_loss" in loss_dict
        assert "hard_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_alpha_weighting(self):
        """测试 alpha 权重"""
        student_logits = torch.randn(8, 10)
        teacher_logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))

        loss_fn_high_alpha = DistillationLoss(alpha=0.9)
        loss_fn_low_alpha = DistillationLoss(alpha=0.1)

        loss_high, _ = loss_fn_high_alpha(student_logits, teacher_logits, labels)
        loss_low, _ = loss_fn_low_alpha(student_logits, teacher_logits, labels)

        # 不同 alpha 应该产生不同的损失
        assert loss_high != loss_low


# ==================== 特征蒸馏测试 ====================

class TestFeatureDistillation:
    """测试特征蒸馏"""

    def test_same_dimension(self):
        """测试相同维度"""
        distiller = FeatureDistillation(64, 64, use_projector=False)

        student_features = torch.randn(8, 64, 4, 4)
        teacher_features = torch.randn(8, 64, 4, 4)

        loss = distiller(student_features, teacher_features)

        assert loss.dim() == 0
        assert loss >= 0

    def test_different_dimension(self):
        """测试不同维度 (需要投影)"""
        distiller = FeatureDistillation(32, 64, use_projector=True)

        student_features = torch.randn(8, 32, 4, 4)
        teacher_features = torch.randn(8, 64, 4, 4)

        loss = distiller(student_features, teacher_features)

        assert loss.dim() == 0
        assert loss >= 0


class TestAttentionTransfer:
    """测试注意力迁移"""

    def test_attention_map(self):
        """测试注意力图计算"""
        transfer = AttentionTransfer(p=2)
        features = torch.randn(8, 64, 4, 4)

        attention = transfer.attention_map(features)

        assert attention.shape == (8, 16)  # 4*4 = 16

    def test_forward(self):
        """测试前向传播"""
        transfer = AttentionTransfer()

        student_features = torch.randn(8, 32, 4, 4)
        teacher_features = torch.randn(8, 64, 4, 4)

        loss = transfer(student_features, teacher_features)

        assert loss.dim() == 0
        assert loss >= 0


# ==================== 关系蒸馏测试 ====================

class TestRelationDistillation:
    """测试关系蒸馏"""

    def test_cosine_similarity(self):
        """测试余弦相似度"""
        distiller = RelationDistillation(distance_type="cosine")

        student_features = torch.randn(8, 64)
        teacher_features = torch.randn(8, 128)

        loss = distiller(student_features, teacher_features)

        assert loss.dim() == 0
        assert loss >= 0

    def test_euclidean_distance(self):
        """测试欧氏距离"""
        distiller = RelationDistillation(distance_type="euclidean")

        student_features = torch.randn(8, 64)
        teacher_features = torch.randn(8, 128)

        loss = distiller(student_features, teacher_features)

        assert loss.dim() == 0
        assert loss >= 0

    def test_similarity_matrix(self):
        """测试相似度矩阵计算"""
        distiller = RelationDistillation(distance_type="cosine")
        features = torch.randn(8, 64)

        similarity = distiller.compute_similarity_matrix(features)

        assert similarity.shape == (8, 8)


class TestDistanceWiseDistillation:
    """测试距离蒸馏"""

    def test_forward(self):
        """测试前向传播"""
        distiller = DistanceWiseDistillation()

        student_features = torch.randn(8, 64)
        teacher_features = torch.randn(8, 128)

        loss = distiller(student_features, teacher_features)

        assert loss.dim() == 0
        assert loss >= 0


class TestAngleWiseDistillation:
    """测试角度蒸馏"""

    def test_forward(self):
        """测试前向传播"""
        distiller = AngleWiseDistillation()

        student_features = torch.randn(8, 64)
        teacher_features = torch.randn(8, 128)

        loss = distiller(student_features, teacher_features)

        assert loss.dim() == 0
        assert loss >= 0

    def test_small_batch(self):
        """测试小批次"""
        distiller = AngleWiseDistillation()

        student_features = torch.randn(2, 64)
        teacher_features = torch.randn(2, 128)

        loss = distiller(student_features, teacher_features)

        # 批次太小时应该返回 0
        assert loss == 0.0


# ==================== FeatureExtractor 测试 ====================

class TestFeatureExtractor:
    """测试特征提取器"""

    def test_extract_features(self):
        """测试特征提取"""
        model = TeacherModel()
        extractor = FeatureExtractor(model, ["fc1", "fc2"])

        x = torch.randn(8, 64)
        features = extractor.extract(x)

        assert "fc1" in features
        assert "fc2" in features

    def test_remove_hooks(self):
        """测试移除钩子"""
        model = TeacherModel()
        extractor = FeatureExtractor(model, ["fc1"])

        extractor.remove_hooks()

        assert len(extractor.hooks) == 0


# ==================== KnowledgeDistiller 测试 ====================

class TestKnowledgeDistiller:
    """测试知识蒸馏器"""

    def test_initialization(self):
        """测试初始化"""
        teacher = TeacherModel()
        student = StudentModel()

        distiller = KnowledgeDistiller(teacher, student)

        # 教师模型应该被冻结
        for param in distiller.teacher.parameters():
            assert not param.requires_grad

    def test_train_basic(self):
        """测试基本训练"""
        teacher = TeacherModel()
        student = StudentModel()

        distiller = KnowledgeDistiller(teacher, student)

        # 创建训练数据
        train_data = [
            (torch.randn(8, 64), torch.randint(0, 10, (8,)))
            for _ in range(5)
        ]

        trained_student = distiller.train(
            train_data,
            num_epochs=2,
            verbose=False
        )

        # 检查学生模型可以前向传播
        x = torch.randn(8, 64)
        y = trained_student(x)
        assert y.shape == (8, 10)

    def test_train_with_relation_distillation(self):
        """测试带关系蒸馏的训练"""
        teacher = TeacherModel()
        student = StudentModel()

        distiller = KnowledgeDistiller(teacher, student)
        distiller.setup_relation_distillation(distance_type="cosine")

        train_data = [
            (torch.randn(8, 64), torch.randint(0, 10, (8,)))
            for _ in range(5)
        ]

        trained_student = distiller.train(
            train_data,
            num_epochs=2,
            verbose=False
        )

        assert trained_student is not None

    def test_evaluate(self):
        """测试评估"""
        teacher = TeacherModel()
        student = StudentModel()

        distiller = KnowledgeDistiller(teacher, student)

        test_data = [
            (torch.randn(8, 64), torch.randint(0, 10, (8,)))
            for _ in range(5)
        ]

        metrics = distiller.evaluate(test_data)

        assert "student_accuracy" in metrics
        assert "teacher_accuracy" in metrics
        assert "total_samples" in metrics

    def test_get_history(self):
        """测试获取历史"""
        teacher = TeacherModel()
        student = StudentModel()

        distiller = KnowledgeDistiller(teacher, student)

        train_data = [
            (torch.randn(8, 64), torch.randint(0, 10, (8,)))
            for _ in range(5)
        ]

        distiller.train(train_data, num_epochs=3, verbose=False)
        history = distiller.get_history()

        assert len(history) == 3


# ==================== 便捷函数测试 ====================

class TestDistillModel:
    """测试 distill_model 便捷函数"""

    def test_basic_distillation(self):
        """测试基本蒸馏"""
        teacher = TeacherModel()
        student = StudentModel()

        train_data = [
            (torch.randn(8, 64), torch.randint(0, 10, (8,)))
            for _ in range(5)
        ]

        trained_student = distill_model(
            teacher, student, train_data,
            temperature=4.0,
            alpha=0.7,
            num_epochs=2
        )

        x = torch.randn(8, 64)
        y = trained_student(x)
        assert y.shape == (8, 10)


# ==================== 配置测试 ====================

class TestDistillationConfig:
    """测试蒸馏配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = DistillationConfig()

        assert config.distillation_type == DistillationType.RESPONSE
        assert config.temperature == 4.0
        assert config.alpha == 0.7

    def test_custom_config(self):
        """测试自定义配置"""
        config = DistillationConfig(
            distillation_type=DistillationType.FEATURE,
            temperature=2.0,
            alpha=0.5,
            feature_weight=2.0
        )

        assert config.distillation_type == DistillationType.FEATURE
        assert config.temperature == 2.0
        assert config.alpha == 0.5
        assert config.feature_weight == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
