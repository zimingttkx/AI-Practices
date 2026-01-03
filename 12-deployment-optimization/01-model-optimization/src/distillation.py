"""
知识蒸馏 (Knowledge Distillation)

本模块实现深度学习模型的知识蒸馏技术，包括：
- 响应蒸馏 (Response-based Distillation)
- 特征蒸馏 (Feature-based Distillation)
- 关系蒸馏 (Relation-based Distillation)

=== 蒸馏原理 ===

知识蒸馏将大型教师模型的知识迁移到小型学生模型：

1. 软标签 (Soft Labels): 使用温度缩放的 softmax 输出
   P_i = exp(z_i / T) / sum(exp(z_j / T))

2. 蒸馏损失: L = α * L_soft + (1 - α) * L_hard
   - L_soft: KL 散度损失 (软标签)
   - L_hard: 交叉熵损失 (硬标签)

=== 参考文献 ===

1. Hinton et al. "Distilling the Knowledge in a Neural Network" 2015
2. Romero et al. "FitNets: Hints for Thin Deep Nets" 2015
3. Park et al. "Relational Knowledge Distillation" 2019
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationType(Enum):
    """蒸馏类型"""
    RESPONSE = "response"      # 响应蒸馏
    FEATURE = "feature"        # 特征蒸馏
    RELATION = "relation"      # 关系蒸馏
    COMBINED = "combined"      # 组合蒸馏


@dataclass
class DistillationConfig:
    """蒸馏配置"""

    # 蒸馏类型
    distillation_type: DistillationType = DistillationType.RESPONSE

    # 温度参数
    temperature: float = 4.0

    # 软标签权重 (alpha)
    alpha: float = 0.7

    # 特征蒸馏配置
    feature_layers: List[str] = field(default_factory=list)
    feature_weight: float = 1.0

    # 关系蒸馏配置
    relation_weight: float = 1.0

    # 训练配置
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 32


def soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0
) -> torch.Tensor:
    """
    软标签交叉熵损失

    Args:
        student_logits: 学生模型输出 [batch, num_classes]
        teacher_logits: 教师模型输出 [batch, num_classes]
        temperature: 温度参数

    Returns:
        软标签损失
    """
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)

    # KL 散度
    loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

    # 乘以 T^2 保持梯度量级
    return loss * (temperature ** 2)


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失函数

    组合软标签损失和硬标签损失。
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7
    ):
        """
        Args:
            temperature: 温度参数，控制软标签的平滑程度
            alpha: 软标签损失的权重
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算蒸馏损失

        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签

        Returns:
            总损失, 损失分解字典
        """
        # 软标签损失
        soft_loss = soft_cross_entropy(
            student_logits, teacher_logits, self.temperature
        )

        # 硬标签损失
        hard_loss = F.cross_entropy(student_logits, labels)

        # 组合损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss, {
            "soft_loss": soft_loss,
            "hard_loss": hard_loss,
            "total_loss": total_loss
        }


class FeatureDistillation(nn.Module):
    """
    特征蒸馏

    匹配教师和学生模型的中间层特征。
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        use_projector: bool = True
    ):
        """
        Args:
            student_channels: 学生特征通道数
            teacher_channels: 教师特征通道数
            use_projector: 是否使用投影层对齐维度
        """
        super().__init__()
        self.use_projector = use_projector

        if use_projector and student_channels != teacher_channels:
            self.projector = nn.Conv2d(
                student_channels, teacher_channels,
                kernel_size=1, bias=False
            )
        else:
            self.projector = nn.Identity()

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算特征蒸馏损失

        Args:
            student_features: 学生特征 [batch, C, H, W] 或 [batch, dim]
            teacher_features: 教师特征

        Returns:
            特征匹配损失
        """
        # 投影学生特征
        if self.use_projector:
            student_features = self.projector(student_features)

        # 归一化
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features.detach(), p=2, dim=1)

        # MSE 损失
        return F.mse_loss(student_norm, teacher_norm)


class AttentionTransfer(nn.Module):
    """
    注意力迁移

    匹配教师和学生的注意力图。
    """

    def __init__(self, p: int = 2):
        """
        Args:
            p: 范数的阶数
        """
        super().__init__()
        self.p = p

    def attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算注意力图

        Args:
            features: 特征图 [batch, C, H, W]

        Returns:
            注意力图 [batch, H, W]
        """
        return F.normalize(
            features.pow(self.p).mean(dim=1).view(features.size(0), -1),
            p=2, dim=1
        )

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算注意力迁移损失

        Args:
            student_features: 学生特征
            teacher_features: 教师特征

        Returns:
            注意力迁移损失
        """
        student_attention = self.attention_map(student_features)
        teacher_attention = self.attention_map(teacher_features.detach())

        return (student_attention - teacher_attention).pow(2).mean()


class RelationDistillation(nn.Module):
    """
    关系蒸馏

    保持样本间的关系结构。
    """

    def __init__(self, distance_type: str = "cosine"):
        """
        Args:
            distance_type: 距离类型 ("cosine", "euclidean")
        """
        super().__init__()
        self.distance_type = distance_type

    def compute_similarity_matrix(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算样本间相似度矩阵

        Args:
            features: 特征 [batch, dim]

        Returns:
            相似度矩阵 [batch, batch]
        """
        if self.distance_type == "cosine":
            # 余弦相似度
            features_norm = F.normalize(features, p=2, dim=1)
            similarity = torch.mm(features_norm, features_norm.t())
        else:
            # 欧氏距离 (转换为相似度)
            dist = torch.cdist(features, features, p=2)
            similarity = 1 / (1 + dist)

        return similarity

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算关系蒸馏损失

        Args:
            student_features: 学生特征 [batch, dim]
            teacher_features: 教师特征 [batch, dim]

        Returns:
            关系蒸馏损失
        """
        # 展平特征
        if student_features.dim() > 2:
            student_features = student_features.view(student_features.size(0), -1)
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.view(teacher_features.size(0), -1)

        # 计算相似度矩阵
        student_sim = self.compute_similarity_matrix(student_features)
        teacher_sim = self.compute_similarity_matrix(teacher_features.detach())

        # MSE 损失
        return F.mse_loss(student_sim, teacher_sim)


class DistanceWiseDistillation(nn.Module):
    """
    距离蒸馏 (RKD-D)

    保持样本对之间的距离关系。
    """

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算距离蒸馏损失
        """
        # 展平
        if student_features.dim() > 2:
            student_features = student_features.view(student_features.size(0), -1)
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.view(teacher_features.size(0), -1)

        # 计算成对距离
        student_dist = torch.cdist(student_features, student_features, p=2)
        teacher_dist = torch.cdist(teacher_features.detach(), teacher_features.detach(), p=2)

        # 归一化
        student_dist = student_dist / (student_dist.mean() + 1e-8)
        teacher_dist = teacher_dist / (teacher_dist.mean() + 1e-8)

        return F.smooth_l1_loss(student_dist, teacher_dist)


class AngleWiseDistillation(nn.Module):
    """
    角度蒸馏 (RKD-A)

    保持样本三元组之间的角度关系。
    """

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算角度蒸馏损失
        """
        # 展平
        if student_features.dim() > 2:
            student_features = student_features.view(student_features.size(0), -1)
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.view(teacher_features.size(0), -1)

        batch_size = student_features.size(0)
        if batch_size < 3:
            return torch.tensor(0.0, device=student_features.device)

        # 计算角度
        student_angles = self._compute_angles(student_features)
        teacher_angles = self._compute_angles(teacher_features.detach())

        return F.smooth_l1_loss(student_angles, teacher_angles)

    def _compute_angles(self, features: torch.Tensor) -> torch.Tensor:
        """计算所有三元组的角度"""
        # 简化实现: 使用 Gram 矩阵
        features_norm = F.normalize(features, p=2, dim=1)
        gram = torch.mm(features_norm, features_norm.t())
        return gram


class FeatureExtractor:
    """
    特征提取器

    用于提取模型中间层的特征。
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        """
        Args:
            model: 模型
            layer_names: 要提取特征的层名称
        """
        self.model = model
        self.layer_names = layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks: List = []

        self._register_hooks()

    def _register_hooks(self):
        """注册前向钩子"""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        """创建钩子函数"""
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取特征

        Args:
            x: 输入

        Returns:
            各层特征字典
        """
        self.features = {}
        _ = self.model(x)
        return self.features.copy()

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class KnowledgeDistiller:
    """
    知识蒸馏器

    完整的知识蒸馏训练流程。
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None
    ):
        """
        Args:
            teacher: 教师模型
            student: 学生模型
            config: 蒸馏配置
        """
        self.teacher = teacher
        self.student = student
        self.config = config or DistillationConfig()

        # 冻结教师模型
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 损失函数
        self.distillation_loss = DistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha
        )

        # 特征蒸馏
        self.feature_distillers: Dict[str, FeatureDistillation] = {}
        self.teacher_extractor: Optional[FeatureExtractor] = None
        self.student_extractor: Optional[FeatureExtractor] = None

        # 关系蒸馏
        self.relation_distiller: Optional[RelationDistillation] = None

        # 训练历史
        self.history: List[Dict] = []

    def setup_feature_distillation(
        self,
        layer_pairs: List[Tuple[str, str, int, int]]
    ):
        """
        设置特征蒸馏

        Args:
            layer_pairs: [(student_layer, teacher_layer, student_dim, teacher_dim), ...]
        """
        student_layers = [p[0] for p in layer_pairs]
        teacher_layers = [p[1] for p in layer_pairs]

        self.student_extractor = FeatureExtractor(self.student, student_layers)
        self.teacher_extractor = FeatureExtractor(self.teacher, teacher_layers)

        for student_layer, teacher_layer, student_dim, teacher_dim in layer_pairs:
            key = f"{student_layer}->{teacher_layer}"
            self.feature_distillers[key] = FeatureDistillation(
                student_dim, teacher_dim
            )

    def setup_relation_distillation(self, distance_type: str = "cosine"):
        """
        设置关系蒸馏

        Args:
            distance_type: 距离类型
        """
        self.relation_distiller = RelationDistillation(distance_type)

    def train(
        self,
        train_loader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: Optional[int] = None,
        device: str = "cpu",
        verbose: bool = True
    ) -> nn.Module:
        """
        训练学生模型

        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            num_epochs: 训练轮数
            device: 设备
            verbose: 是否打印训练信息

        Returns:
            训练后的学生模型
        """
        num_epochs = num_epochs or self.config.num_epochs

        # 移动模型到设备
        self.teacher = self.teacher.to(device)
        self.student = self.student.to(device)

        # 移动特征蒸馏模块
        for distiller in self.feature_distillers.values():
            distiller.to(device)

        # 创建优化器
        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.student.parameters(),
                lr=self.config.learning_rate
            )

        # 训练循环
        for epoch in range(num_epochs):
            self.student.train()
            epoch_losses = {
                "total": 0.0,
                "soft": 0.0,
                "hard": 0.0,
                "feature": 0.0,
                "relation": 0.0
            }
            num_batches = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                else:
                    inputs = batch.to(device)
                    labels = None

                optimizer.zero_grad()

                # 前向传播
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                student_logits = self.student(inputs)

                # 计算蒸馏损失
                if labels is not None:
                    total_loss, loss_dict = self.distillation_loss(
                        student_logits, teacher_logits, labels
                    )
                    epoch_losses["soft"] += loss_dict["soft_loss"].item()
                    epoch_losses["hard"] += loss_dict["hard_loss"].item()
                else:
                    total_loss = soft_cross_entropy(
                        student_logits, teacher_logits, self.config.temperature
                    )
                    epoch_losses["soft"] += total_loss.item()

                # 特征蒸馏损失
                if self.feature_distillers:
                    feature_loss = self._compute_feature_loss(inputs)
                    total_loss = total_loss + self.config.feature_weight * feature_loss
                    epoch_losses["feature"] += feature_loss.item()

                # 关系蒸馏损失
                if self.relation_distiller is not None:
                    relation_loss = self._compute_relation_loss(inputs)
                    total_loss = total_loss + self.config.relation_weight * relation_loss
                    epoch_losses["relation"] += relation_loss.item()

                # 反向传播
                total_loss.backward()
                optimizer.step()

                epoch_losses["total"] += total_loss.item()
                num_batches += 1

            # 计算平均损失
            for key in epoch_losses:
                epoch_losses[key] /= max(num_batches, 1)

            self.history.append(epoch_losses)

            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Loss: {epoch_losses['total']:.4f} "
                      f"(soft: {epoch_losses['soft']:.4f}, "
                      f"hard: {epoch_losses['hard']:.4f})")

        return self.student

    def _compute_feature_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算特征蒸馏损失"""
        if not self.feature_distillers:
            return torch.tensor(0.0, device=inputs.device)

        # 提取特征
        with torch.no_grad():
            teacher_features = self.teacher_extractor.extract(inputs)
        student_features = self.student_extractor.extract(inputs)

        # 计算损失
        total_loss = torch.tensor(0.0, device=inputs.device)
        for key, distiller in self.feature_distillers.items():
            student_layer, teacher_layer = key.split("->")
            if student_layer in student_features and teacher_layer in teacher_features:
                loss = distiller(
                    student_features[student_layer],
                    teacher_features[teacher_layer]
                )
                total_loss = total_loss + loss

        return total_loss

    def _compute_relation_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算关系蒸馏损失"""
        if self.relation_distiller is None:
            return torch.tensor(0.0, device=inputs.device)

        # 获取最后一层特征
        with torch.no_grad():
            teacher_out = self.teacher(inputs)
        student_out = self.student(inputs)

        return self.relation_distiller(student_out, teacher_out)

    def evaluate(
        self,
        test_loader,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        评估学生模型

        Args:
            test_loader: 测试数据加载器
            device: 设备

        Returns:
            评估指标
        """
        self.student.eval()
        self.teacher.eval()

        student_correct = 0
        teacher_correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (tuple, list)):
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                else:
                    continue

                student_logits = self.student(inputs)
                teacher_logits = self.teacher(inputs)

                student_preds = student_logits.argmax(dim=1)
                teacher_preds = teacher_logits.argmax(dim=1)

                student_correct += (student_preds == labels).sum().item()
                teacher_correct += (teacher_preds == labels).sum().item()
                total += labels.size(0)

        return {
            "student_accuracy": student_correct / total if total > 0 else 0.0,
            "teacher_accuracy": teacher_correct / total if total > 0 else 0.0,
            "total_samples": total
        }

    def get_history(self) -> List[Dict]:
        """获取训练历史"""
        return self.history


def distill_model(
    teacher: nn.Module,
    student: nn.Module,
    train_loader,
    temperature: float = 4.0,
    alpha: float = 0.7,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cpu"
) -> nn.Module:
    """
    知识蒸馏的便捷函数

    Args:
        teacher: 教师模型
        student: 学生模型
        train_loader: 训练数据加载器
        temperature: 温度参数
        alpha: 软标签权重
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备

    Returns:
        训练后的学生模型
    """
    config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

    distiller = KnowledgeDistiller(teacher, student, config)
    return distiller.train(train_loader, device=device)
