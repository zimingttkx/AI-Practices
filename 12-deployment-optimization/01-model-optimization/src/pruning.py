"""
模型剪枝 (Model Pruning)

本模块实现深度学习模型的剪枝技术，包括：
- 非结构化剪枝 (Unstructured Pruning)
- 结构化剪枝 (Structured Pruning)
- 迭代剪枝 (Iterative Pruning)

=== 剪枝原理 ===

剪枝通过移除模型中不重要的参数来减小模型大小和计算量：

1. 非结构化剪枝: 移除单个权重，产生稀疏矩阵
2. 结构化剪枝: 移除整个结构（通道、层、注意力头），产生规则的小模型

=== 参考文献 ===

1. Han et al. "Deep Compression: Compressing Deep Neural Networks with
   Pruning, Trained Quantization and Huffman Coding" 2016
2. Li et al. "Pruning Filters for Efficient ConvNets" 2017
3. Molchanov et al. "Importance Estimation for Neural Network Pruning" 2019
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class PruningType(Enum):
    """剪枝类型"""
    UNSTRUCTURED = "unstructured"  # 非结构化剪枝
    STRUCTURED = "structured"      # 结构化剪枝


class ImportanceMetric(Enum):
    """重要性评估指标"""
    MAGNITUDE = "magnitude"        # 幅度 (L1/L2 范数)
    GRADIENT = "gradient"          # 梯度
    TAYLOR = "taylor"              # Taylor 展开
    RANDOM = "random"              # 随机


@dataclass
class PruningConfig:
    """剪枝配置"""

    # 剪枝类型
    pruning_type: PruningType = PruningType.UNSTRUCTURED

    # 稀疏度 (要剪枝的比例)
    sparsity: float = 0.5

    # 重要性评估指标
    importance_metric: ImportanceMetric = ImportanceMetric.MAGNITUDE

    # 结构化剪枝的维度 (0: 输出通道, 1: 输入通道)
    structured_dim: int = 0

    # 迭代剪枝配置
    iterative: bool = False
    num_iterations: int = 5
    finetune_epochs: int = 1

    # 要剪枝的层类型
    prune_layers: List[str] = field(default_factory=lambda: ["Linear", "Conv2d"])

    # 全局剪枝 vs 局部剪枝
    global_pruning: bool = False


class PruningMask:
    """剪枝掩码管理器"""

    def __init__(self):
        self.masks: Dict[str, torch.Tensor] = {}

    def register_mask(self, name: str, mask: torch.Tensor):
        """注册掩码"""
        self.masks[name] = mask

    def get_mask(self, name: str) -> Optional[torch.Tensor]:
        """获取掩码"""
        return self.masks.get(name)

    def apply_masks(self, model: nn.Module):
        """应用所有掩码到模型"""
        for name, module in model.named_modules():
            if name in self.masks:
                if hasattr(module, 'weight'):
                    module.weight.data *= self.masks[name].to(module.weight.device)

    def get_sparsity(self) -> float:
        """计算总体稀疏度"""
        total_params = 0
        pruned_params = 0
        for mask in self.masks.values():
            total_params += mask.numel()
            pruned_params += (mask == 0).sum().item()
        return pruned_params / total_params if total_params > 0 else 0.0


def compute_magnitude_importance(weight: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    计算基于幅度的重要性

    Args:
        weight: 权重张量
        dim: 结构化剪枝的维度 (None 表示非结构化)

    Returns:
        重要性分数
    """
    if dim is None:
        # 非结构化: 每个权重的绝对值
        return weight.abs()
    else:
        # 结构化: 沿指定维度的 L1 范数
        dims_to_reduce = list(range(weight.dim()))
        dims_to_reduce.remove(dim)
        return weight.abs().sum(dim=dims_to_reduce)


def compute_gradient_importance(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    dim: Optional[int] = None
) -> torch.Tensor:
    """
    计算基于梯度的重要性

    Args:
        weight: 权重张量
        gradient: 梯度张量
        dim: 结构化剪枝的维度

    Returns:
        重要性分数
    """
    importance = (weight * gradient).abs()

    if dim is None:
        return importance
    else:
        dims_to_reduce = list(range(importance.dim()))
        dims_to_reduce.remove(dim)
        return importance.sum(dim=dims_to_reduce)


def compute_taylor_importance(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    dim: Optional[int] = None
) -> torch.Tensor:
    """
    计算基于 Taylor 展开的重要性

    一阶近似: |w * g|
    其中 g = dL/dw

    Args:
        weight: 权重张量
        gradient: 梯度张量
        dim: 结构化剪枝的维度

    Returns:
        重要性分数
    """
    # 一阶 Taylor 展开
    importance = (weight * gradient).abs()

    if dim is None:
        return importance
    else:
        dims_to_reduce = list(range(importance.dim()))
        dims_to_reduce.remove(dim)
        return importance.sum(dim=dims_to_reduce)


def create_pruning_mask(
    importance: torch.Tensor,
    sparsity: float,
    structured: bool = False
) -> torch.Tensor:
    """
    创建剪枝掩码

    Args:
        importance: 重要性分数
        sparsity: 稀疏度 (要剪枝的比例)
        structured: 是否结构化剪枝

    Returns:
        二值掩码 (1: 保留, 0: 剪枝)
    """
    if structured:
        # 结构化剪枝: 按重要性排序，剪枝最不重要的
        num_prune = int(len(importance) * sparsity)
        if num_prune == 0:
            return torch.ones_like(importance)

        threshold = torch.topk(importance, num_prune, largest=False)[0].max()
        mask = (importance > threshold).float()
    else:
        # 非结构化剪枝
        flat_importance = importance.flatten()
        num_prune = int(len(flat_importance) * sparsity)
        if num_prune == 0:
            return torch.ones_like(importance)

        threshold = torch.topk(flat_importance, num_prune, largest=False)[0].max()
        mask = (importance > threshold).float().view_as(importance)

    return mask


class MagnitudePruner:
    """
    基于幅度的剪枝器

    移除绝对值最小的权重。
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()
        self.mask_manager = PruningMask()

    def compute_importance(self, module: nn.Module, name: str) -> torch.Tensor:
        """计算模块的重要性"""
        weight = module.weight.data

        if self.config.pruning_type == PruningType.STRUCTURED:
            return compute_magnitude_importance(weight, self.config.structured_dim)
        else:
            return compute_magnitude_importance(weight, None)

    def prune(self, model: nn.Module, sparsity: Optional[float] = None) -> nn.Module:
        """
        对模型进行剪枝

        Args:
            model: 要剪枝的模型
            sparsity: 稀疏度 (覆盖配置)

        Returns:
            剪枝后的模型
        """
        sparsity = sparsity or self.config.sparsity

        if self.config.global_pruning:
            return self._global_prune(model, sparsity)
        else:
            return self._local_prune(model, sparsity)

    def _local_prune(self, model: nn.Module, sparsity: float) -> nn.Module:
        """局部剪枝: 每层独立剪枝"""
        for name, module in model.named_modules():
            if self._should_prune(module):
                importance = self.compute_importance(module, name)
                mask = create_pruning_mask(
                    importance, sparsity,
                    structured=(self.config.pruning_type == PruningType.STRUCTURED)
                )

                # 应用掩码
                if self.config.pruning_type == PruningType.STRUCTURED:
                    # 结构化剪枝: 扩展掩码到完整形状
                    mask = self._expand_structured_mask(mask, module.weight.shape)

                self.mask_manager.register_mask(name, mask)
                module.weight.data *= mask.to(module.weight.device)

        return model

    def _global_prune(self, model: nn.Module, sparsity: float) -> nn.Module:
        """全局剪枝: 所有层一起剪枝"""
        # 收集所有权重的重要性
        all_importance = []
        module_info = []

        for name, module in model.named_modules():
            if self._should_prune(module):
                importance = self.compute_importance(module, name)
                all_importance.append(importance.flatten())
                module_info.append((name, module, importance.shape))

        if not all_importance:
            return model

        # 全局计算阈值
        global_importance = torch.cat(all_importance)
        num_prune = int(len(global_importance) * sparsity)
        threshold = torch.topk(global_importance, num_prune, largest=False)[0].max()

        # 应用掩码
        idx = 0
        for name, module, shape in module_info:
            importance = all_importance[module_info.index((name, module, shape))]
            mask = (importance > threshold).float().view(shape)

            if self.config.pruning_type == PruningType.STRUCTURED:
                mask = self._expand_structured_mask(mask, module.weight.shape)

            self.mask_manager.register_mask(name, mask)
            module.weight.data *= mask.to(module.weight.device)

        return model

    def _should_prune(self, module: nn.Module) -> bool:
        """判断是否应该剪枝该模块"""
        module_type = type(module).__name__
        return module_type in self.config.prune_layers and hasattr(module, 'weight')

    def _expand_structured_mask(self, mask: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """扩展结构化掩码到完整形状"""
        # mask 形状: [num_channels]
        # target_shape: [out_channels, in_channels, ...] 或 [out_features, in_features]
        expanded = mask.clone()
        for _ in range(len(target_shape) - 1):
            expanded = expanded.unsqueeze(-1)
        return expanded.expand(target_shape)

    def get_sparsity(self) -> float:
        """获取当前稀疏度"""
        return self.mask_manager.get_sparsity()


class StructuredPruner:
    """
    结构化剪枝器

    移除整个结构（通道、神经元），产生规则的小模型。
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        config = config or PruningConfig()
        config.pruning_type = PruningType.STRUCTURED
        self.config = config
        self.mask_manager = PruningMask()

    def compute_channel_importance(
        self,
        conv_weight: torch.Tensor,
        bn_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算通道重要性

        Args:
            conv_weight: 卷积权重 [out_channels, in_channels, H, W]
            bn_weight: BN 层权重 [out_channels]

        Returns:
            每个通道的重要性分数
        """
        # L1 范数
        importance = conv_weight.abs().sum(dim=(1, 2, 3))

        # 结合 BN 缩放因子
        if bn_weight is not None:
            importance = importance * bn_weight.abs()

        return importance

    def prune(
        self,
        model: nn.Module,
        sparsity: Optional[float] = None
    ) -> nn.Module:
        """
        对模型进行结构化剪枝

        Args:
            model: 要剪枝的模型
            sparsity: 稀疏度

        Returns:
            剪枝后的模型
        """
        sparsity = sparsity or self.config.sparsity

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                importance = self.compute_channel_importance(module.weight.data)
                mask = create_pruning_mask(importance, sparsity, structured=True)

                # 扩展掩码
                expanded_mask = mask.view(-1, 1, 1, 1).expand_as(module.weight)
                self.mask_manager.register_mask(name, expanded_mask)
                module.weight.data *= expanded_mask

            elif isinstance(module, nn.Linear):
                # 对输出神经元剪枝
                importance = module.weight.data.abs().sum(dim=1)
                mask = create_pruning_mask(importance, sparsity, structured=True)

                expanded_mask = mask.view(-1, 1).expand_as(module.weight)
                self.mask_manager.register_mask(name, expanded_mask)
                module.weight.data *= expanded_mask

        return model

    def get_pruned_indices(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        获取被剪枝的索引

        Returns:
            每层被剪枝的通道/神经元索引
        """
        pruned_indices = {}
        for name, mask in self.mask_manager.masks.items():
            if mask.dim() == 4:  # Conv2d
                channel_mask = mask[:, 0, 0, 0]
            else:  # Linear
                channel_mask = mask[:, 0]
            pruned_indices[name] = (channel_mask == 0).nonzero(as_tuple=True)[0]
        return pruned_indices

    def get_sparsity(self) -> float:
        """获取当前稀疏度"""
        return self.mask_manager.get_sparsity()


class GradientPruner:
    """
    基于梯度的剪枝器

    使用梯度信息评估参数重要性。
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        config = config or PruningConfig()
        config.importance_metric = ImportanceMetric.GRADIENT
        self.config = config
        self.mask_manager = PruningMask()
        self.gradients: Dict[str, torch.Tensor] = {}

    def collect_gradients(
        self,
        model: nn.Module,
        data_loader,
        criterion: Callable,
        num_batches: int = 10
    ):
        """
        收集梯度信息

        Args:
            model: 模型
            data_loader: 数据加载器
            criterion: 损失函数
            num_batches: 收集批次数
        """
        model.train()
        self.gradients = {}

        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None

            model.zero_grad()
            outputs = model(inputs)

            if targets is not None:
                loss = criterion(outputs, targets)
            else:
                loss = outputs.sum()

            loss.backward()

            # 累积梯度
            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    if name not in self.gradients:
                        self.gradients[name] = module.weight.grad.abs().clone()
                    else:
                        self.gradients[name] += module.weight.grad.abs()

        # 平均梯度
        for name in self.gradients:
            self.gradients[name] /= num_batches

    def prune(
        self,
        model: nn.Module,
        sparsity: Optional[float] = None
    ) -> nn.Module:
        """
        基于梯度进行剪枝

        Args:
            model: 要剪枝的模型
            sparsity: 稀疏度

        Returns:
            剪枝后的模型
        """
        if not self.gradients:
            raise RuntimeError("请先调用 collect_gradients() 收集梯度")

        sparsity = sparsity or self.config.sparsity

        for name, module in model.named_modules():
            if name in self.gradients and hasattr(module, 'weight'):
                weight = module.weight.data
                gradient = self.gradients[name]

                # 计算重要性
                if self.config.pruning_type == PruningType.STRUCTURED:
                    importance = compute_gradient_importance(
                        weight, gradient, self.config.structured_dim
                    )
                    mask = create_pruning_mask(importance, sparsity, structured=True)
                    mask = self._expand_structured_mask(mask, weight.shape)
                else:
                    importance = compute_gradient_importance(weight, gradient, None)
                    mask = create_pruning_mask(importance, sparsity, structured=False)

                self.mask_manager.register_mask(name, mask)
                module.weight.data *= mask.to(module.weight.device)

        return model

    def _expand_structured_mask(self, mask: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """扩展结构化掩码"""
        expanded = mask.clone()
        for _ in range(len(target_shape) - 1):
            expanded = expanded.unsqueeze(-1)
        return expanded.expand(target_shape)

    def get_sparsity(self) -> float:
        """获取当前稀疏度"""
        return self.mask_manager.get_sparsity()


class IterativePruner:
    """
    迭代剪枝器

    逐步剪枝并微调，减少精度损失。
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        config = config or PruningConfig()
        config.iterative = True
        self.config = config
        self.base_pruner = MagnitudePruner(config)
        self.pruning_history: List[Dict] = []

    def prune(
        self,
        model: nn.Module,
        train_loader,
        criterion: Callable,
        optimizer_fn: Callable,
        target_sparsity: Optional[float] = None,
        num_iterations: Optional[int] = None,
        finetune_epochs: Optional[int] = None,
        device: str = "cpu"
    ) -> nn.Module:
        """
        迭代剪枝

        Args:
            model: 要剪枝的模型
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer_fn: 优化器构造函数
            target_sparsity: 目标稀疏度
            num_iterations: 迭代次数
            finetune_epochs: 每次迭代的微调轮数
            device: 设备

        Returns:
            剪枝后的模型
        """
        target_sparsity = target_sparsity or self.config.sparsity
        num_iterations = num_iterations or self.config.num_iterations
        finetune_epochs = finetune_epochs or self.config.finetune_epochs

        # 计算每次迭代的剪枝比例
        # 最终稀疏度 = 1 - (1 - p)^n
        # p = 1 - (1 - target_sparsity)^(1/n)
        per_iteration_sparsity = 1 - (1 - target_sparsity) ** (1 / num_iterations)

        model = model.to(device)

        for iteration in range(num_iterations):
            print(f"迭代 {iteration + 1}/{num_iterations}")

            # 剪枝
            current_sparsity = 1 - (1 - per_iteration_sparsity) ** (iteration + 1)
            model = self.base_pruner.prune(model, per_iteration_sparsity)

            actual_sparsity = self.base_pruner.get_sparsity()
            print(f"  当前稀疏度: {actual_sparsity:.2%}")

            # 微调
            optimizer = optimizer_fn(model.parameters())
            model = self._finetune(
                model, train_loader, criterion, optimizer,
                finetune_epochs, device
            )

            # 记录历史
            self.pruning_history.append({
                "iteration": iteration + 1,
                "sparsity": actual_sparsity,
            })

        return model

    def _finetune(
        self,
        model: nn.Module,
        train_loader,
        criterion: Callable,
        optimizer,
        epochs: int,
        device: str
    ) -> nn.Module:
        """微调模型"""
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    continue

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # 应用掩码到梯度
                self.base_pruner.mask_manager.apply_masks(model)

                optimizer.step()

                # 重新应用掩码
                self.base_pruner.mask_manager.apply_masks(model)

                total_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"    微调 Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        return model

    def get_sparsity(self) -> float:
        """获取当前稀疏度"""
        return self.base_pruner.get_sparsity()

    def get_history(self) -> List[Dict]:
        """获取剪枝历史"""
        return self.pruning_history


def prune_model(
    model: nn.Module,
    sparsity: float = 0.5,
    pruning_type: str = "unstructured",
    importance_metric: str = "magnitude",
    global_pruning: bool = False
) -> nn.Module:
    """
    剪枝模型的便捷函数

    Args:
        model: 要剪枝的模型
        sparsity: 稀疏度
        pruning_type: 剪枝类型 ("unstructured", "structured")
        importance_metric: 重要性指标 ("magnitude", "gradient", "taylor")
        global_pruning: 是否全局剪枝

    Returns:
        剪枝后的模型
    """
    config = PruningConfig(
        pruning_type=PruningType(pruning_type),
        sparsity=sparsity,
        importance_metric=ImportanceMetric(importance_metric),
        global_pruning=global_pruning
    )

    if pruning_type == "structured":
        pruner = StructuredPruner(config)
    else:
        pruner = MagnitudePruner(config)

    return pruner.prune(model)


def compute_model_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    计算模型的稀疏度统计

    Args:
        model: 模型

    Returns:
        稀疏度统计字典
    """
    total_params = 0
    zero_params = 0
    layer_sparsity = {}

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            weight = module.weight.data
            num_params = weight.numel()
            num_zeros = (weight == 0).sum().item()

            total_params += num_params
            zero_params += num_zeros

            layer_sparsity[name] = num_zeros / num_params if num_params > 0 else 0.0

    overall_sparsity = zero_params / total_params if total_params > 0 else 0.0

    return {
        "overall": overall_sparsity,
        "layers": layer_sparsity,
        "total_params": total_params,
        "zero_params": zero_params
    }
