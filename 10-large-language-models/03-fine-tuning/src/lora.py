"""
LoRA (Low-Rank Adaptation) 实现

本模块提供 LoRA 的生产级实现，基于论文
"LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)。

核心组件：
    - LoRAConfig: LoRA配置
    - LoRALinear: LoRA线性层
    - apply_lora_to_model: 将LoRA应用到模型

"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "apply_lora_to_model",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_weights",
    "load_lora_weights",
]


@dataclass
class LoRAConfig:
    """LoRA 配置。

    参数：
        r: 低秩矩阵的秩
        alpha: 缩放因子
        dropout: Dropout概率
        target_modules: 要应用LoRA的模块名称模式
        modules_to_save: 需要完整保存的模块
        bias: 偏置处理方式 ("none", "all", "lora_only")
    """
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: Optional[List[str]] = None
    bias: str = "none"
    
    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"r必须为正数，得到 {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"alpha必须为正数，得到 {self.alpha}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout必须在[0,1)范围内，得到 {self.dropout}")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(f"bias必须是'none', 'all', 'lora_only'之一")


class LoRALinear(nn.Module):
    """LoRA线性层。

    将原始线性层包装，添加低秩适配器。
    
    数学原理：
        h = Wx + BAx * (alpha/r)
        
    其中：
        - W: 原始权重（冻结）
        - B: 低秩矩阵 [out_features, r]
        - A: 低秩矩阵 [r, in_features]
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # 冻结原始权重
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
        
        # LoRA矩阵
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self._init_weights()
        
        # 是否已合并
        self.merged = False

    def _init_weights(self) -> None:
        """初始化LoRA权重。A用kaiming，B用零初始化。"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        if self.merged:
            return self.original_layer(x)
        
        # 原始输出
        result = self.original_layer(x)
        
        # LoRA增量: x @ A^T @ B^T * scaling
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_output * self.scaling
        
        return result

    def merge(self) -> None:
        """将LoRA权重合并到原始权重。"""
        if not self.merged:
            self.original_layer.weight.data += (
                self.lora_B @ self.lora_A * self.scaling
            )
            self.merged = True

    def unmerge(self) -> None:
        """从原始权重中移除LoRA权重。"""
        if self.merged:
            self.original_layer.weight.data -= (
                self.lora_B @ self.lora_A * self.scaling
            )
            self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, merged={self.merged}"


def _find_modules_to_modify(
    model: nn.Module,
    target_modules: List[str],
) -> Dict[str, nn.Linear]:
    """查找需要应用LoRA的模块。"""
    modules_to_modify = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否匹配目标模块
            for target in target_modules:
                if re.search(target, name):
                    modules_to_modify[name] = module
                    break
    
    return modules_to_modify


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """将LoRA应用到模型。

    参数：
        model: 原始模型
        config: LoRA配置
        
    返回：
        应用了LoRA的模型
    """
    # 查找目标模块
    modules_to_modify = _find_modules_to_modify(model, config.target_modules)
    
    if not modules_to_modify:
        raise ValueError(f"未找到匹配的模块: {config.target_modules}")
    
    # 替换为LoRA层
    for name, module in modules_to_modify.items():
        # 获取父模块和属性名
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name
        
        # 创建LoRA层
        lora_layer = LoRALinear(
            module,
            r=config.r,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        
        # 替换
        setattr(parent, attr_name, lora_layer)
    
    # 冻结非LoRA参数
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
    
    # 处理bias
    if config.bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif config.bias == "lora_only":
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                if module.original_layer.bias is not None:
                    module.original_layer.bias.requires_grad = True
    
    # 打印信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA应用完成:")
    print(f"  - 修改的模块: {len(modules_to_modify)}")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def get_lora_parameters(model: nn.Module) -> Dict[str, Tensor]:
    """获取所有LoRA参数。"""
    lora_params = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_params[name] = param
    return lora_params


def merge_lora_weights(model: nn.Module) -> None:
    """合并所有LoRA权重到原始权重。"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    """从原始权重中移除所有LoRA权重。"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def save_lora_weights(model: nn.Module, path: str) -> None:
    """保存LoRA权重。"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state_dict[name] = param.data
    torch.save(lora_state_dict, path)
    print(f"LoRA权重已保存到: {path}")


def load_lora_weights(model: nn.Module, path: str) -> None:
    """加载LoRA权重。"""
    lora_state_dict = torch.load(path, map_location="cpu")
    
    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
    
    print(f"LoRA权重已从 {path} 加载")
