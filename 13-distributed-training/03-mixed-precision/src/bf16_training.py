"""
BF16 (Brain Floating Point 16) 训练实现

BF16 是一种16位浮点格式，具有与 FP32 相同的指数范围，
因此不需要梯度缩放，训练更稳定。

核心特点:
    - 8位指数 + 7位尾数 + 1位符号
    - 动态范围与 FP32 相同
    - 无需 GradScaler
    - 需要 Ampere 或更新架构 GPU
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


def is_bf16_supported() -> bool:
    """检查当前设备是否支持 BF16
    
    Returns:
        是否支持 BF16
    """
    if not torch.cuda.is_available():
        return False
    
    # 检查 CUDA 计算能力 (需要 >= 8.0，即 Ampere 架构)
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return capability[0] >= 8


@dataclass
class BF16Config:
    """BF16 训练配置
    
    Attributes:
        enabled: 是否启用 BF16
        convert_weights: 是否转换模型权重为 BF16
        keep_batchnorm_fp32: 是否保持 BatchNorm 为 FP32
        keep_layernorm_fp32: 是否保持 LayerNorm 为 FP32
        master_weights: 是否使用 FP32 主权重
        patch_torch_functions: 是否修补 torch 函数
    """
    enabled: bool = True
    convert_weights: bool = True
    keep_batchnorm_fp32: bool = True
    keep_layernorm_fp32: bool = True
    master_weights: bool = True
    patch_torch_functions: bool = False


def convert_to_bf16(
    model: nn.Module,
    keep_batchnorm_fp32: bool = True,
    keep_layernorm_fp32: bool = True,
) -> nn.Module:
    """将模型转换为 BF16
    
    Args:
        model: PyTorch 模型
        keep_batchnorm_fp32: 保持 BatchNorm 为 FP32
        keep_layernorm_fp32: 保持 LayerNorm 为 FP32
        
    Returns:
        转换后的模型
    """
    # 收集需要保持 FP32 的模块
    fp32_modules = []
    
    for name, module in model.named_modules():
        if keep_batchnorm_fp32 and isinstance(module, (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )):
            fp32_modules.append(name)
        
        if keep_layernorm_fp32 and isinstance(module, nn.LayerNorm):
            fp32_modules.append(name)
    
    # 转换模型为 BF16
    model = model.to(torch.bfloat16)
    
    # 恢复需要保持 FP32 的模块
    for name, module in model.named_modules():
        if name in fp32_modules:
            module.float()
    
    return model


class BF16Trainer:
    """BF16 训练器
    
    Args:
        model: PyTorch 模型
        config: BF16 配置
        device: 训练设备
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[BF16Config] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or BF16Config()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # 检查 BF16 支持
        if self.config.enabled and not is_bf16_supported():
            print("Warning: BF16 not supported, falling back to FP32")
            self.config.enabled = False
        
        # 转换模型
        if self.config.enabled and self.config.convert_weights:
            model = convert_to_bf16(
                model,
                keep_batchnorm_fp32=self.config.keep_batchnorm_fp32,
                keep_layernorm_fp32=self.config.keep_layernorm_fp32,
            )
        
        self.model = model.to(self.device)
        
        # 主权重（FP32 副本）
        self.master_weights: Optional[Dict[str, torch.Tensor]] = None
        if self.config.enabled and self.config.master_weights:
            self._init_master_weights()
    
    def _init_master_weights(self) -> None:
        """初始化 FP32 主权重"""
        self.master_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.master_weights[name] = param.data.float().clone()
    
    def autocast(self):
        """获取 BF16 autocast 上下文"""
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.config.enabled,
        )
    
    def forward(self, *args, **kwargs) -> Any:
        """带 autocast 的前向传播"""
        with self.autocast():
            return self.model(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """反向传播（BF16 不需要梯度缩放）"""
        loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """优化器步骤"""
        if self.master_weights is not None:
            # 使用主权重更新
            self._step_with_master_weights(optimizer)
        else:
            optimizer.step()
    
    def _step_with_master_weights(self, optimizer: torch.optim.Optimizer) -> None:
        """使用 FP32 主权重进行优化器步骤"""
        # 将梯度复制到主权重
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.master_weights:
                    # 创建临时 FP32 梯度
                    self.master_weights[name].grad = param.grad.float()
        
        # 在 FP32 主权重上执行优化器步骤
        # 注意：这需要优化器使用主权重
        optimizer.step()
        
        # 将更新后的主权重复制回 BF16 模型
        for name, param in self.model.named_parameters():
            if name in self.master_weights:
                param.data.copy_(self.master_weights[name].to(torch.bfloat16))
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """清零梯度"""
        self.model.zero_grad(set_to_none=set_to_none)
        if self.master_weights is not None:
            for tensor in self.master_weights.values():
                if tensor.grad is not None:
                    if set_to_none:
                        tensor.grad = None
                    else:
                        tensor.grad.zero_()
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = {"model": self.model.state_dict()}
        if self.master_weights is not None:
            state["master_weights"] = self.master_weights
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """加载状态字典"""
        self.model.load_state_dict(state["model"])
        if "master_weights" in state:
            self.master_weights = state["master_weights"]
