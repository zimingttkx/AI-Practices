"""
BF16 (Brain Floating Point 16) 训练实现

================================================================================
核心思想 (一句话理解)
================================================================================
BF16 = 8位指数(与FP32相同) + 7位尾数 = 无需梯度缩放的16位训练

================================================================================
为什么BF16比FP16更适合训练？
================================================================================

    FP16的问题:
    ┌─────────────────────────────────────────────────────────────────┐
    │  FP16: 1位符号 + 5位指数 + 10位尾数                              │
    │                                                                 │
    │  数值范围: 6×10⁻⁵ ~ 65504 (很窄!)                               │
    │                                                                 │
    │  问题:                                                          │
    │  - 梯度下溢: 小于6×10⁻⁵的梯度变成0                              │
    │  - 梯度溢出: 大于65504的值变成inf                               │
    │  - 需要复杂的损失缩放机制                                        │
    └─────────────────────────────────────────────────────────────────┘

    BF16的解决方案:
    ┌─────────────────────────────────────────────────────────────────┐
    │  BF16: 1位符号 + 8位指数 + 7位尾数                              │
    │                                                                 │
    │  数值范围: ±3.4×10³⁸ (与FP32完全相同!)                          │
    │                                                                 │
    │  优点:                                                          │
    │  - 无需梯度缩放 (范围足够大)                                     │
    │  - 训练更稳定                                                   │
    │  - 代码更简单                                                   │
    │                                                                 │
    │  代价:                                                          │
    │  - 精度略低 (7位尾数 vs FP16的10位)                             │
    │  - 但对训练影响很小                                              │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
数据类型对比 (图解)
================================================================================

    位分配对比:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  FP32:  S | EEEEEEEE | MMMMMMMMMMMMMMMMMMMMMMM                  │
    │         1    8位指数        23位尾数                             │
    │         范围大，精度高                                           │
    │                                                                 │
    │  FP16:  S | EEEEE | MMMMMMMMMM                                  │
    │         1   5位指数   10位尾数                                   │
    │         范围小，精度中                                           │
    │                                                                 │
    │  BF16:  S | EEEEEEEE | MMMMMMM                                  │
    │         1    8位指数    7位尾数                                  │
    │         范围大，精度低                                           │
    │                                                                 │
    │  关键: BF16的指数位与FP32相同 → 数值范围相同!                    │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
硬件要求
================================================================================
    - NVIDIA Ampere架构及以上 (A100, RTX 30系列, RTX 40系列)
    - 计算能力 >= 8.0
    - 旧GPU (V100, RTX 20系列) 不支持BF16

================================================================================
前置知识
================================================================================
- 浮点数的指数和尾数概念
- FP16训练的梯度缩放问题
- PyTorch的autocast机制

================================================================================
参考文献
================================================================================
- Kalamkar et al., "A Study of BFLOAT16 for Deep Learning Training", 2019
- Google TPU documentation on bfloat16
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


# =============================================================================
# 辅助函数
# =============================================================================

def is_bf16_supported() -> bool:
    """检查当前设备是否支持BF16

    BF16需要Ampere架构及以上的GPU (计算能力 >= 8.0)

    Returns:
        True如果支持BF16，否则False

    Example:
        >>> if is_bf16_supported():
        ...     print("可以使用BF16训练")
        ... else:
        ...     print("请使用FP16 + AMP")
    """
    if not torch.cuda.is_available():
        return False

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    # Ampere架构的计算能力是8.0
    return capability[0] >= 8


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class BF16Config:
    """BF16训练配置

    Attributes:
        enabled: 是否启用BF16训练
        convert_weights: 是否将模型权重转换为BF16
        keep_batchnorm_fp32: 是否保持BatchNorm为FP32 (数值稳定性)
        keep_layernorm_fp32: 是否保持LayerNorm为FP32 (数值稳定性)
        master_weights: 是否维护FP32主权重 (更新精度)
        patch_torch_functions: 是否修补torch函数以支持BF16

    为什么某些层要保持FP32?
        - BatchNorm/LayerNorm计算均值和方差
        - 这些统计量对精度敏感
        - 使用FP32可以避免数值不稳定

    Example:
        >>> config = BF16Config(
        ...     enabled=True,
        ...     keep_batchnorm_fp32=True,  # 推荐
        ...     master_weights=True,        # 推荐
        ... )
    """
    enabled: bool = True
    convert_weights: bool = True
    keep_batchnorm_fp32: bool = True
    keep_layernorm_fp32: bool = True
    master_weights: bool = True
    patch_torch_functions: bool = False


# =============================================================================
# 模型转换函数
# =============================================================================

def convert_to_bf16(
    model: nn.Module,
    keep_batchnorm_fp32: bool = True,
    keep_layernorm_fp32: bool = True,
) -> nn.Module:
    """将模型转换为BF16，同时保持归一化层为FP32

    工作流程:
    1. 记录需要保持FP32的层
    2. 将整个模型转换为BF16
    3. 将记录的层转回FP32

    Args:
        model: 要转换的PyTorch模型
        keep_batchnorm_fp32: 是否保持BatchNorm为FP32
        keep_layernorm_fp32: 是否保持LayerNorm为FP32

    Returns:
        转换后的模型 (BF16权重，但归一化层为FP32)

    Example:
        >>> model = MyModel()
        >>> model = convert_to_bf16(model, keep_batchnorm_fp32=True)
        >>> # 大部分层是BF16，BatchNorm是FP32
    """
    # 记录需要保持FP32的层
    fp32_modules = []

    for name, module in model.named_modules():
        # BatchNorm层
        if keep_batchnorm_fp32 and isinstance(module, (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )):
            fp32_modules.append(name)

        # LayerNorm层
        if keep_layernorm_fp32 and isinstance(module, nn.LayerNorm):
            fp32_modules.append(name)

    # 将整个模型转换为BF16
    model = model.to(torch.bfloat16)

    # 将记录的层转回FP32
    for name, module in model.named_modules():
        if name in fp32_modules:
            module.float()

    return model


# =============================================================================
# BF16训练器
# =============================================================================

class BF16Trainer:
    """BF16混合精度训练器

    与FP16不同，BF16不需要梯度缩放，因为它的数值范围与FP32相同。

    工作流程:
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. with trainer.autocast():  # 进入BF16上下文                  │
    │         output = model(input)                                   │
    │         loss = criterion(output, target)                        │
    │                                                                 │
    │  2. trainer.backward(loss)    # 直接反向传播 (无需缩放!)        │
    │                                                                 │
    │  3. trainer.step(optimizer)   # 直接更新 (可选用主权重)         │
    └─────────────────────────────────────────────────────────────────┘

    主权重 (Master Weights) 机制:
    ┌─────────────────────────────────────────────────────────────────┐
    │  问题: BF16精度较低，累积更新可能有误差                          │
    │                                                                 │
    │  解决: 维护FP32主权重                                            │
    │  - 模型权重: BF16 (用于前向/反向)                                │
    │  - 主权重: FP32 (用于优化器更新)                                 │
    │  - 每步: FP32主权重更新 → 复制到BF16模型权重                     │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        model: PyTorch模型
        config: BF16配置
        device: 训练设备

    Example:
        >>> trainer = BF16Trainer(model, BF16Config())
        >>> with trainer.autocast():
        ...     loss = model(batch)
        >>> trainer.backward(loss)  # 无需缩放!
        >>> trainer.step(optimizer)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[BF16Config] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or BF16Config()

        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # 检查BF16支持
        if self.config.enabled and not is_bf16_supported():
            print("警告: 当前GPU不支持BF16，回退到FP32")
            self.config.enabled = False

        # 转换模型权重为BF16
        if self.config.enabled and self.config.convert_weights:
            model = convert_to_bf16(
                model,
                keep_batchnorm_fp32=self.config.keep_batchnorm_fp32,
                keep_layernorm_fp32=self.config.keep_layernorm_fp32,
            )

        self.model = model.to(self.device)

        # 初始化FP32主权重 (可选)
        self.master_weights: Optional[Dict[str, torch.Tensor]] = None
        if self.config.enabled and self.config.master_weights:
            self._init_master_weights()

    def _init_master_weights(self) -> None:
        """初始化FP32主权重

        为每个需要梯度的参数创建一个FP32副本。
        优化器更新在FP32主权重上进行，然后复制回BF16模型权重。
        """
        self.master_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 创建FP32副本
                self.master_weights[name] = param.data.float().clone()

    def autocast(self):
        """获取BF16 autocast上下文管理器

        Returns:
            autocast上下文管理器

        Example:
            >>> with trainer.autocast():
            ...     output = model(input)  # 自动使用BF16
        """
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.config.enabled,
        )

    def forward(self, *args, **kwargs) -> Any:
        """带BF16 autocast的前向传播

        Args:
            *args: 模型输入参数
            **kwargs: 模型输入关键字参数

        Returns:
            模型输出
        """
        with self.autocast():
            return self.model(*args, **kwargs)

    def backward(self, loss: torch.Tensor) -> None:
        """反向传播 (BF16无需梯度缩放!)

        Args:
            loss: 损失张量
        """
        # BF16的数值范围与FP32相同，无需缩放
        loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """优化器更新步骤

        如果启用了主权重:
        1. 将BF16梯度复制到FP32主权重
        2. 在FP32主权重上执行优化器更新
        3. 将更新后的FP32主权重复制回BF16模型权重

        Args:
            optimizer: 优化器
        """
        if self.master_weights is not None:
            self._step_with_master_weights(optimizer)
        else:
            optimizer.step()

    def _step_with_master_weights(self, optimizer: torch.optim.Optimizer) -> None:
        """使用FP32主权重进行更新

        这样可以保证优化器更新的数值精度。

        Args:
            optimizer: 优化器
        """
        # 1. 将BF16梯度复制到FP32主权重
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.master_weights:
                    self.master_weights[name].grad = param.grad.float()

        # 2. 在FP32主权重上执行优化器更新
        optimizer.step()

        # 3. 将更新后的FP32主权重复制回BF16模型权重
        for name, param in self.model.named_parameters():
            if name in self.master_weights:
                param.data.copy_(self.master_weights[name].to(torch.bfloat16))

    def zero_grad(self, set_to_none: bool = True) -> None:
        """清零梯度

        Args:
            set_to_none: 是否将梯度设为None (更省显存)
        """
        self.model.zero_grad(set_to_none=set_to_none)
        # 同时清零主权重的梯度
        if self.master_weights is not None:
            for tensor in self.master_weights.values():
                if tensor.grad is not None:
                    if set_to_none:
                        tensor.grad = None
                    else:
                        tensor.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """获取训练器状态 (用于保存检查点)

        Returns:
            包含模型和主权重状态的字典
        """
        state = {"model": self.model.state_dict()}
        if self.master_weights is not None:
            state["master_weights"] = self.master_weights
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """加载训练器状态 (用于恢复检查点)

        Args:
            state: 状态字典
        """
        self.model.load_state_dict(state["model"])
        if "master_weights" in state:
            self.master_weights = state["master_weights"]
