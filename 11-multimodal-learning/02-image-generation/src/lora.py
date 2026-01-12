"""
LoRA (Low-Rank Adaptation) 和 LyCORIS 实现

支持的方法:
- LoRA: 低秩适应
- LoHA: 低秩 Hadamard 乘积适应
- LoKr: 低秩 Kronecker 乘积适应
- DyLoRA: 动态秩 LoRA
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAType(Enum):
    """LoRA 类型"""
    LORA = "lora"           # 标准 LoRA
    LOHA = "loha"           # Hadamard 乘积
    LOKR = "lokr"           # Kronecker 乘积
    DYLORA = "dylora"       # 动态秩


@dataclass
class LoRAConfig:
    """LoRA 配置"""
    rank: int = 4                           # LoRA 秩
    alpha: float = 1.0                      # 缩放因子
    dropout: float = 0.0                    # Dropout 率
    lora_type: LoRAType = LoRAType.LORA     # LoRA 类型
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    use_bias: bool = False                  # 是否使用偏置
    init_weights: str = "kaiming"           # 权重初始化方法
    
    # DyLoRA 特定参数
    min_rank: int = 1                       # 最小秩
    max_rank: int = 8                       # 最大秩
    
    # LoHA 特定参数
    use_effective_conv2d: bool = False      # 是否使用高效卷积
    
    # LoKr 特定参数
    factor: int = -1                        # Kronecker 分解因子


class LoRALinear(nn.Module):
    """LoRA 线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        use_bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 矩阵
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 返回 LoRA 增量"""
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
    
    def get_merged_weight(self, original_weight: torch.Tensor) -> torch.Tensor:
        """获取合并后的权重"""
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        return original_weight + delta


class LoRAConv2d(nn.Module):
    """LoRA 卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.stride = stride
        self.padding = padding
        
        # LoRA 卷积
        self.lora_A = nn.Conv2d(
            in_channels, rank, self.kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.lora_B = nn.Conv2d(rank, out_channels, (1, 1), bias=False)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 返回 LoRA 增量"""
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoHALinear(nn.Module):
    """LoHA (Low-rank Hadamard Product) 线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 两组 LoRA 矩阵用于 Hadamard 乘积
        self.hada_w1_a = nn.Parameter(torch.empty(rank, in_features))
        self.hada_w1_b = nn.Parameter(torch.empty(out_features, rank))
        self.hada_w2_a = nn.Parameter(torch.empty(rank, in_features))
        self.hada_w2_b = nn.Parameter(torch.empty(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_uniform_(self.hada_w1_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.hada_w2_a, a=math.sqrt(5))
        nn.init.zeros_(self.hada_w1_b)
        nn.init.zeros_(self.hada_w2_b)
        
    def get_delta_weight(self) -> torch.Tensor:
        """计算权重增量"""
        w1 = self.hada_w1_b @ self.hada_w1_a
        w2 = self.hada_w2_b @ self.hada_w2_a
        return (w1 * w2) * self.scaling
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        delta_w = self.get_delta_weight()
        return F.linear(self.dropout(x), delta_w)


class LoKrLinear(nn.Module):
    """LoKr (Low-rank Kronecker Product) 线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        factor: int = -1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 计算 Kronecker 分解的维度
        if factor == -1:
            factor = self._find_factor(in_features, out_features)
        self.factor = factor
        
        in_factor = in_features // factor if in_features % factor == 0 else in_features
        out_factor = out_features // factor if out_features % factor == 0 else out_features
        
        # Kronecker 分解矩阵
        self.lokr_w1 = nn.Parameter(torch.empty(factor, factor))
        self.lokr_w2_a = nn.Parameter(torch.empty(rank, in_factor))
        self.lokr_w2_b = nn.Parameter(torch.empty(out_factor, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
        
    def _find_factor(self, in_features: int, out_features: int) -> int:
        """找到合适的分解因子"""
        for f in [8, 4, 2]:
            if in_features % f == 0 and out_features % f == 0:
                return f
        return 1
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
        nn.init.zeros_(self.lokr_w2_b)
        
    def get_delta_weight(self) -> torch.Tensor:
        """计算权重增量"""
        w2 = self.lokr_w2_b @ self.lokr_w2_a
        delta = torch.kron(self.lokr_w1, w2) * self.scaling
        # 调整到正确的形状
        if delta.shape != (self.out_features, self.in_features):
            delta = delta[:self.out_features, :self.in_features]
        return delta
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        delta_w = self.get_delta_weight()
        return F.linear(self.dropout(x), delta_w)


class DyLoRALinear(nn.Module):
    """DyLoRA (Dynamic LoRA) 线性层 - 支持动态秩"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int = 8,
        min_rank: int = 1,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.alpha = alpha
        self.current_rank = max_rank
        
        # 使用最大秩初始化
        self.lora_A = nn.Linear(in_features, max_rank, bias=False)
        self.lora_B = nn.Linear(max_rank, out_features, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def set_rank(self, rank: int):
        """设置当前使用的秩"""
        self.current_rank = max(self.min_rank, min(rank, self.max_rank))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 使用当前秩"""
        scaling = self.alpha / self.current_rank
        
        # 只使用前 current_rank 个维度
        A_weight = self.lora_A.weight[:self.current_rank, :]
        B_weight = self.lora_B.weight[:, :self.current_rank]
        
        h = F.linear(self.dropout(x), A_weight)
        return F.linear(h, B_weight) * scaling


class LoRAInjectedLinear(nn.Module):
    """注入 LoRA 的线性层包装器"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        lora_type: LoRAType = LoRAType.LORA
    ):
        super().__init__()
        self.original = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.lora_type = lora_type
        self.enabled = True
        
        # 创建对应类型的 LoRA 层
        if lora_type == LoRAType.LORA:
            self.lora = LoRALinear(
                self.in_features, self.out_features, rank, alpha, dropout
            )
        elif lora_type == LoRAType.LOHA:
            self.lora = LoHALinear(
                self.in_features, self.out_features, rank, alpha, dropout
            )
        elif lora_type == LoRAType.LOKR:
            self.lora = LoKrLinear(
                self.in_features, self.out_features, rank, alpha, dropout=dropout
            )
        elif lora_type == LoRAType.DYLORA:
            self.lora = DyLoRALinear(
                self.in_features, self.out_features, max_rank=rank, alpha=alpha, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown LoRA type: {lora_type}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        result = self.original(x)
        if self.enabled:
            result = result + self.lora(x)
        return result
    
    def merge_weights(self):
        """将 LoRA 权重合并到原始层"""
        if self.lora_type == LoRAType.LORA:
            delta = (self.lora.lora_B.weight @ self.lora.lora_A.weight) * self.lora.scaling
        elif self.lora_type in [LoRAType.LOHA, LoRAType.LOKR]:
            delta = self.lora.get_delta_weight()
        else:
            raise NotImplementedError(f"Merge not supported for {self.lora_type}")
            
        self.original.weight.data += delta
        self.enabled = False
        
    def unmerge_weights(self, delta: torch.Tensor):
        """从原始层移除 LoRA 权重"""
        self.original.weight.data -= delta
        self.enabled = True


class LoRAInjectedConv2d(nn.Module):
    """注入 LoRA 的卷积层包装器"""
    
    def __init__(
        self,
        original_layer: nn.Conv2d,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.original = original_layer
        self.enabled = True
        
        self.lora = LoRAConv2d(
            original_layer.in_channels,
            original_layer.out_channels,
            original_layer.kernel_size,
            rank, alpha, dropout,
            original_layer.stride[0],
            original_layer.padding[0]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        result = self.original(x)
        if self.enabled:
            result = result + self.lora(x)
        return result


class LoRAManager:
    """LoRA 管理器 - 用于注入和管理模型中的 LoRA 层"""
    
    def __init__(self, config: LoRAConfig):
        self.config = config
        self.injected_modules: Dict[str, nn.Module] = {}
        
    def inject_lora(self, model: nn.Module) -> nn.Module:
        """向模型注入 LoRA 层"""
        for name, module in model.named_modules():
            if self._should_inject(name, module):
                self._inject_module(model, name, module)
        return model
    
    def _should_inject(self, name: str, module: nn.Module) -> bool:
        """判断是否应该注入 LoRA"""
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            return False
        for target in self.config.target_modules:
            if target in name:
                return True
        return False
    
    def _inject_module(self, model: nn.Module, name: str, module: nn.Module):
        """注入单个模块"""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
            
        if isinstance(module, nn.Linear):
            injected = LoRAInjectedLinear(
                module, self.config.rank, self.config.alpha,
                self.config.dropout, self.config.lora_type
            )
        elif isinstance(module, nn.Conv2d):
            injected = LoRAInjectedConv2d(
                module, self.config.rank, self.config.alpha, self.config.dropout
            )
        else:
            return
            
        setattr(parent, parts[-1], injected)
        self.injected_modules[name] = injected
        
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """获取所有 LoRA 参数"""
        params = []
        for module in self.injected_modules.values():
            params.extend(module.lora.parameters())
        return params
    
    def enable_lora(self):
        """启用所有 LoRA 层"""
        for module in self.injected_modules.values():
            module.enabled = True
            
    def disable_lora(self):
        """禁用所有 LoRA 层"""
        for module in self.injected_modules.values():
            module.enabled = False
            
    def merge_all(self):
        """合并所有 LoRA 权重"""
        for module in self.injected_modules.values():
            if hasattr(module, 'merge_weights'):
                module.merge_weights()
                
    def save_lora_weights(self, path: str):
        """保存 LoRA 权重"""
        state_dict = {}
        for name, module in self.injected_modules.items():
            lora_state = module.lora.state_dict()
            for key, value in lora_state.items():
                state_dict[f"{name}.lora.{key}"] = value
        torch.save(state_dict, path)
        
    def load_lora_weights(self, path: str):
        """加载 LoRA 权重"""
        state_dict = torch.load(path, map_location='cpu')
        for name, module in self.injected_modules.items():
            lora_state = {}
            prefix = f"{name}.lora."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    lora_state[key[len(prefix):]] = value
            if lora_state:
                module.lora.load_state_dict(lora_state)


def create_lora_config(
    rank: int = 4,
    alpha: float = 1.0,
    lora_type: str = "lora",
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0
) -> LoRAConfig:
    """创建 LoRA 配置的工厂函数"""
    type_map = {
        "lora": LoRAType.LORA,
        "loha": LoRAType.LOHA,
        "lokr": LoRAType.LOKR,
        "dylora": LoRAType.DYLORA
    }
    
    if lora_type not in type_map:
        raise ValueError(f"Unknown LoRA type: {lora_type}")
        
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
        
    return LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        lora_type=type_map[lora_type],
        target_modules=target_modules
    )


def inject_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    lora_type: str = "lora",
    target_modules: Optional[List[str]] = None
) -> Tuple[nn.Module, LoRAManager]:
    """便捷函数：向模型注入 LoRA"""
    config = create_lora_config(rank, alpha, lora_type, target_modules)
    manager = LoRAManager(config)
    model = manager.inject_lora(model)
    return model, manager
