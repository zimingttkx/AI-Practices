# 08-efficient-architectures 高效架构

本模块实现了 2024-2026 年最前沿的高效 LLM 架构。

## 模块结构

```
08-efficient-architectures/
├── src/
│   ├── __init__.py
│   └── mamba.py          # Mamba 状态空间模型 (~1000行)
├── tests/
│   ├── __init__.py
│   └── test_mamba.py     # 单元测试 (35 tests)
├── notebooks/
│   └── 01_Mamba_tutorial.ipynb
├── 知识点.md
└── README.md
```

## 已实现功能

### Mamba (选择性状态空间模型)

| 组件 | 描述 |
|:-----|:-----|
| `MambaConfig` | 模型配置 |
| `SelectiveSSM` | 选择性 SSM 核心模块 |
| `MambaBlock` | Mamba 块 (残差 + 归一化) |
| `MambaModel` | 基础模型 |
| `MambaForCausalLM` | 因果语言模型 |
| `Mamba2Block` | Mamba-2 SSD 块 |

### 核心算法

- `discretize_ssm`: SSM 离散化 (ZOH/Bilinear/Euler)
- `selective_scan`: 选择性扫描算法
- `causal_conv1d`: 因果一维卷积

## 快速开始

```python
from src.mamba import create_mamba_model, MambaConfig

# 使用预设配置
model = create_mamba_model("small")  # 130M 参数

# 自定义配置
config = MambaConfig(
    d_model=768,
    n_layers=24,
    d_state=16,
    vocab_size=32000,
)
model = MambaForCausalLM(config)

# 生成
input_ids = np.array([[1, 2, 3, 4, 5]])
generated = model.generate(input_ids, max_new_tokens=50)
```

## 测试

```bash
pytest tests/test_mamba.py -v
```

## 参考文献

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
2. Dao, T., et al. (2024). Transformers are SSMs.
