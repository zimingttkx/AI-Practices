# 常见问题与排错指南 / Troubleshooting Guide

## 环境配置问题

### CUDA 不可用 / CUDA not available

**症状**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
1. 检查 NVIDIA 驱动是否安装：`nvidia-smi`
2. 安装对应 CUDA 版本的 PyTorch：
   ```bash
   # CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. 如果没有 GPU，可以使用 CPU 版本（notebook 会自动检测设备）

### TensorFlow 与 PyTorch 冲突

**症状**: `import tensorflow` 或 `import torch` 报错

**解决方案**:
1. 推荐使用 conda 管理独立环境：
   ```bash
   conda env create -f environment.yml
   conda activate ai-practices
   ```
2. 或使用 Docker：
   ```bash
   docker compose up jupyter
   ```

### 包版本不兼容

**症状**: `ImportError` 或 `AttributeError`

**解决方案**:
1. 确认 Python 版本 ≥ 3.9
2. 按 `requirements.txt` 安装指定版本：
   ```bash
   pip install -r requirements.txt
   ```
3. 如果仍有冲突，逐个安装核心包：
   ```bash
   pip install numpy pandas scikit-learn torch torchvision
   ```

---

## 运行时错误

### CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
1. 减小 `batch_size`（如从 64 减至 32 或 16）
2. 使用 CPU 运行：在代码中将 `device` 设为 `'cpu'`
3. 清空 CUDA 缓存：`torch.cuda.empty_cache()`

### 数据集下载失败

**症状**: `ConnectionError` 或下载超时

**解决方案**:
1. 检查网络连接
2. Fashion-MNIST 手动下载：
   ```python
   from torchvision.datasets import FashionMNIST
   dataset = FashionMNIST(root='./data', download=True)
   ```
3. 如果持续失败，设置镜像源或代理

### 模型权重加载失败

**症状**: `RuntimeError: Error(s) in loading state_dict`

**解决方案**:
1. 确保模型类定义与保存时完全一致
2. 使用 `strict=False` 参数忽略不匹配的键：
   ```python
   model.load_state_dict(torch.load('model.pth'), strict=False)
   ```
3. 检查 PyTorch 版本是否兼容

### 训练损失不下降

**可能原因与解决方案**:

| 原因 | 解决方案 |
|------|---------|
| 学习率过大 | 降低到 1e-3 或 1e-4 |
| 学习率过小 | 增大到 1e-2 |
| 梯度消失 | 使用 BatchNorm、残差连接 |
| 数据未标准化 | 使用 `StandardScaler` 标准化特征 |
| 梯度爆炸 | 使用梯度裁剪 `torch.nn.utils.clip_grad_norm_` |

---

## Notebook 特定问题

### TF 和 PyTorch 结果不同

这是正常现象，原因包括：
- 不同框架的默认初始化方式不同
- 浮点运算顺序差异
- Dropout 的随机性

**关注训练趋势**而非精确数值，两者应表现出相似的学习曲线走势。

### Jupyter Kernel 崩溃

**解决方案**:
1. 重启 Kernel：`Kernel → Restart`
2. 减少内存占用：释放不需要的变量 `del model; torch.cuda.empty_cache()`
3. 使用 `nbdime` 对比 notebook 差异

### Matplotlib 中文显示乱码

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

---

## 获取帮助

- 📖 查看 [08-theory-notes](../08-theory-notes) 的快速参考卡
- 💬 在 [GitHub Discussions](https://github.com/zimingttkx/AI-Practices/discussions) 提问
- 🐛 报告 Bug：[GitHub Issues](https://github.com/zimingttkx/AI-Practices/issues)
