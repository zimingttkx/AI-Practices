# 02-Neural Networks | 神经网络基础

> Keras/TensorFlow 深度学习入门与进阶（含 PyTorch 等价实现）

---

## 前置知识

- ✅ 必需：Python 基础、NumPy、scikit-learn 基础
- 📖 推荐：线性代数、微积分（偏导数、链式法则）
- 🔗 前序模块：[01-foundations](../01-foundations)

---

## 目录结构

```
02-neural-networks/
├── 01-keras-introduction/       # Keras 入门：MLP、回调、TensorBoard
├── 02-training-deep-networks/   # 训练技巧：初始化、BN、Dropout、正则化
├── 03-custom-models-training/   # 自定义模型：Layer、Loss、训练循环
└── 04-data-loading-preprocessing/ # 数据加载：tf.data、TFRecord、预处理
```

> 💡 **PyTorch 版本**：部分 notebook 提供 `_pytorch` 后缀的等价实现，
> 如 `01-顺序API构建回归MLP_pytorch.ipynb`，可与 TF 版本对照学习。

---

## 学习路线

```
Keras 基础 → 训练技巧 → 自定义模型 → 数据管道
```

---

## 核心内容

| 子模块 | 核心概念 | 实践重点 | PyTorch 版本 |
|--------|----------|----------|-------------|
| Keras 入门 | Sequential/Functional API、回调 | MLP 分类、TensorBoard | ✅ 已提供 |
| 训练技巧 | Xavier/He 初始化、BN、Dropout | 梯度消失/爆炸处理 | 🔄 进行中 |
| 自定义模型 | 自定义 Layer/Loss/Metric | 训练循环实现 | 🔄 进行中 |
| 数据管道 | tf.data、混合精度、数据增强 | 大规模数据处理 | 📋 计划中 |

---

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| TensorFlow 与 PyTorch 结果不同 | 正常差异，关键看趋势一致 |
| CUDA out of memory | 减小 batch_size，或使用 CPU |
| `torch.load` 权重不匹配 | 确保 model class 定义与保存时一致 |
| Fashion-MNIST 下载失败 | 检查网络连接，或手动下载数据集 |

---

[返回主页](../README.md)
