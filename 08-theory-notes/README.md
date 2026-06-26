# 08-Theory Notes | 理论笔记

> 深度学习理论的高密度速查表库 - 激活函数、损失函数、架构选型、超参数调优、问题诊断

---

## 前置知识

- ✅ 必需：Python 基础、正在学习 01-06 模块中的任一主题
- 📖 推荐：线性代数、概率统计、微积分
- 🔗 前序模块：[01-foundations](../01-foundations)（可随时参考）

## 📚 目录结构

```
08-theory-notes/
├── 📌 QUICK-REFERENCE.md                    # ⭐ 激活函数与损失函数速查表
├── 📌 ARCHITECTURE-HYPERPARAMETER-TUNING.md # ⭐ 架构选型与超参数调优指南
├── 📌 MODEL-SELECTION-TROUBLESHOOTING.md    # ⭐ 模型选择与问题诊断
│
├── activation-functions/
│   └── activation-functions-complete.md     # 30+ 激活函数详解
├── loss-functions/
│   └── loss-functions-complete.md           # 分类/回归/排序 Loss 详解
└── architectures/
    ├── 机器学习的通用流程.md
    ├── 什么数据用什么网络结构.md
    ├── 人工智能各种方法类别.md
    ├── 如何看待深度学习.md
    ├── 卷积神经网络.ipynb
    ├── 循环神经网络.ipynb
    └── 密集层连接网络.ipynb
```

---

## 🎯 快速导航

### 我需要快速查阅什么？

| 需求 | 推荐文档 | 内容 |
|------|---------|------|
| **激活函数选择** | `QUICK-REFERENCE.md` | 一句话选择指南、对比矩阵、详细速查 |
| **损失函数选择** | `QUICK-REFERENCE.md` | 任务类型→损失函数、对比矩阵、详细速查 |
| **网络架构选择** | `ARCHITECTURE-HYPERPARAMETER-TUNING.md` | 数据类型→架构、架构对比、选择决策树 |
| **超参数调优** | `ARCHITECTURE-HYPERPARAMETER-TUNING.md` | 学习率、Batch Size、正则化、训练技巧 |
| **问题诊断** | `MODEL-SELECTION-TROUBLESHOOTING.md` | 问题诊断决策树、常见错误、解决方案 |
| **深度学习详解** | `activation-functions-complete.md` | 30+ 激活函数的完整理论与实现 |
| **损失函数详解** | `loss-functions-complete.md` | 分类/回归/排序损失的完整理论 |

---

## 📊 核心内容速览

### 🚀 三个快速参考卡

#### 1. QUICK-REFERENCE.md (激活函数与损失函数)
```
✅ 激活函数速查表
   - 一句话选择指南
   - 激活函数对比矩阵
   - 8个常用激活函数详解
   - 快速决策树

✅ 损失函数速查表
   - 一句话选择指南
   - 损失函数对比矩阵
   - 8个常用损失函数详解
   - 快速决策树

✅ 架构选型指南
   - 数据类型→网络架构
   - 任务类型→损失函数→激活函数
   - 超参数速查
```

#### 2. ARCHITECTURE-HYPERPARAMETER-TUNING.md (架构与超参数)
```
✅ 网络架构速查
   - 架构选择决策树
   - 常见架构对比矩阵
   - 5个常见架构详解

✅ 超参数调优指南
   - 调优优先级
   - 学习率调优（范围、衰减策略）
   - Batch Size调优
   - 正则化调优

✅ 训练技巧速查
   - 训练加速技巧
   - 内存优化技巧
   - 收敛性诊断
   - 性能优化清单
```

#### 3. MODEL-SELECTION-TROUBLESHOOTING.md (模型选择与诊断)
```
✅ 模型选择速查
   - 按任务类型选择模型
   - 图像分类/检测/分割
   - 文本分类/序列标注/翻译
   - 时间序列预测

✅ 问题诊断决策树
   - 训练问题诊断
   - 推理问题诊断
   - 常见错误与解决方案

✅ 性能瓶颈分析
   - 性能瓶颈诊断矩阵
   - 训练/推理速度优化
   - 快速决策表
```

---

## 💡 使用建议

### 学习路径

**初学者**：
1. 阅读 `QUICK-REFERENCE.md` 了解基础概念
2. 查看 `ARCHITECTURE-HYPERPARAMETER-TUNING.md` 学习架构选择
3. 参考 `activation-functions-complete.md` 深入理解激活函数

**实践者**：
1. 使用 `QUICK-REFERENCE.md` 快速查阅激活函数和损失函数
2. 参考 `ARCHITECTURE-HYPERPARAMETER-TUNING.md` 调优超参数
3. 使用 `MODEL-SELECTION-TROUBLESHOOTING.md` 诊断问题

**研究者**：
1. 深入阅读 `activation-functions-complete.md` 和 `loss-functions-complete.md`
2. 参考 `architectures/` 中的详细笔记和Jupyter笔记本
3. 结合快速参考卡进行快速查阅

### 快速查阅技巧

- **需要快速答案？** → 查看对应快速参考卡的"一句话选择指南"
- **需要详细对比？** → 查看对应快速参考卡的"对比矩阵"
- **需要深入理解？** → 查看 `activation-functions-complete.md` 或 `loss-functions-complete.md`
- **遇到问题？** → 查看 `MODEL-SELECTION-TROUBLESHOOTING.md` 的诊断决策树
- **需要代码示例？** → 查看快速参考卡末尾的"快速参考代码"

---

## 📈 内容统计

| 文档 | 行数 | 内容量 | 难度 |
|------|------|--------|------|
| QUICK-REFERENCE.md | 400+ | 高密度 | ⭐⭐ |
| ARCHITECTURE-HYPERPARAMETER-TUNING.md | 500+ | 高密度 | ⭐⭐ |
| MODEL-SELECTION-TROUBLESHOOTING.md | 450+ | 高密度 | ⭐⭐ |
| activation-functions-complete.md | 1000+ | 详细 | ⭐⭐⭐ |
| loss-functions-complete.md | 1000+ | 详细 | ⭐⭐⭐ |
| architectures/ | 1000+ | 详细 | ⭐⭐⭐ |

**总计**：4000+ 行高质量理论笔记

---

## 🎓 核心特色

### ✨ 三层递进式学习

1. **快速参考卡**（5分钟）
   - 一句话选择指南
   - 对比矩阵
   - 快速决策树

2. **详细指南**（30分钟）
   - 详细速查表
   - 完整解释
   - 代码示例

3. **深度理论**（2小时）
   - 完整的数学推导
   - 历史背景
   - 前沿研究

### 🎯 高密度知识蒸馏

- **无废话**：每一行都有信息增量
- **易查阅**：表格、决策树、快速指南
- **有代码**：PyTorch代码示例
- **有对比**：矩阵对比、优缺点分析

---

## 🔗 相关资源

### 完整的AI-Practices学习体系

```
AI-Practices/
├── 01-基础数学
├── 02-机器学习基础
├── 03-深度学习基础
├── 04-计算机视觉
├── 05-自然语言处理
├── 06-推荐系统
├── 07-强化学习
│   ├── 01-MDP基础 ✅ (极智重构)
│   ├── 02-Q-Learning ✅ (极智重构)
│   ├── 03-深度Q学习 ✅ (完整实现)
│   └── 04-策略梯度 ✅ (极智重构)
└── 08-理论笔记 ← 你在这里
    ├── QUICK-REFERENCE.md ✅ (新增)
    ├── ARCHITECTURE-HYPERPARAMETER-TUNING.md ✅ (新增)
    └── MODEL-SELECTION-TROUBLESHOOTING.md ✅ (新增)
```

---

## 📝 最后更新

- **版本**: 1.0
- **更新时间**: 2024年12月
- **新增内容**: 3个高密度快速参考卡
- **总行数**: 4000+ 行

---

[返回主页](../README.md)
