"""
MNIST数据加载和预处理模块

本模块提供MNIST手写数字数据集的加载、预处理和增强功能。
主要功能包括:
    - 加载原始MNIST数据集
    - 数据归一化和格式转换
    - 训练集、验证集、测试集划分
    - 数据增强策略
    - 数据分布统计

日期: 2024-01
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


def load_mnist_data(normalize=True):
    """
    从Keras数据集加载MNIST手写数字数据

    MNIST数据集包含60000张训练图像和10000张测试图像，
    每张图像为28x28像素的灰度图，表示0-9的手写数字。

    Args:
        normalize (bool): 是否将像素值从[0,255]归一化到[0,1]
                         归一化有助于神经网络训练的稳定性和收敛速度

    Returns:
        tuple: ((X_train, y_train), (X_test, y_test))
            - X_train: (60000, 28, 28, 1) 训练图像，添加了通道维度
            - y_train: (60000,) 训练标签，取值0-9
            - X_test: (10000, 28, 28, 1) 测试图像
            - y_test: (10000,) 测试标签

    注意:
        - 原始数据为(28,28)形状，本函数会自动添加通道维度变为(28,28,1)
        - 这是为了兼容CNN的输入格式要求(height, width, channels)
    """
    print("正在加载MNIST数据集...")

    # 从Keras加载数据
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    print(f"✓ 训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"✓ 测试集: {X_test.shape}, 标签: {y_test.shape}")

    # 添加通道维度，从(N, 28, 28)变为(N, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # 归一化到[0, 1]区间
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        print("✓ 数据已归一化到[0, 1]")

    return (X_train, y_train), (X_test, y_test)


def prepare_data(test_size=0.1, random_state=42):
    """
    准备训练、验证和测试数据集

    本函数完成以下操作:
    1. 加载原始MNIST数据
    2. 将原训练集划分为新训练集和验证集
    3. 使用分层采样确保各类别比例一致

    Args:
        test_size (float): 验证集占原训练集的比例，默认0.1(10%)
        random_state (int): 随机种子，确保可重复性

    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
            - X_train: (54000, 28, 28, 1) 训练集图像
            - y_train: (54000,) 训练集标签
            - X_val: (6000, 28, 28, 1) 验证集图像
            - y_val: (6000,) 验证集标签
            - X_test: (10000, 28, 28, 1) 测试集图像
            - y_test: (10000,) 测试集标签

    注意:
        验证集用于模型选择和超参数调优，测试集仅用于最终评估
    """
    set_seed(random_state)

    # 加载数据
    (X_train, y_train), (X_test, y_test) = load_mnist_data()

    # 划分训练集和验证集，使用分层采样保持类别分布
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train  # 分层采样，确保各类别比例一致
    )

    print(f"\n数据划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  测试集: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_data_augmentation():
    """
    创建数据增强层用于训练时的数据扩充

    数据增强通过对训练图像进行随机变换来扩充训练集，
    提高模型的泛化能力和鲁棒性。

    应用的增强策略:
        - 随机旋转: ±10%角度范围内随机旋转
        - 随机平移: 在高度和宽度方向±10%范围内随机平移
        - 随机缩放: ±10%缩放比例

    Returns:
        keras.Sequential: 包含数据增强层的序列模型
                         可直接嵌入到模型中或在训练时使用

    使用示例:
        >>> aug = create_data_augmentation()
        >>> augmented_images = aug(images, training=True)

    注意:
        - 仅在training=True时生效，推理时不进行增强
        - 增强范围保守，避免破坏数字的可识别性
    """
    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.1),      # ±10%角度旋转
        keras.layers.RandomTranslation(0.1, 0.1),  # ±10%位置平移
        keras.layers.RandomZoom(0.1),          # ±10%缩放
    ], name='data_augmentation')

    return data_augmentation


def get_class_distribution(y):
    """
    统计并打印数据集的类别分布情况

    用于检查数据集是否平衡，各类别样本数是否合理。

    Args:
        y (np.ndarray): 标签数组，形状为(n_samples,)

    Returns:
        dict: 类别分布字典，键为类别标签，值为该类别的样本数

    输出格式:
        类别分布:
          0:  5923 ( 9.87%)
          1:  6742 (11.24%)
          ...
    """
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))

    print("\n类别分布:")
    total = len(y)
    for label, count in sorted(distribution.items()):
        percentage = count / total * 100
        print(f"  {label}: {count:5d} ({percentage:5.2f}%)")

    return distribution


def print_data_info(X, y, name='数据集'):
    """
    打印数据集的详细统计信息

    包括数据形状、类型、值域、类别数量及分布等信息，
    用于数据探索和质量检查。

    Args:
        X (np.ndarray): 特征数据，通常为图像数组
        y (np.ndarray): 标签数据
        name (str): 数据集名称，用于显示标题

    输出信息包括:
        - 数据形状和类型
        - 像素值范围
        - 标签形状和类别数
        - 各类别的样本分布
    """
    print(f"\n{'=' * 60}")
    print(f"{name}信息")
    print(f"{'=' * 60}")
    print(f"形状: {X.shape}")
    print(f"数据类型: {X.dtype}")
    print(f"值范围: [{X.min():.3f}, {X.max():.3f}]")
    print(f"标签形状: {y.shape}")
    print(f"类别数: {len(np.unique(y))}")

    get_class_distribution(y)


if __name__ == '__main__':
    print("=" * 60)
    print("MNIST数据加载测试")
    print("=" * 60)

    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    # 打印信息
    print_data_info(X_train, y_train, '训练集')
    print_data_info(X_val, y_val, '验证集')
    print_data_info(X_test, y_test, '测试集')

    print("\n✓ 数据加载测试完成！")
