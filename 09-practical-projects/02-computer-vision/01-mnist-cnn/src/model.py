"""
MNIST卷积神经网络模型定义

本模块实现了多种CNN架构用于MNIST手写数字识别:
    - Simple CNN: 基础两层卷积网络
    - Improved CNN: 使用批标准化的改进网络
    - Deep CNN: 更深层的卷积网络架构

主要类:
    MNISTPredictor: 封装模型创建、训练、预测和评估功能

设计特点:
    1. 模块化设计，支持多种模型架构
    2. 内置训练和评估功能
    3. 支持模型保存和加载
    4. 提供预测概率接口

日期: 2024-01
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed


class MNISTPredictor:
    """
    MNIST手写数字识别预测器

    提供完整的模型生命周期管理，包括模型创建、训练、预测和评估。
    支持多种CNN架构，并封装了常用的模型操作接口。

    属性:
        model_type (str): 模型架构类型
        random_state (int): 随机种子
        model (keras.Model): Keras模型实例
        history (History): 训练历史记录

    支持的模型类型:
        - 'simple_cnn': 基础两层卷积网络，适合快速原型
        - 'improved_cnn': 带批标准化的改进网络，性能更优
        - 'deep_cnn': 更深的网络架构，使用全局平均池化

    使用示例:
        >>> predictor = MNISTPredictor(model_type='simple_cnn')
        >>> predictor.train(X_train, y_train, X_val, y_val, epochs=20)
        >>> metrics = predictor.evaluate(X_test, y_test)
        >>> predictions = predictor.predict(X_new)
    """

    def __init__(self, model_type='simple_cnn', random_state=42):
        """
        初始化预测器

        Args:
            model_type (str): 模型类型，可选'simple_cnn', 'improved_cnn', 'deep_cnn'
            random_state (int): 随机种子，用于结果可重复性
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.history = None

        set_seed(random_state)

    def create_simple_cnn(self, input_shape=(28, 28, 1), num_classes=10):
        """
        创建简单的CNN模型

        网络结构:
            Conv2D(32) -> MaxPooling -> Conv2D(64) -> MaxPooling
            -> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(10)

        特点:
            - 两个卷积块，特征图数量递增(32->64)
            - 使用ReLU激活函数
            - Dropout防止过拟合
            - 适合快速训练和原型验证

        Args:
            input_shape (tuple): 输入图像形状，默认(28, 28, 1)
            num_classes (int): 分类类别数，默认10

        Returns:
            keras.Model: 编译前的Keras模型实例

        模型参数量: ~225K
        """
        model = models.Sequential([
            # 第一个卷积块: 提取低层次特征(边缘、纹理)
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),  # 降采样，减少计算量

            # 第二个卷积块: 提取高层次特征(形状、结构)
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # 全连接层: 分类决策
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),  # 50% dropout防止过拟合
            layers.Dense(num_classes, activation='softmax')  # 10类输出
        ], name='simple_cnn')

        return model

    def create_improved_cnn(self, input_shape=(28, 28, 1), num_classes=10):
        """
        创建改进的CNN模型（使用批标准化）

        网络结构:
            [Conv2D(32)->BN->ReLU] x2 -> MaxPooling -> Dropout
            [Conv2D(64)->BN->ReLU] x2 -> MaxPooling -> Dropout
            -> Flatten -> Dense(256)->BN->ReLU -> Dropout -> Dense(10)

        改进点:
            - 批标准化(Batch Normalization): 加速训练，提高稳定性
            - 双卷积结构: 每个池化前使用两次卷积，增强特征提取
            - padding='same': 保持特征图尺寸，避免信息损失
            - 更大的全连接层(256): 增强分类能力

        Args:
            input_shape (tuple): 输入图像形状，默认(28, 28, 1)
            num_classes (int): 分类类别数，默认10

        Returns:
            keras.Model: 编译前的Keras模型实例

        模型参数量: ~280K
        预期性能: 测试准确率 > 99%
        """
        model = models.Sequential([
            # 第一个卷积块: 特征提取 + 批标准化
            layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # 第二个卷积块: 更深层次特征
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # 全连接分类层
            layers.Flatten(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ], name='improved_cnn')

        return model

    def create_deep_cnn(self, input_shape=(28, 28, 1), num_classes=10):
        """
        创建深度CNN模型

        网络结构:
            Conv2D(32)->BN->ReLU -> MaxPooling
            Conv2D(64)->BN->ReLU -> MaxPooling
            Conv2D(128)->BN->ReLU -> GlobalAvgPooling
            -> Dense(256) -> Dropout -> Dense(10)

        特点:
            - 三层卷积，逐层加深特征图数量(32->64->128)
            - 使用全局平均池化(GAP)替代Flatten，减少参数量
            - 更深的网络，更强的特征提取能力
            - 适合追求更高精度的场景

        Args:
            input_shape (tuple): 输入图像形状，默认(28, 28, 1)
            num_classes (int): 分类类别数，默认10

        Returns:
            keras.Model: 编译前的Keras模型实例

        模型参数量: ~320K
        全局平均池化优势: 减少参数，避免过拟合，提高模型泛化能力
        """
        model = models.Sequential([
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),

            # 第二个卷积块
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),

            # 第三个卷积块
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            # 全局平均池化，将每个特征图平均为一个值
            layers.GlobalAveragePooling2D(),

            # 全连接分类层
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ], name='deep_cnn')

        return model

    def create_model(self, input_shape=(28, 28, 1), num_classes=10):
        """
        根据模型类型创建相应的CNN模型

        这是一个工厂方法，根据初始化时指定的model_type参数，
        创建对应的模型架构。

        Args:
            input_shape (tuple): 输入图像形状
            num_classes (int): 分类类别数

        Returns:
            keras.Model: 未编译的Keras模型实例

        Raises:
            ValueError: 当model_type不在支持的类型列表中时抛出

        支持的模型类型:
            - 'simple_cnn': 基础CNN，快速训练
            - 'improved_cnn': 改进CNN，性能更优
            - 'deep_cnn': 深度CNN，追求高精度
        """
        if self.model_type == 'simple_cnn':
            model = self.create_simple_cnn(input_shape, num_classes)
        elif self.model_type == 'improved_cnn':
            model = self.create_improved_cnn(input_shape, num_classes)
        elif self.model_type == 'deep_cnn':
            model = self.create_deep_cnn(input_shape, num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        return model

    def compile_model(self, model, learning_rate=0.001):
        """
        编译模型，配置优化器、损失函数和评估指标

        使用Adam优化器，这是一种自适应学习率的优化算法，
        结合了动量和RMSprop的优点，适合大多数深度学习任务。

        Args:
            model (keras.Model): 待编译的Keras模型
            learning_rate (float): 初始学习率，默认0.001

        Returns:
            keras.Model: 编译后的模型

        配置说明:
            - 优化器: Adam (Adaptive Moment Estimation)
            - 损失函数: sparse_categorical_crossentropy
              (适用于整数标签的多分类交叉熵)
            - 评估指标: accuracy (准确率)
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=20, batch_size=128, callbacks=None):
        """
        训练模型

        完整的模型训练流程，包括模型创建、编译和训练。
        支持验证集监控和自定义回调函数。

        Args:
            X_train (np.ndarray): 训练数据，形状(n_samples, height, width, channels)
            y_train (np.ndarray): 训练标签，形状(n_samples,)
            X_val (np.ndarray, optional): 验证数据
            y_val (np.ndarray, optional): 验证标签
            epochs (int): 训练轮数，默认20
            batch_size (int): 批次大小，默认128
            callbacks (list, optional): Keras回调函数列表

        Returns:
            History: Keras训练历史对象，包含每个epoch的损失和指标

        训练过程:
            1. 创建模型架构
            2. 编译模型(配置优化器、损失函数)
            3. 打印模型结构摘要
            4. 执行训练，记录训练历史
        """
        print(f"\n开始训练模型: {self.model_type}")
        print(f"训练样本数: {len(X_train)}")
        print(f"输入形状: {X_train.shape[1:]}")

        # 创建模型
        self.model = self.create_model(input_shape=X_train.shape[1:])
        self.model = self.compile_model(self.model)

        # 打印模型结构
        print("\n模型结构:")
        self.model.summary()

        # 验证数据
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✓ 模型训练完成")

        return self.history

    def predict(self, X):
        """
        对输入数据进行预测

        返回预测的类别标签(0-9)，而非概率分布。

        Args:
            X (np.ndarray): 输入数据，支持多种形状:
                - (28, 28): 单张图像
                - (n, 28, 28): 批量图像
                - (n, 28, 28, 1): 带通道维度的批量图像

        Returns:
            np.ndarray: 预测的类别标签，形状(n_samples,)

        Raises:
            ValueError: 如果模型未训练

        注意:
            输入数据会自动调整为(n, 28, 28, 1)格式
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        # 确保输入形状正确
        if len(X.shape) == 2:
            X = X.reshape(-1, 28, 28, 1)
        elif len(X.shape) == 3:
            X = X.reshape(-1, 28, 28, 1)

        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        """
        预测每个类别的概率分布

        与predict()不同，本方法返回完整的概率分布，
        可用于分析模型的置信度。

        Args:
            X (np.ndarray): 输入数据

        Returns:
            np.ndarray: 预测概率，形状(n_samples, 10)
                       每行为一个样本的10个类别概率

        Raises:
            ValueError: 如果模型未训练

        使用示例:
            >>> proba = predictor.predict_proba(X_test[:5])
            >>> print(proba[0])  # 第一个样本的概率分布
            [0.01 0.02 0.85 0.03 ...]  # 类别2的概率最高
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        # 确保输入形状正确
        if len(X.shape) == 2:
            X = X.reshape(-1, 28, 28, 1)
        elif len(X.shape) == 3:
            X = X.reshape(-1, 28, 28, 1)

        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        评估模型在给定数据集上的性能

        Args:
            X (np.ndarray): 测试数据
            y (np.ndarray): 测试标签

        Returns:
            dict: 包含损失和准确率的字典
                {'loss': float, 'accuracy': float}

        Raises:
            ValueError: 如果模型未训练
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()方法")

        loss, accuracy = self.model.evaluate(X, y, verbose=0)

        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def save_model(self, filepath):
        """
        保存训练好的模型到文件

        Args:
            filepath (str or Path): 保存路径，建议使用.h5或.keras扩展名

        Raises:
            ValueError: 如果模型未训练

        注意:
            保存的模型包含完整的架构、权重和优化器状态，
            可以直接用于推理或继续训练。
        """
        if self.model is None:
            raise ValueError("模型未训练，无法保存")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(filepath)
        print(f"✓ 模型已保存: {filepath}")

    def load_model(self, filepath):
        """
        从文件加载训练好的模型

        Args:
            filepath (str or Path): 模型文件路径

        Raises:
            FileNotFoundError: 如果模型文件不存在

        使用示例:
            >>> predictor = MNISTPredictor()
            >>> predictor.load_model('models/best_model.h5')
            >>> predictions = predictor.predict(X_test)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        self.model = keras.models.load_model(filepath)
        print(f"✓ 模型已加载: {filepath}")


def get_callbacks(model_path, patience=5):
    """
    获取训练回调函数列表

    配置三个关键回调函数，用于优化训练过程和保存模型：
    1. EarlyStopping: 验证集性能不再提升时提前停止训练
    2. ModelCheckpoint: 自动保存验证集上表现最好的模型
    3. ReduceLROnPlateau: 验证损失停滞时自动降低学习率

    Args:
        model_path (str or Path): 模型保存路径
        patience (int): 早停的耐心值，默认5
                       表示验证集性能连续5个epoch未提升时停止训练

    Returns:
        list: Keras回调函数列表

    回调函数说明:
        - EarlyStopping: 监控val_loss，patience=5，恢复最佳权重
        - ModelCheckpoint: 监控val_accuracy，只保存最佳模型
        - ReduceLROnPlateau: 监控val_loss，patience=3，学习率减半

    使用示例:
        >>> callbacks = get_callbacks('models/best.h5', patience=5)
        >>> model.fit(X, y, callbacks=callbacks)
    """
    callbacks = [
        # 早停: 防止过拟合，节省训练时间
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        # 模型检查点: 保存验证集上表现最好的模型
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),

        # 学习率调度: 在平台期降低学习率
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,            # 学习率减半
            patience=3,            # 3个epoch未改善则降低学习率
            min_lr=1e-7,          # 最小学习率
            verbose=1
        )
    ]

    return callbacks


if __name__ == '__main__':
    from data import load_mnist_data

    print("=" * 60)
    print("MNIST模型测试")
    print("=" * 60)

    # 加载数据
    (X_train, y_train), (X_test, y_test) = load_mnist_data()

    # 使用小样本测试
    X_train_small = X_train[:1000]
    y_train_small = y_train[:1000]
    X_test_small = X_test[:200]
    y_test_small = y_test[:200]

    # 创建并训练模型
    predictor = MNISTPredictor(model_type='simple_cnn')
    predictor.train(
        X_train_small, y_train_small,
        X_test_small, y_test_small,
        epochs=3,
        batch_size=32
    )

    # 评估
    metrics = predictor.evaluate(X_test_small, y_test_small)
    print(f"\n测试集性能:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\n✓ 模型测试完成！")
