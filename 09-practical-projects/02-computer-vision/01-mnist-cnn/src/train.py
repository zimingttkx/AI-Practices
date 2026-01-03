"""
MNIST模型训练脚本

提供命令行接口用于训练MNIST手写数字识别模型。
支持多种模型架构和超参数配置。

使用方法:
    python train.py --model simple_cnn --epochs 20 --batch-size 128

    可用参数:
        --model: 模型类型 (simple_cnn, improved_cnn, deep_cnn)
        --epochs: 训练轮数
        --batch-size: 批次大小
        --seed: 随机种子

功能特性:
    - 自动数据加载和预处理
    - 模型训练和验证
    - 早停和学习率调度
    - 模型保存和性能评估
    - 完整的训练日志

日期: 2024-01
"""

import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.common import set_seed
from data import prepare_data
from model import MNISTPredictor, get_callbacks


def train_model(model_type='simple_cnn', epochs=20, batch_size=128, random_state=42):
    """
    训练MNIST模型

    Args:
        model_type: 模型类型
        epochs: 训练轮数
        batch_size: 批大小
        random_state: 随机种子
    """
    print("=" * 60)
    print("MNIST手写数字识别 - 模型训练")
    print("=" * 60)

    set_seed(random_state)

    # 准备数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data()

    # 创建预测器
    predictor = MNISTPredictor(model_type=model_type, random_state=random_state)

    # 设置回调函数
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{model_type}_best.h5'

    callbacks = get_callbacks(model_path, patience=5)

    # 训练模型
    history = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # 评估模型
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    train_metrics = predictor.evaluate(X_train, y_train)
    val_metrics = predictor.evaluate(X_val, y_val)
    test_metrics = predictor.evaluate(X_test, y_test)

    print("\n训练集性能:")
    for metric, value in train_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    print("\n验证集性能:")
    for metric, value in val_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    print("\n测试集性能:")
    for metric, value in test_metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    # 保存最终模型
    final_model_path = model_dir / f'{model_type}_final.h5'
    predictor.save_model(final_model_path)

    # 保存为best_model
    best_model_path = model_dir / 'best_model.h5'
    predictor.save_model(best_model_path)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳模型: {model_path}")
    print(f"最终模型: {final_model_path}")
    print(f"测试准确率: {test_metrics['accuracy']:.4f}")

    return predictor, history


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MNIST手写数字识别模型训练')

    parser.add_argument(
        '--model',
        type=str,
        default='simple_cnn',
        choices=['simple_cnn', 'improved_cnn', 'deep_cnn'],
        help='模型类型'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='训练轮数'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='批大小'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    args = parser.parse_args()

    # 训练模型
    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.seed
    )


if __name__ == '__main__':
    main()
