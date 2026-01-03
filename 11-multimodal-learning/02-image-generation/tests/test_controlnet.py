"""
ControlNet 模型单元测试

测试覆盖:
    - ControlNetConfig 配置验证
    - ZeroConv 零卷积
    - ControlNetConditioningEmbedding 条件编码
    - ControlNetBlock 控制块
    - ControlNet 完整模型
"""

import unittest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from controlnet import (
    ControlNetConfig,
    ZeroConv,
    ControlNetConditioningEmbedding,
    ControlNetBlock,
    ControlNet,
    create_controlnet,
)


class TestControlNetConfig(unittest.TestCase):
    """测试 ControlNet 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = ControlNetConfig()
        self.assertEqual(config.image_size, 512)
        self.assertEqual(config.conditioning_channels, 3)

    def test_custom_config(self):
        """测试自定义配置"""
        config = ControlNetConfig(image_size=256, conditioning_channels=1)
        self.assertEqual(config.image_size, 256)
        self.assertEqual(config.conditioning_channels, 1)


class TestZeroConv(unittest.TestCase):
    """测试零卷积"""

    def test_zero_initialization(self):
        """测试零初始化"""
        conv = ZeroConv(64, 128)
        # 检查权重是否为零
        self.assertTrue(torch.allclose(conv.conv.weight, torch.zeros_like(conv.conv.weight)))
        self.assertTrue(torch.allclose(conv.conv.bias, torch.zeros_like(conv.conv.bias)))

    def test_output_shape(self):
        """测试输出形状"""
        conv = ZeroConv(64, 128)
        x = torch.randn(2, 64, 32, 32)
        output = conv(x)
        self.assertEqual(output.shape, (2, 128, 32, 32))

    def test_zero_output(self):
        """测试零输出"""
        conv = ZeroConv(64, 128)
        x = torch.randn(2, 64, 32, 32)
        output = conv(x)
        # 初始时输出应该为零
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))


class TestControlNetConditioningEmbedding(unittest.TestCase):
    """测试条件编码器"""

    def test_output_shape(self):
        """测试输出形状"""
        config = ControlNetConfig(
            image_size=256,
            model_channels=128,
            conditioning_channels=3,
            conditioning_embedding_channels=256
        )
        embed = ControlNetConditioningEmbedding(config)
        # 输入是原始图像尺寸
        x = torch.randn(2, 3, 256, 256)
        output = embed(x)
        # 输出应该是潜在空间尺寸
        self.assertEqual(output.shape, (2, 128, 32, 32))


class TestControlNetBlock(unittest.TestCase):
    """测试 ControlNet 块"""

    def test_without_attention(self):
        """测试不带注意力"""
        config = ControlNetConfig(model_channels=128, context_dim=256)
        block = ControlNetBlock(128, 256, time_embed_dim=512, config=config, has_attention=False)
        x = torch.randn(2, 128, 32, 32)
        t_emb = torch.randn(2, 512)
        output = block(x, t_emb)
        self.assertEqual(output.shape, (2, 256, 32, 32))

    def test_with_attention(self):
        """测试带注意力"""
        config = ControlNetConfig(model_channels=128, context_dim=256, num_heads=4)
        block = ControlNetBlock(256, 256, time_embed_dim=512, config=config, has_attention=True)
        x = torch.randn(2, 256, 16, 16)
        t_emb = torch.randn(2, 512)
        context = torch.randn(2, 77, 256)
        output = block(x, t_emb, context)
        self.assertEqual(output.shape, x.shape)


class TestCreateControlNet(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_canny(self):
        """测试创建 Canny ControlNet"""
        model = create_controlnet("canny", "tiny")
        self.assertEqual(model.config.conditioning_channels, 1)

    def test_create_pose(self):
        """测试创建 Pose ControlNet"""
        model = create_controlnet("pose", "tiny")
        self.assertEqual(model.config.conditioning_channels, 3)

    def test_create_depth(self):
        """测试创建 Depth ControlNet"""
        model = create_controlnet("depth", "tiny")
        self.assertEqual(model.config.conditioning_channels, 1)

    def test_invalid_control_type(self):
        """测试无效的控制类型"""
        with self.assertRaises(ValueError):
            create_controlnet("invalid")

    def test_invalid_model_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_controlnet("canny", "invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
