"""
Stable Diffusion 模型单元测试

测试覆盖:
    - SDConfig 配置验证
    - CLIPTextEncoder 文本编码器
    - CrossAttention 交叉注意力
    - SpatialTransformer 空间 Transformer
    - SDUNet 条件 UNet
    - StableDiffusion 完整模型
"""

import unittest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_diffusion import (
    SDConfig,
    CLIPTextEmbedding,
    CLIPAttention,
    CLIPMLP,
    CLIPEncoderLayer,
    CLIPTextEncoder,
    CrossAttention,
    GEGLU,
    FeedForward,
    BasicTransformerBlock,
    SpatialTransformer,
    SDResBlock,
    SDNoiseScheduler,
    create_sd_model,
)


class TestSDConfig(unittest.TestCase):
    """测试 Stable Diffusion 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = SDConfig()
        self.assertEqual(config.image_size, 512)
        self.assertEqual(config.latent_channels, 4)

    def test_custom_config(self):
        """测试自定义配置"""
        config = SDConfig(image_size=256, model_channels=128)
        self.assertEqual(config.image_size, 256)
        self.assertEqual(config.model_channels, 128)


class TestCLIPTextEmbedding(unittest.TestCase):
    """测试 CLIP 文本嵌入"""

    def test_output_shape(self):
        """测试输出形状"""
        config = SDConfig(vocab_size=1000, max_text_length=77, text_embed_dim=256)
        embed = CLIPTextEmbedding(config)
        input_ids = torch.randint(0, 1000, (2, 20))
        output = embed(input_ids)
        self.assertEqual(output.shape, (2, 20, 256))


class TestCLIPAttention(unittest.TestCase):
    """测试 CLIP 注意力"""

    def test_output_shape(self):
        """测试输出形状"""
        attn = CLIPAttention(embed_dim=256, num_heads=4)
        x = torch.randn(2, 20, 256)
        output = attn(x)
        self.assertEqual(output.shape, x.shape)

    def test_without_causal_mask(self):
        """测试不带因果掩码"""
        attn = CLIPAttention(embed_dim=256, num_heads=4)
        x = torch.randn(2, 20, 256)
        output = attn(x, causal_mask=False)
        self.assertEqual(output.shape, x.shape)


class TestCLIPMLP(unittest.TestCase):
    """测试 CLIP MLP"""

    def test_output_shape(self):
        """测试输出形状"""
        mlp = CLIPMLP(embed_dim=256)
        x = torch.randn(2, 20, 256)
        output = mlp(x)
        self.assertEqual(output.shape, x.shape)


class TestCLIPEncoderLayer(unittest.TestCase):
    """测试 CLIP 编码器层"""

    def test_output_shape(self):
        """测试输出形状"""
        layer = CLIPEncoderLayer(embed_dim=256, num_heads=4)
        x = torch.randn(2, 20, 256)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)


class TestCLIPTextEncoder(unittest.TestCase):
    """测试 CLIP 文本编码器"""

    def setUp(self):
        self.config = SDConfig(
            vocab_size=1000,
            max_text_length=77,
            text_embed_dim=256,
            text_num_layers=2,
            text_num_heads=4
        )
        self.encoder = CLIPTextEncoder(self.config)

    def test_output_shape(self):
        """测试输出形状"""
        input_ids = torch.randint(0, 1000, (2, 20))
        output = self.encoder(input_ids)
        self.assertEqual(output.shape, (2, 20, 256))


class TestCrossAttention(unittest.TestCase):
    """测试交叉注意力"""

    def test_self_attention(self):
        """测试自注意力模式"""
        attn = CrossAttention(query_dim=256, context_dim=256, num_heads=4, head_dim=64)
        x = torch.randn(2, 100, 256)
        output = attn(x)
        self.assertEqual(output.shape, x.shape)

    def test_cross_attention(self):
        """测试交叉注意力模式"""
        attn = CrossAttention(query_dim=256, context_dim=512, num_heads=4, head_dim=64)
        x = torch.randn(2, 100, 256)
        context = torch.randn(2, 77, 512)
        output = attn(x, context)
        self.assertEqual(output.shape, x.shape)


class TestGEGLU(unittest.TestCase):
    """测试 GEGLU 激活"""

    def test_output_shape(self):
        """测试输出形状"""
        geglu = GEGLU(dim_in=256, dim_out=512)
        x = torch.randn(2, 100, 256)
        output = geglu(x)
        self.assertEqual(output.shape, (2, 100, 512))


class TestFeedForward(unittest.TestCase):
    """测试前馈网络"""

    def test_output_shape(self):
        """测试输出形状"""
        ff = FeedForward(dim=256, mult=4)
        x = torch.randn(2, 100, 256)
        output = ff(x)
        self.assertEqual(output.shape, x.shape)


class TestBasicTransformerBlock(unittest.TestCase):
    """测试基础 Transformer 块"""

    def test_output_shape(self):
        """测试输出形状"""
        block = BasicTransformerBlock(dim=256, num_heads=4, head_dim=64, context_dim=512)
        x = torch.randn(2, 100, 256)
        context = torch.randn(2, 77, 512)
        output = block(x, context)
        self.assertEqual(output.shape, x.shape)

    def test_without_context(self):
        """测试不带上下文"""
        block = BasicTransformerBlock(dim=256, num_heads=4, head_dim=64, context_dim=256)
        x = torch.randn(2, 100, 256)
        output = block(x)
        self.assertEqual(output.shape, x.shape)


class TestSpatialTransformer(unittest.TestCase):
    """测试空间 Transformer"""

    def test_output_shape(self):
        """测试输出形状"""
        transformer = SpatialTransformer(
            in_channels=256, num_heads=4, head_dim=64,
            depth=1, context_dim=512
        )
        x = torch.randn(2, 256, 16, 16)
        context = torch.randn(2, 77, 512)
        output = transformer(x, context)
        self.assertEqual(output.shape, x.shape)


class TestSDResBlock(unittest.TestCase):
    """测试 SD 残差块"""

    def test_same_channels(self):
        """测试相同通道数"""
        block = SDResBlock(256, 256, time_embed_dim=512)
        x = torch.randn(2, 256, 16, 16)
        t_emb = torch.randn(2, 512)
        output = block(x, t_emb)
        self.assertEqual(output.shape, x.shape)

    def test_different_channels(self):
        """测试不同通道数"""
        block = SDResBlock(256, 512, time_embed_dim=512)
        x = torch.randn(2, 256, 16, 16)
        t_emb = torch.randn(2, 512)
        output = block(x, t_emb)
        self.assertEqual(output.shape, (2, 512, 16, 16))


class TestSDNoiseScheduler(unittest.TestCase):
    """测试 SD 噪声调度器"""

    def setUp(self):
        self.config = SDConfig(num_timesteps=1000)
        self.scheduler = SDNoiseScheduler(self.config)

    def test_add_noise(self):
        """测试添加噪声"""
        latents = torch.randn(2, 4, 32, 32)
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([100, 500])
        noisy = self.scheduler.add_noise(latents, noise, timesteps)
        self.assertEqual(noisy.shape, latents.shape)


class TestCreateSDModel(unittest.TestCase):
    """测试模型创建函数"""

    def test_create_tiny(self):
        """测试创建 tiny 模型"""
        model = create_sd_model("tiny")
        self.assertEqual(model.config.image_size, 256)

    def test_invalid_size(self):
        """测试无效的模型大小"""
        with self.assertRaises(ValueError):
            create_sd_model("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
