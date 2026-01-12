"""
SDXL 单元测试
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sdxl import (
    SDXLConfig,
    SDXLRefinerConfig,
    SDXLModelType,
    CLIPTextEmbedding,
    CLIPAttention,
    CLIPMLP,
    CLIPEncoderLayer,
    CLIPTextEncoderWithPooling,
    SDXLTextEncoder,
    TimestepEmbedding,
    SinusoidalTimestepEmbedding,
    SDXLAdditionEmbedding,
    SDXLCrossAttention,
    GEGLU,
    SDXLFeedForward,
    SDXLTransformerBlock,
    SDXLSpatialTransformer,
    SDXLResBlock,
    SDXLDownsample,
    SDXLUpsample,
    SDXLDownBlock,
    SDXLUpBlock,
    SDXLMidBlock,
    SDXLUNet,
    SDXLNoiseScheduler,
    SDXL,
    create_sdxl_model
)


# ============================================================================
# 测试配置
# ============================================================================

class TestSDXLConfig:
    """测试 SDXL 配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SDXLConfig()
        assert config.image_size == 1024
        assert config.latent_channels == 4
        assert config.model_channels == 320
        assert config.context_dim == 2048
        assert config.pooled_embed_dim == 1280
        
    def test_custom_config(self):
        """测试自定义配置"""
        config = SDXLConfig(
            image_size=512,
            model_channels=256,
            context_dim=1024
        )
        assert config.image_size == 512
        assert config.model_channels == 256
        assert config.context_dim == 1024
        
    def test_refiner_config(self):
        """测试 Refiner 配置"""
        config = SDXLRefinerConfig()
        assert config.model_type == SDXLModelType.REFINER
        assert config.context_dim == 1280


# ============================================================================
# 测试文本编码器组件
# ============================================================================

class TestCLIPTextEmbedding:
    """测试 CLIP 文本嵌入"""
    
    def test_forward(self):
        """测试前向传播"""
        embedding = CLIPTextEmbedding(vocab_size=1000, embed_dim=256, max_length=77)
        input_ids = torch.randint(0, 1000, (2, 50))
        
        output = embedding(input_ids)
        
        assert output.shape == (2, 50, 256)
        

class TestCLIPAttention:
    """测试 CLIP 注意力"""
    
    def test_forward(self):
        """测试前向传播"""
        attn = CLIPAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 50, 256)
        
        output = attn(x)
        
        assert output.shape == x.shape
        
    def test_causal_mask(self):
        """测试因果掩码"""
        attn = CLIPAttention(embed_dim=256, num_heads=8)
        x = torch.randn(2, 50, 256)
        
        output_causal = attn(x, causal_mask=True)
        output_no_causal = attn(x, causal_mask=False)
        
        assert not torch.allclose(output_causal, output_no_causal)


class TestCLIPMLP:
    """测试 CLIP MLP"""
    
    def test_forward(self):
        """测试前向传播"""
        mlp = CLIPMLP(embed_dim=256)
        x = torch.randn(2, 50, 256)
        
        output = mlp(x)
        
        assert output.shape == x.shape


class TestCLIPEncoderLayer:
    """测试 CLIP 编码器层"""
    
    def test_forward(self):
        """测试前向传播"""
        layer = CLIPEncoderLayer(embed_dim=256, num_heads=8)
        x = torch.randn(2, 50, 256)
        
        output = layer(x)
        
        assert output.shape == x.shape


class TestCLIPTextEncoderWithPooling:
    """测试带池化的 CLIP 文本编码器"""
    
    @pytest.fixture
    def encoder(self):
        return CLIPTextEncoderWithPooling(
            vocab_size=1000,
            embed_dim=256,
            num_layers=4,
            num_heads=8,
            max_length=77
        )
        
    def test_forward_with_pooling(self, encoder):
        """测试带池化的前向传播"""
        input_ids = torch.randint(0, 1000, (2, 50))
        
        hidden_states, pooled = encoder(input_ids, return_pooled=True)
        
        assert hidden_states.shape == (2, 50, 256)
        assert pooled.shape == (2, 256)
        
    def test_forward_without_pooling(self, encoder):
        """测试不带池化的前向传播"""
        input_ids = torch.randint(0, 1000, (2, 50))
        
        hidden_states, pooled = encoder(input_ids, return_pooled=False)
        
        assert hidden_states.shape == (2, 50, 256)
        assert pooled is None


class TestSDXLTextEncoder:
    """测试 SDXL 双文本编码器"""
    
    @pytest.fixture
    def config(self):
        return SDXLConfig(
            clip_vocab_size=1000,
            clip_embed_dim=256,
            clip_num_layers=2,
            clip_num_heads=4,
            openclip_vocab_size=1000,
            openclip_embed_dim=512,
            openclip_num_layers=2,
            openclip_num_heads=8
        )
        
    def test_forward(self, config):
        """测试前向传播"""
        encoder = SDXLTextEncoder(config)
        clip_ids = torch.randint(0, 1000, (2, 50))
        openclip_ids = torch.randint(0, 1000, (2, 50))
        
        text_embeds, pooled_embeds = encoder(clip_ids, openclip_ids)
        
        # 拼接后的维度
        assert text_embeds.shape == (2, 50, 256 + 512)
        assert pooled_embeds.shape == (2, 512)


# ============================================================================
# 测试条件嵌入
# ============================================================================

class TestTimestepEmbedding:
    """测试时间步嵌入"""
    
    def test_forward(self):
        """测试前向传播"""
        emb = TimestepEmbedding(256, 1024)
        x = torch.randn(4, 256)
        
        output = emb(x)
        
        assert output.shape == (4, 1024)


class TestSinusoidalTimestepEmbedding:
    """测试正弦时间步嵌入"""
    
    def test_forward(self):
        """测试前向传播"""
        emb = SinusoidalTimestepEmbedding(256)
        timesteps = torch.tensor([0, 100, 500, 999])
        
        output = emb(timesteps)
        
        assert output.shape == (4, 256)
        
    def test_different_timesteps_different_embeddings(self):
        """测试不同时间步产生不同嵌入"""
        emb = SinusoidalTimestepEmbedding(256)
        t1 = torch.tensor([100])
        t2 = torch.tensor([500])
        
        e1 = emb(t1)
        e2 = emb(t2)
        
        assert not torch.allclose(e1, e2)


class TestSDXLAdditionEmbedding:
    """测试 SDXL 额外条件嵌入"""
    
    @pytest.fixture
    def config(self):
        return SDXLConfig(
            model_channels=128,
            pooled_embed_dim=256,
            addition_time_embed_dim=64
        )
        
    def test_forward(self, config):
        """测试前向传播"""
        emb = SDXLAdditionEmbedding(config)
        
        pooled = torch.randn(2, 256)
        original_size = torch.tensor([[1024, 1024], [1024, 1024]]).float()
        crop_coords = torch.zeros(2, 2)
        target_size = torch.tensor([[1024, 1024], [1024, 1024]]).float()
        
        output = emb(pooled, original_size, crop_coords, target_size)
        
        assert output.shape == (2, 128 * 4)  # time_embed_dim


# ============================================================================
# 测试注意力模块
# ============================================================================

class TestSDXLCrossAttention:
    """测试 SDXL 交叉注意力"""
    
    def test_self_attention(self):
        """测试自注意力"""
        attn = SDXLCrossAttention(query_dim=256, num_heads=8, head_dim=32)
        x = torch.randn(2, 100, 256)
        
        output = attn(x)
        
        assert output.shape == x.shape
        
    def test_cross_attention(self):
        """测试交叉注意力"""
        attn = SDXLCrossAttention(query_dim=256, context_dim=512, num_heads=8, head_dim=32)
        x = torch.randn(2, 100, 256)
        context = torch.randn(2, 77, 512)
        
        output = attn(x, context)
        
        assert output.shape == x.shape


class TestGEGLU:
    """测试 GEGLU"""
    
    def test_forward(self):
        """测试前向传播"""
        geglu = GEGLU(256, 512)
        x = torch.randn(2, 100, 256)
        
        output = geglu(x)
        
        assert output.shape == (2, 100, 512)


class TestSDXLFeedForward:
    """测试 SDXL 前馈网络"""
    
    def test_forward(self):
        """测试前向传播"""
        ff = SDXLFeedForward(256)
        x = torch.randn(2, 100, 256)
        
        output = ff(x)
        
        assert output.shape == x.shape


class TestSDXLTransformerBlock:
    """测试 SDXL Transformer 块"""
    
    def test_forward_self_attention(self):
        """测试自注意力"""
        # context_dim 应该与 dim 相同，因为没有 context 时会使用 x
        block = SDXLTransformerBlock(dim=256, num_heads=8, head_dim=32, context_dim=256)
        x = torch.randn(2, 100, 256)
        
        output = block(x)
        
        assert output.shape == x.shape
        
    def test_forward_cross_attention(self):
        """测试交叉注意力"""
        block = SDXLTransformerBlock(dim=256, num_heads=8, head_dim=32, context_dim=512)
        x = torch.randn(2, 100, 256)
        context = torch.randn(2, 77, 512)
        
        output = block(x, context)
        
        assert output.shape == x.shape


class TestSDXLSpatialTransformer:
    """测试 SDXL 空间 Transformer"""
    
    def test_forward(self):
        """测试前向传播"""
        transformer = SDXLSpatialTransformer(
            in_channels=256, num_heads=8, head_dim=32,
            depth=2, context_dim=512
        )
        x = torch.randn(2, 256, 16, 16)
        context = torch.randn(2, 77, 512)
        
        output = transformer(x, context)
        
        assert output.shape == x.shape
        
    def test_residual_connection(self):
        """测试残差连接"""
        # 使用相同的 context_dim 和 inner_dim
        inner_dim = 8 * 32  # num_heads * head_dim = 256
        transformer = SDXLSpatialTransformer(
            in_channels=256, num_heads=8, head_dim=32,
            depth=1, context_dim=256  # 与 inner_dim 相同
        )
        x = torch.randn(2, 256, 8, 8)
        
        output = transformer(x)
        
        # 输出应该与输入不同（有变换）但形状相同
        assert output.shape == x.shape
        assert not torch.allclose(output, x)


# ============================================================================
# 测试 UNet 组件
# ============================================================================

class TestSDXLResBlock:
    """测试 SDXL 残差块"""
    
    def test_same_channels(self):
        """测试相同通道数"""
        block = SDXLResBlock(256, 256, time_embed_dim=512)
        x = torch.randn(2, 256, 16, 16)
        t_emb = torch.randn(2, 512)
        
        output = block(x, t_emb)
        
        assert output.shape == x.shape
        
    def test_different_channels(self):
        """测试不同通道数"""
        block = SDXLResBlock(256, 512, time_embed_dim=512)
        x = torch.randn(2, 256, 16, 16)
        t_emb = torch.randn(2, 512)
        
        output = block(x, t_emb)
        
        assert output.shape == (2, 512, 16, 16)


class TestSDXLDownsample:
    """测试 SDXL 下采样"""
    
    def test_forward(self):
        """测试前向传播"""
        down = SDXLDownsample(256)
        x = torch.randn(2, 256, 32, 32)
        
        output = down(x)
        
        assert output.shape == (2, 256, 16, 16)


class TestSDXLUpsample:
    """测试 SDXL 上采样"""
    
    def test_forward(self):
        """测试前向传播"""
        up = SDXLUpsample(256)
        x = torch.randn(2, 256, 16, 16)
        
        output = up(x)
        
        assert output.shape == (2, 256, 32, 32)


class TestSDXLDownBlock:
    """测试 SDXL 下采样块"""
    
    def test_forward_with_attention(self):
        """测试带注意力的前向传播"""
        block = SDXLDownBlock(
            in_channels=128, out_channels=256, time_embed_dim=512,
            num_res_blocks=2, transformer_depth=1, num_heads=4,
            head_dim=32, context_dim=512, add_downsample=True
        )
        x = torch.randn(2, 128, 32, 32)
        t_emb = torch.randn(2, 512)
        context = torch.randn(2, 77, 512)
        
        output, skips = block(x, t_emb, context)
        
        assert output.shape == (2, 256, 16, 16)
        assert len(skips) == 3  # 2 res blocks + 1 downsample
        
    def test_forward_without_downsample(self):
        """测试不带下采样的前向传播"""
        block = SDXLDownBlock(
            in_channels=256, out_channels=256, time_embed_dim=512,
            num_res_blocks=2, transformer_depth=0, num_heads=4,
            head_dim=32, context_dim=512, add_downsample=False
        )
        x = torch.randn(2, 256, 16, 16)
        t_emb = torch.randn(2, 512)
        
        output, skips = block(x, t_emb)
        
        assert output.shape == x.shape
        assert len(skips) == 2


class TestSDXLUpBlock:
    """测试 SDXL 上采样块"""
    
    def test_forward(self):
        """测试前向传播"""
        block = SDXLUpBlock(
            in_channels=512, out_channels=256, prev_channels=256,
            time_embed_dim=512, num_res_blocks=2, transformer_depth=1,
            num_heads=4, head_dim=32, context_dim=512, add_upsample=True
        )
        x = torch.randn(2, 512, 16, 16)
        t_emb = torch.randn(2, 512)
        context = torch.randn(2, 77, 512)
        skips = [torch.randn(2, 256, 16, 16) for _ in range(2)]
        
        output = block(x, t_emb, skips, context)
        
        assert output.shape == (2, 256, 32, 32)


class TestSDXLMidBlock:
    """测试 SDXL 中间块"""
    
    def test_forward(self):
        """测试前向传播"""
        block = SDXLMidBlock(
            channels=512, time_embed_dim=512, transformer_depth=2,
            num_heads=8, head_dim=64, context_dim=512
        )
        x = torch.randn(2, 512, 8, 8)
        t_emb = torch.randn(2, 512)
        context = torch.randn(2, 77, 512)
        
        output = block(x, t_emb, context)
        
        assert output.shape == x.shape


# ============================================================================
# 测试 SDXL UNet
# ============================================================================

class TestSDXLUNet:
    """测试 SDXL UNet"""
    
    @pytest.fixture
    def tiny_config(self):
        return SDXLConfig(
            image_size=64,
            model_channels=32,
            channel_mult=(1, 2),
            num_res_blocks=1,
            transformer_depth=(1, 1),
            head_dim=32,
            context_dim=128,
            pooled_embed_dim=64,
            addition_time_embed_dim=32
        )
        
    def test_forward(self, tiny_config):
        """测试前向传播"""
        unet = SDXLUNet(tiny_config)
        
        batch_size = 2
        latent_size = tiny_config.image_size // tiny_config.latent_scale_factor
        
        x = torch.randn(batch_size, tiny_config.latent_channels, latent_size, latent_size)
        timesteps = torch.randint(0, 1000, (batch_size,))
        context = torch.randn(batch_size, 77, tiny_config.context_dim)
        pooled = torch.randn(batch_size, tiny_config.pooled_embed_dim)
        original_size = torch.tensor([[64, 64], [64, 64]]).float()
        crop_coords = torch.zeros(batch_size, 2)
        target_size = original_size.clone()
        
        output = unet(x, timesteps, context, pooled, original_size, crop_coords, target_size)
        
        assert output.shape == x.shape
        
    def test_output_is_finite(self, tiny_config):
        """测试输出是有限的"""
        unet = SDXLUNet(tiny_config)
        
        batch_size = 1
        latent_size = tiny_config.image_size // tiny_config.latent_scale_factor
        
        x = torch.randn(batch_size, tiny_config.latent_channels, latent_size, latent_size)
        timesteps = torch.tensor([500])
        context = torch.randn(batch_size, 77, tiny_config.context_dim)
        pooled = torch.randn(batch_size, tiny_config.pooled_embed_dim)
        original_size = torch.tensor([[64, 64]]).float()
        crop_coords = torch.zeros(batch_size, 2)
        target_size = original_size.clone()
        
        output = unet(x, timesteps, context, pooled, original_size, crop_coords, target_size)
        
        assert torch.isfinite(output).all()


# ============================================================================
# 测试噪声调度器
# ============================================================================

class TestSDXLNoiseScheduler:
    """测试 SDXL 噪声调度器"""
    
    @pytest.fixture
    def scheduler(self):
        config = SDXLConfig()
        return SDXLNoiseScheduler(config)
        
    def test_initialization(self, scheduler):
        """测试初始化"""
        assert scheduler.num_timesteps == 1000
        assert len(scheduler.betas) == 1000
        assert len(scheduler.alphas_cumprod) == 1000
        
    def test_alphas_cumprod_decreasing(self, scheduler):
        """测试 alphas_cumprod 递减"""
        for i in range(len(scheduler.alphas_cumprod) - 1):
            assert scheduler.alphas_cumprod[i] > scheduler.alphas_cumprod[i + 1]
            
    def test_add_noise(self, scheduler):
        """测试添加噪声"""
        latents = torch.randn(2, 4, 32, 32)
        noise = torch.randn_like(latents)
        timesteps = torch.tensor([100, 500])
        
        noisy = scheduler.add_noise(latents, noise, timesteps)
        
        assert noisy.shape == latents.shape
        assert not torch.allclose(noisy, latents)


# ============================================================================
# 测试完整 SDXL 模型
# ============================================================================

class TestSDXL:
    """测试完整 SDXL 模型"""
    
    @pytest.fixture
    def tiny_model(self):
        config = SDXLConfig(
            image_size=64,
            model_channels=32,
            channel_mult=(1, 2),
            num_res_blocks=1,
            transformer_depth=(1, 1),
            head_dim=32,
            clip_embed_dim=64,
            clip_num_layers=2,
            clip_num_heads=4,
            clip_vocab_size=1000,
            openclip_embed_dim=64,
            openclip_num_layers=2,
            openclip_num_heads=4,
            openclip_vocab_size=1000,
            context_dim=128,
            pooled_embed_dim=64,
            addition_time_embed_dim=32
        )
        return SDXL(config)
        
    def test_encode_text(self, tiny_model):
        """测试文本编码"""
        clip_ids = torch.randint(0, 1000, (2, 50))
        openclip_ids = torch.randint(0, 1000, (2, 50))
        
        text_embeds, pooled = tiny_model.encode_text(clip_ids, openclip_ids)
        
        assert text_embeds.shape == (2, 50, 128)  # 64 + 64
        assert pooled.shape == (2, 64)
        
    def test_forward(self, tiny_model):
        """测试训练前向传播"""
        batch_size = 2
        latent_size = tiny_model.config.image_size // tiny_model.config.latent_scale_factor
        
        latents = torch.randn(batch_size, 4, latent_size, latent_size)
        clip_ids = torch.randint(0, 1000, (batch_size, 50))
        openclip_ids = torch.randint(0, 1000, (batch_size, 50))
        
        loss = tiny_model(latents, clip_ids, openclip_ids)
        
        assert loss.ndim == 0
        assert loss.item() > 0


# ============================================================================
# 测试工厂函数
# ============================================================================

class TestCreateSDXLModel:
    """测试工厂函数"""
    
    def test_create_tiny(self):
        """测试创建 tiny 模型"""
        model = create_sdxl_model("tiny")
        assert isinstance(model, SDXL)
        assert model.config.image_size == 256
        
    def test_create_unknown_raises(self):
        """测试创建未知模型抛出异常"""
        with pytest.raises(ValueError):
            create_sdxl_model("unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
