"""
IP-Adapter 模块单元测试
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ip_adapter import (
    IPAdapterType, IPAdapterConfig,
    ImageProjection, ImageProjectionPlus, PerceiverAttentionBlock,
    IPAdapterCrossAttention, CLIPVisionEncoder, VisionTransformerBlock,
    IPAdapter, IPAdapterManager, create_ip_adapter
)


class TestIPAdapterConfig:
    """测试 IP-Adapter 配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = IPAdapterConfig()
        
        assert config.adapter_type == IPAdapterType.STANDARD
        assert config.image_encoder_dim == 1024
        assert config.num_image_tokens == 4
        assert config.cross_attention_dim == 768
        
    def test_custom_config(self):
        """测试自定义配置"""
        config = IPAdapterConfig(
            adapter_type=IPAdapterType.PLUS,
            image_encoder_dim=768,
            num_queries=32,
            scale=0.5
        )
        
        assert config.adapter_type == IPAdapterType.PLUS
        assert config.image_encoder_dim == 768
        assert config.num_queries == 32
        assert config.scale == 0.5


class TestImageProjection:
    """测试图像投影层"""
    
    def test_initialization(self):
        """测试初始化"""
        proj = ImageProjection(
            image_embed_dim=1024,
            cross_attention_dim=768,
            num_image_tokens=4
        )
        
        assert proj.image_embed_dim == 1024
        assert proj.cross_attention_dim == 768
        assert proj.num_image_tokens == 4
        
    def test_forward_shape(self):
        """测试前向传播形状"""
        proj = ImageProjection(
            image_embed_dim=1024,
            cross_attention_dim=768,
            num_image_tokens=4
        )
        
        image_embeds = torch.randn(2, 1024)
        output = proj(image_embeds)
        
        assert output.shape == (2, 4, 768)
        
    def test_output_is_finite(self):
        """测试输出是有限的"""
        proj = ImageProjection(1024, 768, 4)
        image_embeds = torch.randn(2, 1024)
        
        output = proj(image_embeds)
        
        assert torch.isfinite(output).all()


class TestImageProjectionPlus:
    """测试 IP-Adapter Plus 图像投影层"""
    
    def test_initialization(self):
        """测试初始化"""
        proj = ImageProjectionPlus(
            image_embed_dim=1024,
            cross_attention_dim=768,
            num_queries=16,
            num_heads=12,
            num_layers=4
        )
        
        assert proj.num_queries == 16
        assert proj.cross_attention_dim == 768
        assert len(proj.layers) == 4
        
    def test_forward_shape(self):
        """测试前向传播形状"""
        proj = ImageProjectionPlus(
            image_embed_dim=1024,
            cross_attention_dim=768,
            num_queries=16,
            num_heads=12,
            num_layers=2
        )
        
        image_embeds = torch.randn(2, 1024)
        output = proj(image_embeds)
        
        assert output.shape == (2, 16, 768)
        
    def test_forward_with_hidden_states(self):
        """测试带隐藏状态的前向传播"""
        proj = ImageProjectionPlus(
            image_embed_dim=1024,
            cross_attention_dim=768,
            num_queries=16,
            num_heads=12,
            num_layers=2
        )
        
        image_embeds = torch.randn(2, 1024)
        hidden_states = torch.randn(2, 50, 1024)  # 细粒度特征
        
        output = proj(image_embeds, hidden_states)
        
        assert output.shape == (2, 16, 768)


class TestPerceiverAttentionBlock:
    """测试 Perceiver 注意力块"""
    
    def test_forward(self):
        """测试前向传播"""
        block = PerceiverAttentionBlock(dim=768, num_heads=12)
        
        queries = torch.randn(2, 16, 768)
        context = torch.randn(2, 50, 768)
        
        output = block(queries, context)
        
        assert output.shape == queries.shape
        
    def test_output_is_finite(self):
        """测试输出是有限的"""
        block = PerceiverAttentionBlock(dim=768, num_heads=12)
        
        queries = torch.randn(2, 16, 768)
        context = torch.randn(2, 50, 768)
        
        output = block(queries, context)
        
        assert torch.isfinite(output).all()


class TestIPAdapterCrossAttention:
    """测试 IP-Adapter 解耦交叉注意力"""
    
    def test_initialization(self):
        """测试初始化"""
        attn = IPAdapterCrossAttention(
            query_dim=768,
            cross_attention_dim=768,
            num_heads=8,
            head_dim=64,
            scale=1.0
        )
        
        assert attn.num_heads == 8
        assert attn.head_dim == 64
        assert attn.scale == 1.0
        
    def test_forward_shape(self):
        """测试前向传播形状"""
        attn = IPAdapterCrossAttention(
            query_dim=768,
            cross_attention_dim=768,
            num_heads=8,
            head_dim=64
        )
        
        hidden_states = torch.randn(2, 100, 768)
        text_embeds = torch.randn(2, 77, 768)
        image_embeds = torch.randn(2, 4, 768)
        
        output = attn(hidden_states, text_embeds, image_embeds)
        
        assert output.shape == hidden_states.shape
        
    def test_scale_affects_output(self):
        """测试缩放因子影响输出"""
        attn_scale1 = IPAdapterCrossAttention(768, 768, 8, 64, scale=1.0)
        attn_scale0 = IPAdapterCrossAttention(768, 768, 8, 64, scale=0.0)
        
        # 复制权重
        attn_scale0.load_state_dict(attn_scale1.state_dict())
        
        hidden_states = torch.randn(2, 100, 768)
        text_embeds = torch.randn(2, 77, 768)
        image_embeds = torch.randn(2, 4, 768)
        
        out1 = attn_scale1(hidden_states, text_embeds, image_embeds)
        out0 = attn_scale0(hidden_states, text_embeds, image_embeds)
        
        # scale=0 时图像不应该影响输出
        assert not torch.allclose(out1, out0)


class TestCLIPVisionEncoder:
    """测试 CLIP 视觉编码器"""
    
    def test_initialization(self):
        """测试初始化"""
        encoder = CLIPVisionEncoder(
            image_size=224,
            patch_size=14,
            embed_dim=768,
            num_layers=4,
            num_heads=12,
            output_dim=512
        )
        
        assert encoder.image_size == 224
        assert encoder.patch_size == 14
        assert encoder.num_patches == 256  # (224/14)^2
        
    def test_forward_shape(self):
        """测试前向传播形状"""
        encoder = CLIPVisionEncoder(
            image_size=224,
            patch_size=14,
            embed_dim=256,
            num_layers=2,
            num_heads=4,
            output_dim=512
        )
        
        pixel_values = torch.randn(2, 3, 224, 224)
        output = encoder(pixel_values)
        
        assert output.shape == (2, 512)
        
    def test_forward_with_hidden_states(self):
        """测试带隐藏状态的前向传播"""
        encoder = CLIPVisionEncoder(
            image_size=224,
            patch_size=14,
            embed_dim=256,
            num_layers=2,
            num_heads=4,
            output_dim=512
        )
        
        pixel_values = torch.randn(2, 3, 224, 224)
        output, hidden_states = encoder(pixel_values, output_hidden_states=True)
        
        assert output.shape == (2, 512)
        assert len(hidden_states) == 2  # num_layers


class TestVisionTransformerBlock:
    """测试 Vision Transformer 块"""
    
    def test_forward(self):
        """测试前向传播"""
        block = VisionTransformerBlock(dim=768, num_heads=12)
        x = torch.randn(2, 197, 768)  # 196 patches + 1 CLS
        
        output = block(x)
        
        assert output.shape == x.shape


class TestIPAdapter:
    """测试 IP-Adapter 主模块"""
    
    def test_standard_adapter(self):
        """测试标准 IP-Adapter"""
        config = IPAdapterConfig(
            adapter_type=IPAdapterType.STANDARD,
            image_encoder_dim=1024,
            cross_attention_dim=768,
            num_image_tokens=4
        )
        adapter = IPAdapter(config)
        
        image_embeds = torch.randn(2, 1024)
        output = adapter(image_embeds)
        
        assert output.shape == (2, 4, 768)
        
    def test_plus_adapter(self):
        """测试 IP-Adapter Plus"""
        config = IPAdapterConfig(
            adapter_type=IPAdapterType.PLUS,
            image_encoder_dim=1024,
            cross_attention_dim=768,
            num_queries=16,
            num_heads=12,
            num_projection_layers=2
        )
        adapter = IPAdapter(config)
        
        image_embeds = torch.randn(2, 1024)
        output = adapter(image_embeds)
        
        assert output.shape == (2, 16, 768)
        
    def test_adapter_with_face_id(self):
        """测试带人脸 ID 的 IP-Adapter"""
        config = IPAdapterConfig(
            adapter_type=IPAdapterType.STANDARD,
            image_encoder_dim=1024,
            cross_attention_dim=768,
            num_image_tokens=4,
            use_face_id=True,
            face_embed_dim=512
        )
        adapter = IPAdapter(config)
        
        image_embeds = torch.randn(2, 1024)
        face_embeds = torch.randn(2, 512)
        
        output = adapter(image_embeds, face_embeds=face_embeds)
        
        # 4 image tokens + 1 face token
        assert output.shape == (2, 5, 768)
        
    def test_encode_image(self):
        """测试图像编码"""
        config = IPAdapterConfig(
            adapter_type=IPAdapterType.STANDARD,
            image_encoder_dim=1024,
            cross_attention_dim=768,
            num_image_tokens=4
        )
        adapter = IPAdapter(config)
        
        image_embeds = torch.randn(2, 1024)
        output = adapter.encode_image(image_embeds)
        
        assert output.shape == (2, 4, 768)


class TestIPAdapterManager:
    """测试 IP-Adapter 管理器"""
    
    def test_initialization(self):
        """测试初始化"""
        config = IPAdapterConfig()
        manager = IPAdapterManager(config)
        
        assert manager.ip_adapter is not None
        assert len(manager.injected_attentions) == 0
        
    def test_get_image_embeds(self):
        """测试获取图像嵌入"""
        config = IPAdapterConfig(
            image_encoder_dim=1024,
            cross_attention_dim=768,
            num_image_tokens=4
        )
        manager = IPAdapterManager(config)
        
        image_embeds = torch.randn(2, 1024)
        output = manager.get_image_embeds(image_embeds)
        
        assert output.shape == (2, 4, 768)
        
    def test_save_load_weights(self):
        """测试保存和加载权重"""
        config = IPAdapterConfig(
            image_encoder_dim=512,
            cross_attention_dim=256,
            num_image_tokens=4
        )
        manager = IPAdapterManager(config)
        
        # 保存权重
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            
        try:
            manager.save_weights(temp_path)
            
            # 创建新管理器并加载
            new_manager = IPAdapterManager(config)
            new_manager.load_weights(temp_path)
            
            # 验证权重相同
            for (name1, param1), (name2, param2) in zip(
                manager.ip_adapter.named_parameters(),
                new_manager.ip_adapter.named_parameters()
            ):
                assert torch.allclose(param1, param2)
        finally:
            os.unlink(temp_path)


class TestCreateIPAdapter:
    """测试 IP-Adapter 工厂函数"""
    
    def test_create_standard(self):
        """测试创建标准 IP-Adapter"""
        adapter, config = create_ip_adapter(adapter_type="standard")
        
        assert config.adapter_type == IPAdapterType.STANDARD
        assert not config.use_fine_grained_features
        
    def test_create_plus(self):
        """测试创建 IP-Adapter Plus"""
        adapter, config = create_ip_adapter(adapter_type="plus")
        
        assert config.adapter_type == IPAdapterType.PLUS
        assert config.use_fine_grained_features
        
    def test_create_plus_face(self):
        """测试创建 IP-Adapter Plus Face"""
        adapter, config = create_ip_adapter(adapter_type="plus_face")
        
        assert config.adapter_type == IPAdapterType.PLUS_FACE
        assert config.use_face_id
        
    def test_create_with_custom_params(self):
        """测试自定义参数"""
        adapter, config = create_ip_adapter(
            adapter_type="standard",
            image_encoder_dim=768,
            cross_attention_dim=512,
            num_tokens=8,
            scale=0.5
        )
        
        assert config.image_encoder_dim == 768
        assert config.cross_attention_dim == 512
        assert config.num_image_tokens == 8
        assert config.scale == 0.5
        
    def test_create_unknown_raises(self):
        """测试未知类型抛出异常"""
        with pytest.raises(ValueError):
            create_ip_adapter(adapter_type="unknown")


class TestIPAdapterIntegration:
    """IP-Adapter 集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        # 创建 IP-Adapter
        config = IPAdapterConfig(
            adapter_type=IPAdapterType.STANDARD,
            image_encoder_dim=512,
            cross_attention_dim=256,
            num_image_tokens=4,
            num_heads=4
        )
        adapter = IPAdapter(config)
        
        # 创建解耦注意力
        cross_attn = IPAdapterCrossAttention(
            query_dim=256,
            cross_attention_dim=256,
            num_heads=4,
            head_dim=64
        )
        
        # 模拟输入
        image_embeds = torch.randn(2, 512)
        hidden_states = torch.randn(2, 100, 256)
        text_embeds = torch.randn(2, 77, 256)
        
        # 获取图像 tokens
        ip_tokens = adapter(image_embeds)
        
        # 通过解耦注意力
        output = cross_attn(hidden_states, text_embeds, ip_tokens)
        
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        
    def test_multi_image_fusion(self):
        """测试多图像融合"""
        config = IPAdapterConfig(
            adapter_type=IPAdapterType.STANDARD,
            image_encoder_dim=512,
            cross_attention_dim=256,
            num_image_tokens=4
        )
        adapter = IPAdapter(config)
        
        # 多个图像
        image_embeds_1 = torch.randn(2, 512)
        image_embeds_2 = torch.randn(2, 512)
        
        # 分别编码
        tokens_1 = adapter(image_embeds_1)
        tokens_2 = adapter(image_embeds_2)
        
        # 融合（简单平均）
        fused_tokens = (tokens_1 + tokens_2) / 2
        
        assert fused_tokens.shape == (2, 4, 256)
        
    def test_scale_interpolation(self):
        """测试缩放插值"""
        config = IPAdapterConfig(
            adapter_type=IPAdapterType.STANDARD,
            image_encoder_dim=512,
            cross_attention_dim=256,
            num_image_tokens=4
        )
        adapter = IPAdapter(config)
        
        image_embeds = torch.randn(2, 512)
        tokens = adapter(image_embeds)
        
        # 不同缩放
        scales = [0.0, 0.5, 1.0]
        outputs = []
        
        for scale in scales:
            scaled_tokens = tokens * scale
            outputs.append(scaled_tokens)
            
        # 验证缩放效果
        assert torch.allclose(outputs[0], torch.zeros_like(outputs[0]))
        assert not torch.allclose(outputs[1], outputs[2])
