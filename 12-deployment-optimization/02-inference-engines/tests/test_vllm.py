"""
vLLM 模块单元测试
"""

import pytest
import numpy as np
import os
import sys

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vllm_inference import (
    SamplingConfig,
    EngineConfig,
    GenerationOutput,
    QuantizationMethod,
    VLLM_AVAILABLE,
)

# 检查 vLLM 是否可用
skip_if_no_vllm = pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="vLLM not available"
)


# ==================== SamplingConfig 测试 ====================

class TestSamplingConfig:
    """测试采样配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = SamplingConfig()

        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.top_k == -1
        assert config.max_tokens == 256
        assert config.min_tokens == 0
        assert config.presence_penalty == 0.0
        assert config.frequency_penalty == 0.0
        assert config.repetition_penalty == 1.0
        assert config.n == 1
        assert config.use_beam_search is False
        assert config.skip_special_tokens is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = SamplingConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=512,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            n=3,
            seed=42
        )

        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.max_tokens == 512
        assert config.presence_penalty == 0.1
        assert config.frequency_penalty == 0.1
        assert config.n == 3
        assert config.seed == 42

    def test_stop_sequences(self):
        """测试停止序列"""
        config = SamplingConfig(
            stop=["</s>", "\n\n"],
            stop_token_ids=[2, 50256]
        )

        assert config.stop == ["</s>", "\n\n"]
        assert config.stop_token_ids == [2, 50256]

    @skip_if_no_vllm
    def test_to_sampling_params(self):
        """测试转换为 SamplingParams"""
        config = SamplingConfig(
            temperature=0.8,
            top_p=0.95,
            max_tokens=128
        )

        params = config.to_sampling_params()
        assert params is not None


# ==================== EngineConfig 测试 ====================

class TestEngineConfig:
    """测试引擎配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = EngineConfig()

        assert config.model == ""
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.gpu_memory_utilization == 0.9
        assert config.swap_space == 4
        assert config.quantization == QuantizationMethod.NONE
        assert config.dtype == "auto"
        assert config.enable_lora is False
        assert config.seed == 0

    def test_custom_config(self):
        """测试自定义配置"""
        config = EngineConfig(
            model="meta-llama/Llama-2-7b-hf",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            quantization=QuantizationMethod.GPTQ,
            dtype="float16",
            enable_lora=True,
            max_loras=4
        )

        assert config.model == "meta-llama/Llama-2-7b-hf"
        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.8
        assert config.quantization == QuantizationMethod.GPTQ
        assert config.dtype == "float16"
        assert config.enable_lora is True
        assert config.max_loras == 4

    def test_speculative_decoding_config(self):
        """测试投机解码配置"""
        config = EngineConfig(
            model="meta-llama/Llama-2-70b-hf",
            speculative_model="meta-llama/Llama-2-7b-hf",
            num_speculative_tokens=5
        )

        assert config.speculative_model == "meta-llama/Llama-2-7b-hf"
        assert config.num_speculative_tokens == 5


# ==================== GenerationOutput 测试 ====================

class TestGenerationOutput:
    """测试生成输出"""

    def test_create_output(self):
        """测试创建输出"""
        output = GenerationOutput(
            prompt="Hello, world!",
            generated_text="How are you?",
            finish_reason="stop",
            prompt_tokens=3,
            generated_tokens=4,
            latency_ms=100.0
        )

        assert output.prompt == "Hello, world!"
        assert output.generated_text == "How are you?"
        assert output.finish_reason == "stop"
        assert output.prompt_tokens == 3
        assert output.generated_tokens == 4
        assert output.latency_ms == 100.0
        assert output.all_outputs is None

    def test_output_with_multiple_generations(self):
        """测试多个生成结果"""
        output = GenerationOutput(
            prompt="Hello",
            generated_text="Hi there!",
            finish_reason="stop",
            prompt_tokens=1,
            generated_tokens=3,
            latency_ms=50.0,
            all_outputs=["Hi there!", "Hello!", "Hey!"]
        )

        assert output.all_outputs is not None
        assert len(output.all_outputs) == 3


# ==================== 枚举测试 ====================

class TestEnums:
    """测试枚举类型"""

    def test_quantization_method(self):
        """测试量化方法枚举"""
        assert QuantizationMethod.NONE.value is None
        assert QuantizationMethod.GPTQ.value == "gptq"
        assert QuantizationMethod.AWQ.value == "awq"
        assert QuantizationMethod.SQUEEZELLM.value == "squeezellm"
        assert QuantizationMethod.FP8.value == "fp8"


# ==================== LoRAAdapter 测试 ====================

class TestLoRAAdapter:
    """测试 LoRA 适配器"""

    @skip_if_no_vllm
    def test_create_adapter(self):
        """测试创建适配器"""
        from vllm_inference import LoRAAdapter

        adapter = LoRAAdapter(
            name="test_adapter",
            adapter_id=1,
            path="/path/to/adapter"
        )

        assert adapter.name == "test_adapter"
        assert adapter.adapter_id == 1
        assert adapter.path == "/path/to/adapter"
        assert adapter.request is not None


# ==================== 便捷函数测试 ====================

class TestConvenienceFunctions:
    """测试便捷函数"""

    @skip_if_no_vllm
    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_create_engine(self):
        """测试 create_engine 函数"""
        from vllm_inference import create_engine

        engine = create_engine(
            model="facebook/opt-125m",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5
        )
        assert engine is not None

    @skip_if_no_vllm
    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_generate_text(self):
        """测试 generate_text 函数"""
        from vllm_inference import generate_text

        outputs = generate_text(
            model="facebook/opt-125m",
            prompts=["Hello, world!"],
            max_tokens=10
        )
        assert len(outputs) == 1


# ==================== VLLMEngine 测试 ====================

class TestVLLMEngine:
    """测试 vLLM 引擎"""

    @skip_if_no_vllm
    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_create_engine(self):
        """测试创建引擎"""
        from vllm_inference import VLLMEngine

        config = EngineConfig(
            model="facebook/opt-125m",
            gpu_memory_utilization=0.5
        )
        engine = VLLMEngine(config)
        assert engine is not None

    @skip_if_no_vllm
    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_generate(self):
        """测试生成"""
        from vllm_inference import VLLMEngine

        config = EngineConfig(
            model="facebook/opt-125m",
            gpu_memory_utilization=0.5
        )
        engine = VLLMEngine(config)

        outputs = engine.generate(["Hello, world!"])
        assert len(outputs) == 1
        assert outputs[0].generated_text is not None

    @skip_if_no_vllm
    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_chat(self):
        """测试聊天"""
        from vllm_inference import VLLMEngine

        config = EngineConfig(
            model="facebook/opt-125m",
            gpu_memory_utilization=0.5
        )
        engine = VLLMEngine(config)

        messages = [
            {"role": "user", "content": "Hello!"}
        ]
        response = engine.chat(messages)
        assert response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
