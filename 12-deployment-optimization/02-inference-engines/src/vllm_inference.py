"""
vLLM 推理模块

提供 vLLM 大语言模型推理的封装和优化功能。

主要功能:
1. VLLMEngine: vLLM 推理引擎封装
2. SamplingConfig: 采样配置
3. 批量推理和流式生成
4. LoRA 适配器支持

注意: vLLM 需要 GPU 支持
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Union
import time

# 检查 vLLM 是否可用
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    LoRARequest = None


class QuantizationMethod(Enum):
    """量化方法"""
    NONE = None
    GPTQ = "gptq"
    AWQ = "awq"
    SQUEEZELLM = "squeezellm"
    FP8 = "fp8"


@dataclass
class SamplingConfig:
    """采样配置"""
    # 温度和 top-p/top-k
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 表示禁用

    # 生成长度
    max_tokens: int = 256
    min_tokens: int = 0

    # 停止条件
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None

    # 重复惩罚
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0

    # 其他选项
    n: int = 1  # 生成数量
    best_of: Optional[int] = None
    use_beam_search: bool = False
    skip_special_tokens: bool = True
    ignore_eos: bool = False

    # 随机种子
    seed: Optional[int] = None

    def to_sampling_params(self) -> "SamplingParams":
        """转换为 vLLM SamplingParams"""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")

        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            n=self.n,
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            skip_special_tokens=self.skip_special_tokens,
            ignore_eos=self.ignore_eos,
            seed=self.seed,
        )


@dataclass
class EngineConfig:
    """vLLM 引擎配置"""
    # 模型配置
    model: str = ""
    tokenizer: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False

    # 并行配置
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # 内存配置
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    swap_space: int = 4  # GB

    # 量化配置
    quantization: QuantizationMethod = QuantizationMethod.NONE
    dtype: str = "auto"  # auto, float16, bfloat16, float32

    # LoRA 配置
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16

    # 投机解码
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 5

    # 其他选项
    seed: int = 0
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = False


@dataclass
class GenerationOutput:
    """生成输出"""
    prompt: str
    generated_text: str
    finish_reason: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float

    # 多个输出 (n > 1)
    all_outputs: Optional[List[str]] = None


class VLLMEngine:
    """vLLM 推理引擎"""

    def __init__(self, config: EngineConfig):
        """
        初始化 vLLM 引擎

        Args:
            config: 引擎配置
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM not available. "
                "Install with: pip install vllm"
            )

        self.config = config

        # 构建 LLM 参数
        llm_kwargs = {
            "model": config.model,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "swap_space": config.swap_space,
            "dtype": config.dtype,
            "seed": config.seed,
            "trust_remote_code": config.trust_remote_code,
            "enforce_eager": config.enforce_eager,
        }

        if config.tokenizer:
            llm_kwargs["tokenizer"] = config.tokenizer
        if config.revision:
            llm_kwargs["revision"] = config.revision
        if config.max_model_len:
            llm_kwargs["max_model_len"] = config.max_model_len
        if config.quantization != QuantizationMethod.NONE:
            llm_kwargs["quantization"] = config.quantization.value
        if config.enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = config.max_loras
            llm_kwargs["max_lora_rank"] = config.max_lora_rank
        if config.speculative_model:
            llm_kwargs["speculative_model"] = config.speculative_model
            llm_kwargs["num_speculative_tokens"] = config.num_speculative_tokens

        # 创建 LLM 实例
        self.llm = LLM(**llm_kwargs)

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_config: Optional[SamplingConfig] = None,
        lora_request: Optional["LoRARequest"] = None
    ) -> List[GenerationOutput]:
        """
        生成文本

        Args:
            prompts: 输入提示 (单个或列表)
            sampling_config: 采样配置
            lora_request: LoRA 请求

        Returns:
            生成输出列表
        """
        # 处理单个输入
        if isinstance(prompts, str):
            prompts = [prompts]

        # 获取采样参数
        sampling_config = sampling_config or SamplingConfig()
        sampling_params = sampling_config.to_sampling_params()

        # 执行生成
        start_time = time.perf_counter()
        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request
        )
        total_time = (time.perf_counter() - start_time) * 1000

        # 处理输出
        results = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason

            # 统计 token 数量
            prompt_tokens = len(output.prompt_token_ids)
            generated_tokens = len(output.outputs[0].token_ids)

            # 多个输出
            all_outputs = None
            if len(output.outputs) > 1:
                all_outputs = [o.text for o in output.outputs]

            results.append(GenerationOutput(
                prompt=prompt,
                generated_text=generated_text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                latency_ms=total_time / len(prompts),
                all_outputs=all_outputs
            ))

        return results

    def generate_single(
        self,
        prompt: str,
        sampling_config: Optional[SamplingConfig] = None
    ) -> str:
        """
        生成单个文本

        Args:
            prompt: 输入提示
            sampling_config: 采样配置

        Returns:
            生成的文本
        """
        outputs = self.generate([prompt], sampling_config)
        return outputs[0].generated_text

    def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_config: Optional[SamplingConfig] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        聊天接口

        Args:
            messages: 消息列表 [{"role": "user/assistant", "content": "..."}]
            sampling_config: 采样配置
            system_prompt: 系统提示

        Returns:
            助手回复
        """
        # 构建聊天提示
        prompt = self._format_chat_prompt(messages, system_prompt)
        return self.generate_single(prompt, sampling_config)

    def _format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        格式化聊天提示

        Args:
            messages: 消息列表
            system_prompt: 系统提示

        Returns:
            格式化的提示
        """
        # 简单的聊天格式 (实际应根据模型调整)
        parts = []

        if system_prompt:
            parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            parts.append(f"{role}: {content}")

        parts.append("Assistant:")

        return "\n".join(parts)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model": self.config.model,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "quantization": self.config.quantization.value,
            "dtype": self.config.dtype,
        }


class LoRAAdapter:
    """LoRA 适配器管理"""

    def __init__(self, name: str, adapter_id: int, path: str):
        """
        初始化 LoRA 适配器

        Args:
            name: 适配器名称
            adapter_id: 适配器 ID
            path: 适配器路径
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")

        self.name = name
        self.adapter_id = adapter_id
        self.path = path
        self._request = LoRARequest(name, adapter_id, path)

    @property
    def request(self) -> "LoRARequest":
        """获取 LoRA 请求对象"""
        return self._request


# 便捷函数
def create_engine(
    model: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    quantization: Optional[str] = None,
    **kwargs
) -> VLLMEngine:
    """
    创建 vLLM 引擎

    Args:
        model: 模型名称或路径
        tensor_parallel_size: 张量并行大小
        gpu_memory_utilization: GPU 内存利用率
        quantization: 量化方法
        **kwargs: 其他配置参数

    Returns:
        vLLM 引擎
    """
    quant_method = QuantizationMethod.NONE
    if quantization:
        quant_method = QuantizationMethod(quantization)

    config = EngineConfig(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quant_method,
        **kwargs
    )

    return VLLMEngine(config)


def generate_text(
    model: str,
    prompts: Union[str, List[str]],
    max_tokens: int = 256,
    temperature: float = 1.0,
    **kwargs
) -> List[str]:
    """
    快速生成文本

    Args:
        model: 模型名称或路径
        prompts: 输入提示
        max_tokens: 最大生成长度
        temperature: 温度
        **kwargs: 其他引擎配置

    Returns:
        生成的文本列表
    """
    engine = create_engine(model, **kwargs)
    sampling_config = SamplingConfig(
        max_tokens=max_tokens,
        temperature=temperature
    )

    outputs = engine.generate(prompts, sampling_config)
    return [o.generated_text for o in outputs]
