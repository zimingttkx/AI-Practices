"""
模型导出 (Model Export)

本模块实现深度学习模型的导出功能，包括：
- ONNX 导出
- TorchScript 导出 (脚本化和追踪)
- 模型优化和验证

=== 导出格式对比 ===

1. ONNX (Open Neural Network Exchange)
   - 跨框架兼容
   - 支持多种推理引擎 (ONNX Runtime, TensorRT, etc.)
   - 适合生产部署

2. TorchScript
   - PyTorch 原生格式
   - 支持 Python 控制流
   - 适合 PyTorch 生态系统

=== 参考文献 ===

1. ONNX: https://onnx.ai/
2. TorchScript: https://pytorch.org/docs/stable/jit.html
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Any, Sequence
from enum import Enum

import torch
import torch.nn as nn


class ExportFormat(Enum):
    """导出格式"""
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TORCHSCRIPT_TRACE = "torchscript_trace"
    TORCHSCRIPT_SCRIPT = "torchscript_script"


@dataclass
class ExportConfig:
    """导出配置"""

    # 导出格式
    format: ExportFormat = ExportFormat.ONNX

    # ONNX 配置
    opset_version: int = 14
    do_constant_folding: bool = True
    export_params: bool = True

    # 输入输出名称
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])

    # 动态维度
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None

    # 验证配置
    verify_export: bool = True
    atol: float = 1e-5
    rtol: float = 1e-5

    # 优化配置
    optimize: bool = True


class ONNXExporter:
    """
    ONNX 导出器

    将 PyTorch 模型导出为 ONNX 格式。
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig(format=ExportFormat.ONNX)

    def export(
        self,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: Optional[int] = None
    ) -> str:
        """
        导出模型为 ONNX 格式

        Args:
            model: PyTorch 模型
            dummy_input: 示例输入
            output_path: 输出路径
            input_names: 输入名称
            output_names: 输出名称
            dynamic_axes: 动态维度
            opset_version: ONNX opset 版本

        Returns:
            导出文件路径
        """
        model.eval()

        # 使用配置或参数
        input_names = input_names or self.config.input_names
        output_names = output_names or self.config.output_names
        dynamic_axes = dynamic_axes or self.config.dynamic_axes
        opset_version = opset_version or self.config.opset_version

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # 导出
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=self.config.do_constant_folding,
                export_params=self.config.export_params,
            )

        # 验证导出
        if self.config.verify_export:
            self._verify_export(model, dummy_input, output_path)

        # 优化
        if self.config.optimize:
            self._optimize_onnx(output_path)

        return output_path

    def _verify_export(
        self,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        onnx_path: str
    ):
        """验证 ONNX 导出"""
        try:
            import onnx
            import onnxruntime as ort

            # 检查 ONNX 模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # 使用 ONNX Runtime 验证
            ort_session = ort.InferenceSession(onnx_path)

            # 准备输入
            if isinstance(dummy_input, torch.Tensor):
                inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            else:
                inputs = {
                    ort_session.get_inputs()[i].name: inp.numpy()
                    for i, inp in enumerate(dummy_input)
                }

            # 运行推理
            ort_outputs = ort_session.run(None, inputs)

            # 比较输出
            with torch.no_grad():
                torch_outputs = model(dummy_input)
                if isinstance(torch_outputs, torch.Tensor):
                    torch_outputs = [torch_outputs]

            for i, (ort_out, torch_out) in enumerate(zip(ort_outputs, torch_outputs)):
                if not torch.allclose(
                    torch.tensor(ort_out),
                    torch_out,
                    atol=self.config.atol,
                    rtol=self.config.rtol
                ):
                    warnings.warn(f"输出 {i} 验证失败: ONNX 和 PyTorch 输出不一致")

            print("ONNX 导出验证通过")

        except ImportError:
            warnings.warn("未安装 onnx 或 onnxruntime，跳过验证")
        except Exception as e:
            warnings.warn(f"ONNX 验证失败: {e}")

    def _optimize_onnx(self, onnx_path: str):
        """优化 ONNX 模型"""
        try:
            import onnx
            from onnx import optimizer

            model = onnx.load(onnx_path)

            # 应用优化
            passes = [
                "eliminate_identity",
                "eliminate_nop_transpose",
                "eliminate_nop_pad",
                "eliminate_unused_initializer",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
                "fuse_bn_into_conv",
            ]

            optimized_model = optimizer.optimize(model, passes)
            onnx.save(optimized_model, onnx_path)
            print("ONNX 模型优化完成")

        except ImportError:
            pass  # 优化是可选的
        except Exception as e:
            warnings.warn(f"ONNX 优化失败: {e}")

    def get_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        获取 ONNX 模型信息

        Args:
            onnx_path: ONNX 模型路径

        Returns:
            模型信息字典
        """
        try:
            import onnx

            model = onnx.load(onnx_path)

            # 获取输入信息
            inputs = []
            for inp in model.graph.input:
                shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                inputs.append({
                    "name": inp.name,
                    "shape": shape,
                    "dtype": inp.type.tensor_type.elem_type
                })

            # 获取输出信息
            outputs = []
            for out in model.graph.output:
                shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
                outputs.append({
                    "name": out.name,
                    "shape": shape,
                    "dtype": out.type.tensor_type.elem_type
                })

            # 计算参数量
            num_params = sum(
                len(init.raw_data) // 4  # 假设 float32
                for init in model.graph.initializer
            )

            return {
                "inputs": inputs,
                "outputs": outputs,
                "num_params": num_params,
                "opset_version": model.opset_import[0].version,
                "ir_version": model.ir_version
            }

        except ImportError:
            return {"error": "onnx not installed"}
        except Exception as e:
            return {"error": str(e)}


class TorchScriptExporter:
    """
    TorchScript 导出器

    将 PyTorch 模型导出为 TorchScript 格式。
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig(format=ExportFormat.TORCHSCRIPT)

    def export_script(
        self,
        model: nn.Module,
        output_path: str,
        optimize: bool = True
    ) -> str:
        """
        使用脚本化导出模型

        脚本化会分析 Python 代码并转换为 TorchScript。
        支持控制流 (if/for/while)。

        Args:
            model: PyTorch 模型
            output_path: 输出路径
            optimize: 是否优化

        Returns:
            导出文件路径
        """
        model.eval()

        # 脚本化
        scripted_model = torch.jit.script(model)

        # 优化
        if optimize:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        scripted_model.save(output_path)

        print(f"TorchScript (script) 导出完成: {output_path}")
        return output_path

    def export_trace(
        self,
        model: nn.Module,
        example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output_path: str,
        optimize: bool = True,
        strict: bool = True
    ) -> str:
        """
        使用追踪导出模型

        追踪会记录模型的执行路径。
        不支持动态控制流。

        Args:
            model: PyTorch 模型
            example_input: 示例输入
            output_path: 输出路径
            optimize: 是否优化
            strict: 是否严格模式

        Returns:
            导出文件路径
        """
        model.eval()

        # 追踪
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model, example_input, strict=strict
            )

        # 优化
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)

        # 保存
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        traced_model.save(output_path)

        # 验证
        if self.config.verify_export:
            self._verify_trace(model, traced_model, example_input)

        print(f"TorchScript (trace) 导出完成: {output_path}")
        return output_path

    def _verify_trace(
        self,
        original_model: nn.Module,
        traced_model: torch.jit.ScriptModule,
        example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ):
        """验证追踪导出"""
        with torch.no_grad():
            original_output = original_model(example_input)
            traced_output = traced_model(example_input)

            if isinstance(original_output, torch.Tensor):
                original_output = [original_output]
                traced_output = [traced_output]

            for i, (orig, traced) in enumerate(zip(original_output, traced_output)):
                if not torch.allclose(
                    orig, traced,
                    atol=self.config.atol,
                    rtol=self.config.rtol
                ):
                    warnings.warn(f"输出 {i} 验证失败")

        print("TorchScript 导出验证通过")

    def load(self, model_path: str) -> torch.jit.ScriptModule:
        """
        加载 TorchScript 模型

        Args:
            model_path: 模型路径

        Returns:
            TorchScript 模型
        """
        return torch.jit.load(model_path)

    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        获取 TorchScript 模型信息

        Args:
            model_path: 模型路径

        Returns:
            模型信息字典
        """
        try:
            model = torch.jit.load(model_path)

            # 获取代码
            code = model.code

            # 计算参数量
            num_params = sum(p.numel() for p in model.parameters())

            return {
                "code": code,
                "num_params": num_params,
                "is_training": model.training
            }

        except Exception as e:
            return {"error": str(e)}


class ModelAnalyzer:
    """
    模型分析器

    分析模型的结构、参数量和计算量。
    """

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        统计模型参数量

        Args:
            model: PyTorch 模型

        Returns:
            参数统计字典
        """
        total_params = 0
        trainable_params = 0
        layer_params = {}

        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            layer_params[name] = num_params

        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params,
            "layers": layer_params
        }

    @staticmethod
    def estimate_model_size(model: nn.Module, dtype: torch.dtype = torch.float32) -> Dict[str, float]:
        """
        估算模型大小

        Args:
            model: PyTorch 模型
            dtype: 数据类型

        Returns:
            大小估算字典 (MB)
        """
        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.int32: 4,
        }

        param_bytes = bytes_per_param.get(dtype, 4)
        total_params = sum(p.numel() for p in model.parameters())
        buffer_params = sum(b.numel() for b in model.buffers())

        param_size_mb = (total_params * param_bytes) / (1024 * 1024)
        buffer_size_mb = (buffer_params * 4) / (1024 * 1024)  # 假设 buffer 是 float32

        return {
            "params_mb": param_size_mb,
            "buffers_mb": buffer_size_mb,
            "total_mb": param_size_mb + buffer_size_mb
        }

    @staticmethod
    def profile_inference(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        分析推理性能

        Args:
            model: PyTorch 模型
            input_shape: 输入形状
            num_runs: 运行次数
            warmup_runs: 预热次数
            device: 设备

        Returns:
            性能统计字典
        """
        import time

        model = model.to(device)
        model.eval()

        dummy_input = torch.randn(*input_shape).to(device)

        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

        # 同步 (GPU)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

        # 计时
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms

        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
            "throughput": 1000 / (sum(times) / len(times))  # samples/sec
        }


def export_to_onnx(
    model: nn.Module,
    dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_path: str,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 14,
    verify: bool = True
) -> str:
    """
    导出模型为 ONNX 格式的便捷函数

    Args:
        model: PyTorch 模型
        dummy_input: 示例输入
        output_path: 输出路径
        input_names: 输入名称
        output_names: 输出名称
        dynamic_axes: 动态维度
        opset_version: ONNX opset 版本
        verify: 是否验证导出

    Returns:
        导出文件路径
    """
    config = ExportConfig(
        format=ExportFormat.ONNX,
        opset_version=opset_version,
        verify_export=verify,
        input_names=input_names or ["input"],
        output_names=output_names or ["output"],
        dynamic_axes=dynamic_axes
    )

    exporter = ONNXExporter(config)
    return exporter.export(model, dummy_input, output_path)


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    method: str = "trace",
    example_input: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    optimize: bool = True
) -> str:
    """
    导出模型为 TorchScript 格式的便捷函数

    Args:
        model: PyTorch 模型
        output_path: 输出路径
        method: 导出方法 ("trace" 或 "script")
        example_input: 示例输入 (trace 方法需要)
        optimize: 是否优化

    Returns:
        导出文件路径
    """
    exporter = TorchScriptExporter()

    if method == "script":
        return exporter.export_script(model, output_path, optimize)
    elif method == "trace":
        if example_input is None:
            raise ValueError("trace 方法需要提供 example_input")
        return exporter.export_trace(model, example_input, output_path, optimize)
    else:
        raise ValueError(f"未知的导出方法: {method}")


def compare_model_outputs(
    model1: nn.Module,
    model2: nn.Module,
    test_inputs: List[torch.Tensor],
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> Dict[str, Any]:
    """
    比较两个模型的输出

    Args:
        model1: 第一个模型
        model2: 第二个模型
        test_inputs: 测试输入列表
        atol: 绝对容差
        rtol: 相对容差

    Returns:
        比较结果字典
    """
    model1.eval()
    model2.eval()

    results = {
        "all_match": True,
        "num_tests": len(test_inputs),
        "max_diff": 0.0,
        "mean_diff": 0.0,
        "mismatches": []
    }

    total_diff = 0.0

    with torch.no_grad():
        for i, inp in enumerate(test_inputs):
            out1 = model1(inp)
            out2 = model2(inp)

            if isinstance(out1, torch.Tensor):
                diff = (out1 - out2).abs().max().item()
                total_diff += diff
                results["max_diff"] = max(results["max_diff"], diff)

                if not torch.allclose(out1, out2, atol=atol, rtol=rtol):
                    results["all_match"] = False
                    results["mismatches"].append(i)

    results["mean_diff"] = total_diff / len(test_inputs) if test_inputs else 0.0

    return results
