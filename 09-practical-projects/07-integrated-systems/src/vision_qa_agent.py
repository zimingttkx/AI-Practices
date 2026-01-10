"""
视觉问答智能体。

基于ReAct框架实现图像理解和问答能力。

核心组件:
    - VisionAction: 视觉动作
    - VisionObservation: 观察结果
    - VisionQAAgent: 视觉问答智能体
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .multimodal_retriever import (
    CLIPEncoder,
    MultimodalDocument,
    MultimodalRetriever,
    SearchResult,
)


class ActionType(Enum):
    """动作类型。"""
    SEARCH = "search"
    DESCRIBE = "describe"
    COMPARE = "compare"
    LOCATE = "locate"
    COUNT = "count"
    ANSWER = "answer"
    THINK = "think"


@dataclass
class VisionAction:
    """视觉动作。"""
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    
    def __repr__(self) -> str:
        return f"VisionAction({self.action_type.value}, params={self.params})"


@dataclass
class VisionObservation:
    """观察结果。"""
    action: VisionAction
    result: Any
    success: bool = True
    error: Optional[str] = None
    
    def __repr__(self) -> str:
        status = "ok" if self.success else "error"
        return f"VisionObservation({self.action.action_type.value}, {status})"


@dataclass
class AgentStep:
    """智能体执行步骤。"""
    step_num: int
    thought: str
    action: VisionAction
    observation: VisionObservation


@dataclass
class AgentResult:
    """智能体执行结果。"""
    question: str
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    sources: List[MultimodalDocument] = field(default_factory=list)
    success: bool = True
    
    @property
    def num_steps(self) -> int:
        return len(self.steps)


class VisionQAAgent:
    """视觉问答智能体。
    
    使用ReAct框架进行多步推理:
        1. Thought: 分析当前状态
        2. Action: 选择并执行动作
        3. Observation: 获取结果
        4. 重复直到得出答案
    
    示例:
        >>> agent = VisionQAAgent(retriever=retriever)
        >>> result = agent.answer("图中有几只猫?", image=cat_image)
    """
    
    def __init__(
        self,
        retriever: Optional[MultimodalRetriever] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        max_steps: int = 10,
        verbose: bool = False,
    ) -> None:
        self.retriever = retriever or MultimodalRetriever()
        self.llm_func = llm_func or self._mock_llm
        self.max_steps = max_steps
        self.verbose = verbose
        
        self._action_handlers: Dict[ActionType, Callable] = {
            ActionType.SEARCH: self._handle_search,
            ActionType.DESCRIBE: self._handle_describe,
            ActionType.COMPARE: self._handle_compare,
            ActionType.LOCATE: self._handle_locate,
            ActionType.COUNT: self._handle_count,
            ActionType.ANSWER: self._handle_answer,
            ActionType.THINK: self._handle_think,
        }
        
        self._current_image: Optional[np.ndarray] = None
        self._context: List[str] = []
    
    def _mock_llm(self, prompt: str) -> str:
        """模拟LLM响应。"""
        if "describe" in prompt.lower():
            return "THOUGHT: 需要描述图像内容\nACTION: describe\nPARAMS: {}"
        elif "search" in prompt.lower():
            return "THOUGHT: 需要搜索相关信息\nACTION: search\nPARAMS: {\"query\": \"relevant content\"}"
        elif "count" in prompt.lower():
            return "THOUGHT: 需要计数\nACTION: count\nPARAMS: {\"target\": \"objects\"}"
        else:
            return "THOUGHT: 已收集足够信息\nACTION: answer\nPARAMS: {\"answer\": \"基于分析，答案是...\"}"
    
    def answer(
        self,
        question: str,
        image: Optional[np.ndarray] = None,
        context: Optional[List[str]] = None,
    ) -> AgentResult:
        """回答视觉问题。
        
        参数:
            question: 问题
            image: 图像 (可选)
            context: 额外上下文
            
        返回:
            智能体结果
        """
        self._current_image = image
        self._context = context or []
        
        steps: List[AgentStep] = []
        sources: List[MultimodalDocument] = []
        final_answer = ""
        
        for step_num in range(1, self.max_steps + 1):
            # 构建提示
            prompt = self._build_prompt(question, steps)
            
            # 获取LLM响应
            response = self.llm_func(prompt)
            
            # 解析动作
            thought, action = self._parse_response(response)
            
            # 执行动作
            observation = self._execute_action(action)
            
            # 记录步骤
            step = AgentStep(
                step_num=step_num,
                thought=thought,
                action=action,
                observation=observation,
            )
            steps.append(step)
            
            if self.verbose:
                print(f"Step {step_num}: {thought}")
                print(f"  Action: {action}")
                print(f"  Observation: {observation}")
            
            # 收集来源
            if action.action_type == ActionType.SEARCH and observation.success:
                if isinstance(observation.result, list):
                    for r in observation.result:
                        if isinstance(r, SearchResult):
                            sources.append(r.document)
            
            # 检查是否完成
            if action.action_type == ActionType.ANSWER:
                final_answer = action.params.get("answer", str(observation.result))
                break
        
        return AgentResult(
            question=question,
            answer=final_answer,
            steps=steps,
            sources=sources,
            success=bool(final_answer),
        )
    
    def _build_prompt(self, question: str, steps: List[AgentStep]) -> str:
        """构建提示。"""
        prompt_parts = [
            "你是一个视觉问答智能体，使用ReAct框架进行推理。",
            "",
            f"问题: {question}",
            "",
        ]
        
        if self._current_image is not None:
            prompt_parts.append(f"[图像已提供: shape={self._current_image.shape}]")
            prompt_parts.append("")
        
        if self._context:
            prompt_parts.append("上下文:")
            for ctx in self._context:
                prompt_parts.append(f"  - {ctx}")
            prompt_parts.append("")
        
        if steps:
            prompt_parts.append("历史步骤:")
            for step in steps:
                prompt_parts.append(f"  Step {step.step_num}:")
                prompt_parts.append(f"    Thought: {step.thought}")
                prompt_parts.append(f"    Action: {step.action.action_type.value}")
                prompt_parts.append(f"    Observation: {step.observation.result}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "可用动作: search, describe, compare, locate, count, answer, think",
            "",
            "请输出:",
            "THOUGHT: <你的思考>",
            "ACTION: <动作名称>",
            "PARAMS: <JSON参数>",
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_response(self, response: str) -> Tuple[str, VisionAction]:
        """解析LLM响应。"""
        thought = ""
        action_type = ActionType.THINK
        params = {}
        
        # 提取THOUGHT
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # 提取ACTION
        action_match = re.search(r"ACTION:\s*(\w+)", response)
        if action_match:
            action_name = action_match.group(1).lower()
            try:
                action_type = ActionType(action_name)
            except ValueError:
                action_type = ActionType.THINK
        
        # 提取PARAMS
        params_match = re.search(r"PARAMS:\s*(\{.*?\})", response, re.DOTALL)
        if params_match:
            try:
                import json
                params = json.loads(params_match.group(1))
            except (json.JSONDecodeError, ValueError):
                params = {}
        
        return thought, VisionAction(
            action_type=action_type,
            params=params,
            reasoning=thought,
        )
    
    def _execute_action(self, action: VisionAction) -> VisionObservation:
        """执行动作。"""
        handler = self._action_handlers.get(action.action_type)
        if handler is None:
            return VisionObservation(
                action=action,
                result=None,
                success=False,
                error=f"未知动作: {action.action_type}",
            )
        
        try:
            result = handler(action.params)
            return VisionObservation(
                action=action,
                result=result,
                success=True,
            )
        except Exception as e:
            return VisionObservation(
                action=action,
                result=None,
                success=False,
                error=str(e),
            )
    
    def _handle_search(self, params: Dict[str, Any]) -> List[SearchResult]:
        """处理搜索动作。"""
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        
        if self._current_image is not None and not query:
            return self.retriever.search_by_image(self._current_image, top_k=top_k)
        elif query:
            return self.retriever.search_by_text(query, top_k=top_k)
        else:
            return []
    
    def _handle_describe(self, params: Dict[str, Any]) -> str:
        """处理描述动作。"""
        if self._current_image is None:
            return "没有可描述的图像"
        
        shape = self._current_image.shape
        if len(shape) == 2:
            return f"灰度图像，尺寸 {shape[0]}x{shape[1]}"
        else:
            return f"彩色图像，尺寸 {shape[0]}x{shape[1]}，{shape[2]}通道"
    
    def _handle_compare(self, params: Dict[str, Any]) -> str:
        """处理比较动作。"""
        target1 = params.get("target1", "")
        target2 = params.get("target2", "")
        return f"比较 '{target1}' 和 '{target2}': 需要更多上下文信息"
    
    def _handle_locate(self, params: Dict[str, Any]) -> str:
        """处理定位动作。"""
        target = params.get("target", "object")
        return f"定位 '{target}': 需要目标检测模型支持"
    
    def _handle_count(self, params: Dict[str, Any]) -> str:
        """处理计数动作。"""
        target = params.get("target", "objects")
        return f"计数 '{target}': 需要目标检测模型支持"
    
    def _handle_answer(self, params: Dict[str, Any]) -> str:
        """处理回答动作。"""
        return params.get("answer", "无法确定答案")
    
    def _handle_think(self, params: Dict[str, Any]) -> str:
        """处理思考动作。"""
        return params.get("thought", "继续分析...")
    
    def register_action(
        self,
        action_type: ActionType,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """注册自定义动作处理器。"""
        self._action_handlers[action_type] = handler
    
    def __repr__(self) -> str:
        return f"VisionQAAgent(max_steps={self.max_steps})"
