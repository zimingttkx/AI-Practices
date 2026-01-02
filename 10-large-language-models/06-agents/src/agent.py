"""
Agent模块 (Agent Module)

============================================================
核心思想 (Core Idea)
============================================================
AI Agent是能够自主规划、决策和执行任务的智能系统。通过结合LLM的
推理能力和外部工具的执行能力，Agent可以完成复杂的多步骤任务。

============================================================
数学基础 (Mathematical Foundation)
============================================================
Agent决策过程可形式化为马尔可夫决策过程(MDP)：

    π(a|s) = LLM(action | state, history, tools)

ReAct模式的状态转移：
    s_{t+1} = (s_t, thought_t, action_t, observation_t)

目标函数：
    max_π E[Σ_t γ^t R(s_t, a_t)]

其中R为任务完成奖励，γ为折扣因子。

============================================================
算法流程 (Algorithm Flow)
============================================================
ReAct循环：
    1. Thought: 分析当前状态，思考下一步
    2. Action: 选择要执行的工具
    3. Observation: 获取工具执行结果
    4. 重复1-3直到任务完成或达到最大迭代

Plan-and-Execute：
    1. Plan: 制定完整执行计划
    2. Execute: 逐步执行计划中的步骤
    3. Replan: 根据执行结果调整计划

============================================================
参考文献 (References)
============================================================
[1] Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting
    in Language Models. arXiv:2210.03629.
[2] Wang, L., et al. (2023). Plan-and-Solve Prompting. ACL 2023.
[3] Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal
    Reinforcement Learning. arXiv:2303.11366.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from .tools import Tool, ToolRegistry, ToolResult
    from .memory import Message, MessageRole, ConversationMemory, BufferMemory
except ImportError:
    from tools import Tool, ToolRegistry, ToolResult
    from memory import Message, MessageRole, ConversationMemory, BufferMemory


__all__ = [
    "AgentConfig",
    "AgentState",
    "AgentAction",
    "AgentFinish",
    "BaseAgent",
    "ReActAgent",
    "ToolCallingAgent",
    "PlanAndExecuteAgent",
]


class AgentStatus(Enum):
    """Agent状态枚举。"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Agent配置。

    参数：
        max_iterations: 最大迭代次数
        max_execution_time: 最大执行时间（秒）
        early_stopping: 是否启用早停
        return_intermediate_steps: 是否返回中间步骤
        handle_parsing_errors: 是否处理解析错误
    """
    max_iterations: int = 10
    max_execution_time: float = 120.0
    early_stopping: bool = True
    return_intermediate_steps: bool = True
    handle_parsing_errors: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations必须为正数，得到 {self.max_iterations}")
        if self.max_execution_time <= 0:
            raise ValueError(f"max_execution_time必须为正数，得到 {self.max_execution_time}")


@dataclass
class AgentState:
    """Agent状态。

    参数：
        status: 当前状态
        iteration: 当前迭代次数
        intermediate_steps: 中间步骤
        final_output: 最终输出
        error: 错误信息
    """
    status: AgentStatus = AgentStatus.IDLE
    iteration: int = 0
    intermediate_steps: List[Tuple[Any, str]] = field(default_factory=list)
    final_output: Optional[str] = None
    error: Optional[str] = None

    def __repr__(self) -> str:
        return f"AgentState(status={self.status.value}, iteration={self.iteration})"


@dataclass
class AgentAction:
    """Agent动作。

    表示Agent决定执行的工具调用。

    参数：
        tool: 工具名称
        tool_input: 工具输入
        log: 思考日志
    """
    tool: str
    tool_input: Dict[str, Any]
    log: str = ""

    def __repr__(self) -> str:
        return f"AgentAction(tool='{self.tool}', input={self.tool_input})"


@dataclass
class AgentFinish:
    """Agent完成。

    表示Agent决定结束并返回最终答案。

    参数：
        output: 最终输出
        log: 思考日志
    """
    output: str
    log: str = ""

    def __repr__(self) -> str:
        preview = self.output[:50] + "..." if len(self.output) > 50 else self.output
        return f"AgentFinish(output='{preview}')"


class BaseAgent(ABC):
    """Agent基类。

    所有Agent必须继承此类并实现plan方法。

    Agent工作流程：
        1. 接收用户输入
        2. 思考并决定动作
        3. 执行动作（调用工具）
        4. 观察结果
        5. 重复2-4直到完成

    示例：
        >>> agent = MyAgent(tools=[calc, search])
        >>> result = agent.run("计算 2+2")
    """

    def __init__(
        self,
        tools: List[Tool],
        config: Optional[AgentConfig] = None,
        memory: Optional[ConversationMemory] = None,
        llm: Optional[Callable[[List[Dict]], str]] = None,
    ) -> None:
        """初始化Agent。

        参数：
            tools: 可用工具列表
            config: Agent配置
            memory: 对话记忆
            llm: LLM调用函数
        """
        self.config = config or AgentConfig()
        self.memory = memory or BufferMemory()
        self.llm = llm or self._default_llm
        
        # 注册工具
        self.tool_registry = ToolRegistry()
        for tool in tools:
            self.tool_registry.register(tool)
        
        self.state = AgentState()

    def _default_llm(self, messages: List[Dict]) -> str:
        """默认LLM（模拟）。"""
        return "[模拟LLM响应] 这是一个模拟的LLM响应。实际应用中请接入真实的LLM API。"

    @abstractmethod
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """规划下一步动作。

        参数：
            intermediate_steps: 中间步骤（动作，观察）列表
            **kwargs: 额外参数

        返回：
            AgentAction（继续执行）或 AgentFinish（结束）
        """
        pass

    def run(self, input_text: str, **kwargs: Any) -> str:
        """运行Agent。

        参数：
            input_text: 用户输入
            **kwargs: 额外参数

        返回：
            最终输出
        """
        # 初始化状态
        self.state = AgentState(status=AgentStatus.THINKING)
        self.memory.add_user_message(input_text)
        
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        
        # 迭代执行
        while self.state.iteration < self.config.max_iterations:
            self.state.iteration += 1
            
            try:
                # 规划
                output = self.plan(intermediate_steps, input=input_text, **kwargs)
                
                # 检查是否完成
                if isinstance(output, AgentFinish):
                    self.state.status = AgentStatus.FINISHED
                    self.state.final_output = output.output
                    self.memory.add_assistant_message(output.output)
                    return output.output
                
                # 执行动作
                self.state.status = AgentStatus.ACTING
                action = output
                observation = self._execute_action(action)
                intermediate_steps.append((action, observation))
                self.state.intermediate_steps = intermediate_steps
                
            except Exception as e:
                if self.config.handle_parsing_errors:
                    observation = f"错误: {str(e)}"
                    intermediate_steps.append((
                        AgentAction(tool="error", tool_input={}, log=str(e)),
                        observation,
                    ))
                else:
                    self.state.status = AgentStatus.ERROR
                    self.state.error = str(e)
                    raise
        
        # 达到最大迭代
        self.state.status = AgentStatus.FINISHED
        final_output = "达到最大迭代次数，无法完成任务。"
        self.state.final_output = final_output
        return final_output

    def _execute_action(self, action: AgentAction) -> str:
        """执行动作。"""
        tool = self.tool_registry.get(action.tool)
        if tool is None:
            return f"错误: 未找到工具 '{action.tool}'"
        
        result = tool.run(**action.tool_input)
        if result.is_success:
            return result.output
        else:
            return f"工具执行错误: {result.error}"

    @property
    def tools(self) -> List[Tool]:
        """获取所有工具。"""
        return self.tool_registry.get_all_tools()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tools={self.tool_registry.list_tools()})"


class ReActAgent(BaseAgent):
    """ReAct Agent实现。

    基于论文 "ReAct: Synergizing Reasoning and Acting in Language Models"。

    ReAct模式：
        Thought: 思考当前情况
        Action: 选择要执行的动作
        Action Input: 动作的输入参数
        Observation: 观察动作结果
        ... (重复)
        Final Answer: 最终答案

    示例：
        >>> agent = ReActAgent(tools=[calc, search])
        >>> result = agent.run("北京的人口是多少？")
    """

    # ReAct提示模板
    REACT_PROMPT = """你是一个有帮助的AI助手，可以使用以下工具来回答问题：

{tools}

使用以下格式：

Question: 需要回答的问题
Thought: 思考应该怎么做
Action: 要使用的工具名称，必须是 [{tool_names}] 之一
Action Input: 工具的输入参数（JSON格式）
Observation: 工具返回的结果
... (可以重复 Thought/Action/Action Input/Observation 多次)
Thought: 我现在知道最终答案了
Final Answer: 问题的最终答案

开始！

Question: {input}
{agent_scratchpad}"""

    def _build_scratchpad(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> str:
        """构建思考过程记录。"""
        scratchpad = ""
        for action, observation in intermediate_steps:
            scratchpad += f"Thought: {action.log}\n"
            scratchpad += f"Action: {action.tool}\n"
            scratchpad += f"Action Input: {json.dumps(action.tool_input, ensure_ascii=False)}\n"
            scratchpad += f"Observation: {observation}\n"
        return scratchpad

    def _parse_output(self, text: str) -> Union[AgentAction, AgentFinish]:
        """解析LLM输出。"""
        # 检查是否有最终答案
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            return AgentFinish(output=answer, log=text)

        # 解析动作
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)
        thought_match = re.search(r"Thought:\s*(.+?)(?:\nAction|$)", text, re.DOTALL)

        if not action_match:
            return AgentFinish(output=text, log=text)

        action = action_match.group(1).strip()
        thought = thought_match.group(1).strip() if thought_match else ""

        # 解析输入
        tool_input = {}
        if input_match:
            input_str = input_match.group(1).strip()
            try:
                tool_input = json.loads(input_str)
            except json.JSONDecodeError:
                # 尝试作为简单字符串
                tool_input = {"input": input_str}

        return AgentAction(tool=action, tool_input=tool_input, log=thought)

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """规划下一步动作。"""
        # 构建提示
        tools_desc = self.tool_registry.get_tools_description()
        tool_names = ", ".join(self.tool_registry.list_tools())
        scratchpad = self._build_scratchpad(intermediate_steps)

        prompt = self.REACT_PROMPT.format(
            tools=tools_desc,
            tool_names=tool_names,
            input=kwargs.get("input", ""),
            agent_scratchpad=scratchpad,
        )

        # 调用LLM
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages)

        # 解析输出
        return self._parse_output(response)


class ToolCallingAgent(BaseAgent):
    """工具调用Agent。

    使用OpenAI风格的函数调用来选择和执行工具。

    特点：
        - 结构化的工具调用
        - 支持并行工具调用
        - 更可靠的参数解析

    示例：
        >>> agent = ToolCallingAgent(tools=[calc, search])
        >>> result = agent.run("计算 sqrt(144)")
    """

    SYSTEM_PROMPT = """你是一个有帮助的AI助手。你可以使用提供的工具来帮助用户完成任务。
当需要使用工具时，请以JSON格式输出工具调用：
{"tool": "工具名", "arguments": {"参数名": "参数值"}}

如果不需要使用工具，直接回答用户的问题。"""

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """规划下一步动作。"""
        # 构建消息
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "system", "content": f"可用工具:\n{self.tool_registry.get_tools_description()}"},
        ]

        # 添加用户输入
        user_input = kwargs.get("input", "")
        messages.append({"role": "user", "content": user_input})

        # 添加中间步骤
        for action, observation in intermediate_steps:
            messages.append({
                "role": "assistant",
                "content": f"调用工具: {action.tool}({action.tool_input})",
            })
            messages.append({
                "role": "user",
                "content": f"工具结果: {observation}",
            })

        # 调用LLM
        response = self.llm(messages)

        # 解析响应
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Union[AgentAction, AgentFinish]:
        """解析LLM响应。"""
        # 尝试解析JSON工具调用
        try:
            # 尝试直接解析整个响应为JSON
            data = json.loads(response.strip())
            if "tool" in data:
                return AgentAction(
                    tool=data.get("tool", ""),
                    tool_input=data.get("arguments", {}),
                    log=response,
                )
        except json.JSONDecodeError:
            pass

        # 尝试查找嵌入的JSON
        try:
            # 查找包含"tool"的JSON对象
            start = response.find('{"tool"')
            if start == -1:
                start = response.find("{'tool")
            if start != -1:
                # 找到匹配的右括号
                depth = 0
                for i, c in enumerate(response[start:]):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = response[start:start + i + 1]
                            data = json.loads(json_str)
                            return AgentAction(
                                tool=data.get("tool", ""),
                                tool_input=data.get("arguments", {}),
                                log=response,
                            )
        except (json.JSONDecodeError, KeyError):
            pass

        # 没有工具调用，返回最终答案
        return AgentFinish(output=response, log=response)


class PlanAndExecuteAgent(BaseAgent):
    """计划执行Agent。

    先制定完整计划，再逐步执行。

    工作流程：
        1. 分析任务，制定步骤计划
        2. 逐步执行计划中的每个步骤
        3. 根据执行结果调整计划
        4. 汇总结果

    示例：
        >>> agent = PlanAndExecuteAgent(tools=[calc, search])
        >>> result = agent.run("研究并总结机器学习的主要算法")
    """

    PLANNER_PROMPT = """你是一个任务规划专家。请为以下任务制定一个详细的执行计划。

任务: {input}

可用工具:
{tools}

请输出一个JSON格式的计划，包含步骤列表：
{{"steps": ["步骤1", "步骤2", ...]}}"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_plan: List[str] = []
        self._current_step: int = 0

    def _create_plan(self, input_text: str) -> List[str]:
        """创建执行计划。"""
        prompt = self.PLANNER_PROMPT.format(
            input=input_text,
            tools=self.tool_registry.get_tools_description(),
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages)

        # 解析计划
        try:
            json_match = re.search(r'\{[^{}]*"steps"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("steps", [])
        except (json.JSONDecodeError, KeyError):
            pass

        # 默认计划
        return [f"完成任务: {input_text}"]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """规划下一步动作。"""
        input_text = kwargs.get("input", "")

        # 首次调用时创建计划
        if not self._current_plan:
            self._current_plan = self._create_plan(input_text)
            self._current_step = 0

        # 检查是否完成所有步骤
        if self._current_step >= len(self._current_plan):
            # 汇总结果
            results = [obs for _, obs in intermediate_steps]
            summary = f"任务完成。执行了{len(results)}个步骤。\n结果摘要:\n" + "\n".join(results[-3:])
            return AgentFinish(output=summary, log="计划执行完成")

        # 获取当前步骤
        current_step = self._current_plan[self._current_step]
        self._current_step += 1

        # 决定使用哪个工具
        tool_name = self._select_tool_for_step(current_step)
        if tool_name:
            return AgentAction(
                tool=tool_name,
                tool_input={"input": current_step},
                log=f"执行步骤: {current_step}",
            )

        # 无需工具，直接完成
        return AgentFinish(output=f"步骤完成: {current_step}", log=current_step)

    def _select_tool_for_step(self, step: str) -> Optional[str]:
        """为步骤选择合适的工具。"""
        step_lower = step.lower()
        for tool in self.tools:
            if tool.name in step_lower:
                return tool.name
            # 简单的关键词匹配
            if "搜索" in step_lower or "查找" in step_lower:
                if tool.name == "search":
                    return tool.name
            if "计算" in step_lower or "算" in step_lower:
                if tool.name == "calculator":
                    return tool.name
        return None

    def reset(self) -> None:
        """重置Agent状态。"""
        self._current_plan = []
        self._current_step = 0
        self.state = AgentState()
