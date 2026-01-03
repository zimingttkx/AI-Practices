"""
Agent模块单元测试 (Agent Module Unit Tests)

测试覆盖：
    - AgentConfig配置验证
    - AgentState状态管理
    - AgentAction动作类
    - AgentFinish完成类
    - ReActAgent实现
    - ToolCallingAgent实现
    - PlanAndExecuteAgent实现

"""

import pytest
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import (
    AgentConfig,
    AgentState,
    AgentStatus,
    AgentAction,
    AgentFinish,
    BaseAgent,
    ReActAgent,
    ToolCallingAgent,
    PlanAndExecuteAgent,
)
from tools import CalculatorTool, SearchTool, ToolRegistry
from memory import BufferMemory


# ==================== AgentConfig Tests ====================

class TestAgentConfig:
    """AgentConfig测试类。"""

    def test_default_config(self):
        """测试默认配置。"""
        config = AgentConfig()
        assert config.max_iterations == 10
        assert config.max_execution_time == 120.0
        assert config.early_stopping is True
        assert config.return_intermediate_steps is True
        assert config.handle_parsing_errors is True

    def test_custom_config(self):
        """测试自定义配置。"""
        config = AgentConfig(
            max_iterations=20,
            max_execution_time=60.0,
            early_stopping=False,
        )
        assert config.max_iterations == 20
        assert config.max_execution_time == 60.0
        assert config.early_stopping is False

    def test_invalid_max_iterations(self):
        """测试无效的max_iterations。"""
        with pytest.raises(ValueError, match="max_iterations必须为正数"):
            AgentConfig(max_iterations=0)
        with pytest.raises(ValueError, match="max_iterations必须为正数"):
            AgentConfig(max_iterations=-1)

    def test_invalid_max_execution_time(self):
        """测试无效的max_execution_time。"""
        with pytest.raises(ValueError, match="max_execution_time必须为正数"):
            AgentConfig(max_execution_time=0)
        with pytest.raises(ValueError, match="max_execution_time必须为正数"):
            AgentConfig(max_execution_time=-1)


# ==================== AgentState Tests ====================

class TestAgentState:
    """AgentState测试类。"""

    def test_default_state(self):
        """测试默认状态。"""
        state = AgentState()
        assert state.status == AgentStatus.IDLE
        assert state.iteration == 0
        assert state.intermediate_steps == []
        assert state.final_output is None
        assert state.error is None

    def test_state_with_values(self):
        """测试带值的状态。"""
        state = AgentState(
            status=AgentStatus.THINKING,
            iteration=5,
            final_output="Done",
        )
        assert state.status == AgentStatus.THINKING
        assert state.iteration == 5
        assert state.final_output == "Done"

    def test_state_repr(self):
        """测试状态的字符串表示。"""
        state = AgentState(status=AgentStatus.ACTING, iteration=3)
        repr_str = repr(state)
        assert "AgentState" in repr_str
        assert "acting" in repr_str


# ==================== AgentStatus Tests ====================

class TestAgentStatus:
    """AgentStatus测试类。"""

    def test_status_values(self):
        """测试状态值。"""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.THINKING.value == "thinking"
        assert AgentStatus.ACTING.value == "acting"
        assert AgentStatus.FINISHED.value == "finished"
        assert AgentStatus.ERROR.value == "error"


# ==================== AgentAction Tests ====================

class TestAgentAction:
    """AgentAction测试类。"""

    def test_create_action(self):
        """测试创建动作。"""
        action = AgentAction(
            tool="calculator",
            tool_input={"expression": "2+2"},
        )
        assert action.tool == "calculator"
        assert action.tool_input == {"expression": "2+2"}

    def test_action_with_log(self):
        """测试带日志的动作。"""
        action = AgentAction(
            tool="search",
            tool_input={"query": "Python"},
            log="I need to search for Python",
        )
        assert action.log == "I need to search for Python"

    def test_action_repr(self):
        """测试动作的字符串表示。"""
        action = AgentAction(tool="calculator", tool_input={"x": 1})
        repr_str = repr(action)
        assert "AgentAction" in repr_str
        assert "calculator" in repr_str


# ==================== AgentFinish Tests ====================

class TestAgentFinish:
    """AgentFinish测试类。"""

    def test_create_finish(self):
        """测试创建完成。"""
        finish = AgentFinish(output="The answer is 42")
        assert finish.output == "The answer is 42"

    def test_finish_with_log(self):
        """测试带日志的完成。"""
        finish = AgentFinish(
            output="Done",
            log="Task completed successfully",
        )
        assert finish.log == "Task completed successfully"

    def test_finish_repr(self):
        """测试完成的字符串表示。"""
        finish = AgentFinish(output="The answer is 42")
        repr_str = repr(finish)
        assert "AgentFinish" in repr_str

    def test_finish_repr_long_output(self):
        """测试长输出的字符串表示。"""
        long_output = "A" * 100
        finish = AgentFinish(output=long_output)
        repr_str = repr(finish)
        assert "..." in repr_str


# ==================== ReActAgent Tests ====================

class TestReActAgent:
    """ReActAgent测试类。"""

    def setup_method(self):
        """测试前初始化。"""
        self.calc = CalculatorTool()
        self.search = SearchTool()

    def test_create_react_agent(self):
        """测试创建ReAct Agent。"""
        agent = ReActAgent(tools=[self.calc, self.search])
        assert len(agent.tools) == 2

    def test_react_agent_with_config(self):
        """测试带配置的ReAct Agent。"""
        config = AgentConfig(max_iterations=5)
        agent = ReActAgent(tools=[self.calc], config=config)
        assert agent.config.max_iterations == 5

    def test_react_agent_with_memory(self):
        """测试带记忆的ReAct Agent。"""
        memory = BufferMemory(system_message="Be helpful")
        agent = ReActAgent(tools=[self.calc], memory=memory)
        assert agent.memory is memory

    def test_parse_final_answer(self):
        """测试解析最终答案。"""
        agent = ReActAgent(tools=[self.calc])
        text = "Thought: I know the answer\nFinal Answer: 42"
        result = agent._parse_output(text)
        assert isinstance(result, AgentFinish)
        assert result.output == "42"

    def test_parse_action(self):
        """测试解析动作。"""
        agent = ReActAgent(tools=[self.calc])
        text = 'Thought: I need to calculate\nAction: calculator\nAction Input: {"expression": "2+2"}'
        result = agent._parse_output(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "calculator"

    def test_parse_action_simple_input(self):
        """测试解析简单输入的动作。"""
        agent = ReActAgent(tools=[self.calc])
        text = "Thought: Calculate\nAction: calculator\nAction Input: 2+2"
        result = agent._parse_output(text)
        assert isinstance(result, AgentAction)
        assert "input" in result.tool_input

    def test_build_scratchpad(self):
        """测试构建思考记录。"""
        agent = ReActAgent(tools=[self.calc])
        steps = [
            (AgentAction(tool="calc", tool_input={"x": 1}, log="thinking"), "result"),
        ]
        scratchpad = agent._build_scratchpad(steps)
        assert "Thought:" in scratchpad
        assert "Action:" in scratchpad
        assert "Observation:" in scratchpad

    def test_react_agent_repr(self):
        """测试ReAct Agent的字符串表示。"""
        agent = ReActAgent(tools=[self.calc])
        repr_str = repr(agent)
        assert "ReActAgent" in repr_str


# ==================== ToolCallingAgent Tests ====================

class TestToolCallingAgent:
    """ToolCallingAgent测试类。"""

    def setup_method(self):
        """测试前初始化。"""
        self.calc = CalculatorTool()

    def test_create_tool_calling_agent(self):
        """测试创建ToolCalling Agent。"""
        agent = ToolCallingAgent(tools=[self.calc])
        assert len(agent.tools) == 1

    def test_parse_json_tool_call(self):
        """测试解析JSON工具调用。"""
        agent = ToolCallingAgent(tools=[self.calc])
        # JSON必须是独立的，不能有前缀文字
        response = '{"tool": "calculator", "arguments": {"expression": "2+2"}}'
        result = agent._parse_response(response)
        assert isinstance(result, AgentAction)
        assert result.tool == "calculator"

    def test_parse_no_tool_call(self):
        """测试解析无工具调用。"""
        agent = ToolCallingAgent(tools=[self.calc])
        response = "The answer is 42"
        result = agent._parse_response(response)
        assert isinstance(result, AgentFinish)
        assert result.output == "The answer is 42"

    def test_tool_calling_agent_repr(self):
        """测试ToolCalling Agent的字符串表示。"""
        agent = ToolCallingAgent(tools=[self.calc])
        repr_str = repr(agent)
        assert "ToolCallingAgent" in repr_str


# ==================== PlanAndExecuteAgent Tests ====================

class TestPlanAndExecuteAgent:
    """PlanAndExecuteAgent测试类。"""

    def setup_method(self):
        """测试前初始化。"""
        self.calc = CalculatorTool()
        self.search = SearchTool()

    def test_create_plan_execute_agent(self):
        """测试创建PlanAndExecute Agent。"""
        agent = PlanAndExecuteAgent(tools=[self.calc, self.search])
        assert len(agent.tools) == 2

    def test_reset_agent(self):
        """测试重置Agent。"""
        agent = PlanAndExecuteAgent(tools=[self.calc])
        agent._current_plan = ["step1", "step2"]
        agent._current_step = 1
        agent.reset()
        assert agent._current_plan == []
        assert agent._current_step == 0

    def test_select_tool_for_step_calculator(self):
        """测试为步骤选择计算器工具。"""
        agent = PlanAndExecuteAgent(tools=[self.calc, self.search])
        tool = agent._select_tool_for_step("计算 2+2 的结果")
        assert tool == "calculator"

    def test_select_tool_for_step_search(self):
        """测试为步骤选择搜索工具。"""
        agent = PlanAndExecuteAgent(tools=[self.calc, self.search])
        tool = agent._select_tool_for_step("搜索Python教程")
        assert tool == "search"

    def test_select_tool_for_step_none(self):
        """测试无匹配工具。"""
        agent = PlanAndExecuteAgent(tools=[self.calc])
        tool = agent._select_tool_for_step("写一首诗")
        assert tool is None

    def test_plan_execute_agent_repr(self):
        """测试PlanAndExecute Agent的字符串表示。"""
        agent = PlanAndExecuteAgent(tools=[self.calc])
        repr_str = repr(agent)
        assert "PlanAndExecuteAgent" in repr_str


# ==================== BaseAgent Tests ====================

class TestBaseAgent:
    """BaseAgent测试类。"""

    def test_agent_has_tools(self):
        """测试Agent有工具。"""
        calc = CalculatorTool()
        agent = ReActAgent(tools=[calc])
        assert len(agent.tools) == 1

    def test_agent_tool_registry(self):
        """测试Agent工具注册表。"""
        calc = CalculatorTool()
        agent = ReActAgent(tools=[calc])
        assert agent.tool_registry.get("calculator") is calc

    def test_agent_default_llm(self):
        """测试Agent默认LLM。"""
        agent = ReActAgent(tools=[CalculatorTool()])
        response = agent._default_llm([{"role": "user", "content": "test"}])
        assert "模拟" in response or "LLM" in response

    def test_agent_custom_llm(self):
        """测试Agent自定义LLM。"""
        def custom_llm(messages):
            return "Final Answer: Custom response"
        
        agent = ReActAgent(tools=[CalculatorTool()], llm=custom_llm)
        # 运行agent应该使用自定义LLM
        result = agent.run("test")
        assert "Custom response" in result

    def test_execute_action_success(self):
        """测试执行动作成功。"""
        calc = CalculatorTool()
        agent = ReActAgent(tools=[calc])
        action = AgentAction(tool="calculator", tool_input={"expression": "2+2"})
        result = agent._execute_action(action)
        assert "4" in result

    def test_execute_action_tool_not_found(self):
        """测试执行动作工具未找到。"""
        agent = ReActAgent(tools=[CalculatorTool()])
        action = AgentAction(tool="nonexistent", tool_input={})
        result = agent._execute_action(action)
        assert "未找到" in result or "错误" in result


# ==================== Agent Run Tests ====================

class TestAgentRun:
    """Agent运行测试。"""

    def test_run_with_immediate_finish(self):
        """测试立即完成的运行。"""
        def llm_finish(messages):
            return "Final Answer: Done"
        
        agent = ReActAgent(tools=[CalculatorTool()], llm=llm_finish)
        result = agent.run("test")
        assert result == "Done"
        assert agent.state.status == AgentStatus.FINISHED

    def test_run_max_iterations(self):
        """测试达到最大迭代。"""
        def llm_loop(messages):
            return 'Thought: Keep going\nAction: calculator\nAction Input: {"expression": "1+1"}'
        
        config = AgentConfig(max_iterations=3)
        agent = ReActAgent(tools=[CalculatorTool()], llm=llm_loop, config=config)
        result = agent.run("test")
        assert "最大迭代" in result
        assert agent.state.iteration == 3

    def test_run_updates_memory(self):
        """测试运行更新记忆。"""
        def llm_finish(messages):
            return "Final Answer: Done"
        
        memory = BufferMemory()
        agent = ReActAgent(tools=[CalculatorTool()], llm=llm_finish, memory=memory)
        agent.run("Hello")
        
        messages = memory.get_messages()
        assert len(messages) >= 2  # 用户消息 + 助手消息


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
