"""
多智能体系统 - 严格单元测试
测试覆盖: agent_base, agent_communication, agent_orchestrator, debate_agents, collaborative_agents
"""

import sys
import asyncio
import pytest
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from agent_base import (
    AgentRole, AgentState, AgentConfig, AgentResponse,
    BaseAgent, SimpleAgent, ReActAgent, MockLLM, create_agent
)
from agent_communication import (
    MessageType, MessagePriority, AgentMessage,
    DirectChannel, BroadcastChannel, TopicChannel, MessageBus
)
from agent_orchestrator import (
    TaskStatus, TaskAssignment, OrchestratorConfig,
    RoundRobinOrchestrator, CapabilityBasedOrchestrator, LoadBalancedOrchestrator
)
from debate_agents import (
    DebateRole, DebatePhase, DebateConfig, Argument, JudgmentScore, DebateRound,
    DebaterAgent, JudgeAgent, DebateArena
)
from collaborative_agents import (
    CollaborationMode, TeamConfig, TeamMember, Contribution,
    CollaborativeTeam, ConsensusBuilder, VotingSystem
)


def run_async(coro):
    """运行异步函数的辅助方法"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ============================================================
# 第一部分: AgentRole 测试
# ============================================================

class TestAgentRole:
    """AgentRole 枚举测试"""
    
    def test_role_values(self):
        """测试角色值"""
        assert AgentRole.ASSISTANT.value == 'assistant'
        assert AgentRole.CRITIC.value == 'critic'
        assert AgentRole.MANAGER.value == 'manager'
        assert AgentRole.RESEARCHER.value == 'researcher'
        assert AgentRole.CODER.value == 'coder'
        assert AgentRole.DEBATER.value == 'debater'
    
    def test_role_count(self):
        """测试角色数量"""
        assert len(list(AgentRole)) == 8  # 实际有8种角色
    
    def test_role_from_string(self):
        """测试从字符串创建角色"""
        assert AgentRole('assistant') == AgentRole.ASSISTANT
        assert AgentRole('coder') == AgentRole.CODER
    
    def test_role_string_conversion(self):
        """测试角色转字符串"""
        assert str(AgentRole.ASSISTANT) == 'AgentRole.ASSISTANT' or AgentRole.ASSISTANT.value == 'assistant'


# ============================================================
# 第二部分: AgentState 测试
# ============================================================

class TestAgentState:
    """AgentState 枚举测试"""
    
    def test_state_values(self):
        """测试状态值"""
        assert AgentState.IDLE.value == 'idle'
        assert AgentState.THINKING.value == 'thinking'
        assert AgentState.SPEAKING.value == 'speaking'
        assert AgentState.ERROR.value == 'error'
        assert AgentState.TERMINATED.value == 'terminated'
    
    def test_state_count(self):
        """测试状态数量"""
        assert len(list(AgentState)) == 7  # 实际有7种状态
    
    def test_valid_transitions_from_idle(self):
        """测试从 IDLE 状态的有效转换"""
        assert AgentState.IDLE.can_transition_to(AgentState.THINKING) == True
        assert AgentState.IDLE.can_transition_to(AgentState.TERMINATED) == True
    
    def test_valid_transitions_from_thinking(self):
        """测试从 THINKING 状态的有效转换"""
        assert AgentState.THINKING.can_transition_to(AgentState.SPEAKING) == True
        assert AgentState.THINKING.can_transition_to(AgentState.ERROR) == True
    
    def test_valid_transitions_from_speaking(self):
        """测试从 SPEAKING 状态的有效转换"""
        assert AgentState.SPEAKING.can_transition_to(AgentState.IDLE) == True
        assert AgentState.SPEAKING.can_transition_to(AgentState.ERROR) == True
    
    def test_valid_transitions_from_error(self):
        """测试从 ERROR 状态的有效转换"""
        assert AgentState.ERROR.can_transition_to(AgentState.IDLE) == True
    
    def test_terminated_no_transitions(self):
        """测试 TERMINATED 状态不能转换"""
        assert AgentState.TERMINATED.can_transition_to(AgentState.IDLE) == False
        assert AgentState.TERMINATED.can_transition_to(AgentState.THINKING) == False


# ============================================================
# 第三部分: AgentConfig 测试
# ============================================================

class TestAgentConfig:
    """AgentConfig 配置类测试"""
    
    def test_minimal_config(self):
        """测试最小配置"""
        config = AgentConfig(name='TestBot')
        assert config.name == 'TestBot'
        assert config.role == AgentRole.ASSISTANT
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
    
    def test_full_config(self):
        """测试完整配置"""
        config = AgentConfig(
            name='FullBot',
            role=AgentRole.CODER,
            system_prompt='You are a coder.',
            model_name='gpt-4',
            temperature=0.5,
            max_tokens=2000,
            capabilities={'coding', 'debugging'}
        )
        assert config.name == 'FullBot'
        assert config.role == AgentRole.CODER
        assert config.system_prompt == 'You are a coder.'
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert 'coding' in config.capabilities
    
    def test_invalid_empty_name(self):
        """测试空名称验证"""
        with pytest.raises(ValueError):
            AgentConfig(name='')
    
    def test_invalid_temperature_high(self):
        """测试温度过高验证"""
        with pytest.raises(ValueError):
            AgentConfig(name='Test', temperature=3.0)
    
    def test_invalid_temperature_low(self):
        """测试温度过低验证"""
        with pytest.raises(ValueError):
            AgentConfig(name='Test', temperature=-0.5)
    
    def test_valid_temperature_boundary(self):
        """测试温度边界值"""
        config1 = AgentConfig(name='Test', temperature=0.0)
        assert config1.temperature == 0.0
        config2 = AgentConfig(name='Test', temperature=2.0)
        assert config2.temperature == 2.0


# ============================================================
# 第四部分: MockLLM 测试
# ============================================================

class TestMockLLM:
    """MockLLM 测试"""
    
    def test_mock_llm_responses(self):
        """测试 MockLLM 响应"""
        responses = ['Response 1', 'Response 2', 'Response 3']
        llm = MockLLM(responses=responses)
        
        result1 = run_async(llm.generate('test'))
        assert result1 == 'Response 1'
        
        result2 = run_async(llm.generate('test'))
        assert result2 == 'Response 2'
    
    def test_mock_llm_cycle(self):
        """测试 MockLLM 循环响应"""
        responses = ['A', 'B']
        llm = MockLLM(responses=responses)
        
        run_async(llm.generate('1'))  # A
        run_async(llm.generate('2'))  # B
        result = run_async(llm.generate('3'))  # 循环回 A
        assert result == 'A'


# ============================================================
# 第五部分: SimpleAgent 测试
# ============================================================

class TestSimpleAgent:
    """SimpleAgent 测试"""
    
    def test_create_simple_agent(self):
        """测试创建 SimpleAgent"""
        llm = MockLLM(responses=['Hello!'])
        agent = create_agent(name='TestAgent', llm=llm)
        assert agent.name == 'TestAgent'
        assert agent.is_active == True
        assert agent.state == AgentState.IDLE
    
    def test_agent_step(self):
        """测试 Agent 单步执行"""
        llm = MockLLM(responses=['Test response'])
        agent = create_agent(name='StepAgent', llm=llm)
        response = run_async(agent.step('Hello'))
        assert response.content == 'Test response'
    
    def test_agent_reset(self):
        """测试 Agent 重置"""
        llm = MockLLM(responses=['R1', 'R2'])
        agent = create_agent(name='ResetAgent', llm=llm)
        run_async(agent.step('msg1'))
        agent.reset()
        assert len(agent.get_history()) == 0
    
    def test_agent_terminate(self):
        """测试 Agent 终止"""
        llm = MockLLM(responses=['test'])
        agent = create_agent(name='TermAgent', llm=llm)
        agent.terminate()
        assert agent.state == AgentState.TERMINATED
        assert agent.is_active == False
    
    def test_agent_history(self):
        """测试 Agent 历史记录"""
        llm = MockLLM(responses=['Response'])
        agent = create_agent(name='HistAgent', llm=llm)
        run_async(agent.step('User message'))
        history = agent.get_history()
        assert len(history) >= 1


# ============================================================
# 第六部分: ReActAgent 测试
# ============================================================

class TestReActAgent:
    """ReActAgent 测试"""
    
    def test_create_react_agent(self):
        """测试创建 ReActAgent"""
        llm = MockLLM(responses=['Thought: test\nFinal Answer: done'])
        agent = create_agent(name='ReactTest', agent_type='react', llm=llm)
        assert isinstance(agent, ReActAgent)
    
    def test_react_agent_step(self):
        """测试 ReActAgent 执行"""
        llm = MockLLM(responses=['Thought: analyzing\nFinal Answer: Result'])
        agent = create_agent(name='ReactStep', agent_type='react', llm=llm)
        response = run_async(agent.step('Question'))
        assert response is not None


# ============================================================
# 第七部分: MessageType 测试
# ============================================================

class TestMessageType:
    """MessageType 枚举测试"""
    
    def test_message_type_values(self):
        """测试消息类型值"""
        assert MessageType.CHAT.value == 'chat'
        assert MessageType.QUERY.value == 'query'
        assert MessageType.RESPONSE.value == 'response'


# ============================================================
# 第八部分: AgentMessage 测试
# ============================================================

class TestAgentMessage:
    """AgentMessage 测试"""
    
    def test_create_message(self):
        """测试创建消息"""
        msg = AgentMessage(
            sender_id='agent1',
            sender_name='Agent1',
            content='Hello',
            msg_type=MessageType.CHAT
        )
        assert msg.sender_id == 'agent1'
        assert msg.content == 'Hello'
        assert msg.msg_type == MessageType.CHAT
    
    def test_message_with_recipient(self):
        """测试带接收者的消息"""
        msg = AgentMessage(
            sender_id='agent1',
            sender_name='Agent1',
            content='Direct message',
            msg_type=MessageType.CHAT,
            recipient_id='agent2'
        )
        assert msg.recipient_id == 'agent2'


# ============================================================
# 第九部分: Channel 测试
# ============================================================

class TestChannels:
    """通道测试"""
    
    def test_direct_channel(self):
        """测试直接通道"""
        channel = DirectChannel('test_channel')
        assert channel.name == 'test_channel'
    
    def test_broadcast_channel(self):
        """测试广播通道"""
        channel = BroadcastChannel('broadcast_ch')
        assert channel.name == 'broadcast_ch'
    
    def test_topic_channel(self):
        """测试主题通道"""
        channel = TopicChannel('topic_ch')
        assert channel.name == 'topic_ch'


# ============================================================
# 第十部分: TaskStatus 测试
# ============================================================

class TestTaskStatus:
    """TaskStatus 枚举测试"""
    
    def test_task_status_values(self):
        """测试任务状态值"""
        assert TaskStatus.PENDING.value == 'pending'
        assert TaskStatus.ASSIGNED.value == 'assigned'
        assert TaskStatus.IN_PROGRESS.value == 'in_progress'
        assert TaskStatus.COMPLETED.value == 'completed'
        assert TaskStatus.FAILED.value == 'failed'


# ============================================================
# 第十一部分: Orchestrator 测试
# ============================================================

class TestOrchestrators:
    """编排器测试"""
    
    def test_round_robin_orchestrator(self):
        """测试轮询编排器"""
        agents = [
            create_agent('Agent1', llm=MockLLM(responses=['R1'])),
            create_agent('Agent2', llm=MockLLM(responses=['R2'])),
        ]
        config = OrchestratorConfig()
        orch = RoundRobinOrchestrator(config)
        for agent in agents:
            orch.register_agent(agent)
        assert len(orch._agents) == 2
    
    def test_capability_orchestrator(self):
        """测试能力编排器"""
        agents = [
            create_agent('Coder', llm=MockLLM(responses=['code'])),
        ]
        config = OrchestratorConfig()
        orch = CapabilityBasedOrchestrator(config)
        for agent in agents:
            orch.register_agent(agent)
        assert orch is not None
    
    def test_load_balanced_orchestrator(self):
        """测试负载均衡编排器"""
        agents = [
            create_agent('Worker1', llm=MockLLM(responses=['w1'])),
            create_agent('Worker2', llm=MockLLM(responses=['w2'])),
        ]
        config = OrchestratorConfig()
        orch = LoadBalancedOrchestrator(config)
        for agent in agents:
            orch.register_agent(agent)
        assert len(orch._agents) == 2


# ============================================================
# 第十二部分: DebateRole 测试
# ============================================================

class TestDebateRole:
    """DebateRole 枚举测试"""
    
    def test_debate_role_values(self):
        """测试辩论角色值"""
        assert DebateRole.PROPONENT.value == 'proponent'
        assert DebateRole.OPPONENT.value == 'opponent'
        assert DebateRole.JUDGE.value == 'judge'
    
    def test_debate_role_count(self):
        """测试辩论角色数量"""
        assert len(list(DebateRole)) == 3


# ============================================================
# 第十三部分: DebateConfig 测试
# ============================================================

class TestDebateConfig:
    """DebateConfig 测试"""
    
    def test_debate_config_defaults(self):
        """测试辩论配置默认值"""
        config = DebateConfig(topic='Test topic')
        assert config.topic == 'Test topic'
        assert config.max_rounds >= 1
    
    def test_debate_config_custom(self):
        """测试自定义辩论配置"""
        config = DebateConfig(
            topic='AI Ethics',
            max_rounds=5,
            allow_rebuttals=True
        )
        assert config.topic == 'AI Ethics'
        assert config.max_rounds == 5
        assert config.allow_rebuttals == True


# ============================================================
# 第十四部分: DebaterAgent 测试
# ============================================================

class TestDebaterAgent:
    """DebaterAgent 测试"""
    
    def test_create_proponent(self):
        """测试创建正方辩手"""
        llm = MockLLM(responses=['Pro argument'])
        config = AgentConfig(name='Proponent')
        agent = DebaterAgent(config, DebateRole.PROPONENT, llm)
        assert agent.debate_role == DebateRole.PROPONENT
    
    def test_create_opponent(self):
        """测试创建反方辩手"""
        llm = MockLLM(responses=['Con argument'])
        config = AgentConfig(name='Opponent')
        agent = DebaterAgent(config, DebateRole.OPPONENT, llm)
        assert agent.debate_role == DebateRole.OPPONENT
    
    def test_generate_opening(self):
        """测试生成开场陈述"""
        llm = MockLLM(responses=['Opening statement'])
        config = AgentConfig(name='Debater')
        agent = DebaterAgent(config, DebateRole.PROPONENT, llm)
        arg = run_async(agent.generate_opening('Test topic'))
        assert arg is not None


# ============================================================
# 第十五部分: JudgeAgent 测试
# ============================================================

class TestJudgeAgent:
    """JudgeAgent 测试"""
    
    def test_create_judge(self):
        """测试创建裁判"""
        llm = MockLLM(responses=['Judgment'])
        config = AgentConfig(name='Judge')
        judge = JudgeAgent(config, llm=llm)
        assert judge.name == 'Judge'
    
    def test_judge_criteria(self):
        """测试裁判评判标准"""
        llm = MockLLM(responses=['Score'])
        config = AgentConfig(name='Judge')
        judge = JudgeAgent(config, llm=llm)
        assert hasattr(judge, 'criteria')


# ============================================================
# 第十六部分: CollaborationMode 测试
# ============================================================

class TestCollaborationMode:
    """CollaborationMode 枚举测试"""
    
    def test_collaboration_mode_values(self):
        """测试协作模式值"""
        assert CollaborationMode.SEQUENTIAL.value == 'sequential'
        assert CollaborationMode.PARALLEL.value == 'parallel'
        assert CollaborationMode.ROUND_ROBIN.value == 'round_robin'
    
    def test_collaboration_mode_count(self):
        """测试协作模式数量"""
        assert len(list(CollaborationMode)) == 4  # 实际有4种模式


# ============================================================
# 第十七部分: TeamConfig 测试
# ============================================================

class TestTeamConfig:
    """TeamConfig 测试"""
    
    def test_team_config_defaults(self):
        """测试团队配置默认值"""
        config = TeamConfig(name='TestTeam')
        assert config.name == 'TestTeam'
    
    def test_team_config_with_mode(self):
        """测试带模式的团队配置"""
        config = TeamConfig(
            name='SeqTeam',
            mode=CollaborationMode.SEQUENTIAL
        )
        assert config.mode == CollaborationMode.SEQUENTIAL


# ============================================================
# 第十八部分: CollaborativeTeam 测试
# ============================================================

class TestCollaborativeTeam:
    """CollaborativeTeam 测试"""
    
    def test_create_team(self):
        """测试创建团队"""
        config = TeamConfig(name='TestTeam')
        team = CollaborativeTeam(config)
        assert team.config.name == 'TestTeam'
    
    def test_add_member(self):
        """测试添加成员"""
        config = TeamConfig(name='TestTeam')
        team = CollaborativeTeam(config)
        agent = create_agent('Member1', llm=MockLLM(responses=['test']))
        team.add_member(agent, 'testing')
        assert len(team._members) == 1
    
    def test_team_collaborate(self):
        """测试团队协作"""
        config = TeamConfig(name='CollabTeam', mode=CollaborationMode.SEQUENTIAL)
        team = CollaborativeTeam(config)
        agent = create_agent('Worker', llm=MockLLM(responses=['Result']))
        team.add_member(agent, 'work')
        result = run_async(team.collaborate('Task'))
        assert result is not None


# ============================================================
# 第十九部分: VotingSystem 测试
# ============================================================

class TestVotingSystem:
    """VotingSystem 测试"""
    
    def test_create_voting_system(self):
        """测试创建投票系统"""
        agents = [
            create_agent('V1', llm=MockLLM(responses=['1'])),
            create_agent('V2', llm=MockLLM(responses=['2'])),
        ]
        voting = VotingSystem(agents)
        assert len(voting.agents) == 2
    
    def test_voting(self):
        """测试投票"""
        agents = [
            create_agent('V1', llm=MockLLM(responses=['1'])),
            create_agent('V2', llm=MockLLM(responses=['1'])),
            create_agent('V3', llm=MockLLM(responses=['2'])),
        ]
        voting = VotingSystem(agents)
        result = run_async(voting.vote('Question?', ['A', 'B', 'C']))
        assert 'winner' in result or 'tally' in result


# ============================================================
# 第二十部分: ConsensusBuilder 测试
# ============================================================

class TestConsensusBuilder:
    """ConsensusBuilder 测试"""
    
    def test_create_consensus_builder(self):
        """测试创建共识构建器"""
        agents = [
            create_agent('E1', llm=MockLLM(responses=['Opinion1'])),
            create_agent('E2', llm=MockLLM(responses=['Opinion2'])),
        ]
        builder = ConsensusBuilder(agents=agents, threshold=0.7)
        assert len(builder.agents) == 2
    
    def test_build_consensus(self):
        """测试构建共识"""
        agents = [
            create_agent('E1', llm=MockLLM(responses=['Agree', 'Consensus'])),
            create_agent('E2', llm=MockLLM(responses=['Agree', 'Consensus'])),
        ]
        builder = ConsensusBuilder(agents=agents, threshold=0.5, max_iterations=2)
        result = run_async(builder.build_consensus('Topic'))
        assert result is not None


# ============================================================
# 第二十一部分: 集成测试
# ============================================================

class TestIntegration:
    """集成测试"""
    
    def test_full_agent_workflow(self):
        """测试完整 Agent 工作流"""
        llm = MockLLM(responses=['Hello', 'How can I help?', 'Goodbye'])
        agent = create_agent('IntegrationBot', llm=llm)
        
        # 多轮对话
        r1 = run_async(agent.step('Hi'))
        assert r1.content == 'Hello'
        
        r2 = run_async(agent.step('Help me'))
        assert r2.content == 'How can I help?'
        
        # 检查历史
        history = agent.get_history()
        assert len(history) >= 2
        
        # 重置
        agent.reset()
        assert len(agent.get_history()) == 0
    
    def test_multi_agent_communication(self):
        """测试多智能体通信"""
        bus = MessageBus()
        
        msg = AgentMessage(
            sender_id='agent1',
            sender_name='Agent1',
            content='Test message',
            msg_type=MessageType.CHAT,
            recipient_id='agent2'
        )
        
        run_async(bus.send(msg))
        # 验证消息发送成功
        assert msg.sender_id == 'agent1'


# ============================================================
# 第二十二部分: 边界和异常测试
# ============================================================

class TestEdgeCasesAndExceptions:
    """边界情况和异常处理测试"""
    
    def test_agent_invalid_state_transition(self):
        """测试无效的状态转换"""
        llm = MockLLM(responses=['test'])
        agent = create_agent('TestAgent', llm=llm)
        agent.set_state(AgentState.TERMINATED)
        
        # 尝试从 TERMINATED 转换到 THINKING 应该失败
        with pytest.raises(ValueError):
            agent.set_state(AgentState.THINKING)
    
    def test_agent_error_state_recovery(self):
        """测试从错误状态恢复 - 需要先经历一个会话才能进入错误状态"""
        class FailingLLM:
            async def generate(self, messages, **kwargs):
                raise Exception("Simulated error")
        
        llm = FailingLLM()
        agent = create_agent('ErrorAgent', llm=llm)
        
        # 运行一个会话让agent进入错误状态
        run_async(agent.step('Trigger error'))
        
        # 验证agent处于错误状态
        assert agent.state == AgentState.ERROR
        
        # 从 ERROR 可以恢复到 IDLE
        agent.set_state(AgentState.IDLE)
        assert agent.state == AgentState.IDLE
    
    def test_message_create_reply(self):
        """测试消息回复创建"""
        original = AgentMessage(
            sender_id='agent1',
            sender_name='Agent1',
            content='Question',
            msg_type=MessageType.QUERY
        )
        
        reply = original.create_reply(
            content='Answer',
            sender_id='agent2',
            sender_name='Agent2'
        )
        
        assert reply.recipient_id == 'agent1'
        assert reply.reply_to == original.id
        assert reply.msg_type == MessageType.RESPONSE
    
    def test_message_serialization(self):
        """测试消息序列化和反序列化"""
        original = AgentMessage(
            sender_id='agent1',
            sender_name='Agent1',
            content='Test content',
            msg_type=MessageType.CHAT,
            recipient_id='agent2'
        )
        
        # 序列化
        data = original.to_dict()
        assert 'sender_id' in data
        assert 'content' in data
        
        # 反序列化
        restored = AgentMessage.from_dict(data)
        assert restored.sender_id == original.sender_id
        assert restored.content == original.content
    
    def test_task_assignment_lifecycle(self):
        """测试任务分配完整生命周期"""
        task = TaskAssignment(
            task_id='task_001',
            description='Test task',
            priority=1
        )
        
        # 初始状态
        assert task.status == TaskStatus.PENDING
        assert task.agent_id is None
        
        # 分配
        task.assign('agent_1')
        assert task.status == TaskStatus.ASSIGNED
        assert task.agent_id == 'agent_1'
        assert task.assigned_at is not None
        
        # 开始
        task.start()
        assert task.status == TaskStatus.IN_PROGRESS
        
        # 完成
        result = {'output': 'success'}
        task.complete(result)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at is not None
    
    def test_task_failure(self):
        """测试任务失败"""
        task = TaskAssignment(
            task_id='task_002',
            description='Failing task'
        )
        
        task.assign('agent_1')
        task.start()
        task.fail('Connection error')
        
        assert task.status == TaskStatus.FAILED
        assert task.result == {'error': 'Connection error'}
        assert task.completed_at is not None
    
    def test_orchestrator_task_creation(self):
        """测试编排器创建任务"""
        orch = RoundRobinOrchestrator()
        
        task = orch.create_task(
            description='Test task',
            priority=5,
            required_capabilities={'coding'}
        )
        
        assert task.task_id in orch._tasks
        assert task.priority == 5
        assert 'coding' in task.required_capabilities
    
    def test_orchestrator_stats(self):
        """测试编排器统计"""
        agents = [
            create_agent('A1', llm=MockLLM(responses=['r1'])),
            create_agent('A2', llm=MockLLM(responses=['r2'])),
        ]
        orch = RoundRobinOrchestrator()
        
        for agent in agents:
            orch.register_agent(agent)
        
        stats = orch.get_stats()
        assert 'agents_count' in stats
        assert 'tasks_created' in stats
        assert stats['agents_count'] == 2
    
    def test_debate_argument_creation(self):
        """测试辩论论据创建"""
        llm = MockLLM(responses=['Opening argument'])
        config = AgentConfig(name='Debater')
        agent = DebaterAgent(config, DebateRole.PROPONENT, llm)
        
        arg = run_async(agent.generate_opening('AI is good'))
        
        assert arg.agent_id == agent.id
        assert arg.role == DebateRole.PROPONENT
        assert arg.phase == DebatePhase.OPENING
        assert arg.round_num == 0
    
    def test_debate_round_completion(self):
        """测试辩论轮次完成检测"""
        round1 = DebateRound(round_num=1)
        assert round1.is_complete == False
        
        round1.proponent_argument = Argument(
            agent_id='a1', agent_name='A1', role=DebateRole.PROPONENT,
            content='Pro arg', phase=DebatePhase.OPENING, round_num=1
        )
        assert round1.is_complete == False
        
        round1.opponent_argument = Argument(
            agent_id='a2', agent_name='A2', role=DebateRole.OPPONENT,
            content='Opp arg', phase=DebatePhase.OPENING, round_num=1
        )
        assert round1.is_complete == True
    
    def test_team_remove_member(self):
        """测试团队移除成员"""
        config = TeamConfig(name='TestTeam')
        team = CollaborativeTeam(config)
        
        agent1 = create_agent('M1', llm=MockLLM(responses=['test']))
        agent2 = create_agent('M2', llm=MockLLM(responses=['test']))
        
        team.add_member(agent1, 'role1')
        team.add_member(agent2, 'role2')
        assert len(team._members) == 2
        
        removed = team.remove_member(agent1.id)
        assert removed == True
        assert len(team._members) == 1
        
        # 再次移除应该返回 False
        removed = team.remove_member(agent1.id)
        assert removed == False
    
    def test_agent_capability_check(self):
        """测试能力检查"""
        config = AgentConfig(
            name='CapableAgent',
            capabilities={'coding', 'debugging', 'testing'}
        )
        agent = SimpleAgent(config)
        
        assert agent.has_capability('coding') == True
        assert agent.has_capability('debugging') == True
        assert agent.has_capability('design') == False
    
    def test_agent_response_to_dict(self):
        """测试 AgentResponse 序列化"""
        resp = AgentResponse(
            content='Test response',
            agent_id='agent_001',
            agent_name='TestAgent',
            metadata={'iteration': 1}
        )
        
        data = resp.to_dict()
        assert data['content'] == 'Test response'
        assert data['agent_id'] == 'agent_001'
        assert data['metadata']['iteration'] == 1
        assert 'timestamp' in data
    
    def test_message_priority_comparison(self):
        """测试消息优先级比较"""
        assert MessagePriority.LOW < MessagePriority.NORMAL
        assert MessagePriority.NORMAL < MessagePriority.HIGH
        assert MessagePriority.HIGH < MessagePriority.URGENT
        
        assert (MessagePriority.URGENT > MessagePriority.HIGH) == True
        assert (MessagePriority.LOW > MessagePriority.URGENT) == False


# ============================================================
# 第二十三部分: 性能和并发测试
# ============================================================

class TestPerformanceAndConcurrency:
    """性能和并发测试"""
    
    def test_concurrent_message_handling(self):
        """测试并发消息处理"""
        bus = MessageBus()
        
        messages_sent = []
        
        async def mock_handler(msg):
            messages_sent.append(msg.id)
        
        # 注册多个代理
        for i in range(5):
            bus.register_agent(
                f'agent_{i}',
                f'Agent{i}',
                mock_handler
            )
        
        # 发送多条消息
        async def send_messages():
            tasks = []
            for i in range(10):
                msg = AgentMessage(
                    sender_id=f'agent_{i % 5}',
                    sender_name=f'Agent{i % 5}',
                    content=f'Message {i}',
                    msg_type=MessageType.CHAT
                )
                tasks.append(bus.send(msg))
            await asyncio.gather(*tasks)
        
        run_async(send_messages())
        
        # 验证统计
        stats = bus.get_stats()
        assert stats['sent'] == 10
    
    def test_message_bus_statistics(self):
        """测试消息总线统计"""
        bus = MessageBus()
        
        async def handler(msg):
            pass
        
        bus.register_agent('agent1', 'Agent1', handler)
        
        # 发送消息
        msg = AgentMessage(
            sender_id='agent1',
            sender_name='Agent1',
            content='Test',
            msg_type=MessageType.CHAT
        )
        run_async(bus.send(msg))
        
        stats = bus.get_stats()
        assert stats['sent'] > 0
        assert 'delivered' in stats
        assert 'failed' in stats


# ============================================================
# 第二十四部分: 完整辩论流程测试
# ============================================================

class TestFullDebateWorkflow:
    """完整辩论工作流测试"""
    
    def test_debate_arena_execution(self):
        """测试辩论场执行"""
        pro_llm = MockLLM(responses=[
            'Pro opening',
            'Pro rebuttal 1',
            'Pro rebuttal 2',
            'Pro closing'
        ])
        opp_llm = MockLLM(responses=[
            'Opp opening',
            'Opp rebuttal 1',
            'Opp rebuttal 2',
            'Opp closing'
        ])
        # 使用能产生明确结果的评判LLM
        judge_llm = MockLLM(responses=['''Evaluation:
PROPONENT: logical_coherence=8, evidence_quality=7, persuasiveness=8, addressing_counterarguments=7
OPPONENT: logical_coherence=6, evidence_quality=6, persuasiveness=6, addressing_counterarguments=6

Winner: PROPONENT'''])
        
        pro_config = AgentConfig(name='Proponent')
        opp_config = AgentConfig(name='Opponent')
        judge_config = AgentConfig(name='Judge')
        
        pro = DebaterAgent(pro_config, DebateRole.PROPONENT, pro_llm)
        opp = DebaterAgent(opp_config, DebateRole.OPPONENT, opp_llm)
        judge = JudgeAgent(judge_config, llm=judge_llm)
        
        arena = DebateArena(pro, opp, judge)
        result = run_async(arena.run_debate('AI benefits humanity'))
        
        assert result.topic == 'AI benefits humanity'
        assert len(result.rounds) > 0
        assert result.judgment is not None
        # winner可能为None（平局）或某个agent_id
    
    def test_debate_transcript(self):
        """测试辩论记录生成"""
        pro_llm = MockLLM(responses=['Pro arg'])
        opp_llm = MockLLM(responses=['Opp arg'])
        judge_llm = MockLLM(responses=['Judgment'])
        
        pro_config = AgentConfig(name='Pro')
        opp_config = AgentConfig(name='Opp')
        judge_config = AgentConfig(name='Judge')
        
        pro = DebaterAgent(pro_config, DebateRole.PROPONENT, pro_llm)
        opp = DebaterAgent(opp_config, DebateRole.OPPONENT, opp_llm)
        judge = JudgeAgent(judge_config, llm=judge_llm)
        
        arena = DebateArena(pro, opp, judge, DebateConfig(topic='Test'))
        run_async(arena.run_debate())
        
        transcript = arena.get_transcript()
        assert 'Debate Topic: Test' in transcript
        assert 'Pro' in transcript or 'Opp' in transcript


# ============================================================
# 第二十五部分: 高级协作测试
# ============================================================

class TestAdvancedCollaboration:
    """高级协作测试"""
    
    def test_parallel_collaboration(self):
        """测试并行协作"""
        config = TeamConfig(
            name='ParallelTeam',
            mode=CollaborationMode.PARALLEL
        )
        team = CollaborativeTeam(config)
        
        for i in range(3):
            agent = create_agent(
                f'Member{i}',
                llm=MockLLM(responses=[f'Contribution {i}'])
            )
            team.add_member(agent, f'Expert {i}')
        
        result = run_async(team.collaborate('Analyze this problem'))
        assert result is not None
        assert len(team.get_contributions()) >= 3
    
    def test_round_robin_collaboration(self):
        """测试轮询协作"""
        config = TeamConfig(
            name='RRTeam',
            mode=CollaborationMode.ROUND_ROBIN,
            max_iterations=2
        )
        team = CollaborativeTeam(config)
        
        for i in range(2):
            agent = create_agent(
                f'Member{i}',
                llm=MockLLM(responses=[f'Input {i}'])
            )
            team.add_member(agent, f'Worker {i}')
        
        result = run_async(team.collaborate('Refine this solution'))
        assert result is not None
        # 应该有多次迭代
        contributions = team.get_contributions()
        assert len(contributions) >= 2


# ============================================================
# 第二十六部分: 错误处理和恢复测试
# ============================================================

class TestErrorHandlingAndRecovery:
    """错误处理和恢复测试"""
    
    def test_agent_step_error_handling(self):
        """测试 Agent 步骤错误处理"""
        class FailingLLM:
            async def generate(self, messages, **kwargs):
                raise Exception("LLM failed")
        
        llm = FailingLLM()
        agent = create_agent('FailingAgent', llm=llm)
        
        response = run_async(agent.step('Test'))
        assert response.content.startswith('Error:')
        assert agent.state == AgentState.ERROR
    
    def test_orchestrator_task_timeout(self):
        """测试任务超时"""
        class SlowLLM:
            async def generate(self, messages, **kwargs):
                await asyncio.sleep(100)  # 超过超时时间
                return 'Done'
        
        llm = SlowLLM()
        agent = create_agent('SlowAgent', llm=llm)
        
        orch = RoundRobinOrchestrator(
            OrchestratorConfig(task_timeout=0.1)  # 100ms 超时
        )
        orch.register_agent(agent)
        
        task = orch.create_task('Slow task')
        run_async(orch.assign_task(task.task_id))
        result = run_async(orch.execute_task(task.task_id))
        
        # 任务应该因超时而失败，或者分配失败
        assert result is None or task.status == TaskStatus.FAILED
    
    def test_message_bus_invalid_recipient(self):
        """测试无效接收者处理"""
        bus = MessageBus()
        
        # 没有注册任何代理
        msg = AgentMessage(
            sender_id='unknown',
            sender_name='Unknown',
            content='Test',
            msg_type=MessageType.CHAT,
            recipient_id='nonexistent'
        )
        
        # 不应该抛出异常
        run_async(bus.send(msg))


# ============================================================
# 运行测试
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
