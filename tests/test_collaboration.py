"""SPEPA 阶段五测试：多Agent协作深化。

测试范围：
1. MessageBus：Agent间通信总线
2. ExperienceFeedback：经验回传与技能提取
3. CollaborationOrchestrator：多协作模式（并行/串行/竞争）
4. SPEPAAgent 集成：collaborate 方法
5. 端到端：多Agent协作完整流程
"""
import numpy as np
import pytest

from spepa.utils.config import get_default_config, SPEPAConfig
from spepa.agent import SPEPAAgent
from spepa.multiagent.communication import MessageBus, Message, MessageType
from spepa.multiagent.experience import (
    ExperienceFeedback, AgentTrajectory, TrajectoryStep
)
from spepa.multiagent.collaboration import (
    CollaborationOrchestrator, CollaborationMode
)
from spepa.multiagent.agent_pool import AgentPool, SubAgent, SubTask
from spepa.skills.skill import Skill, SkillLibrary


# ============================================================
# MessageBus 测试
# ============================================================

class TestMessageBus:
    """通信总线测试。"""

    def test_initialization(self):
        """总线正确初始化。"""
        bus = MessageBus()
        assert bus.total_sent == 0
        assert bus.total_received == 0
        assert len(bus.get_registered_agents()) == 0

    def test_register_agent(self):
        """注册Agent。"""
        bus = MessageBus()
        assert bus.register("agent_1") is True
        assert "agent_1" in bus.get_registered_agents()

    def test_register_duplicate(self):
        """重复注册返回False。"""
        bus = MessageBus()
        bus.register("agent_1")
        assert bus.register("agent_1") is False

    def test_unregister(self):
        """注销Agent。"""
        bus = MessageBus()
        bus.register("agent_1")
        assert bus.unregister("agent_1") is True
        assert "agent_1" not in bus.get_registered_agents()

    def test_publish_broadcast(self):
        """广播消息。"""
        bus = MessageBus()
        bus.register("sender")
        bus.register("receiver_1")
        bus.register("receiver_2")
        msg = Message(
            sender_id="sender",
            msg_type=MessageType.STATE_SHARE,
            content={"key": "value"},
        )
        delivered = bus.publish(msg)
        assert delivered == 2
        assert bus.total_sent == 1

    def test_publish_unicast(self):
        """单播消息。"""
        bus = MessageBus()
        bus.register("sender")
        bus.register("receiver")
        msg = Message(
            sender_id="sender",
            msg_type=MessageType.SKILL_SHARE,
            content={"skill": "test"},
            receiver_id="receiver",
        )
        delivered = bus.publish(msg)
        assert delivered == 1

    def test_receive_message(self):
        """接收消息。"""
        bus = MessageBus()
        bus.register("sender")
        bus.register("receiver")
        msg = Message(
            sender_id="sender",
            msg_type=MessageType.PROGRESS_UPDATE,
            content={"progress": 0.5},
            receiver_id="receiver",
        )
        bus.publish(msg)
        messages = bus.receive("receiver")
        assert len(messages) == 1
        assert messages[0].msg_type == MessageType.PROGRESS_UPDATE
        assert bus.total_received == 1

    def test_receive_empty(self):
        """空队列接收返回空列表。"""
        bus = MessageBus()
        bus.register("agent_1")
        messages = bus.receive("agent_1")
        assert messages == []

    def test_receive_max_messages(self):
        """限制接收消息数。"""
        bus = MessageBus()
        bus.register("sender")
        bus.register("receiver")
        for i in range(5):
            bus.publish(Message(
                sender_id="sender",
                msg_type=MessageType.STATE_SHARE,
                content={"i": i},
                receiver_id="receiver",
            ))
        messages = bus.receive("receiver", max_messages=3)
        assert len(messages) == 3

    def test_peek(self):
        """查看队列中消息数。"""
        bus = MessageBus()
        bus.register("sender")
        bus.register("receiver")
        bus.publish(Message(
            sender_id="sender",
            msg_type=MessageType.STATE_SHARE,
            content={},
            receiver_id="receiver",
        ))
        assert bus.peek("receiver") == 1
        # peek 不取出
        assert bus.peek("receiver") == 1

    def test_publish_unknown_sender(self):
        """未知发送方返回0。"""
        bus = MessageBus()
        bus.register("receiver")
        msg = Message(
            sender_id="unknown",
            msg_type=MessageType.STATE_SHARE,
            content={},
        )
        assert bus.publish(msg) == 0

    def test_clear(self):
        """清空所有队列。"""
        bus = MessageBus()
        bus.register("sender")
        bus.register("receiver")
        bus.publish(Message(
            sender_id="sender",
            msg_type=MessageType.STATE_SHARE,
            content={},
            receiver_id="receiver",
        ))
        bus.clear()
        assert bus.peek("receiver") == 0

    def test_get_state(self):
        """get_state 返回完整状态。"""
        bus = MessageBus()
        state = bus.get_state()
        assert "registered_count" in state
        assert "total_sent" in state
        assert "total_received" in state

    def test_reset(self):
        """reset 清空所有状态。"""
        bus = MessageBus()
        bus.register("agent_1")
        bus.total_sent = 10
        bus.reset()
        assert len(bus.get_registered_agents()) == 0
        assert bus.total_sent == 0

    def test_queue_overflow_drops_oldest(self):
        """队列满时丢弃最旧消息。"""
        bus = MessageBus(max_queue_size=3)
        bus.register("sender")
        bus.register("receiver")
        for i in range(5):
            bus.publish(Message(
                sender_id="sender",
                msg_type=MessageType.STATE_SHARE,
                content={"i": i},
                receiver_id="receiver",
            ))
        # 队列大小为3，应有2个被丢弃
        assert bus.total_dropped == 2
        messages = bus.receive("receiver", max_messages=10)
        assert len(messages) == 3


# ============================================================
# ExperienceFeedback 测试
# ============================================================

class TestExperienceFeedback:
    """经验回传测试。"""

    def test_initialization(self):
        """正确初始化。"""
        config = get_default_config()
        ef = ExperienceFeedback(config)
        assert ef.total_trajectories == 0
        assert ef.total_skills_extracted == 0

    def test_extract_skills_from_successful_trajectory(self):
        """从成功轨迹提取技能。"""
        config = get_default_config()
        ef = ExperienceFeedback(config)
        # 构造成功轨迹
        dim = config.state_dim
        traj = AgentTrajectory(
            agent_id="test_agent",
            task_id="test_task",
            success=True,
            final_error=0.05,
        )
        target = np.ones(dim) * 0.8
        current = np.ones(dim) * 0.2
        for i in range(5):
            action = (target - current) * 0.3
            effect = np.clip(current + action, 0, 1)
            traj.add_step(current, action, effect, reward=0.1)
            current = effect
        skills = ef.extract_skills(traj)
        assert len(skills) > 0
        assert all(isinstance(s, Skill) for s in skills)

    def test_extract_skills_short_trajectory(self):
        """短轨迹不提取技能。"""
        config = get_default_config()
        ef = ExperienceFeedback(config)
        traj = AgentTrajectory(
            agent_id="test", task_id="test",
            success=True, final_error=0.05,
        )
        traj.add_step(np.ones(config.state_dim) * 0.5,
                      np.ones(config.state_dim) * 0.1,
                      np.ones(config.state_dim) * 0.6)
        skills = ef.extract_skills(traj)
        assert skills == []

    def test_extract_skills_failed_high_error(self):
        """失败且高误差不提取。"""
        config = get_default_config()
        ef = ExperienceFeedback(config)
        traj = AgentTrajectory(
            agent_id="test", task_id="test",
            success=False, final_error=0.8,
        )
        for i in range(5):
            traj.add_step(np.ones(config.state_dim) * 0.5,
                          np.ones(config.state_dim) * 0.1,
                          np.ones(config.state_dim) * 0.6)
        skills = ef.extract_skills(traj)
        assert skills == []

    def test_feedback_to_library(self):
        """技能回传到技能库。"""
        config = get_default_config()
        ef = ExperienceFeedback(config)
        library = SkillLibrary(config)
        skill = Skill(
            name="test_skill",
            trigger_state=np.ones(config.state_dim) * 0.5,
            action=np.ones(config.state_dim) * 0.1,
            expected_effect=np.ones(config.state_dim) * 0.05,
            success_count=1,
        )
        merged = ef.feedback_to_library([skill], library)
        assert merged == 1
        assert ef.total_skills_merged == 1

    def test_process_agent(self):
        """处理子Agent完整回传流程。"""
        config = get_default_config()
        ef = ExperienceFeedback(config)
        library = SkillLibrary(config)
        # 创建子Agent
        task = SubTask(
            task_id="test",
            description="test",
            initial_state=np.ones(config.state_dim) * 0.2,
            target_state=np.ones(config.state_dim) * 0.8,
            max_steps=30,
        )
        agent = SubAgent(config, task)
        agent.run()  # 执行完成
        merged = ef.process_agent(agent, library)
        assert ef.total_trajectories == 1
        # 可能提取0个或多个技能
        assert isinstance(merged, int)

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        ef = ExperienceFeedback(config)
        state = ef.get_state()
        assert "total_trajectories" in state
        assert "total_skills_extracted" in state
        assert "extraction_rate" in state

    def test_reset(self):
        """reset 清空状态。"""
        config = get_default_config()
        ef = ExperienceFeedback(config)
        ef.total_trajectories = 5
        ef.total_skills_extracted = 3
        ef.reset()
        assert ef.total_trajectories == 0
        assert ef.total_skills_extracted == 0


# ============================================================
# CollaborationOrchestrator 测试
# ============================================================

class TestCollaborationOrchestrator:
    """协作编排器测试。"""

    def test_initialization(self):
        """正确初始化。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)
        assert orch.total_collaborations == 0
        assert orch.pool is not None
        assert orch.message_bus is not None
        assert orch.experience_feedback is not None

    def test_collaborate_parallel(self):
        """并行协作模式。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)
        initial = np.ones(config.state_dim) * 0.3
        target = np.ones(config.state_dim) * 0.7
        result = orch.collaborate(initial, target, mode=CollaborationMode.PARALLEL)
        assert result["mode"] == "parallel"
        assert "final_state" in result
        assert "success_rate" in result
        assert orch.total_collaborations == 1
        assert orch.mode_stats["parallel"] == 1

    def test_collaborate_sequential(self):
        """串行协作模式。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)
        initial = np.ones(config.state_dim) * 0.2
        target = np.ones(config.state_dim) * 0.8
        result = orch.collaborate(initial, target, mode=CollaborationMode.SEQUENTIAL)
        assert result["mode"] == "sequential"
        assert "final_state" in result
        assert "success_rate" in result
        assert orch.mode_stats["sequential"] == 1

    def test_collaborate_competitive(self):
        """竞争协作模式。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)
        initial = np.ones(config.state_dim) * 0.3
        target = np.ones(config.state_dim) * 0.7
        result = orch.collaborate(
            initial, target,
            mode=CollaborationMode.COMPETITIVE,
            n_competitors=3,
        )
        assert result["mode"] == "competitive"
        assert "final_state" in result
        assert "best_error" in result
        assert orch.mode_stats["competitive"] == 1

    def test_collaborate_with_skill_library(self):
        """协作时回传技能到技能库。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)
        library = SkillLibrary(config)
        initial = np.ones(config.state_dim) * 0.3
        target = np.ones(config.state_dim) * 0.7
        result = orch.collaborate(
            initial, target,
            mode=CollaborationMode.PARALLEL,
            skill_library=library,
        )
        assert "skills_feedback" in result

    def test_collaborate_with_env_fn(self):
        """协作时使用环境函数。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)

        def env_fn(action):
            return np.clip(np.ones(config.state_dim) * 0.5 + action * 0.1, 0, 1)

        initial = np.ones(config.state_dim) * 0.3
        target = np.ones(config.state_dim) * 0.7
        result = orch.collaborate(
            initial, target,
            mode=CollaborationMode.PARALLEL,
            env_fn=env_fn,
        )
        assert "final_state" in result

    def test_collaborate_handles_non_finite(self):
        """协作处理非有限值。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)
        initial = np.ones(config.state_dim) * 0.3
        initial[0] = np.nan
        target = np.ones(config.state_dim) * 0.7
        target[1] = np.inf
        result = orch.collaborate(initial, target, mode=CollaborationMode.PARALLEL)
        assert "final_state" in result
        assert np.all(np.isfinite(result["final_state"]))

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)
        state = orch.get_state()
        assert "total_collaborations" in state
        assert "mode_stats" in state
        assert "pool_state" in state
        assert "message_bus_state" in state
        assert "experience_state" in state

    def test_reset(self):
        """reset 清空状态。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)
        orch.collaborate(
            np.ones(config.state_dim) * 0.3,
            np.ones(config.state_dim) * 0.7,
        )
        orch.reset()
        assert orch.total_collaborations == 0
        assert all(v == 0 for v in orch.mode_stats.values())

    def test_competitive_selects_best(self):
        """竞争模式选择最优结果。"""
        config = get_default_config()
        orch = CollaborationOrchestrator(config)
        initial = np.ones(config.state_dim) * 0.5
        target = np.ones(config.state_dim) * 0.5  # 目标=初始，误差为0
        result = orch.collaborate(
            initial, target,
            mode=CollaborationMode.COMPETITIVE,
            n_competitors=3,
        )
        # 目标就是初始状态，应快速达成
        assert result["best_error"] < 0.5


# ============================================================
# SPEPAAgent 集成测试
# ============================================================

class TestAgentCollaborationIntegration:
    """Agent 协作集成测试。"""

    def test_agent_has_collaboration_orchestrator(self):
        """Agent 包含协作编排器。"""
        agent = SPEPAAgent(get_default_config())
        assert hasattr(agent, 'collaboration_orchestrator')
        assert isinstance(agent.collaboration_orchestrator, CollaborationOrchestrator)

    def test_agent_collaborate_parallel(self):
        """Agent 并行协作。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        target = np.ones(config.state_dim) * 0.7
        result = agent.collaborate(target, mode=CollaborationMode.PARALLEL)
        assert "mode" in result
        assert result["mode"] == "parallel"
        assert "final_state" in result

    def test_agent_collaborate_sequential(self):
        """Agent 串行协作。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        target = np.ones(config.state_dim) * 0.8
        result = agent.collaborate(target, mode=CollaborationMode.SEQUENTIAL)
        assert result["mode"] == "sequential"

    def test_agent_collaborate_competitive(self):
        """Agent 竞争协作。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        target = np.ones(config.state_dim) * 0.7
        result = agent.collaborate(
            target, mode=CollaborationMode.COMPETITIVE, n_competitors=2
        )
        assert result["mode"] == "competitive"

    def test_agent_collaborate_updates_state(self):
        """协作后更新当前状态。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        original_state = agent.current_state.copy()
        target = np.ones(config.state_dim) * 0.7
        agent.collaborate(target, mode=CollaborationMode.PARALLEL)
        # 状态应可能改变
        assert agent.current_state.shape == original_state.shape

    def test_get_state_includes_collaboration(self):
        """get_state 包含协作状态。"""
        agent = SPEPAAgent(get_default_config())
        state = agent.get_state()
        assert "collaboration_orchestrator" in state

    def test_reset_resets_collaboration(self):
        """reset 重置协作编排器。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.collaborate(np.ones(config.state_dim) * 0.7)
        agent.reset()
        assert agent.collaboration_orchestrator.total_collaborations == 0


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEndCollaboration:
    """端到端协作场景测试。"""

    def test_parallel_decomposition_completes(self):
        """并行分解任务完成。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        target = np.ones(config.state_dim) * 0.9
        result = agent.collaborate(target, mode=CollaborationMode.PARALLEL)
        assert result["total_count"] > 0
        assert result["success_rate"] >= 0.0

    def test_sequential_chain_progresses(self):
        """串行链逐步推进。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        initial = agent.current_state.copy()
        target = np.ones(config.state_dim) * 0.9
        result = agent.collaborate(target, mode=CollaborationMode.SEQUENTIAL)
        # 串行模式应有多阶段
        assert result["total_count"] >= 1

    def test_competitive_best_result(self):
        """竞争模式选出最优。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        target = agent.current_state.copy()  # 目标=当前
        result = agent.collaborate(
            target, mode=CollaborationMode.COMPETITIVE, n_competitors=3
        )
        # 目标就是当前状态，最优误差应很小
        assert result.get("best_error", float('inf')) < 0.5

    def test_skill_feedback_enhances_library(self):
        """协作后技能库可能增强。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        initial_skill_count = len(agent.skill_learner.library.skills)
        target = np.ones(config.state_dim) * 0.9
        agent.collaborate(target, mode=CollaborationMode.PARALLEL)
        # 技能库可能增加（取决于子Agent是否成功）
        final_skill_count = len(agent.skill_learner.library.skills)
        assert final_skill_count >= initial_skill_count

    def test_multiple_collaborations_accumulate(self):
        """多次协作累积统计。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        for i in range(3):
            target = np.ones(config.state_dim) * (0.5 + 0.1 * i)
            agent.collaborate(target, mode=CollaborationMode.PARALLEL)
        state = agent.collaboration_orchestrator.get_state()
        assert state["total_collaborations"] == 3

    def test_mixed_mode_collaboration(self):
        """混合模式协作。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 先并行
        agent.collaborate(
            np.ones(config.state_dim) * 0.7,
            mode=CollaborationMode.PARALLEL,
        )
        # 再竞争
        agent.collaborate(
            np.ones(config.state_dim) * 0.3,
            mode=CollaborationMode.COMPETITIVE,
        )
        # 最后串行
        agent.collaborate(
            np.ones(config.state_dim) * 0.8,
            mode=CollaborationMode.SEQUENTIAL,
        )
        state = agent.collaboration_orchestrator.get_state()
        assert state["total_collaborations"] == 3
        assert state["mode_stats"]["parallel"] == 1
        assert state["mode_stats"]["competitive"] == 1
        assert state["mode_stats"]["sequential"] == 1
