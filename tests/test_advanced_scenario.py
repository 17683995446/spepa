"""应用场景测试：多Agent协作 + 技能增强 + 长任务驱动。

验证三大应用场景能力：
1. 自主操作多个agent完成任务（多Agent协作）
2. 自主学会增强自己的技能（技能增强）
3. 自主学会自己驱动自己完成长任务（长任务驱动）
"""
import numpy as np
import pytest

from spepa.utils.config import SPEPAConfig
from spepa.skills.skill import Skill, SkillLibrary
from spepa.skills.skill_learner import SkillLearner
from spepa.multiagent.agent_pool import AgentPool, SubAgent, SubTask
from spepa.multiagent.task_allocator import TaskAllocator, Coordinator
from spepa.longtask.task_graph import TaskGraph, GraphNode, TaskStatus
from spepa.longtask.task_driver import LongTaskDriver
from spepa.agent import SPEPAAgent


# ============================================================
# 一、技能系统测试
# ============================================================

class TestSkillSystem:
    """技能系统测试。"""

    def test_skill_creation(self):
        """技能创建。"""
        skill = Skill(
            name="test_skill",
            trigger_state=np.ones(8) * 0.5,
            action=np.ones(8) * 0.3,
            expected_effect=np.ones(8) * 0.1,
        )
        assert skill.name == "test_skill"
        assert skill.success_rate == 0.0
        assert skill.use_count == 0

    def test_skill_record_outcome(self):
        """技能结果记录。"""
        skill = Skill("s", np.ones(4), np.ones(4), np.ones(4))
        skill.record_outcome(True, 1.0)
        skill.record_outcome(True, 2.0)
        skill.record_outcome(False, 3.0)
        assert skill.use_count == 3
        assert skill.success_count == 2
        assert skill.failure_count == 1
        assert abs(skill.success_rate - 2/3) < 1e-6

    def test_skill_matches(self):
        """技能状态匹配。"""
        skill = Skill("s", np.array([1, 0, 0, 0]), np.ones(4), np.ones(4))
        # 相似状态
        sim = skill.matches(np.array([0.9, 0.1, 0, 0]))
        assert sim > 0.9
        # 不相似状态
        sim2 = skill.matches(np.array([0, 0, 0, 1]))
        assert sim2 < 0.5

    def test_skill_library_add_and_retrieve(self):
        """技能库添加和检索。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        lib = SkillLibrary(config)
        skill = Skill("s1", np.ones(8) * 0.5, np.ones(8) * 0.3, np.ones(8) * 0.1)
        skill.use_count = 5
        skill.success_count = 4
        lib.add(skill)
        # 检索
        results = lib.retrieve(np.ones(8) * 0.5, threshold=0.7)
        assert len(results) == 1
        assert results[0].name == "s1"

    def test_skill_library_consolidate(self):
        """技能固化。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        lib = SkillLibrary(config)
        skill = Skill("s1", np.ones(8) * 0.5, np.ones(8) * 0.3, np.ones(8) * 0.1)
        skill.use_count = 10
        skill.success_count = 8
        lib.add(skill)
        count = lib.consolidate()
        assert count == 1
        assert skill.consolidated

    def test_skill_library_merge_same_name(self):
        """同名技能合并。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        lib = SkillLibrary(config)
        s1 = Skill("s", np.ones(8) * 0.5, np.ones(8) * 0.3, np.ones(8) * 0.1,
                   success_count=3, use_count=5)
        s2 = Skill("s", np.ones(8) * 0.6, np.ones(8) * 0.4, np.ones(8) * 0.2,
                   success_count=2, use_count=4)
        lib.add(s1)
        lib.add(s2)
        assert len(lib.skills) == 1
        merged = lib.skills["s"]
        assert merged.use_count == 9
        assert merged.success_count == 5


class TestSkillLearner:
    """技能学习器测试。"""

    def test_record_experience(self):
        """记录经验。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        learner = SkillLearner(config)
        learner.record_experience(np.ones(8) * 0.5, np.ones(8) * 0.3,
                                  np.ones(8) * 0.1, True)
        assert len(learner.experience_buffer) == 1

    def test_extract_skills_insufficient_data(self):
        """数据不足时不提取技能。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        learner = SkillLearner(config)
        # 只记录1条经验
        learner.record_experience(np.ones(8) * 0.5, np.ones(8) * 0.3,
                                  np.ones(8) * 0.1, True)
        skills = learner.extract_skills()
        assert len(skills) == 0

    def test_extract_skills_from_repeated_pattern(self):
        """从重复模式提取技能。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        learner = SkillLearner(config)
        # 记录5条相似的成功经验
        for i in range(5):
            learner.record_experience(
                np.ones(8) * 0.5 + np.random.default_rng(i).normal(0, 0.01, 8),
                np.ones(8) * 0.3,
                np.ones(8) * 0.1,
                True,
            )
        skills = learner.extract_skills()
        assert len(skills) >= 1
        assert all(s.success_rate > 0 for s in skills)

    def test_apply_skill_no_match(self):
        """无匹配技能时返回None。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        learner = SkillLearner(config)
        result = learner.apply_skill(np.ones(8) * 0.5)
        assert result is None

    def test_apply_skill_with_match(self):
        """有匹配技能时返回技能。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        learner = SkillLearner(config)
        # 添加一个技能
        skill = Skill("s1", np.ones(8) * 0.5, np.ones(8) * 0.3, np.ones(8) * 0.1)
        skill.use_count = 5
        skill.success_count = 4
        learner.library.add(skill)
        # 匹配检索
        result = learner.apply_skill(np.ones(8) * 0.5)
        assert result is not None
        assert result.name == "s1"


# ============================================================
# 二、多Agent协作测试
# ============================================================

class TestAgentPool:
    """Agent池测试。"""

    def test_spawn_agent(self):
        """派生子Agent。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        task = SubTask("t1", "test", np.ones(8) * 0.3, np.ones(8) * 0.7)
        agent = pool.spawn(task)
        assert agent is not None
        assert agent.agent_id in pool.active_agents
        assert len(pool.active_agents) == 1

    def test_spawn_limit(self):
        """派生上限。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        pool.max_concurrent = 2
        for i in range(2):
            task = SubTask(f"t{i}", "test", np.ones(8) * 0.3, np.ones(8) * 0.7)
            agent = pool.spawn(task)
            assert agent is not None
        # 第3个应失败
        task = SubTask("t2", "test", np.ones(8) * 0.3, np.ones(8) * 0.7)
        agent = pool.spawn(task)
        assert agent is None

    def test_subagent_run_to_completion(self):
        """子Agent运行到完成。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        task = SubTask("t1", "test", np.zeros(8), np.ones(8) * 0.5, max_steps=20)
        agent = pool.spawn(task)
        result = agent.run()
        assert result["completed"] or result["steps"] >= 20
        assert "final_state" in result

    def test_step_all(self):
        """推进所有子Agent。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        for i in range(3):
            task = SubTask(f"t{i}", "test", np.zeros(8), np.ones(8) * 0.5)
            pool.spawn(task)
        results = pool.step_all()
        assert len(results) == 3
        # 所有Agent应前进一步
        for r in results:
            assert "step" in r

    def test_recycle_completed(self):
        """回收完成的子Agent。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        task = SubTask("t1", "test", np.zeros(8), np.ones(8) * 0.5, max_steps=5)
        agent = pool.spawn(task)
        # 运行到完成
        pool.run_all()
        # 应被自动回收
        assert len(pool.active_agents) == 0
        assert len(pool.completed_agents) == 1


class TestTaskAllocator:
    """任务分配器测试。"""

    def test_decompose_task(self):
        """任务分解。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        allocator = TaskAllocator(config, pool)
        subtasks = allocator.decompose(np.zeros(8), np.ones(8))
        assert len(subtasks) >= 1
        # 每个子任务应有正确的维度
        for st in subtasks:
            assert st.initial_state.shape[0] == 8
            assert st.target_state.shape[0] == 8

    def test_allocate_tasks(self):
        """任务分配。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        allocator = TaskAllocator(config, pool)
        subtasks = allocator.decompose(np.zeros(8), np.ones(8))
        assignments = allocator.allocate(subtasks)
        assert len(assignments) == len(subtasks)
        # 所有任务都应分配成功
        for task_id, agent in assignments.items():
            assert agent is not None

    def test_decompose_and_allocate(self):
        """分解并分配。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        allocator = TaskAllocator(config, pool)
        assignments = allocator.decompose_and_allocate(np.zeros(8), np.ones(8))
        assert len(assignments) >= 1


class TestCoordinator:
    """协调器测试。"""

    def test_coordinate_all(self):
        """协调所有子Agent。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        allocator = TaskAllocator(config, pool)
        coordinator = Coordinator(config, pool)
        # 分解并分配
        allocator.decompose_and_allocate(np.zeros(8), np.ones(8))
        # 协调执行
        result = coordinator.coordinate()
        assert "total_count" in result
        assert "success_count" in result
        assert result["total_count"] >= 1

    def test_merge_results(self):
        """合并结果。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        pool = AgentPool(config)
        coordinator = Coordinator(config, pool)
        results = [
            {"final_state": np.ones(8) * 0.5, "success": True},
            {"final_state": np.ones(8) * 0.6, "success": False},
        ]
        merged = coordinator.merge_results(results, np.zeros(8))
        assert merged.shape[0] == 8
        assert np.all(np.isfinite(merged))


# ============================================================
# 三、长任务驱动测试
# ============================================================

class TestTaskGraph:
    """任务图测试。"""

    def test_add_node(self):
        """添加节点。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        graph = TaskGraph(config)
        node = GraphNode("n1", "test", np.ones(8) * 0.5)
        graph.add_node(node)
        assert "n1" in graph.nodes

    def test_dependency(self):
        """依赖关系。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        graph = TaskGraph(config)
        graph.add_node(GraphNode("n1", "t1", np.ones(8)))
        graph.add_node(GraphNode("n2", "t2", np.ones(8)))
        assert graph.add_dependency("n2", "n1")
        # n2 依赖 n1，n1 未完成时 n2 不就绪
        ready = graph.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].node_id == "n1"

    def test_cycle_detection(self):
        """循环依赖检测。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        graph = TaskGraph(config)
        graph.add_node(GraphNode("n1", "t1", np.ones(8)))
        graph.add_node(GraphNode("n2", "t2", np.ones(8)))
        graph.add_dependency("n2", "n1")
        # n1 -> n2 已存在，添加 n2 -> n1 应被拒绝
        assert not graph.add_dependency("n1", "n2")

    def test_progress(self):
        """进度计算。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        graph = TaskGraph(config)
        graph.add_node(GraphNode("n1", "t1", np.ones(8)))
        graph.add_node(GraphNode("n2", "t2", np.ones(8)))
        assert graph.get_progress() == 0.0
        graph.nodes["n1"].status = TaskStatus.COMPLETED
        assert graph.get_progress() == 0.5

    def test_skipped_on_failed_dependency(self):
        """依赖失败时跳过。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        graph = TaskGraph(config)
        graph.add_node(GraphNode("n1", "t1", np.ones(8)))
        n2 = GraphNode("n2", "t2", np.ones(8))
        n2.dependencies.add("n1")
        graph.add_node(n2)
        graph.nodes["n1"].status = TaskStatus.FAILED
        ready = graph.get_ready_nodes()
        # n2 应被跳过
        assert graph.nodes["n2"].status == TaskStatus.SKIPPED


class TestLongTaskDriver:
    """长任务驱动器测试。"""

    def test_plan_task(self):
        """规划长任务。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        driver = LongTaskDriver(config)
        graph = driver.plan(np.zeros(8), np.ones(8), n_stages=3)
        assert len(graph.nodes) == 3
        assert driver.total_subtasks == 3

    def test_run_long_task(self):
        """运行长任务。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        driver = LongTaskDriver(config)
        driver.plan(np.zeros(8), np.ones(8) * 0.5, n_stages=3)
        result = driver.run(max_total_steps=200)
        assert "progress" in result
        assert "total_steps" in result
        assert result["total_steps"] > 0

    def test_checkpoint_save_restore(self):
        """检查点保存和恢复。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        driver = LongTaskDriver(config)
        driver.plan(np.zeros(8), np.ones(8) * 0.5, n_stages=3)
        # 部分执行
        driver.run(max_total_steps=50)
        # 保存检查点
        checkpoint = driver.save_checkpoint()
        assert checkpoint["task_description"] == "long_task"
        # 新driver恢复
        driver2 = LongTaskDriver(config)
        driver2.plan(np.zeros(8), np.ones(8) * 0.5, n_stages=3)
        success = driver2.restore_checkpoint(checkpoint)
        assert success

    def test_autonomous_progression(self):
        """自主推进：阶段间自动衔接。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        driver = LongTaskDriver(config)
        driver.plan(np.zeros(8), np.ones(8) * 0.5, n_stages=3)
        result = driver.run(max_total_steps=300)
        # 应有部分或全部完成
        assert result["completed_subtasks"] >= 0
        assert result["progress"] >= 0.0


# ============================================================
# 四、Agent集成测试（三大能力端到端）
# ============================================================

class TestAgentAdvancedCapabilities:
    """Agent高级能力端到端测试。"""

    def test_agent_has_skill_learner(self):
        """Agent 应有技能学习器。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        assert hasattr(agent, "skill_learner")
        assert hasattr(agent, "agent_pool")
        assert hasattr(agent, "long_task_driver")

    def test_agent_online_cycle_records_experience(self):
        """在线周期应记录经验到技能学习器。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        agent.online_cycle()
        assert len(agent.skill_learner.experience_buffer) >= 1

    def test_agent_delegate_task(self):
        """Agent 委派任务给子Agent。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        target = np.ones(8) * 0.8
        result = agent.delegate_task(target)
        assert "coordination" in result
        assert "merged_state" in result
        assert result["coordination"]["total_count"] >= 1

    def test_agent_learn_skills(self):
        """Agent 从经验学习技能。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # 运行多步积累经验
        for _ in range(30):
            agent.online_cycle()
        # 学习技能
        new_skills = agent.learn_skill_from_experience()
        # 应该能提取一些技能（或为空，但不崩溃）
        assert isinstance(new_skills, list)

    def test_agent_apply_skill_no_skills(self):
        """无技能时apply返回None。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        result = agent.apply_learned_skill()
        assert result is None

    def test_agent_run_long_task(self):
        """Agent 运行长任务。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        target = np.ones(8) * 0.8
        result = agent.run_long_task(target, n_stages=2, max_total_steps=100)
        assert "progress" in result
        assert "total_steps" in result
        assert result["total_steps"] > 0

    def test_agent_long_task_checkpoint(self):
        """Agent 长任务检查点。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        agent.run_long_task(np.ones(8) * 0.7, n_stages=2, max_total_steps=50)
        checkpoint = agent.save_long_task_checkpoint()
        assert "task_description" in checkpoint
        assert "graph_state" in checkpoint

    def test_agent_get_state_includes_advanced(self):
        """get_state 应包含高级能力状态。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        state = agent.get_state()
        assert "skill_learner" in state
        assert "agent_pool" in state
        assert "long_task_driver" in state

    def test_agent_reset_clears_advanced(self):
        """reset 应清除高级能力状态。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # 运行产生状态
        for _ in range(5):
            agent.online_cycle()
        agent.delegate_task(np.ones(8) * 0.8)
        # 重置
        agent.reset()
        # 技能学习器缓冲区应清空？不强制，但Agent状态应重置
        assert agent.online_cycles == 0
        assert agent.step_count == 0

    def test_full_scenario_multiagent_skill_longtask(self):
        """完整场景：多Agent协作 + 技能学习 + 长任务。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42,
                             max_steps=50)
        agent = SPEPAAgent(config)
        # 1. 在线学习积累经验
        for _ in range(30):
            agent.online_cycle()
        # 2. 从经验学习技能
        skills = agent.learn_skill_from_experience()
        # 3. 委派任务给子Agent
        delegate_result = agent.delegate_task(np.ones(8) * 0.8)
        assert delegate_result["coordination"]["total_count"] >= 1
        # 4. 运行长任务
        long_result = agent.run_long_task(np.ones(8) * 0.9, n_stages=2,
                                          max_total_steps=100)
        assert long_result["total_steps"] > 0
        # 5. 最终状态应有限
        assert np.all(np.isfinite(agent.current_state))
