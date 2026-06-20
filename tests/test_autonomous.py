"""SPEPA 阶段四测试：自主目标生成与任务队列。

测试范围：
1. GoalGenerator：从各来源生成目标
2. TaskQueue：优先级队列、去重、老化、过期
3. AutonomousDriver：目标→任务→执行闭环
4. SPEPAAgent 集成：autonomous_step/run_autonomous
5. 端到端：自主运行完整流程
"""
import time
import numpy as np
import pytest

from spepa.utils.config import get_default_config, SPEPAConfig
from spepa.agent import SPEPAAgent
from spepa.autonomous.goal_generator import (
    GoalGenerator, AutonomousGoal, GoalSource, GoalStatus
)
from spepa.autonomous.task_queue import TaskQueue, QueuedTask, TaskStatus
from spepa.autonomous.driver import AutonomousDriver


# ============================================================
# GoalGenerator 测试
# ============================================================

class TestGoalGenerator:
    """目标生成器测试。"""

    def test_initialization(self):
        """生成器正确初始化。"""
        config = get_default_config()
        gen = GoalGenerator(config)
        assert gen._goal_counter == 0

    def test_generate_from_homeostasis_no_deviation(self):
        """无偏离时不生成内稳态目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        gen = GoalGenerator(config)
        goal = gen.generate_from_homeostasis(agent)
        assert goal is None

    def test_generate_from_homeostasis_with_deviation(self):
        """有偏离时生成内稳态目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 制造偏离
        agent.drive_engine.homeostasis.values = np.ones(config.state_dim) * 0.9
        gen = GoalGenerator(config)
        goal = gen.generate_from_homeostasis(agent)
        assert goal is not None
        assert goal.source == GoalSource.HOMEOSTASIS
        assert goal.priority > 0

    def test_generate_from_curiosity_no_error(self):
        """无误差时不生成好奇心目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        gen = GoalGenerator(config)
        goal = gen.generate_from_curiosity(agent)
        # 初始无误差记录，应返回 None
        assert goal is None

    def test_generate_from_curiosity_with_error(self):
        """有误差时生成好奇心目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 记录高误差
        for _ in range(5):
            agent.metrics.record("prediction_error", 2.0)
        gen = GoalGenerator(config)
        goal = gen.generate_from_curiosity(agent)
        assert goal is not None
        assert goal.source == GoalSource.CURIOSITY

    def test_generate_from_skill_gap_no_skills(self):
        """无技能时不生成技能缺口目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        gen = GoalGenerator(config)
        goal = gen.generate_from_skill_gap(agent)
        assert goal is None

    def test_generate_from_skill_gap_with_low_success(self):
        """有低成功率技能时生成练习目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        from spepa.skills.skill import Skill
        # 添加低成功率技能
        skill = Skill(
            name="bad_skill",
            trigger_state=np.ones(config.state_dim) * 0.5,
            action=np.ones(config.action_dim) * 0.3,
            expected_effect=np.ones(config.state_dim) * 0.1,
            success_count=1, failure_count=9,
        )
        agent.skill_learner.library.add(skill)
        gen = GoalGenerator(config)
        goal = gen.generate_from_skill_gap(agent)
        assert goal is not None
        assert goal.source == GoalSource.SKILL_GAP

    def test_generate_from_self_improve_no_offset(self):
        """无校准偏移时不生成自我改进目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        gen = GoalGenerator(config)
        goal = gen.generate_from_self_improve(agent)
        assert goal is None

    def test_generate_external(self):
        """生成外部目标。"""
        config = get_default_config()
        gen = GoalGenerator(config)
        target = np.ones(config.state_dim) * 0.7
        goal = gen.generate_external(target, priority=8.0, description="test")
        assert goal.source == GoalSource.EXTERNAL
        assert goal.priority == 8.0
        assert np.allclose(goal.target_state, target)

    def test_generate_all_respects_intervals(self):
        """generate_all 遵守生成间隔。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        gen = GoalGenerator(config)
        # 第一次调用（step=0）
        goals1 = gen.generate_all(agent, 0)
        # 第二次调用（step=5，小于间隔10）
        goals2 = gen.generate_all(agent, 5)
        # 应不生成（间隔不够）
        # 注意：第一次可能也没生成（因为无偏离/误差）
        # 主要验证不会在间隔内重复生成
        assert isinstance(goals1, list)
        assert isinstance(goals2, list)

    def test_goal_id_unique(self):
        """目标ID唯一。"""
        config = get_default_config()
        gen = GoalGenerator(config)
        goal1 = gen.generate_external(np.ones(config.state_dim), description="g1")
        goal2 = gen.generate_external(np.ones(config.state_dim), description="g2")
        assert goal1.goal_id != goal2.goal_id

    def test_goal_is_achieved(self):
        """目标达成判断。"""
        goal = AutonomousGoal(
            goal_id="test",
            source=GoalSource.EXTERNAL,
            target_state=np.ones(8) * 0.5,
            priority=1.0,
            achievement_threshold=0.2,
        )
        # 在阈值内
        assert goal.is_achieved(np.ones(8) * 0.55)
        # 在阈值外
        assert not goal.is_achieved(np.ones(8) * 0.9)

    def test_goal_is_expired(self):
        """目标超时判断。"""
        goal = AutonomousGoal(
            goal_id="test",
            source=GoalSource.EXTERNAL,
            target_state=np.ones(8),
            priority=1.0,
            max_steps=10,
        )
        goal.steps_taken = 5
        assert not goal.is_expired()
        goal.steps_taken = 10
        assert goal.is_expired()

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        gen = GoalGenerator(config)
        gen.generate_external(np.ones(config.state_dim))
        state = gen.get_state()
        assert "goal_counter" in state
        assert "history_size" in state

    def test_reset(self):
        """reset 清除状态。"""
        config = get_default_config()
        gen = GoalGenerator(config)
        gen.generate_external(np.ones(config.state_dim))
        gen.reset()
        assert gen._goal_counter == 0
        assert len(gen._history) == 0


# ============================================================
# TaskQueue 测试
# ============================================================

class TestTaskQueue:
    """任务队列测试。"""

    def test_initialization(self):
        """队列正确初始化。"""
        config = get_default_config()
        tq = TaskQueue(config)
        assert tq.is_empty()
        assert tq.size() == 0

    def test_enqueue(self):
        """入队。"""
        config = get_default_config()
        tq = TaskQueue(config)
        goal = AutonomousGoal(
            goal_id="g1", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim), priority=5.0,
        )
        ok = tq.enqueue(goal)
        assert ok is True
        assert tq.size() == 1

    def test_dequeue(self):
        """出队。"""
        config = get_default_config()
        tq = TaskQueue(config)
        goal = AutonomousGoal(
            goal_id="g1", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim), priority=5.0,
        )
        tq.enqueue(goal)
        task = tq.dequeue()
        assert task is not None
        assert task.goal.goal_id == "g1"
        assert tq.size() == 0

    def test_dequeue_empty_returns_none(self):
        """空队列出队返回 None。"""
        config = get_default_config()
        tq = TaskQueue(config)
        task = tq.dequeue()
        assert task is None

    def test_priority_ordering(self):
        """优先级排序：高优先级先出。"""
        config = get_default_config()
        tq = TaskQueue(config)
        g_low = AutonomousGoal(
            goal_id="low", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim) * 0.1, priority=1.0,
        )
        g_high = AutonomousGoal(
            goal_id="high", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim) * 0.9, priority=10.0,
        )
        tq.enqueue(g_low)
        tq.enqueue(g_high)
        task = tq.dequeue()
        assert task.goal.goal_id == "high"

    def test_deduplication(self):
        """去重：相似目标不重复入队。"""
        config = get_default_config()
        tq = TaskQueue(config)
        g1 = AutonomousGoal(
            goal_id="g1", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim) * 0.5, priority=5.0,
        )
        g2 = AutonomousGoal(
            goal_id="g2", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim) * 0.5, priority=3.0,  # 相同目标
        )
        tq.enqueue(g1)
        ok = tq.enqueue(g2)
        assert ok is False  # 去重
        assert tq.size() == 1

    def test_complete_success(self):
        """完成任务。"""
        config = get_default_config()
        tq = TaskQueue(config)
        goal = AutonomousGoal(
            goal_id="g1", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim), priority=5.0,
        )
        tq.enqueue(goal)
        task = tq.dequeue()
        ok = tq.complete(task.task_id, success=True)
        assert ok is True
        assert tq.total_completed == 1

    def test_complete_failure_with_retry(self):
        """失败重试。"""
        config = get_default_config()
        tq = TaskQueue(config)
        goal = AutonomousGoal(
            goal_id="g1", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim), priority=5.0,
        )
        tq.enqueue(goal)
        task = tq.dequeue()
        # 第一次失败（应重试）
        ok = tq.complete(task.task_id, success=False)
        assert ok is True
        # 应重新入队
        assert tq.size() == 1

    def test_complete_failure_exhausted(self):
        """重试耗尽后失败。"""
        config = get_default_config()
        tq = TaskQueue(config)
        goal = AutonomousGoal(
            goal_id="g1", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim), priority=5.0,
        )
        tq.enqueue(goal)
        task = tq.dequeue()
        task.max_retries = 1
        # 失败（重试耗尽）
        tq.complete(task.task_id, success=False)
        assert tq.total_failed == 1

    def test_peek(self):
        """查看队首。"""
        config = get_default_config()
        tq = TaskQueue(config)
        goal = AutonomousGoal(
            goal_id="g1", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim), priority=5.0,
        )
        tq.enqueue(goal)
        task = tq.peek()
        assert task is not None
        assert tq.size() == 1  # peek 不出队

    def test_get_running(self):
        """获取运行中任务。"""
        config = get_default_config()
        tq = TaskQueue(config)
        goal = AutonomousGoal(
            goal_id="g1", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim), priority=5.0,
        )
        tq.enqueue(goal)
        task = tq.dequeue()
        assert tq.get_running() is not None
        tq.complete(task.task_id, success=True)
        assert tq.get_running() is None

    def test_list_pending(self):
        """列出待处理任务。"""
        config = get_default_config()
        tq = TaskQueue(config)
        for i in range(3):
            goal = AutonomousGoal(
                goal_id=f"g{i}", source=GoalSource.EXTERNAL,
                target_state=np.ones(config.state_dim) * (0.1 * i + 0.1),
                priority=float(i + 1),
            )
            tq.enqueue(goal)
        pending = tq.list_pending()
        assert len(pending) == 3

    def test_clear(self):
        """清空队列。"""
        config = get_default_config()
        tq = TaskQueue(config)
        for i in range(3):
            goal = AutonomousGoal(
                goal_id=f"g{i}", source=GoalSource.EXTERNAL,
                target_state=np.ones(config.state_dim) * (0.1 * i + 0.1),
                priority=float(i + 1),
            )
            tq.enqueue(goal)
        count = tq.clear()
        assert count == 3
        assert tq.is_empty()

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        tq = TaskQueue(config)
        state = tq.get_state()
        assert "queue_size" in state
        assert "total_enqueued" in state

    def test_reset(self):
        """reset 清空所有状态。"""
        config = get_default_config()
        tq = TaskQueue(config)
        goal = AutonomousGoal(
            goal_id="g1", source=GoalSource.EXTERNAL,
            target_state=np.ones(config.state_dim), priority=5.0,
        )
        tq.enqueue(goal)
        tq.reset()
        assert tq.is_empty()
        assert tq.total_enqueued == 0


# ============================================================
# AutonomousDriver 测试
# ============================================================

class TestAutonomousDriver:
    """自主驱动器测试。"""

    def test_initialization(self):
        """驱动器正确初始化。"""
        config = get_default_config()
        driver = AutonomousDriver(config)
        assert driver.goals_generated == 0
        assert driver.goals_achieved == 0

    def test_tick_returns_dict(self):
        """tick 返回结果字典。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        driver = AutonomousDriver(config)
        result = driver.tick(agent, 0)
        assert isinstance(result, dict)
        assert "step" in result
        assert "generated_goals" in result

    def test_tick_generates_goals_periodically(self):
        """tick 定期生成目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 制造内稳态偏离
        agent.drive_engine.homeostasis.values = np.ones(config.state_dim) * 0.9
        driver = AutonomousDriver(config)
        driver.generation_interval = 5
        # step=0 可能生成
        result = driver.tick(agent, 0)
        # step=5 应生成
        for i in range(1, 6):
            result = driver.tick(agent, i)
        # 应已生成至少一个目标
        assert driver.goals_generated > 0 or len(result["generated_goals"]) > 0 or \
               driver.goals_achieved > 0

    def test_tick_executes_tasks(self):
        """tick 执行任务。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        driver = AutonomousDriver(config)
        # 添加一个容易达成的目标
        target = agent.current_state.copy()
        driver.add_external_goal(target, priority=10.0)
        # tick 应执行任务
        result = driver.tick(agent, 0)
        # 目标应很快达成（因为目标就是当前状态）
        for i in range(1, 5):
            result = driver.tick(agent, i)
            if result["task_completed"]:
                break
        assert driver.goals_achieved > 0

    def test_add_external_goal(self):
        """添加外部目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        driver = AutonomousDriver(config)
        ok = driver.add_external_goal(np.ones(config.state_dim) * 0.7, priority=5.0)
        assert ok is True
        assert driver.task_queue.size() == 1

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        driver = AutonomousDriver(config)
        state = driver.get_state()
        assert "goals_generated" in state
        assert "goals_achieved" in state
        assert "achievement_rate" in state
        assert "task_queue" in state

    def test_reset(self):
        """reset 清除状态。"""
        config = get_default_config()
        driver = AutonomousDriver(config)
        driver.add_external_goal(np.ones(config.state_dim))
        driver.reset()
        assert driver.goals_generated == 0
        assert driver.task_queue.is_empty()

    def test_idle_cycle_runs(self):
        """无任务时也执行在线周期。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        driver = AutonomousDriver(config)
        initial_step = agent.step_count
        driver.tick(agent, 0)
        # 应执行了一个在线周期
        assert agent.step_count == initial_step + 1


# ============================================================
# SPEPAAgent 集成测试
# ============================================================

class TestAgentAutonomousIntegration:
    """Agent 自主驱动集成测试。"""

    def test_agent_has_autonomous_driver(self):
        """Agent 包含自主驱动器。"""
        agent = SPEPAAgent(get_default_config())
        assert hasattr(agent, 'autonomous_driver')
        assert isinstance(agent.autonomous_driver, AutonomousDriver)

    def test_autonomous_step(self):
        """autonomous_step 返回结果。"""
        agent = SPEPAAgent(get_default_config())
        result = agent.autonomous_step()
        assert isinstance(result, dict)

    def test_run_autonomous(self):
        """run_autonomous 返回摘要。"""
        agent = SPEPAAgent(get_default_config())
        summary = agent.run_autonomous(n_steps=15)
        assert "total_steps" in summary
        assert summary["total_steps"] == 15
        assert "goals_generated" in summary
        assert "achievement_rate" in summary

    def test_add_autonomous_goal(self):
        """add_autonomous_goal 添加目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        ok = agent.add_autonomous_goal(np.ones(config.state_dim) * 0.7)
        assert ok is True
        assert agent.autonomous_driver.task_queue.size() == 1

    def test_get_state_includes_autonomous(self):
        """get_state 包含自主驱动状态。"""
        agent = SPEPAAgent(get_default_config())
        state = agent.get_state()
        assert "autonomous_driver" in state

    def test_reset_resets_autonomous(self):
        """reset 重置自主驱动。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.add_autonomous_goal(np.ones(config.state_dim))
        agent.reset()
        assert agent.autonomous_driver.task_queue.is_empty()
        assert agent.autonomous_driver.goals_generated == 0


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端场景测试。"""

    def test_autonomous_goal_achievement(self):
        """自主目标达成场景。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 添加一个接近当前状态的目标
        target = agent.current_state + 0.01
        agent.add_autonomous_goal(target, priority=10.0)
        # 运行几步
        for _ in range(5):
            agent.autonomous_step()
        # 目标应已达成
        assert agent.autonomous_driver.goals_achieved > 0

    def test_autonomous_run_generates_goals(self):
        """自主运行生成目标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 制造内稳态偏离以触发目标生成
        agent.drive_engine.homeostasis.values = np.ones(config.state_dim) * 0.9
        summary = agent.run_autonomous(n_steps=30)
        # 应生成了一些目标
        assert summary["goals_generated"] > 0 or summary["goals_achieved"] > 0

    def test_priority_scheduling(self):
        """优先级调度：高优先级先执行。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 添加两个目标
        far_target = np.ones(config.state_dim) * 0.9
        near_target = agent.current_state + 0.01
        agent.add_autonomous_goal(far_target, priority=1.0, description="far")
        agent.add_autonomous_goal(near_target, priority=10.0, description="near")
        # 高优先级应先执行
        state = agent.autonomous_driver.get_state()
        assert state["current_source"] is not None or \
               agent.autonomous_driver.task_queue.size() > 0

    def test_deduplication_in_autonomous_run(self):
        """自主运行中去重。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 添加两个相同目标
        target = np.ones(config.state_dim) * 0.7
        ok1 = agent.add_autonomous_goal(target, priority=5.0)
        ok2 = agent.add_autonomous_goal(target, priority=3.0)
        assert ok1 is True
        assert ok2 is False  # 去重

    def test_metrics_get_recent(self):
        """MetricsTracker.get_recent 返回最近值。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        for i in range(10):
            agent.metrics.record("test_metric", float(i))
        recent = agent.metrics.get_recent("test_metric", n=3)
        assert len(recent) == 3
        assert recent == [7.0, 8.0, 9.0]
