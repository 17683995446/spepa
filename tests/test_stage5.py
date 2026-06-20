"""阶段五测试：驱力引擎副作用消除与内稳态重构。

验证：
1. reward 计算是纯函数（不修改内稳态状态）
2. 内稳态更新通过显式 update_homeostasis 调用
3. Homeostasis.simulate 不修改实际状态
4. Homeostasis.reset 重置到目标值
5. Agent.reset 重置内稳态
6. online_cycle 显式更新内稳态
7. 因果性：内稳态由动作驱动，非由 reward 计算
8. 数值稳定性：NaN/Inf 输入处理
"""
import numpy as np
import pytest

from spepa.utils.config import SPEPAConfig
from spepa.core.drive_engine import DriveEngine, Homeostasis
from spepa.agent import SPEPAAgent


class TestRewardPureFunction:
    """验证 reward 计算是纯函数。"""

    def test_reward_safe_no_state_change(self):
        """reward_safe 不应修改 homeostasis.values。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        values_before = de.homeostasis.values.copy()
        de.reward_safe(np.ones(8) * 0.5, np.ones(8) * 0.3)
        values_after = de.homeostasis.values.copy()
        assert np.allclose(values_before, values_after), \
            "reward_safe 不应修改 homeostasis.values"

    def test_reward_safe_idempotent(self):
        """reward_safe 幂等性：多次调用结果相同且状态不变。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        state = np.ones(8) * 0.5
        action = np.ones(8) * 0.3
        rewards = []
        for _ in range(10):
            rewards.append(de.reward_safe(state, action))
        # 所有 reward 应相同
        assert all(abs(r - rewards[0]) < 1e-10 for r in rewards), \
            "reward_safe 不幂等：相同输入产生不同输出"

    def test_total_reward_no_state_change(self):
        """total_reward 不应修改 homeostasis 状态。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        values_before = de.homeostasis.values.copy()
        for _ in range(5):
            de.total_reward(np.ones(8) * 0.5, np.ones(8) * 0.2,
                            np.ones(8) * 0.7, 0.5)
        values_after = de.homeostasis.values.copy()
        assert np.allclose(values_before, values_after), \
            "total_reward 不应修改 homeostasis 状态"

    def test_total_reward_idempotent(self):
        """total_reward 幂等性。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        state = np.ones(8) * 0.5
        action = np.ones(8) * 0.2
        goal = np.ones(8) * 0.7
        r1 = de.total_reward(state, action, goal, 0.5)
        r2 = de.total_reward(state, action, goal, 0.5)
        r3 = de.total_reward(state, action, goal, 0.5)
        assert abs(r1 - r2) < 1e-10 and abs(r2 - r3) < 1e-10, \
            "total_reward 不幂等"

    def test_reward_gain_no_state_change(self):
        """reward_gain 不应修改任何状态。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        values_before = de.homeostasis.values.copy()
        de.reward_gain(np.ones(8) * 0.5, np.ones(8) * 0.3, np.ones(8) * 0.7)
        values_after = de.homeostasis.values.copy()
        assert np.allclose(values_before, values_after)

    def test_reward_eff_no_state_change(self):
        """reward_eff 不应修改任何状态。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        values_before = de.homeostasis.values.copy()
        de.reward_eff(np.ones(8) * 0.3, 0.5)
        values_after = de.homeostasis.values.copy()
        assert np.allclose(values_before, values_after)


class TestExplicitHomeostasisUpdate:
    """验证内稳态通过显式 update_homeostasis 更新。"""

    def test_update_homeostasis_modifies_state(self):
        """update_homeostasis 应修改内稳态状态。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        values_before = de.homeostasis.values.copy()
        de.update_homeostasis(np.ones(8) * 0.5)
        values_after = de.homeostasis.values.copy()
        assert not np.allclose(values_before, values_after), \
            "update_homeostasis 应修改内稳态状态"

    def test_update_homeostasis_returns_deviation(self):
        """update_homeostasis 应返回偏离量。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        deviation = de.update_homeostasis(np.ones(8) * 0.3)
        assert isinstance(deviation, np.ndarray)
        assert deviation.shape[0] == 8
        assert np.all(np.isfinite(deviation))

    def test_update_homeostasis_accumulates(self):
        """多次 update_homeostasis 应累积修改。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        dev1 = de.homeostasis.deviation()
        de.update_homeostasis(np.ones(8) * 0.3)
        dev2 = de.homeostasis.deviation()
        de.update_homeostasis(np.ones(8) * 0.3)
        dev3 = de.homeostasis.deviation()
        # 偏离应累积
        assert dev2 >= dev1
        assert dev3 >= dev2

    def test_update_homeostasis_with_nan_action(self):
        """update_homeostasis 处理 NaN 动作。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        action = np.ones(8) * 0.3
        action[0] = np.nan
        # 不应崩溃
        deviation = de.update_homeostasis(action)
        assert np.all(np.isfinite(deviation)), "NaN 动作应被清洗"

    def test_update_homeostasis_dimension_mismatch(self):
        """update_homeostasis 处理维度不匹配。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        # 16 维动作传入 8 维系统
        deviation = de.update_homeostasis(np.ones(16) * 0.3)
        assert deviation.shape[0] == 8


class TestHomeostasisSimulate:
    """验证 Homeostasis.simulate 不修改实际状态。"""

    def test_simulate_no_mutation(self):
        """simulate 不应修改 values。"""
        homeo = Homeostasis(dim=8, target=0.5)
        values_before = homeo.values.copy()
        homeo.simulate(np.ones(8) * 0.3)
        values_after = homeo.values.copy()
        assert np.allclose(values_before, values_after)

    def test_simulate_returns_deviation(self):
        """simulate 应返回假设的偏离量。"""
        homeo = Homeostasis(dim=8, target=0.5)
        # 当前 values = target = 0.5，模拟 delta=0.3
        # simulated_values = 0.5 + 0.3 = 0.8
        # deviation = 0.8 - 0.5 = 0.3
        dev = homeo.simulate(np.ones(8) * 0.3)
        assert np.allclose(dev, np.ones(8) * 0.3), \
            f"simulate 偏离计算错误: {dev}"

    def test_simulate_idempotent(self):
        """simulate 多次调用结果相同且状态不变。"""
        homeo = Homeostasis(dim=8, target=0.5)
        dev1 = homeo.simulate(np.ones(8) * 0.3)
        dev2 = homeo.simulate(np.ones(8) * 0.3)
        assert np.allclose(dev1, dev2), "simulate 不幂等"

    def test_simulate_with_nan(self):
        """simulate 处理 NaN 输入。"""
        homeo = Homeostasis(dim=8, target=0.5)
        delta = np.ones(8) * 0.3
        delta[0] = np.nan
        dev = homeo.simulate(delta)
        assert np.all(np.isfinite(dev)), "simulate 应清洗 NaN"


class TestHomeostasisReset:
    """验证 Homeostasis.reset。"""

    def test_reset_to_target(self):
        """reset 应重置到目标值。"""
        homeo = Homeostasis(dim=8, target=0.5)
        homeo.update(np.ones(8) * 0.3)
        assert homeo.deviation() > 0
        homeo.reset()
        assert abs(homeo.deviation()) < 1e-10
        assert np.allclose(homeo.values, homeo.target)

    def test_reset_idempotent(self):
        """多次 reset 结果相同。"""
        homeo = Homeostasis(dim=8, target=0.5)
        homeo.update(np.ones(8) * 0.3)
        homeo.reset()
        values_after_1 = homeo.values.copy()
        homeo.reset()
        values_after_2 = homeo.values.copy()
        assert np.allclose(values_after_1, values_after_2)

    def test_drive_engine_reset_homeostasis(self):
        """DriveEngine.reset_homeostasis 应重置内稳态。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        de.update_homeostasis(np.ones(8) * 0.5)
        assert de.homeostasis.deviation() > 0
        de.reset_homeostasis()
        assert abs(de.homeostasis.deviation()) < 1e-10


class TestAgentHomeostasisIntegration:
    """验证 Agent 集成后的内稳态行为。"""

    def test_agent_reset_resets_homeostasis(self):
        """Agent.reset 应重置内稳态。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        for _ in range(20):
            agent.online_cycle()
        assert agent.drive_engine.homeostasis.deviation() > 0
        agent.reset()
        assert abs(agent.drive_engine.homeostasis.deviation()) < 1e-10

    def test_online_cycle_explicit_homeostasis_update(self):
        """online_cycle 应显式更新内稳态。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # 记录 update_homeostasis 调用
        call_count = [0]
        original_update = agent.drive_engine.update_homeostasis

        def counting_update(action):
            call_count[0] += 1
            return original_update(action)

        agent.drive_engine.update_homeostasis = counting_update
        agent.online_cycle()
        assert call_count[0] == 1, \
            f"online_cycle 应调用 update_homeostasis 一次，实际 {call_count[0]} 次"

    def test_reward_computation_no_homeostasis_mutation(self):
        """Agent 运行时 reward 计算不修改内稳态。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # 在 reward 计算前后检查内稳态
        # online_cycle 中 reward 计算在 step 6，update_homeostasis 在 step 6.5
        # 我们验证：reward 计算本身不修改状态
        de = agent.drive_engine
        values_before = de.homeostasis.values.copy()
        # 直接调用 total_reward（模拟 reward 计算部分）
        de.total_reward(agent.current_state, np.ones(8) * 0.2,
                        np.ones(8) * 0.7, 0.5)
        values_after_reward = de.homeostasis.values.copy()
        # reward 计算不应修改状态
        assert np.allclose(values_before, values_after_reward), \
            "reward 计算不应修改内稳态状态"

    def test_homeostasis_deviation_metric_tracked(self):
        """homeostasis_deviation 指标应被跟踪。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        agent.online_cycle()
        assert "homeostasis_deviation" in agent.metrics._history
        assert len(agent.metrics._history["homeostasis_deviation"]) > 0

    def test_long_run_homeostasis_stable(self):
        """长时间运行内稳态应保持稳定（不发散）。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        max_deviation = 0.0
        for _ in range(200):
            agent.online_cycle()
            dev = agent.drive_engine.homeostasis.deviation()
            max_deviation = max(max_deviation, dev)
        # 内稳态偏离不应过大（动作幅值被裁剪到 [-1,1]，0.1 倍后影响有限）
        assert max_deviation < 50.0, \
            f"内稳态偏离过大: {max_deviation}"
        # 所有值应在 bounds [0, 1] 内
        assert np.all(agent.drive_engine.homeostasis.values >= 0 - 1e-6)
        assert np.all(agent.drive_engine.homeostasis.values <= 1 + 1e-6)
