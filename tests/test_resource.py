"""SPEPA 阶段三测试：资源自调节与背压。

测试范围：
1. ResourceController：PID 控制器、节流信号计算
2. Backpressure：背压信号传播、聚合、衰减
3. DegradationMatrix：降级策略矩阵、级别映射
4. SPEPAAgent 集成：资源调节集成到在线闭环
5. PerpetualRunner 集成：软重启重置资源组件
6. 端到端：高负载下降级、低负载时恢复
"""
import time
import numpy as np
import pytest

from spepa.utils.config import get_default_config, SPEPAConfig
from spepa.agent import SPEPAAgent, ResourceMonitor
from spepa.resource.controller import ResourceController, ThrottleSignal
from spepa.resource.backpressure import Backpressure, BackpressureLevel, BackpressureSignal
from spepa.resource.degradation import DegradationMatrix, DegradationLevel, DegradationParams


# ============================================================
# ResourceController 测试
# ============================================================

class TestResourceController:
    """PID 式资源控制器测试。"""

    def test_initialization(self):
        """控制器正确初始化。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        assert ctrl.cpu_target == config.resources.cpu_max_percent
        assert ctrl.memory_target == config.resources.memory_max_percent
        assert ctrl.current_signal.throttle == 0.0

    def test_update_returns_throttle_signal(self):
        """update 返回 ThrottleSignal。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        signal = ctrl.update(10.0, 10.0)
        assert isinstance(signal, ThrottleSignal)

    def test_low_load_no_throttle(self):
        """低负载时无节流。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        signal = ctrl.update(5.0, 5.0)
        assert signal.throttle < 0.1
        assert signal.n_policies == 8

    def test_high_load_throttle(self):
        """高负载时节流。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        signal = ctrl.update(50.0, 50.0)
        assert signal.throttle > 0.5
        assert signal.n_policies <= 2

    def test_very_high_load_max_throttle(self):
        """极高负载时最大节流。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        signal = ctrl.update(95.0, 95.0)
        assert signal.throttle >= 0.85
        assert signal.skip_learning is True

    def test_throttle_signal_n_policies_decreases_with_load(self):
        """策略数随负载增加而递减。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        # 低负载
        s_low = ctrl.update(5.0, 5.0)
        # 中负载
        s_mid = ctrl.update(25.0, 25.0)
        # 重置后高负载
        ctrl.reset()
        s_high = ctrl.update(60.0, 60.0)
        assert s_low.n_policies >= s_mid.n_policies
        assert s_mid.n_policies >= s_high.n_policies

    def test_throttle_signal_skip_learning_at_high_load(self):
        """高负载时跳过学习。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        signal = ctrl.update(90.0, 90.0)
        assert signal.skip_learning is True

    def test_throttle_signal_skip_offline_at_moderate_load(self):
        """中高负载时跳过离线复盘。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        signal = ctrl.update(50.0, 50.0)
        assert signal.skip_offline is True

    def test_pid_integral_accumulates(self):
        """PID 积分项累积。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        # 持续高负载
        for _ in range(10):
            ctrl.update(50.0, 50.0)
        # 积分项应为负（持续超限）
        assert ctrl._integral_cpu < 0
        assert ctrl._integral_mem < 0

    def test_pid_integral_anti_windup(self):
        """积分抗饱和：积分项有上限。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        # 极长时间高负载
        for _ in range(100):
            ctrl.update(99.0, 99.0)
        # 积分项不应无限增长
        assert ctrl._integral_cpu >= -50.0
        assert ctrl._integral_mem >= -50.0

    def test_reset_clears_state(self):
        """reset 清除控制器状态。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        ctrl.update(50.0, 50.0)
        ctrl.reset()
        assert ctrl._integral_cpu == 0.0
        assert ctrl.current_signal.throttle == 0.0

    def test_history_recorded(self):
        """历史记录被保存。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        for _ in range(5):
            ctrl.update(10.0, 10.0)
        history = ctrl.get_history()
        assert len(history) == 5

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        ctrl.update(10.0, 10.0)
        state = ctrl.get_state()
        assert "cpu_usage" in state
        assert "throttle" in state
        assert "n_policies" in state

    def test_throttle_signal_is_throttled_property(self):
        """ThrottleSignal.is_throttled 属性。"""
        sig1 = ThrottleSignal(throttle=0.0)
        sig2 = ThrottleSignal(throttle=0.5)
        assert not sig1.is_throttled
        assert sig2.is_throttled


# ============================================================
# Backpressure 测试
# ============================================================

class TestBackpressure:
    """背压机制测试。"""

    def test_initialization(self):
        """背压管理器正确初始化。"""
        config = get_default_config()
        bp = Backpressure(config)
        assert bp.get_pressure() == BackpressureLevel.NONE

    def test_signal_sets_level(self):
        """signal 设置背压级别。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("perception", BackpressureLevel.MEDIUM, "test")
        assert bp.get_pressure() == BackpressureLevel.MEDIUM

    def test_multiple_signals_aggregate_max(self):
        """多个信号聚合取最大值。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("perception", BackpressureLevel.LOW)
        bp.signal("learning", BackpressureLevel.HIGH)
        assert bp.get_pressure() == BackpressureLevel.HIGH

    def test_clear_signal(self):
        """清除信号后级别降低。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("perception", BackpressureLevel.HIGH)
        bp.signal("learning", BackpressureLevel.LOW)
        bp.clear_signal("perception")
        assert bp.get_pressure() == BackpressureLevel.LOW

    def test_clear_all_signals(self):
        """清除所有信号后回到 NONE。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("perception", BackpressureLevel.HIGH)
        bp.clear_signal("perception")
        assert bp.get_pressure() == BackpressureLevel.NONE

    def test_is_overloaded(self):
        """is_overloaded 在 MEDIUM 及以上为 True。"""
        config = get_default_config()
        bp = Backpressure(config)
        assert not bp.is_overloaded()
        bp.signal("test", BackpressureLevel.MEDIUM)
        assert bp.is_overloaded()

    def test_is_critical(self):
        """is_critical 在 CRITICAL 时为 True。"""
        config = get_default_config()
        bp = Backpressure(config)
        assert not bp.is_critical()
        bp.signal("test", BackpressureLevel.CRITICAL)
        assert bp.is_critical()

    def test_get_pressure_for_excludes_self(self):
        """get_pressure_for 排除自身的信号。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("perception", BackpressureLevel.HIGH)
        # perception 查询时不应看到自己的信号
        pressure = bp.get_pressure_for("perception")
        assert pressure == BackpressureLevel.NONE

    def test_get_pressure_for_includes_others(self):
        """get_pressure_for 包含其他组件的信号。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("perception", BackpressureLevel.HIGH)
        pressure = bp.get_pressure_for("learning")
        assert pressure >= BackpressureLevel.LOW  # 衰减后可能降低

    def test_backpressure_level_ordering(self):
        """背压级别有序。"""
        assert BackpressureLevel.NONE < BackpressureLevel.LOW
        assert BackpressureLevel.LOW < BackpressureLevel.MEDIUM
        assert BackpressureLevel.MEDIUM < BackpressureLevel.HIGH
        assert BackpressureLevel.HIGH < BackpressureLevel.CRITICAL

    def test_get_active_sources(self):
        """get_active_sources 返回活跃信号源。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("perception", BackpressureLevel.MEDIUM)
        bp.signal("learning", BackpressureLevel.HIGH)
        sources = bp.get_active_sources()
        assert "perception" in sources
        assert "learning" in sources

    def test_reset(self):
        """reset 清除所有信号。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("test", BackpressureLevel.HIGH)
        bp.reset()
        assert bp.get_pressure() == BackpressureLevel.NONE

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("test", BackpressureLevel.MEDIUM)
        state = bp.get_state()
        assert state["current_level"] == "MEDIUM"
        assert state["is_overloaded"] is True

    def test_signal_with_reason(self):
        """信号带原因。"""
        config = get_default_config()
        bp = Backpressure(config)
        bp.signal("perception", BackpressureLevel.HIGH, "high_error")
        sources = bp.get_active_sources()
        assert "perception" in sources


# ============================================================
# DegradationMatrix 测试
# ============================================================

class TestDegradationMatrix:
    """降级策略矩阵测试。"""

    def test_initialization(self):
        """矩阵正确初始化。"""
        dm = DegradationMatrix()
        assert dm.current_level == DegradationLevel.L0_NORMAL

    def test_matrix_has_all_levels(self):
        """矩阵包含所有级别。"""
        dm = DegradationMatrix()
        for level in DegradationLevel:
            assert level in dm.MATRIX

    def test_get_params(self):
        """get_params 返回指定级别的参数。"""
        dm = DegradationMatrix()
        params = dm.get_params(DegradationLevel.L0_NORMAL)
        assert isinstance(params, DegradationParams)
        assert params.n_policies == 8
        assert params.learning_enabled is True

    def test_l4_critical_params(self):
        """L4 临界级别参数最小化。"""
        dm = DegradationMatrix()
        params = dm.get_params(DegradationLevel.L4_CRITICAL)
        assert params.n_policies == 1
        assert params.learning_enabled is False
        assert params.offline_enabled is False
        assert params.memory_consolidation is False

    def test_level_from_throttle(self):
        """从节流信号映射降级级别。"""
        dm = DegradationMatrix()
        assert dm.level_from_throttle(0.0) == DegradationLevel.L0_NORMAL
        assert dm.level_from_throttle(0.2) == DegradationLevel.L1_MILD
        assert dm.level_from_throttle(0.5) == DegradationLevel.L2_MODERATE
        assert dm.level_from_throttle(0.7) == DegradationLevel.L3_SEVERE
        assert dm.level_from_throttle(0.9) == DegradationLevel.L4_CRITICAL

    def test_level_from_load(self):
        """从负载率映射降级级别。"""
        dm = DegradationMatrix()
        assert dm.level_from_load(0.5, 1.0) == DegradationLevel.L0_NORMAL
        assert dm.level_from_load(0.8, 1.0) == DegradationLevel.L1_MILD
        assert dm.level_from_load(0.95, 1.0) == DegradationLevel.L2_MODERATE
        assert dm.level_from_load(1.1, 1.0) == DegradationLevel.L3_SEVERE
        assert dm.level_from_load(1.5, 1.0) == DegradationLevel.L4_CRITICAL

    def test_update_from_throttle(self):
        """update_from_throttle 更新级别并返回参数。"""
        dm = DegradationMatrix()
        params = dm.update_from_throttle(0.5)
        assert dm.current_level == DegradationLevel.L2_MODERATE
        assert params.level == DegradationLevel.L2_MODERATE

    def test_set_level(self):
        """set_level 设置当前级别。"""
        dm = DegradationMatrix()
        params = dm.set_level(DegradationLevel.L3_SEVERE)
        assert dm.current_level == DegradationLevel.L3_SEVERE
        assert params.n_policies == 2

    def test_reset(self):
        """reset 回到正常级别。"""
        dm = DegradationMatrix()
        dm.set_level(DegradationLevel.L4_CRITICAL)
        dm.reset()
        assert dm.current_level == DegradationLevel.L0_NORMAL

    def test_degradation_progressive(self):
        """降级是渐进的：策略数递减。"""
        dm = DegradationMatrix()
        policies = [dm.get_params(level).n_policies
                    for level in DegradationLevel]
        # 应单调递减
        for i in range(len(policies) - 1):
            assert policies[i] >= policies[i + 1]

    def test_get_state(self):
        """get_state 返回完整状态。"""
        dm = DegradationMatrix()
        state = dm.get_state()
        assert "current_level" in state
        assert "n_policies" in state
        assert "learning_enabled" in state

    def test_get_current_params(self):
        """get_current_params 返回当前级别参数。"""
        dm = DegradationMatrix()
        dm.set_level(DegradationLevel.L2_MODERATE)
        params = dm.get_current_params()
        assert params.level == DegradationLevel.L2_MODERATE


# ============================================================
# ResourceMonitor 集成测试
# ============================================================

class TestResourceMonitorIntegration:
    """ResourceMonitor 集成测试。"""

    def test_monitor_has_controller(self):
        """ResourceMonitor 内含 ResourceController。"""
        config = get_default_config()
        monitor = ResourceMonitor(config)
        assert isinstance(monitor.controller, ResourceController)

    def test_monitor_update_returns_resources(self):
        """update 返回资源使用字典。"""
        config = get_default_config()
        monitor = ResourceMonitor(config)
        result = monitor.update(0.5)
        assert "cpu" in result
        assert "memory" in result

    def test_monitor_get_throttle_signal(self):
        """get_throttle_signal 返回节流信号。"""
        config = get_default_config()
        monitor = ResourceMonitor(config)
        monitor.update(0.5)
        signal = monitor.get_throttle_signal()
        assert isinstance(signal, ThrottleSignal)

    def test_monitor_backward_compatible(self):
        """兼容旧接口：cpu_usage/memory_usage/throttle_active。"""
        config = get_default_config()
        monitor = ResourceMonitor(config)
        monitor.update(0.5)
        assert hasattr(monitor, 'cpu_usage')
        assert hasattr(monitor, 'memory_usage')
        assert hasattr(monitor, 'throttle_active')
        assert hasattr(monitor, 'within_limits')
        assert hasattr(monitor, 'should_throttle')
        assert hasattr(monitor, 'current_load')


# ============================================================
# SPEPAAgent 集成测试
# ============================================================

class TestAgentResourceIntegration:
    """Agent 资源调节集成测试。"""

    def test_agent_has_resource_components(self):
        """Agent 包含所有资源组件。"""
        agent = SPEPAAgent(get_default_config())
        assert hasattr(agent, 'resource_monitor')
        assert hasattr(agent, 'backpressure')
        assert hasattr(agent, 'degradation_matrix')

    def test_agent_online_cycle_updates_degradation(self):
        """在线闭环更新降级级别。"""
        agent = SPEPAAgent(get_default_config())
        agent.online_cycle()
        state = agent.get_state()
        assert "degradation" in state
        assert "backpressure" in state

    def test_agent_get_state_includes_resource_info(self):
        """get_state 包含资源信息。"""
        agent = SPEPAAgent(get_default_config())
        state = agent.get_state()
        assert "resource_controller" in state
        assert "backpressure" in state
        assert "degradation" in state

    def test_agent_reset_resets_resources(self):
        """reset 重置资源组件。"""
        agent = SPEPAAgent(get_default_config())
        # 运行几个周期
        for _ in range(3):
            agent.online_cycle()
        # reset
        agent.reset()
        assert agent.degradation_matrix.current_level == DegradationLevel.L0_NORMAL
        assert agent.backpressure.get_pressure() == BackpressureLevel.NONE

    def test_high_load_reduces_policies(self):
        """高负载时减少策略数。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 模拟高负载
        agent.resource_monitor.cpu_usage = 50.0
        agent.resource_monitor.memory_usage = 50.0
        agent.resource_monitor.throttle_active = True
        # 设置降级级别
        agent.degradation_matrix.set_level(DegradationLevel.L3_SEVERE)
        params = agent.degradation_matrix.get_current_params()
        assert params.n_policies <= 2

    def test_backpressure_signals_on_high_error(self):
        """高误差时产生背压信号。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 注入高误差观测
        high_error_obs = np.ones(config.obs_dim) * 100  # 极端值产生高误差
        agent.online_cycle(observation=high_error_obs)
        # 应有背压信号
        assert agent.backpressure.get_pressure() >= BackpressureLevel.MEDIUM


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端场景测试。"""

    def test_degradation_and_recovery_cycle(self):
        """降级与恢复循环：高负载降级→低负载恢复。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 模拟高负载：直接通过 PID 控制器和降级矩阵
        for _ in range(5):
            signal = agent.resource_monitor.controller.update(80.0, 80.0)
            agent.degradation_matrix.update_from_throttle(signal.throttle)
        # 应已降级
        assert agent.degradation_matrix.current_level != DegradationLevel.L0_NORMAL
        # 模拟低负载恢复
        for _ in range(10):
            signal = agent.resource_monitor.controller.update(5.0, 5.0)
            agent.degradation_matrix.update_from_throttle(signal.throttle)
        # 应恢复到正常或轻度
        assert agent.degradation_matrix.current_level <= DegradationLevel.L1_MILD

    def test_resource_monitoring_during_run(self):
        """运行过程中资源监控持续工作。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.run(n_steps=10)
        state = agent.get_state()
        assert state["resources"]["cpu"] >= 0
        assert state["resources"]["memory"] >= 0

    def test_perpetual_runner_soft_restart_resets_resources(self):
        """PerpetualRunner 软重启重置资源组件。"""
        from spepa.perpetual.runner import PerpetualRunner
        config = get_default_config()
        agent = SPEPAAgent(config)
        # 创建 runner
        def factory():
            return SPEPAAgent(config)
        runner = PerpetualRunner(config, factory)
        runner.agent = agent
        # 设置一些资源状态
        agent.backpressure.signal("test", BackpressureLevel.HIGH)
        agent.degradation_matrix.set_level(DegradationLevel.L3_SEVERE)
        # 软重启
        runner._soft_restart()
        # 应已重置
        assert agent.backpressure.get_pressure() == BackpressureLevel.NONE
        assert agent.degradation_matrix.current_level == DegradationLevel.L0_NORMAL

    def test_degradation_matrix_levels_cover_all_scenarios(self):
        """降级矩阵覆盖所有场景。"""
        dm = DegradationMatrix()
        # 每个级别都应有合理的参数
        for level in DegradationLevel:
            params = dm.get_params(level)
            assert params.n_policies >= 1
            assert 0.0 <= params.exploration_ratio <= 1.0
            # 更高级别应更保守
            if level > DegradationLevel.L0_NORMAL:
                prev_params = dm.get_params(DegradationLevel(level - 1))
                assert params.n_policies <= prev_params.n_policies

    def test_pid_controller_responds_to_changing_load(self):
        """PID 控制器响应负载变化。"""
        config = get_default_config()
        ctrl = ResourceController(config)
        # 低负载
        s1 = ctrl.update(5.0, 5.0)
        # 突然高负载
        s2 = ctrl.update(80.0, 80.0)
        # 节流应增加
        assert s2.throttle > s1.throttle
        # 持续高负载后
        for _ in range(5):
            s3 = ctrl.update(80.0, 80.0)
        # 节流应维持高位
        assert s3.throttle > 0.5
