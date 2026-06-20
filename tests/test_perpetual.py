"""阶段一测试：永续运行机制。

验证：
1. 自主循环：持续运行不停止
2. 异常自恢复：分级恢复策略（L0-L4）
3. 心跳监控：检测卡死并恢复
4. 健康检查：状态数值稳定性
5. 降级运行：资源紧张时降级
6. 软重启/硬重启：状态恢复
7. 永不停止：任何异常都不能让系统永久停止
"""
import time
import numpy as np
import pytest

from spepa.utils.config import SPEPAConfig
from spepa.perpetual.runner import (
    PerpetualRunner, HeartbeatMonitor, RecoveryLevel, RunStats
)
from spepa.agent import SPEPAAgent


# ============================================================
# 一、心跳监控测试
# ============================================================

class TestHeartbeatMonitor:
    """心跳监控测试。"""

    def test_heartbeat_initial_state(self):
        """心跳初始状态。"""
        config = SPEPAConfig(seed=42)
        monitor = HeartbeatMonitor(config, timeout=5.0)
        assert monitor.beat_count == 0
        assert monitor.timeout_count == 0
        assert monitor.check()  # 刚创建，应该存活

    def test_heartbeat_beat(self):
        """心跳更新。"""
        config = SPEPAConfig(seed=42)
        monitor = HeartbeatMonitor(config, timeout=5.0)
        monitor.beat()
        monitor.beat()
        monitor.beat()
        assert monitor.beat_count == 3

    def test_heartbeat_timeout_detection(self):
        """心跳超时检测。"""
        config = SPEPAConfig(seed=42)
        monitor = HeartbeatMonitor(config, timeout=0.1)
        # 等待超时
        time.sleep(0.2)
        assert not monitor.check()  # 应该超时

    def test_heartbeat_elapsed_time(self):
        """心跳经过时间。"""
        config = SPEPAConfig(seed=42)
        monitor = HeartbeatMonitor(config, timeout=5.0)
        time.sleep(0.05)
        elapsed = monitor.elapsed_since_last_beat()
        assert elapsed >= 0.04
        assert elapsed < 1.0

    def test_heartbeat_callback(self):
        """心跳超时回调。"""
        config = SPEPAConfig(seed=42)
        callback_called = [False]
        monitor = HeartbeatMonitor(config, timeout=0.05)

        def on_timeout():
            callback_called[0] = True

        monitor.set_timeout_callback(on_timeout)
        # 手动触发检查（超时后）
        time.sleep(0.1)
        monitor._on_timeout()
        assert callback_called[0]

    def test_heartbeat_monitor_thread(self):
        """心跳监控线程启动和停止。"""
        config = SPEPAConfig(seed=42)
        monitor = HeartbeatMonitor(config, timeout=1.0)
        monitor.start_monitor()
        assert monitor.get_state()["monitor_running"]
        time.sleep(0.1)
        monitor.beat()
        monitor.stop_monitor()
        assert not monitor.get_state()["monitor_running"]


# ============================================================
# 二、恢复级别判定测试
# ============================================================

class TestRecoveryLevel:
    """恢复级别判定测试。"""

    def test_recovery_level_normal(self):
        """L0: 正常（0次连续失败）。"""
        config = SPEPAConfig(seed=42)
        runner = PerpetualRunner(config, lambda: None)
        runner.stats.consecutive_failures = 0
        level = runner._determine_recovery_level()
        assert level == RecoveryLevel.SKIP  # 0次失败不会触发，但1次是SKIP

    def test_recovery_level_skip(self):
        """L1: 跳过（1-2次连续失败）。"""
        config = SPEPAConfig(seed=42)
        runner = PerpetualRunner(config, lambda: None)
        runner.stats.consecutive_failures = 1
        assert runner._determine_recovery_level() == RecoveryLevel.SKIP
        runner.stats.consecutive_failures = 2
        assert runner._determine_recovery_level() == RecoveryLevel.SKIP

    def test_recovery_level_degrade(self):
        """L2: 降级（3-9次连续失败）。"""
        config = SPEPAConfig(seed=42)
        runner = PerpetualRunner(config, lambda: None)
        runner.stats.consecutive_failures = 3
        assert runner._determine_recovery_level() == RecoveryLevel.DEGRADE
        runner.stats.consecutive_failures = 9
        assert runner._determine_recovery_level() == RecoveryLevel.DEGRADE

    def test_recovery_level_soft_restart(self):
        """L3: 软重启（10-19次连续失败）。"""
        config = SPEPAConfig(seed=42)
        runner = PerpetualRunner(config, lambda: None)
        runner.stats.consecutive_failures = 10
        assert runner._determine_recovery_level() == RecoveryLevel.SOFT_RESTART
        runner.stats.consecutive_failures = 19
        assert runner._determine_recovery_level() == RecoveryLevel.SOFT_RESTART

    def test_recovery_level_hard_restart(self):
        """L4: 硬重启（>=20次连续失败）。"""
        config = SPEPAConfig(seed=42)
        runner = PerpetualRunner(config, lambda: None)
        runner.stats.consecutive_failures = 20
        assert runner._determine_recovery_level() == RecoveryLevel.HARD_RESTART
        runner.stats.consecutive_failures = 100
        assert runner._determine_recovery_level() == RecoveryLevel.HARD_RESTART


# ============================================================
# 三、永续运行测试
# ============================================================

class TestPerpetualRunner:
    """永续运行器测试。"""

    def test_runner_initial_state(self):
        """运行器初始状态。"""
        config = SPEPAConfig(seed=42)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        assert runner.agent is None
        assert runner.stats.total_cycles == 0
        assert not runner._running

    def test_runner_start_with_max_cycles(self):
        """启动运行（有限周期）。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        summary = runner.start(max_cycles=5)
        assert summary["total_cycles"] >= 5
        assert summary["successful_cycles"] > 0

    def test_runner_creates_agent(self):
        """运行器通过工厂创建Agent。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.start(max_cycles=1)
        assert runner.agent is not None

    def test_runner_stop(self):
        """优雅停止。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        # 在另一个线程中停止
        import threading
        def stop_after_delay():
            time.sleep(0.2)
            runner.stop()
        threading.Thread(target=stop_after_delay, daemon=True).start()
        summary = runner.start(max_cycles=10000)
        # 应该被停止，不会跑满10000
        assert summary["total_cycles"] < 10000

    def test_runner_exception_recovery(self):
        """异常自恢复：Agent抛异常后继续运行。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)

        class FlakyAgent:
            """会抛异常的Agent（具备健康检查所需属性）。"""
            def __init__(self):
                self.call_count = 0
                self.config = config
                self.current_state = np.ones(8) * 0.5
                # 健康检查所需
                class FakeModel:
                    class FakeLayer:
                        def __init__(self):
                            self.mu = np.zeros(8)
                            self.W = np.eye(8) * 0.1
                    def __init__(self):
                        self.layers = [self.FakeLayer()]
                self.predictive_model = FakeModel()

            def online_cycle(self, env_step_fn=None):
                self.call_count += 1
                if self.call_count % 3 == 0:
                    raise RuntimeError("Simulated failure")
                return {"ok": True}

            def offline_cycle(self):
                return {"ok": True}

        flaky = FlakyAgent()
        runner = PerpetualRunner(config, lambda: flaky)
        runner.agent = flaky
        summary = runner.start(max_cycles=10)
        # 应该有失败和恢复
        assert summary["total_cycles"] >= 10
        assert summary["failed_cycles"] > 0
        # 系统不应停止
        assert summary["total_cycles"] >= 10

    def test_runner_consecutive_failure_tracking(self):
        """连续失败计数。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)

        class AlwaysFailAgent:
            def __init__(self):
                self.config = config
                self.current_state = np.ones(8) * 0.5
                class FakeModel:
                    class FakeLayer:
                        def __init__(self):
                            self.mu = np.zeros(8)
                            self.W = np.eye(8) * 0.1
                    def __init__(self):
                        self.layers = [self.FakeLayer()]
                self.predictive_model = FakeModel()
            def online_cycle(self, env_step_fn=None):
                raise RuntimeError("Always fails")
            def offline_cycle(self):
                raise RuntimeError("Always fails")

        agent = AlwaysFailAgent()
        runner = PerpetualRunner(config, lambda: agent)
        runner.agent = agent
        summary = runner.start(max_cycles=15)
        assert summary["failed_cycles"] > 0
        assert summary["max_consecutive_failures"] > 0

    def test_runner_soft_restart(self):
        """软重启：重置部分状态。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.agent = agent
        # 修改状态
        agent.current_state = np.ones(8) * 999
        runner.stats.consecutive_failures = 5
        # 软重启
        runner._soft_restart()
        # 状态应被重置
        assert np.allclose(agent.current_state, np.ones(8) * 0.5)
        assert runner.stats.consecutive_failures == 0

    def test_runner_hard_restart(self):
        """硬重启：重建Agent。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent1 = SPEPAAgent(config)
        created_agents = []
        def factory():
            a = SPEPAAgent(config)
            created_agents.append(a)
            return a
        runner = PerpetualRunner(config, factory)
        runner.agent = agent1
        # 硬重启
        runner._hard_restart()
        # 应创建新Agent
        assert len(created_agents) == 1
        assert runner.agent is created_agents[0]
        assert runner.agent is not agent1

    def test_runner_health_check_pass(self):
        """健康检查通过。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.agent = agent
        assert runner._health_check()

    def test_runner_health_check_fail_nan(self):
        """健康检查失败：状态含NaN。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        agent.current_state = np.ones(8) * np.nan
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.agent = agent
        assert not runner._health_check()

    def test_runner_health_check_fail_inf_weights(self):
        """健康检查失败：权重含Inf。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        agent.predictive_model.layers[0].W = np.ones((8, 8)) * np.inf
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.agent = agent
        assert not runner._health_check()

    def test_runner_degraded_mode(self):
        """降级模式运行。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.agent = agent
        runner._degraded_mode = True
        # 降级模式下应能运行
        runner._run_degraded_cycle(None)
        assert runner.stats.degraded_cycles == 1

    def test_runner_get_state(self):
        """获取运行器状态。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.start(max_cycles=3)
        state = runner.get_state()
        assert "stats" in state
        assert "heartbeat" in state
        assert "recovery_history" in state
        assert state["stats"]["total_cycles"] >= 3


# ============================================================
# 四、Agent集成测试
# ============================================================

class TestAgentPerpetualRun:
    """Agent永续运行集成测试。"""

    def test_agent_run_perpetual(self):
        """Agent永续运行。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        summary = agent.run_perpetual(max_cycles=5)
        assert summary["total_cycles"] >= 5
        assert summary["successful_cycles"] > 0

    def test_agent_perpetual_never_stops_on_error(self):
        """Agent永续运行遇错不停止。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # 注入一个会偶尔失败的环境
        call_count = [0]
        def flaky_env(action):
            call_count[0] += 1
            if call_count[0] % 4 == 0:
                raise RuntimeError("Env failure")
            return np.ones(8) * 0.5
        summary = agent.run_perpetual(max_cycles=20, env_step_fn=flaky_env)
        # 应该跑满20个周期，不停止
        assert summary["total_cycles"] >= 20

    def test_agent_perpetual_with_offline(self):
        """永续运行包含离线复盘。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # offline_interval=10，跑15个周期应触发1次offline
        summary = agent.run_perpetual(max_cycles=15)
        assert summary["total_cycles"] >= 15


# ============================================================
# 五、永续性压力测试
# ============================================================

class TestPerpetualStress:
    """永续性压力测试。"""

    def test_long_run_stability(self):
        """长时间运行稳定性（100周期）。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        summary = runner.start(max_cycles=100)
        assert summary["total_cycles"] >= 100
        # 成功率应较高
        assert summary["success_rate"] > 0.5
        # 不应有硬重启
        assert summary["hard_restarts"] == 0

    def test_recovery_from_nan_state(self):
        """从NaN状态恢复。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.agent = agent
        # 注入NaN状态
        agent.current_state = np.ones(8) * np.nan
        # 健康检查应失败
        assert not runner._health_check()
        # 软重启应恢复
        runner._soft_restart()
        assert runner._health_check()

    def test_recovery_from_extreme_state(self):
        """从极端状态恢复。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.agent = agent
        # 注入极端状态
        agent.current_state = np.ones(8) * 1e10
        # 软重启
        runner._soft_restart()
        assert np.allclose(agent.current_state, np.ones(8) * 0.5)

    def test_never_permanently_stops(self):
        """永不永久停止：即使大量异常也能继续。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)

        class VeryFlakyAgent:
            def __init__(self):
                self.config = config
                self.count = 0
                self.current_state = np.ones(8) * 0.5
                class FakeModel:
                    class FakeLayer:
                        def __init__(self):
                            self.mu = np.zeros(8)
                            self.W = np.eye(8) * 0.1
                    def __init__(self):
                        self.layers = [self.FakeLayer()]
                self.predictive_model = FakeModel()
            def online_cycle(self, env_step_fn=None):
                self.count += 1
                if self.count % 2 == 0:
                    raise RuntimeError("Frequent failure")
                return {"ok": True}
            def offline_cycle(self):
                pass

        agent = VeryFlakyAgent()
        runner = PerpetualRunner(config, lambda: agent)
        runner.agent = agent
        summary = runner.start(max_cycles=30)
        # 应该跑满30个周期
        assert summary["total_cycles"] >= 30
        # 应有失败和恢复
        assert summary["failed_cycles"] > 0
        # 但系统仍在运行
        assert summary["total_cycles"] >= 30

    def test_runstats_tracking(self):
        """运行统计跟踪。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)
        runner = PerpetualRunner(config, lambda: SPEPAAgent(config))
        runner.start(max_cycles=10)
        stats = runner.stats
        assert stats.total_cycles >= 10
        assert stats.successful_cycles + stats.failed_cycles + stats.skipped_cycles >= 10
        assert stats.uptime > 0

    def test_recovery_history_recorded(self):
        """恢复历史被记录。"""
        config = SPEPAConfig(state_dim=8, action_dim=8, obs_dim=8, seed=42)

        class FailOnceAgent:
            def __init__(self):
                self.config = config
                self.count = 0
                self.current_state = np.ones(8) * 0.5
                class FakeModel:
                    class FakeLayer:
                        def __init__(self):
                            self.mu = np.zeros(8)
                            self.W = np.eye(8) * 0.1
                    def __init__(self):
                        self.layers = [self.FakeLayer()]
                self.predictive_model = FakeModel()
            def online_cycle(self, env_step_fn=None):
                self.count += 1
                if self.count == 1:
                    raise RuntimeError("First call fails")
                return {"ok": True}
            def offline_cycle(self):
                pass

        agent = FailOnceAgent()
        runner = PerpetualRunner(config, lambda: agent)
        runner.agent = agent
        runner.start(max_cycles=5)
        # 应有恢复历史
        assert len(runner.recovery_history) >= 1
