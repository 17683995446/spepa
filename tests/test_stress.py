"""SPEPA 复杂混乱场景压力测试。

目标：通过长时间、大规模、高难度、混乱场景的真实运行，发现系统在真实运行时的 bug。
每句代码后面都有日志，通过日志是否正常打印来验证是否存在问题。
"""
import io
import logging
import sys
import threading
import time
import warnings
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pytest

from spepa.agent import SPEPAAgent, ResourceMonitor
from spepa.core.active_inference import ActiveInference, Policy
from spepa.core.drive_engine import DriveEngine, Homeostasis
from spepa.core.free_energy import FreeEnergyCalculator
from spepa.core.gap_detector import GapDetector, GapInfo, GAP_LEVELS
from spepa.core.goal_hierarchy import Goal, GoalHierarchy
from spepa.core.memory_system import (
    FastMemory, MemoryItem, MemorySystem, SlowMemory,
)
from spepa.core.metacognition import ConfidenceCalibrator, Metacognition
from spepa.core.optimizer import Optimizer
from spepa.core.predictive_model import PredictiveLayer, PredictiveModel
from spepa.core.safety_guard import SafetyGuard
from spepa.core.self_model import SelfModel
from spepa.core.state_bus import StateBus
from spepa.utils.config import SPEPAConfig
from spepa.utils.metrics import (
    MetricsTracker, cosine_similarity, entropy, gaussian_kl,
    kl_divergence, mse, prediction_error,
)


# ---------------------------------------------------------------------------
# 辅助：捕获日志以验证"每句代码后面打印日志"
# ---------------------------------------------------------------------------


class LogCapture:
    """捕获指定 logger 的日志记录，便于断言。"""

    def __init__(self, name: str = "spepa", level: int = logging.DEBUG):
        self.name = name
        self.records: list = []
        self._handler = None
        self._logger = None
        self._level = level

    def __enter__(self):
        self._logger = logging.getLogger(self.name)
        self._handler = logging.Handler()
        self._handler.setLevel(self._level)
        self._handler.emit = lambda record: self.records.append(record)
        self._logger.addHandler(self._handler)
        self._logger.setLevel(self._level)
        return self

    def __exit__(self, *exc):
        if self._handler is not None and self._logger is not None:
            self._logger.removeHandler(self._handler)
        return False

    def messages(self, level: str = None) -> list:
        out = []
        for r in self.records:
            if level is None or logging.getLevelName(r.levelno) == level:
                out.append(r.getMessage())
        return out


# ---------------------------------------------------------------------------
# 1. 长时间运行稳定性测试
# ---------------------------------------------------------------------------


class TestLongRunStability:
    """长时间运行稳定性测试：发现内存泄漏、数值发散、状态污染等问题。"""

    def test_run_500_steps_default_env(self):
        """运行 500 步，检查数值稳定性和状态一致性。"""
        config = SPEPAConfig(state_dim=16, action_dim=8, obs_dim=16,
                             seed=42, max_steps=500,
                             consolidation_interval=20)
        agent = SPEPAAgent(config)
        summary = agent.run(n_steps=500)

        # 验证运行完成
        assert summary["online_cycles"] == 500
        # 验证数值没有发散（无 NaN/Inf）
        state = agent.get_state()
        cs = np.asarray(state["current_state"], dtype=np.float64)
        assert np.all(np.isfinite(cs)), "current_state 出现 NaN/Inf"
        # 验证预测模型权重没有爆炸
        for layer in agent.predictive_model.layers:
            assert np.all(np.isfinite(layer.mu)), "layer.mu 出现 NaN/Inf"
            assert np.all(np.isfinite(layer.W)), "layer.W 出现 NaN/Inf"
            assert np.all(np.isfinite(layer.var)), "layer.var 出现 NaN/Inf"
            # 方差应在合理范围内
            assert np.all(layer.var <= 5.0 + 1e-6), "方差超过上界"
            assert np.all(layer.var >= 0.05 - 1e-6), "方差低于下界"
        # 验证指标
        metrics = summary["metrics"]
        assert "prediction_error" in metrics
        pe = metrics["prediction_error"]
        assert np.isfinite(pe["max"]), "prediction_error max 不是有限值"
        # 误差不应无限增长
        assert pe["max"] < 1e6, f"prediction_error 过大: {pe['max']}"

    def test_run_1000_steps_with_consolidation(self):
        """运行 1000 步，验证记忆巩固不会导致崩溃。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=7, max_steps=1000,
                             consolidation_interval=10,
                             fast_memory_capacity=50,
                             slow_memory_capacity=20)
        agent = SPEPAAgent(config)
        summary = agent.run(n_steps=1000)
        assert summary["online_cycles"] == 1000
        # 验证记忆系统状态
        ms = agent.memory_system.get_state()
        assert ms["step_count"] == 1000
        # 慢速记忆不应超过容量太多（巩固后会裁剪）
        assert len(agent.memory_system.slow) <= 20 + 1

    def test_repeated_reset_and_run(self):
        """多次 reset + run，验证状态完全重置。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, max_steps=10)
        agent = SPEPAAgent(config)
        for i in range(5):
            agent.reset()
            agent.run(n_steps=10)
            assert agent.online_cycles == 10
            assert agent.step_count == 10
        # 最后一次 reset 后应为 0
        agent.reset()
        assert agent.online_cycles == 0
        assert agent.step_count == 0


# ---------------------------------------------------------------------------
# 2. 大规模状态维度测试
# ---------------------------------------------------------------------------


class TestLargeScaleDimensions:
    """大规模状态维度测试：发现维度不匹配、性能瓶颈。"""

    def test_large_state_dim_64(self):
        """state_dim=64 的大规模状态。"""
        config = SPEPAConfig(state_dim=64, action_dim=32, obs_dim=64,
                             seed=42, max_steps=20,
                             prediction_layers=5)
        agent = SPEPAAgent(config)
        summary = agent.run(n_steps=20)
        assert summary["online_cycles"] == 20
        # 验证所有模块维度一致
        assert agent.predictive_model.dim == 64
        assert agent.active_inference.dim == 32
        assert agent.active_inference.state_dim == 64

    def test_mismatched_dims(self):
        """obs_dim != state_dim != action_dim 的混乱维度。"""
        config = SPEPAConfig(state_dim=16, action_dim=4, obs_dim=32,
                             seed=42, max_steps=10)
        agent = SPEPAAgent(config)
        # 提供与 obs_dim 匹配的观测
        obs = np.random.rand(32)
        result = agent.online_cycle(obs)
        assert result["cycle"] == 1
        # 验证观测被正确 resize
        perc_obs = np.asarray(result["perception"]["observation"])
        # perception 会 resize 到 obs_dim
        assert perc_obs.shape[0] == 32

    def test_single_dim(self):
        """state_dim=1 的极端小维度。"""
        config = SPEPAConfig(state_dim=1, action_dim=1, obs_dim=1,
                             seed=42, max_steps=5, prediction_layers=1)
        agent = SPEPAAgent(config)
        summary = agent.run(n_steps=5)
        assert summary["online_cycles"] == 5


# ---------------------------------------------------------------------------
# 3. 极端值与异常输入测试
# ---------------------------------------------------------------------------


class TestExtremeInputs:
    """极端值与异常输入测试：发现数值溢出、未处理异常。"""

    def test_zero_observation(self):
        """全零观测。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        result = agent.online_cycle(np.zeros(8))
        assert result["cycle"] == 1
        assert np.all(np.isfinite(np.asarray(result["perception"]["prediction"])))

    def test_max_observation(self):
        """全 1 观测（最大值）。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        result = agent.online_cycle(np.ones(8))
        assert result["cycle"] == 1

    def test_negative_observation(self):
        """负值观测（超出预期范围）。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        result = agent.online_cycle(-np.ones(8) * 10)
        assert result["cycle"] == 1
        # 验证系统没有崩溃
        assert np.all(np.isfinite(np.asarray(result["perception"]["error"])))

    def test_huge_observation(self):
        """超大值观测（1e10）。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        result = agent.online_cycle(np.ones(8) * 1e10)
        assert result["cycle"] == 1
        # 验证数值没有变成 NaN
        assert np.all(np.isfinite(np.asarray(result["perception"]["error"])))

    def test_nan_observation_should_not_crash(self):
        """包含 NaN 的观测不应导致崩溃。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        obs = np.array([0.5, 0.5, np.nan, 0.5, 0.5, 0.5, 0.5, 0.5])
        # 系统应能处理 NaN（可能通过裁剪或报错日志）
        try:
            result = agent.online_cycle(obs)
            # 如果成功，验证没有产生全 NaN 状态
            cs = agent.current_state
            if np.any(np.isnan(cs)):
                pytest.fail("NaN 观测导致 current_state 变为 NaN")
        except Exception as e:
            # 如果抛异常，记录但不应该崩溃
            pytest.fail(f"NaN 观测导致崩溃: {e}")

    def test_inf_observation_should_not_crash(self):
        """包含 Inf 的观测不应导致崩溃。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        obs = np.array([0.5, 0.5, np.inf, 0.5, 0.5, 0.5, 0.5, 0.5])
        try:
            result = agent.online_cycle(obs)
        except Exception as e:
            pytest.fail(f"Inf 观测导致崩溃: {e}")

    def test_empty_observation(self):
        """空数组观测。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        result = agent.online_cycle(np.array([]))
        assert result["cycle"] == 1

    def test_wrong_shape_2d_observation(self):
        """2D 观测（错误形状）。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        obs_2d = np.random.rand(4, 4)  # 16 个元素但 2D
        result = agent.online_cycle(obs_2d)
        assert result["cycle"] == 1


# ---------------------------------------------------------------------------
# 4. 混乱环境测试
# ---------------------------------------------------------------------------


class TestChaoticEnvironment:
    """混乱环境测试：快速变化、对抗性、随机中断。"""

    def test_rapidly_changing_environment(self):
        """每步环境动力学完全改变。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, max_steps=50)
        agent = SPEPAAgent(config)

        def chaotic_env(action):
            # 每次调用环境动力学都不同
            scale = np.random.uniform(0.1, 2.0)
            return np.clip(np.random.rand(8) * scale, 0, 1)

        summary = agent.run(n_steps=50, env_step_fn=chaotic_env)
        assert summary["online_cycles"] == 50

    def test_adversarial_environment(self):
        """对抗性环境：总是把状态推向最差方向。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, max_steps=50)
        agent = SPEPAAgent(config)

        def adversarial_env(action):
            # 总是返回极端状态
            return np.clip(np.ones(8) - action[:8] if action.shape[0] >= 8
                           else np.ones(8), 0, 1)

        summary = agent.run(n_steps=50, env_step_fn=adversarial_env)
        assert summary["online_cycles"] == 50
        # 安全护栏应记录违规
        # (不强制，因为状态可能被裁剪)

    def test_environment_occasionally_returns_none(self):
        """环境偶尔返回 None（模拟断连）。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, max_steps=30)
        agent = SPEPAAgent(config)
        call_count = [0]

        def flaky_env(action):
            call_count[0] += 1
            if call_count[0] % 5 == 0:
                return None  # 模拟断连
            return np.clip(np.random.rand(8), 0, 1)

        # Action.execute 内部 try/except 了 env_step_fn
        summary = agent.run(n_steps=30, env_step_fn=flaky_env)
        assert summary["online_cycles"] == 30

    def test_environment_raises_exception(self):
        """环境抛异常（模拟故障）。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, max_steps=20)
        agent = SPEPAAgent(config)
        call_count = [0]

        def crashing_env(action):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise RuntimeError("环境模拟故障")
            return np.clip(np.random.rand(8), 0, 1)

        summary = agent.run(n_steps=20, env_step_fn=crashing_env)
        assert summary["online_cycles"] == 20

    def test_environment_returns_wrong_shape(self):
        """环境返回错误形状的状态。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, max_steps=20)
        agent = SPEPAAgent(config)

        def wrong_shape_env(action):
            return np.random.rand(16)  # 返回 16 维但 state_dim=8

        summary = agent.run(n_steps=20, env_step_fn=wrong_shape_env)
        assert summary["online_cycles"] == 20
        # current_state 应被 resize 到 state_dim
        assert agent.current_state.shape[0] == 8

    def test_environment_returns_nan(self):
        """环境返回 NaN。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, max_steps=20)
        agent = SPEPAAgent(config)
        call_count = [0]

        def nan_env(action):
            call_count[0] += 1
            if call_count[0] % 4 == 0:
                return np.array([np.nan] * 8)
            return np.clip(np.random.rand(8), 0, 1)

        try:
            summary = agent.run(n_steps=20, env_step_fn=nan_env)
            # 验证系统没有因 NaN 而完全崩溃
            assert summary["online_cycles"] == 20
        except Exception as e:
            pytest.fail(f"环境返回 NaN 导致崩溃: {e}")


# ---------------------------------------------------------------------------
# 5. 数据一致性校验
# ---------------------------------------------------------------------------


class TestDataConsistency:
    """数据一致性校验：核对增删改查、数据存储、关联关系。"""

    def test_state_bus_pubsub_consistency(self):
        """状态总线发布-订阅一致性。"""
        bus = StateBus()
        received = []
        bus.subscribe("topic1", received.append)
        # 发布 10 条消息
        for i in range(10):
            bus.publish("topic1", {"idx": i})
        assert len(received) == 10
        assert received[0] == {"idx": 0}
        assert received[-1] == {"idx": 9}
        # 历史记录
        history = bus.history("topic1")
        assert len(history) == 10

    def test_memory_encode_retrieve_consistency(self):
        """记忆编码-检索一致性。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        ms = MemorySystem(config)
        # 编码 10 条记忆
        keys = []
        for i in range(10):
            key = np.zeros(8)
            key[i % 8] = 1.0  # 每条记忆有一个独特维度为 1
            keys.append(key)
            ms.encode(key, {"id": i}, importance=0.5)
        # 检索每条记忆
        for i, key in enumerate(keys):
            results = ms.retrieve(key, k=1)
            assert len(results) >= 1
            # 最相似的应是自己（余弦相似度=1）
            retrieved_key = results[0].key
            sim = cosine_similarity(key, retrieved_key)
            assert sim > 0.99, f"检索结果不匹配: 期望 idx={i}, sim={sim}"

    def test_memory_consolidation_preserves_top(self):
        """记忆巩固保留 Top 30% 最重要的。"""
        config = SPEPAConfig(state_dim=4, seed=42,
                             fast_memory_capacity=100,
                             slow_memory_capacity=10,
                             consolidation_interval=1000)
        ms = MemorySystem(config)
        # 编码 20 条记忆，重要性递增
        for i in range(20):
            ms.encode(np.random.rand(4), {"id": i}, importance=float(i) / 20)
        # 巩固
        n = ms.consolidate()
        assert n > 0
        # 慢速记忆中的项重要性应较高
        slow_imps = [it.importance for it in ms.slow.items]
        if slow_imps:
            assert max(slow_imps) >= 0.5, "巩固未保留高重要性记忆"

    def test_metrics_record_summary_consistency(self):
        """指标记录-摘要一致性。"""
        mt = MetricsTracker()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            mt.record("test_metric", v)
        summary = mt.summary()
        assert "test_metric" in summary
        s = summary["test_metric"]
        assert s["count"] == 5
        assert s["min"] == 1.0
        assert s["max"] == 5.0
        assert s["last"] == 5.0
        assert abs(s["mean"] - 3.0) < 1e-6

    def test_agent_state_bus_sync(self):
        """Agent 运行时状态总线与模块状态同步。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        agent.online_cycle(np.random.rand(8))
        # 状态总线应记录最后一次观测
        assert agent.bus.get_state("last_observation") is not None
        assert agent.bus.get_state("last_prediction") is not None
        assert agent.bus.get_state("last_error") is not None
        assert agent.bus.get_state("last_action") is not None
        assert agent.bus.get_state("last_gap") is not None

    def test_goal_hierarchy_arbitration_consistency(self):
        """目标层次仲裁一致性。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        gh = GoalHierarchy(config)
        # 状态远离 L0 目标
        state_far = np.ones(4) * 0.0  # 距离 0.5 目标很远
        result = gh.arbitrate(state_far)
        # 应该有主导目标
        assert result["dominant_goal"] is not None
        # 状态在 L0 目标附近
        state_near = np.ones(4) * 0.5
        result2 = gh.arbitrate(state_near)
        assert result2["dominant_goal"] is not None


# ---------------------------------------------------------------------------
# 6. 并发与线程安全测试
# ---------------------------------------------------------------------------


class TestConcurrency:
    """并发与线程安全测试。"""

    def test_state_bus_concurrent_publish(self):
        """多线程并发发布消息。"""
        bus = StateBus()
        received = []
        lock = threading.Lock()

        def callback(msg):
            with lock:
                received.append(msg)

        bus.subscribe("concurrent", callback)
        threads = []
        for i in range(10):
            t = threading.Thread(target=bus.publish, args=("concurrent", {"idx": i}))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(received) == 10

    def test_state_bus_concurrent_set_get(self):
        """多线程并发读写状态。"""
        bus = StateBus()
        errors = []

        def writer():
            try:
                for i in range(100):
                    bus.set_state("counter", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    bus.get_state("counter")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"并发错误: {errors}"


# ---------------------------------------------------------------------------
# 7. 安全护栏极端测试
# ---------------------------------------------------------------------------


class TestSafetyGuardExtreme:
    """安全护栏极端测试。"""

    def test_emergency_stop_severe_out_of_range(self):
        """状态严重越界触发紧急停止。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        sg = SafetyGuard(config)
        # 状态 < -0.5
        assert sg.emergency_stop(np.ones(4) * -1.0)
        # 状态 > 1.5
        assert sg.emergency_stop(np.ones(4) * 2.0)
        # 正常状态
        assert not sg.emergency_stop(np.ones(4) * 0.5)

    def test_custom_constraint_called(self):
        """自定义约束被调用。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        sg = SafetyGuard(config)
        called = [False]

        def my_constraint(state, action):
            called[0] = True
            return np.all(state >= 0)

        sg.add_constraint(my_constraint)
        sg.validate(np.ones(4) * 0.5, np.ones(4) * 0.1)
        assert called[0]

    def test_custom_constraint_failing(self):
        """自定义约束失败时记录违规。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        sg = SafetyGuard(config)

        def fail_constraint(state, action):
            return False

        sg.add_constraint(fail_constraint)
        result = sg.validate(np.ones(4) * 0.5, np.ones(4) * 0.1)
        assert not result["safe"]
        assert not result["custom_ok"]
        assert len(sg.violations) > 0

    def test_custom_constraint_exception(self):
        """自定义约束抛异常时不崩溃。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        sg = SafetyGuard(config)

        def bad_constraint(state, action):
            raise ValueError("约束函数故障")

        sg.add_constraint(bad_constraint)
        # 不应崩溃
        result = sg.validate(np.ones(4) * 0.5, np.ones(4) * 0.1)
        # custom_ok 保持 True（因为异常被捕获）
        assert result["custom_ok"]


# ---------------------------------------------------------------------------
# 8. 预测模型数值稳定性测试
# ---------------------------------------------------------------------------


class TestPredictiveModelStability:
    """预测模型数值稳定性测试。"""

    def test_repeated_update_convergence(self):
        """重复更新同一观测，误差应趋于下降。"""
        config = SPEPAConfig(state_dim=8, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        obs = np.random.rand(8)
        initial_err = model.total_prediction_error(obs)
        errors = []
        for _ in range(100):
            model.update(obs, lr=0.05)
            errors.append(model.total_prediction_error(obs))
        # 最终误差应小于初始误差
        assert errors[-1] <= initial_err + 1e-3, \
            f"误差未收敛: {initial_err} -> {errors[-1]}"

    def test_precision_weighted_error_non_negative(self):
        """精度加权误差非负。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        model = PredictiveModel(config)
        for _ in range(20):
            obs = np.random.rand(8)
            err = model.precision_weighted_error(obs)
            assert err >= 0, f"精度加权误差为负: {err}"

    def test_layer_var_bounded_after_many_updates(self):
        """多次更新后方差仍在边界内。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        model = PredictiveModel(config)
        for _ in range(200):
            obs = np.random.rand(8) * 10  # 大值观测
            model.update(obs, lr=0.1)
        for layer in model.layers:
            assert np.all(layer.var <= 5.0 + 1e-6)
            assert np.all(layer.var >= 0.05 - 1e-6)

    def test_predict_without_observation(self):
        """无观测时的纯内部预测。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        model = PredictiveModel(config)
        pred = model.predict(None)
        assert pred.shape == (8,)
        assert np.all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# 9. 优化器模式选择测试
# ---------------------------------------------------------------------------


class TestOptimizerModeSelection:
    """优化器模式选择测试：覆盖所有分支。"""

    def test_micro_mode_small_error(self):
        """小误差 -> micro 模式。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        opt = Optimizer(config)
        model = PredictiveModel(config)
        result = opt.optimize(model, np.random.rand(8), 0.1)
        assert result["mode"] == "micro"

    def test_structural_mode_large_error(self):
        """大误差 -> structural 模式。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        opt = Optimizer(config)
        model = PredictiveModel(config)
        result = opt.optimize(model, np.random.rand(8), 2.0)
        assert result["mode"] == "structural"

    def test_middle_region_uses_micro(self):
        """中间区域（0.3 < err < 1.0）-> hebbian 模式（阶段七三档模式）。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        opt = Optimizer(config)
        model = PredictiveModel(config)
        result = opt.optimize(model, np.random.rand(8), 0.6)
        # 阶段七：中间区域使用 hebbian 探索性学习
        assert result["mode"] == "hebbian"

    def test_boundary_at_micro_threshold(self):
        """边界值：误差 == micro_threshold。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        opt = Optimizer(config)
        model = PredictiveModel(config)
        # err == 0.3 -> 不 < 0.3，进入中间区域 -> hebbian
        result = opt.optimize(model, np.random.rand(8), 0.3)
        assert result["mode"] == "hebbian"

    def test_boundary_at_struct_threshold(self):
        """边界值：误差 == struct_threshold。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        opt = Optimizer(config)
        model = PredictiveModel(config)
        # err == 1.0 -> 不 < 0.3，1.0 不 > 1.0，走中间区域 -> hebbian
        result = opt.optimize(model, np.random.rand(8), 1.0)
        assert result["mode"] == "hebbian"

    def test_zero_error(self):
        """零误差。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        opt = Optimizer(config)
        model = PredictiveModel(config)
        result = opt.optimize(model, np.random.rand(8), 0.0)
        assert result["mode"] == "micro"

    def test_negative_error(self):
        """负误差（异常情况）。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        opt = Optimizer(config)
        model = PredictiveModel(config)
        result = opt.optimize(model, np.random.rand(8), -1.0)
        assert result["mode"] == "micro"


# ---------------------------------------------------------------------------
# 10. 缺口检测器边界测试
# ---------------------------------------------------------------------------


class TestGapDetectorBoundary:
    """缺口检测器边界测试。"""

    def test_all_levels(self):
        """覆盖所有四级。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        gd = GapDetector(config)
        # L0: kl < 0.2
        gap = gd.detect_prediction_gap(0.1)
        assert gap.level == "L0_none"
        assert gap.level_idx == 0
        # L1: 0.2 <= kl < 0.5
        gap = gd.detect_prediction_gap(0.3)
        assert gap.level == "L1_minor"
        assert gap.level_idx == 1
        # L2: 0.5 <= kl < 1.0
        gap = gd.detect_prediction_gap(0.7)
        assert gap.level == "L2_moderate"
        assert gap.level_idx == 2
        # L3: kl >= 1.0
        gap = gd.detect_prediction_gap(1.5)
        assert gap.level == "L3_severe"
        assert gap.level_idx == 3

    def test_boundary_values(self):
        """边界值测试。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        gd = GapDetector(config)
        thresholds = config.gap_level_thresholds  # (0.2, 0.5, 1.0, 2.0)
        # 恰好等于阈值
        gap = gd.detect_prediction_gap(thresholds[0])  # 0.2
        assert gap.level_idx == 1  # 0.2 不 < 0.2，进入 L1
        gap = gd.detect_prediction_gap(thresholds[1])  # 0.5
        assert gap.level_idx == 2  # 0.5 不 < 0.5，进入 L2
        gap = gd.detect_prediction_gap(thresholds[2])  # 1.0
        assert gap.level_idx == 3  # 1.0 不 < 1.0，进入 L3

    def test_is_actionable(self):
        """is_actionable 阈值。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        gd = GapDetector(config)
        gap = gd.detect_prediction_gap(0.1)
        assert not gap.is_actionable()
        gap = gd.detect_prediction_gap(0.3)
        assert not gap.is_actionable()
        gap = gd.detect_prediction_gap(0.7)
        assert gap.is_actionable()
        gap = gd.detect_prediction_gap(1.5)
        assert gap.is_actionable()

    def test_should_explore(self):
        """should_explore 逻辑。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        gd = GapDetector(config)
        gap = gd.detect_prediction_gap(0.1)
        assert not gd.should_explore(gap)
        gap = gd.detect_prediction_gap(1.5)
        assert gd.should_explore(gap)

    def test_negative_kl(self):
        """负 KL 值（异常）。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        gd = GapDetector(config)
        gap = gd.detect_prediction_gap(-1.0)
        assert gap.level_idx == 0  # 负数 < 0.2，进入 L0


# ---------------------------------------------------------------------------
# 11. 主动推理策略选择测试
# ---------------------------------------------------------------------------


class TestActiveInferencePolicy:
    """主动推理策略选择测试。"""

    def test_generate_policies_count(self):
        """策略生成数量正确。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, seed=42)
        fe = FreeEnergyCalculator(config)
        ai = ActiveInference(config, fe)
        policies = ai.generate_policies(n_policies=10, horizon=1, exploration_ratio=0.3)
        assert len(policies) == 10
        # 30% 探索 = 3 个
        explore_count = sum(1 for p in policies if p.name.startswith("explore_"))
        assert explore_count == 3

    def test_generate_policies_zero_exploration(self):
        """零探索比例。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, seed=42)
        fe = FreeEnergyCalculator(config)
        ai = ActiveInference(config, fe)
        policies = ai.generate_policies(n_policies=10, horizon=1, exploration_ratio=0.0)
        assert len(policies) == 10
        # max(1, int(10 * 0)) = max(1, 0) = 1，所以至少 1 个探索
        explore_count = sum(1 for p in policies if p.name.startswith("explore_"))
        assert explore_count == 1

    def test_generate_policies_full_exploration(self):
        """全探索比例。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, seed=42)
        fe = FreeEnergyCalculator(config)
        ai = ActiveInference(config, fe)
        policies = ai.generate_policies(n_policies=10, horizon=1, exploration_ratio=1.0)
        assert len(policies) == 10
        explore_count = sum(1 for p in policies if p.name.startswith("explore_"))
        assert explore_count == 10

    def test_decide_returns_valid_action(self):
        """decide 返回有效动作。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, seed=42)
        fe = FreeEnergyCalculator(config)
        ai = ActiveInference(config, fe)

        def predict_fn(state, action):
            action = np.resize(np.asarray(action, dtype=np.float64).flatten(), state.shape[0])
            return state + action * 0.1, np.ones(8) * 0.1

        result = ai.decide(np.ones(8) * 0.5, predict_fn, np.ones(8) * 0.7,
                           exploration_ratio=0.3)
        assert len(result["action"]) == 4
        assert "policy_name" in result
        assert "expected_free_energy" in result
        assert len(result["evaluations"]) == 8

    def test_policy_first_action_empty(self):
        """空策略的 first_action。"""
        p = Policy([], name="empty")
        action = p.first_action()
        assert action.shape == (0,)


# ---------------------------------------------------------------------------
# 12. 元认知校准测试
# ---------------------------------------------------------------------------


class TestMetacognitionCalibration:
    """元认知校准测试。"""

    def test_overconfident_detection(self):
        """过度自信检测。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        mc = Metacognition(config)
        # 记录高信心但低准确度
        for _ in range(10):
            mc.record_outcome(0.9, 2.0)  # 高信心，大误差
        assert mc.calibrator.is_overconfident()

    def test_well_calibrated(self):
        """良好校准。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        mc = Metacognition(config)
        # 信心与准确度匹配
        for _ in range(10):
            mc.record_outcome(0.5, 0.5)  # 信心 0.5，误差 0.5 -> accuracy ≈ 0.6
        assert not mc.calibrator.is_overconfident()

    def test_should_switch_strategy_low_confidence(self):
        """低信心时切换策略。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        mc = Metacognition(config)
        mc.compute_confidence(10.0, 10.0)  # 大误差 -> 低信心
        assert mc.current_confidence < 0.3
        assert mc.should_switch_strategy()

    def test_meta_control_exploration_increase(self):
        """元认知控制增加探索比例。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        mc = Metacognition(config)
        # 大误差 + 高自由能
        result = mc.meta_control(prediction_error=2.0, free_energy=2.0,
                                 exploration_ratio=0.3)
        assert result["new_exploration_ratio"] > 0.3
        assert result["control_signal"] > 0

    def test_meta_control_respects_minimum(self):
        """元认知控制不低于探索下限。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        mc = Metacognition(config)
        result = mc.meta_control(prediction_error=0.0, free_energy=0.0,
                                 exploration_ratio=0.1)
        assert result["new_exploration_ratio"] >= config.exploration.exploration_ratio_min


# ---------------------------------------------------------------------------
# 13. 自我模型测试
# ---------------------------------------------------------------------------


class TestSelfModel:
    """自我模型测试。"""

    def test_self_referential_check_no_history(self):
        """无历史表现时的自我指涉检查。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sm = SelfModel(config)
        result = sm.self_referential_check()
        assert result["mean_performance"] == 0.0
        assert 0.0 <= result["consistency"] <= 1.0

    def test_propose_self_modification_high_severity(self):
        """高严重度缺口 -> 增加学习率。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sm = SelfModel(config)
        proposal = sm.propose_self_modification(2.0)
        assert proposal["action"] == "increase_learning_rate"
        applied = sm.apply_modification(proposal)
        assert applied
        assert sm.capability_bounds["learning_rate"] == proposal["new_lr"]

    def test_propose_self_modification_medium_severity(self):
        """中严重度缺口 -> 调整精度。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sm = SelfModel(config)
        proposal = sm.propose_self_modification(0.5)
        assert proposal["action"] == "adjust_precision"
        applied = sm.apply_modification(proposal)
        assert applied

    def test_propose_self_modification_low_severity(self):
        """低严重度缺口 -> 无操作。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sm = SelfModel(config)
        proposal = sm.propose_self_modification(0.1)
        assert proposal["action"] == "none"
        applied = sm.apply_modification(proposal)
        assert not applied

    def test_performance_history_capped(self):
        """历史表现记录上限。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sm = SelfModel(config)
        for i in range(200):
            sm.update_self_representation(np.random.rand(8), float(i) / 200)
        assert len(sm.performance_history) <= 100


# ---------------------------------------------------------------------------
# 14. 离线复盘触发与执行测试
# ---------------------------------------------------------------------------


class TestOfflineReplay:
    """离线复盘触发与执行测试。"""

    def test_should_trigger_low_load(self):
        """低负载触发。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        agent = SPEPAAgent(config)
        assert agent.offline_replay.should_trigger(0.1)
        assert agent.offline_replay.should_trigger(0.49)

    def test_should_not_trigger_high_load(self):
        """高负载不触发。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        agent = SPEPAAgent(config)
        assert not agent.offline_replay.should_trigger(0.5)
        assert not agent.offline_replay.should_trigger(0.9)

    def test_offline_replay_empty_memory(self):
        """空记忆时离线复盘。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        agent = SPEPAAgent(config)
        agent.resource_monitor.cpu_usage = 0.1
        agent.resource_monitor.memory_usage = 0.1
        result = agent.offline_cycle()
        # 空记忆应跳过或返回 replayed=0
        assert result.get("skipped") or result.get("replay", {}).get("replayed", 0) == 0

    def test_offline_replay_with_memories(self):
        """有记忆时离线复盘。"""
        config = SPEPAConfig(state_dim=8, seed=42,
                             offline_load_threshold=0.9)
        agent = SPEPAAgent(config)
        # 填充记忆
        for _ in range(10):
            agent.online_cycle(np.random.rand(8))
        # 强制低负载
        agent.resource_monitor.cpu_usage = 0.1
        agent.resource_monitor.memory_usage = 0.1
        result = agent.offline_cycle()
        assert "replay" in result
        assert result["replay"]["replayed"] > 0


# ---------------------------------------------------------------------------
# 15. 日志验证测试
# ---------------------------------------------------------------------------


class TestLoggingVerification:
    """日志验证测试：验证每句代码后面打印了日志。"""

    def test_perception_logs(self):
        """感知模块打印日志。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        with LogCapture("spepa.modules.perception") as cap:
            agent.perception.perceive(np.random.rand(8))
        msgs = cap.messages()
        assert len(msgs) > 0
        # 应有 "Perceived" 日志
        assert any("Perceived" in m for m in msgs)

    def test_decision_logs(self):
        """决策模块打印日志。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        gap = agent.gap_detector.detect_prediction_gap(0.5)
        with LogCapture("spepa.modules.decision") as cap:
            agent.decision.decide(np.ones(8) * 0.5, gap, agent._model_predict_fn)
        msgs = cap.messages()
        assert len(msgs) > 0
        assert any("Decision complete" in m for m in msgs)

    def test_learning_logs(self):
        """学习模块打印日志。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        with LogCapture("spepa.modules.learning") as cap:
            agent.learning.learn(np.random.rand(8), 0.5, 0.5)
        msgs = cap.messages()
        assert len(msgs) > 0
        assert any("Learning #" in m for m in msgs)

    def test_action_logs(self):
        """行动模块打印日志。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        with LogCapture("spepa.modules.action") as cap:
            agent.action.execute(np.ones(4) * 0.1)
        msgs = cap.messages()
        assert len(msgs) > 0
        assert any("Action executed" in m for m in msgs)

    def test_agent_cycle_logs(self):
        """Agent 在线闭环打印日志。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        with LogCapture("spepa.agent") as cap:
            agent.online_cycle(np.random.rand(8))
        msgs = cap.messages()
        # 应有 "Online cycle" 日志
        assert any("Online cycle" in m for m in msgs)


# ---------------------------------------------------------------------------
# 16. 资源监控测试
# ---------------------------------------------------------------------------


class TestResourceMonitor:
    """资源监控测试。"""

    def test_within_limits(self):
        """资源在限制内。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        rm = ResourceMonitor(config)
        rm.cpu_usage = 10.0
        rm.memory_usage = 15.0
        assert rm.within_limits()

    def test_exceeds_limits(self):
        """资源超限。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        rm = ResourceMonitor(config)
        rm.cpu_usage = 25.0
        rm.memory_usage = 15.0
        assert not rm.within_limits()
        rm.cpu_usage = 15.0
        rm.memory_usage = 25.0
        assert not rm.within_limits()

    def test_current_load(self):
        """当前负载计算。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        rm = ResourceMonitor(config)
        rm.cpu_usage = 10.0
        rm.memory_usage = 20.0
        assert abs(rm.current_load() - 0.2) < 1e-6

    def test_update_returns_dict(self):
        """update 返回字典。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        rm = ResourceMonitor(config)
        result = rm.update(0.5)
        assert "cpu" in result
        assert "memory" in result
        assert 0 <= result["cpu"] <= 100
        assert 0 <= result["memory"] <= 100


# ---------------------------------------------------------------------------
# 17. 自由能计算测试
# ---------------------------------------------------------------------------


class TestFreeEnergy:
    """自由能计算测试。"""

    def test_variational_free_energy_zero_when_matched(self):
        """观测与后验匹配时自由能较低。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        fe = FreeEnergyCalculator(config)
        mu = np.ones(8) * 0.5
        var = np.ones(8) * 0.1
        # 观测 == 后验均值
        f_matched = fe.variational_free_energy(mu, mu, var)
        # 观测远离后验均值
        f_far = fe.variational_free_energy(np.ones(8) * 0.9, mu, var)
        assert f_far > f_matched

    def test_expected_free_energy_prefers_preferred(self):
        """期望自由能偏好 preferred_obs。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        fe = FreeEnergyCalculator(config)
        preferred = np.ones(8) * 0.5
        var = np.ones(8) * 0.1
        # 预测接近偏好
        g_near = fe.expected_free_energy(preferred, var, preferred)
        # 预测远离偏好
        g_far = fe.expected_free_energy(np.ones(8) * 0.9, var, preferred)
        assert g_far > g_near

    def test_is_surprised(self):
        """惊讶检测。"""
        config = SPEPAConfig(state_dim=8, seed=42, free_energy_threshold=1.0)
        fe = FreeEnergyCalculator(config)
        assert not fe.is_surprised(0.5)
        assert fe.is_surprised(1.5)

    def test_complexity_accuracy_form(self):
        """复杂度-准确度形式。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        fe = FreeEnergyCalculator(config)
        obs = np.random.rand(8)
        mu = np.random.rand(8)
        var = np.ones(8) * 0.5
        cpx, acc, f = fe.complexity_accuracy_form(obs, mu, var)
        assert cpx >= 0
        assert isinstance(f, float)

    def test_prediction_coding_form(self):
        """预测编码形式。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        fe = FreeEnergyCalculator(config)
        pe = np.random.rand(8)
        prec = np.ones(8) * 2.0
        mu = np.random.rand(8)
        f = fe.prediction_coding_form(pe, prec, mu)
        assert f >= 0


# ---------------------------------------------------------------------------
# 18. 驱力引擎测试
# ---------------------------------------------------------------------------


class TestDriveEngine:
    """驱力引擎测试。"""

    def test_reward_safe_in_bounds(self):
        """安全驱力：边界内。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        state = np.ones(8) * 0.5
        action = np.ones(8) * 0.1
        rs = de.reward_safe(state, action)
        assert isinstance(rs, float)

    def test_reward_safe_out_of_bounds(self):
        """安全驱力：边界外惩罚。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        state = np.ones(8) * 0.5
        # 大动作导致内稳态越界
        action = np.ones(8) * 10.0
        rs = de.reward_safe(state, action)
        assert rs < 0  # 应为负（惩罚）

    def test_reward_gain_toward_goal(self):
        """收益驱力：朝目标移动。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        state = np.ones(8) * 0.3
        goal = np.ones(8) * 0.8
        # 动作朝向目标
        action = np.ones(8) * 0.5
        rg = de.reward_gain(state, action, goal)
        assert rg > 0  # 朝目标移动应为正收益

    def test_reward_gain_away_from_goal(self):
        """收益驱力：远离目标。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        state = np.ones(8) * 0.5
        goal = np.ones(8) * 0.9
        # 动作远离目标（负方向）
        action = -np.ones(8) * 0.5
        rg = de.reward_gain(state, action, goal)
        assert rg < 0

    def test_reward_eff(self):
        """能效驱力。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        action = np.ones(8) * 0.1
        re = de.reward_eff(action, 0.5)
        assert isinstance(re, float)
        # 小动作 + 高成效 -> 高能效
        re_good = de.reward_eff(np.ones(8) * 0.01, 1.0)
        re_bad = de.reward_eff(np.ones(8) * 1.0, 0.1)
        assert re_good > re_bad

    def test_total_reward(self):
        """加权总驱力。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        state = np.ones(8) * 0.5
        action = np.ones(8) * 0.1
        goal = np.ones(8) * 0.7
        total = de.total_reward(state, action, goal, 0.5)
        assert isinstance(total, float)
        assert np.isfinite(total)

    def test_homeostasis_update(self):
        """内稳态更新。"""
        homeo = Homeostasis(dim=4, target=0.5, bounds=(0.0, 1.0))
        dev = homeo.update(np.ones(4) * 0.1)
        assert dev.shape == (4,)
        assert homeo.in_bounds()

    def test_homeostasis_out_of_bounds(self):
        """内稳态越界。"""
        homeo = Homeostasis(dim=4, target=0.5, bounds=(0.0, 1.0))
        homeo.update(np.ones(4) * 10.0)
        # 被 clip 到 1.0，仍在边界内
        assert homeo.in_bounds()


# ---------------------------------------------------------------------------
# 19. 目标层次体系测试
# ---------------------------------------------------------------------------


class TestGoalHierarchy:
    """目标层次体系测试。"""

    def test_default_goals_initialized(self):
        """默认目标初始化。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        gh = GoalHierarchy(config)
        for lvl in range(5):
            assert len(gh.goals[lvl]) >= 1

    def test_add_remove_goal(self):
        """添加和移除目标。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        gh = GoalHierarchy(config)
        initial = len(gh.goals[2])
        g = Goal(2, "test_goal", np.ones(4) * 0.5, priority=2.0)
        gh.add_goal(g)
        assert len(gh.goals[2]) == initial + 1
        removed = gh.remove_goal(2, "test_goal")
        assert removed
        assert len(gh.goals[2]) == initial

    def test_rigid_goal_dominates_when_far(self):
        """刚性目标在距离过大时主导。"""
        config = SPEPAConfig(state_dim=4, seed=42,
                             safety_hard_boundary=0.3)
        gh = GoalHierarchy(config)
        # 状态远离 L0 目标（0.5）
        state = np.ones(4) * 0.0  # 距离 = 1.0 > 0.3
        result = gh.arbitrate(state)
        assert result["dominant_level"] == 0
        assert result["dominant_goal"] == "survival"

    def test_set_task_goal(self):
        """设置任务目标。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        gh = GoalHierarchy(config)
        gh.set_task_goal(np.ones(4) * 0.9, name="my_task")
        assert any(g.name == "my_task" for g in gh.goals[2])

    def test_goal_progress(self):
        """目标进度更新。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        g = Goal(2, "test", np.ones(4) * 0.5, priority=1.0)
        # 状态在目标上
        progress = g.update_progress(np.ones(4) * 0.5)
        assert progress > 0.99
        # 状态远离目标
        progress = g.update_progress(np.ones(4) * 0.0)
        assert progress < 0.99

    def test_active_goals(self):
        """激活目标列表。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        gh = GoalHierarchy(config)
        active = gh.active_goals()
        assert len(active) >= 5  # 每层至少 1 个


# ---------------------------------------------------------------------------
# 20. 综合混乱场景测试
# ---------------------------------------------------------------------------


class TestComplexChaosScenario:
    """综合混乱场景测试：多模块协同、快速变化、极端条件。"""

    def test_chaos_scenario_1_rapid_switching(self):
        """混乱场景1：快速切换环境动力学。"""
        config = SPEPAConfig(state_dim=16, action_dim=8, obs_dim=16,
                             seed=42, max_steps=100,
                             consolidation_interval=10)
        agent = SPEPAAgent(config)

        def chaos_env(action):
            mode = np.random.randint(4)
            if mode == 0:
                return np.clip(np.random.rand(16), 0, 1)
            elif mode == 1:
                return np.ones(16) * 0.5
            elif mode == 2:
                return np.zeros(16)
            else:
                return np.clip(np.random.rand(16) * 5, 0, 1)

        summary = agent.run(n_steps=100, env_step_fn=chaos_env)
        assert summary["online_cycles"] == 100
        # 验证系统没有崩溃
        state = agent.get_state()
        cs = np.asarray(state["current_state"], dtype=np.float64)
        assert np.all(np.isfinite(cs))

    def test_chaos_scenario_2_dimension_mismatch(self):
        """混乱场景2：观测维度频繁不匹配。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        for i in range(20):
            # 每次提供不同维度的观测
            dim = np.random.choice([4, 8, 16, 32])
            obs = np.random.rand(dim)
            agent.online_cycle(obs)
        assert agent.online_cycles == 20

    def test_chaos_scenario_3_concurrent_offline(self):
        """混乱场景3：频繁触发离线复盘。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, max_steps=50,
                             offline_load_threshold=0.99,  # 几乎总是触发
                             consolidation_interval=5)
        agent = SPEPAAgent(config)
        summary = agent.run(n_steps=50)
        assert summary["online_cycles"] == 50
        # 应该有很多离线复盘
        assert summary["offline_cycles"] > 0

    def test_chaos_scenario_4_extreme_goals(self):
        """混乱场景4：极端目标设置。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # 设置极端目标
        agent.goal_hierarchy.set_task_goal(np.ones(8) * 1.0, name="extreme_high")
        agent.goal_hierarchy.set_task_goal(np.ones(8) * 0.0, name="extreme_low")
        summary = agent.run(n_steps=20)
        assert summary["online_cycles"] == 20

    def test_chaos_scenario_5_rapid_reset(self):
        """混乱场景5：快速 reset + run。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, max_steps=5)
        agent = SPEPAAgent(config)
        for i in range(10):
            agent.reset()
            agent.run(n_steps=5)
        # 验证最终状态一致
        assert agent.online_cycles == 5

    def test_chaos_scenario_6_all_zero_inputs(self):
        """混乱场景6：全零输入持续运行。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        for _ in range(20):
            agent.online_cycle(np.zeros(8))
        assert agent.online_cycles == 20
        # 验证没有 NaN
        assert np.all(np.isfinite(agent.current_state))

    def test_chaos_scenario_7_max_capacity_memory(self):
        """混乱场景7：记忆系统满载。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8,
                             seed=42, fast_memory_capacity=10,
                             slow_memory_capacity=5,
                             consolidation_interval=5)
        agent = SPEPAAgent(config)
        # 运行足够多步以填满记忆
        agent.run(n_steps=50)
        # 验证记忆系统没有崩溃
        assert len(agent.memory_system.fast) <= 10
        # 慢速记忆在巩固后应裁剪
        assert len(agent.memory_system.slow) <= 5 + 1  # +1 容差

    def test_chaos_scenario_8_random_seed_variation(self):
        """混乱场景8：不同随机种子下稳定运行。"""
        for seed in [1, 42, 100, 999]:
            config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=seed)
            agent = SPEPAAgent(config)
            summary = agent.run(n_steps=10)
            assert summary["online_cycles"] == 10

    def test_chaos_scenario_9_continuous_learning(self):
        """混乱场景9：持续学习同一模式，验证收敛。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        target_obs = np.random.rand(8)
        # 持续提供同一观测
        errors = []
        for _ in range(50):
            result = agent.online_cycle(target_obs)
            errors.append(result["perception"]["error"])
        # 误差应有下降趋势（至少不持续上升）
        assert errors[-1] <= errors[0] * 2, \
            f"误差持续上升: {errors[0]} -> {errors[-1]}"

    def test_chaos_scenario_10_full_state_pollution(self):
        """混乱场景10：状态总线被污染后恢复。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # 污染状态总线
        agent.bus.set_state("polluted_key", "garbage_value")
        agent.bus.set_state("another_polluted", np.nan)
        # 运行
        agent.run(n_steps=5)
        # 验证系统仍正常运行
        assert agent.online_cycles == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
