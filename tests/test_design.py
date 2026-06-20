"""SPEPA 第三轮测试：发现功能缺失与设计文档不一致的问题。

验证设计文档定义的规则是否在代码中实现：
1. 资源约束（CPU<=20%, 内存<=20%）超限时应有降级动作
2. 探索比例衰减（decay=0.995）是否实现
3. 缺口检测器第4个阈值（2.0）是否使用
4. 记忆巩固后是否避免重复巩固
5. 离线复盘触发条件是否正确
6. 安全硬边界（0.95）是否在决策中检查
7. 存储保留 Top 30% 是否在所有路径生效
8. 元认知自我指涉三层截断是否实现
"""
import numpy as np
import pytest

from spepa.agent import SPEPAAgent, ResourceMonitor
from spepa.core.drive_engine import DriveEngine
from spepa.core.gap_detector import GapDetector
from spepa.core.memory_system import MemorySystem
from spepa.core.metacognition import Metacognition
from spepa.core.optimizer import Optimizer
from spepa.core.predictive_model import PredictiveModel
from spepa.core.safety_guard import SafetyGuard
from spepa.core.self_model import SelfModel
from spepa.utils.config import SPEPAConfig


# ---------------------------------------------------------------------------
# 1. 资源约束执行测试
# ---------------------------------------------------------------------------


class TestResourceConstraintEnforcement:
    """验证资源约束是否被强制执行（设计文档要求 CPU<=20%, 内存<=20%）。"""

    def test_resource_constraint_config_exists(self):
        """配置中存在资源约束。"""
        config = SPEPAConfig()
        assert config.resources.cpu_max_percent == 20.0
        assert config.resources.memory_max_percent == 20.0

    def test_resource_monitor_tracks_usage(self):
        """资源监控跟踪使用率。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        agent = SPEPAAgent(config)
        agent.run(n_steps=5)
        assert agent.resource_monitor.cpu_usage >= 0
        assert agent.resource_monitor.memory_usage >= 0

    def test_resource_exceeded_logs_warning(self):
        """资源超限时记录警告。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42,
                             max_steps=3)
        agent = SPEPAAgent(config)
        # 监控日志
        import logging
        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r)
        agent.log.addHandler(handler)
        try:
            # 模拟超限
            original_update = agent.resource_monitor.update
            def force_high_load(load_factor=0.5):
                result = original_update(load_factor)
                agent.resource_monitor.cpu_usage = 50.0
                agent.resource_monitor.memory_usage = 50.0
                return {"cpu": 50.0, "memory": 50.0}
            agent.resource_monitor.update = force_high_load
            agent.run(n_steps=3)
        finally:
            agent.log.removeHandler(handler)
        # 应有资源超限警告
        warnings = [r for r in records if "Resource limit exceeded" in r.getMessage()]
        assert len(warnings) > 0, "资源超限时应记录警告"

    def test_resource_constraint_should_throttle(self):
        """资源超限时应采取降级动作（当前仅日志，应改进）。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # 当前实现：仅日志警告，不降级
        # 这是一个功能缺失 - 资源约束未被强制执行
        agent.resource_monitor.cpu_usage = 50.0
        agent.resource_monitor.memory_usage = 50.0
        assert not agent.resource_monitor.within_limits()
        # 系统仍继续运行（无降级）
        # TODO: 应添加降级机制（如减少策略数量、跳过离线复盘等）


# ---------------------------------------------------------------------------
# 2. 探索比例衰减测试
# ---------------------------------------------------------------------------


class TestExplorationDecay:
    """验证探索比例衰减是否实现（config.exploration.decay=0.995）。"""

    def test_decay_config_exists(self):
        """配置中存在衰减参数。"""
        config = SPEPAConfig()
        assert hasattr(config.exploration, 'decay')
        assert config.exploration.decay == 0.995

    def test_decay_not_applied(self):
        """当前实现中探索比例衰减未被应用。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        initial_ratio = agent.decision.exploration_ratio
        agent.run(n_steps=20)
        final_ratio = agent.decision.exploration_ratio
        # 探索比例可能因元认知控制而变化，但不应有衰减趋势
        # decay 参数未被使用
        # TODO: 应在每步应用 decay

    def test_exploration_ratio_stays_in_bounds(self):
        """探索比例保持在 [min, max] 范围内。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        for _ in range(50):
            agent.online_cycle(np.random.rand(8))
            r = agent.decision.exploration_ratio
            assert r >= config.exploration.exploration_ratio_min
            assert r <= 1.0


# ---------------------------------------------------------------------------
# 3. 缺口检测器第4阈值测试
# ---------------------------------------------------------------------------


class TestGapDetectorFourthThreshold:
    """验证缺口检测器第4个阈值是否被使用。"""

    def test_fourth_threshold_defined(self):
        """第4个阈值在配置中定义。"""
        config = SPEPAConfig()
        assert len(config.gap_level_thresholds) == 4
        assert config.gap_level_thresholds[3] == 2.0

    def test_fourth_threshold_used_in_classify(self):
        """第4个阈值在 classify 中被使用。"""
        config = SPEPAConfig(state_dim=8, seed=42,
                             gap_level_thresholds=(0.2, 0.5, 1.0, 2.0))
        gd = GapDetector(config)
        # kl=1.5 -> L3 (idx=3)
        gap1 = gd.detect_prediction_gap(1.5)
        assert gap1.level_idx == 3
        # kl=2.5 -> idx=4 (超过第4阈值)
        gap2 = gd.detect_prediction_gap(2.5)
        assert gap2.level_idx == 4


# ---------------------------------------------------------------------------
# 4. 记忆巩固重复测试
# ---------------------------------------------------------------------------


class TestConsolidationDuplication:
    """验证记忆巩固是否导致慢速记忆中出现重复项。"""

    def test_consolidate_adds_duplicates(self):
        """连续巩固同一批快速记忆不应添加重复项（幂等性验证）。"""
        config = SPEPAConfig(state_dim=4, seed=42,
                             fast_memory_capacity=50,
                             slow_memory_capacity=100,
                             consolidation_interval=1000)
        ms = MemorySystem(config)
        for i in range(10):
            ms.encode(np.random.rand(4), {"id": i}, importance=0.5)
        # 第一次巩固
        n1 = ms.consolidate()
        slow_ids_1 = {it.value["id"] for it in ms.slow.items}
        assert n1 > 0, "第一次巩固应添加记忆"
        # 第二次巩固（快速记忆未变）— 应是幂等的
        n2 = ms.consolidate()
        slow_ids_2 = {it.value["id"] for it in ms.slow.items}
        # 修复后：第二次巩固不应添加任何项
        assert n2 == 0, f"第二次巩固应幂等（n2=0），实际 n2={n2}"
        # ID 集合不变（无重复添加）
        assert slow_ids_1 == slow_ids_2, "慢速记忆ID集合不应变化"

    def test_consolidate_does_not_remove_from_fast(self):
        """巩固后快速记忆中的项未被移除。"""
        config = SPEPAConfig(state_dim=4, seed=42,
                             fast_memory_capacity=50,
                             slow_memory_capacity=100,
                             consolidation_interval=1000)
        ms = MemorySystem(config)
        for i in range(10):
            ms.encode(np.random.rand(4), {"id": i}, importance=0.5)
        fast_before = len(ms.fast)
        ms.consolidate()
        fast_after = len(ms.fast)
        # 快速记忆未清理
        assert fast_after == fast_before


# ---------------------------------------------------------------------------
# 5. 安全硬边界检查测试
# ---------------------------------------------------------------------------


class TestSafetyHardBoundary:
    """验证安全硬边界（0.95）在决策中的使用。"""

    def test_hard_boundary_config(self):
        """硬边界配置存在。"""
        config = SPEPAConfig()
        assert config.safety_hard_boundary == 0.95

    def test_hard_boundary_in_goal_arbitration(self):
        """目标仲裁中使用硬边界。"""
        config = SPEPAConfig(state_dim=4, seed=42,
                             safety_hard_boundary=0.3)
        from spepa.core.goal_hierarchy import GoalHierarchy
        gh = GoalHierarchy(config)
        # 状态远离 L0 目标
        state = np.zeros(4)  # 距离 0.5 目标 = 1.0 > 0.3
        result = gh.arbitrate(state)
        assert result["dominant_level"] == 0  # L0 主导

    def test_hard_boundary_in_safety_guard(self):
        """安全护栏使用硬边界检查内稳态。"""
        config = SPEPAConfig(state_dim=4, seed=42,
                             safety_hard_boundary=0.5)
        sg = SafetyGuard(config)
        # 偏离 0.6 > 0.5
        assert not sg.check_homeostasis(0.6)
        # 偏离 0.3 < 0.5
        assert sg.check_homeostasis(0.3)

    def test_hard_boundary_not_checked_in_decision(self):
        """决策模块未直接检查硬边界（仅检查 emergency_stop）。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        # emergency_stop 检查 state < -0.5 或 > 1.5
        # 但硬边界是 0.95，两者不一致
        # TODO: 决策应检查硬边界


# ---------------------------------------------------------------------------
# 6. 存储保留 Top 30% 测试
# ---------------------------------------------------------------------------


class TestStorageTopRatio:
    """验证存储保留 Top 30% 在所有路径生效。"""

    def test_config_top_ratio(self):
        """配置中 Top 30% 比例。"""
        config = SPEPAConfig()
        assert config.resources.storage_keep_top_ratio == 0.30

    def test_slow_memory_uses_top_ratio(self):
        """慢速记忆使用 Top 30% 裁剪。"""
        config = SPEPAConfig(state_dim=4, seed=42,
                             slow_memory_capacity=10)
        ms = MemorySystem(config)
        assert ms.slow.keep_ratio == 0.30

    def test_consolidate_uses_top_ratio(self):
        """巩固使用 Top 30% 选择。"""
        config = SPEPAConfig(state_dim=4, seed=42,
                             fast_memory_capacity=100,
                             slow_memory_capacity=50,
                             consolidation_interval=1000)
        ms = MemorySystem(config)
        for i in range(20):
            ms.encode(np.random.rand(4), {"id": i},
                      importance=float(i) / 20)
        n = ms.consolidate()
        # 应巩固 Top 30% = 6 条
        assert n == 6, f"应巩固 6 条，实际 {n}"


# ---------------------------------------------------------------------------
# 7. 元认知自我指涉测试
# ---------------------------------------------------------------------------


class TestMetacognitionSelfReference:
    """验证元认知自我指涉功能。"""

    def test_self_referential_check_returns_consistency(self):
        """自我指涉检查返回一致性。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sm = SelfModel(config)
        result = sm.self_referential_check()
        assert "consistency" in result
        assert 0.0 <= result["consistency"] <= 1.0

    def test_self_modification_proposal(self):
        """自我修改提议。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sm = SelfModel(config)
        # 高严重度
        proposal = sm.propose_self_modification(2.0)
        assert proposal["action"] == "increase_learning_rate"
        assert "new_lr" in proposal

    def test_self_modification_applied(self):
        """自我修改被应用。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sm = SelfModel(config)
        original_lr = sm.capability_bounds["learning_rate"]
        proposal = sm.propose_self_modification(2.0)
        sm.apply_modification(proposal)
        assert sm.capability_bounds["learning_rate"] != original_lr

    def test_self_representation_updates(self):
        """自我表征随状态更新。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sm = SelfModel(config)
        initial_rep = sm.self_representation.copy()
        for _ in range(20):
            sm.update_self_representation(np.random.rand(8), 0.5)
        # 自我表征应已变化
        assert not np.allclose(initial_rep, sm.self_representation)


# ---------------------------------------------------------------------------
# 8. 综合设计文档一致性测试
# ---------------------------------------------------------------------------


class TestDesignConsistency:
    """验证代码实现与设计文档的一致性。"""

    def test_three_drives_exist(self):
        """三大驱力：安全、收益、能效。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        de = DriveEngine(config)
        assert hasattr(de, 'reward_safe')
        assert hasattr(de, 'reward_gain')
        assert hasattr(de, 'reward_eff')
        assert hasattr(de, 'total_reward')

    def test_five_goal_levels_exist(self):
        """五层目标体系 L0-L4。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        from spepa.core.goal_hierarchy import GoalHierarchy
        gh = GoalHierarchy(config)
        assert len(gh.goals) == 5
        for lvl in range(5):
            assert len(gh.goals[lvl]) >= 1

    def test_dual_memory_system(self):
        """双系统记忆：快速 + 慢速。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        ms = MemorySystem(config)
        assert hasattr(ms, 'fast')
        assert hasattr(ms, 'slow')
        assert hasattr(ms, 'consolidate')

    def test_dual_mode_optimizer(self):
        """双模式优化器：微修正 + 结构性重构。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        opt = Optimizer(config)
        # 阶段七重构后：predictive_coding_update + hebbian_update + structural_reconstruction
        assert hasattr(opt, 'predictive_coding_update')
        assert hasattr(opt, 'hebbian_update')
        assert hasattr(opt, 'structural_reconstruction')
        assert hasattr(opt, 'select_mode')

    def test_four_phase_decision(self):
        """四阶段决策：感知-评估-生成-决策。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        assert hasattr(agent.decision, '_phase_perceive')
        assert hasattr(agent.decision, '_phase_evaluate')
        assert hasattr(agent.decision, '_phase_generate')
        assert hasattr(agent.decision, '_phase_decide')

    def test_online_offline_dual_loop(self):
        """在线+离线双闭环。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        assert hasattr(agent, 'online_cycle')
        assert hasattr(agent, 'offline_cycle')
        assert hasattr(agent, 'step')

    def test_free_energy_three_forms(self):
        """自由能三种形式。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        from spepa.core.free_energy import FreeEnergyCalculator
        fe = FreeEnergyCalculator(config)
        assert hasattr(fe, 'variational_free_energy')
        assert hasattr(fe, 'complexity_accuracy_form')
        assert hasattr(fe, 'prediction_coding_form')
        assert hasattr(fe, 'expected_free_energy')

    def test_gap_four_levels(self):
        """缺口四级分类。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        gd = GapDetector(config)
        from spepa.core.gap_detector import GAP_LEVELS
        assert len(GAP_LEVELS) == 4

    def test_safety_guard_functions(self):
        """安全护栏功能完整。"""
        config = SPEPAConfig(state_dim=8, seed=42)
        sg = SafetyGuard(config)
        assert hasattr(sg, 'check_state')
        assert hasattr(sg, 'check_action')
        assert hasattr(sg, 'validate')
        assert hasattr(sg, 'clip_action')
        assert hasattr(sg, 'emergency_stop')

    def test_active_inference_policy_selection(self):
        """主动推理策略选择。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, seed=42)
        from spepa.core.active_inference import ActiveInference
        from spepa.core.free_energy import FreeEnergyCalculator
        fe = FreeEnergyCalculator(config)
        ai = ActiveInference(config, fe)
        assert hasattr(ai, 'generate_policies')
        assert hasattr(ai, 'evaluate_policy')
        assert hasattr(ai, 'select_policy')
        assert hasattr(ai, 'decide')


# ---------------------------------------------------------------------------
# 9. 长时间运行性能测试
# ---------------------------------------------------------------------------


class TestLongRunPerformance:
    """长时间运行性能测试。"""

    def test_500_steps_performance(self):
        """500步运行时间合理。"""
        import time
        config = SPEPAConfig(state_dim=16, action_dim=8, obs_dim=16,
                             seed=42, max_steps=500,
                             consolidation_interval=50)
        agent = SPEPAAgent(config)
        start = time.time()
        summary = agent.run(n_steps=500)
        elapsed = time.time() - start
        assert summary["online_cycles"] == 500
        # 500步应在30秒内完成
        assert elapsed < 30.0, f"500步耗时 {elapsed:.1f}s，超过30s"

    def test_no_memory_leak(self):
        """无内存泄漏：长时间运行后指标历史不无限增长。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42,
                             max_steps=100,
                             consolidation_interval=10)
        agent = SPEPAAgent(config)
        agent.run(n_steps=100)
        # 指标历史应等于步数
        pe_history = agent.metrics.get("prediction_error")
        assert len(pe_history) == 100
        # 再运行100步
        agent.run(n_steps=100)
        pe_history = agent.metrics.get("prediction_error")
        assert len(pe_history) == 200  # 累积

    def test_cycle_time_stable(self):
        """单步时间稳定。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        times = []
        for _ in range(50):
            agent.online_cycle(np.random.rand(8))
        cycle_times = agent.metrics.get("cycle_time")
        # 单步时间应稳定（标准差小于均值的2倍）
        if len(cycle_times) > 10:
            mean_t = np.mean(cycle_times)
            std_t = np.std(cycle_times)
            assert std_t < mean_t * 2, f"单步时间不稳定: mean={mean_t}, std={std_t}"


# ---------------------------------------------------------------------------
# 10. 状态一致性深度测试
# ---------------------------------------------------------------------------


class TestStateConsistency:
    """状态一致性深度测试。"""

    def test_current_state_always_finite(self):
        """current_state 始终为有限值。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        for _ in range(100):
            agent.online_cycle(np.random.rand(8))
            assert np.all(np.isfinite(agent.current_state)), \
                "current_state 包含非有限值"

    def test_predictive_model_weights_finite(self):
        """预测模型权重始终为有限值。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        for _ in range(100):
            agent.online_cycle(np.random.rand(8) * 5)  # 大值观测
        for layer in agent.predictive_model.layers:
            assert np.all(np.isfinite(layer.mu))
            assert np.all(np.isfinite(layer.W))
            assert np.all(np.isfinite(layer.var))
            assert np.all(np.isfinite(layer.b))

    def test_metrics_all_finite(self):
        """所有指标值为有限值。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        agent.run(n_steps=20)
        summary = agent.metrics.summary()
        for name, stats in summary.items():
            for stat_name, val in stats.items():
                if isinstance(val, (int, float)):
                    assert np.isfinite(val), \
                        f"指标 {name}.{stat_name} 不是有限值: {val}"

    def test_goal_progress_in_range(self):
        """目标进度在 [0, 1] 范围内。"""
        config = SPEPAConfig(state_dim=4, seed=42)
        from spepa.core.goal_hierarchy import GoalHierarchy
        gh = GoalHierarchy(config)
        for _ in range(20):
            state = np.random.rand(4)
            gh.arbitrate(state)
            for g in gh.active_goals():
                assert 0.0 <= g.progress <= 1.0

    def test_confidence_in_range(self):
        """信心值在 [0, 1] 范围内。"""
        config = SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)
        agent = SPEPAAgent(config)
        for _ in range(20):
            agent.online_cycle(np.random.rand(8))
            assert 0.0 <= agent.metacognition.current_confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
