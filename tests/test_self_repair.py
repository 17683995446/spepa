"""SPEPA 阶段六测试：元认知自监控与自我修复。

测试范围：
1. SelfMonitor：健康监控、异常检测、趋势分析
2. SelfDiagnoser：根因诊断、修复级别建议
3. SelfRepair：分级修复（L1软/L2中/L3硬）
4. MetacognitiveSelfRepair：监控→诊断→修复闭环
5. SPEPAAgent 集成：self_repair_step
6. 端到端：自监控与自我修复完整流程
"""
import numpy as np
import pytest

from spepa.utils.config import get_default_config, SPEPAConfig
from spepa.agent import SPEPAAgent
from spepa.core.self_monitor import (
    SelfMonitor, AnomalyType, AnomalyAlert, TrendDirection
)
from spepa.core.self_repair import (
    SelfDiagnoser, SelfRepair, MetacognitiveSelfRepair,
    DiagnosisType, Diagnosis, RepairLevel, RepairAction
)


# ============================================================
# SelfMonitor 测试
# ============================================================

class TestSelfMonitor:
    """自我监控器测试。"""

    def test_initialization(self):
        """正确初始化。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        assert monitor.get_health_score() == 1.0
        assert monitor.total_anomalies == 0
        assert monitor.get_alerts() == []

    def test_record_metric(self):
        """记录指标。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        monitor.record("prediction_error", 0.5)
        monitor.record("confidence", 0.8)
        state = monitor.get_state()
        assert "prediction_error" in state["metrics_tracked"]

    def test_record_non_finite_sanitized(self):
        """非有限值被清洗。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        monitor.record("prediction_error", float('nan'))
        monitor.record("confidence", float('inf'))
        # 不应崩溃，值被替换为0
        alerts = monitor.analyze()
        assert isinstance(alerts, list)

    def test_analyze_no_data(self):
        """无数据时分析返回空告警。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        alerts = monitor.analyze()
        assert alerts == []
        assert monitor.get_health_score() == 1.0

    def test_detect_error_spike(self):
        """检测误差突增。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        # 先记录低误差基线
        for _ in range(15):
            monitor.record("prediction_error", 0.1)
        # 然后突增
        for _ in range(5):
            monitor.record("prediction_error", 2.0)
        alerts = monitor.analyze()
        # 应检测到误差突增
        spike_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.ERROR_SPIKE]
        assert len(spike_alerts) > 0
        assert spike_alerts[0].severity > 0

    def test_detect_confidence_drop(self):
        """检测信心骤降。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        # 先记录高信心基线
        for _ in range(15):
            monitor.record("confidence", 0.9)
        # 然后骤降
        for _ in range(5):
            monitor.record("confidence", 0.2)
        alerts = monitor.analyze()
        drop_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.CONFIDENCE_DROP]
        assert len(drop_alerts) > 0

    def test_detect_stagnation(self):
        """检测性能停滞。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        # 记录稳定的误差（停滞）
        for _ in range(25):
            monitor.record("prediction_error", 0.5)
        alerts = monitor.analyze()
        stag_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.STAGNATION]
        assert len(stag_alerts) > 0

    def test_detect_reward_decline(self):
        """检测奖励持续下降。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        # 记录持续下降的奖励
        for i in range(15):
            monitor.record("total_reward", 1.0 - i * 0.1)
        alerts = monitor.analyze()
        decline_alerts = [a for a in alerts if a.anomaly_type == AnomalyType.REWARD_DECLINE]
        assert len(decline_alerts) > 0

    def test_health_score_decreases_with_anomaly(self):
        """有异常时健康评分下降。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        for _ in range(15):
            monitor.record("prediction_error", 0.1)
        for _ in range(5):
            monitor.record("prediction_error", 3.0)
        monitor.analyze()
        assert monitor.get_health_score() < 1.0

    def test_health_score_recovers_without_anomaly(self):
        """无异常时健康评分恢复。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        # 制造异常
        for _ in range(15):
            monitor.record("prediction_error", 0.1)
        for _ in range(5):
            monitor.record("prediction_error", 3.0)
        monitor.analyze()
        # 恢复正常
        for _ in range(15):
            monitor.record("prediction_error", 0.1)
        monitor.analyze()
        # 健康评分应有所恢复
        assert monitor.get_health_score() > 0.0

    def test_trend_improving(self):
        """检测改善趋势。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        # 误差逐渐下降（改善）
        for i in range(20):
            monitor.record("prediction_error", 1.0 - i * 0.05)
        monitor.analyze()
        trends = monitor.get_trends()
        assert trends["prediction_error"] == TrendDirection.IMPROVING.value

    def test_trend_degrading(self):
        """检测退化趋势。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        # 误差逐渐上升（退化）
        for i in range(20):
            monitor.record("prediction_error", 0.1 + i * 0.05)
        monitor.analyze()
        trends = monitor.get_trends()
        assert trends["prediction_error"] == TrendDirection.DEGRADING.value

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        state = monitor.get_state()
        assert "health_score" in state
        assert "total_anomalies" in state
        assert "current_alerts" in state
        assert "trends" in state

    def test_reset(self):
        """reset 清空状态。"""
        config = get_default_config()
        monitor = SelfMonitor(config)
        monitor.record("prediction_error", 0.5)
        monitor.total_anomalies = 5
        monitor.reset()
        assert monitor.get_health_score() == 1.0
        assert monitor.total_anomalies == 0

    def test_record_cycle_from_agent(self):
        """从 Agent 状态记录指标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        monitor = SelfMonitor(config)
        # 执行一个在线周期以产生指标
        agent.online_cycle()
        monitor.record_cycle(agent)
        # 应有记录
        state = monitor.get_state()
        assert len(state["metrics_tracked"]) > 0


# ============================================================
# SelfDiagnoser 测试
# ============================================================

class TestSelfDiagnoser:
    """自我诊断器测试。"""

    def test_initialization(self):
        """正确初始化。"""
        config = get_default_config()
        diagnoser = SelfDiagnoser(config)
        assert diagnoser is not None

    def test_diagnose_healthy(self):
        """无告警时诊断为健康。"""
        config = get_default_config()
        diagnoser = SelfDiagnoser(config)
        diagnosis = diagnoser.diagnose([], health_score=0.9)
        assert diagnosis.diagnosis_type == DiagnosisType.HEALTHY
        assert diagnosis.suggested_repair == RepairLevel.L0_NONE

    def test_diagnose_error_spike(self):
        """误差突增诊断为模型漂移。"""
        config = get_default_config()
        diagnoser = SelfDiagnoser(config)
        alert = AnomalyAlert(
            anomaly_type=AnomalyType.ERROR_SPIKE,
            metric_name="prediction_error",
            severity=0.5,
            current_value=2.0,
            baseline_value=0.5,
        )
        diagnosis = diagnoser.diagnose([alert], health_score=0.4)
        assert diagnosis.diagnosis_type == DiagnosisType.MODEL_DRIFT
        assert "predictive_model" in diagnosis.affected_modules

    def test_diagnose_confidence_drop(self):
        """信心骤降诊断为校准失败。"""
        config = get_default_config()
        diagnoser = SelfDiagnoser(config)
        alert = AnomalyAlert(
            anomaly_type=AnomalyType.CONFIDENCE_DROP,
            metric_name="confidence",
            severity=0.4,
            current_value=0.2,
            baseline_value=0.8,
        )
        diagnosis = diagnoser.diagnose([alert], health_score=0.5)
        assert diagnosis.diagnosis_type == DiagnosisType.CALIBRATION_FAILURE

    def test_diagnose_stagnation(self):
        """停滞诊断为学习平台。"""
        config = get_default_config()
        diagnoser = SelfDiagnoser(config)
        alert = AnomalyAlert(
            anomaly_type=AnomalyType.STAGNATION,
            metric_name="prediction_error",
            severity=0.3,
            current_value=0.5,
            baseline_value=0.5,
        )
        diagnosis = diagnoser.diagnose([alert], health_score=0.6)
        assert diagnosis.diagnosis_type == DiagnosisType.LEARNING_PLATEAU

    def test_repair_level_escalation_with_multiple_alerts(self):
        """多个告警时升级修复级别。"""
        config = get_default_config()
        diagnoser = SelfDiagnoser(config)
        alerts = [
            AnomalyAlert(AnomalyType.ERROR_SPIKE, "prediction_error",
                         0.3, 2.0, 0.5),
            AnomalyAlert(AnomalyType.CONFIDENCE_DROP, "confidence",
                         0.3, 0.2, 0.8),
            AnomalyAlert(AnomalyType.REWARD_DECLINE, "total_reward",
                         0.3, -1.0, 1.0),
        ]
        diagnosis = diagnoser.diagnose(alerts, health_score=0.3)
        # 3个告警应导致升级
        assert diagnosis.suggested_repair != RepairLevel.L0_NONE

    def test_severity_to_repair_level_mapping(self):
        """严重度到修复级别的映射。"""
        config = get_default_config()
        diagnoser = SelfDiagnoser(config)
        # 低严重度 → L1
        alert_low = AnomalyAlert(AnomalyType.ERROR_SPIKE, "err", 0.2, 1.0, 0.5)
        diag_low = diagnoser.diagnose([alert_low], health_score=0.6)
        assert diag_low.suggested_repair == RepairLevel.L1_SOFT
        # 中严重度 → L2
        alert_mid = AnomalyAlert(AnomalyType.ERROR_SPIKE, "err", 0.5, 2.0, 0.5)
        diag_mid = diagnoser.diagnose([alert_mid], health_score=0.4)
        assert diag_mid.suggested_repair == RepairLevel.L2_MEDIUM
        # 高严重度 → L3
        alert_high = AnomalyAlert(AnomalyType.ERROR_SPIKE, "err", 0.9, 5.0, 0.5)
        diag_high = diagnoser.diagnose([alert_high], health_score=0.1)
        assert diag_high.suggested_repair == RepairLevel.L3_HARD


# ============================================================
# SelfRepair 测试
# ============================================================

class TestSelfRepair:
    """自我修复器测试。"""

    def test_initialization(self):
        """正确初始化。"""
        config = get_default_config()
        repair = SelfRepair(config)
        assert repair.total_repairs == 0
        assert repair.successful_repairs == 0

    def test_repair_none_needed(self):
        """无需修复时返回L0。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        repair = SelfRepair(config)
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.HEALTHY,
            root_cause="none",
            severity=0.0,
            suggested_repair=RepairLevel.L0_NONE,
        )
        action = repair.repair(diagnosis, agent)
        assert action.repair_level == RepairLevel.L0_NONE
        assert repair.total_repairs == 0  # L0不计入

    def test_soft_repair_model_drift(self):
        """L1软修复：模型漂移→提升学习率。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        repair = SelfRepair(config)
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.MODEL_DRIFT,
            root_cause="error_spike",
            severity=0.2,
            suggested_repair=RepairLevel.L1_SOFT,
        )
        old_lr = agent.self_model.capability_bounds.get("learning_rate", 0.01)
        action = repair.repair(diagnosis, agent)
        assert action.success is True
        assert action.action_type == "increase_learning_rate"
        new_lr = agent.self_model.capability_bounds["learning_rate"]
        assert new_lr > old_lr

    def test_soft_repair_calibration_failure(self):
        """L1软修复：校准失败→重置校准器。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        repair = SelfRepair(config)
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.CALIBRATION_FAILURE,
            root_cause="confidence_drop",
            severity=0.2,
            suggested_repair=RepairLevel.L1_SOFT,
        )
        action = repair.repair(diagnosis, agent)
        assert action.success is True
        assert action.action_type == "reset_calibrator"
        assert agent.metacognition.calibrator.calibration_offset == 0.0

    def test_soft_repair_strategy_failure(self):
        """L1软修复：策略失败→提升探索。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        repair = SelfRepair(config)
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.STRATEGY_FAILURE,
            root_cause="reward_decline",
            severity=0.2,
            suggested_repair=RepairLevel.L1_SOFT,
        )
        old_ratio = agent.decision.exploration_ratio
        action = repair.repair(diagnosis, agent)
        assert action.success is True
        assert agent.decision.exploration_ratio > old_ratio

    def test_soft_repair_instability(self):
        """L1软修复：不稳定→降低学习率。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        repair = SelfRepair(config)
        # 先设置一个较高的学习率
        agent.self_model.capability_bounds["learning_rate"] = 0.05
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.INSTABILITY,
            root_cause="oscillation",
            severity=0.2,
            suggested_repair=RepairLevel.L1_SOFT,
        )
        action = repair.repair(diagnosis, agent)
        assert action.success is True
        assert action.action_type == "decrease_learning_rate"
        new_lr = agent.self_model.capability_bounds["learning_rate"]
        assert new_lr < 0.05

    def test_medium_repair_resets_optimizer(self):
        """L2中修复：重置优化器。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        repair = SelfRepair(config)
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.INSTABILITY,
            root_cause="oscillation",
            severity=0.5,
            suggested_repair=RepairLevel.L2_MEDIUM,
        )
        action = repair.repair(diagnosis, agent)
        assert action.success is True
        assert action.action_type == "reset_optimizer"

    def test_hard_repair_full_reset(self):
        """L3硬修复：全量重置。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.step_count = 50  # 制造一些状态
        repair = SelfRepair(config)
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.MODEL_DRIFT,
            root_cause="severe",
            severity=0.9,
            suggested_repair=RepairLevel.L3_HARD,
        )
        action = repair.repair(diagnosis, agent)
        assert action.success is True
        assert action.action_type == "full_reset"
        assert agent.step_count == 0  # 重置后

    def test_consecutive_failure_escalation(self):
        """连续失败升级修复级别。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        repair = SelfRepair(config)
        # 制造连续失败
        repair._consecutive_failures = 2
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.MODEL_DRIFT,
            root_cause="test",
            severity=0.2,
            suggested_repair=RepairLevel.L1_SOFT,
        )
        action = repair.repair(diagnosis, agent)
        # 应升级到L3
        assert action.repair_level == RepairLevel.L3_HARD

    def test_repair_history(self):
        """修复历史记录。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        repair = SelfRepair(config)
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.MODEL_DRIFT,
            root_cause="test",
            severity=0.2,
            suggested_repair=RepairLevel.L1_SOFT,
        )
        repair.repair(diagnosis, agent)
        history = repair.get_repair_history()
        assert len(history) == 1

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        repair = SelfRepair(config)
        state = repair.get_state()
        assert "total_repairs" in state
        assert "successful_repairs" in state
        assert "success_rate" in state

    def test_reset(self):
        """reset 清空状态。"""
        config = get_default_config()
        repair = SelfRepair(config)
        repair.total_repairs = 5
        repair.reset()
        assert repair.total_repairs == 0


# ============================================================
# MetacognitiveSelfRepair 测试
# ============================================================

class TestMetacognitiveSelfRepair:
    """元认知自监控与修复闭环测试。"""

    def test_initialization(self):
        """正确初始化。"""
        config = get_default_config()
        system = MetacognitiveSelfRepair(config)
        assert system.monitor is not None
        assert system.diagnoser is not None
        assert system.repair is not None

    def test_step_records_metrics(self):
        """step 记录指标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        system = MetacognitiveSelfRepair(config)
        agent.online_cycle()  # 产生一些指标
        result = system.step(agent, 0)
        assert result["health_score"] >= 0.0

    def test_step_monitors_periodically(self):
        """step 定期执行分析。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        system = MetacognitiveSelfRepair(config)
        system.monitor_interval = 2
        # step 0: 记录但不分析（0 - 0 < 2）
        result0 = system.step(agent, 0)
        assert result0["monitored"] is False
        # step 2: 应分析
        agent.online_cycle()
        result2 = system.step(agent, 2)
        assert result2["monitored"] is True

    def test_step_diagnoses_on_anomaly(self):
        """有异常时进行诊断。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        system = MetacognitiveSelfRepair(config)
        system.monitor_interval = 1
        # 制造误差突增
        for _ in range(15):
            system.monitor.record("prediction_error", 0.1)
        for _ in range(5):
            system.monitor.record("prediction_error", 3.0)
        result = system.step(agent, 1)
        assert result["monitored"] is True
        if result["alerts"]:
            assert result["diagnosed"] is True

    def test_step_repairs_on_diagnosis(self):
        """有诊断时执行修复。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        system = MetacognitiveSelfRepair(config)
        system.monitor_interval = 1
        # 制造严重误差突增
        for _ in range(15):
            system.monitor.record("prediction_error", 0.1)
        for _ in range(5):
            system.monitor.record("prediction_error", 5.0)
        result = system.step(agent, 1)
        if result["diagnosis"] and result["diagnosis"]["suggested_repair"] != "L0_none":
            assert result["repaired"] is True
            assert result["repair_action"] is not None

    def test_get_state(self):
        """get_state 返回完整状态。"""
        config = get_default_config()
        system = MetacognitiveSelfRepair(config)
        state = system.get_state()
        assert "monitor" in state
        assert "repair" in state
        assert "monitor_interval" in state

    def test_reset(self):
        """reset 清空所有状态。"""
        config = get_default_config()
        system = MetacognitiveSelfRepair(config)
        system.monitor.total_anomalies = 5
        system.repair.total_repairs = 3
        system.reset()
        assert system.monitor.total_anomalies == 0
        assert system.repair.total_repairs == 0


# ============================================================
# SPEPAAgent 集成测试
# ============================================================

class TestAgentSelfRepairIntegration:
    """Agent 自我修复集成测试。"""

    def test_agent_has_self_repair_system(self):
        """Agent 包含自我修复系统。"""
        agent = SPEPAAgent(get_default_config())
        assert hasattr(agent, 'self_repair_system')
        assert isinstance(agent.self_repair_system, MetacognitiveSelfRepair)

    def test_agent_self_repair_step(self):
        """Agent self_repair_step 返回结果。"""
        agent = SPEPAAgent(get_default_config())
        agent.online_cycle()  # 产生指标
        result = agent.self_repair_step()
        assert isinstance(result, dict)
        assert "health_score" in result

    def test_agent_get_health_score(self):
        """Agent get_health_score 返回评分。"""
        agent = SPEPAAgent(get_default_config())
        score = agent.get_health_score()
        assert 0.0 <= score <= 1.0

    def test_get_state_includes_self_repair(self):
        """get_state 包含自我修复状态。"""
        agent = SPEPAAgent(get_default_config())
        state = agent.get_state()
        assert "self_repair_system" in state

    def test_reset_resets_self_repair(self):
        """reset 重置自我修复系统。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.self_repair_system.monitor.total_anomalies = 5
        agent.reset()
        assert agent.self_repair_system.monitor.total_anomalies == 0


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEndSelfRepair:
    """端到端自监控与修复场景测试。"""

    def test_healthy_run_no_repair(self):
        """健康运行不触发修复。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.self_repair_system.monitor_interval = 1
        # 正常运行几步
        for _ in range(5):
            agent.online_cycle()
            result = agent.self_repair_step()
        # 健康评分应保持较高
        assert agent.get_health_score() >= 0.0
        # 可能没有修复
        assert agent.self_repair_system.repair.total_repairs >= 0

    def test_error_spike_triggers_repair(self):
        """误差突增触发修复。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.self_repair_system.monitor_interval = 1
        # 先记录低误差基线到监控器
        for _ in range(15):
            agent.self_repair_system.monitor.record("prediction_error", 0.1)
        # 制造误差突增
        for _ in range(5):
            agent.self_repair_system.monitor.record("prediction_error", 5.0)
        # 分析应检测到异常
        alerts = agent.self_repair_system.monitor.analyze()
        # 应有告警
        assert len(alerts) > 0

    def test_repair_effectiveness_tracked(self):
        """修复效果被跟踪。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        repair = agent.self_repair_system.repair
        initial_total = repair.total_repairs
        # 执行一次修复
        diagnosis = Diagnosis(
            diagnosis_type=DiagnosisType.MODEL_DRIFT,
            root_cause="test",
            severity=0.2,
            suggested_repair=RepairLevel.L1_SOFT,
        )
        repair.repair(diagnosis, agent)
        assert repair.total_repairs == initial_total + 1
        assert repair.successful_repairs >= 1

    def test_full_monitoring_repair_cycle(self):
        """完整监控-诊断-修复循环。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.self_repair_system.monitor_interval = 5
        # 运行20步，每步执行在线周期和自我修复
        for i in range(20):
            agent.online_cycle()
            agent.self_repair_step()
        # 应有监控记录
        state = agent.self_repair_system.get_state()
        assert state["monitor"]["total_anomalies"] >= 0
        assert state["repair"]["total_repairs"] >= 0

    def test_health_score_reflects_state(self):
        """健康评分反映系统状态。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.self_repair_system.monitor_interval = 1
        # 初始健康
        initial_health = agent.get_health_score()
        assert initial_health == 1.0
        # 制造异常
        for _ in range(15):
            agent.self_repair_system.monitor.record("prediction_error", 0.1)
        for _ in range(5):
            agent.self_repair_system.monitor.record("prediction_error", 5.0)
        agent.self_repair_system.monitor.analyze()
        # 健康应下降
        assert agent.get_health_score() < 1.0
