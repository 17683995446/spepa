"""SPEPA 自我诊断与自我修复。

设计原则（第一性原理）：
- 诊断原理：异常告警应映射到根因，而非盲目修复
- 分级修复原理：从最轻的修复开始，逐步升级
- 反馈原理：修复后观察效果，有效则保持，无效则升级

诊断→修复映射：
- ERROR_SPIKE → model_drift → L1:提升学习率 / L2:重置预测模型层
- CONFIDENCE_DROP → calibration_failure → L1:重置校准器 / L2:提升探索
- REWARD_DECLINE → strategy_failure → L1:提升探索 / L2:巩固技能
- STAGNATION → learning_plateau → L1:提升探索 / L2:触发离线复盘
- OSCILLATION → instability → L1:降低学习率 / L2:重置优化器

修复级别：
- L0: 无需修复
- L1: 软修复（调整超参数）
- L2: 中修复（重置特定模块）
- L3: 硬修复（全量重置，最后手段）
"""
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.core.self_monitor import (
    SelfMonitor, AnomalyType, AnomalyAlert, TrendDirection
)


class DiagnosisType(Enum):
    """诊断类型（根因分类）。"""
    HEALTHY = "healthy"
    MODEL_DRIFT = "model_drift"                # 模型漂移
    CALIBRATION_FAILURE = "calibration_failure"  # 校准失败
    STRATEGY_FAILURE = "strategy_failure"      # 策略失败
    LEARNING_PLATEAU = "learning_plateau"      # 学习停滞
    INSTABILITY = "instability"                # 不稳定
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # 资源耗尽


class RepairLevel(Enum):
    """修复级别。"""
    L0_NONE = "L0_none"          # 无需修复
    L1_SOFT = "L1_soft"          # 软修复：调整超参数
    L2_MEDIUM = "L2_medium"      # 中修复：重置模块
    L3_HARD = "L3_hard"          # 硬修复：全量重置


@dataclass
class Diagnosis:
    """诊断结果。"""
    diagnosis_type: DiagnosisType
    root_cause: str
    severity: float  # 0~1
    suggested_repair: RepairLevel
    affected_modules: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis_type": self.diagnosis_type.value,
            "root_cause": self.root_cause,
            "severity": self.severity,
            "suggested_repair": self.suggested_repair.value,
            "affected_modules": self.affected_modules,
            "description": self.description,
        }


@dataclass
class RepairAction:
    """修复动作记录。"""
    action_type: str
    target_module: str
    repair_level: RepairLevel = RepairLevel.L0_NONE
    params_before: Dict[str, Any] = field(default_factory=dict)
    params_after: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repair_level": self.repair_level.value,
            "action_type": self.action_type,
            "target_module": self.target_module,
            "success": self.success,
            "description": self.description,
        }


class SelfDiagnoser(LoggedMixin):
    """自我诊断器：将异常告警映射到根因诊断。

    诊断逻辑：
    1. 收集 SelfMonitor 的告警
    2. 根据告警类型和组合，推断根因
    3. 建议修复级别
    """

    # 异常类型 → 诊断类型的映射
    ANOMALY_TO_DIAGNOSIS = {
        AnomalyType.ERROR_SPIKE: DiagnosisType.MODEL_DRIFT,
        AnomalyType.CONFIDENCE_DROP: DiagnosisType.CALIBRATION_FAILURE,
        AnomalyType.REWARD_DECLINE: DiagnosisType.STRATEGY_FAILURE,
        AnomalyType.STAGNATION: DiagnosisType.LEARNING_PLATEAU,
        AnomalyType.OSCILLATION: DiagnosisType.INSTABILITY,
    }

    # 诊断类型 → 受影响模块
    DIAGNOSIS_TO_MODULES = {
        DiagnosisType.MODEL_DRIFT: ["predictive_model", "optimizer"],
        DiagnosisType.CALIBRATION_FAILURE: ["metacognition", "calibrator"],
        DiagnosisType.STRATEGY_FAILURE: ["decision", "active_inference"],
        DiagnosisType.LEARNING_PLATEAU: ["learning", "memory_system"],
        DiagnosisType.INSTABILITY: ["optimizer", "safety_guard"],
        DiagnosisType.RESOURCE_EXHAUSTION: ["resource_controller"],
    }

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.self_diagnoser")
        self.config = config
        self.log.info("SelfDiagnoser initialized")

    def diagnose(self, alerts: List[AnomalyAlert],
                 health_score: float) -> Diagnosis:
        """根据告警和健康评分进行诊断。

        参数：
        - alerts: SelfMonitor 检测到的告警列表
        - health_score: 当前健康评分

        返回：诊断结果
        """
        if not alerts or health_score > 0.7:
            return Diagnosis(
                diagnosis_type=DiagnosisType.HEALTHY,
                root_cause="no significant anomaly",
                severity=0.0,
                suggested_repair=RepairLevel.L0_NONE,
                description="System is healthy",
            )
        # 取最严重的告警作为主要诊断
        primary_alert = max(alerts, key=lambda a: a.severity)
        diag_type = self.ANOMALY_TO_DIAGNOSIS.get(
            primary_alert.anomaly_type, DiagnosisType.STRATEGY_FAILURE)
        affected = self.DIAGNOSIS_TO_MODULES.get(diag_type, [])
        severity = primary_alert.severity
        # 根据严重度建议修复级别
        if severity < 0.3:
            suggested = RepairLevel.L1_SOFT
        elif severity < 0.7:
            suggested = RepairLevel.L2_MEDIUM
        else:
            suggested = RepairLevel.L3_HARD
        # 多个告警同时出现，升级修复级别
        if len(alerts) >= 3:
            levels = list(RepairLevel)
            current_idx = levels.index(suggested)
            suggested = levels[min(current_idx + 1, len(levels) - 1)]
        root_cause = f"{primary_alert.anomaly_type.value} on {primary_alert.metric_name}"
        diagnosis = Diagnosis(
            diagnosis_type=diag_type,
            root_cause=root_cause,
            severity=severity,
            suggested_repair=suggested,
            affected_modules=affected,
            description=f"Primary anomaly: {primary_alert.description}",
        )
        self.log.info("Diagnosis: %s (severity=%.2f, repair=%s)",
                      diag_type.value, severity, suggested.value)
        return diagnosis


class SelfRepair(LoggedMixin):
    """自我修复器：根据诊断执行分级修复。

    修复策略（从轻到重）：
    - L1 软修复：调整超参数（学习率、探索比例）
    - L2 中修复：重置特定模块（校准器、优化器、技能库巩固）
    - L3 硬修复：全量重置（最后手段）
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.self_repair")
        self.config = config
        # 修复历史
        self._repair_history: List[RepairAction] = []
        # 统计
        self.total_repairs = 0
        self.successful_repairs = 0
        self.failed_repairs = 0
        # 连续修复失败计数（用于升级）
        self._consecutive_failures = 0
        self.log.info("SelfRepair initialized")

    def repair(self, diagnosis: Diagnosis, agent) -> RepairAction:
        """根据诊断执行修复。

        参数：
        - diagnosis: 诊断结果
        - agent: SPEPA Agent（用于访问和修改模块）

        返回：修复动作记录
        """
        if diagnosis.suggested_repair == RepairLevel.L0_NONE:
            # 重置连续失败计数
            self._consecutive_failures = 0
            return RepairAction(
                repair_level=RepairLevel.L0_NONE,
                action_type="none",
                target_module="none",
                description="No repair needed",
            )
        self.total_repairs += 1
        level = diagnosis.suggested_repair
        # 如果连续失败，升级修复级别
        if self._consecutive_failures >= 2 and level != RepairLevel.L3_HARD:
            level = RepairLevel.L3_HARD
            self.log.warning("Escalating to L3 hard repair after %d consecutive failures",
                             self._consecutive_failures)
        # 执行修复
        if level == RepairLevel.L1_SOFT:
            action = self._soft_repair(diagnosis, agent)
        elif level == RepairLevel.L2_MEDIUM:
            action = self._medium_repair(diagnosis, agent)
        else:
            action = self._hard_repair(diagnosis, agent)
        action.repair_level = level
        # 记录历史
        self._repair_history.append(action)
        if len(self._repair_history) > 100:
            self._repair_history = self._repair_history[-50:]
        if action.success:
            self.successful_repairs += 1
            self._consecutive_failures = 0
        else:
            self.failed_repairs += 1
            self._consecutive_failures += 1
        self.log.info("Repair executed: level=%s type=%s success=%s",
                      level.value, action.action_type, action.success)
        return action

    def _soft_repair(self, diagnosis: Diagnosis, agent) -> RepairAction:
        """L1 软修复：调整超参数。"""
        diag_type = diagnosis.diagnosis_type
        try:
            if diag_type == DiagnosisType.MODEL_DRIFT:
                # 提升学习率
                old_lr = agent.self_model.capability_bounds.get("learning_rate", 0.01)
                new_lr = min(0.1, old_lr * 2.0)
                agent.self_model.capability_bounds["learning_rate"] = new_lr
                return RepairAction(
                    action_type="increase_learning_rate",
                    target_module="self_model",
                    params_before={"learning_rate": old_lr},
                    params_after={"learning_rate": new_lr},
                    success=True,
                    description=f"Learning rate {old_lr:.4f} -> {new_lr:.4f}",
                )
            elif diag_type == DiagnosisType.CALIBRATION_FAILURE:
                # 重置校准器
                cal = agent.metacognition.calibrator
                old_offset = cal.calibration_offset
                cal.calibration_offset = 0.0
                cal.confidences.clear()
                cal.accuracies.clear()
                return RepairAction(
                    action_type="reset_calibrator",
                    target_module="metacognition.calibrator",
                    params_before={"offset": old_offset},
                    params_after={"offset": 0.0},
                    success=True,
                    description="Calibrator reset to zero offset",
                )
            elif diag_type == DiagnosisType.STRATEGY_FAILURE:
                # 提升探索比例
                old_ratio = agent.decision.exploration_ratio
                agent.decision.exploration_ratio = min(1.0, old_ratio + 0.2)
                return RepairAction(
                    action_type="increase_exploration",
                    target_module="decision",
                    params_before={"exploration_ratio": old_ratio},
                    params_after={"exploration_ratio": agent.decision.exploration_ratio},
                    success=True,
                    description=f"Exploration {old_ratio:.2f} -> {agent.decision.exploration_ratio:.2f}",
                )
            elif diag_type == DiagnosisType.LEARNING_PLATEAU:
                # 提升探索 + 触发离线复盘
                old_ratio = agent.decision.exploration_ratio
                agent.decision.exploration_ratio = min(1.0, old_ratio + 0.15)
                return RepairAction(
                    action_type="break_plateau",
                    target_module="decision",
                    params_before={"exploration_ratio": old_ratio},
                    params_after={"exploration_ratio": agent.decision.exploration_ratio},
                    success=True,
                    description="Increased exploration to break plateau",
                )
            elif diag_type == DiagnosisType.INSTABILITY:
                # 降低学习率
                old_lr = agent.self_model.capability_bounds.get("learning_rate", 0.01)
                new_lr = max(0.001, old_lr * 0.5)
                agent.self_model.capability_bounds["learning_rate"] = new_lr
                return RepairAction(
                    action_type="decrease_learning_rate",
                    target_module="self_model",
                    params_before={"learning_rate": old_lr},
                    params_after={"learning_rate": new_lr},
                    success=True,
                    description=f"Learning rate {old_lr:.4f} -> {new_lr:.4f}",
                )
            else:
                return RepairAction(
                    action_type="unknown_soft_repair",
                    target_module="none",
                    success=False,
                    description=f"No soft repair for {diag_type.value}",
                )
        except Exception as e:
            self.log.warning("Soft repair failed: %s", e)
            return RepairAction(
                action_type="soft_repair_error",
                target_module="unknown",
                success=False,
                description=str(e),
            )

    def _medium_repair(self, diagnosis: Diagnosis, agent) -> RepairAction:
        """L2 中修复：重置特定模块。"""
        diag_type = diagnosis.diagnosis_type
        try:
            if diag_type == DiagnosisType.MODEL_DRIFT:
                # 重置预测模型的部分层
                model = agent.predictive_model
                n_layers = len(model.layers)
                # 重置最后一层（最可能漂移的）
                if n_layers > 0:
                    last_layer = model.layers[-1]
                    old_var = float(np.mean(last_layer.var))
                    last_layer.var = np.ones_like(last_layer.var) * 0.5
                    return RepairAction(
                        action_type="reset_model_layer",
                        target_module="predictive_model",
                        params_before={"last_layer_var": old_var},
                        params_after={"last_layer_var": 0.5},
                        success=True,
                        description="Reset last predictive model layer variance",
                    )
            elif diag_type == DiagnosisType.CALIBRATION_FAILURE:
                # 重置整个元认知
                old_conf = agent.metacognition.current_confidence
                agent.metacognition.calibrator = type(agent.metacognition.calibrator)(
                    window=agent.metacognition.calibrator.window
                )
                agent.metacognition.current_confidence = 0.5
                return RepairAction(
                    action_type="reset_metacognition",
                    target_module="metacognition",
                    params_before={"confidence": old_conf},
                    params_after={"confidence": 0.5},
                    success=True,
                    description="Reset metacognition calibrator",
                )
            elif diag_type == DiagnosisType.STRATEGY_FAILURE:
                # 巩固技能库
                consolidated = agent.skill_learner.consolidate()
                return RepairAction(
                    action_type="consolidate_skills",
                    target_module="skill_learner",
                    params_before={},
                    params_after={"consolidated": consolidated},
                    success=True,
                    description=f"Consolidated {consolidated} skills",
                )
            elif diag_type == DiagnosisType.LEARNING_PLATEAU:
                # 触发离线复盘
                try:
                    agent.offline_replay.replay()
                    return RepairAction(
                        action_type="offline_replay",
                        target_module="offline_replay",
                        success=True,
                        description="Triggered offline replay",
                    )
                except Exception:
                    # 离线复盘不可用，退化为巩固技能
                    consolidated = agent.skill_learner.consolidate()
                    return RepairAction(
                        action_type="consolidate_skills",
                        target_module="skill_learner",
                        params_after={"consolidated": consolidated},
                        success=True,
                        description="Offline replay unavailable, consolidated skills instead",
                    )
            elif diag_type == DiagnosisType.INSTABILITY:
                # 重置优化器
                agent.optimizer.reset()
                return RepairAction(
                    action_type="reset_optimizer",
                    target_module="optimizer",
                    success=True,
                    description="Reset optimizer state",
                )
            # 默认：执行软修复
            return self._soft_repair(diagnosis, agent)
        except Exception as e:
            self.log.warning("Medium repair failed: %s", e)
            return RepairAction(
                action_type="medium_repair_error",
                target_module="unknown",
                success=False,
                description=str(e),
            )

    def _hard_repair(self, diagnosis: Diagnosis, agent) -> RepairAction:
        """L3 硬修复：全量重置。"""
        try:
            # 记录重置前状态
            old_state = agent.current_state.copy()
            old_step = agent.step_count
            # 执行全量重置
            agent.reset()
            return RepairAction(
                action_type="full_reset",
                target_module="agent",
                params_before={"step_count": old_step,
                               "state_norm": float(np.linalg.norm(old_state))},
                params_after={"step_count": 0,
                              "state_norm": float(np.linalg.norm(agent.current_state))},
                success=True,
                description="Full agent reset (L3 hard repair)",
            )
        except Exception as e:
            self.log.error("Hard repair failed: %s", e)
            return RepairAction(
                action_type="hard_repair_error",
                target_module="agent",
                success=False,
                description=str(e),
            )

    def get_repair_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """获取最近 n 次修复历史。"""
        return [a.to_dict() for a in self._repair_history[-n:]]

    def get_state(self) -> Dict[str, Any]:
        return {
            "total_repairs": self.total_repairs,
            "successful_repairs": self.successful_repairs,
            "failed_repairs": self.failed_repairs,
            "success_rate": (
                self.successful_repairs / max(1, self.total_repairs)
            ),
            "consecutive_failures": self._consecutive_failures,
            "history_size": len(self._repair_history),
        }

    def reset(self) -> None:
        """重置修复器。"""
        self._repair_history.clear()
        self.total_repairs = 0
        self.successful_repairs = 0
        self.failed_repairs = 0
        self._consecutive_failures = 0
        self.log.info("SelfRepair reset")


class MetacognitiveSelfRepair(LoggedMixin):
    """元认知自监控与自我修复的统一入口。

    集成 SelfMonitor + SelfDiagnoser + SelfRepair，
    提供一键式的"监控→诊断→修复"闭环。
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.metacognitive_repair")
        self.config = config
        self.monitor = SelfMonitor(config)
        self.diagnoser = SelfDiagnoser(config)
        self.repair = SelfRepair(config)
        # 监控间隔（每 N 步执行一次完整分析）
        self.monitor_interval = 10
        self._last_monitor_step = 0
        self.log.info("MetacognitiveSelfRepair initialized")

    def step(self, agent, step_count: int) -> Dict[str, Any]:
        """执行一轮监控-诊断-修复。

        参数：
        - agent: SPEPA Agent
        - step_count: 当前步数

        返回：执行结果字典
        """
        result = {
            "monitored": False,
            "diagnosed": False,
            "repaired": False,
            "health_score": self.monitor.get_health_score(),
            "alerts": [],
            "diagnosis": None,
            "repair_action": None,
        }
        # 1. 记录指标
        self.monitor.record_cycle(agent)
        # 2. 定期分析
        if step_count - self._last_monitor_step >= self.monitor_interval:
            self._last_monitor_step = step_count
            alerts = self.monitor.analyze()
            result["monitored"] = True
            result["alerts"] = [a.to_dict() for a in alerts]
            result["health_score"] = self.monitor.get_health_score()
            # 3. 诊断
            if alerts:
                diagnosis = self.diagnoser.diagnose(
                    alerts, self.monitor.get_health_score())
                result["diagnosed"] = True
                result["diagnosis"] = diagnosis.to_dict()
                # 4. 修复
                if diagnosis.suggested_repair != RepairLevel.L0_NONE:
                    repair_action = self.repair.repair(diagnosis, agent)
                    result["repaired"] = True
                    result["repair_action"] = repair_action.to_dict()
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "monitor": self.monitor.get_state(),
            "diagnoser": {
                "available": True,
            },
            "repair": self.repair.get_state(),
            "monitor_interval": self.monitor_interval,
        }

    def reset(self) -> None:
        """重置整个自监控-修复系统。"""
        self.monitor.reset()
        self.repair.reset()
        self._last_monitor_step = 0
        self.log.info("MetacognitiveSelfRepair reset")
