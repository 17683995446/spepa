"""SPEPA 自我监控器：持续健康监控与异常检测。

设计原则（第一性原理）：
- 内省原理：系统应能观察自身性能指标，发现退化趋势
- 趋势原理：单点指标无意义，趋势（改善/稳定/退化）才是健康信号
- 预警原理：在严重故障前检测到异常，争取修复窗口

监控指标：
- prediction_error: 预测误差趋势
- confidence: 信心趋势
- total_reward: 奖励趋势
- achievement_rate: 目标达成率趋势

异常类型：
- ERROR_SPIKE: 误差突增
- CONFIDENCE_DROP: 信心骤降
- REWARD_DECLINE: 奖励持续下降
- STAGNATION: 性能停滞（无改善）
- OSCILLATION: 指标振荡（不稳定）
"""
import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class AnomalyType(Enum):
    """异常类型。"""
    NONE = "none"
    ERROR_SPIKE = "error_spike"            # 误差突增
    CONFIDENCE_DROP = "confidence_drop"    # 信心骤降
    REWARD_DECLINE = "reward_decline"      # 奖励下降
    STAGNATION = "stagnation"              # 性能停滞
    OSCILLATION = "oscillation"            # 指标振荡


class TrendDirection(Enum):
    """趋势方向。"""
    IMPROVING = "improving"    # 改善中
    STABLE = "stable"          # 稳定
    DEGRADING = "degrading"    # 退化中


@dataclass
class AnomalyAlert:
    """异常告警。"""
    anomaly_type: AnomalyType
    metric_name: str
    severity: float  # 0~1，1 = 最严重
    current_value: float
    baseline_value: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type.value,
            "metric_name": self.metric_name,
            "severity": self.severity,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "description": self.description,
        }


class SelfMonitor(LoggedMixin):
    """自我监控器：持续跟踪性能指标，检测异常趋势。

    工作流程：
    1. 每个 online_cycle 后，record 各项指标
    2. 定期 analyze 趋势和异常
    3. 输出健康评分和异常告警
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.self_monitor")
        self.config = config
        # 指标历史窗口
        self.window_size = 50
        self._metrics_history: Dict[str, deque] = {
            "prediction_error": deque(maxlen=self.window_size),
            "confidence": deque(maxlen=self.window_size),
            "total_reward": deque(maxlen=self.window_size),
            "achievement_rate": deque(maxlen=self.window_size),
        }
        # 异常检测阈值
        self.error_spike_threshold = 2.0      # 误差突增倍数
        self.confidence_drop_threshold = 0.3   # 信心下降阈值
        self.reward_decline_steps = 10         # 奖励连续下降步数
        self.stagnation_steps = 20             # 停滞检测步数
        self.stagnation_improvement = 0.01     # 停滞改善阈值
        self.oscillation_threshold = 0.5       # 振荡变异系数阈值
        # 当前状态
        self._alerts: List[AnomalyAlert] = []
        self._health_score: float = 1.0  # 0~1，1 = 最健康
        self._trends: Dict[str, TrendDirection] = {
            m: TrendDirection.STABLE for m in self._metrics_history
        }
        # 统计
        self.total_anomalies = 0
        self.log.info("SelfMonitor initialized (window=%d)", self.window_size)

    def record(self, metric_name: str, value: float) -> None:
        """记录一个指标值。"""
        if metric_name not in self._metrics_history:
            self._metrics_history[metric_name] = deque(maxlen=self.window_size)
        # 清洗
        if not np.isfinite(value):
            value = 0.0
        self._metrics_history[metric_name].append(float(value))

    def record_cycle(self, agent) -> None:
        """从 Agent 状态记录一轮周期的所有指标。"""
        # 预测误差
        recent_errors = agent.metrics.get_recent("prediction_error", n=1)
        if recent_errors:
            self.record("prediction_error", recent_errors[-1])
        # 信心
        self.record("confidence", agent.metacognition.current_confidence)
        # 奖励
        recent_rewards = agent.metrics.get_recent("total_reward", n=1)
        if recent_rewards:
            self.record("total_reward", recent_rewards[-1])
        # 目标达成率
        ad_state = agent.autonomous_driver.get_state()
        self.record("achievement_rate", ad_state.get("achievement_rate", 0.0))

    def analyze(self) -> List[AnomalyAlert]:
        """分析所有指标，检测异常。返回当前告警列表。"""
        alerts = []
        # 检测误差突增
        alert = self._detect_error_spike()
        if alert:
            alerts.append(alert)
        # 检测信心骤降
        alert = self._detect_confidence_drop()
        if alert:
            alerts.append(alert)
        # 检测奖励下降
        alert = self._detect_reward_decline()
        if alert:
            alerts.append(alert)
        # 检测停滞
        alert = self._detect_stagnation()
        if alert:
            alerts.append(alert)
        # 检测振荡
        alert = self._detect_oscillation()
        if alert:
            alerts.append(alert)
        # 更新趋势
        self._update_trends()
        # 更新健康评分
        self._update_health_score(alerts)
        # 更新告警列表
        self._alerts = alerts
        if alerts:
            self.total_anomalies += len(alerts)
            for a in alerts:
                self.log.warning("Anomaly detected: %s on %s (severity=%.2f)",
                                 a.anomaly_type.value, a.metric_name, a.severity)
        return alerts

    def _compute_trend(self, metric_name: str) -> TrendDirection:
        """计算指标趋势。"""
        history = list(self._metrics_history.get(metric_name, []))
        if len(history) < 5:
            return TrendDirection.STABLE
        # 比较前半段和后半段的均值
        mid = len(history) // 2
        first_half = float(np.mean(history[:mid]))
        second_half = float(np.mean(history[mid:]))
        diff = second_half - first_half
        # 对于 error 和 reward，趋势方向含义不同
        if metric_name in ("prediction_error",):
            # 误差下降 = 改善
            if diff < -0.05:
                return TrendDirection.IMPROVING
            elif diff > 0.05:
                return TrendDirection.DEGRADING
        elif metric_name in ("confidence", "total_reward", "achievement_rate"):
            # 这些指标上升 = 改善
            if diff > 0.02:
                return TrendDirection.IMPROVING
            elif diff < -0.02:
                return TrendDirection.DEGRADING
        return TrendDirection.STABLE

    def _update_trends(self) -> None:
        """更新所有指标的趋势。"""
        for metric in self._metrics_history:
            self._trends[metric] = self._compute_trend(metric)

    def _detect_error_spike(self) -> Optional[AnomalyAlert]:
        """检测误差突增。"""
        history = list(self._metrics_history.get("prediction_error", []))
        if len(history) < 10:
            return None
        baseline = float(np.mean(history[:-3]))
        current = float(np.mean(history[-3:]))
        if baseline < 1e-6:
            return None
        ratio = current / baseline
        if ratio > self.error_spike_threshold:
            severity = min(1.0, (ratio - 1.0) / 3.0)
            return AnomalyAlert(
                anomaly_type=AnomalyType.ERROR_SPIKE,
                metric_name="prediction_error",
                severity=severity,
                current_value=current,
                baseline_value=baseline,
                description=f"Error spiked {ratio:.1f}x above baseline",
            )
        return None

    def _detect_confidence_drop(self) -> Optional[AnomalyAlert]:
        """检测信心骤降。"""
        history = list(self._metrics_history.get("confidence", []))
        if len(history) < 10:
            return None
        baseline = float(np.mean(history[:-3]))
        current = float(np.mean(history[-3:]))
        drop = baseline - current
        if drop > self.confidence_drop_threshold:
            severity = min(1.0, drop / 0.5)
            return AnomalyAlert(
                anomaly_type=AnomalyType.CONFIDENCE_DROP,
                metric_name="confidence",
                severity=severity,
                current_value=current,
                baseline_value=baseline,
                description=f"Confidence dropped by {drop:.2f}",
            )
        return None

    def _detect_reward_decline(self) -> Optional[AnomalyAlert]:
        """检测奖励持续下降。"""
        history = list(self._metrics_history.get("total_reward", []))
        if len(history) < self.reward_decline_steps:
            return None
        recent = history[-self.reward_decline_steps:]
        # 检查是否持续下降
        declining = all(recent[i] >= recent[i + 1]
                        for i in range(len(recent) - 1))
        if declining:
            total_drop = recent[0] - recent[-1]
            severity = min(1.0, abs(total_drop) / 5.0)
            return AnomalyAlert(
                anomaly_type=AnomalyType.REWARD_DECLINE,
                metric_name="total_reward",
                severity=severity,
                current_value=recent[-1],
                baseline_value=recent[0],
                description=f"Reward declining for {len(recent)} steps",
            )
        return None

    def _detect_stagnation(self) -> Optional[AnomalyAlert]:
        """检测性能停滞。"""
        history = list(self._metrics_history.get("prediction_error", []))
        if len(history) < self.stagnation_steps:
            return None
        recent = history[-self.stagnation_steps:]
        # 停滞 = 误差变化很小
        improvement = abs(float(np.max(recent)) - float(np.min(recent)))
        if improvement < self.stagnation_improvement:
            severity = min(1.0, 1.0 - improvement / self.stagnation_improvement)
            return AnomalyAlert(
                anomaly_type=AnomalyType.STAGNATION,
                metric_name="prediction_error",
                severity=severity,
                current_value=float(np.mean(recent)),
                baseline_value=float(np.mean(recent)),
                description=f"Performance stagnant for {len(recent)} steps",
            )
        return None

    def _detect_oscillation(self) -> Optional[AnomalyAlert]:
        """检测指标振荡。"""
        history = list(self._metrics_history.get("prediction_error", []))
        if len(history) < 10:
            return None
        recent = history[-10:]
        mean_val = float(np.mean(recent))
        if mean_val < 1e-6:
            return None
        std_val = float(np.std(recent))
        cv = std_val / mean_val  # 变异系数
        if cv > self.oscillation_threshold:
            severity = min(1.0, cv / 1.0)
            return AnomalyAlert(
                anomaly_type=AnomalyType.OSCILLATION,
                metric_name="prediction_error",
                severity=severity,
                current_value=cv,
                baseline_value=0.0,
                description=f"Error oscillating (CV={cv:.2f})",
            )
        return None

    def _update_health_score(self, alerts: List[AnomalyAlert]) -> None:
        """根据告警更新健康评分。"""
        if not alerts:
            self._health_score = min(1.0, self._health_score + 0.05)
        else:
            # 最严重的告警决定健康评分下降幅度
            max_severity = max(a.severity for a in alerts)
            self._health_score = max(0.0, 1.0 - max_severity)

    def get_health_score(self) -> float:
        """获取当前健康评分（0~1）。"""
        return self._health_score

    def get_alerts(self) -> List[AnomalyAlert]:
        """获取当前告警列表。"""
        return list(self._alerts)

    def get_trends(self) -> Dict[str, str]:
        """获取所有指标趋势。"""
        return {m: t.value for m, t in self._trends.items()}

    def get_state(self) -> Dict[str, Any]:
        return {
            "health_score": self._health_score,
            "total_anomalies": self.total_anomalies,
            "current_alerts": [a.to_dict() for a in self._alerts],
            "trends": {m: t.value for m, t in self._trends.items()},
            "metrics_tracked": list(self._metrics_history.keys()),
        }

    def reset(self) -> None:
        """重置监控器。"""
        for queue in self._metrics_history.values():
            queue.clear()
        self._alerts.clear()
        self._health_score = 1.0
        self._trends = {m: TrendDirection.STABLE for m in self._metrics_history}
        self.total_anomalies = 0
        self.log.info("SelfMonitor reset")
