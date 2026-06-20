"""SPEPA 元认知监控器（自我模型，信心校准）。

监控：
- 系统对自身预测的信心
- 信心校准（预测信心 vs 实际准确度）
- 元认知控制（何时切换策略、何时探索）
"""
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class ConfidenceCalibrator:
    """信心校准器：跟踪预测信心与实际误差的关系。"""

    def __init__(self, window: int = 50):
        self.window = window
        self.confidences = deque(maxlen=window)
        self.accuracies = deque(maxlen=window)
        self.calibration_offset = 0.0

    def record(self, confidence: float, accuracy: float) -> None:
        """记录一次预测的信心与准确度。"""
        self.confidences.append(float(confidence))
        # accuracy: 0~1，1 表示完全准确
        self.accuracies.append(float(accuracy))
        # 校准偏移 = 平均信心 - 平均准确度（>0 表示过度自信）
        if len(self.confidences) >= 5:
            self.calibration_offset = float(np.mean(self.confidences) - np.mean(self.accuracies))

    def calibrated_confidence(self, raw_confidence: float) -> float:
        """返回校准后的信心。"""
        return float(np.clip(raw_confidence - self.calibration_offset, 0.0, 1.0))

    def is_overconfident(self) -> bool:
        """是否过度自信。"""
        return self.calibration_offset > 0.1

    def summary(self) -> Dict[str, float]:
        if not self.confidences:
            return {"mean_confidence": 0.0, "mean_accuracy": 0.0, "offset": 0.0}
        return {
            "mean_confidence": float(np.mean(self.confidences)),
            "mean_accuracy": float(np.mean(self.accuracies)),
            "offset": self.calibration_offset,
        }


class Metacognition(LoggedMixin):
    """元认知监控器。"""

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.metacognition")
        self.config = config
        self.calibrator = ConfidenceCalibrator(window=config.confidence_calibration_window)
        self.current_confidence: float = 0.5
        self.meta_control_signal: float = 0.0  # >0 倾向探索，<0 倾向利用
        self.log.info("Metacognition initialized (window=%d)",
                      config.confidence_calibration_window)

    def compute_confidence(self, prediction_error: float, uncertainty: float) -> float:
        """根据预测误差和不确定性计算信心（0~1）。"""
        # 误差越小、不确定性越低 -> 信心越高
        err_term = np.exp(-prediction_error)
        unc_term = np.exp(-uncertainty)
        raw = 0.5 * (err_term + unc_term)
        self.current_confidence = self.calibrator.calibrated_confidence(raw)
        self.log.debug("Confidence: raw=%.4f calibrated=%.4f (err=%.4f unc=%.4f)",
                       raw, self.current_confidence, prediction_error, uncertainty)
        return self.current_confidence

    def record_outcome(self, confidence: float, actual_error: float) -> None:
        """记录预测结果以校准信心。"""
        accuracy = float(np.exp(-actual_error))
        self.calibrator.record(confidence, accuracy)
        self.log.debug("Outcome recorded: conf=%.4f accuracy=%.4f offset=%.4f",
                       confidence, accuracy, self.calibrator.calibration_offset)

    def meta_control(self, prediction_error: float, free_energy: float,
                     exploration_ratio: float) -> Dict[str, Any]:
        """元认知控制：决定是否调整探索比例。"""
        # 误差大、自由能高 -> 增加探索
        signal = 0.0
        if prediction_error > self.config.micro_correction_threshold:
            signal += 0.1
        if free_energy > self.config.free_energy_threshold:
            signal += 0.1
        if self.calibrator.is_overconfident():
            signal += 0.05  # 过度自信时也增加探索
        self.meta_control_signal = signal

        # 调整探索比例
        new_ratio = float(np.clip(exploration_ratio + signal, 0.0, 1.0))
        # 确保不低于下限
        new_ratio = max(new_ratio, self.config.exploration.exploration_ratio_min)
        self.log.info("Meta-control: signal=%.4f exploration %.4f -> %.4f",
                      signal, exploration_ratio, new_ratio)
        return {
            "control_signal": signal,
            "new_exploration_ratio": new_ratio,
            "confidence": self.current_confidence,
            "overconfident": self.calibrator.is_overconfident(),
        }

    def should_switch_strategy(self) -> bool:
        """是否应切换策略（信心过低或过度自信）。"""
        switch = (self.current_confidence < 0.3) or self.calibrator.is_overconfident()
        self.log.debug("Should switch strategy? conf=%.4f -> %s",
                       self.current_confidence, switch)
        return switch

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_confidence": self.current_confidence,
            "meta_control_signal": self.meta_control_signal,
            "calibration": self.calibrator.summary(),
        }
