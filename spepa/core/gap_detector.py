"""SPEPA 缺口检测器（信息缺口 = KL 散度），四级分类。

缺口分级：
- L0: 无缺口（KL < 0.2）
- L1: 轻微缺口（0.2 <= KL < 0.5）
- L2: 中度缺口（0.5 <= KL < 1.0）
- L3: 严重缺口（KL >= 1.0）
"""
import numpy as np
from typing import Dict, Any, Tuple

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.utils.metrics import kl_divergence, gaussian_kl


GAP_LEVELS = ["L0_none", "L1_minor", "L2_moderate", "L3_severe"]


class GapInfo:
    """缺口信息载体。"""

    def __init__(self, level: str, level_idx: int, kl_value: float,
                 source: str = "", details: Dict[str, Any] = None):
        self.level = level
        self.level_idx = level_idx
        self.kl_value = kl_value
        self.source = source
        self.details = details or {}

    def is_actionable(self) -> bool:
        """是否需要采取行动（L2 及以上）。"""
        return self.level_idx >= 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "level_idx": self.level_idx,
            "kl_value": self.kl_value,
            "source": self.source,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"GapInfo(level={self.level}, kl={self.kl_value:.4f}, source={self.source})"


class GapDetector(LoggedMixin):
    """缺口检测器：基于 KL 散度检测预测与观测之间的信息缺口。"""

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.gap_detector")
        self.config = config
        self.thresholds = config.gap_level_thresholds
        self.last_gap: GapInfo = None
        self.log.info("GapDetector initialized thresholds=%s", self.thresholds)

    def classify(self, kl_value: float) -> Tuple[str, int]:
        """根据 KL 值分级（使用全部4个阈值）。"""
        if kl_value < self.thresholds[0]:
            # L0: kl < 0.2
            return GAP_LEVELS[0], 0
        elif kl_value < self.thresholds[1]:
            # L1: 0.2 <= kl < 0.5
            return GAP_LEVELS[1], 1
        elif kl_value < self.thresholds[2]:
            # L2: 0.5 <= kl < 1.0
            return GAP_LEVELS[2], 2
        elif kl_value < self.thresholds[3]:
            # L3: 1.0 <= kl < 2.0
            return GAP_LEVELS[3], 3
        else:
            # L4: kl >= 2.0（极端严重，使用 L3 标签但标记为更高级别）
            return GAP_LEVELS[3], 4

    def detect_distribution_gap(self, predicted: np.ndarray, actual: np.ndarray,
                                source: str = "perception") -> GapInfo:
        """检测两个分布之间的缺口（KL 散度）。"""
        kl = kl_divergence(predicted, actual)
        level, idx = self.classify(kl)
        gap = GapInfo(level, idx, kl, source=source,
                      details={"predicted_norm": float(np.linalg.norm(predicted)),
                               "actual_norm": float(np.linalg.norm(actual))})
        self.last_gap = gap
        self.log.info("Gap detected source='%s' kl=%.4f level=%s", source, kl, level)
        return gap

    def detect_gaussian_gap(self, pred_mu, pred_var, act_mu, act_var,
                            source: str = "model") -> GapInfo:
        """检测两个高斯分布之间的缺口。"""
        kl = gaussian_kl(pred_mu, act_mu, pred_var, act_var)
        level, idx = self.classify(kl)
        gap = GapInfo(level, idx, kl, source=source,
                      details={"pred_mu_norm": float(np.linalg.norm(pred_mu)),
                               "act_mu_norm": float(np.linalg.norm(act_mu))})
        self.last_gap = gap
        self.log.info("Gaussian gap detected source='%s' kl=%.4f level=%s", source, kl, level)
        return gap

    def detect_prediction_gap(self, prediction_error: float, source: str = "prediction") -> GapInfo:
        """基于预测误差直接检测缺口（误差即缺口）。"""
        kl = prediction_error  # 预测误差作为缺口度量
        level, idx = self.classify(kl)
        gap = GapInfo(level, idx, kl, source=source, details={"raw_error": prediction_error})
        self.last_gap = gap
        self.log.info("Prediction gap detected source='%s' err=%.4f level=%s",
                      source, prediction_error, level)
        return gap

    def should_explore(self, gap: GapInfo) -> bool:
        """根据缺口决定是否需要探索。L2+ 触发探索。"""
        need = gap.is_actionable()
        self.log.debug("Should explore? gap=%s -> %s", gap.level, need)
        return need

    def get_state(self) -> Dict[str, Any]:
        if self.last_gap is None:
            return {"last_gap": None}
        return {"last_gap": self.last_gap.to_dict()}
