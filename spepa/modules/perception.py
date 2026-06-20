"""SPEPA 感知模块。

接收原始观测，通过预测模型生成预测，检测缺口，编码到记忆。
"""
import numpy as np
from typing import Dict, Any

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.core.predictive_model import PredictiveModel
from spepa.core.gap_detector import GapDetector, GapInfo
from spepa.core.memory_system import MemorySystem
from spepa.core.state_bus import StateBus


class Perception(LoggedMixin):
    """感知模块。"""

    def __init__(self, config: SPEPAConfig, predictive_model: PredictiveModel,
                 gap_detector: GapDetector, memory: MemorySystem,
                 state_bus: StateBus):
        self._set_log_name("spepa.modules.perception")
        self.config = config
        self.model = predictive_model
        self.gap_detector = gap_detector
        self.memory = memory
        self.bus = state_bus
        self.last_observation: np.ndarray = None
        self.last_prediction: np.ndarray = None
        self.last_gap: GapInfo = None
        self.log.info("Perception module initialized")

    def perceive(self, observation: np.ndarray) -> Dict[str, Any]:
        """感知流程：预测 -> 比对 -> 缺口检测 -> 记忆编码。"""
        obs = np.asarray(observation, dtype=np.float64).flatten()
        if obs.shape[0] != self.config.obs_dim:
            obs = np.resize(obs, self.config.obs_dim)
        # NaN/Inf 清洗，避免污染后续计算
        if not np.all(np.isfinite(obs)):
            bad = ~np.isfinite(obs)
            obs[bad] = 0.0
            self.log.warning("Observation contained non-finite values, sanitized")
        self.last_observation = obs

        # 1. 生成预测（模型内部会 resize 到 state_dim）
        prediction = self.model.predict(obs)
        self.last_prediction = prediction

        # 2. 计算预测误差（将 prediction 对齐到 obs_dim）
        pred_aligned = np.resize(prediction, obs.shape[0])
        error = float(np.linalg.norm(pred_aligned - obs))

        # 3. 缺口检测
        gap = self.gap_detector.detect_prediction_gap(error, source="perception")
        self.last_gap = gap

        # 4. 记忆编码（重要性 = 误差，误差越大越重要）
        importance = float(np.tanh(error))
        self.memory.encode(obs, value={"prediction": prediction.tolist(),
                                       "error": error,
                                       "gap_level": gap.level},
                           importance=importance)

        # 5. 发布到状态总线
        self.bus.set_state("last_observation", obs.tolist())
        self.bus.set_state("last_prediction", prediction.tolist())
        self.bus.set_state("last_error", error)
        self.bus.set_state("last_gap", gap.to_dict())
        self.bus.publish("perception", {"observation": obs.tolist(),
                                        "prediction": prediction.tolist(),
                                        "error": error,
                                        "gap": gap.to_dict()})

        self.log.info("Perceived: error=%.4f gap=%s importance=%.4f",
                      error, gap.level, importance)
        return {
            "observation": obs.tolist(),
            "prediction": prediction.tolist(),
            "error": error,
            "gap": gap.to_dict(),
            "importance": importance,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_error": float(np.linalg.norm(self.last_prediction - self.last_observation))
            if self.last_observation is not None and self.last_prediction is not None else 0.0,
            "last_gap": self.last_gap.to_dict() if self.last_gap else None,
        }
