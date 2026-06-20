"""SPEPA 学习模块（Hebb-like 微修正 + 结构性重构）。

根据预测误差驱动模型更新：
- 误差小：Hebb-like 微修正
- 误差大：结构性重构
- 同时更新元认知校准与自我模型
"""
import numpy as np
from typing import Dict, Any

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.core.predictive_model import PredictiveModel
from spepa.core.optimizer import Optimizer
from spepa.core.metacognition import Metacognition
from spepa.core.self_model import SelfModel
from spepa.core.state_bus import StateBus


class Learning(LoggedMixin):
    """学习模块。"""

    def __init__(self, config: SPEPAConfig, model: PredictiveModel,
                 optimizer: Optimizer, metacog: Metacognition,
                 self_model: SelfModel, state_bus: StateBus):
        self._set_log_name("spepa.modules.learning")
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.metacog = metacog
        self.self_model = self_model
        self.bus = state_bus
        self.learning_count = 0
        self.log.info("Learning module initialized")

    def learn(self, observation: np.ndarray, prediction_error: float,
              confidence: float) -> Dict[str, Any]:
        """学习流程：优化模型 -> 校准信心 -> 更新自我模型。"""
        # 1. 双模式优化
        opt_result = self.optimizer.optimize(self.model, observation, prediction_error)

        # 2. 重新计算误差
        new_error = self.model.total_prediction_error(observation)

        # 3. 元认知校准
        self.metacog.record_outcome(confidence, new_error)

        # 4. 自我模型更新
        self.self_model.update_self_representation(observation, 1.0 - float(np.tanh(new_error)))

        # 5. 自我修改建议
        modification = self.self_model.propose_self_modification(prediction_error)
        if modification["action"] != "none":
            self.self_model.apply_modification(modification)

        self.learning_count += 1
        result = {
            "optimization": opt_result,
            "pre_error": prediction_error,
            "post_error": new_error,
            "modification": modification,
            "learning_count": self.learning_count,
        }
        self.bus.set_state("last_learning", result)
        self.bus.publish("learning", result)
        self.log.info("Learning #%d: mode=%s err %.4f -> %.4f",
                      self.learning_count, opt_result["mode"],
                      prediction_error, new_error)
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "learning_count": self.learning_count,
            "optimizer_state": self.optimizer.get_state(),
        }
