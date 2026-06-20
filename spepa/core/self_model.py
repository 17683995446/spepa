"""SPEPA 自我模型（元认知自我指涉）。

维护系统对"自身"的描述：能力边界、当前状态、历史表现。
支持自我指涉：模型可以描述并修改自身。
"""
import numpy as np
from typing import Dict, Any, List

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class SelfModel(LoggedMixin):
    """自我模型：系统对自身的内部表征。"""

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.self_model")
        self.config = config
        self.dim = config.state_dim
        # 自我描述向量：能力、信心、资源使用等
        self.self_representation = np.ones(self.dim) * 0.5
        # 能力边界估计
        self.capability_bounds = {
            "prediction_accuracy": 0.5,
            "action_effectiveness": 0.5,
            "learning_rate": config.learning_rate_micro,
            "resource_usage": 0.0,
        }
        # 历史表现
        self.performance_history: List[float] = []
        self.self_modifications: List[Dict[str, Any]] = []
        self.log.info("SelfModel initialized (dim=%d)", self.dim)

    def update_self_representation(self, state: np.ndarray, performance: float) -> None:
        """根据当前状态与表现更新自我表征。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        if state.shape[0] != self.dim:
            state = np.resize(state, self.dim)
        # 平滑更新
        self.self_representation = 0.9 * self.self_representation + 0.1 * state
        self.performance_history.append(float(performance))
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        # 更新能力边界
        if self.performance_history:
            recent = np.mean(self.performance_history[-20:])
            self.capability_bounds["prediction_accuracy"] = (
                0.9 * self.capability_bounds["prediction_accuracy"] + 0.1 * recent
            )
        self.log.debug("Self-representation updated: perf=%.4f acc=%.4f",
                       performance, self.capability_bounds["prediction_accuracy"])

    def estimate_capability(self, task_type: str = "prediction") -> float:
        """估计自身在某类任务上的能力。"""
        if task_type == "prediction":
            cap = self.capability_bounds["prediction_accuracy"]
        elif task_type == "action":
            cap = self.capability_bounds["action_effectiveness"]
        else:
            cap = 0.5
        self.log.debug("Capability estimate '%s'=%.4f", task_type, cap)
        return float(cap)

    def self_referential_check(self) -> Dict[str, Any]:
        """自我指涉检查：模型对自身的认知是否一致。"""
        rep_norm = float(np.linalg.norm(self.self_representation))
        mean_perf = float(np.mean(self.performance_history[-20:])) if self.performance_history else 0.0
        # 一致性：自我表征与实际表现的吻合度
        consistency = 1.0 - abs(rep_norm / self.dim - mean_perf)
        consistency = float(np.clip(consistency, 0.0, 1.0))
        self.log.debug("Self-referential check: consistency=%.4f", consistency)
        return {
            "self_representation_norm": rep_norm,
            "mean_performance": mean_perf,
            "consistency": consistency,
            "capability_bounds": dict(self.capability_bounds),
        }

    def propose_self_modification(self, gap_severity: float) -> Dict[str, Any]:
        """根据缺口严重度提出自我修改建议。"""
        proposal: Dict[str, Any] = {}
        if gap_severity > self.config.structural_reconstruction_threshold:
            proposal["action"] = "increase_learning_rate"
            proposal["new_lr"] = min(self.config.learning_rate_micro * 2, 0.1)
        elif gap_severity > self.config.micro_correction_threshold:
            proposal["action"] = "adjust_precision"
            proposal["precision_factor"] = 1.2
        else:
            proposal["action"] = "none"
        proposal["gap_severity"] = float(gap_severity)
        self.self_modifications.append(proposal)
        self.log.info("Self-modification proposed: %s", proposal["action"])
        return proposal

    def apply_modification(self, proposal: Dict[str, Any]) -> bool:
        """应用自我修改。"""
        action = proposal.get("action", "none")
        if action == "increase_learning_rate":
            self.capability_bounds["learning_rate"] = proposal.get("new_lr",
                                                                   self.capability_bounds["learning_rate"])
            self.log.info("Applied: learning_rate -> %.4f",
                          self.capability_bounds["learning_rate"])
            return True
        elif action == "adjust_precision":
            self.log.info("Applied: precision adjustment (factor=%.2f)",
                          proposal.get("precision_factor", 1.0))
            return True
        return False

    def get_state(self) -> Dict[str, Any]:
        return {
            "self_representation": self.self_representation.tolist(),
            "capability_bounds": dict(self.capability_bounds),
            "performance_history_size": len(self.performance_history),
            "self_modifications_count": len(self.self_modifications),
        }
