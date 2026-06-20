"""SPEPA 决策模块（感知-评估-生成-决策 四阶段）。

四阶段：
1. 感知阶段：整合感知结果与目标
2. 评估阶段：评估当前态势（自由能、缺口、驱力）
3. 生成阶段：生成候选策略
4. 决策阶段：选择最优策略
"""
import numpy as np
from typing import Dict, Any, Callable, Optional

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.core.active_inference import ActiveInference
from spepa.core.free_energy import FreeEnergyCalculator
from spepa.core.gap_detector import GapInfo
from spepa.core.goal_hierarchy import GoalHierarchy
from spepa.core.drive_engine import DriveEngine
from spepa.core.metacognition import Metacognition
from spepa.core.safety_guard import SafetyGuard
from spepa.core.state_bus import StateBus


class Decision(LoggedMixin):
    """决策模块：四阶段决策。"""

    def __init__(self, config: SPEPAConfig, active_inference: ActiveInference,
                 fe_calculator: FreeEnergyCalculator, goals: GoalHierarchy,
                 drives: DriveEngine, metacog: Metacognition,
                 safety: SafetyGuard, state_bus: StateBus):
        self._set_log_name("spepa.modules.decision")
        self.config = config
        self.ai = active_inference
        self.fe = fe_calculator
        self.goals = goals
        self.drives = drives
        self.metacog = metacog
        self.safety = safety
        self.bus = state_bus
        self.exploration_ratio = config.exploration.initial_exploration_ratio
        self.last_decision: Dict[str, Any] = None
        self.log.info("Decision module initialized (exploration=%.2f)",
                      self.exploration_ratio)

    def _phase_perceive(self, state: np.ndarray, gap: GapInfo) -> Dict[str, Any]:
        """阶段1：感知 - 整合状态与目标。"""
        arbitration = self.goals.arbitrate(state)
        self.log.debug("Phase1 perceive: dominant=%s", arbitration["dominant_goal"])
        return arbitration

    def _phase_evaluate(self, state: np.ndarray, gap: GapInfo,
                        arbitration: Dict[str, Any]) -> Dict[str, Any]:
        """阶段2：评估 - 计算自由能、驱力、元认知。"""
        # 用预测误差作为自由能的近似
        fe_value = gap.kl_value
        homeostatic = self.drives.homeostatic_drive()
        meta = self.metacog.meta_control(
            prediction_error=gap.kl_value,
            free_energy=fe_value,
            exploration_ratio=self.exploration_ratio,
        )
        # 更新探索比例（元认知控制 + 衰减）
        self.exploration_ratio = meta["new_exploration_ratio"]
        # 应用探索比例衰减（每步衰减）
        self.exploration_ratio *= self.config.exploration.decay
        # 确保不低于下限
        self.exploration_ratio = max(self.exploration_ratio,
                                     self.config.exploration.exploration_ratio_min)
        self.log.debug("Phase2 evaluate: fe=%.4f homeo=%.4f explore=%.4f",
                       fe_value, homeostatic, self.exploration_ratio)
        return {
            "free_energy": fe_value,
            "homeostatic_drive": homeostatic,
            "metacognition": meta,
            "exploration_ratio": self.exploration_ratio,
        }

    def _phase_generate(self, state: np.ndarray, preferred: np.ndarray,
                        model_predict_fn: Callable,
                        n_policies: int = 8) -> Dict[str, Any]:
        """阶段3：生成 - 生成候选策略并评估。"""
        decision = self.ai.decide(
            current_state=state,
            model_predict_fn=model_predict_fn,
            preferred_state=preferred,
            exploration_ratio=self.exploration_ratio,
            n_policies=n_policies,
        )
        self.log.debug("Phase3 generate: policy=%s G=%.4f",
                       decision["policy_name"], decision["expected_free_energy"])
        return decision

    def _phase_decide(self, state: np.ndarray, decision: Dict[str, Any]) -> Dict[str, Any]:
        """阶段4：决策 - 安全校验并最终确定动作。"""
        action = np.asarray(decision["action"], dtype=np.float64).flatten()
        # 安全校验
        validation = self.safety.validate(state, action)
        if not validation["safe"]:
            action = self.safety.clip_action(action)
            self.log.warning("Action unsafe, clipped to norm=%.4f",
                             float(np.linalg.norm(action)))
        # 紧急停止检查
        emergency = self.safety.emergency_stop(state)
        if emergency:
            action = np.zeros_like(action)
            self.log.error("Emergency stop triggered, action zeroed")
        # 安全硬边界检查：内稳态偏离超过硬边界时强制保守
        homeostatic_dev = self.drives.homeostatic_drive()
        if not self.safety.check_homeostasis(homeostatic_dev):
            # 硬边界违反：减小动作幅值（保守化）
            action = action * 0.5
            self.log.warning("Hard boundary breached (homeostasis=%.4f), action halved",
                             homeostatic_dev)
        decision["action"] = action.tolist()
        decision["safe"] = validation["safe"]
        decision["emergency_stop"] = emergency
        self.log.info("Phase4 decide: action_norm=%.4f safe=%s",
                      float(np.linalg.norm(action)), validation["safe"])
        return decision

    def decide(self, state: np.ndarray, gap: GapInfo,
               model_predict_fn: Callable,
               n_policies: int = 8) -> Dict[str, Any]:
        """完整四阶段决策。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        # 阶段1
        arbitration = self._phase_perceive(state, gap)
        preferred = np.asarray(arbitration.get("dominant_target") or state, dtype=np.float64).flatten()
        # 阶段2
        evaluation = self._phase_evaluate(state, gap, arbitration)
        # 阶段3
        generation = self._phase_generate(state, preferred, model_predict_fn,
                                          n_policies=n_policies)
        # 阶段4
        final = self._phase_decide(state, generation)

        result = {
            "arbitration": arbitration,
            "evaluation": evaluation,
            "decision": final,
            "action": final["action"],
        }
        self.last_decision = result
        self.bus.set_state("last_action", final["action"])
        self.bus.set_state("exploration_ratio", self.exploration_ratio)
        self.bus.publish("decision", result)
        self.log.info("Decision complete: action_norm=%.4f explore=%.4f",
                      float(np.linalg.norm(final["action"])), self.exploration_ratio)
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "exploration_ratio": self.exploration_ratio,
            "last_decision_exists": self.last_decision is not None,
        }
