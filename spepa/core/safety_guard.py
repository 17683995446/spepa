"""SPEPA 安全护栏（刚性边界约束）。

定义不可逾越的刚性边界，违反时强制中止/回退。
"""
import numpy as np
from typing import Dict, Any, List, Callable

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class SafetyGuard(LoggedMixin):
    """安全护栏。"""

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.safety_guard")
        self.config = config
        self.hard_boundary = config.safety_hard_boundary
        self.state_lower = np.zeros(config.state_dim)
        self.state_upper = np.ones(config.state_dim)
        self.action_limit = 1.0  # 单个动作分量最大幅值
        self.violations: List[Dict[str, Any]] = []
        self.custom_constraints: List[Callable[[np.ndarray, np.ndarray], bool]] = []
        self.log.info("SafetyGuard initialized (hard_boundary=%.2f)", self.hard_boundary)

    def add_constraint(self, fn: Callable[[np.ndarray, np.ndarray], bool]) -> None:
        """添加自定义约束函数 fn(state, action) -> 是否安全。"""
        self.custom_constraints.append(fn)
        self.log.info("Custom constraint added (total=%d)", len(self.custom_constraints))

    def check_state(self, state: np.ndarray) -> bool:
        """检查状态是否在安全边界内。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        in_bounds = bool(np.all(state >= self.state_lower - 1e-6) and
                         np.all(state <= self.state_upper + 1e-6))
        if not in_bounds:
            self._record_violation("state_out_of_bounds", state=state.tolist())
            self.log.warning("State out of bounds: %s", state)
        return in_bounds

    def check_action(self, action: np.ndarray) -> bool:
        """检查动作是否在幅值限制内。"""
        action = np.asarray(action, dtype=np.float64).flatten()
        within = bool(np.all(np.abs(action) <= self.action_limit + 1e-6))
        if not within:
            self._record_violation("action_exceeds_limit", action=action.tolist())
            self.log.warning("Action exceeds limit: %s", action)
        return within

    def check_homeostasis(self, deviation: float) -> bool:
        """检查内稳态偏离是否超过硬边界。"""
        safe = deviation < self.hard_boundary
        if not safe:
            self._record_violation("homeostasis_breach", deviation=deviation)
            self.log.error("Homeostasis hard boundary breached: deviation=%.4f", deviation)
        return safe

    def validate(self, state: np.ndarray, action: np.ndarray) -> Dict[str, Any]:
        """综合校验状态与动作。"""
        state_ok = self.check_state(state)
        action_ok = self.check_action(action)
        custom_ok = True
        for fn in self.custom_constraints:
            try:
                if not fn(state, action):
                    custom_ok = False
                    self._record_violation("custom_constraint",
                                           state=state.tolist(), action=action.tolist())
                    break
            except Exception as e:
                self.log.error("Custom constraint error: %s", e)
        all_ok = state_ok and action_ok and custom_ok
        self.log.debug("Validate: state_ok=%s action_ok=%s custom_ok=%s -> %s",
                       state_ok, action_ok, custom_ok, all_ok)
        return {
            "safe": all_ok,
            "state_ok": state_ok,
            "action_ok": action_ok,
            "custom_ok": custom_ok,
        }

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """将动作裁剪到安全范围内。"""
        action = np.asarray(action, dtype=np.float64).flatten()
        clipped = np.clip(action, -self.action_limit, self.action_limit)
        if not np.allclose(action, clipped):
            self.log.info("Action clipped: norm %.4f -> %.4f",
                          float(np.linalg.norm(action)), float(np.linalg.norm(clipped)))
        return clipped

    def emergency_stop(self, state: np.ndarray) -> bool:
        """是否需要紧急停止。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        # 状态严重越界
        if np.any(state < -0.5) or np.any(state > 1.5):
            self._record_violation("emergency_stop_state", state=state.tolist())
            self.log.error("EMERGENCY STOP: state severely out of range")
            return True
        return False

    def _record_violation(self, reason: str, **info) -> None:
        self.violations.append({"reason": reason, **info})

    def get_state(self) -> Dict[str, Any]:
        return {
            "violation_count": len(self.violations),
            "last_violations": self.violations[-5:],
            "n_custom_constraints": len(self.custom_constraints),
        }
