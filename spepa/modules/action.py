"""SPEPA 行动模块。

执行决策产生的动作，作用于环境，收集反馈。
"""
import numpy as np
from typing import Dict, Any, Callable, Optional

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.core.safety_guard import SafetyGuard
from spepa.core.state_bus import StateBus


class Action(LoggedMixin):
    """行动模块。"""

    def __init__(self, config: SPEPAConfig, safety: SafetyGuard, state_bus: StateBus):
        self._set_log_name("spepa.modules.action")
        self.config = config
        self.safety = safety
        self.bus = state_bus
        self.last_action: np.ndarray = None
        self.last_effect: np.ndarray = None
        self.action_count = 0
        self.log.info("Action module initialized")

    def execute(self, action: np.ndarray,
                env_step_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Dict[str, Any]:
        """执行动作。

        env_step_fn(action) -> next_state（若不提供则仅记录动作）。
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape[0] != self.config.action_dim:
            action = np.resize(action, self.config.action_dim)
        # 再次安全裁剪
        action = self.safety.clip_action(action)
        self.last_action = action
        self.action_count += 1

        effect = None
        if env_step_fn is not None:
            try:
                raw_effect = env_step_fn(action)
                # 环境可能返回 None（模拟断连）
                if raw_effect is None:
                    self.log.warning("Environment returned None, treating as no-op")
                else:
                    effect = np.asarray(raw_effect, dtype=np.float64).flatten()
                    # NaN/Inf 清洗
                    if not np.all(np.isfinite(effect)):
                        bad = ~np.isfinite(effect)
                        effect[bad] = 0.0
                        self.log.warning("Environment effect contained non-finite values, sanitized")
                    self.last_effect = effect
            except Exception as e:
                self.log.error("Environment step error: %s", e)
                effect = None

        self.bus.set_state("last_executed_action", action.tolist())
        self.bus.publish("action", {"action": action.tolist(),
                                    "effect": effect.tolist() if effect is not None else None})
        self.log.info("Action executed #%d: norm=%.4f effect=%s",
                      self.action_count, float(np.linalg.norm(action)),
                      "yes" if effect is not None else "no")
        return {
            "action": action.tolist(),
            "effect": effect.tolist() if effect is not None else None,
            "action_count": self.action_count,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "action_count": self.action_count,
            "last_action_norm": float(np.linalg.norm(self.last_action))
            if self.last_action is not None else 0.0,
        }
