"""SPEPA 三大驱力引擎（安全/收益/能效），含内稳态调节。

驱力：
- Reward_safe: 安全驱力（维持内稳态边界）
- Reward_gain: 收益驱力（追求目标收益）
- Reward_eff:  能效驱力（资源使用效率）

设计原则（第一性原理）：
- 纯函数原则：reward 计算只读取状态，不修改状态
- 显式性原则：内稳态变更必须通过显式调用 update_homeostasis
- 因果性原则：内稳态由"动作对环境的影响"驱动，而非由 reward 计算驱动
"""
import numpy as np
from typing import Dict, Any, Tuple

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class Homeostasis:
    """内稳态调节器：维护一组生理变量的目标区间。

    状态变更方法：
    - update(delta): 应用扰动并更新 values（显式状态变更）
    - simulate(delta): 模拟更新但不修改实际状态（用于 reward 计算）
    - reset(): 重置到目标值
    """

    def __init__(self, dim: int, target: float = 0.5, bounds: Tuple[float, float] = (0.0, 1.0)):
        self.dim = dim
        self.target = np.ones(dim) * target
        self.lower = np.ones(dim) * bounds[0]
        self.upper = np.ones(dim) * bounds[1]
        self.values = np.ones(dim) * target

    def update(self, delta: np.ndarray) -> np.ndarray:
        """显式更新内稳态变量（状态变更入口）。

        根据外部扰动更新 values，返回更新后的偏离量。
        此方法是唯一的状态变更入口，reward 计算不应调用此方法。
        """
        delta = np.asarray(delta, dtype=np.float64).flatten()
        if delta.shape[0] != self.dim:
            delta = np.resize(delta, self.dim)
        # 清洗 NaN/Inf
        if not np.all(np.isfinite(delta)):
            delta = np.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=-1.0)
        self.values = np.clip(self.values + delta, self.lower, self.upper)
        deviation = self.values - self.target
        return deviation

    def simulate(self, delta: np.ndarray) -> np.ndarray:
        """模拟更新但不修改实际状态（纯函数，用于 reward 计算）。

        返回假设应用 delta 后的偏离量，但不改变 self.values。
        """
        delta = np.asarray(delta, dtype=np.float64).flatten()
        if delta.shape[0] != self.dim:
            delta = np.resize(delta, self.dim)
        # 清洗 NaN/Inf
        if not np.all(np.isfinite(delta)):
            delta = np.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=-1.0)
        simulated_values = np.clip(self.values + delta, self.lower, self.upper)
        deviation = simulated_values - self.target
        return deviation

    def deviation(self) -> float:
        """当前偏离目标的总量。"""
        return float(np.sum((self.values - self.target) ** 2))

    def in_bounds(self) -> bool:
        """是否所有变量都在安全边界内。"""
        return bool(np.all(self.values >= self.lower) and np.all(self.values <= self.upper))

    def reset(self) -> None:
        """重置内稳态到目标值。"""
        self.values = self.target.copy()

    def get_state(self) -> Dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "target": self.target.tolist(),
            "deviation": self.deviation(),
            "in_bounds": self.in_bounds(),
        }


class DriveEngine(LoggedMixin):
    """三大驱力引擎。

    reward 计算方法均为纯函数（只读取状态，不修改状态）。
    内稳态更新通过 update_homeostasis 显式调用。
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.drive_engine")
        self.config = config
        self.dim = config.state_dim
        self.homeostasis = Homeostasis(self.dim)
        self.weights = config.drives
        self.last_rewards: Dict[str, float] = {"safe": 0.0, "gain": 0.0, "eff": 0.0}
        self.log.info("DriveEngine initialized (safe=%.2f gain=%.2f eff=%.2f)",
                      self.weights.safety, self.weights.gain, self.weights.efficiency)

    def update_homeostasis(self, action: np.ndarray) -> np.ndarray:
        """显式更新内稳态（基于动作对环境的影响）。

        这是内稳态状态变更的唯一入口，应在 online_cycle 中显式调用。
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape[0] != self.dim:
            action = np.resize(action, self.dim)
        # 清洗 NaN/Inf
        if not np.all(np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        deviation = self.homeostasis.update(action * 0.1)
        self.log.debug("Homeostasis updated: deviation=%.4f in_bounds=%s",
                       self.homeostasis.deviation(), self.homeostasis.in_bounds())
        return deviation

    def reward_safe(self, state: np.ndarray, action: np.ndarray) -> float:
        """安全驱力：偏离内稳态越小，奖励越高。

        纯函数：只读取 homeostasis 状态，不修改。
        使用 simulate 模拟动作对内稳态的影响，用于评估动作的安全度。
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape[0] != self.dim:
            action = np.resize(action, self.dim)
        # 清洗 NaN/Inf
        if not np.all(np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        # 模拟动作对内稳态的影响（不修改实际状态）
        simulated_deviation = self.homeostasis.simulate(action * 0.1)
        safe_reward = -float(np.sum(simulated_deviation ** 2))
        # 边界检查（基于模拟后的状态）
        simulated_in_bounds = bool(
            np.all(self.homeostasis.values + action * 0.1 >= self.homeostasis.lower - 1e-6) and
            np.all(self.homeostasis.values + action * 0.1 <= self.homeostasis.upper + 1e-6)
        )
        if not simulated_in_bounds:
            safe_reward -= 10.0  # 越界惩罚
        self.last_rewards["safe"] = safe_reward
        self.log.debug("Reward_safe=%.4f (simulated, no state change)", safe_reward)
        return safe_reward

    def reward_gain(self, state: np.ndarray, action: np.ndarray, goal: np.ndarray) -> float:
        """收益驱力：朝目标推进的收益。纯函数。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        goal = np.asarray(goal, dtype=np.float64).flatten()
        action = np.asarray(action, dtype=np.float64).flatten()
        # 动作维度对齐到状态维度
        if action.shape[0] != state.shape[0]:
            action = np.resize(action, state.shape[0])
        # 清洗 NaN/Inf
        if not np.all(np.isfinite(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        if not np.all(np.isfinite(goal)):
            goal = np.nan_to_num(goal, nan=0.0, posinf=1.0, neginf=0.0)
        if not np.all(np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        # 收益 = 状态朝目标移动的量
        progress = float(np.dot(action, (goal - state)))
        gain_reward = progress
        self.last_rewards["gain"] = gain_reward
        self.log.debug("Reward_gain=%.4f", gain_reward)
        return gain_reward

    def reward_eff(self, action: np.ndarray, achieved: float) -> float:
        """能效驱力：单位资源消耗取得的成效。纯函数。"""
        action = np.asarray(action, dtype=np.float64).flatten()
        # 清洗 NaN/Inf
        if not np.all(np.isfinite(action)):
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        cost = float(np.sum(action ** 2)) + 1e-6
        eff_reward = float(achieved) / cost
        self.last_rewards["eff"] = eff_reward
        self.log.debug("Reward_eff=%.4f", eff_reward)
        return eff_reward

    def total_reward(self, state: np.ndarray, action: np.ndarray,
                     goal: np.ndarray, achieved: float) -> float:
        """加权总驱力。纯函数（不修改任何状态）。"""
        rs = self.reward_safe(state, action)
        rg = self.reward_gain(state, action, goal)
        re = self.reward_eff(action, achieved)
        total = (self.weights.safety * rs +
                 self.weights.gain * rg +
                 self.weights.efficiency * re)
        self.log.debug("Total reward=%.4f (safe=%.4f gain=%.4f eff=%.4f)",
                       total, rs, rg, re)
        return float(total)

    def homeostatic_drive(self) -> float:
        """返回当前内稳态驱力（偏离越大驱力越强）。纯函数。"""
        d = self.homeostasis.deviation()
        self.log.debug("Homeostatic drive=%.4f", d)
        return d

    def reset_homeostasis(self) -> None:
        """重置内稳态到目标值。"""
        self.homeostasis.reset()
        self.log.info("Homeostasis reset to target values")

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_rewards": dict(self.last_rewards),
            "homeostasis_deviation": self.homeostasis.deviation(),
            "in_bounds": self.homeostasis.in_bounds(),
            "homeostasis_values": self.homeostasis.values.tolist(),
        }
