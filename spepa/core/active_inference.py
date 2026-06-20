"""SPEPA 主动推理决策器（期望自由能最小化）。

基于主动推理框架：选择使期望自由能 G 最小化的策略。
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.core.free_energy import FreeEnergyCalculator


class Policy:
    """一个候选策略：动作序列。"""

    def __init__(self, actions: List[np.ndarray], name: str = "policy"):
        self.actions = [np.asarray(a, dtype=np.float64).flatten() for a in actions]
        self.name = name
        self.expected_free_energy: float = 0.0
        self.uncertainty: float = 0.0

    def __len__(self) -> int:
        return len(self.actions)

    def first_action(self) -> np.ndarray:
        return self.actions[0].copy() if self.actions else np.zeros(0)


class ActiveInference(LoggedMixin):
    """主动推理决策器。"""

    def __init__(self, config: SPEPAConfig, fe_calculator: FreeEnergyCalculator):
        self._set_log_name("spepa.core.active_inference")
        self.config = config
        self.fe = fe_calculator
        self.dim = config.action_dim
        self.state_dim = config.state_dim
        self.rng = np.random.default_rng(config.seed)
        self.log.info("ActiveInference initialized (action_dim=%d)", self.dim)

    def generate_policies(self, n_policies: int = 8, horizon: int = 1,
                          exploration_ratio: float = 0.3) -> List[Policy]:
        """生成候选策略：70% 利用（朝目标），30% 探索（随机）。"""
        # 确保 n_policies 至少为 1，避免空策略列表
        n_policies = max(1, int(n_policies))
        policies: List[Policy] = []
        n_explore = max(1, int(n_policies * exploration_ratio))
        n_exploit = max(0, n_policies - n_explore)
        # 利用：小幅扰动
        for i in range(n_exploit):
            actions = []
            for _ in range(horizon):
                a = self.rng.normal(0, 0.1, self.dim)
                actions.append(a)
            policies.append(Policy(actions, name=f"exploit_{i}"))
        # 探索：大幅随机
        for i in range(n_explore):
            actions = []
            for _ in range(horizon):
                a = self.rng.normal(0, 0.5, self.dim)
                actions.append(a)
            policies.append(Policy(actions, name=f"explore_{i}"))
        self.log.info("Generated %d policies (exploit=%d explore=%d)",
                      len(policies), n_exploit, n_explore)
        return policies

    def evaluate_policy(self, policy: Policy, current_state: np.ndarray,
                        predicted_next_state: np.ndarray, predicted_var: np.ndarray,
                        preferred_state: np.ndarray) -> float:
        """评估单个策略的期望自由能。"""
        g = self.fe.expected_free_energy(
            predicted_obs=predicted_next_state,
            predicted_var=predicted_var,
            preferred_obs=preferred_state,
        )
        # 加上动作成本（能效）
        action_cost = sum(float(np.sum(a ** 2)) for a in policy.actions) * 0.1
        # 加上不确定性（认知价值：降低不确定性是好事，所以减去）
        policy.uncertainty = float(np.mean(predicted_var))
        policy.expected_free_energy = g + action_cost - 0.05 * policy.uncertainty
        self.log.debug("Policy '%s' G=%.4f cost=%.4f unc=%.4f",
                       policy.name, g, action_cost, policy.uncertainty)
        return policy.expected_free_energy

    def select_policy(self, policies: List[Policy], current_state: np.ndarray,
                      model_predict_fn, preferred_state: np.ndarray,
                      exploration_ratio: float = 0.3) -> Tuple[Policy, List[Dict[str, Any]]]:
        """选择期望自由能最小的策略。

        model_predict_fn(state, action) -> (next_state, var)
        """
        # 空策略列表防御：生成至少一个默认策略
        if not policies:
            self.log.warning("Empty policies list, generating default policy")
            default_action = np.zeros(self.dim)
            policies = [Policy([default_action], name="default")]
        # 对齐 preferred_state 到 state_dim
        preferred_state = np.asarray(preferred_state, dtype=np.float64).flatten()
        if preferred_state.shape[0] != self.state_dim:
            preferred_state = np.resize(preferred_state, self.state_dim)
        # 对齐 current_state 到 state_dim
        current_state = np.asarray(current_state, dtype=np.float64).flatten()
        if current_state.shape[0] != self.state_dim:
            current_state = np.resize(current_state, self.state_dim)

        evals = []
        for p in policies:
            next_state, var = model_predict_fn(current_state, p.first_action())
            next_state = np.asarray(next_state, dtype=np.float64).flatten()
            var = np.asarray(var, dtype=np.float64).flatten()
            # 对齐 next_state 和 var 到 state_dim
            if next_state.shape[0] != self.state_dim:
                next_state = np.resize(next_state, self.state_dim)
            if var.shape[0] != self.state_dim:
                var = np.resize(var, self.state_dim)
            # 清洗 NaN/Inf
            if not np.all(np.isfinite(next_state)):
                next_state = np.nan_to_num(next_state, nan=0.0, posinf=1.0, neginf=0.0)
            if not np.all(np.isfinite(var)):
                var = np.nan_to_num(var, nan=0.1, posinf=1.0, neginf=0.01)
            self.evaluate_policy(p, current_state, next_state, var, preferred_state)
            evals.append({
                "name": p.name,
                "G": p.expected_free_energy,
                "uncertainty": p.uncertainty,
            })
        # softmax 选择（带温度）
        Gs = np.array([p.expected_free_energy for p in policies])
        # 数值稳定性：清洗 NaN/Inf
        if not np.all(np.isfinite(Gs)):
            self.log.warning("Policy G values contain non-finite, sanitizing")
            Gs = np.nan_to_num(Gs, nan=0.0, posinf=1e6, neginf=-1e6)
        temp = 0.5
        # 越小越好 -> 用 -G
        logits = -Gs / temp
        logits -= logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        # 再次确保概率有效
        if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
            self.log.warning("Policy probs invalid, using uniform distribution")
            probs = np.ones(len(policies)) / len(policies)
        idx = int(self.rng.choice(len(policies), p=probs))
        chosen = policies[idx]
        self.log.info("Policy selected: '%s' G=%.4f (probs=%s)",
                      chosen.name, chosen.expected_free_energy,
                      np.round(probs, 3).tolist())
        return chosen, evals

    def decide(self, current_state: np.ndarray, model_predict_fn,
               preferred_state: np.ndarray, exploration_ratio: float = 0.3,
               n_policies: int = 8) -> Dict[str, Any]:
        """完整决策流程：生成策略 -> 评估 -> 选择。"""
        policies = self.generate_policies(n_policies=n_policies, horizon=1,
                                          exploration_ratio=exploration_ratio)
        chosen, evals = self.select_policy(
            policies, current_state, model_predict_fn, preferred_state,
            exploration_ratio=exploration_ratio,
        )
        action = chosen.first_action()
        self.log.info("Decision made: action_norm=%.4f", float(np.linalg.norm(action)))
        return {
            "action": action.tolist(),
            "policy_name": chosen.name,
            "expected_free_energy": chosen.expected_free_energy,
            "evaluations": evals,
        }
