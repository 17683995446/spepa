"""SPEPA 自由能计算（变分自由能）。

实现三种形式：
1. 经典变分自由能 F = E_q[-log p(o,s)] + H[q(s)]
2. 简化形式 F = 复杂度 - 准确度
3. 预测编码形式 F = 精度加权预测误差 + KL(后验||先验)
"""
import numpy as np
from typing import Optional, Tuple

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.utils.metrics import gaussian_kl


class FreeEnergyCalculator(LoggedMixin):
    """变分自由能计算器。"""

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.free_energy")
        self.config = config
        self.dim = config.state_dim
        self.log.info("FreeEnergyCalculator initialized (dim=%d)", self.dim)

    def variational_free_energy(
        self,
        obs: np.ndarray,
        posterior_mu: np.ndarray,
        posterior_var: np.ndarray,
        prior_mu: Optional[np.ndarray] = None,
        prior_var: Optional[np.ndarray] = None,
    ) -> float:
        """形式1：经典变分自由能 F = E_q[-log p(o,s)] + H[q(s)]。

        近似为：F ≈ KL(q(s)||p(s|o)) - E_q[log p(o|s)] + const
        这里用高斯近似。
        """
        obs = np.asarray(obs, dtype=np.float64).flatten()
        post_mu = np.asarray(posterior_mu, dtype=np.float64).flatten()
        post_var = np.asarray(posterior_var, dtype=np.float64).flatten() + 1e-10
        if prior_mu is None:
            prior_mu = np.zeros_like(post_mu)
        if prior_var is None:
            prior_var = np.ones_like(post_mu)

        # KL(q||p) 复杂度项
        kl = gaussian_kl(post_mu, prior_mu, post_var, prior_var)
        # 准确度项：-E_q[log p(o|s)] ≈ 0.5 * ||o - mu||^2 / var_o
        accuracy = 0.5 * float(np.sum((obs - post_mu) ** 2 / post_var))
        fe = kl + accuracy
        self.log.debug("Variational FE: kl=%.4f acc=%.4f F=%.4f", kl, accuracy, fe)
        return float(fe)

    def complexity_accuracy_form(
        self,
        obs: np.ndarray,
        posterior_mu: np.ndarray,
        posterior_var: np.ndarray,
        prior_mu: Optional[np.ndarray] = None,
        prior_var: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """形式2：F = 复杂度 - 准确度。返回 (复杂度, 准确度, F)。"""
        if prior_mu is None:
            prior_mu = np.zeros_like(posterior_mu)
        if prior_var is None:
            prior_var = np.ones_like(posterior_var)
        complexity = gaussian_kl(posterior_mu, prior_mu, posterior_var, prior_var)
        obs = np.asarray(obs, dtype=np.float64).flatten()
        post_mu = np.asarray(posterior_mu, dtype=np.float64).flatten()
        post_var = np.asarray(posterior_var, dtype=np.float64).flatten() + 1e-10
        accuracy = -0.5 * float(np.sum((obs - post_mu) ** 2 / post_var))
        fe = complexity - accuracy
        self.log.debug("Complexity-Accuracy: cpx=%.4f acc=%.4f F=%.4f",
                       complexity, accuracy, fe)
        return float(complexity), float(accuracy), float(fe)

    def prediction_coding_form(
        self,
        prediction_errors: np.ndarray,
        precisions: np.ndarray,
        posterior_mu: np.ndarray,
        prior_mu: Optional[np.ndarray] = None,
    ) -> float:
        """形式3：预测编码形式 F = 精度加权预测误差 + KL(后验||先验)。"""
        pe = np.asarray(prediction_errors, dtype=np.float64).flatten()
        prec = np.asarray(precisions, dtype=np.float64).flatten()
        weighted_err = float(np.sum(prec * pe ** 2))
        if prior_mu is None:
            prior_mu = np.zeros_like(posterior_mu)
        kl = 0.5 * float(np.sum((posterior_mu - prior_mu) ** 2))
        fe = weighted_err + kl
        self.log.debug("Prediction-coding FE: weighted_err=%.4f kl=%.4f F=%.4f",
                       weighted_err, kl, fe)
        return float(fe)

    def expected_free_energy(
        self,
        predicted_obs: np.ndarray,
        predicted_var: np.ndarray,
        preferred_obs: np.ndarray,
        preferred_var: Optional[np.ndarray] = None,
    ) -> float:
        """期望自由能 G = E_q[F]：用于主动推理策略选择。

        G ≈ KL(q(o|pi) || p(o))  即预测观测与偏好观测之间的散度。
        """
        pred = np.asarray(predicted_obs, dtype=np.float64).flatten()
        pref = np.asarray(preferred_obs, dtype=np.float64).flatten()
        pv = np.asarray(predicted_var, dtype=np.float64).flatten() + 1e-10
        if preferred_var is None:
            preferred_var = np.ones_like(pref)
        preferred_var = np.asarray(preferred_var, dtype=np.float64).flatten() + 1e-10
        # 用高斯 KL 近似
        g = gaussian_kl(pred, pref, pv, preferred_var)
        self.log.debug("Expected free energy G=%.4f", g)
        return float(g)

    def is_surprised(self, fe: float) -> bool:
        """自由能是否超过阈值（惊讶）。"""
        surprised = fe > self.config.free_energy_threshold
        self.log.debug("Surprise check: fe=%.4f threshold=%.4f surprised=%s",
                       fe, self.config.free_energy_threshold, surprised)
        return surprised
