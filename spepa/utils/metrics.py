"""SPEPA 评估指标。

提供自由能、KL散度、精度、召回等指标的实现。
"""
import numpy as np
from typing import Dict, Any, List

from spepa.utils.logger import get_logger


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """计算 KL(p||q) 散度。

    输入会被裁剪到非负区间并归一化为概率分布，避免负值导致 log 产生 NaN。
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # 裁剪到非负区间（KL 散度要求概率非负）
    p = np.clip(np.abs(p), 0.0, None) + eps
    q = np.clip(np.abs(q), 0.0, None) + eps
    # 归一化为概率分布
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    val = float(np.sum(p * np.log(p / q)))
    return max(val, 0.0)


def gaussian_kl(mu_p, mu_q, var_p, var_q) -> float:
    """两个高斯分布之间的 KL 散度（保证非负）。"""
    mu_p = np.asarray(mu_p, dtype=np.float64)
    mu_q = np.asarray(mu_q, dtype=np.float64)
    var_p = np.asarray(var_p, dtype=np.float64) + 1e-10
    var_q = np.asarray(var_q, dtype=np.float64) + 1e-10
    # 清洗 NaN/Inf
    mu_p = np.nan_to_num(mu_p, nan=0.0, posinf=1.0, neginf=0.0)
    mu_q = np.nan_to_num(mu_q, nan=0.0, posinf=1.0, neginf=0.0)
    var_p = np.nan_to_num(var_p, nan=0.1, posinf=1.0, neginf=0.1)
    var_q = np.nan_to_num(var_q, nan=0.1, posinf=1.0, neginf=0.1)
    term1 = np.log(var_q / var_p)
    term2 = (var_p + (mu_p - mu_q) ** 2) / var_q
    val = float(0.5 * np.sum(term1 + term2 - 1.0))
    # KL 散度数学上恒非负
    return max(val, 0.0)


def prediction_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """预测误差（L2 范数）。"""
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    return float(np.linalg.norm(pred - actual))


def mse(pred: np.ndarray, actual: np.ndarray) -> float:
    """均方误差。"""
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    return float(np.mean((pred - actual) ** 2))


def entropy(p: np.ndarray, eps: float = 1e-10) -> float:
    """香农熵。"""
    p = np.asarray(p, dtype=np.float64) + eps
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度。"""
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    na = np.linalg.norm(a) + 1e-10
    nb = np.linalg.norm(b) + 1e-10
    return float(np.dot(a, b) / (na * nb))


class MetricsTracker:
    """跟踪并聚合运行时指标。"""

    def __init__(self):
        self._logger = get_logger("spepa.metrics")
        self._history: Dict[str, List[float]] = {}
        self._logger.info("MetricsTracker initialized")

    def record(self, name: str, value: float) -> None:
        """记录一个指标值。"""
        if name not in self._history:
            self._history[name] = []
        self._history[name].append(float(value))
        self._logger.debug("Metric recorded: %s=%.4f (n=%d)", name, value, len(self._history[name]))

    def summary(self) -> Dict[str, Dict[str, float]]:
        """返回所有指标的统计摘要。"""
        out = {}
        for name, vals in self._history.items():
            if not vals:
                continue
            arr = np.asarray(vals)
            out[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "last": float(vals[-1]),
                "count": int(len(vals)),
            }
        self._logger.info("Metrics summary generated for %d metrics", len(out))
        return out

    def get(self, name: str) -> List[float]:
        """获取某个指标的完整历史。"""
        return list(self._history.get(name, []))

    def get_recent(self, name: str, n: int = 10) -> List[float]:
        """获取某个指标最近 n 个值。"""
        vals = self._history.get(name, [])
        return list(vals[-n:]) if vals else []

    def reset(self) -> None:
        """清空所有指标。"""
        self._history.clear()
        self._logger.info("MetricsTracker reset")

    def to_dict(self) -> Dict[str, Any]:
        return {k: list(v) for k, v in self._history.items()}
