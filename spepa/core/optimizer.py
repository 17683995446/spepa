"""SPEPA 双模式优化器（微修正 + 结构性重构）。

设计原则（第一性原理）：
- 单一性原则：每次优化只应用一种权重更新策略，避免叠加
- 因果性原则：权重更新由明确的策略驱动，预测编码与 Hebb-like 二选一
- 自适应性原则：学习率根据误差变化率动态调整

两种更新策略（互斥）：
1. 预测编码更新（predictive_coding_update）：基于预测误差的梯度下降
   - model.update 内部完成，更新 mu/W/b/var
2. Hebb-like 更新（hebbian_update）：基于神经元共激活的连接增强
   - 只更新 W/b，不调用 model.update

模式选择：
- micro 模式：误差小，使用预测编码更新（稳定收敛）
- hebbian 模式：误差中等，使用 Hebb-like 更新（探索性学习）
- structural 模式：误差大，结构性重构（大幅调整）
"""
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.core.predictive_model import PredictiveModel


class Optimizer(LoggedMixin):
    """双模式优化器（含学习率自适应）。"""

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.optimizer")
        self.config = config
        self.micro_lr = config.learning_rate_micro
        self.struct_lr = config.learning_rate_struct
        self.micro_threshold = config.micro_correction_threshold
        self.struct_threshold = config.structural_reconstruction_threshold
        self.mode_history: List[str] = []
        self.rng = np.random.default_rng(config.seed)
        # 学习率自适应：记录最近误差历史
        self._error_history: deque = deque(maxlen=10)
        self._adaptive_lr: float = self.micro_lr
        # 连续高误差计数（用于触发结构性重构）
        self._high_error_streak: int = 0
        self.log.info("Optimizer initialized (micro_lr=%.4f struct_lr=%.4f)",
                      self.micro_lr, self.struct_lr)

    def select_mode(self, prediction_error: float) -> str:
        """根据预测误差选择优化模式。

        三档模式：
        - micro (err < micro_threshold)：预测编码更新，稳定收敛
        - hebbian (micro_threshold <= err <= struct_threshold)：Hebb-like 探索性学习
        - structural (err > struct_threshold 或连续高误差)：结构性重构
        """
        # 清洗误差
        if not np.isfinite(prediction_error):
            prediction_error = 0.0
        prediction_error = max(0.0, prediction_error)

        # 连续高误差检测
        if prediction_error > self.struct_threshold:
            self._high_error_streak += 1
        else:
            self._high_error_streak = 0

        # 模式选择
        if prediction_error < self.micro_threshold:
            mode = "micro"
        elif prediction_error > self.struct_threshold:
            mode = "structural"
        else:
            # 中间区域：使用 Hebb-like 探索性学习
            mode = "hebbian"

        # 连续高误差超过阈值时强制结构性重构
        if self._high_error_streak >= 5 and mode != "structural":
            mode = "structural"
            self.log.warning("Forced structural mode after %d high-error steps",
                             self._high_error_streak)

        self.mode_history.append(mode)
        self.log.debug("Optimization mode selected: %s (err=%.4f streak=%d)",
                       mode, prediction_error, self._high_error_streak)
        return mode

    def _update_adaptive_lr(self, prediction_error: float) -> float:
        """学习率自适应：根据误差变化率调整。

        - 误差下降：增大学习率（加速收敛）
        - 误差上升：减小学习率（避免震荡）
        - 误差稳定：保持当前学习率
        """
        self._error_history.append(prediction_error)
        if len(self._error_history) < 2:
            self._adaptive_lr = self.micro_lr
            return self._adaptive_lr

        # 计算误差变化率
        recent = list(self._error_history)
        # 使用最近几个误差的均值与之前的均值比较
        n = len(recent)
        if n >= 4:
            recent_mean = np.mean(recent[-2:])
            prev_mean = np.mean(recent[-4:-2])
            if prev_mean > 1e-10:
                change_rate = (recent_mean - prev_mean) / prev_mean
            else:
                change_rate = 0.0
        else:
            change_rate = 0.0

        # 自适应调整
        base_lr = self.micro_lr
        if change_rate < -0.05:
            # 误差下降超过5%：增大学习率（最多2倍）
            self._adaptive_lr = min(base_lr * 1.5, self._adaptive_lr * 1.1)
        elif change_rate > 0.05:
            # 误差上升超过5%：减小学习率（最少0.5倍）
            self._adaptive_lr = max(base_lr * 0.5, self._adaptive_lr * 0.9)
        # else: 保持当前学习率

        # 确保学习率在合理范围
        self._adaptive_lr = float(np.clip(self._adaptive_lr, base_lr * 0.1, base_lr * 3.0))
        self.log.debug("Adaptive lr: %.6f (change_rate=%.4f)", self._adaptive_lr, change_rate)
        return self._adaptive_lr

    def predictive_coding_update(self, model: PredictiveModel,
                                  observation: np.ndarray,
                                  prediction_error: float) -> float:
        """预测编码更新：基于预测误差的梯度下降（单一更新）。

        只调用 model.update，不额外做 Hebb-like 更新。
        这是 micro 模式使用的更新策略。
        """
        obs = np.asarray(observation, dtype=np.float64).flatten()
        if obs.shape[0] != model.dim:
            obs = np.resize(obs, model.dim)
        # 防御性清洗
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            self.log.warning("Predictive coding: observation contained non-finite values, sanitized")
        # 使用自适应学习率
        lr = self._update_adaptive_lr(prediction_error)
        # 单一更新：只调用 model.update
        errors = model.update(obs, lr=lr)
        total_err = float(sum(np.sum(e ** 2) for e in errors))
        if not np.isfinite(total_err):
            total_err = 0.0
        self.log.info("Predictive coding update applied: lr=%.6f total_err=%.4f",
                      lr, total_err)
        return total_err

    def hebbian_update(self, model: PredictiveModel, observation: np.ndarray,
                       prediction_error: float) -> float:
        """Hebb-like 更新：基于神经元共激活的连接增强（单一更新）。

        只做 Hebb-like 权重更新，不调用 model.update。
        这是 hebbian 模式使用的更新策略。
        """
        obs = np.asarray(observation, dtype=np.float64).flatten()
        if obs.shape[0] != model.dim:
            obs = np.resize(obs, model.dim)
        # 防御性清洗
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            self.log.warning("Hebbian: observation contained non-finite values, sanitized")

        # 先 forward 获取当前误差（不更新权重）
        _, errors = model.forward(obs)
        # 只更新 mu（感知），不更新 W/b（避免与 Hebb-like 叠加）
        lr = self._update_adaptive_lr(prediction_error)
        for i, layer in enumerate(model.layers):
            err = errors[i]
            # 只更新 mu（感知阶段）
            prec = layer.precision
            delta = lr * prec * err
            delta = np.clip(delta, -1.0, 1.0)
            layer.mu = np.clip(layer.mu + delta, -10.0, 10.0)
            # 方差更新
            layer.var = np.clip(layer.var + 0.01 * np.abs(err), 0.05, 5.0)

        # Hebb-like 权重更新：连接同时激活则增强
        for i, layer in enumerate(model.layers):
            pre = obs if i == 0 else model.layers[i - 1].mu
            post = layer.mu
            # 防御性清洗
            if not np.all(np.isfinite(pre)):
                pre = np.nan_to_num(pre, nan=0.0, posinf=1.0, neginf=0.0)
            if not np.all(np.isfinite(post)):
                post = np.nan_to_num(post, nan=0.0, posinf=1.0, neginf=0.0)
            # ΔW = lr * post * pre^T (外积)
            delta_W = lr * 0.1 * np.outer(post, pre)
            layer.W += delta_W
            layer.W = np.clip(layer.W, -5.0, 5.0)
            # 偏置更新
            layer.b += lr * 0.1 * post
            layer.b = np.clip(layer.b, -5.0, 5.0)

        total_err = float(sum(np.sum(e ** 2) for e in errors))
        if not np.isfinite(total_err):
            total_err = 0.0
        self.log.info("Hebbian update applied: lr=%.6f total_err=%.4f",
                      lr, total_err)
        return total_err

    def structural_reconstruction(self, model: PredictiveModel, observation: np.ndarray,
                                  prediction_error: float) -> float:
        """结构性重构：大幅调整权重与方差（单一更新）。

        只做结构性重构，不调用 model.update 的权重更新部分。
        """
        obs = np.asarray(observation, dtype=np.float64).flatten()
        if obs.shape[0] != model.dim:
            obs = np.resize(obs, model.dim)
        # 防御性清洗
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            self.log.warning("Structural: observation contained non-finite values, sanitized")

        # 先 forward 获取误差（用于 mu 更新）
        _, errors = model.forward(obs)
        # 大幅更新 mu（使用更大的学习率）
        big_lr = self.struct_lr * 10
        for i, layer in enumerate(model.layers):
            err = errors[i]
            prec = layer.precision
            delta = big_lr * prec * err
            delta = np.clip(delta, -1.0, 1.0)
            layer.mu = np.clip(layer.mu + delta, -10.0, 10.0)

        # 重构权重：注入随机性 + 大幅调整（使用种子化 RNG）
        for i, layer in enumerate(model.layers):
            mask = self.rng.random(layer.W.shape) < 0.3
            n_masked = int(np.sum(mask))
            if n_masked > 0:
                layer.W[mask] = self.rng.normal(0, 0.1, n_masked)
            # 增大方差（提升探索）
            layer.var = np.clip(layer.var * 1.5, 0.05, 5.0)

        total_err = float(sum(np.sum(e ** 2) for e in errors))
        if not np.isfinite(total_err):
            total_err = 0.0
        self.log.warning("Structural reconstruction applied: total_err=%.4f", total_err)
        return total_err

    def optimize(self, model: PredictiveModel, observation: np.ndarray,
                 prediction_error: float) -> Dict[str, Any]:
        """根据误差自动选择模式并优化（每次只应用一种更新策略）。"""
        mode = self.select_mode(prediction_error)
        if mode == "micro":
            err = self.predictive_coding_update(model, observation, prediction_error)
        elif mode == "hebbian":
            err = self.hebbian_update(model, observation, prediction_error)
        else:  # structural
            err = self.structural_reconstruction(model, observation, prediction_error)
        self.log.info("Optimize done: mode=%s pre_err=%.4f post_err=%.4f",
                      mode, prediction_error, err)
        return {
            "mode": mode,
            "pre_error": prediction_error,
            "post_error": err,
            "adaptive_lr": self._adaptive_lr,
        }

    def get_state(self) -> Dict[str, Any]:
        from collections import Counter
        counts = Counter(self.mode_history)
        return {
            "mode_history_size": len(self.mode_history),
            "mode_counts": dict(counts),
            "adaptive_lr": self._adaptive_lr,
            "high_error_streak": self._high_error_streak,
        }

    def reset(self) -> None:
        """重置优化器内部状态（保留配置）。"""
        self.mode_history.clear()
        self._error_history.clear()
        self._adaptive_lr = self.micro_lr
        self._high_error_streak = 0
        self.log.info("Optimizer state reset")
