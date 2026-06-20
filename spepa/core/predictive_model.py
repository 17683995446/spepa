"""SPEPA 层级预测模型（多层预测编码）。

实现自上而下的预测与自下而上的误差传播，每层带精度加权。

设计原则（第一性原理 - 预测编码）：
- 自上而下（生成通路 generate）：顶层状态 → 预测下层状态 → ... → 预测观测
- 自下而上（编码通路 encode）：观测 → 计算底层误差 → 误差向上传播 → 更新各层
- 误差定义：每层误差 = 该层实际状态 - 上层对该层的预测
- predict_down(parent_mu) 语义：基于上层状态 parent_mu 生成对当前层的预测

层级索引约定：
- layers[0] = 底层（最接近观测）
- layers[-1] = 顶层（最高层抽象）
- 信息流：观测 → L0 → L1 → ... → L(top)
- 预测流：L(top) → ... → L1 → L0 → 观测预测
"""
import numpy as np
from typing import List, Tuple, Optional

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class PredictiveLayer:
    """单层预测编码单元。

    维护该层的均值向量、方差（精度）与权重矩阵。
    - mu: 该层隐状态均值
    - var: 该层隐状态方差（精度=1/var）
    - W: 自上而下预测权重（上层 -> 本层）
    - b: 偏置
    """

    def __init__(self, dim: int, name: str = "layer"):
        self.dim = dim
        self.name = name
        self.mu = np.zeros(dim)              # 该层隐状态均值
        self.var = np.ones(dim) * 0.5        # 该层隐状态方差（精度=1/var）
        self.W = np.eye(dim) * 0.1           # 自上而下预测权重（上层->本层）
        self.b = np.zeros(dim)               # 偏置
        self._log_name = "spepa.core.predictive_model." + name

    @property
    def precision(self) -> np.ndarray:
        """精度 = 1/方差。"""
        return 1.0 / (self.var + 1e-10)

    def predict_down(self, parent_mu: Optional[np.ndarray] = None) -> np.ndarray:
        """自上而下生成对当前层的预测（基于上层状态）。

        语义：给定上层状态 parent_mu，生成本层的预测值。
        - parent_mu=None 时返回本层自身均值（纯内部预测的顶层起点）
        - parent_mu 给定时：pred = W @ parent_mu + b
        """
        if parent_mu is None:
            return self.mu.copy()
        parent_mu = np.asarray(parent_mu, dtype=np.float64).flatten()
        # 裁剪输入，避免矩阵乘法溢出
        parent_mu = np.clip(parent_mu, -10.0, 10.0)
        if not np.all(np.isfinite(parent_mu)):
            parent_mu = np.nan_to_num(parent_mu, nan=0.0, posinf=10.0, neginf=-10.0)
        pred = self.W @ parent_mu + self.b
        # 清洗预测结果
        if not np.all(np.isfinite(pred)):
            pred = np.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
        return pred

    def update(self, prediction_error: np.ndarray, lr: float,
               parent_mu: Optional[np.ndarray] = None) -> None:
        """根据预测误差更新该层状态与权重（精度加权）。

        - mu 更新：感知（朝误差方向移动，降低误差）
        - W/b 更新：学习（降低该层预测误差，梯度下降）
        - var 更新：误差大则不确定性提升

        参数：
        - prediction_error: 本层误差 = 本层实际状态 - 上层对本层的预测
        - parent_mu: 上层状态（用于 W 的梯度更新）
        """
        prediction_error = np.asarray(prediction_error, dtype=np.float64).flatten()
        # 数值稳定性：裁剪预测误差，避免溢出
        prediction_error = np.clip(prediction_error, -10.0, 10.0)
        # 清洗 NaN/Inf
        if not np.all(np.isfinite(prediction_error)):
            prediction_error = np.nan_to_num(prediction_error, nan=0.0,
                                             posinf=10.0, neginf=-10.0)
        prec = self.precision
        delta = lr * prec * prediction_error
        # 裁剪 delta 避免单步更新过大
        delta = np.clip(delta, -1.0, 1.0)
        self.mu = self.mu + delta
        # mu 本身也裁剪，避免发散
        self.mu = np.clip(self.mu, -10.0, 10.0)
        # 权重/偏置更新以降低预测误差（梯度下降方向）
        if parent_mu is not None:
            parent_mu = np.asarray(parent_mu, dtype=np.float64).flatten()
            parent_mu = np.clip(parent_mu, -10.0, 10.0)
            self.W += lr * np.outer(prediction_error, parent_mu)
            # 权重裁剪，避免数值爆炸
            self.W = np.clip(self.W, -5.0, 5.0)
            self.b += lr * prediction_error
            self.b = np.clip(self.b, -5.0, 5.0)
        # 方差随误差增大而增大（不确定性提升）
        self.var = np.clip(self.var + 0.01 * np.abs(prediction_error), 0.05, 5.0)


class PredictiveModel(LoggedMixin):
    """层级预测模型：多层预测编码，含精度加权。

    层级结构（自底向上）：
        layers[0]（底层，对接观测） -> layers[1] -> ... -> layers[-1]（顶层）

    两条通路：
    - encode（自下而上）：观测 -> L0 误差 -> L1 误差 -> ... -> L(top) 误差
    - generate（自上而下）：L(top) -> 预测 L(top-1) -> ... -> 预测观测
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.predictive_model")
        self.config = config
        self.dim = config.state_dim
        self.layers: List[PredictiveLayer] = []
        for i in range(config.prediction_layers):
            layer = PredictiveLayer(self.dim, name=f"L{i}")
            self.layers.append(layer)
        # 显式初始化 _last_inputs，消除 hasattr 检查
        self._last_inputs: List[Optional[np.ndarray]] = [None] * len(self.layers)
        # 显式初始化 _last_predictions，记录每层从上层收到的预测
        self._last_predictions: List[Optional[np.ndarray]] = [None] * len(self.layers)
        self.log.info("PredictiveModel initialized with %d layers (dim=%d)",
                      len(self.layers), self.dim)

    def forward(self, observation: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """自下而上编码观测，返回顶层状态与各层误差。

        编码通路（自下而上）：
        - L0 误差 = obs - L1对L0的预测（若只有1层，则 = obs - L0.mu）
        - L1 误差 = L0.mu - L2对L1的预测
        - ...
        - L(top) 误差 = L(top-1).mu - L(top)自身预测（顶层无上层，用自身mu）

        注意：predict_down(parent_mu) 中 parent_mu 是上层状态，生成对当前层的预测。
        因此 L(i) 的预测来自 L(i+1)：pred_i = L(i).predict_down(L(i+1).mu)
        但由于 W 是每层自己的，更准确的语义是：
        L(i) 的误差 = L(i).mu 的更新目标 - L(i+1) 通过 L(i).W 对 L(i) 的预测

        简化实现（保持与原架构兼容）：
        - 底层 L0：误差 = obs - L0.predict_down(L1.mu)（L1 预测 L0）
        - 中间层 Li：误差 = Li-1.mu - Li.predict_down(L(i+1).mu)（L(i+1) 预测 Li）
        - 顶层 L(top)：误差 = L(top-1).mu - L(top).mu（无上层，用自身mu做预测）

        但为保持向后兼容和数值稳定性，采用如下简化：
        - 每层误差 = 该层输入 - 该层基于上层状态的预测
        - 底层输入 = obs
        - 中间层输入 = 下层 mu
        - 顶层输入 = 下层 mu（顶层无上层预测，误差=下层mu-顶层mu）
        """
        obs = np.asarray(observation, dtype=np.float64).flatten()
        if obs.shape[0] != self.dim:
            obs = np.resize(obs, self.dim)
        # 清洗 NaN/Inf
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            self.log.warning("Forward pass: observation contained non-finite values, sanitized")

        n_layers = len(self.layers)
        errors: List[np.ndarray] = []
        # 重置 _last_inputs 和 _last_predictions
        self._last_inputs = [None] * n_layers
        self._last_predictions = [None] * n_layers

        # 自下而上编码
        # 每层的"输入"（即该层需要预测的目标）：
        #   L0 的输入 = obs
        #   Li 的输入 = L(i-1).mu（下层状态向上传递）
        # 每层的"预测"（来自上层）：
        #   L(top) 无上层，预测 = 自身 mu
        #   Li 的预测 = L(i).predict_down(L(i+1).mu)（上层状态通过本层W预测本层）
        # 但这里有个语义问题：predict_down 是"本层基于上层状态预测本层"
        # 所以 L(i) 的预测 = L(i).predict_down(parent_mu=L(i+1).mu)

        # 为保持与原实现兼容（原实现中 parent=current if i==0 else layers[i-1].mu），
        # 且避免破坏现有行为，我们采用如下清晰定义：
        # - 每层误差 = 该层目标 - 该层预测
        # - 底层目标 = obs
        # - 中间层目标 = 下层 mu
        # - 每层预测 = predict_down(上层 mu)，顶层无上层则预测=自身mu

        current_target = obs  # 底层目标是观测
        for i, layer in enumerate(self.layers):
            # 记录该层输入（目标）
            self._last_inputs[i] = current_target.copy()
            # 获取上层状态（用于生成对当前层的预测）
            if i < n_layers - 1:
                # 有上层：上层是 layers[i+1]，用其 mu 生成对当前层的预测
                parent_mu = self.layers[i + 1].mu
            else:
                # 顶层：无上层，用自身 mu 作为预测（纯内部）
                parent_mu = None
            # 生成对当前层的预测
            pred = layer.predict_down(parent_mu)
            self._last_predictions[i] = pred.copy()
            # 误差 = 目标 - 预测
            err = current_target - pred
            # 清洗误差
            if not np.all(np.isfinite(err)):
                err = np.nan_to_num(err, nan=0.0, posinf=10.0, neginf=-10.0)
            errors.append(err)
            # 向上传递：下层 mu 成为上层的输入目标
            current_target = layer.mu

        top_pred = self.layers[-1].mu.copy()
        self.log.debug("Forward pass done, top_pred norm=%.4f, n_errors=%d",
                       float(np.linalg.norm(top_pred)), len(errors))
        return top_pred, errors

    def predict(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """生成对观测的预测（自上而下生成通路）。

        - observation=None：纯内部预测（从顶层 mu 自上而下生成）
        - observation 给定：先 forward 更新内部状态，再自上而下生成观测预测
        """
        if observation is None:
            # 纯内部预测：从顶层开始自上而下生成
            pred = self.layers[-1].mu.copy()
            for i in range(len(self.layers) - 2, -1, -1):
                pred = self.layers[i].predict_down(pred)
            self.log.debug("Internal prediction generated, norm=%.4f",
                           float(np.linalg.norm(pred)))
            return pred
        # 基于当前模型预测观测：先 forward 编码，再自上而下生成
        top_pred, _ = self.forward(observation)
        pred = top_pred
        for i in range(len(self.layers) - 2, -1, -1):
            pred = self.layers[i].predict_down(pred)
        self.log.debug("Prediction generated from observation, norm=%.4f",
                       float(np.linalg.norm(pred)))
        return pred

    def update(self, observation: np.ndarray, lr: Optional[float] = None) -> List[np.ndarray]:
        """根据观测更新各层（精度加权预测编码）。

        流程：
        1. forward 编码观测，得到各层误差和 _last_inputs
        2. 对每层调用 layer.update(error, lr, parent_mu)
           - parent_mu 是用于 W 梯度更新的上层状态
           - 底层 parent_mu = obs（观测作为最底层输入）
           - 中间层 parent_mu = 下层 mu
        """
        lr = lr if lr is not None else self.config.learning_rate_micro
        obs = np.asarray(observation, dtype=np.float64).flatten()
        if obs.shape[0] != self.dim:
            obs = np.resize(obs, self.dim)
        _, errors = self.forward(obs)
        # 对每层更新：parent_mu 是该层的输入（用于 W 梯度）
        for i, layer in enumerate(self.layers):
            parent = self._last_inputs[i]  # 已在 forward 中显式设置
            layer.update(errors[i], lr, parent_mu=parent)
        total_err = float(sum(np.sum(e ** 2) for e in errors))
        if not np.isfinite(total_err):
            total_err = 1e10
        self.log.debug("Model updated with observation, total_err=%.4f", total_err)
        return errors

    def total_prediction_error(self, observation: np.ndarray) -> float:
        """返回对观测的总预测误差。"""
        _, errors = self.forward(observation)
        total = float(sum(np.sum(e ** 2) for e in errors))
        # 清洗溢出
        if not np.isfinite(total):
            total = 1e10
        self.log.debug("Total prediction error=%.4f", total)
        return total

    def precision_weighted_error(self, observation: np.ndarray) -> float:
        """返回精度加权后的预测误差。"""
        _, errors = self.forward(observation)
        total = 0.0
        for layer, err in zip(self.layers, errors):
            total += float(np.sum(layer.precision * err ** 2))
        # 清洗溢出
        if not np.isfinite(total):
            total = 1e10
        self.log.debug("Precision-weighted error=%.4f", total)
        return total

    def get_state(self) -> dict:
        """返回模型状态摘要。"""
        return {
            "n_layers": len(self.layers),
            "mu_norms": [float(np.linalg.norm(l.mu)) for l in self.layers],
            "var_means": [float(np.mean(l.var)) for l in self.layers],
        }
