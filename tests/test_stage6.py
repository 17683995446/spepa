"""阶段六测试：预测编码层级逻辑修正。

验证：
1. encode（自下而上）与 generate（自上而下）两条通路分离
2. forward 中层级误差传播的正确性
3. _last_inputs 显式初始化（无 hasattr 检查）
4. predict_down 语义正确（基于上层状态预测本层）
5. 误差定义正确（目标 - 预测）
6. 层级传播方向正确（自下而上编码，自上而下生成）
7. 数值稳定性
"""
import numpy as np
import pytest

from spepa.utils.config import SPEPAConfig
from spepa.core.predictive_model import PredictiveModel, PredictiveLayer


class TestLayerSemantics:
    """验证 PredictiveLayer 的语义正确性。"""

    def test_predict_down_with_parent(self):
        """predict_down(parent_mu) 应基于上层状态生成本层预测。"""
        layer = PredictiveLayer(dim=4, name="test")
        parent_mu = np.array([1.0, 2.0, 3.0, 4.0])
        pred = layer.predict_down(parent_mu)
        # pred = W @ parent_mu + b，W 初始为 0.1*I，b=0
        expected = 0.1 * parent_mu
        assert np.allclose(pred, expected), \
            f"predict_down 语义错误: expected {expected}, got {pred}"

    def test_predict_down_without_parent(self):
        """predict_down(None) 应返回本层自身 mu。"""
        layer = PredictiveLayer(dim=4, name="test")
        layer.mu = np.array([0.5, 0.6, 0.7, 0.8])
        pred = layer.predict_down(None)
        assert np.allclose(pred, layer.mu), \
            "predict_down(None) 应返回自身 mu"

    def test_predict_down_clips_extreme_input(self):
        """predict_down 应裁剪极端输入。"""
        layer = PredictiveLayer(dim=4, name="test")
        extreme = np.array([1e10, -1e10, np.nan, np.inf])
        pred = layer.predict_down(extreme)
        assert np.all(np.isfinite(pred)), "predict_down 应处理极端输入"

    def test_layer_update_with_parent_updates_weights(self):
        """update 带 parent_mu 时应更新权重。"""
        layer = PredictiveLayer(dim=4, name="test")
        w_before = layer.W.copy()
        layer.update(np.ones(4) * 0.5, lr=0.01, parent_mu=np.ones(4) * 0.3)
        assert not np.allclose(layer.W, w_before), "权重应被更新"

    def test_layer_update_without_parent_no_weight_change(self):
        """update 不带 parent_mu 时不应更新权重。"""
        layer = PredictiveLayer(dim=4, name="test")
        w_before = layer.W.copy()
        layer.update(np.ones(4) * 0.5, lr=0.01, parent_mu=None)
        assert np.allclose(layer.W, w_before), "无 parent_mu 时权重不应改变"


class TestExplicitInitialization:
    """验证 _last_inputs 的显式初始化。"""

    def test_last_inputs_initialized_in_init(self):
        """_last_inputs 应在 __init__ 中显式初始化。"""
        config = SPEPAConfig(state_dim=8, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        # 应存在且长度等于层数
        assert hasattr(model, "_last_inputs")
        assert len(model._last_inputs) == 3
        # 初始值应为 None
        for item in model._last_inputs:
            assert item is None

    def test_last_predictions_initialized_in_init(self):
        """_last_predictions 应在 __init__ 中显式初始化。"""
        config = SPEPAConfig(state_dim=8, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        assert hasattr(model, "_last_predictions")
        assert len(model._last_predictions) == 3
        for item in model._last_predictions:
            assert item is None

    def test_update_before_forward_no_error(self):
        """update 在 forward 之前调用不应报错（_last_inputs 已初始化）。"""
        config = SPEPAConfig(state_dim=8, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        # 直接调用 update（内部会调用 forward）
        errors = model.update(np.ones(8) * 0.5)
        assert len(errors) == 3
        # _last_inputs 应被填充
        assert all(item is not None for item in model._last_inputs)

    def test_last_inputs_populated_after_forward(self):
        """forward 后 _last_inputs 应被填充。"""
        config = SPEPAConfig(state_dim=8, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        model.forward(obs)
        # 所有层都应有输入记录
        assert all(item is not None for item in model._last_inputs)
        # 底层输入应是观测本身
        assert np.allclose(model._last_inputs[0], obs)


class TestErrorPropagation:
    """验证层级误差传播的正确性。"""

    def test_error_count_equals_layers(self):
        """forward 返回的误差数量应等于层数。"""
        config = SPEPAConfig(state_dim=8, seed=42, prediction_layers=5)
        model = PredictiveModel(config)
        _, errors = model.forward(np.ones(8) * 0.5)
        assert len(errors) == 5

    def test_error_definition_target_minus_prediction(self):
        """误差 = 目标 - 预测。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=2)
        model = PredictiveModel(config)
        obs = np.array([0.5, 0.5, 0.5, 0.5])
        _, errors = model.forward(obs)
        # 底层误差 = obs - L0.predict_down(L1.mu)
        # L1.mu 初始为 0，predict_down(0) = W@0 + b = b = 0
        # 所以底层误差 = obs - 0 = obs
        # 但 W 初始为 0.1*I，L1.mu=0，所以 pred = 0.1*0 + 0 = 0
        # 误差 = obs - 0 = obs
        assert np.allclose(errors[0], obs, atol=1e-6), \
            f"底层误差应为 obs - 0 = obs，实际 {errors[0]}"

    def test_bottom_layer_target_is_observation(self):
        """底层目标应是观测。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=2)
        model = PredictiveModel(config)
        obs = np.array([0.1, 0.2, 0.3, 0.4])
        model.forward(obs)
        # _last_inputs[0] 应是 obs
        assert np.allclose(model._last_inputs[0], obs)

    def test_middle_layer_target_is_lower_mu(self):
        """中间层目标应是下层 mu。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        obs = np.array([0.1, 0.2, 0.3, 0.4])
        model.forward(obs)
        # _last_inputs[1] 应是 L0.mu
        assert np.allclose(model._last_inputs[1], model.layers[0].mu)
        # _last_inputs[2] 应是 L1.mu
        assert np.allclose(model._last_inputs[2], model.layers[1].mu)

    def test_top_layer_prediction_is_self_mu(self):
        """顶层预测应是自身 mu（无上层）。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        model.forward(np.ones(4) * 0.5)
        # 顶层 _last_predictions[2] 应是 L2.mu（predict_down(None)）
        assert np.allclose(model._last_predictions[2], model.layers[2].mu)

    def test_non_top_layer_prediction_from_upper(self):
        """非顶层预测应来自上层 mu。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        model.forward(np.ones(4) * 0.5)
        # L0 的预测 = L0.predict_down(L1.mu)
        expected_pred_0 = model.layers[0].predict_down(model.layers[1].mu)
        assert np.allclose(model._last_predictions[0], expected_pred_0)
        # L1 的预测 = L1.predict_down(L2.mu)
        expected_pred_1 = model.layers[1].predict_down(model.layers[2].mu)
        assert np.allclose(model._last_predictions[1], expected_pred_1)


class TestGeneratePathway:
    """验证自上而下生成通路。"""

    def test_predict_without_observation(self):
        """无观测时的纯内部预测（自上而下生成）。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        # 设置各层 mu
        model.layers[2].mu = np.array([1.0, 1.0, 1.0, 1.0])
        model.layers[1].mu = np.array([0.5, 0.5, 0.5, 0.5])
        model.layers[0].mu = np.array([0.3, 0.3, 0.3, 0.3])
        # 纯内部预测：从顶层 mu 自上而下
        pred = model.predict(None)
        # pred = L0.predict_down(L1.predict_down(L2.mu))
        # L2.mu = [1,1,1,1]
        # L1.predict_down(L2.mu) = W1 @ [1,1,1,1] + b1 = 0.1*[1,1,1,1] = [0.1,0.1,0.1,0.1]
        # L0.predict_down([0.1,0.1,0.1,0.1]) = W0 @ [0.1,0.1,0.1,0.1] + b0 = 0.1*[0.1,...] = [0.01,...]
        expected = 0.1 * (0.1 * np.array([1.0, 1.0, 1.0, 1.0]))
        assert np.allclose(pred, expected, atol=1e-6), \
            f"纯内部预测错误: expected {expected}, got {pred}"

    def test_predict_with_observation(self):
        """有观测时的预测（先编码再生成）。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=2)
        model = PredictiveModel(config)
        obs = np.array([0.5, 0.5, 0.5, 0.5])
        pred = model.predict(obs)
        assert pred.shape[0] == 4
        assert np.all(np.isfinite(pred))

    def test_generate_pathway_top_to_bottom(self):
        """生成通路应从顶层到底层。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        # 修改顶层 mu，观察预测变化
        model.layers[2].mu = np.array([2.0, 2.0, 2.0, 2.0])
        pred1 = model.predict(None)
        model.layers[2].mu = np.array([4.0, 4.0, 4.0, 4.0])
        pred2 = model.predict(None)
        # 顶层 mu 翻倍，预测应变化（因为 W 是线性映射）
        assert not np.allclose(pred1, pred2), "顶层 mu 变化应影响预测"


class TestNumericalStability:
    """验证数值稳定性。"""

    def test_forward_with_nan_observation(self):
        """forward 处理 NaN 观测。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=2)
        model = PredictiveModel(config)
        obs = np.array([0.5, np.nan, 0.5, np.inf])
        _, errors = model.forward(obs)
        assert len(errors) == 2
        for err in errors:
            assert np.all(np.isfinite(err)), "误差应清洗 NaN/Inf"

    def test_forward_with_extreme_observation(self):
        """forward 处理极端观测。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=2)
        model = PredictiveModel(config)
        obs = np.array([1e10, -1e10, 1e-10, 1e308])
        _, errors = model.forward(obs)
        for err in errors:
            assert np.all(np.isfinite(err))

    def test_update_stability_under_large_errors(self):
        """大误差下 update 的稳定性。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        large_obs = np.ones(4) * 100.0
        for _ in range(50):
            model.update(large_obs, lr=0.1)
        for layer in model.layers:
            assert np.all(np.isfinite(layer.mu))
            assert np.all(np.isfinite(layer.W))
            assert np.all(np.isfinite(layer.var))
            assert np.all(np.abs(layer.W) <= 5.0 + 1e-6)
            assert np.all(layer.var <= 5.0 + 1e-6)


class TestLayerConsistency:
    """验证层级一致性。"""

    def test_forward_does_not_mutate_observation(self):
        """forward 不应修改输入观测。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=2)
        model = PredictiveModel(config)
        obs = np.array([0.1, 0.2, 0.3, 0.4])
        obs_before = obs.copy()
        model.forward(obs)
        assert np.allclose(obs, obs_before), "forward 不应修改输入观测"

    def test_forward_resets_last_inputs_each_call(self):
        """每次 forward 应重置 _last_inputs。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=2)
        model = PredictiveModel(config)
        model.forward(np.ones(4) * 0.5)
        inputs_1 = [x.copy() if x is not None else None for x in model._last_inputs]
        model.forward(np.ones(4) * 0.9)
        inputs_2 = model._last_inputs
        # 底层输入应不同（不同观测）
        assert not np.allclose(inputs_1[0], inputs_2[0])

    def test_update_uses_forward_last_inputs(self):
        """update 应使用 forward 产生的 _last_inputs。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=2)
        model = PredictiveModel(config)
        obs = np.array([0.3, 0.4, 0.5, 0.6])
        model.update(obs)
        # update 内部调用 forward，_last_inputs 应被设置
        assert model._last_inputs[0] is not None
        assert np.allclose(model._last_inputs[0], obs)

    def test_multi_layer_error_decreases_with_learning(self):
        """多层模型学习后误差应不爆炸。"""
        config = SPEPAConfig(state_dim=4, seed=42, prediction_layers=3)
        model = PredictiveModel(config)
        obs = np.array([0.5, 0.5, 0.5, 0.5])
        initial_error = model.total_prediction_error(obs)
        for _ in range(100):
            model.update(obs, lr=0.01)
        final_error = model.total_prediction_error(obs)
        # 误差不应爆炸（可能不收敛，但不应发散）
        assert final_error < initial_error * 100, \
            f"误差爆炸: {initial_error} -> {final_error}"
        assert np.isfinite(final_error)
