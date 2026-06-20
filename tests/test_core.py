"""SPEPA 核心模块测试。"""
import numpy as np
import pytest

from spepa.utils.config import SPEPAConfig
from spepa.core.state_bus import StateBus
from spepa.core.predictive_model import PredictiveModel
from spepa.core.free_energy import FreeEnergyCalculator
from spepa.core.drive_engine import DriveEngine
from spepa.core.gap_detector import GapDetector, GAP_LEVELS
from spepa.core.goal_hierarchy import GoalHierarchy
from spepa.core.active_inference import ActiveInference
from spepa.core.metacognition import Metacognition
from spepa.core.memory_system import MemorySystem
from spepa.core.optimizer import Optimizer
from spepa.core.safety_guard import SafetyGuard
from spepa.core.self_model import SelfModel


@pytest.fixture
def config():
    return SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42)


def test_state_bus_pubsub(config):
    bus = StateBus()
    received = []
    bus.subscribe("test", received.append)
    n = bus.publish("test", {"data": 1})
    assert n == 1
    assert received == [{"data": 1}]
    bus.set_state("key", "value")
    assert bus.get_state("key") == "value"
    snap = bus.snapshot()
    assert "key" in snap


def test_predictive_model(config):
    model = PredictiveModel(config)
    obs = np.random.rand(8)
    pred, errors = model.forward(obs)
    assert pred.shape == (8,)
    assert len(errors) == config.prediction_layers
    err = model.total_prediction_error(obs)
    assert err >= 0
    # 更新后误差应下降
    for _ in range(10):
        model.update(obs, lr=0.05)
    err_after = model.total_prediction_error(obs)
    assert err_after <= err + 1e-6


def test_free_energy_three_forms(config):
    fe = FreeEnergyCalculator(config)
    obs = np.random.rand(8)
    mu = np.random.rand(8)
    var = np.ones(8) * 0.5
    # 形式1
    f1 = fe.variational_free_energy(obs, mu, var)
    assert f1 >= 0
    # 形式2
    cpx, acc, f2 = fe.complexity_accuracy_form(obs, mu, var)
    assert isinstance(cpx, float)
    assert isinstance(acc, float)
    # 形式3
    pe = np.random.rand(8)
    prec = np.ones(8) * 2.0
    f3 = fe.prediction_coding_form(pe, prec, mu)
    assert f3 >= 0
    # 期望自由能
    g = fe.expected_free_energy(mu, var, obs)
    assert g >= 0


def test_drive_engine(config):
    de = DriveEngine(config)
    state = np.ones(8) * 0.5
    action = np.ones(4) * 0.1
    action = np.resize(action, 8)
    goal = np.ones(8) * 0.7
    rs = de.reward_safe(state, action)
    rg = de.reward_gain(state, action, goal)
    re = de.reward_eff(action, 0.5)
    total = de.total_reward(state, action, goal, 0.5)
    assert isinstance(rs, float)
    assert isinstance(rg, float)
    assert isinstance(re, float)
    assert isinstance(total, float)


def test_gap_detector_levels(config):
    gd = GapDetector(config)
    # L0
    gap = gd.detect_prediction_gap(0.1, "test")
    assert gap.level == GAP_LEVELS[0]
    assert not gap.is_actionable()
    # L3
    gap = gd.detect_prediction_gap(1.5, "test")
    assert gap.level == GAP_LEVELS[3]
    assert gap.is_actionable()
    # 分布缺口
    p = np.array([0.5, 0.5])
    q = np.array([0.3, 0.7])
    gap = gd.detect_distribution_gap(p, q)
    assert gap.kl_value >= 0


def test_goal_hierarchy(config):
    gh = GoalHierarchy(config)
    state = np.ones(8) * 0.5
    result = gh.arbitrate(state)
    assert "dominant_goal" in result
    assert "dominant_level" in result
    # 设置任务目标
    gh.set_task_goal(np.ones(8) * 0.9)
    assert len(gh.goals[2]) >= 1


def test_active_inference(config):
    fe = FreeEnergyCalculator(config)
    ai = ActiveInference(config, fe)
    policies = ai.generate_policies(n_policies=8, horizon=1, exploration_ratio=0.3)
    assert len(policies) == 8

    def predict_fn(state, action):
        action = np.resize(np.asarray(action, dtype=np.float64).flatten(), state.shape[0])
        return state + action, np.ones(8) * 0.1

    state = np.ones(8) * 0.5
    preferred = np.ones(8) * 0.7
    result = ai.decide(state, predict_fn, preferred, exploration_ratio=0.3)
    assert "action" in result
    assert len(result["action"]) == config.action_dim


def test_metacognition(config):
    mc = Metacognition(config)
    conf = mc.compute_confidence(0.5, 0.3)
    assert 0.0 <= conf <= 1.0
    mc.record_outcome(conf, 0.4)
    ctrl = mc.meta_control(0.5, 1.5, 0.3)
    assert "new_exploration_ratio" in ctrl
    assert ctrl["new_exploration_ratio"] >= config.exploration.exploration_ratio_min


def test_memory_system(config):
    ms = MemorySystem(config)
    for i in range(20):
        ms.encode(np.random.rand(8), {"id": i}, importance=float(i) / 20)
    results = ms.retrieve(np.random.rand(8), k=5)
    assert len(results) <= 5
    n = ms.consolidate()
    assert n >= 0
    assert len(ms.slow) >= 0


def test_optimizer(config):
    model = PredictiveModel(config)
    opt = Optimizer(config)
    obs = np.random.rand(8)
    # 微修正模式
    res = opt.optimize(model, obs, 0.1)
    assert res["mode"] == "micro"
    # 结构性重构模式
    res = opt.optimize(model, obs, 2.0)
    assert res["mode"] == "structural"


def test_safety_guard(config):
    sg = SafetyGuard(config)
    assert sg.check_state(np.ones(8) * 0.5)
    assert not sg.check_state(np.ones(8) * 2.0)
    assert sg.check_action(np.ones(4) * 0.5)
    assert not sg.check_action(np.ones(4) * 2.0)
    clipped = sg.clip_action(np.ones(4) * 5.0)
    assert np.all(np.abs(clipped) <= sg.action_limit)
    val = sg.validate(np.ones(8) * 0.5, np.ones(4) * 0.1)
    assert val["safe"]


def test_self_model(config):
    sm = SelfModel(config)
    sm.update_self_representation(np.random.rand(8), 0.7)
    cap = sm.estimate_capability("prediction")
    assert 0.0 <= cap <= 1.0
    check = sm.self_referential_check()
    assert "consistency" in check
    prop = sm.propose_self_modification(1.5)
    assert prop["action"] in ("increase_learning_rate", "adjust_precision", "none")
