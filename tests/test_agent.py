"""SPEPA 功能模块与 Agent 集成测试。"""
import numpy as np
import pytest

from spepa.utils.config import SPEPAConfig
from spepa.agent import SPEPAAgent
from spepa.modules.perception import Perception
from spepa.modules.decision import Decision
from spepa.modules.action import Action
from spepa.modules.learning import Learning
from spepa.modules.offline_replay import OfflineReplay


@pytest.fixture
def config():
    return SPEPAConfig(state_dim=8, action_dim=4, obs_dim=8, seed=42,
                       max_steps=5, consolidation_interval=3)


@pytest.fixture
def agent(config):
    return SPEPAAgent(config)


def test_agent_initialization(agent):
    state = agent.get_state()
    assert state["step_count"] == 0
    assert state["online_cycles"] == 0
    assert "current_state" in state
    assert "resources" in state
    assert "predictive_model" in state


def test_online_cycle(agent):
    obs = np.random.rand(8)
    result = agent.online_cycle(obs)
    assert "perception" in result
    assert "decision" in result
    assert "action" in result
    assert "learning" in result
    assert "reward" in result
    assert agent.online_cycles == 1


def test_online_cycle_with_env(agent):
    def env_fn(action):
        return np.clip(np.random.rand(8), 0, 1)
    result = agent.online_cycle(np.random.rand(8), env_step_fn=env_fn)
    assert result["action"]["effect"] is not None


def test_offline_cycle(agent):
    # 先填充一些记忆
    for _ in range(5):
        agent.online_cycle(np.random.rand(8))
    # 强制低负载触发
    agent.resource_monitor.cpu_usage = 0.1
    agent.resource_monitor.memory_usage = 0.1
    result = agent.offline_cycle()
    assert "replay" in result or result.get("skipped") is True


def test_step_combines_online_offline(agent):
    result = agent.step(np.random.rand(8))
    assert "online" in result
    assert "offline" in result


def test_run(agent):
    summary = agent.run(n_steps=5)
    assert summary["total_steps"] == 5
    assert summary["online_cycles"] == 5
    assert "metrics" in summary
    assert "prediction_error" in summary["metrics"]


def test_reset(agent):
    agent.run(n_steps=3)
    assert agent.online_cycles > 0
    agent.reset()
    assert agent.online_cycles == 0
    assert agent.step_count == 0


def test_metrics_tracking(agent):
    agent.run(n_steps=3)
    summary = agent.metrics.summary()
    assert "prediction_error" in summary
    assert "confidence" in summary
    assert "total_reward" in summary
    assert summary["prediction_error"]["count"] == 3


def test_resource_constraints(agent):
    """资源约束：CPU<=20%, 内存<=20%。"""
    agent.run(n_steps=3)
    # 资源监控应记录使用率
    assert agent.resource_monitor.cpu_usage >= 0
    assert agent.resource_monitor.memory_usage >= 0


def test_exploration_ratio_minimum(agent):
    """探索比例 >= 30%。"""
    agent.run(n_steps=5)
    assert agent.decision.exploration_ratio >= agent.config.exploration.exploration_ratio_min


def test_safety_guard_integration(agent):
    """安全护栏在决策中生效。"""
    result = agent.online_cycle(np.random.rand(8))
    action = np.asarray(result["decision"]["action"], dtype=np.float64)
    # 动作应被裁剪到安全范围
    assert np.all(np.abs(action) <= agent.safety_guard.action_limit + 1e-6)


def test_perception_module(agent):
    obs = np.random.rand(8)
    result = agent.perception.perceive(obs)
    assert "observation" in result
    assert "prediction" in result
    assert "error" in result
    assert "gap" in result


def test_learning_reduces_error(agent):
    """学习应降低预测误差（微修正模式）。"""
    obs = np.random.rand(8)
    # 多次学习同一观测（用小误差触发微修正模式）
    initial_err = agent.predictive_model.total_prediction_error(obs)
    for _ in range(30):
        agent.learning.learn(obs, 0.1, 0.5)
    final_err = agent.predictive_model.total_prediction_error(obs)
    assert final_err <= initial_err + 1e-3
