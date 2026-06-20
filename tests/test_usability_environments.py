"""阶段二测试：环境适配器与标准接口。

测试范围：
1. Environment 抽象基类接口规范
2. SimpleEnvironment 行为
3. StaticEnvironment 行为
4. FunctionEnvironment 行为
5. GymEnvironment（可选，跳过如未安装）
6. make_environment 工厂函数
7. Agent + Environment 集成
8. to_step_fn 兼容性
9. 上下文管理器（with 语句）
"""
import os
import sys
import tempfile
import numpy as np
import pytest

from spepa.environments import (
    Environment, SimpleEnvironment, StaticEnvironment,
    FunctionEnvironment, GymEnvironment,
    make_environment, list_environments,
)
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_default_config, get_preset_config


# ============================================================
# 抽象基类测试
# ============================================================

class TestEnvironmentInterface:
    """Environment 抽象基类接口规范。"""

    def test_cannot_instantiate_abstract(self):
        """不能直接实例化抽象基类。"""
        with pytest.raises(TypeError):
            Environment()

    def test_subclass_must_implement_all_abstract(self):
        """子类必须实现所有抽象方法。"""

        class IncompleteEnv(Environment):
            def reset(self):
                return np.array([0.0])

        with pytest.raises(TypeError):
            IncompleteEnv()

    def test_complete_subclass_works(self):
        """完整实现的子类可以实例化。"""

        class MinimalEnv(Environment):
            @property
            def observation_dim(self):
                return 4

            @property
            def action_dim(self):
                return 2

            def reset(self):
                return np.zeros(4)

            def step(self, action):
                return np.zeros(4), 0.0, False, {}

        env = MinimalEnv()
        assert env.observation_dim == 4
        assert env.action_dim == 2
        obs = env.reset()
        assert obs.shape == (4,)
        obs, r, d, info = env.step(np.array([0.5, 0.5]))
        assert obs.shape == (4,)
        assert r == 0.0
        assert d is False
        assert isinstance(info, dict)

    def test_close_default_noop(self):
        """close() 默认空实现。"""

        class TestEnv(Environment):
            @property
            def observation_dim(self):
                return 1

            @property
            def action_dim(self):
                return 1

            def reset(self):
                return np.zeros(1)

            def step(self, action):
                return np.zeros(1), 0.0, False, {}

        env = TestEnv()
        env.close()  # 不应抛异常

    def test_render_default_noop(self):
        """render() 默认空实现。"""

        class TestEnv(Environment):
            @property
            def observation_dim(self):
                return 1

            @property
            def action_dim(self):
                return 1

            def reset(self):
                return np.zeros(1)

            def step(self, action):
                return np.zeros(1), 0.0, False, {}

        env = TestEnv()
        env.render()  # 不应抛异常


# ============================================================
# SimpleEnvironment 测试
# ============================================================

class TestSimpleEnvironment:
    """SimpleEnvironment 行为测试。"""

    def test_default_dimensions(self):
        """默认维度为 16/8。"""
        env = SimpleEnvironment()
        assert env.observation_dim == 16
        assert env.action_dim == 8

    def test_custom_dimensions(self):
        """自定义维度。"""
        env = SimpleEnvironment(state_dim=32, action_dim=16)
        assert env.observation_dim == 32
        assert env.action_dim == 16

    def test_reset_returns_initial_state(self):
        """reset 返回全 0.5 的初始状态。"""
        env = SimpleEnvironment(state_dim=4, seed=42)
        obs = env.reset()
        assert obs.shape == (4,)
        assert np.allclose(obs, 0.5)

    def test_reset_resets_step_count(self):
        """reset 重置步数计数。"""
        env = SimpleEnvironment(state_dim=4, action_dim=2, seed=42)
        env.step(np.array([0.5, 0.5]))
        env.step(np.array([0.5, 0.5]))
        assert env._step_count == 2
        env.reset()
        assert env._step_count == 0

    def test_step_returns_correct_shapes(self):
        """step 返回正确形状的元组。"""
        env = SimpleEnvironment(state_dim=8, action_dim=4, seed=42)
        env.reset()
        obs, reward, done, info = env.step(np.array([0.1, 0.2, 0.3, 0.4]))
        assert obs.shape == (8,)
        assert isinstance(reward, float)
        assert done is False
        assert isinstance(info, dict)
        assert "step" in info

    def test_step_state_in_range(self):
        """step 后状态在 [0, 1] 范围内。"""
        env = SimpleEnvironment(state_dim=4, action_dim=2, seed=42, noise_std=0.1)
        env.reset()
        for _ in range(20):
            action = np.random.randn(2) * 5  # 大动作
            obs, _, _, _ = env.step(action)
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_step_reward_is_negative(self):
        """奖励为负（鼓励状态趋零）。"""
        env = SimpleEnvironment(state_dim=4, action_dim=2, seed=42)
        env.reset()
        _, reward, _, _ = env.step(np.array([0.0, 0.0]))
        assert reward < 0  # 状态为 0.5，sum(0.5^2) > 0

    def test_reproducible_with_seed(self):
        """相同种子产生相同序列。"""
        env1 = SimpleEnvironment(state_dim=4, action_dim=2, seed=123)
        env2 = SimpleEnvironment(state_dim=4, action_dim=2, seed=123)
        env1.reset()
        env2.reset()
        action = np.array([0.3, 0.7])
        obs1, _, _, _ = env1.step(action)
        obs2, _, _, _ = env2.step(action)
        assert np.allclose(obs1, obs2)

    def test_action_dim_mismatch_resized(self):
        """动作维度不匹配时自动 resize。"""
        env = SimpleEnvironment(state_dim=4, action_dim=4, seed=42)
        env.reset()
        # 传入维度 2 的动作，应自动 resize 到 4
        obs, _, _, _ = env.step(np.array([0.5, 0.5]))
        assert obs.shape == (4,)


# ============================================================
# StaticEnvironment 测试
# ============================================================

class TestStaticEnvironment:
    """StaticEnvironment 行为测试。"""

    def test_fixed_observation(self):
        """观测始终固定。"""
        obs_val = np.array([0.1, 0.2, 0.3])
        env = StaticEnvironment(obs_val, action_dim=4)
        assert env.observation_dim == 3
        assert env.action_dim == 4
        o1 = env.reset()
        o2, _, _, _ = env.step(np.zeros(4))
        assert np.allclose(o1, obs_val)
        assert np.allclose(o2, obs_val)

    def test_scalar_observation(self):
        """标量观测扩展为 1 维。"""
        env = StaticEnvironment(0.5)
        assert env.observation_dim == 1
        obs = env.reset()
        assert obs.shape == (1,)

    def test_custom_reward(self):
        """自定义固定奖励。"""
        env = StaticEnvironment([0.5, 0.5], action_dim=2, reward=1.5)
        env.reset()
        _, reward, _, _ = env.step(np.zeros(2))
        assert reward == 1.5

    def test_step_count_increments(self):
        """步数计数递增。"""
        env = StaticEnvironment([0.5], action_dim=1)
        env.reset()
        _, _, _, info1 = env.step(np.zeros(1))
        _, _, _, info2 = env.step(np.zeros(1))
        assert info1["step"] == 1
        assert info2["step"] == 2


# ============================================================
# FunctionEnvironment 测试
# ============================================================

class TestFunctionEnvironment:
    """FunctionEnvironment 行为测试。"""

    def test_wraps_step_fn(self):
        """包装现有 env_step_fn。"""
        def my_fn(action):
            return np.clip(action * 2, 0, 1)

        env = FunctionEnvironment(my_fn, observation_dim=4, action_dim=4)
        assert env.observation_dim == 4
        assert env.action_dim == 4
        env.reset()
        obs, _, _, _ = env.step(np.array([0.1, 0.2, 0.3, 0.4]))
        assert np.allclose(obs, [0.2, 0.4, 0.6, 0.8])

    def test_initial_state(self):
        """自定义初始状态。"""
        def my_fn(action):
            return action

        initial = np.array([0.9, 0.8])
        env = FunctionEnvironment(my_fn, observation_dim=2, action_dim=2,
                                  initial_state=initial)
        obs = env.reset()
        assert np.allclose(obs, [0.9, 0.8])

    def test_step_fn_exception_handled(self):
        """step_fn 抛异常时环境保持状态。"""
        def bad_fn(action):
            raise RuntimeError("boom")

        env = FunctionEnvironment(bad_fn, observation_dim=2, action_dim=2)
        env.reset()
        # 不应抛异常
        obs, _, _, _ = env.step(np.array([0.5, 0.5]))
        assert obs.shape == (2,)


# ============================================================
# GymEnvironment 测试（可选）
# ============================================================

class TestGymEnvironment:
    """GymEnvironment 测试（需安装 gymnasium）。"""

    def test_import_error_without_gym(self):
        """未安装 gym 时抛 ImportError。"""
        # 这个测试只在 gym 未安装时有意义
        try:
            import gymnasium  # noqa
            import gym  # noqa
            pytest.skip("gym/gymnasium 已安装，跳过 ImportError 测试")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="gymnasium"):
            GymEnvironment("CartPole-v1")

    def test_gym_environment_basic(self):
        """基本 Gym 环境测试（需安装）。"""
        try:
            import gymnasium  # noqa
        except ImportError:
            try:
                import gym  # noqa
            except ImportError:
                pytest.skip("未安装 gymnasium/gym")

        try:
            env = GymEnvironment("CartPole-v1", seed=42)
        except Exception:
            pytest.skip("无法创建 CartPole-v1 环境")

        obs = env.reset()
        assert obs.ndim == 1
        assert env.observation_dim > 0
        assert env.action_dim > 0
        action = np.zeros(env.action_dim)
        obs2, reward, done, info = env.step(action)
        assert obs2.shape == obs.shape
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        env.close()


# ============================================================
# 工厂函数测试
# ============================================================

class TestMakeEnvironment:
    """make_environment 工厂函数测试。"""

    def test_make_simple(self):
        """创建 SimpleEnvironment。"""
        env = make_environment("simple", state_dim=8, action_dim=4, seed=42)
        assert isinstance(env, SimpleEnvironment)
        assert env.observation_dim == 8

    def test_make_static(self):
        """创建 StaticEnvironment。"""
        env = make_environment("static", observation=[0.5, 0.5], action_dim=2)
        assert isinstance(env, StaticEnvironment)
        assert env.observation_dim == 2

    def test_make_function(self):
        """创建 FunctionEnvironment。"""
        env = make_environment("function",
                               env_step_fn=lambda a: a,
                               observation_dim=4, action_dim=4)
        assert isinstance(env, FunctionEnvironment)

    def test_make_unknown_raises(self):
        """未知类型抛 ValueError。"""
        with pytest.raises(ValueError, match="未知环境类型"):
            make_environment("unknown")

    def test_list_environments(self):
        """list_environments 返回所有类型。"""
        envs = list_environments()
        assert "simple" in envs
        assert "static" in envs
        assert "function" in envs
        assert "gym" in envs


# ============================================================
# to_step_fn 兼容性测试
# ============================================================

class TestToStepFn:
    """to_step_fn() 兼容性测试。"""

    def test_to_step_fn_returns_callable(self):
        """to_step_fn 返回可调用对象。"""
        env = SimpleEnvironment(state_dim=4, action_dim=2, seed=42)
        env.reset()
        fn = env.to_step_fn()
        assert callable(fn)
        result = fn(np.array([0.5, 0.5]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_to_step_fn_only_returns_observation(self):
        """to_step_fn 只返回观测（不含 reward/done/info）。"""
        env = SimpleEnvironment(state_dim=4, action_dim=2, seed=42)
        env.reset()
        fn = env.to_step_fn()
        result = fn(np.array([0.5, 0.5]))
        # 不是元组，是 ndarray
        assert not isinstance(result, tuple)
        assert isinstance(result, np.ndarray)

    def test_to_step_fn_compatible_with_agent(self):
        """to_step_fn 产生的回调可直接传给 Agent.run(env_step_fn=)。"""
        env = SimpleEnvironment(state_dim=16, action_dim=8, seed=42)
        env.reset()
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        fn = env.to_step_fn()
        result = agent.run(n_steps=3, env_step_fn=fn)
        assert result["total_steps"] == 3


# ============================================================
# 上下文管理器测试
# ============================================================

class TestContextManager:
    """with 语句支持测试。"""

    def test_with_statement(self):
        """with 语句自动调用 close。"""
        env = SimpleEnvironment(state_dim=4, action_dim=2, seed=42)
        with env as e:
            assert e is env
            e.reset()
            e.step(np.array([0.5, 0.5]))
        # with 结束后 close 已调用，不应抛异常

    def test_close_idempotent(self):
        """close 可多次调用。"""
        env = SimpleEnvironment(state_dim=4, action_dim=2, seed=42)
        env.close()
        env.close()  # 不应抛异常


# ============================================================
# Agent + Environment 集成测试
# ============================================================

class TestAgentEnvironmentIntegration:
    """Agent 与 Environment 集成测试。"""

    def test_agent_accepts_environment_param(self):
        """Agent.__init__ 接受 environment 参数。"""
        env = SimpleEnvironment(state_dim=16, action_dim=8, seed=42)
        config = get_default_config()
        agent = SPEPAAgent(config, environment=env)
        assert agent._external_env is env

    def test_agent_uses_external_environment(self):
        """Agent 使用外部环境步进。"""
        env = StaticEnvironment(np.zeros(16), action_dim=8)
        config = get_default_config()
        agent = SPEPAAgent(config, environment=env)
        # 运行 3 步，状态应保持为 0（静态环境）
        agent.run(n_steps=3)
        # current_state 应接近 0（静态环境返回全 0）
        assert np.allclose(agent.current_state, 0.0, atol=1e-6)

    def test_agent_without_environment_uses_default(self):
        """不传 environment 时使用默认动力学。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        assert agent._external_env is None
        # 运行不应抛异常
        agent.run(n_steps=3)

    def test_agent_with_simple_environment(self):
        """Agent + SimpleEnvironment 完整运行。"""
        env = SimpleEnvironment(state_dim=16, action_dim=8, seed=42)
        config = get_default_config()
        agent = SPEPAAgent(config, environment=env)
        result = agent.run(n_steps=5)
        assert result["total_steps"] == 5
        # 状态应在 [0, 1] 范围内
        assert np.all(agent.current_state >= 0.0)
        assert np.all(agent.current_state <= 1.0)

    def test_agent_with_function_environment(self):
        """Agent + FunctionEnvironment 向后兼容。"""
        # 模拟旧的 env_step_fn
        def old_fn(action):
            action = np.asarray(action).flatten()
            return np.clip(np.ones(16) * 0.5 + action * 0.1, 0, 1)

        env = FunctionEnvironment(old_fn, observation_dim=16, action_dim=8)
        config = get_default_config()
        agent = SPEPAAgent(config, environment=env)
        result = agent.run(n_steps=3)
        assert result["total_steps"] == 3

    def test_external_env_overridden_by_run_arg(self):
        """run(env_step_fn=) 优先于注入的 environment。"""
        env = StaticEnvironment(np.zeros(16), action_dim=8)
        config = get_default_config()
        agent = SPEPAAgent(config, environment=env)

        # 传入自定义 env_step_fn，应覆盖注入的环境
        def custom_fn(action):
            return np.ones(16) * 0.9

        agent.run(n_steps=2, env_step_fn=custom_fn)
        # current_state 应为 0.9（来自 custom_fn），而非 0（来自静态环境）
        assert np.allclose(agent.current_state, 0.9, atol=0.01)

    def test_environment_with_autonomous_run(self):
        """环境注入与自主驱动兼容。"""
        env = SimpleEnvironment(state_dim=16, action_dim=8, seed=42)
        config = get_default_config()
        agent = SPEPAAgent(config, environment=env)
        result = agent.run_autonomous(n_steps=3)
        assert result["total_steps"] >= 1


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端工作流测试。"""

    def test_full_workflow(self):
        """完整工作流：创建环境 -> 注入 Agent -> 运行 -> 检查结果。"""
        # 1. 创建环境
        env = make_environment("simple", state_dim=16, action_dim=8, seed=42)
        initial_obs = env.reset()
        assert initial_obs.shape == (16,)

        # 2. 创建 Agent 并注入环境
        config = get_default_config()
        agent = SPEPAAgent(config, environment=env)

        # 3. 运行
        result = agent.run(n_steps=10)
        assert result["total_steps"] == 10

        # 4. 验证状态在合理范围
        assert np.all(np.isfinite(agent.current_state))

    def test_environment_swapping(self):
        """运行中切换环境（通过 run 参数）。"""
        config = get_default_config()
        agent = SPEPAAgent(config)

        # 第一次用默认环境运行
        agent.run(n_steps=3)

        # 第二次用静态环境运行
        env = StaticEnvironment(np.zeros(16), action_dim=8)
        agent.run(n_steps=3, env_step_fn=env.to_step_fn())
        assert np.allclose(agent.current_state, 0.0, atol=1e-6)

    def test_multiple_agents_share_environment(self):
        """多个 Agent 可以共享同一环境类型（各自独立实例）。"""
        config = get_preset_config("minimal")

        env1 = SimpleEnvironment(state_dim=8, action_dim=4, seed=1)
        env2 = SimpleEnvironment(state_dim=8, action_dim=4, seed=2)

        agent1 = SPEPAAgent(config, environment=env1)
        agent2 = SPEPAAgent(config, environment=env2)

        agent1.run(n_steps=3)
        agent2.run(n_steps=3)

        # 两个 Agent 状态不同（不同种子）
        assert not np.allclose(agent1.current_state, agent2.current_state)
