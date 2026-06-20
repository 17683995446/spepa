"""SPEPA 环境适配器与标准接口。

设计原则（第一性原理）：
- 标准接口：所有环境遵循统一的 reset/step 协议，与 RL 社区惯例一致
- 向后兼容：通过 to_step_fn() 将 Environment 转为现有 env_step_fn 回调
- 渐进式接入：内置简单环境即可用，Gym 等外部库按需导入
- 零依赖：核心接口与内置环境不依赖第三方库

标准接口（Environment）：
    reset() -> observation
    step(action) -> (observation, reward, done, info)

内置环境：
    - SimpleEnvironment : 线性动力学 + 噪声（SPEPA 默认环境的独立封装）
    - StaticEnvironment : 固定观测（用于测试/调试）
    - FunctionEnvironment : 包装现有 env_step_fn 回调
    - GymEnvironment : 适配 OpenAI Gym/Gymnasium（可选依赖）
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Callable, Union

import numpy as np

from spepa.utils.logger import get_logger

logger = get_logger("spepa.environments")


# ============================================================
# 标准环境接口
# ============================================================

class Environment(ABC):
    """SPEPA 标准环境接口。

    所有环境必须实现：
    - observation_dim : 观测维度
    - action_dim : 动作维度
    - reset() : 重置环境，返回初始观测
    - step(action) : 执行动作，返回 (obs, reward, done, info)

    可选实现：
    - close() : 关闭环境，释放资源
    - render() : 渲染环境（可视化）
    """

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """观测向量维度。"""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """动作向量维度。"""
        ...

    @abstractmethod
    def reset(self) -> np.ndarray:
        """重置环境到初始状态，返回初始观测。

        返回：
            np.ndarray : 初始观测向量，shape=(observation_dim,)
        """
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一个动作，返回 (observation, reward, done, info)。

        参数：
            action : 动作向量，shape=(action_dim,)

        返回：
            observation : 新观测向量
            reward : 即时奖励
            done : 是否终止
            info : 额外信息字典
        """
        ...

    def close(self) -> None:
        """关闭环境，释放资源（默认空实现）。"""
        pass

    def render(self) -> None:
        """渲染环境（默认空实现）。"""
        pass

    def to_step_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """转换为 env_step_fn 回调，兼容现有 Agent 接口。

        返回的回调签名：env_step_fn(action) -> next_state
        注意：回调只返回 next_state（即 observation），不返回 reward/done/info。
        如需完整反馈，请直接使用 step() 方法。

        返回：
            Callable[[np.ndarray], np.ndarray] : 环境步进函数
        """
        def env_step_fn(action: np.ndarray) -> np.ndarray:
            obs, _reward, _done, _info = self.step(action)
            return obs
        return env_step_fn

    def __enter__(self):
        """支持 with 语句。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 语句时自动关闭。"""
        self.close()
        return False


# ============================================================
# 内置环境
# ============================================================

class SimpleEnvironment(Environment):
    """简单线性动力学环境（SPEPA 默认环境的独立封装）。

    动力学：next_state = clip(current_state + gain * action + noise, 0, 1)
    奖励：-sum(state^2)（鼓励状态趋近零）

    这是 SPEPA Agent 内置默认环境的等价实现，提取为独立类后可复用、可配置。
    """

    def __init__(self,
                 state_dim: int = 16,
                 action_dim: int = 8,
                 noise_std: float = 0.05,
                 gain: float = 0.3,
                 seed: Optional[int] = None):
        """初始化简单环境。

        参数：
            state_dim : 状态/观测维度
            action_dim : 动作维度
            noise_std : 高斯噪声标准差
            gain : 动作对状态的影响系数
            seed : 随机种子（None 则不固定）
        """
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._noise_std = noise_std
        self._gain = gain
        self._rng = np.random.default_rng(seed)
        self._state = np.ones(state_dim) * 0.5
        self._step_count = 0

    @property
    def observation_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self) -> np.ndarray:
        """重置到初始状态（全 0.5）。"""
        self._state = np.ones(self._state_dim) * 0.5
        self._step_count = 0
        return self._state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作，返回 (obs, reward, done, info)。"""
        action = np.asarray(action, dtype=np.float64).flatten()
        # 动力学计算需要 action 对齐到 state_dim
        if action.shape[0] != self._state_dim:
            action = np.resize(action, self._state_dim)
        noise = self._rng.normal(0, self._noise_std, self._state_dim)
        self._state = np.clip(
            self._state + self._gain * action + noise, 0.0, 1.0
        )
        self._step_count += 1
        reward = -float(np.sum(self._state ** 2))
        info = {"step": self._step_count}
        return self._state.copy(), reward, False, info


class StaticEnvironment(Environment):
    """静态环境，始终返回固定观测。

    用于测试、调试、基准测试等场景。环境不随动作变化。
    """

    def __init__(self,
                 observation: Union[np.ndarray, list, float],
                 action_dim: int = 8,
                 reward: float = 0.0):
        """初始化静态环境。

        参数：
            observation : 固定观测值（标量会被扩展为 1 维数组）
            action_dim : 动作维度
            reward : 每步固定奖励
        """
        obs = np.asarray(observation, dtype=np.float64).flatten()
        if obs.shape[0] == 0:
            obs = np.array([0.0])
        self._obs = obs
        self._action_dim = action_dim
        self._reward = reward
        self._step_count = 0

    @property
    def observation_dim(self) -> int:
        return self._obs.shape[0]

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return self._obs.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._step_count += 1
        return self._obs.copy(), self._reward, False, {"step": self._step_count}


class FunctionEnvironment(Environment):
    """把现有的 env_step_fn 回调包装成标准 Environment 接口。

    用于向后兼容：用户已有 env_step_fn(action) -> next_state 函数，
    可用此类包装后作为 Environment 传入 Agent。
    """

    def __init__(self,
                 env_step_fn: Callable[[np.ndarray], np.ndarray],
                 observation_dim: int,
                 action_dim: int,
                 initial_state: Optional[np.ndarray] = None):
        """初始化函数环境。

        参数：
            env_step_fn : 环境步进函数，输入 action 返回 next_state
            observation_dim : 观测维度
            action_dim : 动作维度
            initial_state : 初始状态（None 则用全 0.5）
        """
        self._fn = env_step_fn
        self._obs_dim = observation_dim
        self._action_dim = action_dim
        if initial_state is not None:
            self._state = np.asarray(initial_state, dtype=np.float64).flatten().copy()
        else:
            self._state = np.ones(observation_dim) * 0.5
        self._step_count = 0

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self) -> np.ndarray:
        self._step_count = 0
        return self._state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        try:
            next_state = self._fn(action)
            self._state = np.asarray(next_state, dtype=np.float64).flatten()
        except Exception as e:
            logger.warning("FunctionEnvironment step failed: %s", e)
        self._step_count += 1
        return self._state.copy(), 0.0, False, {"step": self._step_count}


class GymEnvironment(Environment):
    """OpenAI Gym / Gymnasium 环境适配器。

    需要 pip install gymnasium（或 gym）。
    自动适配 gym/gymnasium 两种 API（4 元组 / 5 元组）。
    """

    def __init__(self, env_name: str, seed: Optional[int] = None, **kwargs):
        """初始化 Gym 环境。

        参数：
            env_name : 环境名称（如 "CartPole-v1", "MountainCar-v0"）
            seed : 随机种子
            **kwargs : 传递给 gym.make 的额外参数
        """
        try:
            import gymnasium as gym
            self._gym_version = "gymnasium"
        except ImportError:
            try:
                import gym as gym
                self._gym_version = "gym"
            except ImportError:
                raise ImportError(
                    "使用 GymEnvironment 需要安装 gymnasium 或 gym。\n"
                    "请运行: pip install gymnasium"
                )
        self._env_name = env_name
        self._env = gym.make(env_name, **kwargs)
        self._seed = seed
        self._obs = self._reset_env(seed)
        logger.info("GymEnvironment created: %s (%s)", env_name, self._gym_version)

    def _reset_env(self, seed: Optional[int] = None) -> np.ndarray:
        """重置 Gym 环境并返回观测。"""
        if self._gym_version == "gymnasium":
            result = self._env.reset(seed=seed) if seed is not None else self._env.reset()
            obs = result[0] if isinstance(result, tuple) else result
        else:
            if seed is not None:
                self._env.seed(seed)
            obs = self._env.reset()
        return np.asarray(obs, dtype=np.float64).flatten()

    @property
    def observation_dim(self) -> int:
        return self._obs.shape[0]

    @property
    def action_dim(self) -> int:
        space = self._env.action_space
        if hasattr(space, 'shape') and space.shape:
            return int(np.prod(space.shape))
        elif hasattr(space, 'n'):
            return int(space.n)
        return 1

    def reset(self) -> np.ndarray:
        self._obs = self._reset_env()
        return self._obs.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # 适配 Discrete 动作空间：取 argmax
        space = self._env.action_space
        if hasattr(space, 'n') and not hasattr(space, 'shape'):
            action = int(np.argmax(action)) if action.ndim > 0 else int(action)
        result = self._env.step(action)
        if len(result) == 5:  # gymnasium: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # gym: (obs, reward, done, info)
            obs, reward, done, info = result
        self._obs = np.asarray(obs, dtype=np.float64).flatten()
        info = info if isinstance(info, dict) else {}
        return self._obs.copy(), float(reward), bool(done), info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def render(self) -> None:
        try:
            self._env.render()
        except Exception as e:
            logger.warning("Render failed: %s", e)


# ============================================================
# 工具函数
# ============================================================

def make_environment(env_type: str = "simple", **kwargs) -> Environment:
    """工厂函数：创建环境实例。

    参数：
        env_type : 环境类型
            - "simple" : SimpleEnvironment（默认）
            - "static" : StaticEnvironment
            - "function" : FunctionEnvironment（需提供 env_step_fn）
            - "gym" : GymEnvironment（需提供 env_name）
        **kwargs : 传递给环境构造函数的参数

    返回：
        Environment 实例

    示例：
        >>> env = make_environment("simple", state_dim=8, seed=42)
        >>> env = make_environment("gym", env_name="CartPole-v1")
    """
    env_type = env_type.lower()
    factories = {
        "simple": SimpleEnvironment,
        "static": StaticEnvironment,
        "function": FunctionEnvironment,
        "gym": GymEnvironment,
    }
    if env_type not in factories:
        raise ValueError(
            f"未知环境类型: {env_type}。可用: {', '.join(factories.keys())}"
        )
    return factories[env_type](**kwargs)


def list_environments() -> Dict[str, str]:
    """列出所有可用环境类型及描述。"""
    return {
        "simple": "简单线性动力学环境（默认，无外部依赖）",
        "static": "静态环境，固定观测（用于测试/调试）",
        "function": "函数环境，包装现有 env_step_fn 回调",
        "gym": "OpenAI Gym/Gymnasium 适配器（需安装 gymnasium）",
    }
