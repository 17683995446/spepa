"""SPEPA 降级策略矩阵：结构化的降级级别与参数映射。

设计原则（第一性原理）：
- 分级降级原理：降级应是渐进的，而非跳变
- 可逆性原理：降级和恢复都应平滑过渡
- 优先级原理：非关键功能先降级，核心功能最后降级
- 显式性原理：每个降级级别的行为参数应明确指定

降级级别：
- L0_NORMAL: 全速运行，所有功能开启
- L1_MILD: 轻度降级，减少策略数，降低探索
- L2_MODERATE: 中度降级，跳过离线复盘，减少学习频率
- L3_SEVERE: 严重降级，跳过记忆巩固，最小化学习
- L4_CRITICAL: 临界模式，只保留感知+决策+行动
"""
from enum import IntEnum
from typing import Dict, Any
from dataclasses import dataclass

from spepa.utils.logger import LoggedMixin


class DegradationLevel(IntEnum):
    """降级级别。"""
    L0_NORMAL = 0      # 正常
    L1_MILD = 1        # 轻度
    L2_MODERATE = 2    # 中度
    L3_SEVERE = 3      # 严重
    L4_CRITICAL = 4    # 临界


@dataclass
class DegradationParams:
    """降级参数：每个级别的具体行为配置。"""
    level: DegradationLevel
    n_policies: int            # 策略数量
    exploration_ratio: float   # 探索比例
    learning_enabled: bool     # 是否学习
    learning_interval: int     # 学习间隔（每N步学一次）
    offline_enabled: bool      # 是否离线复盘
    memory_consolidation: bool # 是否记忆巩固
    memory_step_interval: int  # 记忆步进间隔
    skill_learning: bool       # 是否技能学习
    multiagent_enabled: bool   # 是否多Agent
    checkpoint_interval: int   # 检查点间隔（步）


class DegradationMatrix(LoggedMixin):
    """降级策略矩阵：定义每个降级级别的行为参数。

    提供从资源使用率/背压级别到降级参数的映射。
    """

    # 降级参数表：每个级别的完整配置
    MATRIX: Dict[DegradationLevel, DegradationParams] = {
        DegradationLevel.L0_NORMAL: DegradationParams(
            level=DegradationLevel.L0_NORMAL,
            n_policies=8, exploration_ratio=0.3,
            learning_enabled=True, learning_interval=1,
            offline_enabled=True, memory_consolidation=True,
            memory_step_interval=1, skill_learning=True,
            multiagent_enabled=True, checkpoint_interval=50,
        ),
        DegradationLevel.L1_MILD: DegradationParams(
            level=DegradationLevel.L1_MILD,
            n_policies=6, exploration_ratio=0.2,
            learning_enabled=True, learning_interval=1,
            offline_enabled=True, memory_consolidation=True,
            memory_step_interval=1, skill_learning=True,
            multiagent_enabled=True, checkpoint_interval=50,
        ),
        DegradationLevel.L2_MODERATE: DegradationParams(
            level=DegradationLevel.L2_MODERATE,
            n_policies=4, exploration_ratio=0.15,
            learning_enabled=True, learning_interval=2,
            offline_enabled=False, memory_consolidation=True,
            memory_step_interval=2, skill_learning=False,
            multiagent_enabled=True, checkpoint_interval=100,
        ),
        DegradationLevel.L3_SEVERE: DegradationParams(
            level=DegradationLevel.L3_SEVERE,
            n_policies=2, exploration_ratio=0.1,
            learning_enabled=True, learning_interval=5,
            offline_enabled=False, memory_consolidation=False,
            memory_step_interval=5, skill_learning=False,
            multiagent_enabled=False, checkpoint_interval=200,
        ),
        DegradationLevel.L4_CRITICAL: DegradationParams(
            level=DegradationLevel.L4_CRITICAL,
            n_policies=1, exploration_ratio=0.05,
            learning_enabled=False, learning_interval=999,
            offline_enabled=False, memory_consolidation=False,
            memory_step_interval=999, skill_learning=False,
            multiagent_enabled=False, checkpoint_interval=500,
        ),
    }

    def __init__(self):
        self._set_log_name("spepa.resource.degradation")
        self._current_level: DegradationLevel = DegradationLevel.L0_NORMAL
        self.log.info("DegradationMatrix initialized (5 levels)")

    def get_params(self, level: DegradationLevel) -> DegradationParams:
        """获取指定级别的降级参数。"""
        return self.MATRIX[level]

    def get_current_params(self) -> DegradationParams:
        """获取当前级别的降级参数。"""
        return self.MATRIX[self._current_level]

    def set_level(self, level: DegradationLevel) -> DegradationParams:
        """设置当前降级级别。"""
        if level != self._current_level:
            self.log.info("Degradation level: %s -> %s",
                          self._current_level.name, level.name)
            self._current_level = level
        return self.get_current_params()

    def level_from_throttle(self, throttle: float) -> DegradationLevel:
        """从节流信号（0-1）映射到降级级别。"""
        if throttle < 0.15:
            return DegradationLevel.L0_NORMAL
        elif throttle < 0.35:
            return DegradationLevel.L1_MILD
        elif throttle < 0.6:
            return DegradationLevel.L2_MODERATE
        elif throttle < 0.85:
            return DegradationLevel.L3_SEVERE
        else:
            return DegradationLevel.L4_CRITICAL

    def level_from_load(self, load: float, limit: float = 1.0) -> DegradationLevel:
        """从负载率（0-1）和限制映射到降级级别。

        参数：
        - load: 当前负载（0-1）
        - limit: 负载限制（0-1），超过则降级
        """
        ratio = load / max(limit, 1e-6)
        if ratio < 0.7:
            return DegradationLevel.L0_NORMAL
        elif ratio < 0.9:
            return DegradationLevel.L1_MILD
        elif ratio < 1.0:
            return DegradationLevel.L2_MODERATE
        elif ratio < 1.2:
            return DegradationLevel.L3_SEVERE
        else:
            return DegradationLevel.L4_CRITICAL

    def update_from_throttle(self, throttle: float) -> DegradationParams:
        """根据节流信号更新降级级别并返回参数。"""
        level = self.level_from_throttle(throttle)
        return self.set_level(level)

    def reset(self) -> None:
        """重置到正常级别。"""
        self._current_level = DegradationLevel.L0_NORMAL
        self.log.info("DegradationMatrix reset to L0_NORMAL")

    @property
    def current_level(self) -> DegradationLevel:
        return self._current_level

    def get_state(self) -> Dict[str, Any]:
        params = self.get_current_params()
        return {
            "current_level": self._current_level.name,
            "n_policies": params.n_policies,
            "learning_enabled": params.learning_enabled,
            "offline_enabled": params.offline_enabled,
            "memory_consolidation": params.memory_consolidation,
            "skill_learning": params.skill_learning,
            "multiagent_enabled": params.multiagent_enabled,
        }
