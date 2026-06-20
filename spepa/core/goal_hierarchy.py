"""SPEPA 目标层次体系（五层 L0-L4），含冲突仲裁。

层级：
- L0: 生存/安全（最高优先级，刚性）
- L1: 内稳态维持
- L2: 任务/收益目标
- L3: 探索/学习目标
- L4: 元目标（自我改进、好奇心）（最低优先级）
"""
import numpy as np
from typing import List, Dict, Any, Optional

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class Goal:
    """单个目标。"""

    def __init__(self, level: int, name: str, target: np.ndarray,
                 priority: float = 1.0, rigid: bool = False):
        self.level = level
        self.name = name
        self.target = np.asarray(target, dtype=np.float64).flatten()
        self.priority = priority
        self.rigid = rigid  # 刚性目标不可被压制
        self.active = True
        self.progress = 0.0

    def distance(self, state: np.ndarray) -> float:
        """当前状态到目标的距离。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        return float(np.linalg.norm(state - self.target))

    def update_progress(self, state: np.ndarray) -> float:
        """更新进度（0~1）。"""
        d = self.distance(state)
        self.progress = float(np.exp(-d))
        return self.progress

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "name": self.name,
            "priority": self.priority,
            "rigid": self.rigid,
            "active": self.active,
            "progress": self.progress,
        }


class GoalHierarchy(LoggedMixin):
    """五层目标层次体系，含冲突仲裁。"""

    LEVEL_NAMES = ["L0_survival", "L1_homeostasis", "L2_task", "L3_explore", "L4_meta"]

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.goal_hierarchy")
        self.config = config
        self.dim = config.state_dim
        self.rng = np.random.default_rng(config.seed)  # 种子化 RNG
        self.goals: Dict[int, List[Goal]] = {i: [] for i in range(5)}
        self._init_default_goals()
        self.log.info("GoalHierarchy initialized with %d levels", 5)

    def _init_default_goals(self) -> None:
        """初始化默认目标。"""
        # L0 生存：状态保持在安全区间
        self.add_goal(Goal(0, "survival", np.ones(self.dim) * 0.5, priority=10.0, rigid=True))
        # L1 内稳态
        self.add_goal(Goal(1, "homeostasis", np.ones(self.dim) * 0.5, priority=5.0))
        # L2 任务目标（默认居中）
        self.add_goal(Goal(2, "task_default", np.ones(self.dim) * 0.7, priority=3.0))
        # L3 探索（使用种子化 RNG，保证可复现）
        self.add_goal(Goal(3, "explore", self.rng.random(self.dim), priority=1.0))
        # L4 元目标
        self.add_goal(Goal(4, "self_improve", np.ones(self.dim) * 0.9, priority=0.5))

    def add_goal(self, goal: Goal) -> None:
        """添加目标。"""
        self.goals[goal.level].append(goal)
        self.log.info("Goal added L%d '%s' priority=%.2f rigid=%s",
                      goal.level, goal.name, goal.priority, goal.rigid)

    def remove_goal(self, level: int, name: str) -> bool:
        """移除目标。"""
        before = len(self.goals[level])
        self.goals[level] = [g for g in self.goals[level] if g.name != name]
        removed = len(self.goals[level]) < before
        self.log.info("Goal removed L%d '%s' success=%s", level, name, removed)
        return removed

    def active_goals(self) -> List[Goal]:
        """返回所有激活的目标（按层级排序）。"""
        out = []
        for lvl in range(5):
            for g in self.goals[lvl]:
                if g.active:
                    out.append(g)
        return out

    def arbitrate(self, state: np.ndarray) -> Dict[str, Any]:
        """冲突仲裁：根据层级优先级和当前距离选择主导目标。

        刚性目标（L0）若距离过大则直接主导。
        """
        state = np.asarray(state, dtype=np.float64).flatten()
        # 更新所有目标进度
        for g in self.active_goals():
            g.update_progress(state)

        # 检查 L0 刚性目标
        dominant: Optional[Goal] = None
        for g in self.goals[0]:
            if g.active and g.distance(state) > self.config.safety_hard_boundary:
                dominant = g
                self.log.warning("Rigid L0 goal '%s' dominates (distance=%.4f)",
                                 g.name, g.distance(state))
                break

        if dominant is None:
            # 否则按 (层级权重 * 距离紧迫度) 选择
            best_score = -np.inf
            for g in self.active_goals():
                urgency = g.distance(state)  # 距离越远越紧迫
                score = (5 - g.level) * g.priority * urgency
                if score > best_score:
                    best_score = score
                    dominant = g
            self.log.debug("Arbitrated dominant goal: '%s' (score=%.4f)",
                           dominant.name if dominant else None, best_score)

        result = {
            "dominant_goal": dominant.name if dominant else None,
            "dominant_level": dominant.level if dominant else None,
            "dominant_target": dominant.target.tolist() if dominant else None,
            "dominant_distance": dominant.distance(state) if dominant else 0.0,
        }
        return result

    def set_task_goal(self, target: np.ndarray, name: str = "task_active") -> None:
        """设置当前任务目标（L2）。"""
        self.goals[2] = [g for g in self.goals[2] if g.name != name]
        self.add_goal(Goal(2, name, target, priority=3.0))
        self.log.info("Task goal set: '%s'", name)

    def get_state(self) -> Dict[str, Any]:
        return {
            "goals": {lvl: [g.to_dict() for g in gs] for lvl, gs in self.goals.items()},
            "n_active": len(self.active_goals()),
        }
