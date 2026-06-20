"""SPEPA 自主目标生成器：从驱力、好奇心、误差信号生成目标。

设计原则（第一性原理）：
- 内驱力原理：目标应由系统内部状态驱动，而非外部指令
- 好奇心原理：预测误差高的区域应产生探索目标
- 内稳态原理：偏离内稳态时应产生恢复目标
- 技能缺口原理：未掌握的技能应产生学习目标
- 多样性原理：目标应有多样性，避免陷入局部最优

目标生成来源：
1. 内稳态恢复：homeostasis 偏离大时，生成恢复目标
2. 好奇心探索：预测误差高的状态区域，生成探索目标
3. 技能缺口：成功率低的技能，生成练习目标
4. 自我改进：元认知检测到过自信/欠自信时，生成校准目标
"""
import time
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class GoalSource(Enum):
    """目标来源。"""
    HOMEOSTASIS = "homeostasis"    # 内稳态恢复
    CURIOSITY = "curiosity"        # 好奇心探索
    SKILL_GAP = "skill_gap"        # 技能缺口
    SELF_IMPROVE = "self_improve"  # 自我改进
    EXTERNAL = "external"          # 外部指令


class GoalStatus(Enum):
    """目标状态。"""
    PENDING = "pending"      # 待执行
    ACTIVE = "active"        # 执行中
    ACHIEVED = "achieved"    # 已达成
    FAILED = "failed"        # 失败
    ABANDONED = "abandoned"  # 放弃


@dataclass
class AutonomousGoal:
    """自主生成的目标。"""
    goal_id: str
    source: GoalSource
    target_state: np.ndarray
    priority: float
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    # 达成阈值（距离小于此值视为达成）
    achievement_threshold: float = 0.3
    # 最大尝试步数
    max_steps: int = 100
    # 已执行步数
    steps_taken: int = 0
    # 失败次数
    failure_count: int = 0

    def distance(self, current_state: np.ndarray) -> float:
        """当前状态到目标的距离。"""
        return float(np.linalg.norm(
            np.asarray(current_state, dtype=np.float64).flatten() - self.target_state
        ))

    def is_achieved(self, current_state: np.ndarray) -> bool:
        """是否已达成。"""
        return self.distance(current_state) < self.achievement_threshold

    def is_expired(self) -> bool:
        """是否已超时。"""
        return self.steps_taken >= self.max_steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "source": self.source.value,
            "target_state": self.target_state.tolist(),
            "priority": self.priority,
            "description": self.description,
            "status": self.status.value,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "distance": self.distance(np.zeros_like(self.target_state)),
        }


class GoalGenerator(LoggedMixin):
    """自主目标生成器。

    从系统内部状态生成目标，驱动自主行为。
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.autonomous.goal_generator")
        self.config = config
        self._goal_counter = 0
        # 生成历史
        self._history: List[AutonomousGoal] = []
        # 各来源的生成间隔（避免过于频繁）
        # 初始化为较大负值，确保首次调用（step=0）即能生成
        self._last_generation: Dict[GoalSource, float] = {
            src: -1e9 for src in GoalSource
        }
        # 生成间隔（步）
        self.generation_intervals = {
            GoalSource.HOMEOSTASIS: 10,
            GoalSource.CURIOSITY: 20,
            GoalSource.SKILL_GAP: 50,
            GoalSource.SELF_IMPROVE: 100,
        }
        # 阈值
        self.homeostasis_threshold = 0.5      # 内稳态偏离阈值
        self.curiosity_threshold = 1.0        # 预测误差阈值
        self.skill_gap_threshold = 0.4        # 技能成功率阈值
        self.overconfidence_threshold = 0.1   # 过自信校准偏移阈值
        self.log.info("GoalGenerator initialized")

    def _next_goal_id(self) -> str:
        """生成唯一目标ID。"""
        self._goal_counter += 1
        return f"goal_{self._goal_counter:06d}"

    def generate_from_homeostasis(self, agent: Any) -> Optional[AutonomousGoal]:
        """从内稳态偏离生成恢复目标。

        当内稳态偏离超过阈值时，生成目标将状态拉回内稳态目标。
        """
        deviation = agent.drive_engine.homeostasis.deviation()
        if deviation < self.homeostasis_threshold:
            return None
        # 目标 = 内稳态目标值
        target = agent.drive_engine.homeostasis.target.copy()
        priority = min(10.0, deviation * 5.0)  # 偏离越大优先级越高
        goal = AutonomousGoal(
            goal_id=self._next_goal_id(),
            source=GoalSource.HOMEOSTASIS,
            target_state=target,
            priority=priority,
            description=f"homeostasis_recovery(dev={deviation:.3f})",
            achievement_threshold=0.2,
            max_steps=50,
        )
        self.log.info("Generated homeostasis goal: %s (dev=%.3f, priority=%.1f)",
                      goal.goal_id, deviation, priority)
        return goal

    def generate_from_curiosity(self, agent: Any) -> Optional[AutonomousGoal]:
        """从预测误差生成探索目标。

        当预测误差高时，生成目标探索高误差区域。
        """
        # 获取最近的预测误差
        recent_errors = agent.metrics.get_recent("prediction_error", n=5)
        if not recent_errors:
            return None
        avg_error = float(np.mean(recent_errors))
        if avg_error < self.curiosity_threshold:
            return None
        # 目标 = 当前状态 + 朝高误差方向的扰动
        current = agent.current_state.copy()
        # GapInfo 只携带标量 KL 值，不包含方向信息；
        # 好奇心探索使用随机方向（误差越大，探索范围越大）
        direction = np.random.randn(*current.shape)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.zeros_like(current)
        # 扰动幅度与误差成正比
        scale = min(0.5, avg_error * 0.2)
        target = np.clip(current + direction * scale, 0.0, 1.0)
        priority = min(5.0, avg_error * 2.0)
        goal = AutonomousGoal(
            goal_id=self._next_goal_id(),
            source=GoalSource.CURIOSITY,
            target_state=target,
            priority=priority,
            description=f"curiosity_explore(err={avg_error:.3f})",
            achievement_threshold=0.3,
            max_steps=30,
        )
        self.log.info("Generated curiosity goal: %s (err=%.3f, priority=%.1f)",
                      goal.goal_id, avg_error, priority)
        return goal

    def generate_from_skill_gap(self, agent: Any) -> Optional[AutonomousGoal]:
        """从技能缺口生成练习目标。

        当存在成功率低的技能时，生成目标练习该技能的触发状态。
        """
        skills = list(agent.skill_learner.library.skills.values())
        if not skills:
            return None
        # 找成功率最低的技能
        worst_skill = min(skills, key=lambda s: s.success_rate)
        if worst_skill.success_rate > self.skill_gap_threshold:
            return None
        # 目标 = 技能的触发状态
        target = worst_skill.trigger_state.copy()
        priority = 2.0 + (1.0 - worst_skill.success_rate) * 3.0
        goal = AutonomousGoal(
            goal_id=self._next_goal_id(),
            source=GoalSource.SKILL_GAP,
            target_state=target,
            priority=priority,
            description=f"skill_practice({worst_skill.name}, rate={worst_skill.success_rate:.2f})",
            achievement_threshold=0.4,
            max_steps=80,
        )
        self.log.info("Generated skill-gap goal: %s (skill=%s, rate=%.2f)",
                      goal.goal_id, worst_skill.name, worst_skill.success_rate)
        return goal

    def generate_from_self_improve(self, agent: Any) -> Optional[AutonomousGoal]:
        """从元认知状态生成自我改进目标。

        当校准偏移大时（过自信或欠自信），生成校准目标。
        """
        cal = agent.metacognition.calibrator
        offset = cal.calibration_offset
        if abs(offset) < self.overconfidence_threshold:
            return None
        # 目标 = 当前状态（原地校准，通过多次尝试校准信心）
        target = agent.current_state.copy()
        priority = 1.0 + abs(offset) * 5.0
        direction = "overconfident" if offset > 0 else "underconfident"
        goal = AutonomousGoal(
            goal_id=self._next_goal_id(),
            source=GoalSource.SELF_IMPROVE,
            target_state=target,
            priority=priority,
            description=f"self_calibrate({direction}, offset={offset:.3f})",
            achievement_threshold=0.5,
            max_steps=20,
        )
        self.log.info("Generated self-improve goal: %s (%s, offset=%.3f)",
                      goal.goal_id, direction, offset)
        return goal

    def generate_all(self, agent: Any, step_count: int) -> List[AutonomousGoal]:
        """生成所有适用的目标。

        参数：
        - agent: SPEPA Agent
        - step_count: 当前步数（用于控制生成频率）
        """
        goals = []
        for source in GoalSource:
            if source == GoalSource.EXTERNAL:
                continue
            # 检查生成间隔
            interval = self.generation_intervals.get(source, 100)
            last = self._last_generation.get(source, 0)
            if step_count - last < interval:
                continue
            # 尝试生成
            goal = None
            if source == GoalSource.HOMEOSTASIS:
                goal = self.generate_from_homeostasis(agent)
            elif source == GoalSource.CURIOSITY:
                goal = self.generate_from_curiosity(agent)
            elif source == GoalSource.SKILL_GAP:
                goal = self.generate_from_skill_gap(agent)
            elif source == GoalSource.SELF_IMPROVE:
                goal = self.generate_from_self_improve(agent)
            if goal is not None:
                goals.append(goal)
                self._last_generation[source] = step_count
                self._history.append(goal)
        if goals:
            self.log.info("Generated %d goals at step %d", len(goals), step_count)
        return goals

    def generate_external(self, target_state: np.ndarray,
                          priority: float = 5.0,
                          description: str = "external_goal") -> AutonomousGoal:
        """生成外部指令目标。"""
        target = np.asarray(target_state, dtype=np.float64).flatten()
        goal = AutonomousGoal(
            goal_id=self._next_goal_id(),
            source=GoalSource.EXTERNAL,
            target_state=target,
            priority=priority,
            description=description,
            achievement_threshold=0.2,
            max_steps=200,
        )
        self._history.append(goal)
        self.log.info("Generated external goal: %s", goal.goal_id)
        return goal

    def get_state(self) -> Dict[str, Any]:
        return {
            "goal_counter": self._goal_counter,
            "history_size": len(self._history),
            "last_generation": {src.value: step
                                for src, step in self._last_generation.items()},
        }

    def get_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """获取最近 n 个生成历史。"""
        return [g.to_dict() for g in self._history[-n:]]

    def reset(self) -> None:
        """重置生成器。"""
        self._goal_counter = 0
        self._history.clear()
        for src in self._last_generation:
            self._last_generation[src] = -1e9
        self.log.info("GoalGenerator reset")
