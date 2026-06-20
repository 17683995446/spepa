"""SPEPA 经验回传与技能共享。

设计原则（第一性原理）：
- 经验积累原理：子Agent的经验应反哺主Agent，避免重复学习
- 技能提取原理：从成功的状态-动作序列中提取可复用的技能
- 质量过滤原理：只回传高质量经验（成功、低误差）

工作流程：
1. 子Agent执行任务时记录轨迹（state, action, effect）
2. 任务完成后，从轨迹中提取技能候选
3. 过滤低质量技能（误差大、不稳定的）
4. 合并到主Agent的技能库
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.skills.skill import Skill, SkillLibrary


@dataclass
class TrajectoryStep:
    """轨迹中的一步。"""
    state: np.ndarray
    action: np.ndarray
    effect: np.ndarray  # 执行动作后的状态
    reward: float = 0.0


@dataclass
class AgentTrajectory:
    """子Agent的执行轨迹。"""
    agent_id: str
    task_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    success: bool = False
    final_error: float = 0.0

    def add_step(self, state: np.ndarray, action: np.ndarray,
                 effect: np.ndarray, reward: float = 0.0) -> None:
        """添加一步到轨迹。"""
        self.steps.append(TrajectoryStep(
            state=np.asarray(state, dtype=np.float64).flatten().copy(),
            action=np.asarray(action, dtype=np.float64).flatten().copy(),
            effect=np.asarray(effect, dtype=np.float64).flatten().copy(),
            reward=reward,
        ))


class ExperienceFeedback(LoggedMixin):
    """经验回传：从子Agent轨迹提取技能并回传到主Agent。

    技能提取策略：
    - 从成功轨迹中找"关键步骤"（误差下降最大的步骤）
    - 以关键步骤的 state 为 trigger，action 为技能动作
    - expected_effect = effect - state（状态变化）
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.multiagent.experience")
        self.config = config
        self.dim = config.state_dim
        # 技能提取参数
        self.min_trajectory_length = 3     # 最短轨迹长度
        self.max_skills_per_trajectory = 3  # 每条轨迹最多提取技能数
        self.min_improvement = 0.01        # 最小改善量（才算关键步骤）
        # 统计
        self.total_trajectories = 0
        self.total_skills_extracted = 0
        self.total_skills_merged = 0
        self.log.info("ExperienceFeedback initialized")

    def collect_trajectory(self, sub_agent) -> AgentTrajectory:
        """从子Agent收集执行轨迹。

        子Agent需要有 _action_history 和 current_state, target_state。
        """
        trajectory = AgentTrajectory(
            agent_id=sub_agent.agent_id,
            task_id=sub_agent.task.task_id,
            success=sub_agent.success,
            final_error=sub_agent.error_history[-1] if sub_agent.error_history else 0.0,
        )
        # 从动作历史重建轨迹
        # SubAgent 记录了 _action_history，但没有记录每步的 state
        # 用 error_history 和 current_state 反推
        action_history = list(getattr(sub_agent, '_action_history', []))
        error_history = list(getattr(sub_agent, 'error_history', []))
        if not action_history:
            return trajectory
        # 重建状态序列：从初始状态开始
        initial_state = sub_agent.task.initial_state.copy()
        current = initial_state.copy()
        for i, action in enumerate(action_history):
            action = np.asarray(action, dtype=np.float64).flatten()
            if action.shape[0] != self.dim:
                action = np.resize(action, self.dim)
            # 模拟执行后的状态
            effect = np.clip(current + action, 0.0, 1.0)
            reward = 0.0
            if i < len(error_history):
                # 误差下降 = 正奖励
                if i > 0:
                    reward = error_history[i - 1] - error_history[i]
                else:
                    reward = 0.0
            trajectory.add_step(current, action, effect, reward)
            current = effect
        return trajectory

    def extract_skills(self, trajectory: AgentTrajectory) -> List[Skill]:
        """从轨迹中提取技能候选。

        策略：找误差下降最大的步骤（关键转折点）。
        """
        if len(trajectory.steps) < self.min_trajectory_length:
            return []
        if not trajectory.success and trajectory.final_error > 0.5:
            # 失败且误差大，不提取
            return []
        # 计算每步的改善量
        improvements = []
        for i, step in enumerate(trajectory.steps):
            if i == 0:
                improvements.append(0.0)
            else:
                prev_state = trajectory.steps[i - 1].state
                # 改善 = 到目标的距离减少量
                prev_dist = float(np.linalg.norm(
                    trajectory.steps[-1].effect - prev_state))
                curr_dist = float(np.linalg.norm(
                    trajectory.steps[-1].effect - step.state))
                improvements.append(prev_dist - curr_dist)
        # 选改善量最大的步骤作为技能
        ranked = sorted(enumerate(improvements),
                        key=lambda x: x[1], reverse=True)
        skills = []
        for idx, improvement in ranked[:self.max_skills_per_trajectory]:
            if improvement < self.min_improvement:
                continue
            step = trajectory.steps[idx]
            # 清洗
            trigger = np.nan_to_num(step.state, nan=0.5, posinf=1.0, neginf=0.0)
            action = np.nan_to_num(step.action, nan=0.0, posinf=1.0, neginf=-1.0)
            effect = np.nan_to_num(step.effect - step.state,
                                   nan=0.0, posinf=1.0, neginf=-1.0)
            skill = Skill(
                name=f"skill_from_{trajectory.agent_id}_{idx}",
                trigger_state=trigger,
                action=action,
                expected_effect=effect,
                success_count=1 if trajectory.success else 0,
                failure_count=0 if trajectory.success else 1,
            )
            skills.append(skill)
        self.log.debug("Extracted %d skills from trajectory %s (success=%s)",
                       len(skills), trajectory.agent_id, trajectory.success)
        return skills

    def feedback_to_library(self, skills: List[Skill],
                            library: SkillLibrary) -> int:
        """将提取的技能回传到技能库。

        返回成功合并的技能数。
        """
        merged = 0
        for skill in skills:
            try:
                library.add(skill)
                merged += 1
            except Exception as e:
                self.log.warning("Failed to merge skill %s: %s",
                                 skill.name, e)
        self.total_skills_merged += merged
        self.log.info("Feedback %d skills to library (merged=%d)",
                      len(skills), merged)
        return merged

    def process_agent(self, sub_agent, library: SkillLibrary) -> int:
        """处理一个子Agent的完整经验回传流程。

        返回回传的技能数。
        """
        self.total_trajectories += 1
        trajectory = self.collect_trajectory(sub_agent)
        skills = self.extract_skills(trajectory)
        self.total_skills_extracted += len(skills)
        return self.feedback_to_library(skills, library)

    def get_state(self) -> Dict[str, Any]:
        return {
            "total_trajectories": self.total_trajectories,
            "total_skills_extracted": self.total_skills_extracted,
            "total_skills_merged": self.total_skills_merged,
            "extraction_rate": (
                self.total_skills_extracted / max(1, self.total_trajectories)
            ),
            "merge_rate": (
                self.total_skills_merged / max(1, self.total_skills_extracted)
            ),
        }

    def reset(self) -> None:
        """重置经验回传。"""
        self.total_trajectories = 0
        self.total_skills_extracted = 0
        self.total_skills_merged = 0
        self.log.info("ExperienceFeedback reset")
