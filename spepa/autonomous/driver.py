"""SPEPA 自主驱动器：目标→任务→执行闭环。

设计原则（第一性原理）：
- 自主性原理：系统应能自主生成目标、排队任务、执行完成，无需外部指令
- 闭环原理：生成→入队→执行→评估→反馈，形成完整闭环
- 优先级原理：高优先级目标先执行，低优先级不饥饿
- 反馈原理：执行结果反馈到目标生成，调整未来目标

工作流程：
1. 每隔N步，GoalGenerator 生成新目标
2. 新目标入队 TaskQueue
3. AutonomousDriver 从队列取出最高优先级任务
4. 驱动 Agent 朝目标执行
5. 评估目标是否达成
6. 反馈结果到队列和生成器
"""
import time
import numpy as np
from typing import Dict, Any, Optional, Callable, List

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.autonomous.goal_generator import GoalGenerator, AutonomousGoal, GoalSource, GoalStatus
from spepa.autonomous.task_queue import TaskQueue, QueuedTask, TaskStatus


class AutonomousDriver(LoggedMixin):
    """自主驱动器：协调目标生成、任务队列和执行。

    这是"自驱动"的核心：系统自主决定做什么、何时做、如何做。
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.autonomous.driver")
        self.config = config
        self.goal_generator = GoalGenerator(config)
        self.task_queue = TaskQueue(config)
        # 当前执行的任务
        self._current_task: Optional[QueuedTask] = None
        # 执行统计
        self.goals_generated = 0
        self.goals_achieved = 0
        self.goals_failed = 0
        # 目标生成间隔（步）
        self.generation_interval = 10
        # 上次生成的步数
        self._last_generation_step = 0
        # 每个任务的执行步数
        self._task_step_count = 0
        self.log.info("AutonomousDriver initialized")

    def tick(self, agent: Any, step_count: int,
             env_step_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """驱动一个步进。

        这是自主驱动的核心入口，每个 Agent step 调用一次。
        流程：
        1. 检查是否需要生成新目标
        2. 如果没有当前任务，从队列取出一个
        3. 如果有当前任务，驱动 Agent 朝目标执行
        4. 评估任务完成情况

        参数：
        - agent: SPEPA Agent
        - step_count: 当前步数
        - env_step_fn: 环境步进函数

        返回：执行结果字典
        """
        result = {
            "step": step_count,
            "generated_goals": [],
            "task_action": None,
            "task_completed": False,
            "task_failed": False,
        }
        # 1. 定期生成新目标
        if step_count - self._last_generation_step >= self.generation_interval:
            new_goals = self.goal_generator.generate_all(agent, step_count)
            for goal in new_goals:
                self.task_queue.enqueue(goal)
            self.goals_generated += len(new_goals)
            self._last_generation_step = step_count
            result["generated_goals"] = [g.goal_id for g in new_goals]
        # 2. 如果没有当前任务，从队列取一个
        if self._current_task is None or self._current_task.is_terminal():
            self._current_task = self.task_queue.dequeue()
            self._task_step_count = 0
            if self._current_task is not None:
                self.log.info("Started task %s (goal=%s, source=%s)",
                              self._current_task.task_id,
                              self._current_task.goal.goal_id,
                              self._current_task.goal.source.value)
        # 3. 如果有当前任务，驱动执行
        if self._current_task is not None:
            goal = self._current_task.goal
            # 先检查是否已达成（避免不必要的执行导致状态偏离）
            if goal.is_achieved(agent.current_state):
                self.task_queue.complete(self._current_task.task_id, success=True)
                self.goals_achieved += 1
                result["task_completed"] = True
                self.log.info("Task %s already achieved (steps=%d)",
                              self._current_task.task_id, self._task_step_count)
                self._current_task = None
            else:
                task_result = self._execute_task_step(agent, env_step_fn)
                result["task_action"] = task_result
                # 4. 评估任务完成
                self._task_step_count += 1
                goal.steps_taken = self._task_step_count
                # 检查达成
                if goal.is_achieved(agent.current_state):
                    self.task_queue.complete(self._current_task.task_id, success=True)
                    self.goals_achieved += 1
                    result["task_completed"] = True
                    self.log.info("Task %s achieved (steps=%d)",
                                  self._current_task.task_id, self._task_step_count)
                    self._current_task = None
                elif goal.is_expired():
                    self.task_queue.complete(self._current_task.task_id, success=False)
                    self.goals_failed += 1
                    result["task_failed"] = True
                    self.log.info("Task %s expired (steps=%d/%d)",
                                  self._current_task.task_id,
                                  self._task_step_count, goal.max_steps)
                    self._current_task = None
        else:
            # 无任务时也执行在线周期，持续积累经验
            try:
                agent.online_cycle(env_step_fn=env_step_fn)
            except Exception as e:
                self.log.warning("Idle cycle error: %s", e)
        return result

    def _execute_task_step(self, agent: Any,
                           env_step_fn: Optional[Callable]) -> Dict[str, Any]:
        """执行当前任务的一步。

        策略：将任务目标作为 Agent 的临时目标，驱动 Agent 朝目标执行。
        """
        if self._current_task is None:
            return {"action": "idle"}
        goal = self._current_task.goal
        # 计算朝目标的动作方向
        current = agent.current_state.copy()
        target = goal.target_state.copy()
        direction = target - current
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return {"action": "reached", "distance": 0.0}
        # 归一化方向
        direction = direction / (dist + 1e-6)
        # 将目标方向注入 Agent 的 goal_hierarchy（L2 任务目标）
        # 这里不直接修改 Agent 内部状态，而是通过 observation 引导
        # 使用目标状态作为观测，让 Agent 的预测-纠错机制自然朝目标移动
        # 同时执行一个在线周期
        try:
            online_result = agent.online_cycle(
                observation=target,  # 用目标作为观测，引导预测模型
                env_step_fn=env_step_fn,
            )
            return {
                "action": "execute",
                "distance": dist,
                "online_result": online_result is not None,
            }
        except Exception as e:
            self.log.warning("Task execution error: %s", e)
            return {"action": "error", "error": str(e)}

    def add_external_goal(self, target_state: np.ndarray,
                          priority: float = 5.0,
                          description: str = "external") -> bool:
        """添加外部目标。"""
        goal = self.goal_generator.generate_external(
            target_state, priority, description
        )
        return self.task_queue.enqueue(goal)

    def get_current_task(self) -> Optional[QueuedTask]:
        """获取当前执行的任务。"""
        return self._current_task

    def get_state(self) -> Dict[str, Any]:
        return {
            "goals_generated": self.goals_generated,
            "goals_achieved": self.goals_achieved,
            "goals_failed": self.goals_failed,
            "achievement_rate": (
                self.goals_achieved / max(1, self.goals_achieved + self.goals_failed)
            ),
            "current_task": (
                self._current_task.task_id if self._current_task else None
            ),
            "current_goal": (
                self._current_task.goal.goal_id if self._current_task else None
            ),
            "current_source": (
                self._current_task.goal.source.value if self._current_task else None
            ),
            "task_queue": self.task_queue.get_state(),
            "goal_generator": self.goal_generator.get_state(),
        }

    def reset(self) -> None:
        """重置驱动器。"""
        self.goal_generator.reset()
        self.task_queue.reset()
        self._current_task = None
        self.goals_generated = 0
        self.goals_achieved = 0
        self.goals_failed = 0
        self._last_generation_step = 0
        self._task_step_count = 0
        self.log.info("AutonomousDriver reset")
