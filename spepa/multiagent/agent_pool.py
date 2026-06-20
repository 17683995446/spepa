"""SPEPA 多Agent协作框架。

设计原则（第一性原理 - 自由能最小化）：
- 主Agent将大任务分解为子任务，派生子Agent处理
- 子Agent的预测误差反馈给主Agent（自由能汇总）
- 任务分配基于子Agent的能力匹配度（状态-动作相似度）

组件：
- AgentPool: 管理子Agent的派生、注册、回收
- TaskAllocator: 基于自由能最小化的任务分配
- Coordinator: 协调子Agent执行，汇总结果
"""
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.utils.metrics import cosine_similarity


@dataclass
class SubTask:
    """子任务。"""
    task_id: str
    description: str
    initial_state: np.ndarray
    target_state: np.ndarray
    priority: float = 1.0
    max_steps: int = 50
    # 执行结果
    assigned_to: Optional[str] = None  # 分配给哪个子Agent
    result: Optional[Dict[str, Any]] = None
    completed: bool = False
    success: bool = False


class SubAgent:
    """子Agent：轻量级Agent，处理单一子任务。

    子Agent复用主Agent的核心能力（预测模型、记忆、决策），
    但有独立的状态和任务目标。
    """

    _id_counter = 0

    def __init__(self, config: SPEPAConfig, task: SubTask):
        SubAgent._id_counter += 1
        self.agent_id = f"subagent_{SubAgent._id_counter:04d}"
        self.config = config
        self.task = task
        self.current_state = task.initial_state.copy()
        self.target_state = task.target_state.copy()
        self.step_count = 0
        self.error_history: List[float] = []
        self.completed = False
        self.success = False
        # 简单的内部状态-动作映射（子Agent不需要完整Agent）
        self._action_history: deque = deque(maxlen=20)
        self.log_name = f"spepa.multiagent.{self.agent_id}"

    def step(self, env_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """执行一步：朝目标移动。"""
        if self.completed:
            return {"completed": True, "success": self.success}
        # 简单策略：朝目标方向移动（带探索噪声）
        direction = self.target_state - self.current_state
        dist = float(np.linalg.norm(direction))
        # 步长随距离衰减
        step_size = min(0.2, dist * 0.3)
        if dist > 1e-6:
            action = direction / dist * step_size
        else:
            action = np.zeros_like(direction)
        # 记录动作
        self._action_history.append(action.copy())
        # 执行动作
        if env_fn is not None:
            try:
                effect = env_fn(action)
                if effect is not None:
                    effect = np.asarray(effect, dtype=np.float64).flatten()
                    if effect.shape[0] != self.current_state.shape[0]:
                        effect = np.resize(effect, self.current_state.shape[0])
                    if not np.all(np.isfinite(effect)):
                        effect = np.nan_to_num(effect, nan=0.0, posinf=1.0, neginf=0.0)
                    self.current_state = effect
                else:
                    # 无环境反馈，用动作直接更新
                    self.current_state = np.clip(self.current_state + action, 0, 1)
            except Exception:
                self.current_state = np.clip(self.current_state + action, 0, 1)
        else:
            # 无环境，用动作直接更新
            self.current_state = np.clip(self.current_state + action, 0, 1)
        # 计算误差
        error = float(np.linalg.norm(self.target_state - self.current_state))
        self.error_history.append(error)
        self.step_count += 1
        # 检查完成
        if error < 0.1:
            self.completed = True
            self.success = True
        elif self.step_count >= self.task.max_steps:
            self.completed = True
            self.success = error < 0.5
        return {
            "step": self.step_count,
            "error": error,
            "completed": self.completed,
            "success": self.success,
            "current_state": self.current_state.copy(),
        }

    def run(self, env_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """运行直到完成或超时。"""
        while not self.completed:
            result = self.step(env_fn)
            if result["completed"]:
                break
        return {
            "task_id": self.task.task_id,
            "agent_id": self.agent_id,
            "steps": self.step_count,
            "completed": self.completed,
            "success": self.success,
            "final_error": self.error_history[-1] if self.error_history else 0.0,
            "final_state": self.current_state.copy(),
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "task_id": self.task.task_id,
            "step_count": self.step_count,
            "completed": self.completed,
            "success": self.success,
            "current_error": (self.error_history[-1] if self.error_history else 0.0),
        }


class AgentPool(LoggedMixin):
    """Agent池：管理子Agent的派生、注册、回收。

    功能：
    - 派生子Agent处理子任务
    - 注册子Agent
    - 回收已完成的子Agent
    - 限制并发子Agent数量（资源约束）
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.multiagent.agent_pool")
        self.config = config
        self.dim = config.state_dim
        # 活跃子Agent
        self.active_agents: Dict[str, SubAgent] = {}
        # 已完成子Agent（历史记录）
        self.completed_agents: List[Dict[str, Any]] = []
        # 最大并发数（资源约束）
        self.max_concurrent = 5
        self.log.info("AgentPool initialized (max_concurrent=%d)", self.max_concurrent)

    def spawn(self, task: SubTask) -> Optional[SubAgent]:
        """派生子Agent处理子任务。"""
        if len(self.active_agents) >= self.max_concurrent:
            self.log.warning("AgentPool full (%d/%d), cannot spawn",
                             len(self.active_agents), self.max_concurrent)
            return None
        # 维度对齐
        if task.initial_state.shape[0] != self.dim:
            task.initial_state = np.resize(task.initial_state, self.dim)
        if task.target_state.shape[0] != self.dim:
            task.target_state = np.resize(task.target_state, self.dim)
        agent = SubAgent(self.config, task)
        self.active_agents[agent.agent_id] = agent
        self.log.info("Spawned %s for task '%s' (active=%d)",
                      agent.agent_id, task.task_id, len(self.active_agents))
        return agent

    def get_agent(self, agent_id: str) -> Optional[SubAgent]:
        """获取子Agent。"""
        return self.active_agents.get(agent_id)

    def step_all(self, env_fn: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """推进所有活跃子Agent一步。"""
        results = []
        to_remove = []
        for agent_id, agent in self.active_agents.items():
            result = agent.step(env_fn)
            result["agent_id"] = agent_id
            results.append(result)
            if result["completed"]:
                to_remove.append(agent_id)
        # 回收已完成的Agent
        for agent_id in to_remove:
            agent = self.active_agents.pop(agent_id)
            summary = agent.run.__wrapped__ if hasattr(agent.run, '__wrapped__') else None
            self.completed_agents.append({
                "agent_id": agent_id,
                "task_id": agent.task.task_id,
                "success": agent.success,
                "steps": agent.step_count,
                "final_error": agent.error_history[-1] if agent.error_history else 0.0,
            })
            self.log.info("Recycled %s (success=%s, steps=%d)",
                          agent_id, agent.success, agent.step_count)
        return results

    def run_all(self, env_fn: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """运行所有活跃子Agent直到完成。"""
        all_results = []
        while self.active_agents:
            results = self.step_all(env_fn)
            all_results.extend(results)
        return all_results

    def recycle(self, agent_id: str) -> bool:
        """手动回收子Agent。"""
        if agent_id in self.active_agents:
            agent = self.active_agents.pop(agent_id)
            self.completed_agents.append({
                "agent_id": agent_id,
                "task_id": agent.task.task_id,
                "success": agent.success,
                "steps": agent.step_count,
                "final_error": agent.error_history[-1] if agent.error_history else 0.0,
            })
            self.log.info("Manually recycled %s", agent_id)
            return True
        return False

    def get_state(self) -> Dict[str, Any]:
        return {
            "active_count": len(self.active_agents),
            "completed_count": len(self.completed_agents),
            "max_concurrent": self.max_concurrent,
            "active_agents": [a.get_state() for a in self.active_agents.values()],
        }
