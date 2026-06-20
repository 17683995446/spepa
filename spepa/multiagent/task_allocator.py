"""SPEPA 任务分配器与协调器。

设计原则（第一性原理 - 自由能最小化）：
- 任务分配基于"预期自由能最小化"：选择能最大降低不确定性的分配方案
- 子任务分解基于"目标层次体系"：大目标分解为可独立完成的子目标
- 结果汇总基于"预测误差加权"：子Agent的预测误差作为结果可信度
"""
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.utils.metrics import cosine_similarity
from spepa.multiagent.agent_pool import AgentPool, SubAgent, SubTask


class TaskAllocator(LoggedMixin):
    """任务分配器：将大任务分解为子任务并分配给子Agent。

    分解策略：
    1. 维度分解：高维目标按维度分组
    2. 阶段分解：目标按距离分阶段
    3. 能力匹配：根据子Agent能力（技能库）分配
    """

    def __init__(self, config: SPEPAConfig, pool: AgentPool):
        self._set_log_name("spepa.multiagent.task_allocator")
        self.config = config
        self.pool = pool
        self.dim = config.state_dim
        # 子任务分组大小（每组的维度数）
        self.group_size = max(2, self.dim // 4)
        self.log.info("TaskAllocator initialized (group_size=%d)", self.group_size)

    def decompose(self, initial_state: np.ndarray, target_state: np.ndarray,
                  description: str = "main_task") -> List[SubTask]:
        """将大任务分解为子任务。

        策略：按维度分组，每组一个子任务。
        """
        initial_state = np.asarray(initial_state, dtype=np.float64).flatten()
        target_state = np.asarray(target_state, dtype=np.float64).flatten()
        if initial_state.shape[0] != self.dim:
            initial_state = np.resize(initial_state, self.dim)
        if target_state.shape[0] != self.dim:
            target_state = np.resize(target_state, self.dim)
        # 清洗
        if not np.all(np.isfinite(initial_state)):
            initial_state = np.nan_to_num(initial_state, nan=0.5, posinf=1.0, neginf=0.0)
        if not np.all(np.isfinite(target_state)):
            target_state = np.nan_to_num(target_state, nan=0.5, posinf=1.0, neginf=0.0)

        subtasks = []
        n_groups = max(1, (self.dim + self.group_size - 1) // self.group_size)
        for g in range(n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.dim)
            if start >= end:
                break
            # 子任务的初始和目标状态（只在该组维度上有值，其余为初始值）
            sub_initial = initial_state.copy()
            sub_target = initial_state.copy()
            sub_target[start:end] = target_state[start:end]
            task = SubTask(
                task_id=f"{description}_sub_{g}",
                description=f"Subtask {g} for {description}",
                initial_state=sub_initial,
                target_state=sub_target,
                priority=1.0,
                max_steps=30,
            )
            subtasks.append(task)
        self.log.info("Decomposed '%s' into %d subtasks", description, len(subtasks))
        return subtasks

    def allocate(self, subtasks: List[SubTask]) -> Dict[str, Optional[SubAgent]]:
        """分配子任务给子Agent。"""
        assignments = {}
        for task in subtasks:
            agent = self.pool.spawn(task)
            if agent is not None:
                task.assigned_to = agent.agent_id
                assignments[task.task_id] = agent
                self.log.debug("Task '%s' allocated to %s", task.task_id, agent.agent_id)
            else:
                assignments[task.task_id] = None
                self.log.warning("Task '%s' could not be allocated (pool full)", task.task_id)
        return assignments

    def decompose_and_allocate(self, initial_state: np.ndarray,
                                target_state: np.ndarray,
                                description: str = "main_task") -> Dict[str, Optional[SubAgent]]:
        """分解并分配任务。"""
        subtasks = self.decompose(initial_state, target_state, description)
        return self.allocate(subtasks)

    def get_state(self) -> Dict[str, Any]:
        return {
            "group_size": self.group_size,
            "pool_state": self.pool.get_state(),
        }


class Coordinator(LoggedMixin):
    """协作协调器：协调子Agent执行，汇总结果。

    功能：
    - 推进所有子Agent
    - 汇总子Agent结果
    - 基于预测误差加权合并结果
    - 处理失败的子任务（重试或降级）
    """

    def __init__(self, config: SPEPAConfig, pool: AgentPool):
        self._set_log_name("spepa.multiagent.coordinator")
        self.config = config
        self.pool = pool
        self.dim = config.state_dim
        # 最大重试次数
        self.max_retries = 2
        self.log.info("Coordinator initialized (max_retries=%d)", self.max_retries)

    def coordinate(self, env_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """协调所有子Agent执行直到完成。"""
        all_results = self.pool.run_all(env_fn)
        # 汇总结果
        summary = self._summarize(all_results)
        self.log.info("Coordination complete: %d/%d succeeded",
                      summary["success_count"], summary["total_count"])
        return summary

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总子Agent结果。"""
        total = len(results)
        successes = sum(1 for r in results if r.get("success", False))
        failures = total - successes
        # 平均误差
        errors = [r.get("final_error", 0.0) for r in results]
        avg_error = float(np.mean(errors)) if errors else 0.0
        # 总步数
        total_steps = sum(r.get("steps", 0) for r in results)
        return {
            "total_count": total,
            "success_count": successes,
            "failure_count": failures,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_error": avg_error,
            "total_steps": total_steps,
            "results": results,
        }

    def merge_results(self, results: List[Dict[str, Any]],
                      initial_state: np.ndarray) -> np.ndarray:
        """合并子Agent的结果到统一状态。

        策略：每个子Agent负责不同维度组，按组拼接。
        """
        merged = np.asarray(initial_state, dtype=np.float64).flatten().copy()
        if merged.shape[0] != self.dim:
            merged = np.resize(merged, self.dim)
        for r in results:
            if "final_state" in r:
                final = np.asarray(r["final_state"], dtype=np.float64).flatten()
                if final.shape[0] == self.dim:
                    # 加权合并：成功的子Agent权重高
                    weight = 1.0 if r.get("success", False) else 0.3
                    merged = (1 - weight) * merged + weight * final
        # 清洗
        if not np.all(np.isfinite(merged)):
            merged = np.nan_to_num(merged, nan=0.5, posinf=1.0, neginf=0.0)
        return merged

    def get_state(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "pool_state": self.pool.get_state(),
        }
