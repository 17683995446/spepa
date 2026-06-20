"""SPEPA 长任务驱动器：自主推进长任务执行。

设计原则（第一性原理 - 自驱永续进化）：
- 自主推进：完成一个子任务后自动推进到下一个
- 中断恢复：记录进度，可从断点恢复
- 自主续跑：所有子任务完成后，评估是否需要追加新子任务
- 自我驱动：基于目标层次体系，从 L4 元目标向下驱动
"""
import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.longtask.task_graph import TaskGraph, GraphNode, TaskStatus
from spepa.multiagent.agent_pool import AgentPool, SubTask
from spepa.multiagent.task_allocator import TaskAllocator, Coordinator


class LongTaskDriver(LoggedMixin):
    """长任务驱动器：自主推进长任务。

    工作流程：
    1. 接收长任务（最终目标状态）
    2. 将长任务分解为子任务序列（任务图）
    3. 自主推进：逐个执行子任务
    4. 中断恢复：可保存和恢复进度
    5. 自主续跑：完成后评估是否需要追加
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.longtask.task_driver")
        self.config = config
        self.dim = config.state_dim
        self.graph = TaskGraph(config)
        self.pool = AgentPool(config)
        self.allocator = TaskAllocator(config, self.pool)
        self.coordinator = Coordinator(config, self.pool)
        # 长任务元数据
        self.task_description: str = ""
        self.initial_state: Optional[np.ndarray] = None
        self.final_target: Optional[np.ndarray] = None
        self.current_state: Optional[np.ndarray] = None
        # 执行统计
        self.total_steps: int = 0
        self.total_subtasks: int = 0
        self.completed_subtasks: int = 0
        self.start_time: float = 0.0
        self.is_running: bool = False
        self.is_completed: bool = False
        # 检查点（用于中断恢复）
        self._checkpoint: Optional[Dict[str, Any]] = None
        self.log.info("LongTaskDriver initialized")

    def plan(self, initial_state: np.ndarray, final_target: np.ndarray,
             description: str = "long_task",
             n_stages: int = 3) -> TaskGraph:
        """规划长任务：将最终目标分解为多阶段子任务。

        策略：从初始状态到最终目标，按阶段插值生成中间目标。
        """
        initial_state = np.asarray(initial_state, dtype=np.float64).flatten()
        final_target = np.asarray(final_target, dtype=np.float64).flatten()
        if initial_state.shape[0] != self.dim:
            initial_state = np.resize(initial_state, self.dim)
        if final_target.shape[0] != self.dim:
            final_target = np.resize(final_target, self.dim)
        # 清洗
        if not np.all(np.isfinite(initial_state)):
            initial_state = np.nan_to_num(initial_state, nan=0.5, posinf=1.0, neginf=0.0)
        if not np.all(np.isfinite(final_target)):
            final_target = np.nan_to_num(final_target, nan=0.5, posinf=1.0, neginf=0.0)

        self.task_description = description
        self.initial_state = initial_state.copy()
        self.final_target = final_target.copy()
        self.current_state = initial_state.copy()

        # 清空旧图
        self.graph = TaskGraph(self.config)

        # 按阶段插值生成中间目标
        prev_node_id = None
        for stage in range(n_stages):
            alpha = (stage + 1) / n_stages
            stage_target = (1 - alpha) * initial_state + alpha * final_target
            node_id = f"stage_{stage}"
            node = GraphNode(
                node_id=node_id,
                description=f"Stage {stage}: reach {alpha*100:.0f}% of target",
                target_state=stage_target,
            )
            if prev_node_id is not None:
                node.dependencies.add(prev_node_id)
            self.graph.add_node(node)
            prev_node_id = node_id

        self.total_subtasks = len(self.graph.nodes)
        self.log.info("Long task '%s' planned: %d stages", description, self.total_subtasks)
        return self.graph

    def run(self, env_fn: Optional[Callable] = None,
            max_total_steps: int = 1000) -> Dict[str, Any]:
        """自主驱动长任务执行直到完成。

        自主推进逻辑：
        1. 获取就绪节点
        2. 为每个就绪节点派生子Agent
        3. 推进子Agent执行
        4. 完成后更新任务图状态
        5. 重复直到所有节点完成
        """
        if self.is_completed:
            return {"completed": True, "already_done": True}
        self.is_running = True
        self.start_time = time.time()
        all_results = []

        while not self.graph.is_all_done() and self.total_steps < max_total_steps:
            # 1. 获取就绪节点
            ready_nodes = self.graph.get_ready_nodes()
            if not ready_nodes:
                # 没有就绪节点，检查是否有运行中的
                if not self.pool.active_agents:
                    # 没有运行中的，可能所有剩余节点都被跳过
                    break
                # 推进运行中的
                step_results = self.pool.step_all(env_fn)
                self.total_steps += 1
                # 检查完成的子Agent，更新任务图
                for sr in step_results:
                    if sr.get("completed"):
                        self._update_node_from_result(sr)
                        all_results.append(sr)
                continue

            # 2. 为就绪节点派生子Agent
            for node in ready_nodes:
                node.status = TaskStatus.RUNNING
                node.started_at = time.time()
                # 子任务的初始状态 = 当前状态（或依赖节点的最终状态）
                init_state = self._get_node_initial_state(node)
                subtask = SubTask(
                    task_id=node.node_id,
                    description=node.description,
                    initial_state=init_state,
                    target_state=node.target_state,
                    max_steps=30,
                )
                self.allocator.pool.spawn(subtask)

            # 3. 推进所有子Agent一步
            step_results = self.pool.step_all(env_fn)
            self.total_steps += 1
            # 4. 更新完成的节点
            for sr in step_results:
                if sr.get("completed"):
                    self._update_node_from_result(sr)
                    all_results.append(sr)

        # 5. 运行剩余的子Agent
        remaining = self.pool.run_all(env_fn)
        for sr in remaining:
            self._update_node_from_result(sr)
            all_results.append(sr)
        self.total_steps += len(remaining)

        self.is_running = False
        self.is_completed = self.graph.is_all_done()
        elapsed = time.time() - self.start_time
        summary = {
            "completed": self.is_completed,
            "total_steps": self.total_steps,
            "total_subtasks": self.total_subtasks,
            "completed_subtasks": self.graph.get_completed_count(),
            "progress": self.graph.get_progress(),
            "elapsed_time": elapsed,
            "final_state": self.current_state.tolist() if self.current_state is not None else None,
            "results": all_results,
        }
        self.log.info("Long task '%s' finished: progress=%.2f, steps=%d, time=%.2fs",
                      self.task_description, summary["progress"],
                      summary["total_steps"], elapsed)
        return summary

    def _get_node_initial_state(self, node: GraphNode) -> np.ndarray:
        """获取节点的初始状态（基于依赖节点的最终状态）。"""
        if not node.dependencies:
            return self.initial_state.copy() if self.initial_state is not None else np.ones(self.dim) * 0.5
        # 使用依赖节点的最终状态
        for dep_id in node.dependencies:
            dep_node = self.graph.nodes.get(dep_id)
            if dep_node and dep_node.result and "final_state" in dep_node.result:
                return np.asarray(dep_node.result["final_state"], dtype=np.float64).flatten()
        return self.current_state.copy() if self.current_state is not None else np.ones(self.dim) * 0.5

    def _update_node_from_result(self, result: Dict[str, Any]) -> None:
        """根据子Agent结果更新任务图节点。"""
        task_id = result.get("task_id")
        if not task_id or task_id not in self.graph.nodes:
            return
        node = self.graph.nodes[task_id]
        node.result = result
        node.completed_at = time.time()
        node.steps_taken = result.get("steps", 0)
        if result.get("success", False):
            node.status = TaskStatus.COMPLETED
            self.completed_subtasks += 1
            # 更新当前状态
            if "final_state" in result:
                self.current_state = np.asarray(result["final_state"], dtype=np.float64).flatten()
        else:
            node.retry_count += 1
            if node.retry_count >= node.max_retries:
                node.status = TaskStatus.FAILED
                self.log.warning("Node '%s' failed after %d retries",
                                 task_id, node.retry_count)
            else:
                # 重试：重置为 pending
                node.status = TaskStatus.PENDING
                self.log.info("Node '%s' retrying (%d/%d)",
                              task_id, node.retry_count, node.max_retries)

    def save_checkpoint(self) -> Dict[str, Any]:
        """保存检查点（用于中断恢复）。"""
        checkpoint = {
            "task_description": self.task_description,
            "initial_state": self.initial_state.tolist() if self.initial_state is not None else None,
            "final_target": self.final_target.tolist() if self.final_target is not None else None,
            "current_state": self.current_state.tolist() if self.current_state is not None else None,
            "total_steps": self.total_steps,
            "completed_subtasks": self.completed_subtasks,
            "is_completed": self.is_completed,
            "graph_state": self.graph.get_state(),
            "nodes": {
                nid: {
                    "status": n.status.value,
                    "retry_count": n.retry_count,
                    "steps_taken": n.steps_taken,
                }
                for nid, n in self.graph.nodes.items()
            },
        }
        self._checkpoint = checkpoint
        self.log.info("Checkpoint saved (progress=%.2f)", checkpoint["graph_state"]["progress"])
        return checkpoint

    def restore_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """从检查点恢复。"""
        try:
            self.task_description = checkpoint["task_description"]
            self.initial_state = np.asarray(checkpoint["initial_state"], dtype=np.float64)
            self.final_target = np.asarray(checkpoint["final_target"], dtype=np.float64)
            self.current_state = np.asarray(checkpoint["current_state"], dtype=np.float64)
            self.total_steps = checkpoint["total_steps"]
            self.completed_subtasks = checkpoint["completed_subtasks"]
            self.is_completed = checkpoint["is_completed"]
            # 恢复节点状态
            for nid, ns in checkpoint["nodes"].items():
                if nid in self.graph.nodes:
                    node = self.graph.nodes[nid]
                    node.status = TaskStatus(ns["status"])
                    node.retry_count = ns["retry_count"]
                    node.steps_taken = ns["steps_taken"]
            self.log.info("Checkpoint restored (progress=%.2f)",
                          self.graph.get_progress())
            return True
        except Exception as e:
            self.log.error("Checkpoint restore failed: %s", e)
            return False

    def get_state(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "is_running": self.is_running,
            "is_completed": self.is_completed,
            "total_steps": self.total_steps,
            "total_subtasks": self.total_subtasks,
            "completed_subtasks": self.completed_subtasks,
            "progress": self.graph.get_progress(),
            "graph_state": self.graph.get_state(),
            "pool_state": self.pool.get_state(),
        }
