"""SPEPA 长任务自主驱动：任务图与任务驱动器。

设计原则（第一性原理 - 目标层次体系）：
- 长任务 = 目标层次体系的 L2 任务目标
- 长任务分解为有序子任务（任务图）
- 自主推进：完成一个子任务后自动推进到下一个
- 中断恢复：记录进度，可从断点恢复
- 自主续跑：所有子任务完成后，评估是否需要追加新子任务
"""
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class TaskStatus(Enum):
    """子任务状态。"""
    PENDING = "pending"       # 待执行
    RUNNING = "running"       # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    SKIPPED = "skipped"       # 跳过（依赖失败）


@dataclass
class GraphNode:
    """任务图节点。"""
    node_id: str
    description: str
    target_state: np.ndarray
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    dependencies: Set[str] = field(default_factory=set)  # 依赖的节点ID
    max_retries: int = 2
    retry_count: int = 0
    # 执行元数据
    started_at: float = 0.0
    completed_at: float = 0.0
    steps_taken: int = 0

    @property
    def is_ready(self) -> bool:
        """是否就绪可执行（所有依赖已完成）。"""
        return self.status == TaskStatus.PENDING

    @property
    def is_terminal(self) -> bool:
        """是否处于终态。"""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)


class TaskGraph(LoggedMixin):
    """任务图：管理子任务及其依赖关系。

    支持有向无环图（DAG）结构，子任务可有依赖。
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.longtask.task_graph")
        self.config = config
        self.dim = config.state_dim
        self.nodes: Dict[str, GraphNode] = {}
        self.log.info("TaskGraph initialized")

    def add_node(self, node: GraphNode) -> None:
        """添加节点。"""
        if node.target_state.shape[0] != self.dim:
            node.target_state = np.resize(node.target_state, self.dim)
        self.nodes[node.node_id] = node
        self.log.debug("Node '%s' added (deps=%s)", node.node_id, node.dependencies)

    def add_dependency(self, node_id: str, depends_on: str) -> bool:
        """添加依赖关系。"""
        if node_id not in self.nodes or depends_on not in self.nodes:
            return False
        # 检查循环依赖
        if self._would_create_cycle(node_id, depends_on):
            self.log.warning("Dependency %s -> %s would create cycle, rejected",
                             depends_on, node_id)
            return False
        self.nodes[node_id].dependencies.add(depends_on)
        return True

    def _would_create_cycle(self, node_id: str, depends_on: str) -> bool:
        """检查添加依赖是否会创建循环。"""
        # BFS：从 depends_on 出发，看能否回到 node_id
        visited = set()
        queue = [depends_on]
        while queue:
            current = queue.pop(0)
            if current == node_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            for dep in self.nodes.get(current, GraphNode("", "", np.zeros(0))).dependencies:
                queue.append(dep)
        return False

    def get_ready_nodes(self) -> List[GraphNode]:
        """获取所有就绪可执行的节点（依赖全部完成）。

        优先检查失败依赖：如果有依赖失败/跳过，则跳过此节点。
        """
        ready = []
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            # 先检查是否有失败的依赖（如果有，跳过此节点）
            failed_deps = [
                dep for dep in node.dependencies
                if dep in self.nodes and self.nodes[dep].status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
            ]
            if failed_deps:
                node.status = TaskStatus.SKIPPED
                self.log.info("Node '%s' skipped (failed deps: %s)",
                              node.node_id, failed_deps)
                continue
            # 检查所有依赖是否完成
            deps_ok = all(
                self.nodes[dep].status == TaskStatus.COMPLETED
                for dep in node.dependencies
                if dep in self.nodes
            )
            if deps_ok:
                ready.append(node)
        return ready

    def get_pending_count(self) -> int:
        """待执行节点数。"""
        return sum(1 for n in self.nodes.values() if n.status == TaskStatus.PENDING)

    def get_completed_count(self) -> int:
        """已完成节点数。"""
        return sum(1 for n in self.nodes.values() if n.status == TaskStatus.COMPLETED)

    def is_all_done(self) -> bool:
        """所有节点是否处于终态。"""
        return all(n.is_terminal for n in self.nodes.values())

    def get_progress(self) -> float:
        """完成进度 [0, 1]。"""
        if not self.nodes:
            return 1.0
        done = self.get_completed_count()
        return done / len(self.nodes)

    def get_state(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "pending": self.get_pending_count(),
            "completed": self.get_completed_count(),
            "failed": sum(1 for n in self.nodes.values() if n.status == TaskStatus.FAILED),
            "skipped": sum(1 for n in self.nodes.values() if n.status == TaskStatus.SKIPPED),
            "progress": self.get_progress(),
        }
