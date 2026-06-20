"""SPEPA 任务队列：优先级队列 + 去重 + 过期清理。

设计原则（第一性原理）：
- 优先级原理：高优先级任务先执行
- 去重原理：相同目标的任务不重复入队
- 公平性原理：低优先级任务不会无限饥饿（老化机制）
- 过期清理：超时任务自动清理

任务生命周期：
- PENDING: 待执行
- RUNNING: 执行中
- COMPLETED: 已完成
- FAILED: 失败
- EXPIRED: 过期
"""
import time
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import heapq

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.autonomous.goal_generator import AutonomousGoal


class TaskStatus(Enum):
    """任务状态。"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(order=True)
class QueuedTask:
    """队列中的任务。"""
    # heapq 按优先级排序（负值 = 高优先级先出）
    sort_priority: float
    task_id: str = field(compare=False)
    goal: AutonomousGoal = field(compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    enqueued_at: float = field(default_factory=time.time, compare=False)
    started_at: float = field(default=0.0, compare=False)
    completed_at: float = field(default=0.0, compare=False)
    # 老化因子：等待越久优先级越高
    age_boost: float = field(default=0.0, compare=False)
    # 重试次数
    retry_count: int = field(default=0, compare=False)
    max_retries: int = field(default=2, compare=False)

    @property
    def effective_priority(self) -> float:
        """有效优先级（含老化加成）。"""
        return self.sort_priority - self.age_boost

    def is_terminal(self) -> bool:
        """是否处于终态。"""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.EXPIRED)


class TaskQueue(LoggedMixin):
    """任务队列：优先级调度 + 去重 + 老化 + 过期清理。"""

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.autonomous.task_queue")
        self.config = config
        # 优先级队列（最小堆）
        self._heap: List[QueuedTask] = []
        # 任务ID到任务的映射（用于去重和查找）
        self._task_map: Dict[str, QueuedTask] = {}
        # 已完成的任务历史
        self._completed: List[QueuedTask] = []
        # 当前运行中的任务
        self._running: Optional[QueuedTask] = None
        # 统计
        self.total_enqueued = 0
        self.total_completed = 0
        self.total_failed = 0
        self.total_expired = 0
        # 过期时间（秒）
        self.task_timeout = 300.0
        # 老化速率（每秒增加的优先级）
        self.aging_rate = 0.01
        # 去重距离阈值
        self.dedup_threshold = 0.15
        self.log.info("TaskQueue initialized")

    def enqueue(self, goal: AutonomousGoal) -> bool:
        """入队一个目标。

        返回 True 如果入队成功，False 如果因去重被拒绝。
        """
        # 去重：检查是否已有相似目标
        for existing in self._task_map.values():
            if existing.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                dist = float(np.linalg.norm(
                    existing.goal.target_state - goal.target_state
                ))
                if dist < self.dedup_threshold:
                    self.log.debug("Goal %s deduplicated (similar to %s, dist=%.3f)",
                                   goal.goal_id, existing.task_id, dist)
                    return False
        # 创建任务
        task = QueuedTask(
            sort_priority=-goal.priority,  # 负值使高优先级先出
            task_id=f"task_{self.total_enqueued + 1:06d}",
            goal=goal,
        )
        heapq.heappush(self._heap, task)
        self._task_map[task.task_id] = task
        self.total_enqueued += 1
        self.log.debug("Enqueued task %s (goal=%s, priority=%.1f)",
                       task.task_id, goal.goal_id, goal.priority)
        return True

    def dequeue(self) -> Optional[QueuedTask]:
        """出队最高优先级任务。"""
        # 清理过期任务
        self._cleanup_expired()
        # 跳过已终态的任务
        while self._heap:
            task = heapq.heappop(self._heap)
            if task.is_terminal():
                continue
            # 应用老化
            self._apply_aging(task)
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self._running = task
            self.log.debug("Dequeued task %s (goal=%s, eff_priority=%.2f)",
                           task.task_id, task.goal.goal_id,
                           task.effective_priority)
            return task
        self._running = None
        return None

    def complete(self, task_id: str, success: bool = True) -> bool:
        """完成当前运行中的任务。"""
        task = self._task_map.get(task_id)
        if task is None or task.status != TaskStatus.RUNNING:
            return False
        task.completed_at = time.time()
        if success:
            task.status = TaskStatus.COMPLETED
            task.goal.status = task.goal.status  # 保持 goal 状态
            self.total_completed += 1
            self.log.debug("Task %s completed", task_id)
        else:
            task.retry_count += 1
            if task.retry_count >= task.max_retries:
                task.status = TaskStatus.FAILED
                self.total_failed += 1
                self.log.debug("Task %s failed (retries exhausted)", task_id)
            else:
                # 重试：重新入队
                task.status = TaskStatus.PENDING
                heapq.heappush(self._heap, task)
                self.log.debug("Task %s retrying (%d/%d)",
                               task_id, task.retry_count, task.max_retries)
                return True
        self._completed.append(task)
        if self._running is not None and self._running.task_id == task_id:
            self._running = None
        # 清理历史
        if len(self._completed) > 100:
            self._completed = self._completed[-50:]
        return True

    def peek(self) -> Optional[QueuedTask]:
        """查看队首任务（不出队）。"""
        self._cleanup_expired()
        for task in sorted(self._heap):
            if not task.is_terminal():
                return task
        return None

    def size(self) -> int:
        """队列中待处理任务数。"""
        return sum(1 for t in self._heap if t.status == TaskStatus.PENDING)

    def is_empty(self) -> bool:
        """队列是否为空。"""
        return self.size() == 0

    def _apply_aging(self, task: QueuedTask) -> None:
        """应用老化：等待越久优先级越高。"""
        if task.status == TaskStatus.PENDING:
            wait_time = time.time() - task.enqueued_at
            task.age_boost = wait_time * self.aging_rate

    def _cleanup_expired(self) -> None:
        """清理过期任务。"""
        now = time.time()
        for task in list(self._task_map.values()):
            if task.status == TaskStatus.PENDING:
                if now - task.enqueued_at > self.task_timeout:
                    task.status = TaskStatus.EXPIRED
                    self.total_expired += 1
                    self.log.debug("Task %s expired", task.task_id)

    def get_running(self) -> Optional[QueuedTask]:
        """获取当前运行中的任务。"""
        return self._running

    def list_pending(self) -> List[Dict[str, Any]]:
        """列出所有待处理任务。"""
        return [
            {
                "task_id": t.task_id,
                "goal_id": t.goal.goal_id,
                "source": t.goal.source.value,
                "priority": t.goal.priority,
                "effective_priority": t.effective_priority,
                "status": t.status.value,
                "enqueued_at": t.enqueued_at,
            }
            for t in sorted(self._heap) if t.status == TaskStatus.PENDING
        ]

    def clear(self) -> int:
        """清空队列。返回清除数量。"""
        count = len(self._heap)
        self._heap.clear()
        self._task_map.clear()
        self._running = None
        self.log.info("TaskQueue cleared (%d tasks)", count)
        return count

    def get_state(self) -> Dict[str, Any]:
        return {
            "queue_size": self.size(),
            "total_enqueued": self.total_enqueued,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_expired": self.total_expired,
            "has_running": self._running is not None,
            "running_task": (self._running.task_id if self._running else None),
            "completed_history_size": len(self._completed),
        }

    def reset(self) -> None:
        """重置队列。"""
        self._heap.clear()
        self._task_map.clear()
        self._completed.clear()
        self._running = None
        self.total_enqueued = 0
        self.total_completed = 0
        self.total_failed = 0
        self.total_expired = 0
        self.log.info("TaskQueue reset")
