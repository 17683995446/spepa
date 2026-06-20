"""SPEPA 记忆系统（双系统：快速 + 慢速巩固）。

- 快速记忆：短期、大容量、易遗忘（类似工作记忆/海马）
- 慢速记忆：长期、小容量、巩固后（类似新皮层）
- 巩固：定期将快速记忆中的重要项迁移到慢速记忆
"""
import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class MemoryItem:
    """单条记忆。"""

    def __init__(self, key: np.ndarray, value: Any, importance: float = 0.0,
                 timestamp: float = 0.0):
        self.key = np.asarray(key, dtype=np.float64).flatten()
        self.value = value
        self.importance = float(importance)
        self.timestamp = float(timestamp)
        self.access_count = 0
        self.consolidated = False  # 是否已被巩固到慢速记忆

    def touch(self) -> None:
        self.access_count += 1

    def score(self) -> float:
        """综合得分：重要性 * 访问频率。"""
        return self.importance * (1.0 + 0.1 * self.access_count)


class FastMemory:
    """快速记忆：大容量、FIFO + 重要性保留。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)

    def retrieve(self, query: np.ndarray, k: int = 5) -> List[MemoryItem]:
        """按余弦相似度检索 top-k。"""
        if not self.items:
            return []
        query = np.asarray(query, dtype=np.float64).flatten()
        scored = []
        for it in self.items:
            na = np.linalg.norm(query) + 1e-10
            nb = np.linalg.norm(it.key) + 1e-10
            sim = float(np.dot(query, it.key) / (na * nb))
            scored.append((sim, it))
        scored.sort(key=lambda x: -x[0])
        out = []
        for sim, it in scored[:k]:
            it.touch()
            out.append(it)
        return out

    def __len__(self) -> int:
        return len(self.items)


class SlowMemory:
    """慢速记忆：小容量、按重要性保留 Top 30%。

    始终保持项数不超过 capacity * keep_ratio，每次添加后立即裁剪。
    """

    def __init__(self, capacity: int, keep_ratio: float = 0.30):
        self.capacity = capacity
        self.keep_ratio = keep_ratio
        # 目标保留数量（始终不超过此值）
        self.keep_n = max(1, int(capacity * keep_ratio))
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        # 超过目标保留数量时立即裁剪
        if len(self.items) > self.keep_n:
            self._prune()

    def _prune(self) -> None:
        """保留得分最高的 Top keep_n 项。"""
        self.items.sort(key=lambda it: -it.score())
        self.items = self.items[:self.keep_n]

    def retrieve(self, query: np.ndarray, k: int = 5) -> List[MemoryItem]:
        if not self.items:
            return []
        query = np.asarray(query, dtype=np.float64).flatten()
        scored = []
        for it in self.items:
            na = np.linalg.norm(query) + 1e-10
            nb = np.linalg.norm(it.key) + 1e-10
            sim = float(np.dot(query, it.key) / (na * nb))
            scored.append((sim, it))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:k]]

    def __len__(self) -> int:
        return len(self.items)


class MemorySystem(LoggedMixin):
    """双系统记忆：快速 + 慢速巩固。"""

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.core.memory_system")
        self.config = config
        self.fast = FastMemory(config.fast_memory_capacity)
        self.slow = SlowMemory(config.slow_memory_capacity,
                               keep_ratio=config.resources.storage_keep_top_ratio)
        self.step_count = 0
        self.log.info("MemorySystem initialized (fast=%d slow=%d keep_ratio=%.2f)",
                      config.fast_memory_capacity, config.slow_memory_capacity,
                      config.resources.storage_keep_top_ratio)

    def encode(self, key: np.ndarray, value: Any, importance: float = 0.0) -> None:
        """编码一条记忆到快速记忆。"""
        item = MemoryItem(key, value, importance=importance,
                          timestamp=float(self.step_count))
        self.fast.add(item)
        self.log.debug("Encoded memory: importance=%.4f fast_size=%d",
                       importance, len(self.fast))

    def retrieve(self, query: np.ndarray, k: int = 5) -> List[MemoryItem]:
        """从快速+慢速记忆检索（按相似度排序）。"""
        query = np.asarray(query, dtype=np.float64).flatten()
        fast_results = self.fast.retrieve(query, k=k)
        slow_results = self.slow.retrieve(query, k=k)
        combined = fast_results + slow_results
        # 按与 query 的相似度重新排序
        qn = np.linalg.norm(query) + 1e-10
        scored = []
        for it in combined:
            kn = np.linalg.norm(it.key) + 1e-10
            sim = float(np.dot(query, it.key) / (qn * kn))
            scored.append((sim, it))
        scored.sort(key=lambda x: -x[0])
        out = [it for _, it in scored[:k]]
        self.log.debug("Retrieved memories: fast=%d slow=%d combined=%d -> %d",
                       len(fast_results), len(slow_results), len(combined), len(out))
        return out

    def consolidate(self) -> int:
        """巩固：将快速记忆中得分高的迁移到慢速记忆。返回巩固数量。

        幂等性：基于总项数计算目标巩固数，已巩固项不再重复巩固。
        """
        if not self.fast.items:
            self.log.debug("Consolidation skipped: fast memory empty")
            return 0
        total = len(self.fast.items)
        keep_ratio = self.config.resources.storage_keep_top_ratio
        # 目标巩固总数 = 总项数 * keep_ratio
        n_target = max(1, int(total * keep_ratio))
        # 已巩固数量
        already_consolidated = sum(1 for it in self.fast.items if it.consolidated)
        # 本次还需巩固的数量
        n_to_consolidate = max(0, n_target - already_consolidated)
        if n_to_consolidate == 0:
            self.log.debug("Consolidation skipped: target already reached (%d/%d)",
                           already_consolidated, n_target)
            return 0
        # 按得分排序，跳过已巩固的
        candidates = [it for it in self.fast.items if not it.consolidated]
        if not candidates:
            self.log.debug("Consolidation skipped: all items already consolidated")
            return 0
        sorted_items = sorted(candidates, key=lambda it: -it.score())
        consolidated = 0
        for it in sorted_items[:n_to_consolidate]:
            # 创建副本添加到慢速记忆（避免引用共享）
            new_item = MemoryItem(it.key, it.value, importance=it.importance,
                                  timestamp=it.timestamp)
            new_item.access_count = it.access_count
            self.slow.add(new_item)
            it.consolidated = True  # 标记已巩固
            consolidated += 1
        self.log.info("Consolidated %d memories to slow memory (slow_size=%d, total=%d, target=%d)",
                      consolidated, len(self.slow), total, n_target)
        return consolidated

    def step(self) -> None:
        """每步调用：检查是否触发巩固。"""
        self.step_count += 1
        if self.step_count % self.config.consolidation_interval == 0:
            self.consolidate()

    def replay(self, n: int = 16) -> List[MemoryItem]:
        """返回 n 条重要记忆用于离线复盘（去重：跳过已巩固的快速记忆）。"""
        # 只取未巩固的快速记忆 + 全部慢速记忆（避免重复）
        fast_items = [it for it in self.fast.items if not it.consolidated]
        all_items = fast_items + list(self.slow.items)
        all_items.sort(key=lambda it: -it.score())
        out = all_items[:n]
        self.log.debug("Replay selected %d memories (fast_unconsolidated=%d slow=%d)",
                       len(out), len(fast_items), len(self.slow.items))
        return out

    def get_state(self) -> Dict[str, Any]:
        return {
            "fast_size": len(self.fast),
            "slow_size": len(self.slow),
            "step_count": self.step_count,
        }
