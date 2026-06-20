"""SPEPA 背压机制：过载信号传播。

设计原则（第一性原理）：
- 反压原理：当下游组件过载时，应向上游发送反压信号，让上游减速
- 传播原理：背压信号应沿数据流方向反向传播
- 累积原理：多个下游的背压信号应取最大值（最严格的约束）
- 衰减原理：背压信号在传播过程中应适当衰减（避免过度反应）

背压级别：
- NONE: 无背压，全速运行
- LOW: 轻度背压，减少非关键计算
- MEDIUM: 中度背压，跳过可选操作
- HIGH: 高度背压，只保留核心流程
- CRITICAL: 临界背压，最小化计算
"""
import time
from enum import IntEnum
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import deque

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class BackpressureLevel(IntEnum):
    """背压级别（值越大越严重）。"""
    NONE = 0      # 无背压
    LOW = 1       # 轻度
    MEDIUM = 2    # 中度
    HIGH = 3      # 高度
    CRITICAL = 4  # 临界


@dataclass
class BackpressureSignal:
    """单个背压信号。"""
    source: str           # 信号来源（组件名）
    level: BackpressureLevel
    reason: str = ""      # 背压原因
    timestamp: float = field(default_factory=time.time)
    # 衰减因子（传播时使用，0-1）
    decay: float = 1.0


class Backpressure(LoggedMixin):
    """背压管理器：收集、聚合、传播背压信号。

    工作流程：
    1. 各组件调用 signal() 报告自身负载状态
    2. 背压管理器聚合所有信号（取最大级别）
    3. 上游组件通过 get_pressure() 查询当前背压级别
    4. 根据背压级别调整自身行为
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.resource.backpressure")
        self.config = config
        # 当前各组件的背压信号
        self._signals: Dict[str, BackpressureSignal] = {}
        # 聚合后的当前背压级别
        self._current_level: BackpressureLevel = BackpressureLevel.NONE
        # 背压历史
        self._history: deque = deque(maxlen=100)
        # 信号超时时间（秒）：超过此时间未更新的信号被视为过期
        self._signal_timeout = 30.0
        # 传播衰减率（每跳衰减）
        self._propagation_decay = 0.9
        self.log.info("Backpressure manager initialized")

    def signal(self, source: str, level: BackpressureLevel,
               reason: str = "") -> None:
        """报告背压信号。

        参数：
        - source: 信号来源（组件名，如 "perception", "learning"）
        - level: 背压级别
        - reason: 背压原因
        """
        self._signals[source] = BackpressureSignal(
            source=source, level=level, reason=reason
        )
        # 重新聚合
        self._aggregate()
        if level >= BackpressureLevel.MEDIUM:
            self.log.debug("Backpressure from %s: %s (%s)",
                           source, level.name, reason)

    def clear_signal(self, source: str) -> None:
        """清除某个组件的背压信号。"""
        if source in self._signals:
            del self._signals[source]
            self._aggregate()

    def _aggregate(self) -> None:
        """聚合所有信号（取最大级别），清除过期信号。"""
        now = time.time()
        # 清除过期信号
        expired = [src for src, sig in self._signals.items()
                   if now - sig.timestamp > self._signal_timeout]
        for src in expired:
            del self._signals[src]
        # 聚合：取最大级别
        if not self._signals:
            new_level = BackpressureLevel.NONE
        else:
            new_level = max(sig.level for sig in self._signals.values())
        # 记录级别变化
        if new_level != self._current_level:
            self.log.info("Backpressure level changed: %s -> %s",
                          self._current_level.name, new_level.name)
            self._current_level = new_level
            self._history.append({
                "level": new_level.value,
                "level_name": new_level.name,
                "sources": {s: sig.level.name for s, sig in self._signals.items()},
                "timestamp": now,
            })

    def get_pressure(self) -> BackpressureLevel:
        """获取当前聚合背压级别。"""
        self._aggregate()  # 清除过期信号
        return self._current_level

    def get_pressure_for(self, component: str) -> BackpressureLevel:
        """获取针对特定组件的背压级别（考虑传播衰减）。

        参数：
        - component: 查询的组件名
        """
        self._aggregate()
        if not self._signals:
            return BackpressureLevel.NONE
        # 排除自身的信号（不背压自己）
        external_signals = [sig for src, sig in self._signals.items()
                           if src != component]
        if not external_signals:
            return BackpressureLevel.NONE
        # 取最大级别，但应用传播衰减
        max_level = max(sig.level for sig in external_signals)
        # 衰减：每级衰减一次
        decayed = int(max_level * self._propagation_decay)
        return BackpressureLevel(decayed)

    def is_overloaded(self) -> bool:
        """系统是否过载（MEDIUM 及以上）。"""
        return self.get_pressure() >= BackpressureLevel.MEDIUM

    def is_critical(self) -> bool:
        """系统是否处于临界状态。"""
        return self.get_pressure() >= BackpressureLevel.CRITICAL

    def get_active_sources(self) -> Dict[str, str]:
        """获取当前活跃的背压源及其级别。"""
        self._aggregate()
        return {src: sig.level.name for src, sig in self._signals.items()}

    def reset(self) -> None:
        """重置背压管理器。"""
        self._signals.clear()
        self._current_level = BackpressureLevel.NONE
        self._history.clear()
        self.log.info("Backpressure reset")

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_level": self._current_level.name,
            "current_level_value": self._current_level.value,
            "is_overloaded": self.is_overloaded(),
            "is_critical": self.is_critical(),
            "active_sources": self.get_active_sources(),
            "history_size": len(self._history),
        }

    def get_history(self, n: int = 20) -> list:
        return list(self._history)[-n:]
