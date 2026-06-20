"""SPEPA 事件回调与进度通知系统。

设计原则（第一性原理）：
- 观察者原理：用户应能在运行时观察系统内部事件，无需侵入式修改
- 解耦原理：事件发布者不依赖具体的回调实现
- 渐进式原理：用户可以不注册任何回调（零开销），也可以注册多个

事件类型：
- STEP_START    : 每步开始
- STEP_END      : 每步结束（含指标）
- CYCLE_COMPLETE: 一个完整周期结束
- GOAL_GENERATED: 自主目标生成
- GOAL_ACHIEVED : 自主目标达成
- ANOMALY       : 异常检测
- REPAIR        : 自我修复执行
- ERROR         : 运行错误

使用方式：
    agent = SPEPAAgent(config)
    agent.on(EventType.STEP_END, lambda e: print(f"Step {e.data['step']}: err={e.data['error']:.3f}"))
    agent.run(n_steps=100)
"""
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional

from spepa.utils.logger import LoggedMixin


class EventType(Enum):
    """事件类型。"""
    STEP_START = "step_start"
    STEP_END = "step_end"
    CYCLE_COMPLETE = "cycle_complete"
    GOAL_GENERATED = "goal_generated"
    GOAL_ACHIEVED = "goal_achieved"
    ANOMALY = "anomaly"
    REPAIR = "repair"
    ERROR = "error"


@dataclass
class Event:
    """事件对象。"""
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# 回调函数类型：接收 Event 对象
EventCallback = Callable[[Event], None]


class EventBus(LoggedMixin):
    """事件总线：管理事件订阅与发布。

    特性：
    - 多订阅者：同一事件可有多个回调
    - 异常隔离：单个回调异常不影响其他回调和系统运行
    - 零开销：无订阅者时 publish 直接返回
    """

    def __init__(self):
        self._set_log_name("spepa.events")
        self._subscribers: Dict[EventType, List[EventCallback]] = {
            et: [] for et in EventType
        }
        # 统计
        self.total_published = 0
        self.total_delivered = 0
        self.total_errors = 0
        self.log.debug("EventBus initialized")

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """订阅事件。

        参数：
        - event_type: 事件类型
        - callback: 回调函数，接收 Event 对象
        """
        if not callable(callback):
            raise ValueError(f"回调必须是可调用对象，得到 {type(callback)}")
        self._subscribers[event_type].append(callback)
        self.log.debug("Subscribed to %s (total=%d)",
                       event_type.value, len(self._subscribers[event_type]))

    def unsubscribe(self, event_type: EventType, callback: EventCallback) -> bool:
        """取消订阅。返回是否成功移除。"""
        subs = self._subscribers.get(event_type, [])
        try:
            subs.remove(callback)
            return True
        except ValueError:
            return False

    def publish(self, event: Event) -> int:
        """发布事件，返回成功投递的回调数。"""
        self.total_published += 1
        callbacks = self._subscribers.get(event.event_type, [])
        if not callbacks:
            return 0
        delivered = 0
        for cb in callbacks:
            try:
                cb(event)
                delivered += 1
            except Exception as e:
                self.total_errors += 1
                self.log.warning("Event callback error on %s: %s",
                                 event.event_type.value, e)
        self.total_delivered += delivered
        return delivered

    def clear(self) -> None:
        """清空所有订阅。"""
        for subs in self._subscribers.values():
            subs.clear()
        self.log.debug("EventBus cleared")

    def subscriber_count(self, event_type: EventType) -> int:
        """获取某事件的订阅者数量。"""
        return len(self._subscribers.get(event_type, []))

    def get_state(self) -> Dict[str, Any]:
        return {
            "total_published": self.total_published,
            "total_delivered": self.total_delivered,
            "total_errors": self.total_errors,
            "subscriber_counts": {
                et.value: len(subs)
                for et, subs in self._subscribers.items()
            },
        }

    def reset(self) -> None:
        """重置事件总线。"""
        self.clear()
        self.total_published = 0
        self.total_delivered = 0
        self.total_errors = 0


class ProgressReporter:
    """进度报告器：将事件转换为人类可读的进度输出。

    使用方式：
        reporter = ProgressReporter(agent)
        agent.run(n_steps=100)
        # 运行过程中自动打印进度
    """

    def __init__(self, agent, interval: int = 10, show_metrics: bool = True):
        """初始化进度报告器并自动订阅事件。

        参数：
        - agent: SPEPAAgent 实例
        - interval: 每隔多少步打印一次进度
        - show_metrics: 是否显示指标
        """
        self.agent = agent
        self.interval = max(1, interval)
        self.show_metrics = show_metrics
        self._start_time = 0.0
        self._last_print_step = -1
        # 订阅事件
        agent.event_bus.subscribe(EventType.STEP_END, self._on_step_end)
        agent.event_bus.subscribe(EventType.GOAL_ACHIEVED, self._on_goal_achieved)
        agent.event_bus.subscribe(EventType.ANOMALY, self._on_anomaly)

    def start(self, total_steps: int) -> None:
        """开始进度跟踪。"""
        self._start_time = time.time()
        self._last_print_step = -1
        print(f"[SPEPA] 开始运行，共 {total_steps} 步...")

    def _on_step_end(self, event: Event) -> None:
        """步结束回调。"""
        step = event.data.get("step", 0)
        if step % self.interval != 0 and step != self._last_print_step:
            return
        if step == self._last_print_step:
            return
        self._last_print_step = step
        elapsed = time.time() - self._start_time if self._start_time else 0
        parts = [f"[SPEPA] Step {step}"]
        if self.show_metrics:
            error = event.data.get("error", 0.0)
            confidence = event.data.get("confidence", 0.0)
            parts.append(f"err={error:.4f}")
            parts.append(f"conf={confidence:.3f}")
        if elapsed > 0:
            parts.append(f"({elapsed:.1f}s)")
        print(" ".join(parts))

    def _on_goal_achieved(self, event: Event) -> None:
        """目标达成回调。"""
        goal_id = event.data.get("goal_id", "?")
        source = event.data.get("source", "?")
        print(f"[SPEPA] ✓ 目标达成: {goal_id} (来源: {source})")

    def _on_anomaly(self, event: Event) -> None:
        """异常回调。"""
        anomaly = event.data.get("anomaly_type", "unknown")
        severity = event.data.get("severity", 0.0)
        print(f"[SPEPA] ⚠ 异常检测: {anomaly} (严重度={severity:.2f})")

    def finish(self, summary: Dict[str, Any]) -> None:
        """结束进度跟踪。"""
        elapsed = time.time() - self._start_time if self._start_time else 0
        print(f"[SPEPA] 运行完成，耗时 {elapsed:.1f}s")
        if summary:
            for key in ["total_steps", "goals_achieved", "health_score"]:
                if key in summary:
                    print(f"       {key}: {summary[key]}")

    def stop(self) -> None:
        """停止进度报告（取消订阅）。"""
        self.agent.event_bus.unsubscribe(EventType.STEP_END, self._on_step_end)
        self.agent.event_bus.unsubscribe(EventType.GOAL_ACHIEVED, self._on_goal_achieved)
        self.agent.event_bus.unsubscribe(EventType.ANOMALY, self._on_anomaly)
