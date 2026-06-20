"""SPEPA 全局状态总线（发布-订阅模式）。

所有模块通过状态总线通信，解耦模块间依赖。每次发布/订阅都会记录日志。
"""
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional

from spepa.utils.logger import get_logger, LoggedMixin


class StateBus(LoggedMixin):
    """线程安全的发布-订阅状态总线。

    - publish(topic, payload): 发布消息
    - subscribe(topic, callback): 订阅主题
    - get_state(key): 获取最新状态
    - set_state(key, value): 设置全局状态
    """

    def __init__(self, history_size: int = 1000):
        self._set_log_name("spepa.core.state_bus")
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._state: Dict[str, Any] = {}
        self._history: deque = deque(maxlen=history_size)
        self._history_size = history_size
        self.log.info("StateBus initialized (history_size=%d)", history_size)

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """订阅某个主题。"""
        with self._lock:
            self._subscribers[topic].append(callback)
        self.log.info("Subscribed to topic='%s' (total=%d)", topic, len(self._subscribers[topic]))

    def unsubscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        """取消订阅。"""
        with self._lock:
            if topic in self._subscribers and callback in self._subscribers[topic]:
                self._subscribers[topic].remove(callback)
                self.log.info("Unsubscribed from topic='%s'", topic)

    def publish(self, topic: str, payload: Any) -> int:
        """向某主题发布消息，返回被通知的订阅者数量。"""
        notified = 0
        with self._lock:
            subs = list(self._subscribers.get(topic, []))
            self._history.append({
                "topic": topic,
                "payload": payload,
                "ts": time.time(),
            })
        # 在锁外调用回调，避免死锁
        for cb in subs:
            try:
                cb(payload)
                notified += 1
            except Exception as e:
                self.log.error("Subscriber error on topic='%s': %s", topic, e)
        self.log.debug("Published topic='%s' notified=%d", topic, notified)
        return notified

    def set_state(self, key: str, value: Any) -> None:
        """设置全局状态键值。"""
        with self._lock:
            old = self._state.get(key)
            self._state[key] = value
        self.log.debug("State set key='%s' old=%s new=%s", key, _short(old), _short(value))

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取全局状态键值。"""
        with self._lock:
            val = self._state.get(key, default)
        self.log.debug("State get key='%s' val=%s", key, _short(val))
        return val

    def update_state(self, updates: Dict[str, Any]) -> None:
        """批量更新状态。"""
        with self._lock:
            self._state.update(updates)
        self.log.info("State updated keys=%s", list(updates.keys()))

    def snapshot(self) -> Dict[str, Any]:
        """返回当前状态的快照。"""
        with self._lock:
            snap = dict(self._state)
        self.log.debug("State snapshot taken (%d keys)", len(snap))
        return snap

    def history(self, topic: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """返回历史消息。"""
        with self._lock:
            items = list(self._history)
        if topic is not None:
            items = [it for it in items if it["topic"] == topic]
        items = items[-limit:]
        self.log.debug("History queried topic=%s returned=%d", topic, len(items))
        return items

    def clear(self) -> None:
        """清空状态与历史。"""
        with self._lock:
            self._state.clear()
            self._history.clear()
        self.log.info("StateBus cleared")


def _short(val: Any, maxlen: int = 50) -> str:
    """将值转为简短字符串用于日志。"""
    s = repr(val)
    if len(s) > maxlen:
        s = s[:maxlen] + "..."
    return s
