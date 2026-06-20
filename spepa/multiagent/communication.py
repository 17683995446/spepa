"""SPEPA Agent 间通信总线。

设计原则（第一性原理）：
- 信息共享原理：子Agent的发现应能传播给其他Agent和主Agent
- 解耦原理：发送方和接收方不直接耦合，通过总线间接通信
- 异步原理：消息排队，接收方按需读取，不阻塞发送方

消息类型：
- STATE_SHARE: 共享状态发现（如找到好路径）
- SKILL_SHARE: 共享学到的技能
- HELP_REQUEST: 请求帮助（如卡住时）
- RESULT_REPORT: 报告任务结果
- PROGRESS_UPDATE: 进度更新
"""
import time
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from spepa.utils.logger import LoggedMixin


class MessageType(Enum):
    """消息类型。"""
    STATE_SHARE = "state_share"        # 共享状态发现
    SKILL_SHARE = "skill_share"        # 共享技能
    HELP_REQUEST = "help_request"      # 请求帮助
    RESULT_REPORT = "result_report"    # 报告结果
    PROGRESS_UPDATE = "progress_update"  # 进度更新


@dataclass
class Message:
    """Agent 间消息。"""
    sender_id: str
    msg_type: MessageType
    content: Dict[str, Any]
    receiver_id: str = "broadcast"  # "broadcast" = 广播
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0  # 消息优先级

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "msg_type": self.msg_type.value,
            "receiver_id": self.receiver_id,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "content_keys": list(self.content.keys()),
        }


class MessageBus(LoggedMixin):
    """Agent 间通信总线。

    提供 publish/subscribe 模式的消息传递：
    - publish: 发送消息（单播或广播）
    - receive: 接收消息（从个人队列读取）
    - broadcast: 广播给所有订阅者
    """

    def __init__(self, max_queue_size: int = 100):
        self._set_log_name("spepa.multiagent.communication")
        # 每个 agent 的消息队列
        self._queues: Dict[str, deque] = {}
        # 已注册的 agent ID
        self._registered: set = set()
        # 统计
        self.total_sent = 0
        self.total_received = 0
        self.total_dropped = 0
        self.max_queue_size = max_queue_size
        self.log.info("MessageBus initialized (max_queue=%d)", max_queue_size)

    def register(self, agent_id: str) -> bool:
        """注册一个 agent。"""
        if agent_id in self._registered:
            return False
        self._registered.add(agent_id)
        self._queues[agent_id] = deque(maxlen=self.max_queue_size)
        self.log.debug("Registered agent: %s", agent_id)
        return True

    def unregister(self, agent_id: str) -> bool:
        """注销一个 agent。"""
        if agent_id not in self._registered:
            return False
        self._registered.discard(agent_id)
        self._queues.pop(agent_id, None)
        self.log.debug("Unregistered agent: %s", agent_id)
        return True

    def publish(self, message: Message) -> int:
        """发布消息。

        返回成功投递的接收方数量。
        """
        if message.sender_id not in self._registered:
            self.log.warning("Unknown sender: %s", message.sender_id)
            return 0
        delivered = 0
        if message.receiver_id == "broadcast":
            # 广播给所有已注册 agent（除发送方）
            for agent_id in self._registered:
                if agent_id == message.sender_id:
                    continue
                if self._enqueue(agent_id, message):
                    delivered += 1
        else:
            # 单播
            if message.receiver_id in self._registered:
                if self._enqueue(message.receiver_id, message):
                    delivered = 1
            else:
                self.log.warning("Unknown receiver: %s", message.receiver_id)
        self.total_sent += 1
        return delivered

    def _enqueue(self, agent_id: str, message: Message) -> bool:
        """将消息放入 agent 的队列。"""
        queue = self._queues.get(agent_id)
        if queue is None:
            return False
        if len(queue) >= self.max_queue_size:
            # 队列满，丢弃最旧的消息
            queue.popleft()
            self.total_dropped += 1
        queue.append(message)
        return True

    def receive(self, agent_id: str, max_messages: int = 10) -> List[Message]:
        """接收消息（从队列中取出）。"""
        if agent_id not in self._registered:
            return []
        queue = self._queues.get(agent_id)
        if queue is None or len(queue) == 0:
            return []
        messages = []
        count = min(max_messages, len(queue))
        for _ in range(count):
            messages.append(queue.popleft())
        self.total_received += len(messages)
        return messages

    def peek(self, agent_id: str) -> int:
        """查看队列中待处理消息数（不取出）。"""
        queue = self._queues.get(agent_id)
        return len(queue) if queue else 0

    def get_registered_agents(self) -> List[str]:
        """获取已注册的 agent 列表。"""
        return sorted(self._registered)

    def clear(self) -> None:
        """清空所有消息队列。"""
        for queue in self._queues.values():
            queue.clear()
        self.log.info("MessageBus cleared")

    def get_state(self) -> Dict[str, Any]:
        return {
            "registered_count": len(self._registered),
            "total_sent": self.total_sent,
            "total_received": self.total_received,
            "total_dropped": self.total_dropped,
            "pending_messages": sum(len(q) for q in self._queues.values()),
        }

    def reset(self) -> None:
        """重置总线。"""
        self._queues.clear()
        self._registered.clear()
        self.total_sent = 0
        self.total_received = 0
        self.total_dropped = 0
        self.log.info("MessageBus reset")
