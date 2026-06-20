"""SPEPA 易用性阶段三测试：事件回调与进度通知。

测试范围：
1. EventBus：订阅/发布/取消订阅
2. 事件类型：STEP_END/CYCLE_COMPLETE 等
3. 异常隔离：回调异常不影响系统
4. Agent 集成：on() 方法
5. ProgressReporter：进度报告
"""
import pytest
import numpy as np

from spepa.events import EventBus, EventType, Event, ProgressReporter
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_default_config


# ============================================================
# EventBus 测试
# ============================================================

class TestEventBus:
    """事件总线测试。"""

    def test_initialization(self):
        """正确初始化。"""
        bus = EventBus()
        assert bus.total_published == 0
        assert bus.total_delivered == 0

    def test_subscribe_and_publish(self):
        """订阅并发布事件。"""
        bus = EventBus()
        received = []
        bus.subscribe(EventType.STEP_END, lambda e: received.append(e))
        event = Event(EventType.STEP_END, {"step": 1})
        delivered = bus.publish(event)
        assert delivered == 1
        assert len(received) == 1
        assert received[0].data["step"] == 1
        assert bus.total_published == 1
        assert bus.total_delivered == 1

    def test_multiple_subscribers(self):
        """多个订阅者都收到事件。"""
        bus = EventBus()
        received1 = []
        received2 = []
        bus.subscribe(EventType.STEP_END, lambda e: received1.append(e))
        bus.subscribe(EventType.STEP_END, lambda e: received2.append(e))
        bus.publish(Event(EventType.STEP_END, {"step": 1}))
        assert len(received1) == 1
        assert len(received2) == 1

    def test_no_subscribers_zero_overhead(self):
        """无订阅者时 publish 返回 0。"""
        bus = EventBus()
        delivered = bus.publish(Event(EventType.STEP_END, {}))
        assert delivered == 0
        assert bus.total_published == 1
        assert bus.total_delivered == 0

    def test_unsubscribe(self):
        """取消订阅。"""
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.subscribe(EventType.STEP_END, cb)
        bus.publish(Event(EventType.STEP_END, {}))
        assert len(received) == 1
        bus.unsubscribe(EventType.STEP_END, cb)
        bus.publish(Event(EventType.STEP_END, {}))
        assert len(received) == 1  # 不再收到

    def test_callback_exception_isolated(self):
        """回调异常不影响其他回调和系统。"""
        bus = EventBus()
        received = []
        def bad_callback(e):
            raise RuntimeError("故意出错")
        bus.subscribe(EventType.STEP_END, bad_callback)
        bus.subscribe(EventType.STEP_END, lambda e: received.append(e))
        delivered = bus.publish(Event(EventType.STEP_END, {}))
        # 异常回调不计入 delivered，但正常回调仍执行
        assert delivered == 1
        assert len(received) == 1
        assert bus.total_errors == 1

    def test_subscribe_non_callable_raises(self):
        """非可调用对象订阅报错。"""
        bus = EventBus()
        with pytest.raises(ValueError, match="可调用"):
            bus.subscribe(EventType.STEP_END, "not_a_function")

    def test_subscriber_count(self):
        """订阅者计数。"""
        bus = EventBus()
        bus.subscribe(EventType.STEP_END, lambda e: None)
        bus.subscribe(EventType.STEP_END, lambda e: None)
        assert bus.subscriber_count(EventType.STEP_END) == 2
        assert bus.subscriber_count(EventType.CYCLE_COMPLETE) == 0

    def test_clear(self):
        """清空所有订阅。"""
        bus = EventBus()
        bus.subscribe(EventType.STEP_END, lambda e: None)
        bus.clear()
        assert bus.subscriber_count(EventType.STEP_END) == 0

    def test_get_state(self):
        """get_state 返回完整状态。"""
        bus = EventBus()
        state = bus.get_state()
        assert "total_published" in state
        assert "total_delivered" in state
        assert "subscriber_counts" in state

    def test_reset(self):
        """reset 清空所有状态。"""
        bus = EventBus()
        bus.subscribe(EventType.STEP_END, lambda e: None)
        bus.total_published = 5
        bus.reset()
        assert bus.total_published == 0
        assert bus.subscriber_count(EventType.STEP_END) == 0


# ============================================================
# Event 对象测试
# ============================================================

class TestEvent:
    """事件对象测试。"""

    def test_event_creation(self):
        """事件创建。"""
        event = Event(EventType.STEP_END, {"step": 1, "error": 0.5})
        assert event.event_type == EventType.STEP_END
        assert event.data["step"] == 1
        assert event.timestamp > 0

    def test_event_to_dict(self):
        """事件转字典。"""
        event = Event(EventType.GOAL_ACHIEVED, {"goal_id": "g1"})
        d = event.to_dict()
        assert d["event_type"] == "goal_achieved"
        assert d["data"]["goal_id"] == "g1"
        assert "timestamp" in d


# ============================================================
# Agent 集成测试
# ============================================================

class TestAgentEventIntegration:
    """Agent 事件集成测试。"""

    def test_agent_has_event_bus(self):
        """Agent 包含事件总线。"""
        agent = SPEPAAgent(get_default_config())
        assert hasattr(agent, 'event_bus')
        assert isinstance(agent.event_bus, EventBus)

    def test_on_method_registers_callback(self):
        """on() 方法注册回调。"""
        agent = SPEPAAgent(get_default_config())
        received = []
        agent.on(EventType.STEP_END, lambda e: received.append(e))
        assert agent.event_bus.subscriber_count(EventType.STEP_END) == 1

    def test_step_end_event_fired(self):
        """online_cycle 触发 STEP_END 事件。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        received = []
        agent.on(EventType.STEP_END, lambda e: received.append(e))
        agent.online_cycle()
        assert len(received) == 1
        assert "step" in received[0].data
        assert "error" in received[0].data

    def test_cycle_complete_event_fired(self):
        """run() 触发 CYCLE_COMPLETE 事件。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        received = []
        agent.on(EventType.CYCLE_COMPLETE, lambda e: received.append(e))
        agent.run(n_steps=3)
        assert len(received) == 1
        assert received[0].data["total_steps"] == 3

    def test_multiple_events_during_run(self):
        """运行过程中触发多个 STEP_END 事件。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        steps = []
        agent.on(EventType.STEP_END, lambda e: steps.append(e.data["step"]))
        agent.run(n_steps=5)
        assert len(steps) == 5
        assert steps == [1, 2, 3, 4, 5]

    def test_event_data_contains_metrics(self):
        """事件数据包含关键指标。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        event_data = []
        agent.on(EventType.STEP_END, lambda e: event_data.append(e.data))
        agent.online_cycle()
        data = event_data[0]
        assert "error" in data
        assert "reward" in data
        assert "confidence" in data
        assert "cycle_time" in data

    def test_get_state_includes_event_bus(self):
        """get_state 包含事件总线状态。"""
        agent = SPEPAAgent(get_default_config())
        state = agent.get_state()
        assert "event_bus" in state

    def test_reset_resets_event_bus(self):
        """reset 重置事件总线。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        agent.on(EventType.STEP_END, lambda e: None)
        agent.event_bus.total_published = 10
        agent.reset()
        assert agent.event_bus.total_published == 0
        assert agent.event_bus.subscriber_count(EventType.STEP_END) == 0


# ============================================================
# ProgressReporter 测试
# ============================================================

class TestProgressReporter:
    """进度报告器测试。"""

    def test_initialization(self):
        """正确初始化。"""
        agent = SPEPAAgent(get_default_config())
        reporter = ProgressReporter(agent, interval=5)
        assert reporter.interval == 5
        # 应已订阅事件
        assert agent.event_bus.subscriber_count(EventType.STEP_END) >= 1

    def test_start_prints_message(self, capsys):
        """start 打印开始消息。"""
        agent = SPEPAAgent(get_default_config())
        reporter = ProgressReporter(agent)
        reporter.start(100)
        captured = capsys.readouterr()
        assert "100" in captured.out

    def test_step_end_prints_progress(self, capsys):
        """步结束打印进度。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        reporter = ProgressReporter(agent, interval=1)
        reporter.start(5)
        agent.online_cycle()
        captured = capsys.readouterr()
        assert "Step" in captured.out or "err" in captured.out

    def test_goal_achieved_prints(self, capsys):
        """目标达成打印。"""
        agent = SPEPAAgent(get_default_config())
        reporter = ProgressReporter(agent)
        agent.event_bus.publish(Event(EventType.GOAL_ACHIEVED, {
            "goal_id": "g1", "source": "homeostasis"
        }))
        captured = capsys.readouterr()
        assert "目标达成" in captured.out

    def test_anomaly_prints(self, capsys):
        """异常打印。"""
        agent = SPEPAAgent(get_default_config())
        reporter = ProgressReporter(agent)
        agent.event_bus.publish(Event(EventType.ANOMALY, {
            "anomaly_type": "error_spike", "severity": 0.8
        }))
        captured = capsys.readouterr()
        assert "异常" in captured.out

    def test_finish_prints_summary(self, capsys):
        """finish 打印摘要。"""
        agent = SPEPAAgent(get_default_config())
        reporter = ProgressReporter(agent)
        reporter.start(10)
        reporter.finish({"total_steps": 10, "goals_achieved": 2})
        captured = capsys.readouterr()
        assert "完成" in captured.out

    def test_stop_unsubscribes(self):
        """stop 取消订阅。"""
        agent = SPEPAAgent(get_default_config())
        reporter = ProgressReporter(agent)
        count_before = agent.event_bus.subscriber_count(EventType.STEP_END)
        reporter.stop()
        count_after = agent.event_bus.subscriber_count(EventType.STEP_END)
        assert count_after < count_before


# ============================================================
# 端到端测试
# ============================================================

class TestEventEndToEnd:
    """事件端到端测试。"""

    def test_monitor_during_run(self):
        """运行过程中监控事件。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        events = []
        agent.on(EventType.STEP_END, lambda e: events.append(e))
        agent.on(EventType.CYCLE_COMPLETE, lambda e: events.append(e))
        agent.run(n_steps=5)
        # 5个STEP_END + 1个CYCLE_COMPLETE
        assert len(events) == 6
        step_events = [e for e in events if e.event_type == EventType.STEP_END]
        complete_events = [e for e in events if e.event_type == EventType.CYCLE_COMPLETE]
        assert len(step_events) == 5
        assert len(complete_events) == 1

    def test_callback_receives_correct_data(self):
        """回调接收正确的数据。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        errors = []
        agent.on(EventType.STEP_END, lambda e: errors.append(e.data["error"]))
        agent.run(n_steps=3)
        assert len(errors) == 3
        assert all(isinstance(e, float) for e in errors)

    def test_progress_reporter_full_cycle(self, capsys):
        """ProgressReporter 完整周期。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        reporter = ProgressReporter(agent, interval=2)
        reporter.start(4)
        agent.run(n_steps=4)
        reporter.finish({"total_steps": 4})
        captured = capsys.readouterr()
        assert "开始" in captured.out
        assert "完成" in captured.out
