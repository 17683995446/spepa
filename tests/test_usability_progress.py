"""SPEPA 进度通知系统测试。

测试范围：
- ProgressInfo 数据结构
- ProgressTracker（start/update/increment/finish/reset/回调/节流）
- ProgressBar（渲染/输出）
- AgentProgressTracker（EventBus 集成）
- 便捷函数与上下文管理器
- format_progress
- 端到端
"""
import time
import pytest
from spepa.progress import (
    ProgressInfo, ProgressTracker, ProgressBar, AgentProgressTracker,
    track_progress, track_agent_progress, format_progress,
)
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_preset_config


# ============================================================
# ProgressInfo 数据结构
# ============================================================

class TestProgressInfo:
    """ProgressInfo 数据结构测试。"""

    def test_create_default(self):
        info = ProgressInfo()
        assert info.current == 0
        assert info.total == 0
        assert info.elapsed == 0.0
        assert info.eta == 0.0
        assert info.percent == 0.0
        assert info.finished is False

    def test_create_with_values(self):
        info = ProgressInfo(current=50, total=100, elapsed=5.0, percent=50.0)
        assert info.current == 50
        assert info.total == 100
        assert info.elapsed == 5.0
        assert info.percent == 50.0

    def test_as_dict(self):
        info = ProgressInfo(current=5, total=10, elapsed=2.0, percent=50.0)
        d = info.as_dict()
        assert d["current"] == 5
        assert d["total"] == 10
        assert d["percent"] == 50.0
        assert "elapsed" in d
        assert "eta" in d
        assert "speed" in d
        assert "finished" in d


# ============================================================
# ProgressTracker
# ============================================================

class TestProgressTracker:
    """进度追踪器测试。"""

    def test_start(self):
        tracker = ProgressTracker(total=100)
        tracker.start()
        assert tracker.info.total == 100
        assert tracker.info.started_at > 0
        assert tracker.info.current == 0

    def test_update(self):
        tracker = ProgressTracker(total=100)
        tracker.start()
        tracker.update(50)
        assert tracker.info.current == 50
        assert tracker.info.percent == 50.0

    def test_update_calculates_speed(self):
        tracker = ProgressTracker(total=100)
        tracker.start()
        time.sleep(0.01)
        tracker.update(10)
        assert tracker.info.speed > 0

    def test_update_calculates_eta(self):
        tracker = ProgressTracker(total=100)
        tracker.start()
        time.sleep(0.01)
        tracker.update(10)
        assert tracker.info.eta > 0

    def test_update_with_new_total(self):
        tracker = ProgressTracker(total=100)
        tracker.start()
        tracker.update(50, total=200)
        assert tracker.info.total == 200
        assert tracker.info.percent == 25.0  # 50/200

    def test_increment(self):
        tracker = ProgressTracker(total=10)
        tracker.start()
        tracker.update(5)
        tracker.increment()
        assert tracker.info.current == 6
        tracker.increment(2)
        assert tracker.info.current == 8

    def test_finish(self):
        tracker = ProgressTracker(total=100)
        tracker.start()
        tracker.update(50)
        tracker.finish()
        assert tracker.info.finished is True
        assert tracker.info.current == 100
        assert tracker.info.percent == 100.0
        assert tracker.info.eta == 0.0

    def test_finish_unknown_total(self):
        tracker = ProgressTracker(total=0)
        tracker.start()
        tracker.update(50)
        tracker.finish()
        assert tracker.info.finished is True

    def test_reset(self):
        tracker = ProgressTracker(total=100)
        tracker.start()
        tracker.update(50)
        tracker.reset()
        assert tracker.info.current == 0
        assert tracker.info.total == 100

    def test_callback_called(self):
        calls = []
        tracker = ProgressTracker(total=10, on_update=lambda info: calls.append(info.current))
        tracker.start()
        tracker.update(5)
        assert len(calls) >= 2  # start + update
        assert 5 in calls

    def test_callback_exception_ignored(self):
        def bad_callback(info):
            raise RuntimeError("callback error")
        tracker = ProgressTracker(total=10, on_update=bad_callback)
        tracker.start()  # 不应抛异常
        tracker.update(5)  # 不应抛异常

    def test_update_interval_throttling(self):
        calls = []
        tracker = ProgressTracker(
            total=100,
            on_update=lambda info: calls.append(info.current),
            update_interval=1.0,  # 1 秒间隔
        )
        tracker.start()
        tracker.update(10)
        tracker.update(20)
        tracker.update(30)
        # 由于间隔 1 秒，中间的 update 可能被节流
        # 但 start 和最后一次应触发
        assert len(calls) >= 1

    def test_update_without_start_auto_starts(self):
        tracker = ProgressTracker(total=10)
        tracker.update(5)  # 未调用 start，应自动开始
        assert tracker.info.current == 5

    def test_zero_total_no_division_error(self):
        tracker = ProgressTracker(total=0)
        tracker.start()
        tracker.update(5)  # 不应抛除零异常
        assert tracker.info.current == 5


# ============================================================
# ProgressBar
# ============================================================

class TestProgressBar:
    """进度条测试。"""

    def test_create_bar(self):
        outputs = []
        bar = ProgressBar(total=100, output_func=outputs.append)
        assert bar.total == 100
        assert bar.width >= 10

    def test_start_renders(self):
        outputs = []
        bar = ProgressBar(total=100, output_func=outputs.append)
        bar.start()
        assert len(outputs) > 0

    def test_update_renders(self):
        outputs = []
        bar = ProgressBar(total=100, output_func=outputs.append)
        bar.start()
        bar.update(50)
        assert len(outputs) >= 2

    def test_finish_adds_newline(self):
        outputs = []
        bar = ProgressBar(total=100, output_func=outputs.append)
        bar.start()
        bar.update(100)
        bar.finish()
        # 最后应有换行
        assert any("\n" in o for o in outputs)

    def test_render_contains_percent(self):
        outputs = []
        bar = ProgressBar(total=100, output_func=outputs.append)
        bar.start()
        bar.update(50)
        last_output = outputs[-1]
        assert "50.0" in last_output or "50" in last_output

    def test_render_contains_bar_chars(self):
        outputs = []
        bar = ProgressBar(total=100, output_func=outputs.append)
        bar.start()
        bar.update(50)
        last_output = outputs[-1]
        assert "=" in last_output or "-" in last_output

    def test_label_in_output(self):
        outputs = []
        bar = ProgressBar(total=100, label="任务", output_func=outputs.append)
        bar.start()
        assert any("任务" in o for o in outputs)

    def test_unknown_total_spinner(self):
        outputs = []
        bar = ProgressBar(total=0, output_func=outputs.append)
        bar.start()
        bar.update(1)
        # 未知总量应显示旋转动画
        last_output = outputs[-1]
        assert any(c in last_output for c in "|/-\\")

    def test_increment(self):
        outputs = []
        bar = ProgressBar(total=10, output_func=outputs.append)
        bar.start()
        bar.update(5)
        bar.increment()
        assert bar.tracker.info.current == 6

    def test_custom_width(self):
        bar = ProgressBar(total=100, width=20)
        assert bar.width == 20

    def test_min_width(self):
        bar = ProgressBar(total=100, width=5)  # 太小
        assert bar.width >= 10


# ============================================================
# AgentProgressTracker
# ============================================================

class TestAgentProgressTracker:
    """Agent 进度追踪器测试。"""

    def test_create(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        tracker = AgentProgressTracker(agent, total_steps=10)
        assert tracker.total_steps == 10
        assert tracker.agent is agent

    def test_start_subscribes(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        tracker = AgentProgressTracker(agent, total_steps=10)
        tracker.start()
        assert tracker._handler is not None
        tracker.finish()

    def test_finish_unsubscribes(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        tracker = AgentProgressTracker(agent, total_steps=10)
        tracker.start()
        tracker.finish()
        assert tracker._handler is None

    def test_tracks_agent_run(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        tracker = AgentProgressTracker(agent, total_steps=5)
        tracker.start()
        agent.run(n_steps=5)
        tracker.finish()
        assert tracker.info.current >= 5
        assert tracker.info.finished is True

    def test_callback_called_on_run(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        calls = []
        tracker = AgentProgressTracker(
            agent, total_steps=5,
            on_update=lambda info: calls.append(info.current),
        )
        tracker.start()
        agent.run(n_steps=5)
        tracker.finish()
        assert len(calls) > 0

    def test_default_total_from_config(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        tracker = AgentProgressTracker(agent)  # 不指定 total_steps
        assert tracker.total_steps == agent.config.max_steps

    def test_info_property(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        tracker = AgentProgressTracker(agent, total_steps=10)
        tracker.start()
        assert isinstance(tracker.info, ProgressInfo)
        tracker.finish()


# ============================================================
# 上下文管理器
# ============================================================

class TestContextManagers:
    """上下文管理器测试。"""

    def test_track_progress_basic(self):
        with track_progress(total=10) as tracker:
            for i in range(10):
                tracker.update(i + 1)
        assert tracker.info.finished is True
        assert tracker.info.current == 10

    def test_track_progress_with_callback(self):
        calls = []
        with track_progress(total=5, on_update=lambda info: calls.append(info.current)) as tracker:
            for i in range(5):
                tracker.update(i + 1)
        assert len(calls) > 0

    def test_track_progress_with_bar(self):
        outputs = []
        # 通过修改 bar 的 output_func 来捕获
        with track_progress(total=5, show_bar=True) as tracker:
            # show_bar=True 会创建内部 bar，无法直接注入 output_func
            for i in range(5):
                tracker.update(i + 1)
        assert tracker.info.finished is True

    def test_track_agent_progress(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        with track_agent_progress(agent, total_steps=5) as tracker:
            agent.run(n_steps=5)
        assert tracker.info.finished is True
        assert tracker.info.current >= 5

    def test_track_progress_zero_total(self):
        with track_progress(total=0) as tracker:
            tracker.update(5)
        assert tracker.info.finished is True


# ============================================================
# format_progress
# ============================================================

class TestFormatProgress:
    """格式化进度信息测试。"""

    def test_format_in_progress(self):
        info = ProgressInfo(current=50, total=100, elapsed=5.0, percent=50.0, eta=5.0)
        text = format_progress(info)
        assert "进行中" in text
        assert "50" in text
        assert "100" in text

    def test_format_finished(self):
        info = ProgressInfo(current=100, total=100, elapsed=10.0, percent=100.0, finished=True)
        text = format_progress(info)
        assert "完成" in text

    def test_format_verbose(self):
        info = ProgressInfo(current=50, total=100, elapsed=5.0, percent=50.0, speed=10.0)
        text = format_progress(info, verbose=True)
        assert "速度" in text

    def test_format_non_verbose_no_speed(self):
        info = ProgressInfo(current=50, total=100, elapsed=5.0, percent=50.0, speed=10.0)
        text = format_progress(info, verbose=False)
        assert "速度" not in text

    def test_format_contains_eta(self):
        info = ProgressInfo(current=50, total=100, eta=5.0)
        text = format_progress(info)
        assert "预计剩余" in text

    def test_format_no_eta_when_zero(self):
        info = ProgressInfo(current=50, total=100, eta=0.0)
        text = format_progress(info)
        assert "预计剩余" not in text


# ============================================================
# 端到端
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_full_tracking_workflow(self):
        """完整追踪工作流。"""
        tracker = ProgressTracker(total=10)
        tracker.start()
        for i in range(1, 11):
            tracker.update(i)
        tracker.finish()
        text = format_progress(tracker.info)
        assert "完成" in text
        assert tracker.info.percent == 100.0

    def test_agent_tracking_workflow(self):
        """Agent 追踪工作流。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        with track_agent_progress(agent, total_steps=5) as tracker:
            agent.run(n_steps=5)
        assert tracker.info.finished is True
        text = format_progress(tracker.info)
        assert "完成" in text

    def test_progress_bar_workflow(self):
        """进度条工作流。"""
        outputs = []
        bar = ProgressBar(total=10, output_func=outputs.append, label="测试")
        bar.start()
        for i in range(1, 11):
            bar.update(i)
        bar.finish()
        assert len(outputs) > 0
        assert any("测试" in o for o in outputs)

    def test_callback_receives_updates(self):
        """回调接收更新。"""
        updates = []
        tracker = ProgressTracker(total=5, on_update=lambda info: updates.append(info.as_dict()))
        tracker.start()
        for i in range(1, 6):
            tracker.update(i)
        tracker.finish()
        # 应有多次更新
        assert len(updates) >= 2
        # 最后一次应为完成状态
        assert updates[-1]["finished"] is True

    def test_reset_and_reuse(self):
        """重置后可复用。"""
        tracker = ProgressTracker(total=10)
        tracker.start()
        tracker.update(5)
        tracker.reset()
        tracker.start()
        tracker.update(3)
        assert tracker.info.current == 3
        assert tracker.info.total == 10
