"""SPEPA 进度通知系统。

设计原则（第一性原理）：
- 实时性：基于 EventBus 事件实时追踪进度，而非轮询
- 可估算：根据已用时间估算剩余时间
- 可定制：支持多种通知方式（回调、打印、日志）
- 非侵入：通过事件订阅实现，不修改 Agent 核心逻辑

与 batch_scheduler 的 on_progress 的区别：
- batch_scheduler.on_progress : 任务级进度（哪个任务完成了）
- progress            : 步级进度（当前任务运行到第几步，预计多久完成）

主要功能：
- ProgressTracker   : 进度追踪器
- ProgressInfo      : 进度信息数据结构
- ProgressBar       : 终端进度条
- track_progress()  : 上下文管理器，便捷追踪
- format_progress() : 格式化进度信息
"""
import time
import sys
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

from spepa.utils.logger import get_logger
from spepa.events import EventType, Event

logger = get_logger("spepa.progress")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ProgressInfo:
    """进度信息。"""
    current: int = 0          # 当前进度
    total: int = 0            # 总量
    elapsed: float = 0.0      # 已用时间（秒）
    eta: float = 0.0          # 预计剩余时间（秒）
    speed: float = 0.0        # 速度（单位/秒）
    percent: float = 0.0      # 百分比（0-100）
    started_at: float = 0.0   # 开始时间戳
    finished: bool = False    # 是否完成

    def as_dict(self) -> Dict[str, Any]:
        """转为字典。"""
        return {
            "current": self.current,
            "total": self.total,
            "elapsed": round(self.elapsed, 3),
            "eta": round(self.eta, 3),
            "speed": round(self.speed, 3),
            "percent": round(self.percent, 2),
            "finished": self.finished,
        }


# ============================================================
# 进度追踪器
# ============================================================

class ProgressTracker:
    """进度追踪器。

    使用方式：
        tracker = ProgressTracker(total=100)
        tracker.start()
        for i in range(100):
            tracker.update(i + 1)
        tracker.finish()
        print(tracker.info)
    """

    def __init__(self,
                 total: int = 0,
                 on_update: Optional[Callable[[ProgressInfo], None]] = None,
                 update_interval: float = 0.0):
        """初始化进度追踪器。

        参数：
            total : 总量（0 表示未知）
            on_update : 更新回调
            update_interval : 最小更新间隔（秒），0 表示每次都更新
        """
        self.total = total
        self.on_update = on_update
        self.update_interval = update_interval
        self.info = ProgressInfo(total=total)
        self._last_update_time: float = 0.0
        self._started = False

    def start(self) -> "ProgressTracker":
        """开始追踪。"""
        self.info = ProgressInfo(total=self.total, started_at=time.time())
        self._started = True
        self._last_update_time = 0.0
        self._notify()
        return self

    def update(self, current: int, total: Optional[int] = None) -> ProgressInfo:
        """更新进度。

        参数：
            current : 当前进度
            total : 新的总量（可选，用于动态调整）

        返回：
            更新后的 ProgressInfo
        """
        if not self._started:
            self.start()

        if total is not None:
            self.info.total = total

        now = time.time()
        self.info.current = current
        self.info.elapsed = now - self.info.started_at

        # 计算速度和 ETA
        if self.info.total > 0 and self.info.current > 0:
            self.info.percent = (self.info.current / self.info.total) * 100
            if self.info.elapsed > 0:
                self.info.speed = self.info.current / self.info.elapsed
                if self.info.speed > 0:
                    remaining = self.info.total - self.info.current
                    self.info.eta = remaining / self.info.speed

        # 检查是否需要通知（节流）
        should_notify = (
            self.update_interval <= 0
            or (now - self._last_update_time) >= self.update_interval
            or self.info.current >= self.info.total
        )
        if should_notify:
            self._last_update_time = now
            self._notify()

        return self.info

    def increment(self, step: int = 1) -> ProgressInfo:
        """增量更新进度。

        参数：
            step : 增量步数

        返回：
            更新后的 ProgressInfo
        """
        return self.update(self.info.current + step)

    def finish(self) -> ProgressInfo:
        """完成追踪。"""
        self.info.finished = True
        if self.info.total > 0:
            self.info.current = self.info.total
            self.info.percent = 100.0
        self.info.eta = 0.0
        now = time.time()
        self.info.elapsed = now - self.info.started_at
        self._notify()
        return self.info

    def _notify(self) -> None:
        """通知回调。"""
        if self.on_update:
            try:
                self.on_update(self.info)
            except Exception:
                pass  # 回调失败不影响主流程

    def reset(self) -> None:
        """重置追踪器。"""
        self.info = ProgressInfo(total=self.total)
        self._started = False
        self._last_update_time = 0.0


# ============================================================
# 进度条
# ============================================================

class ProgressBar:
    """终端进度条。

    使用方式：
        bar = ProgressBar(total=100, width=40)
        bar.start()
        for i in range(100):
            bar.update(i + 1)
        bar.finish()
    """

    def __init__(self,
                 total: int = 0,
                 width: int = 40,
                 label: str = "",
                 output_func: Optional[Callable[[str], None]] = None,
                 update_interval: float = 0.1):
        """初始化进度条。

        参数：
            total : 总量
            width : 进度条宽度（字符数）
            label : 标签
            output_func : 输出函数（默认 sys.stderr.write）
            update_interval : 最小刷新间隔（秒）
        """
        self.total = total
        self.width = max(10, width)
        self.label = label
        self.output_func = output_func or (lambda s: sys.stderr.write(s))
        self.tracker = ProgressTracker(
            total=total,
            on_update=self._render,
            update_interval=update_interval,
        )

    def start(self) -> "ProgressBar":
        """开始进度条。"""
        self.tracker.start()
        return self

    def update(self, current: int, total: Optional[int] = None) -> "ProgressBar":
        """更新进度。"""
        self.tracker.update(current, total)
        return self

    def increment(self, step: int = 1) -> "ProgressBar":
        """增量更新。"""
        self.tracker.increment(step)
        return self

    def finish(self) -> "ProgressBar":
        """完成进度条。"""
        self.tracker.finish()
        self.output_func("\n")
        return self

    def _render(self, info: ProgressInfo) -> None:
        """渲染进度条。"""
        if info.total > 0:
            filled = int(self.width * info.current / info.total)
            bar = "=" * filled + "-" * (self.width - filled)
            percent = info.percent
        else:
            # 未知总量，显示旋转动画
            spinner_chars = "|/-\\"
            idx = info.current % len(spinner_chars)
            bar = spinner_chars[idx] + " " * (self.width - 1)
            percent = 0.0

        label_str = f"{self.label} " if self.label else ""
        elapsed_str = f"{info.elapsed:.1f}s"
        eta_str = f"ETA {info.eta:.1f}s" if info.eta > 0 else "ETA --"
        line = f"\r{label_str}[{bar}] {percent:5.1f}% {info.current}/{info.total} {elapsed_str} {eta_str}"
        self.output_func(line)


# ============================================================
# Agent 进度追踪
# ============================================================

class AgentProgressTracker:
    """基于 EventBus 的 Agent 进度追踪器。

    通过订阅 STEP_END 事件自动追踪 Agent 运行进度。

    使用方式：
        tracker = AgentProgressTracker(agent, total_steps=100)
        tracker.start()
        agent.run(n_steps=100)
        tracker.finish()
    """

    def __init__(self,
                 agent,
                 total_steps: int = 0,
                 on_update: Optional[Callable[[ProgressInfo], None]] = None):
        """初始化。

        参数：
            agent : SPEPAAgent 实例
            total_steps : 总步数（0 表示使用 agent.config.max_steps）
            on_update : 更新回调
        """
        self.agent = agent
        self.total_steps = total_steps or agent.config.max_steps
        self.tracker = ProgressTracker(total=self.total_steps, on_update=on_update)
        self._handler: Optional[Callable] = None

    def start(self) -> "AgentProgressTracker":
        """开始追踪（订阅事件）。"""
        self.tracker.start()

        def on_step_end(event: Event):
            step = event.data.get("step", 0)
            self.tracker.update(step)

        self._handler = on_step_end
        self.agent.event_bus.subscribe(EventType.STEP_END, on_step_end)
        return self

    def finish(self) -> ProgressInfo:
        """完成追踪（取消订阅）。"""
        if self._handler is not None:
            self.agent.event_bus.unsubscribe(EventType.STEP_END, self._handler)
            self._handler = None
        return self.tracker.finish()

    @property
    def info(self) -> ProgressInfo:
        """当前进度信息。"""
        return self.tracker.info


# ============================================================
# 便捷函数与上下文管理器
# ============================================================

@contextmanager
def track_progress(total: int = 0,
                   on_update: Optional[Callable[[ProgressInfo], None]] = None,
                   show_bar: bool = False,
                   label: str = "") -> ProgressTracker:
    """进度追踪上下文管理器。

    参数：
        total : 总量
        on_update : 更新回调
        show_bar : 是否显示进度条
        label : 进度条标签

    使用方式：
        with track_progress(total=100) as tracker:
            for i in range(100):
                tracker.update(i + 1)
    """
    if show_bar:
        bar = ProgressBar(total=total, label=label)
        tracker = ProgressTracker(total=total, on_update=bar._render)
        bar.tracker = tracker
        bar.start()
    else:
        tracker = ProgressTracker(total=total, on_update=on_update)
        tracker.start()
    try:
        yield tracker
    finally:
        tracker.finish()
        if show_bar:
            bar.output_func("\n")


@contextmanager
def track_agent_progress(agent,
                         total_steps: int = 0,
                         on_update: Optional[Callable[[ProgressInfo], None]] = None
                         ) -> AgentProgressTracker:
    """Agent 进度追踪上下文管理器。

    参数：
        agent : SPEPAAgent 实例
        total_steps : 总步数
        on_update : 更新回调

    使用方式：
        with track_agent_progress(agent, total_steps=100) as tracker:
            agent.run(n_steps=100)
    """
    tracker = AgentProgressTracker(agent, total_steps=total_steps, on_update=on_update)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.finish()


def format_progress(info: ProgressInfo, verbose: bool = False) -> str:
    """格式化进度信息为可读文本。

    参数：
        info : ProgressInfo 实例
        verbose : 是否显示详细信息

    返回：
        格式化的字符串
    """
    lines = []
    status = "完成" if info.finished else "进行中"
    lines.append(f"[进度] {status}")
    lines.append(f"  当前: {info.current} / {info.total}")
    lines.append(f"  百分比: {info.percent:.1f}%")
    lines.append(f"  已用时间: {info.elapsed:.2f}s")
    if info.eta > 0:
        lines.append(f"  预计剩余: {info.eta:.2f}s")
    if verbose and info.speed > 0:
        lines.append(f"  速度: {info.speed:.2f} 单位/秒")
    return "\n".join(lines)
