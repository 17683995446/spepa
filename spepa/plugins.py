"""SPEPA 插件系统与扩展机制。

设计原则（第一性原理）：
- 基于事件：插件通过订阅 EventBus 事件来扩展功能，无需修改核心代码
- 生命周期管理：插件有 init/enable/disable/cleanup 完整生命周期
- 零侵入：插件系统不修改 Agent 核心逻辑，仅通过事件总线交互
- 可组合：多个插件可同时工作，互不干扰

插件接口：
    Plugin 基类定义了插件的生命周期方法：
    - init(agent)      : 初始化（获取 agent 引用）
    - on_event(event)  : 处理事件（由订阅的事件触发）
    - cleanup()        : 清理资源

    插件通过 subscribe_events 声明感兴趣的事件类型。

内置插件：
    - LoggingPlugin    : 增强日志记录
    - MetricPlugin     : 自定义指标收集
    - StateMonitorPlugin : 状态监控与告警
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Type
from dataclasses import dataclass, field

from spepa.events import EventBus, EventType, Event
from spepa.utils.logger import get_logger

logger = get_logger("spepa.plugins")


# ============================================================
# 插件基类
# ============================================================

class Plugin(ABC):
    """插件抽象基类。

    子类必须实现：
    - name : 插件名称属性
    - init(agent) : 初始化方法

    可选实现：
    - on_event(event) : 事件处理（默认空实现）
    - cleanup() : 清理资源（默认空实现）
    - subscribe_events : 声明订阅的事件类型集合
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """插件唯一名称。"""
        ...

    @property
    def subscribe_events(self) -> Set[EventType]:
        """该插件订阅的事件类型集合（默认空集）。"""
        return set()

    @abstractmethod
    def init(self, agent) -> None:
        """初始化插件。

        参数：
            agent : SPEPAAgent 实例
        """
        ...

    def on_event(self, event: Event) -> None:
        """处理事件（默认空实现）。

        参数：
            event : 事件对象
        """
        pass

    def cleanup(self) -> None:
        """清理资源（默认空实现）。"""
        pass

    def __repr__(self) -> str:
        return f"<Plugin {self.name}>"


# ============================================================
# 插件管理器
# ============================================================

class PluginManager:
    """插件管理器：管理插件的注册、初始化、事件分发、清理。

    使用方式：
        manager = PluginManager(agent)
        manager.register(LoggingPlugin())
        manager.register(MetricPlugin())
        # 运行 Agent，插件自动接收事件
        agent.run(n_steps=100)
        manager.cleanup()
    """

    def __init__(self, agent):
        """初始化插件管理器。

        参数：
            agent : SPEPAAgent 实例（需有 event_bus 属性）
        """
        self.agent = agent
        self.event_bus: EventBus = agent.event_bus
        self._plugins: Dict[str, Plugin] = {}
        self._enabled: Dict[str, bool] = {}
        self._callbacks: Dict[str, Dict[EventType, Any]] = {}  # 用于取消订阅

    def register(self, plugin: Plugin) -> None:
        """注册并初始化插件。

        参数：
            plugin : Plugin 实例

        异常：
            ValueError : 插件名称已存在
        """
        if plugin.name in self._plugins:
            raise ValueError(f"插件 '{plugin.name}' 已注册")

        # 初始化插件
        plugin.init(self.agent)
        self._plugins[plugin.name] = plugin
        self._enabled[plugin.name] = True

        # 订阅事件
        callbacks = {}
        for event_type in plugin.subscribe_events:
            cb = self._make_callback(plugin)
            self.event_bus.subscribe(event_type, cb)
            callbacks[event_type] = cb
        self._callbacks[plugin.name] = callbacks

        logger.info("Plugin registered: %s (events=%d)",
                    plugin.name, len(plugin.subscribe_events))

    def unregister(self, name: str) -> bool:
        """注销插件。

        参数：
            name : 插件名称

        返回：
            是否成功注销
        """
        if name not in self._plugins:
            return False

        # 取消订阅
        callbacks = self._callbacks.get(name, {})
        for event_type, cb in callbacks.items():
            self.event_bus.unsubscribe(event_type, cb)

        # 清理插件
        plugin = self._plugins[name]
        try:
            plugin.cleanup()
        except Exception as e:
            logger.warning("Plugin %s cleanup error: %s", name, e)

        del self._plugins[name]
        del self._enabled[name]
        if name in self._callbacks:
            del self._callbacks[name]

        logger.info("Plugin unregistered: %s", name)
        return True

    def enable(self, name: str) -> bool:
        """启用插件。"""
        if name not in self._plugins:
            return False
        if self._enabled.get(name, False):
            return True  # 已启用
        # 重新订阅
        plugin = self._plugins[name]
        callbacks = {}
        for event_type in plugin.subscribe_events:
            cb = self._make_callback(plugin)
            self.event_bus.subscribe(event_type, cb)
            callbacks[event_type] = cb
        self._callbacks[name] = callbacks
        self._enabled[name] = True
        logger.info("Plugin enabled: %s", name)
        return True

    def disable(self, name: str) -> bool:
        """禁用插件。"""
        if name not in self._plugins:
            return False
        if not self._enabled.get(name, False):
            return True  # 已禁用
        # 取消订阅
        callbacks = self._callbacks.get(name, {})
        for event_type, cb in callbacks.items():
            self.event_bus.unsubscribe(event_type, cb)
        self._callbacks[name] = {}
        self._enabled[name] = False
        logger.info("Plugin disabled: %s", name)
        return True

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """获取插件实例。"""
        return self._plugins.get(name)

    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """列出所有已注册插件。

        返回：
            {name: {enabled, events: [event_type_names]}}
        """
        result = {}
        for name, plugin in self._plugins.items():
            result[name] = {
                "enabled": self._enabled.get(name, False),
                "events": [et.value for et in plugin.subscribe_events],
            }
        return result

    def cleanup(self) -> None:
        """清理所有插件。"""
        for name in list(self._plugins.keys()):
            self.unregister(name)

    def _make_callback(self, plugin: Plugin):
        """为插件创建事件回调（检查启用状态）。"""
        def callback(event: Event) -> None:
            if self._enabled.get(plugin.name, False):
                try:
                    plugin.on_event(event)
                except Exception as e:
                    logger.warning("Plugin %s event handler error: %s",
                                   plugin.name, e)
        return callback


# ============================================================
# 内置插件
# ============================================================

class LoggingPlugin(Plugin):
    """增强日志插件：记录关键事件到文件或自定义日志器。

    订阅事件：STEP_END, GOAL_ACHIEVED, ANOMALY, REPAIR
    """

    def __init__(self, filepath: Optional[str] = None,
                 log_level: str = "INFO"):
        """初始化日志插件。

        参数：
            filepath : 日志文件路径（None=仅控制台输出）
            log_level : 日志级别
        """
        self._filepath = filepath
        self._log_level = log_level
        self._file_handler = None
        self._plugin_logger = None
        self._event_count = 0

    @property
    def name(self) -> str:
        return "logging"

    @property
    def subscribe_events(self) -> Set[EventType]:
        return {EventType.STEP_END, EventType.GOAL_ACHIEVED,
                EventType.ANOMALY, EventType.REPAIR}

    def init(self, agent) -> None:
        if self._filepath:
            from spepa.utils.logger import add_file_handler, remove_handlers
            import logging
            self._file_handler = add_file_handler(
                self._filepath, self._log_level, "spepa.plugins.logging"
            )
        self._plugin_logger = get_logger("spepa.plugins.logging")
        self._plugin_logger.info("LoggingPlugin initialized")

    def on_event(self, event: Event) -> None:
        self._event_count += 1
        if event.event_type == EventType.STEP_END:
            step = event.data.get("step", "?")
            error = event.data.get("error", 0)
            self._plugin_logger.info(
                "[LoggingPlugin] Step %s: error=%.4f", step, error
            )
        elif event.event_type == EventType.GOAL_ACHIEVED:
            goal_id = event.data.get("goal_id", "?")
            self._plugin_logger.info(
                "[LoggingPlugin] Goal achieved: %s", goal_id
            )
        elif event.event_type == EventType.ANOMALY:
            anomaly = event.data.get("anomaly_type", "unknown")
            self._plugin_logger.warning(
                "[LoggingPlugin] Anomaly: %s", anomaly
            )
        elif event.event_type == EventType.REPAIR:
            level = event.data.get("repair_level", "?")
            self._plugin_logger.info(
                "[LoggingPlugin] Repair: level=%s", level
            )

    def cleanup(self) -> None:
        if self._file_handler:
            import logging
            from spepa.utils.logger import remove_handlers
            remove_handlers(handler_type=logging.FileHandler)
        self._plugin_logger.info(
            "LoggingPlugin cleanup (total_events=%d)", self._event_count
        )


class MetricPlugin(Plugin):
    """自定义指标收集插件：收集运行时指标并计算统计。

    订阅事件：STEP_END, CYCLE_COMPLETE
    """

    def __init__(self):
        self._steps: List[Dict[str, Any]] = []
        self._cycles: List[Dict[str, Any]] = []
        self._agent = None

    @property
    def name(self) -> str:
        return "metric"

    @property
    def subscribe_events(self) -> Set[EventType]:
        return {EventType.STEP_END, EventType.CYCLE_COMPLETE}

    def init(self, agent) -> None:
        self._agent = agent

    def on_event(self, event: Event) -> None:
        if event.event_type == EventType.STEP_END:
            self._steps.append(dict(event.data))
        elif event.event_type == EventType.CYCLE_COMPLETE:
            self._cycles.append(dict(event.data))

    def get_step_metrics(self) -> List[Dict[str, Any]]:
        """获取所有步级指标。"""
        return self._steps.copy()

    def get_cycle_metrics(self) -> List[Dict[str, Any]]:
        """获取所有周期级指标。"""
        return self._cycles.copy()

    def summary(self) -> Dict[str, Any]:
        """获取指标摘要。"""
        import numpy as np
        result = {
            "total_steps": len(self._steps),
            "total_cycles": len(self._cycles),
        }
        if self._steps:
            errors = [s.get("error", 0) for s in self._steps]
            confidences = [s.get("confidence", 0) for s in self._steps]
            result["error_mean"] = float(np.mean(errors))
            result["error_std"] = float(np.std(errors))
            result["confidence_mean"] = float(np.mean(confidences))
        return result

    def cleanup(self) -> None:
        self._steps.clear()
        self._cycles.clear()


class StateMonitorPlugin(Plugin):
    """状态监控插件：监控 Agent 状态变化，检测异常。

    订阅事件：STEP_END
    """

    def __init__(self, alert_threshold: float = 2.0):
        """初始化状态监控插件。

        参数：
            alert_threshold : 状态范数告警阈值
        """
        self._threshold = alert_threshold
        self._agent = None
        self._alerts: List[Dict[str, Any]] = []
        self._last_norm = 0.0

    @property
    def name(self) -> str:
        return "state_monitor"

    @property
    def subscribe_events(self) -> Set[EventType]:
        return {EventType.STEP_END}

    def init(self, agent) -> None:
        self._agent = agent

    def on_event(self, event: Event) -> None:
        import numpy as np
        if self._agent is None:
            return
        norm = float(np.linalg.norm(self._agent.current_state))
        if norm > self._threshold:
            self._alerts.append({
                "step": event.data.get("step", 0),
                "norm": norm,
                "threshold": self._threshold,
                "message": f"State norm {norm:.4f} exceeds threshold {self._threshold}",
            })
        if not np.all(np.isfinite(self._agent.current_state)):
            self._alerts.append({
                "step": event.data.get("step", 0),
                "message": "State contains non-finite values",
            })
        self._last_norm = norm

    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取告警列表。"""
        return self._alerts.copy()

    @property
    def last_norm(self) -> float:
        """最后一次状态范数。"""
        return self._last_norm

    def cleanup(self) -> None:
        self._alerts.clear()


# ============================================================
# 工具函数
# ============================================================

def list_builtin_plugins() -> Dict[str, str]:
    """列出所有内置插件及描述。"""
    return {
        "logging": "LoggingPlugin - 增强日志记录（文件+控制台）",
        "metric": "MetricPlugin - 自定义指标收集与统计",
        "state_monitor": "StateMonitorPlugin - 状态监控与告警",
    }


def create_plugin(name: str, **kwargs) -> Plugin:
    """工厂函数：创建内置插件实例。

    参数：
        name : 插件名称
        **kwargs : 传递给插件构造函数的参数

    返回：
        Plugin 实例

    示例：
        >>> plugin = create_plugin("logging", filepath="spepa.log")
    """
    factories = {
        "logging": LoggingPlugin,
        "metric": MetricPlugin,
        "state_monitor": StateMonitorPlugin,
    }
    if name not in factories:
        raise ValueError(
            f"未知插件: {name}。可用: {', '.join(factories.keys())}"
        )
    return factories[name](**kwargs)
