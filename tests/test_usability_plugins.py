"""阶段六测试：插件系统与扩展机制。

测试范围：
1. Plugin 抽象基类
2. PluginManager 注册/注销/启用/禁用
3. LoggingPlugin 内置插件
4. MetricPlugin 内置插件
5. StateMonitorPlugin 内置插件
6. create_plugin 工厂函数
7. 端到端：Agent + 插件集成运行
"""
import os
import tempfile
import numpy as np
import pytest

from spepa.plugins import (
    Plugin, PluginManager,
    LoggingPlugin, MetricPlugin, StateMonitorPlugin,
    list_builtin_plugins, create_plugin,
)
from spepa.events import EventBus, EventType, Event
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_preset_config


# ============================================================
# Plugin 基类测试
# ============================================================

class TestPluginBase:
    """Plugin 抽象基类测试。"""

    def test_cannot_instantiate_abstract(self):
        """不能直接实例化抽象基类。"""
        with pytest.raises(TypeError):
            Plugin()

    def test_subclass_must_implement_name_and_init(self):
        """子类必须实现 name 和 init。"""

        class IncompletePlugin(Plugin):
            pass

        with pytest.raises(TypeError):
            IncompletePlugin()

    def test_complete_subclass(self):
        """完整实现的子类可以实例化。"""

        class TestPlugin(Plugin):
            @property
            def name(self):
                return "test"

            def init(self, agent):
                self._agent = agent

        plugin = TestPlugin()
        assert plugin.name == "test"
        assert plugin.subscribe_events == set()

    def test_default_methods(self):
        """默认方法不抛异常。"""

        class TestPlugin(Plugin):
            @property
            def name(self):
                return "test"

            def init(self, agent):
                pass

        plugin = TestPlugin()
        # on_event 和 cleanup 默认空实现
        plugin.on_event(Event(EventType.STEP_END))
        plugin.cleanup()

    def test_repr(self):
        """__repr__ 返回正确格式。"""

        class TestPlugin(Plugin):
            @property
            def name(self):
                return "my_plugin"

            def init(self, agent):
                pass

        plugin = TestPlugin()
        assert "my_plugin" in repr(plugin)


# ============================================================
# PluginManager 测试
# ============================================================

class TestPluginManager:
    """PluginManager 测试。"""

    @pytest.fixture
    def agent(self):
        return SPEPAAgent(get_preset_config("minimal"))

    def test_register_plugin(self, agent):
        """注册插件。"""

        class TestPlugin(Plugin):
            @property
            def name(self):
                return "test_register"

            def init(self, agent):
                self._agent = agent

        manager = PluginManager(agent)
        plugin = TestPlugin()
        manager.register(plugin)
        assert manager.get_plugin("test_register") is plugin
        plugins = manager.list_plugins()
        assert "test_register" in plugins

    def test_register_duplicate_raises(self, agent):
        """重复注册抛异常。"""

        class TestPlugin(Plugin):
            @property
            def name(self):
                return "dup"

            def init(self, agent):
                pass

        manager = PluginManager(agent)
        manager.register(TestPlugin())
        with pytest.raises(ValueError, match="已注册"):
            manager.register(TestPlugin())

    def test_unregister_plugin(self, agent):
        """注销插件。"""

        class TestPlugin(Plugin):
            @property
            def name(self):
                return "test_unreg"

            def init(self, agent):
                pass

        manager = PluginManager(agent)
        manager.register(TestPlugin())
        assert manager.unregister("test_unreg") is True
        assert manager.get_plugin("test_unreg") is None
        # 再次注销返回 False
        assert manager.unregister("test_unreg") is False

    def test_enable_disable(self, agent):
        """启用/禁用插件。"""

        class TestPlugin(Plugin):
            @property
            def name(self):
                return "test_toggle"

            @property
            def subscribe_events(self):
                return {EventType.STEP_END}

            def init(self, agent):
                self.events = []

            def on_event(self, event):
                self.events.append(event)

        manager = PluginManager(agent)
        plugin = TestPlugin()
        manager.register(plugin)

        # 初始已启用
        assert manager.list_plugins()["test_toggle"]["enabled"] is True

        # 发布事件，插件应收到
        agent.event_bus.publish(Event(EventType.STEP_END, {"step": 1}))
        assert len(plugin.events) == 1

        # 禁用后不应收到
        manager.disable("test_toggle")
        assert manager.list_plugins()["test_toggle"]["enabled"] is False
        agent.event_bus.publish(Event(EventType.STEP_END, {"step": 2}))
        assert len(plugin.events) == 1  # 仍然只有1个

        # 重新启用后应收到
        manager.enable("test_toggle")
        agent.event_bus.publish(Event(EventType.STEP_END, {"step": 3}))
        assert len(plugin.events) == 2

    def test_cleanup_all(self, agent):
        """清理所有插件。"""

        class TestPlugin(Plugin):
            @property
            def name(self):
                return "test_cleanup"

            def init(self, agent):
                pass

            def cleanup(self):
                self.cleaned = True

        manager = PluginManager(agent)
        plugin = TestPlugin()
        manager.register(plugin)
        manager.cleanup()
        assert not hasattr(plugin, 'cleaned') or plugin.cleaned is True
        assert len(manager.list_plugins()) == 0

    def test_plugin_event_error_isolated(self, agent):
        """插件事件处理异常不影响其他插件。"""

        class BadPlugin(Plugin):
            @property
            def name(self):
                return "bad"

            @property
            def subscribe_events(self):
                return {EventType.STEP_END}

            def init(self, agent):
                pass

            def on_event(self, event):
                raise RuntimeError("boom")

        class GoodPlugin(Plugin):
            @property
            def name(self):
                return "good"

            @property
            def subscribe_events(self):
                return {EventType.STEP_END}

            def init(self, agent):
                self.events = []

            def on_event(self, event):
                self.events.append(event)

        manager = PluginManager(agent)
        good = GoodPlugin()
        manager.register(BadPlugin())
        manager.register(good)

        # 发布事件，BadPlugin 抛异常但 GoodPlugin 应正常收到
        agent.event_bus.publish(Event(EventType.STEP_END))
        assert len(good.events) == 1


# ============================================================
# LoggingPlugin 测试
# ============================================================

class TestLoggingPlugin:
    """LoggingPlugin 测试。"""

    def test_plugin_name(self):
        assert LoggingPlugin().name == "logging"

    def test_subscribe_events(self):
        plugin = LoggingPlugin()
        assert EventType.STEP_END in plugin.subscribe_events
        assert EventType.GOAL_ACHIEVED in plugin.subscribe_events
        assert EventType.ANOMALY in plugin.subscribe_events
        assert EventType.REPAIR in plugin.subscribe_events

    def test_init_and_cleanup(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        plugin = LoggingPlugin()
        plugin.init(agent)
        assert plugin._plugin_logger is not None
        plugin.cleanup()  # 不应抛异常

    def test_file_logging(self):
        """文件日志记录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "plugin.log")
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = PluginManager(agent)
            manager.register(LoggingPlugin(filepath=filepath))

            # 运行几步产生事件
            agent.run(n_steps=3)

            manager.cleanup()
            assert os.path.exists(filepath)

    def test_with_agent_run(self):
        """与 Agent 一起运行。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)
        plugin = LoggingPlugin()
        manager.register(plugin)

        agent.run(n_steps=3)
        assert plugin._event_count > 0
        manager.cleanup()


# ============================================================
# MetricPlugin 测试
# ============================================================

class TestMetricPlugin:
    """MetricPlugin 测试。"""

    def test_plugin_name(self):
        assert MetricPlugin().name == "metric"

    def test_subscribe_events(self):
        plugin = MetricPlugin()
        assert EventType.STEP_END in plugin.subscribe_events
        assert EventType.CYCLE_COMPLETE in plugin.subscribe_events

    def test_collects_step_metrics(self):
        """收集步级指标。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)
        plugin = MetricPlugin()
        manager.register(plugin)

        agent.run(n_steps=3)
        steps = plugin.get_step_metrics()
        assert len(steps) > 0

    def test_summary(self):
        """指标摘要。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)
        plugin = MetricPlugin()
        manager.register(plugin)

        agent.run(n_steps=3)
        summary = plugin.summary()
        assert summary["total_steps"] > 0
        assert "error_mean" in summary
        manager.cleanup()

    def test_cleanup_clears_data(self):
        """清理后数据清空。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)
        plugin = MetricPlugin()
        manager.register(plugin)

        agent.run(n_steps=2)
        assert len(plugin.get_step_metrics()) > 0
        manager.cleanup()
        assert len(plugin.get_step_metrics()) == 0


# ============================================================
# StateMonitorPlugin 测试
# ============================================================

class TestStateMonitorPlugin:
    """StateMonitorPlugin 测试。"""

    def test_plugin_name(self):
        assert StateMonitorPlugin().name == "state_monitor"

    def test_subscribe_events(self):
        plugin = StateMonitorPlugin()
        assert EventType.STEP_END in plugin.subscribe_events

    def test_monitors_state(self):
        """监控状态变化。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)
        plugin = StateMonitorPlugin(alert_threshold=10.0)  # 高阈值，不应告警
        manager.register(plugin)

        agent.run(n_steps=3)
        assert plugin.last_norm > 0
        assert len(plugin.get_alerts()) == 0  # 阈值高，无告警
        manager.cleanup()

    def test_alerts_on_high_norm(self):
        """高范数触发告警。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)
        plugin = StateMonitorPlugin(alert_threshold=0.01)  # 极低阈值，必告警
        manager.register(plugin)

        agent.run(n_steps=2)
        alerts = plugin.get_alerts()
        assert len(alerts) > 0
        manager.cleanup()

    def test_alerts_on_non_finite(self):
        """非有限值触发告警。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)
        plugin = StateMonitorPlugin()
        manager.register(plugin)

        # 设置非有限状态
        agent.current_state = np.full(agent.config.state_dim, np.nan)

        # 发布事件触发检查
        agent.event_bus.publish(Event(EventType.STEP_END, {"step": 1}))

        alerts = plugin.get_alerts()
        assert any("non-finite" in a["message"] for a in alerts)
        manager.cleanup()


# ============================================================
# 工厂函数测试
# ============================================================

class TestCreatePlugin:
    """create_plugin 工厂函数测试。"""

    def test_create_logging(self):
        plugin = create_plugin("logging")
        assert isinstance(plugin, LoggingPlugin)
        assert plugin.name == "logging"

    def test_create_metric(self):
        plugin = create_plugin("metric")
        assert isinstance(plugin, MetricPlugin)

    def test_create_state_monitor(self):
        plugin = create_plugin("state_monitor")
        assert isinstance(plugin, StateMonitorPlugin)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="未知插件"):
            create_plugin("unknown")

    def test_create_with_kwargs(self):
        plugin = create_plugin("state_monitor", alert_threshold=5.0)
        assert plugin._threshold == 5.0

    def test_list_builtin_plugins(self):
        plugins = list_builtin_plugins()
        assert "logging" in plugins
        assert "metric" in plugins
        assert "state_monitor" in plugins


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_multiple_plugins_together(self):
        """多个插件同时工作。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)

        # 注册3个插件
        manager.register(LoggingPlugin())
        manager.register(MetricPlugin())
        manager.register(StateMonitorPlugin())

        # 运行
        agent.run(n_steps=5)

        # 验证各插件都正常工作
        logging_plugin = manager.get_plugin("logging")
        metric_plugin = manager.get_plugin("metric")
        monitor_plugin = manager.get_plugin("state_monitor")

        assert logging_plugin._event_count > 0
        assert len(metric_plugin.get_step_metrics()) > 0
        assert monitor_plugin.last_norm > 0

        manager.cleanup()

    def test_plugin_lifecycle(self):
        """插件完整生命周期。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)

        # 1. 注册
        manager.register(MetricPlugin())
        assert len(manager.list_plugins()) == 1

        # 2. 运行
        agent.run(n_steps=2)
        plugin = manager.get_plugin("metric")
        assert len(plugin.get_step_metrics()) > 0

        # 3. 禁用
        manager.disable("metric")
        agent.run(n_steps=2)
        # 禁用后不应收集新数据
        count_after_disable = len(plugin.get_step_metrics())
        agent.run(n_steps=2)
        assert len(plugin.get_step_metrics()) == count_after_disable

        # 4. 重新启用
        manager.enable("metric")
        agent.run(n_steps=2)
        assert len(plugin.get_step_metrics()) > count_after_disable

        # 5. 注销
        manager.unregister("metric")
        assert len(manager.list_plugins()) == 0

        manager.cleanup()

    def test_custom_plugin(self):
        """自定义插件。"""

        class StepCounterPlugin(Plugin):
            @property
            def name(self):
                return "counter"

            @property
            def subscribe_events(self):
                return {EventType.STEP_END}

            def init(self, agent):
                self.count = 0

            def on_event(self, event):
                self.count += 1

        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)
        plugin = StepCounterPlugin()
        manager.register(plugin)

        agent.run(n_steps=5)
        assert plugin.count > 0
        manager.cleanup()

    def test_plugin_with_all_builtin(self):
        """使用所有内置插件运行。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        manager = PluginManager(agent)

        # 用工厂函数创建所有内置插件
        for name in list_builtin_plugins():
            plugin = create_plugin(name)
            manager.register(plugin)

        # 运行
        agent.run(n_steps=3)

        # 所有插件都应正常工作
        plugins = manager.list_plugins()
        assert len(plugins) == 3
        for name, info in plugins.items():
            assert info["enabled"] is True

        manager.cleanup()
        assert len(manager.list_plugins()) == 0
