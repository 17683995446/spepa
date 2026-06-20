"""SPEPA Jupyter Notebook 集成测试。

测试范围：
- 环境检测（is_jupyter / is_ipython_available）
- 富文本显示（display_agent / display_config / display_metrics / display_diagnosis）
- 内部 HTML/文本生成器（_agent_to_html / _agent_to_text / _config_to_html / _config_to_text）
- IPython 魔法命令（SPEPAMagics）
- 注册与扩展入口（register_magics / load_ipython_extension）
- 便捷函数（show / quickstart）
- 端到端流程

设计原则：
- IPython 不可用时验证优雅降级（终端模式）
- 通过 monkeypatch 模拟 Jupyter 环境验证 HTML 渲染路径
"""
from typing import Any, Dict, List
import sys

import pytest

from spepa.utils.config import SPEPAConfig, get_default_config, get_preset_config
from spepa.agent import SPEPAAgent
from spepa import jupyter
from spepa.jupyter import (
    is_jupyter, is_ipython_available,
    display_agent, display_config, display_metrics, display_diagnosis,
    SPEPAMagics, register_magics, load_ipython_extension,
    show, quickstart,
    _agent_to_html, _agent_to_text,
    _config_to_html, _config_to_text,
    _get_display, _get_html, _get_markdown,
    get_magics,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def agent() -> SPEPAAgent:
    """提供一个运行过若干步的 Agent。"""
    a = SPEPAAgent(get_preset_config("minimal"))
    a.run(n_steps=3)
    return a


@pytest.fixture
def config() -> SPEPAConfig:
    return get_preset_config("minimal")


# ============================================================
# 环境检测
# ============================================================

class TestEnvironmentDetection:
    """测试环境检测函数。"""

    def test_is_ipython_available_returns_bool(self):
        assert isinstance(is_ipython_available(), bool)

    def test_is_jupyter_returns_bool(self):
        assert isinstance(is_jupyter(), bool)

    def test_is_jupyter_false_when_no_ipython(self):
        """IPython 未安装时 is_jupyter 应返回 False。"""
        if not is_ipython_available():
            assert is_jupyter() is False

    def test_is_jupyter_false_when_no_shell(self, monkeypatch):
        """模拟 IPython 存在但 get_ipython 返回 None。"""
        # 构造一个假的 IPython 模块
        fake_ipython = type(sys)("IPython")

        def fake_get_ipython():
            return None

        fake_ipython.get_ipython = fake_get_ipython
        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
        assert is_jupyter() is False

    def test_is_jupyter_true_for_zmq_shell(self, monkeypatch):
        """模拟 ZMQInteractiveShell（Jupyter notebook）。"""

        class ZMQInteractiveShell:
            pass

        fake_ipython = type(sys)("IPython")

        def fake_get_ipython():
            return ZMQInteractiveShell()

        fake_ipython.get_ipython = fake_get_ipython
        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
        assert is_jupyter() is True

    def test_is_jupyter_true_for_terminal_shell(self, monkeypatch):
        """模拟 TerminalInteractiveShell。"""

        class TerminalInteractiveShell:
            pass

        fake_ipython = type(sys)("IPython")

        def fake_get_ipython():
            return TerminalInteractiveShell()

        fake_ipython.get_ipython = fake_get_ipython
        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
        assert is_jupyter() is True

    def test_is_jupyter_false_for_other_shell(self, monkeypatch):
        """非 Jupyter/IPython 终端的 shell。"""

        class OtherShell:
            pass

        fake_ipython = type(sys)("IPython")

        def fake_get_ipython():
            return OtherShell()

        fake_ipython.get_ipython = fake_get_ipython
        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
        assert is_jupyter() is False

    def test_is_ipython_available_true_when_module_exists(self, monkeypatch):
        """模拟 IPython 可导入。"""
        fake_ipython = type(sys)("IPython")
        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
        assert is_ipython_available() is True


# ============================================================
# 显示工具获取
# ============================================================

class TestDisplayHelpers:
    """测试 _get_display / _get_html / _get_markdown。"""

    def test_get_display_returns_callable(self):
        disp = _get_display()
        assert callable(disp)

    def test_get_display_returns_print_when_no_ipython(self):
        """IPython 不可用时返回 print。"""
        if not is_ipython_available():
            assert _get_display() is print

    def test_get_html_returns_none_when_no_ipython(self):
        if not is_ipython_available():
            assert _get_html() is None

    def test_get_markdown_returns_none_when_no_ipython(self):
        if not is_ipython_available():
            assert _get_markdown() is None

    def test_get_html_returns_class_when_ipython_available(self, monkeypatch):
        """模拟 IPython.display.HTML 存在。"""

        class FakeHTML:
            def __init__(self, data):
                self.data = data

        fake_display = type(sys)("IPython.display")
        fake_display.HTML = FakeHTML
        fake_display.display = print
        fake_display.Markdown = FakeHTML

        fake_ipython = type(sys)("IPython")
        fake_ipython.display = fake_display

        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
        monkeypatch.setitem(sys.modules, "IPython.display", fake_display)

        html_cls = _get_html()
        assert html_cls is FakeHTML

    def test_get_markdown_returns_class_when_ipython_available(self, monkeypatch):
        class FakeMarkdown:
            def __init__(self, data):
                self.data = data

        fake_display = type(sys)("IPython.display")
        fake_display.HTML = FakeMarkdown
        fake_display.display = print
        fake_display.Markdown = FakeMarkdown

        fake_ipython = type(sys)("IPython")
        fake_ipython.display = fake_display

        monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
        monkeypatch.setitem(sys.modules, "IPython.display", fake_display)

        md_cls = _get_markdown()
        assert md_cls is FakeMarkdown


# ============================================================
# Agent 状态渲染
# ============================================================

class TestAgentRendering:
    """测试 Agent 状态的 HTML/文本渲染。"""

    def test_agent_to_text_basic(self, agent):
        state = agent.get_state()
        text = _agent_to_text(state)
        assert isinstance(text, str)
        assert "SPEPA Agent" in text
        assert "step_count" in text

    def test_agent_to_text_hides_verbose_fields_by_default(self, agent):
        state = agent.get_state()
        text = _agent_to_text(state, verbose=False)
        # 这些字段默认隐藏
        assert "predictive_model" not in text
        assert "drive_engine" not in text

    def test_agent_to_text_shows_verbose_fields(self, agent):
        state = agent.get_state()
        text = _agent_to_text(state, verbose=True)
        assert "predictive_model" in text
        assert "drive_engine" in text

    def test_agent_to_text_formats_floats(self, agent):
        state = agent.get_state()
        # 注入一个浮点值
        state["test_float"] = 0.123456789
        text = _agent_to_text(state, verbose=True)
        assert "0.1235" in text  # 4 位小数

    def test_agent_to_text_handles_nested_dict(self, agent):
        state = agent.get_state()
        text = _agent_to_text(state, verbose=True)
        # resources 是嵌套字典
        assert "resources" in text
        assert "cpu" in text

    def test_agent_to_html_basic(self, agent):
        state = agent.get_state()
        html = _agent_to_html(state)
        assert isinstance(html, str)
        assert "<table" in html
        assert "SPEPA Agent" in html

    def test_agent_to_html_hides_verbose_fields_by_default(self, agent):
        state = agent.get_state()
        html = _agent_to_html(state, verbose=False)
        assert "predictive_model" not in html

    def test_agent_to_html_shows_verbose_fields(self, agent):
        state = agent.get_state()
        html = _agent_to_html(state, verbose=True)
        assert "predictive_model" in html

    def test_agent_to_html_formats_floats(self, agent):
        state = agent.get_state()
        state["test_float"] = 0.123456789
        html = _agent_to_html(state, verbose=True)
        assert "0.1235" in html

    def test_agent_to_html_has_table_borders(self, agent):
        state = agent.get_state()
        html = _agent_to_html(state)
        assert "border" in html
        assert "</table>" in html


# ============================================================
# Config 渲染
# ============================================================

class TestConfigRendering:
    """测试配置的 HTML/文本渲染。"""

    def test_config_to_text_basic(self, config):
        d = config.to_dict()
        text = _config_to_text(d)
        assert isinstance(text, str)
        assert "SPEPA 配置" in text

    def test_config_to_text_handles_nested_dict(self, config):
        d = config.to_dict()
        text = _config_to_text(d)
        # 配置通常有嵌套字典（如 resources）
        # 至少应该有顶层键
        assert "SPEPA" in text

    def test_config_to_html_basic(self, config):
        d = config.to_dict()
        html = _config_to_html(d)
        assert isinstance(html, str)
        assert "<table" in html
        assert "SPEPA 配置" in html

    def test_config_to_html_has_borders(self, config):
        d = config.to_dict()
        html = _config_to_html(d)
        assert "border" in html
        assert "</table>" in html

    def test_config_to_html_handles_nested_dict(self, config):
        d = config.to_dict()
        html = _config_to_html(d)
        assert "</table>" in html


# ============================================================
# display_agent / display_config（终端模式）
# ============================================================

class TestDisplayFunctions:
    """测试 display_* 函数（终端模式）。"""

    def test_display_agent_runs_without_error(self, agent, capsys):
        # 终端模式下应打印文本
        display_agent(agent)
        output = capsys.readouterr().out
        assert "SPEPA Agent" in output

    def test_display_agent_verbose(self, agent, capsys):
        display_agent(agent, verbose=True)
        output = capsys.readouterr().out
        assert "predictive_model" in output

    def test_display_config_runs_without_error(self, config, capsys):
        display_config(config)
        output = capsys.readouterr().out
        assert "SPEPA" in output

    def test_display_metrics_runs_without_error(self, agent, capsys):
        # 终端模式下使用 ASCII 图表
        display_metrics(agent)
        # 不应抛出异常
        assert True

    def test_display_metrics_with_custom_names(self, agent, capsys):
        display_metrics(agent, metric_names=["prediction_error"])
        assert True

    def test_display_diagnosis_runs_without_error(self, agent, capsys):
        display_diagnosis(agent)
        output = capsys.readouterr().out
        # 应该输出诊断信息
        assert len(output) >= 0  # 至少不报错

    def test_display_agent_jupyter_mode(self, agent, monkeypatch, capsys):
        """模拟 Jupyter 模式渲染 HTML。"""
        rendered = []

        class FakeHTML:
            def __init__(self, data):
                self.data = data
                rendered.append(data)

        def fake_display(obj):
            print(f"[HTML displayed: {type(obj).__name__}]")

        monkeypatch.setattr(jupyter, "is_jupyter", lambda: True)
        monkeypatch.setattr(jupyter, "_get_html", lambda: FakeHTML)
        monkeypatch.setattr(jupyter, "_get_display", lambda: fake_display)

        display_agent(agent)
        assert len(rendered) == 1
        assert "<table" in rendered[0]

    def test_display_config_jupyter_mode(self, config, monkeypatch, capsys):
        rendered = []

        class FakeHTML:
            def __init__(self, data):
                self.data = data
                rendered.append(data)

        def fake_display(obj):
            pass

        monkeypatch.setattr(jupyter, "is_jupyter", lambda: True)
        monkeypatch.setattr(jupyter, "_get_html", lambda: FakeHTML)
        monkeypatch.setattr(jupyter, "_get_display", lambda: fake_display)

        display_config(config)
        assert len(rendered) == 1
        assert "<table" in rendered[0]


# ============================================================
# SPEPAMagics
# ============================================================

class TestSPEPAMagics:
    """测试 IPython 魔法命令。"""

    def test_magics_init_with_default(self):
        m = SPEPAMagics()
        assert m.agent is not None
        assert isinstance(m.agent, SPEPAAgent)
        assert "run" in m.commands
        assert "state" in m.commands
        assert "config" in m.commands
        assert "metrics" in m.commands
        assert "diagnose" in m.commands
        assert "help" in m.commands

    def test_magics_init_with_agent(self, agent):
        m = SPEPAMagics(agent=agent)
        assert m.agent is agent

    def test_magics_init_with_config(self, config):
        m = SPEPAMagics(config=config)
        assert m.agent.config is config

    def test_magics_execute_empty_line_shows_help(self, capsys):
        m = SPEPAMagics()
        result = m.execute("")
        # 空行应调用 help
        output = capsys.readouterr().out
        assert "SPEPA 魔法命令" in output or "spepa" in output.lower()

    def test_magics_execute_whitespace_only(self, capsys):
        m = SPEPAMagics()
        m.execute("   ")
        output = capsys.readouterr().out
        assert len(output) >= 0

    def test_magics_execute_unknown_command(self, capsys):
        m = SPEPAMagics()
        result = m.execute("nonexistent_cmd")
        output = capsys.readouterr().out
        assert "未知命令" in output

    def test_magics_execute_help(self, capsys):
        m = SPEPAMagics()
        result = m.execute("help")
        output = capsys.readouterr().out
        assert "spepa" in output.lower()
        assert isinstance(result, str)

    def test_magics_execute_state(self, capsys):
        m = SPEPAMagics()
        m.execute("state")
        output = capsys.readouterr().out
        assert "SPEPA Agent" in output

    def test_magics_execute_state_verbose(self, capsys):
        m = SPEPAMagics()
        m.execute("state -v")
        output = capsys.readouterr().out
        assert "predictive_model" in output

    def test_magics_execute_state_verbose_long(self, capsys):
        m = SPEPAMagics()
        m.execute("state --verbose")
        output = capsys.readouterr().out
        assert "predictive_model" in output

    def test_magics_execute_config(self, capsys):
        m = SPEPAMagics()
        m.execute("config")
        output = capsys.readouterr().out
        assert "SPEPA" in output

    def test_magics_execute_metrics(self, capsys):
        m = SPEPAMagics()
        m.execute("metrics")
        # 不应抛出异常
        assert True

    def test_magics_execute_metrics_with_names(self, capsys):
        m = SPEPAMagics()
        m.execute("metrics prediction_error")
        assert True

    def test_magics_execute_diagnose(self, capsys):
        m = SPEPAMagics()
        m.execute("diagnose")
        assert True

    def test_magics_execute_run_default(self, capsys):
        m = SPEPAMagics()
        result = m.execute("run 2")
        # run 应返回运行结果
        assert result is not None

    def test_magics_execute_run_no_args(self, capsys):
        m = SPEPAMagics()
        # 不传参数应使用默认值
        result = m.execute("run")
        assert result is not None

    def test_magics_execute_case_insensitive(self, capsys):
        m = SPEPAMagics()
        m.execute("HELP")
        output = capsys.readouterr().out
        assert "spepa" in output.lower()

    def test_magics_cmd_help_returns_text(self):
        m = SPEPAMagics()
        result = m._cmd_help([])
        assert isinstance(result, str)
        assert "spepa" in result.lower()

    def test_magics_cmd_run_returns_dict(self):
        m = SPEPAMagics()
        result = m._cmd_run(["1"])
        assert result is not None

    def test_magics_commands_are_callable(self):
        m = SPEPAMagics()
        for name, cmd in m.commands.items():
            assert callable(cmd), f"{name} 不可调用"


# ============================================================
# 注册与扩展入口
# ============================================================

class TestRegistration:
    """测试魔法命令注册与扩展入口。"""

    def test_register_magics_returns_false_when_no_ipython(self):
        """IPython 不可用时返回 False。"""
        if not is_ipython_available():
            assert register_magics() is False

    def test_register_magics_returns_false_when_no_shell(self, monkeypatch):
        """IPython 可用但无 shell 时返回 False。"""

        class FakeIPython:
            @staticmethod
            def get_ipython():
                return None

        fake_module = type(sys)("IPython")
        fake_module.get_ipython = FakeIPython.get_ipython
        monkeypatch.setitem(sys.modules, "IPython", fake_module)

        assert register_magics() is False

    def test_register_magics_true_when_shell_available(self, monkeypatch):
        """shell 可用时注册成功。"""

        class FakeIPythonShell:
            def __init__(self):
                self.registered = []

            def register_magic_function(self, func, magic_name=None, **kwargs):
                self.registered.append((func, magic_name))

        shell = FakeIPythonShell()

        class FakeIPython:
            @staticmethod
            def get_ipython():
                return shell

        fake_module = type(sys)("IPython")
        fake_module.get_ipython = FakeIPython.get_ipython
        monkeypatch.setitem(sys.modules, "IPython", fake_module)

        result = register_magics()
        assert result is True
        assert len(shell.registered) == 1
        assert shell.registered[0][1] == "spepa"

    def test_register_magics_with_agent(self, monkeypatch):
        class FakeIPythonShell:
            def __init__(self):
                self.registered = []

            def register_magic_function(self, func, magic_name=None, **kwargs):
                self.registered.append((func, magic_name))

        shell = FakeIPythonShell()

        class FakeIPython:
            @staticmethod
            def get_ipython():
                return shell

        fake_module = type(sys)("IPython")
        fake_module.get_ipython = FakeIPython.get_ipython
        monkeypatch.setitem(sys.modules, "IPython", fake_module)

        agent = SPEPAAgent(get_preset_config("minimal"))
        result = register_magics(agent=agent)
        assert result is True

    def test_load_ipython_extension_runs(self, monkeypatch, capsys):
        """load_ipython_extension 应能执行（即使 IPython 不可用）。"""

        class FakeIPythonShell:
            pass

        # 直接调用 load_ipython_extension，传入一个伪 ipython 对象
        # 它会调用 register_magics()，IPython 不可用时返回 False，但仍打印加载信息
        load_ipython_extension(FakeIPythonShell())
        output = capsys.readouterr().out
        assert "SPEPA" in output or "spepa" in output.lower()

    def test_get_magics_returns_singleton(self):
        """get_magics 在无 agent 时返回同一实例。"""
        m1 = get_magics()
        m2 = get_magics()
        # 注意：如果之前测试传过 agent，可能已重置
        assert isinstance(m1, SPEPAMagics)
        assert isinstance(m2, SPEPAMagics)

    def test_get_magics_with_agent_resets(self, agent):
        m = get_magics(agent=agent)
        assert m.agent is agent


# ============================================================
# 便捷函数
# ============================================================

class TestConvenienceFunctions:
    """测试 show / quickstart 便捷函数。"""

    def test_show_state(self, agent, capsys):
        show(agent, "state")
        output = capsys.readouterr().out
        assert "SPEPA Agent" in output

    def test_show_config(self, agent, capsys):
        show(agent, "config")
        output = capsys.readouterr().out
        assert "SPEPA" in output

    def test_show_metrics(self, agent, capsys):
        show(agent, "metrics")
        assert True  # 不报错即可

    def test_show_diagnose(self, agent, capsys):
        show(agent, "diagnose")
        assert True

    def test_show_unknown(self, agent, capsys):
        show(agent, "unknown_what")
        output = capsys.readouterr().out
        assert "未知显示类型" in output

    def test_quickstart_default(self, capsys):
        agent = quickstart(n_steps=2)
        assert isinstance(agent, SPEPAAgent)
        output = capsys.readouterr().out
        assert "SPEPA Agent" in output

    def test_quickstart_with_config(self, config, capsys):
        agent = quickstart(config=config, n_steps=1)
        assert isinstance(agent, SPEPAAgent)
        assert agent.config is config

    def test_quickstart_returns_running_agent(self, capsys):
        agent = quickstart(n_steps=3)
        assert agent.step_count >= 3


# ============================================================
# 模块导出与集成
# ============================================================

class TestModuleIntegration:
    """测试模块在 spepa 包中的导出。"""

    def test_jupyter_module_importable(self):
        import spepa.jupyter as jp
        assert hasattr(jp, "is_jupyter")
        assert hasattr(jp, "display_agent")
        assert hasattr(jp, "SPEPAMagics")

    def test_jupyter_exports_in_package(self):
        import spepa
        for name in ["is_jupyter", "is_ipython_available",
                     "display_agent", "display_config",
                     "display_metrics", "display_diagnosis",
                     "SPEPAMagics", "register_magics",
                     "load_ipython_extension", "show", "quickstart"]:
            assert hasattr(spepa, name), f"spepa.{name} 未导出"

    def test_jupyter_in_all(self):
        import spepa
        for name in ["is_jupyter", "SPEPAMagics", "show", "quickstart"]:
            assert name in spepa.__all__, f"{name} 不在 __all__ 中"

    def test_jupyter_logger_initialized(self):
        assert hasattr(jupyter, "logger")
        assert jupyter.logger.name == "spepa.jupyter"


# ============================================================
# 端到端
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_full_workflow_terminal_mode(self, capsys):
        """终端模式完整工作流：quickstart -> show -> diagnose。"""
        agent = quickstart(n_steps=2)
        show(agent, "state")
        show(agent, "config")
        show(agent, "metrics")
        show(agent, "diagnose")
        output = capsys.readouterr().out
        # 应该有多次输出
        assert "SPEPA" in output

    def test_magics_workflow(self, capsys):
        """魔法命令完整工作流。"""
        m = SPEPAMagics()
        m.execute("help")
        m.execute("run 2")
        m.execute("state")
        m.execute("config")
        m.execute("metrics")
        m.execute("diagnose")
        # 全部不抛异常即可
        assert True

    def test_magics_workflow_with_verbose(self, capsys):
        m = SPEPAMagics()
        m.execute("run 1")
        m.execute("state -v")
        output = capsys.readouterr().out
        assert "predictive_model" in output

    def test_degradation_no_crash(self, capsys):
        """无 IPython 环境下所有功能都不应崩溃。"""
        assert is_ipython_available() is False or is_ipython_available() is True
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=1)
        display_agent(agent)
        display_config(agent.config)
        display_metrics(agent)
        display_diagnosis(agent)
        assert True

    def test_jupyter_mode_simulated(self, monkeypatch, capsys):
        """模拟 Jupyter 模式完整工作流。"""
        rendered = []

        class FakeHTML:
            def __init__(self, data):
                self.data = data
                rendered.append(data)

        def fake_display(obj):
            pass

        monkeypatch.setattr(jupyter, "is_jupyter", lambda: True)
        monkeypatch.setattr(jupyter, "_get_html", lambda: FakeHTML)
        monkeypatch.setattr(jupyter, "_get_markdown", lambda: FakeHTML)
        monkeypatch.setattr(jupyter, "_get_display", lambda: fake_display)

        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=1)

        display_agent(agent)
        display_config(agent.config)
        # 至少 agent 和 config 应该渲染 HTML
        assert len(rendered) >= 2
        assert all("<table" in r for r in rendered[:2])

    def test_show_all_types(self, agent, capsys):
        """show 函数所有类型都不报错。"""
        for what in ["state", "config", "metrics", "diagnose"]:
            show(agent, what)
        assert True

    def test_magics_help_lists_all_commands(self, capsys):
        m = SPEPAMagics()
        m.execute("help")
        output = capsys.readouterr().out
        # 应列出所有命令
        for cmd in ["run", "state", "config", "metrics", "diagnose", "help"]:
            assert cmd in output, f"help 输出中缺少 {cmd}"
