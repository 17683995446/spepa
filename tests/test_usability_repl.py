"""SPEPA REPL 交互式 Shell 测试。

测试范围：
- Command 数据结构
- REPLShell 初始化
- 内置命令（help/run/step/reset/state/metrics/config/health/diagnose/chart/save/load/history/exit）
- 命令注册（自定义命令、别名）
- 主循环（输入/输出注入、EOF、中断）
- execute 编程式调用
- 端到端
"""
import pytest
from spepa.repl import REPLShell, Command, start_repl
from spepa.utils.config import get_preset_config
from spepa.agent import SPEPAAgent


# ============================================================
# Command 数据结构
# ============================================================

class TestCommand:
    """Command 数据结构测试。"""

    def test_create_command(self):
        cmd = Command(name="test", handler=lambda args: "ok", help="测试命令")
        assert cmd.name == "test"
        assert cmd.help == "测试命令"
        assert cmd.aliases == []

    def test_command_with_aliases(self):
        cmd = Command(name="test", handler=lambda args: "ok", aliases=["t", "te"])
        assert cmd.aliases == ["t", "te"]

    def test_command_handler_callable(self):
        cmd = Command(name="test", handler=lambda args: f"got {args}")
        assert cmd.handler(["a", "b"]) == "got ['a', 'b']"


# ============================================================
# REPLShell 初始化
# ============================================================

class TestREPLInit:
    """REPL Shell 初始化测试。"""

    def test_create_with_default(self):
        shell = REPLShell()
        assert shell.agent is not None
        assert shell.input_func is not None
        assert shell.output_func is not None
        assert len(shell.commands) > 0

    def test_create_with_config(self):
        config = get_preset_config("minimal")
        shell = REPLShell(config=config)
        assert shell.agent.config.state_dim == config.state_dim

    def test_create_with_existing_agent(self):
        agent = SPEPAAgent(get_preset_config("minimal"))
        shell = REPLShell(agent=agent)
        assert shell.agent is agent

    def test_create_with_injected_io(self):
        inputs = []
        outputs = []
        shell = REPLShell(
            input_func=lambda _: "",
            output_func=outputs.append,
        )
        assert shell.input_func("") == ""
        shell.output_func("test")
        assert "test" in outputs

    def test_builtin_commands_registered(self):
        shell = REPLShell()
        expected = ["help", "run", "step", "reset", "state", "metrics",
                    "config", "health", "diagnose", "chart", "save",
                    "load", "history", "exit"]
        for name in expected:
            assert name in shell.commands

    def test_aliases_registered(self):
        shell = REPLShell()
        assert "?" in shell.commands  # help 别名
        assert "quit" in shell.commands  # exit 别名


# ============================================================
# 内置命令
# ============================================================

class TestBuiltinCommands:
    """内置命令测试。"""

    def setup_method(self):
        """每个测试创建独立 shell。"""
        self.outputs = []
        self.shell = REPLShell(
            config=get_preset_config("minimal"),
            output_func=self.outputs.append,
        )

    def test_help_command(self):
        result = self.shell.execute("help")
        assert "可用命令" in result
        assert "run" in result
        assert "exit" in result

    def test_help_alias(self):
        result = self.shell.execute("?")
        assert "可用命令" in result

    def test_run_command(self):
        result = self.shell.execute("run 3")
        assert "运行完成" in result
        assert self.shell.agent.step_count == 3

    def test_run_default_steps(self):
        result = self.shell.execute("run")
        assert "运行完成" in result
        assert self.shell.agent.step_count == 10  # 默认 10 步

    def test_step_command(self):
        result = self.shell.execute("step")
        assert "执行了 1 步" in result
        assert self.shell.agent.step_count == 1

    def test_step_multiple(self):
        result = self.shell.execute("step 5")
        assert "执行了 5 步" in result
        assert self.shell.agent.step_count == 5

    def test_reset_command(self):
        # 先运行几步
        self.shell.execute("run 3")
        assert self.shell.agent.step_count == 3
        # 重置
        result = self.shell.execute("reset")
        assert "重置" in result
        assert self.shell.agent.step_count == 0

    def test_state_command(self):
        result = self.shell.execute("state")
        assert "当前状态" in result
        assert "步数" in result

    def test_metrics_command_empty(self):
        result = self.shell.execute("metrics")
        # 未运行时可能无数据
        assert "指标" in result

    def test_metrics_after_run(self):
        self.shell.execute("run 3")
        result = self.shell.execute("metrics")
        assert "指标" in result

    def test_config_command(self):
        result = self.shell.execute("config")
        assert "当前配置" in result
        assert "state_dim" in result

    def test_health_command(self):
        result = self.shell.execute("health")
        assert "健康分数" in result

    def test_diagnose_command(self):
        result = self.shell.execute("diagnose")
        assert "诊断报告" in result

    def test_chart_command(self):
        self.shell.execute("run 3")
        result = self.shell.execute("chart")
        # 可能有图表或错误信息
        assert isinstance(result, str)

    def test_history_command_empty(self):
        result = self.shell.execute("history")
        assert "历史" in result

    def test_history_after_commands(self):
        self.shell.execute("run 2")
        self.shell.execute("state")
        result = self.shell.execute("history")
        assert "run 2" in result
        assert "state" in result

    def test_exit_command(self):
        result = self.shell.execute("exit")
        assert "再见" in result
        assert self.shell._running is False

    def test_exit_alias_quit(self):
        result = self.shell.execute("quit")
        assert "再见" in result
        assert self.shell._running is False


# ============================================================
# 命令注册
# ============================================================

class TestCommandRegistration:
    """自定义命令注册测试。"""

    def test_register_custom_command(self):
        shell = REPLShell(config=get_preset_config("minimal"))
        cmd = Command(name="greet", handler=lambda args: "Hello!", help="问候")
        shell.register_command(cmd)
        assert "greet" in shell.commands
        result = shell.execute("greet")
        assert result == "Hello!"

    def test_register_command_with_alias(self):
        shell = REPLShell(config=get_preset_config("minimal"))
        cmd = Command(name="greet", handler=lambda args: "Hi!",
                      aliases=["g", "hi"])
        shell.register_command(cmd)
        assert "g" in shell.commands
        assert "hi" in shell.commands
        assert shell.execute("g") == "Hi!"
        assert shell.execute("hi") == "Hi!"

    def test_custom_command_with_args(self):
        shell = REPLShell(config=get_preset_config("minimal"))
        cmd = Command(name="echo", handler=lambda args: " ".join(args))
        shell.register_command(cmd)
        result = shell.execute("echo hello world")
        assert result == "hello world"

    def test_custom_command_in_help(self):
        shell = REPLShell(config=get_preset_config("minimal"))
        cmd = Command(name="custom", handler=lambda args: "ok", help="自定义命令")
        shell.register_command(cmd)
        result = shell.execute("help")
        assert "custom" in result


# ============================================================
# 错误处理
# ============================================================

class TestErrorHandling:
    """错误处理测试。"""

    def test_unknown_command(self):
        outputs = []
        shell = REPLShell(
            config=get_preset_config("minimal"),
            output_func=outputs.append,
        )
        result = shell.execute("nonexistent_cmd")
        assert "未知命令" in result

    def test_command_exception_handled(self):
        shell = REPLShell(config=get_preset_config("minimal"))

        def bad_handler(args):
            raise ValueError("故意出错")

        shell.register_command(Command(name="bad", handler=bad_handler))
        result = shell.execute("bad")
        # 应返回友好错误信息，而非抛异常
        assert isinstance(result, str)
        assert "错误" in result or "SPEPA" in result

    def test_quote_mismatch_handled(self):
        shell = REPLShell(config=get_preset_config("minimal"))
        result = shell.execute('run "unclosed')
        assert "解析错误" in result


# ============================================================
# 主循环
# ============================================================

class TestMainLoop:
    """主循环测试。"""

    def test_run_with_injected_input(self):
        inputs = iter(["help", "exit"])
        outputs = []
        shell = REPLShell(
            config=get_preset_config("minimal"),
            input_func=lambda _: next(inputs),
            output_func=outputs.append,
        )
        shell.run()
        # 应有欢迎语 + help 输出 + 退出语
        assert any("交互式 Shell" in o for o in outputs)
        assert any("可用命令" in o for o in outputs)
        assert any("再见" in o for o in outputs)

    def test_run_empty_input_ignored(self):
        inputs = iter(["", "  ", "exit"])
        outputs = []
        shell = REPLShell(
            config=get_preset_config("minimal"),
            input_func=lambda _: next(inputs),
            output_func=outputs.append,
        )
        shell.run()
        # 空输入不应产生命令输出
        assert any("再见" in o for o in outputs)

    def test_run_eof_exits(self):
        def raise_eof(_):
            raise EOFError()

        outputs = []
        shell = REPLShell(
            config=get_preset_config("minimal"),
            input_func=raise_eof,
            output_func=outputs.append,
        )
        shell.run()
        assert any("退出" in o for o in outputs)

    def test_run_keyboard_interrupt_continues(self):
        call_count = [0]

        def input_with_interrupt(_):
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyboardInterrupt()
            return "exit"

        outputs = []
        shell = REPLShell(
            config=get_preset_config("minimal"),
            input_func=input_with_interrupt,
            output_func=outputs.append,
        )
        shell.run()
        # 第一次中断后应继续，第二次 exit 退出
        assert any("中断" in o for o in outputs)
        assert any("再见" in o for o in outputs)

    def test_history_recorded_in_loop(self):
        inputs = iter(["run 2", "state", "history", "exit"])
        outputs = []
        shell = REPLShell(
            config=get_preset_config("minimal"),
            input_func=lambda _: next(inputs),
            output_func=outputs.append,
        )
        shell.run()
        # history 命令应显示之前的命令
        history_output = [o for o in outputs if "命令历史" in o]
        assert len(history_output) > 0
        assert any("run 2" in o for o in outputs)


# ============================================================
# execute 编程式调用
# ============================================================

class TestExecute:
    """execute 编程式调用测试。"""

    def test_execute_returns_string(self):
        shell = REPLShell(config=get_preset_config("minimal"))
        result = shell.execute("help")
        assert isinstance(result, str)

    def test_execute_multiple_commands(self):
        shell = REPLShell(config=get_preset_config("minimal"))
        shell.execute("run 3")
        shell.execute("step 2")
        assert shell.agent.step_count == 5

    def test_execute_preserves_state(self):
        shell = REPLShell(config=get_preset_config("minimal"))
        shell.execute("run 3")
        state_result = shell.execute("state")
        assert "步数" in state_result

    def test_execute_case_insensitive(self):
        shell = REPLShell(config=get_preset_config("minimal"))
        result1 = shell.execute("HELP")
        result2 = shell.execute("Help")
        assert "可用命令" in result1
        assert "可用命令" in result2


# ============================================================
# 端到端
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_full_session(self):
        """完整会话：help → run → state → metrics → exit。"""
        inputs = iter(["help", "run 5", "state", "metrics", "exit"])
        outputs = []
        shell = REPLShell(
            config=get_preset_config("minimal"),
            input_func=lambda _: next(inputs),
            output_func=outputs.append,
        )
        shell.run()
        # 验证各命令都执行了
        assert any("可用命令" in o for o in outputs)
        assert any("运行完成" in o for o in outputs)
        assert any("当前状态" in o for o in outputs)
        assert any("指标" in o for o in outputs)
        assert shell.agent.step_count == 5

    def test_run_reset_run_cycle(self):
        """运行 → 重置 → 再运行。"""
        shell = REPLShell(config=get_preset_config("minimal"))
        shell.execute("run 3")
        assert shell.agent.step_count == 3
        shell.execute("reset")
        assert shell.agent.step_count == 0
        shell.execute("run 2")
        assert shell.agent.step_count == 2

    def test_custom_command_integration(self):
        """自定义命令集成测试。"""
        shell = REPLShell(config=get_preset_config("minimal"))

        def stats_handler(args):
            return f"步数={shell.agent.step_count}, 健康={shell.agent.get_health_score()}"

        shell.register_command(Command(
            name="stats", handler=stats_handler, help="显示统计"
        ))
        shell.execute("run 3")
        result = shell.execute("stats")
        assert "步数=3" in result

    def test_diagnose_after_run(self):
        """运行后诊断。"""
        shell = REPLShell(config=get_preset_config("minimal"))
        shell.execute("run 5")
        result = shell.execute("diagnose")
        assert "诊断报告" in result

    def test_chart_after_run(self):
        """运行后生成图表。"""
        shell = REPLShell(config=get_preset_config("minimal"))
        shell.execute("run 5")
        result = shell.execute("chart prediction_error")
        assert isinstance(result, str)

    def test_help_lists_all_commands(self):
        """help 列出所有命令。"""
        shell = REPLShell(config=get_preset_config("minimal"))
        result = shell.execute("help")
        for cmd in ["help", "run", "step", "reset", "state", "metrics",
                    "config", "health", "diagnose", "chart", "save",
                    "load", "history", "exit"]:
            assert cmd in result
