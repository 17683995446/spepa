"""SPEPA REPL 交互式 Shell。

设计原则（第一性原理）：
- 即时反馈：每条命令立即执行并返回结果
- 可发现：help 命令列出所有可用操作
- 可组合：命令可链式执行，结果可累积
- 可扩展：支持自定义命令注册
- 可测试：输入/输出函数可注入，无需真实终端

与 CLI 的区别：
- cli.py : 一次性命令行执行（spepa run --steps 100）
- repl.py : 交互式循环，保持 Agent 状态，支持探索式操作

主要功能：
- REPLShell        : 交互式 Shell
- start_repl()     : 一键启动 REPL
- Command          : 命令数据结构
"""
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
import shlex

from spepa.utils.config import SPEPAConfig, get_preset_config
from spepa.utils.logger import get_logger
from spepa.agent import SPEPAAgent
from spepa.diagnostics import diagnose, diagnose_config, format_diagnosis
from spepa.state_manager import StateManager
from spepa.visualization import render_ascii_chart

logger = get_logger("spepa.repl")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Command:
    """REPL 命令定义。"""
    name: str                                # 命令名
    handler: Callable[..., str]              # 处理函数 (args, kwargs) -> 输出字符串
    help: str = ""                           # 帮助文本
    aliases: List[str] = field(default_factory=list)  # 别名


# ============================================================
# REPL Shell
# ============================================================

class REPLShell:
    """SPEPA 交互式 Shell。

    使用方式：
        shell = REPLShell()
        shell.run()  # 启动交互循环

    测试方式（注入输入/输出）：
        inputs = iter(["help", "state", "exit"])
        outputs = []
        shell = REPLShell(input_func=lambda _: next(inputs),
                          output_func=outputs.append)
        shell.run()
    """

    PROMPT = "spepa> "

    def __init__(self,
                 agent: Optional[SPEPAAgent] = None,
                 config: Optional[SPEPAConfig] = None,
                 input_func: Optional[Callable[[str], str]] = None,
                 output_func: Optional[Callable[[str], None]] = None):
        """初始化 REPL Shell。

        参数：
            agent : 已存在的 Agent（若为 None 则自动创建）
            config : 创建 Agent 使用的配置（agent 为 None 时生效）
            input_func : 输入函数（默认 input，测试时可注入）
            output_func : 输出函数（默认 print，测试时可注入）
        """
        self.input_func = input_func or input
        self.output_func = output_func or print
        self.agent = agent or SPEPAAgent(config or get_preset_config("minimal"))
        self.state_manager = StateManager(self.agent)
        self.commands: Dict[str, Command] = {}
        self.history: List[str] = []
        self._running = False
        self._register_builtin_commands()

    # --------------------------------------------------------
    # 命令注册
    # --------------------------------------------------------

    def register_command(self, command: Command) -> None:
        """注册自定义命令。

        参数：
            command : Command 实例
        """
        self.commands[command.name] = command
        for alias in command.aliases:
            self.commands[alias] = command

    def _register_builtin_commands(self) -> None:
        """注册内置命令。"""
        builtins = [
            Command(name="help", handler=self._cmd_help,
                    help="显示所有可用命令", aliases=["?"]),
            Command(name="run", handler=self._cmd_run,
                    help="运行 N 步: run <n>"),
            Command(name="step", handler=self._cmd_step,
                    help="执行单步或 N 步: step [n]"),
            Command(name="reset", handler=self._cmd_reset,
                    help="重置 Agent 状态"),
            Command(name="state", handler=self._cmd_state,
                    help="查看当前状态"),
            Command(name="metrics", handler=self._cmd_metrics,
                    help="查看指标摘要"),
            Command(name="config", handler=self._cmd_config,
                    help="查看当前配置"),
            Command(name="health", handler=self._cmd_health,
                    help="查看健康分数"),
            Command(name="diagnose", handler=self._cmd_diagnose,
                    help="诊断 Agent 状态"),
            Command(name="chart", handler=self._cmd_chart,
                    help="生成 ASCII 图表: chart [metric_name]"),
            Command(name="save", handler=self._cmd_save,
                    help="保存状态: save <path>"),
            Command(name="load", handler=self._cmd_load,
                    help="加载状态: load <path>"),
            Command(name="history", handler=self._cmd_history,
                    help="查看命令历史"),
            Command(name="exit", handler=self._cmd_exit,
                    help="退出 REPL", aliases=["quit"]),
        ]
        for cmd in builtins:
            self.register_command(cmd)

    # --------------------------------------------------------
    # 主循环
    # --------------------------------------------------------

    def run(self) -> None:
        """启动 REPL 主循环。"""
        self._running = True
        self._output("SPEPA 交互式 Shell（输入 help 查看命令，exit 退出）")
        while self._running:
            try:
                line = self.input_func(self.PROMPT)
            except EOFError:
                self._output("退出")
                break
            except KeyboardInterrupt:
                self._output("\n中断（输入 exit 退出）")
                continue

            line = line.strip()
            if not line:
                continue

            self._execute(line)

    def execute(self, line: str) -> str:
        """执行单条命令（不进入主循环，用于编程式调用）。

        参数：
            line : 命令行字符串

        返回：
            命令输出字符串
        """
        return self._execute(line, capture=True) or ""

    def _execute(self, line: str, capture: bool = False) -> Optional[str]:
        """执行一条命令。

        参数：
            line : 命令行
            capture : 是否捕获输出（True=返回字符串，False=直接输出）
        """
        line = line.strip()
        if not line:
            return "" if capture else None

        # 记录到历史（无论主循环还是编程式调用）
        self.history.append(line)

        try:
            parts = shlex.split(line)
        except ValueError:
            msg = f"解析错误: 引号不匹配"
            if capture:
                return msg
            self._output(msg)
            return msg

        if not parts:
            return "" if capture else None

        cmd_name = parts[0].lower()
        args = parts[1:]

        if cmd_name not in self.commands:
            msg = f"未知命令: {cmd_name}（输入 help 查看可用命令）"
            if capture:
                return msg
            self._output(msg)
            return msg

        command = self.commands[cmd_name]
        try:
            result = command.handler(args)
            if result:
                if capture:
                    return result
                self._output(result)
            return result if capture else None
        except Exception as e:
            from spepa.diagnostics import friendly_error
            msg = friendly_error(e, context=f"REPL 命令 '{cmd_name}'")
            if capture:
                return msg
            self._output(msg)
            return msg

    def _output(self, text: str) -> None:
        """输出文本。"""
        self.output_func(text)

    # --------------------------------------------------------
    # 内置命令处理
    # --------------------------------------------------------

    def _cmd_help(self, args: List[str]) -> str:
        """help 命令。"""
        lines = ["可用命令:"]
        # 去重（别名指向同一 Command 对象）
        seen = set()
        for name, cmd in sorted(self.commands.items()):
            if cmd.name in seen:
                continue
            seen.add(cmd.name)
            aliases = f" (别名: {', '.join(cmd.aliases)})" if cmd.aliases else ""
            lines.append(f"  {cmd.name:<12} {cmd.help}{aliases}")
        return "\n".join(lines)

    def _cmd_run(self, args: List[str]) -> str:
        """run <n> 命令。"""
        n = int(args[0]) if args else 10
        n = max(1, min(n, 10000))  # 安全限制
        result = self.agent.run(n_steps=n)
        return (f"运行完成: {result['total_steps']} 步, "
                f"耗时 {result['total_time']:.2f}s, "
                f"在线周期 {result['online_cycles']}, "
                f"离线周期 {result['offline_cycles']}")

    def _cmd_step(self, args: List[str]) -> str:
        """step [n] 命令。"""
        n = int(args[0]) if args else 1
        n = max(1, min(n, 1000))
        for _ in range(n):
            self.agent.step()
        return f"执行了 {n} 步，当前步数: {self.agent.step_count}"

    def _cmd_reset(self, args: List[str]) -> str:
        """reset 命令。"""
        self.agent.reset()
        return "Agent 已重置"

    def _cmd_state(self, args: List[str]) -> str:
        """state 命令。"""
        state = self.agent.get_state()
        lines = ["[当前状态]"]
        lines.append(f"  步数: {state.get('step_count', 0)}")
        lines.append(f"  在线周期: {state.get('online_cycles', 0)}")
        lines.append(f"  离线周期: {state.get('offline_cycles', 0)}")
        lines.append(f"  探索比例: {state.get('exploration_ratio', 0):.4f}")
        res = state.get("resources", {})
        lines.append(f"  CPU: {res.get('cpu', 0):.1f}%")
        lines.append(f"  内存: {res.get('memory', 0):.1f}%")
        lines.append(f"  资源受限: {res.get('within_limits', True)}")
        lines.append(f"  降级级别: {state.get('degradation', {}).get('current_level', 'UNKNOWN')}")
        return "\n".join(lines)

    def _cmd_metrics(self, args: List[str]) -> str:
        """metrics 命令。"""
        summary = self.agent.metrics.summary()
        if not summary:
            return "[指标] 暂无指标数据（请先运行几步）"
        lines = ["[指标摘要]"]
        for name, value in summary.items():
            if isinstance(value, dict):
                lines.append(f"  {name}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        lines.append(f"    {k}: {v:.6f}")
                    else:
                        lines.append(f"    {k}: {v}")
            elif isinstance(value, float):
                lines.append(f"  {name}: {value:.6f}")
            else:
                lines.append(f"  {name}: {value}")
        return "\n".join(lines)

    def _cmd_config(self, args: List[str]) -> str:
        """config 命令。"""
        config = self.agent.config
        lines = ["[当前配置]"]
        lines.append(f"  state_dim: {config.state_dim}")
        lines.append(f"  action_dim: {config.action_dim}")
        lines.append(f"  obs_dim: {config.obs_dim}")
        lines.append(f"  learning_rate_micro: {config.learning_rate_micro}")
        lines.append(f"  learning_rate_struct: {config.learning_rate_struct}")
        lines.append(f"  max_steps: {config.max_steps}")
        lines.append(f"  seed: {config.seed}")
        lines.append(f"  fast_memory_capacity: {config.fast_memory_capacity}")
        lines.append(f"  slow_memory_capacity: {config.slow_memory_capacity}")
        lines.append(f"  safety_hard_boundary: {config.safety_hard_boundary}")
        lines.append(f"  CPU 上限: {config.resources.cpu_max_percent}%")
        lines.append(f"  内存上限: {config.resources.memory_max_percent}%")
        return "\n".join(lines)

    def _cmd_health(self, args: List[str]) -> str:
        """health 命令。"""
        try:
            score = self.agent.get_health_score()
            if score is None:
                return "[健康分数] 暂无数据"
            if score >= 0.7:
                status = "良好"
            elif score >= 0.4:
                status = "一般"
            else:
                status = "较差"
            return f"[健康分数] {score:.4f} ({status})"
        except Exception as e:
            return f"[健康分数] 无法获取: {e}"

    def _cmd_diagnose(self, args: List[str]) -> str:
        """diagnose 命令。"""
        result = diagnose(self.agent)
        return format_diagnosis(result)

    def _cmd_chart(self, args: List[str]) -> str:
        """chart [metric] 命令。"""
        metric_name = args[0] if args else "prediction_error"
        try:
            return render_ascii_chart(self.agent, metric_name=metric_name)
        except Exception as e:
            return f"无法生成图表: {e}"

    def _cmd_save(self, args: List[str]) -> str:
        """save <path> 命令。"""
        if not args:
            return "用法: save <path>"
        path = args[0]
        success = self.state_manager.save(force=True)
        if success:
            return f"状态已保存"
        return "保存失败"

    def _cmd_load(self, args: List[str]) -> str:
        """load <path> 命令。"""
        if not args:
            return "用法: load <path>"
        path = args[0]
        success = self.state_manager.load()
        if success:
            return f"状态已加载，当前步数: {self.agent.step_count}"
        return "加载失败（无检查点）"

    def _cmd_history(self, args: List[str]) -> str:
        """history 命令。"""
        if not self.history:
            return "[历史] 暂无命令历史"
        lines = ["[命令历史]"]
        for i, cmd in enumerate(self.history[-20:], 1):  # 最近 20 条
            lines.append(f"  {i}. {cmd}")
        return "\n".join(lines)

    def _cmd_exit(self, args: List[str]) -> str:
        """exit 命令。"""
        self._running = False
        return "再见！"


# ============================================================
# 便捷函数
# ============================================================

def start_repl(config: Optional[SPEPAConfig] = None,
               agent: Optional[SPEPAAgent] = None) -> None:
    """一键启动 REPL Shell。

    参数：
        config : 使用的配置（agent 为 None 时生效）
        agent : 已存在的 Agent
    """
    shell = REPLShell(agent=agent, config=config)
    shell.run()
