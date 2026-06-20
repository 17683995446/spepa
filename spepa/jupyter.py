"""SPEPA Jupyter Notebook 集成。

设计原则（第一性原理）：
- 可选依赖：IPython 不存在时优雅降级，不报错
- 富文本显示：在 Notebook 中自动渲染 HTML/SVG，而非纯文本
- 魔法命令：提供 %spepa 等魔法命令，降低交互成本
- 零配置：import spepa 后自动注册，无需手动调用

与 repl 模块的区别：
- repl  : 终端交互式 Shell（基于 input/print）
- jupyter : Notebook 富文本交互（基于 IPython display）

主要功能：
- is_jupyter()       : 检测是否在 Jupyter 环境
- display_agent()    : 富文本显示 Agent 状态
- display_config()   : 富文本显示配置
- display_metrics()  : 富文本显示指标图表
- SPEPAMagics        : IPython 魔法命令
- register_magics()  : 注册魔法命令
- load_ipython_extension() : IPython 扩展入口
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from spepa.utils.logger import get_logger
from spepa.utils.config import SPEPAConfig, get_preset_config
from spepa.agent import SPEPAAgent
from spepa.visualization import generate_chart, ASCIILineChart

logger = get_logger("spepa.jupyter")


# ============================================================
# 环境检测
# ============================================================

def is_jupyter() -> bool:
    """检测是否在 Jupyter/IPython 环境中。

    返回：
        True 如果在 Jupyter 环境中
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        # 检查是否为 ZMQInteractiveShell（Jupyter notebook/lab）
        shell_class = type(ip).__name__
        return "ZMQ" in shell_class or "Terminal" in shell_class
    except ImportError:
        return False


def is_ipython_available() -> bool:
    """检测 IPython 是否可用。"""
    try:
        import IPython  # noqa: F401
        return True
    except ImportError:
        return False


def _get_display():
    """获取 IPython display 函数（不可用则返回 print）。"""
    if is_ipython_available():
        try:
            from IPython.display import display
            return display
        except ImportError:
            pass
    return print


def _get_html():
    """获取 IPython HTML 类（不可用则返回 None）。"""
    if is_ipython_available():
        try:
            from IPython.display import HTML
            return HTML
        except ImportError:
            pass
    return None


def _get_markdown():
    """获取 IPython Markdown 类（不可用则返回 None）。"""
    if is_ipython_available():
        try:
            from IPython.display import Markdown
            return Markdown
        except ImportError:
            pass
    return None


# ============================================================
# 富文本显示
# ============================================================

def display_agent(agent: SPEPAAgent, verbose: bool = False) -> None:
    """富文本显示 Agent 状态。

    在 Jupyter 中渲染为 HTML 表格，在终端中打印文本。

    参数：
        agent : SPEPAAgent 实例
        verbose : 是否显示详细信息
    """
    state = agent.get_state()
    display = _get_display()
    html_cls = _get_html()

    if html_cls is not None and is_jupyter():
        # Jupyter 环境：渲染 HTML
        html = _agent_to_html(state, verbose=verbose)
        display(html_cls(html))
    else:
        # 终端环境：打印文本
        text = _agent_to_text(state, verbose=verbose)
        display(text)


def _agent_to_html(state: Dict[str, Any], verbose: bool = False) -> str:
    """将 Agent 状态转为 HTML。"""
    rows = []
    rows.append("<h3>SPEPA Agent 状态</h3>")
    rows.append("<table style='border-collapse: collapse;'>")

    def add_row(key, value, indent=0):
        pad = "&nbsp;" * (indent * 20)
        if isinstance(value, dict):
            rows.append(f"<tr><td style='border:1px solid #ddd;padding:4px;'><b>{pad}{key}</b></td>"
                        f"<td style='border:1px solid #ddd;padding:4px;'></td></tr>")
            for k, v in value.items():
                add_row(k, v, indent + 1)
        elif isinstance(value, float):
            rows.append(f"<tr><td style='border:1px solid #ddd;padding:4px;'>{pad}{key}</td>"
                        f"<td style='border:1px solid #ddd;padding:4px;'>{value:.4f}</td></tr>")
        else:
            rows.append(f"<tr><td style='border:1px solid #ddd;padding:4px;'>{pad}{key}</td>"
                        f"<td style='border:1px solid #ddd;padding:4px;'>{value}</td></tr>")

    for key, value in state.items():
        if not verbose and key in ("predictive_model", "drive_engine", "gap_detector",
                                    "goal_hierarchy", "resource_controller", "backpressure",
                                    "degradation"):
            continue
        add_row(key, value)

    rows.append("</table>")
    return "\n".join(rows)


def _agent_to_text(state: Dict[str, Any], verbose: bool = False) -> str:
    """将 Agent 状态转为文本。"""
    lines = ["[SPEPA Agent 状态]"]
    for key, value in state.items():
        if not verbose and key in ("predictive_model", "drive_engine", "gap_detector",
                                    "goal_hierarchy", "resource_controller", "backpressure",
                                    "degradation"):
            continue
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")
        elif isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def display_config(config: SPEPAConfig) -> None:
    """富文本显示配置。

    参数：
        config : SPEPAConfig 实例
    """
    display = _get_display()
    html_cls = _get_html()
    config_dict = config.to_dict()

    if html_cls is not None and is_jupyter():
        html = _config_to_html(config_dict)
        display(html_cls(html))
    else:
        text = _config_to_text(config_dict)
        display(text)


def _config_to_html(config_dict: Dict[str, Any]) -> str:
    """将配置转为 HTML。"""
    rows = ["<h3>SPEPA 配置</h3>",
            "<table style='border-collapse: collapse;'>"]
    for key, value in config_dict.items():
        if isinstance(value, dict):
            rows.append(f"<tr><td style='border:1px solid #ddd;padding:4px;'><b>{key}</b></td>"
                        f"<td style='border:1px solid #ddd;padding:4px;'></td></tr>")
            for k, v in value.items():
                rows.append(f"<tr><td style='border:1px solid #ddd;padding:4px;'>&nbsp;&nbsp;{k}</td>"
                            f"<td style='border:1px solid #ddd;padding:4px;'>{v}</td></tr>")
        else:
            rows.append(f"<tr><td style='border:1px solid #ddd;padding:4px;'><b>{key}</b></td>"
                        f"<td style='border:1px solid #ddd;padding:4px;'>{value}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _config_to_text(config_dict: Dict[str, Any]) -> str:
    """将配置转为文本。"""
    lines = ["[SPEPA 配置]"]
    for key, value in config_dict.items():
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for k, v in value.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def display_metrics(agent: SPEPAAgent,
                    metric_names: Optional[List[str]] = None) -> None:
    """富文本显示指标图表。

    在 Jupyter 中渲染 SVG 图表，在终端中打印 ASCII 图表。

    参数：
        agent : SPEPAAgent 实例
        metric_names : 要显示的指标名列表（默认显示 prediction_error）
    """
    if metric_names is None:
        metric_names = ["prediction_error"]

    display = _get_display()
    html_cls = _get_html()

    if html_cls is not None and is_jupyter():
        # Jupyter：渲染 SVG
        from spepa.visualization import plot_metrics
        svg = plot_metrics(agent, metric_names)
        if svg:
            display(html_cls(svg))
        else:
            display("无可用指标数据")
    else:
        # 终端：ASCII 图表
        chart = ASCIILineChart()
        for name in metric_names:
            text = chart.render({}, title=name)
            display(text)


def display_diagnosis(agent: SPEPAAgent) -> None:
    """富文本显示诊断结果。

    参数：
        agent : SPEPAAgent 实例
    """
    from spepa.diagnostics import diagnose, format_diagnosis
    display = _get_display()
    html_cls = _get_html()
    markdown_cls = _get_markdown()

    result = diagnose(agent)

    if is_jupyter() and (html_cls is not None or markdown_cls is not None):
        # Jupyter：渲染 Markdown
        text = format_diagnosis(result)
        # 转为简单的 Markdown
        md_text = text.replace("=", "## ").replace("[", "**[").replace("]", "]**")
        if markdown_cls is not None:
            display(markdown_cls(text))
        else:
            display(text)
    else:
        display(format_diagnosis(result))


# ============================================================
# IPython 魔法命令
# ============================================================

class SPEPAMagics:
    """SPEPA IPython 魔法命令。

    提供以下魔法命令（需通过 register_magics 注册后使用）：
    - %spepa_run <n>      : 运行 n 步
    - %spepa_state        : 显示 Agent 状态
    - %spepa_config       : 显示配置
    - %spepa_metrics      : 显示指标图表
    - %spepa_diagnose     : 诊断 Agent
    - %spepa_help         : 显示帮助
    """

    def __init__(self, agent: Optional[SPEPAAgent] = None,
                 config: Optional[SPEPAConfig] = None):
        """初始化。

        参数：
            agent : 已存在的 Agent（None 则自动创建）
            config : 创建 Agent 的配置
        """
        self.agent = agent or SPEPAAgent(config or get_preset_config("minimal"))
        self.commands = {
            "run": self._cmd_run,
            "state": self._cmd_state,
            "config": self._cmd_config,
            "metrics": self._cmd_metrics,
            "diagnose": self._cmd_diagnose,
            "help": self._cmd_help,
        }

    def execute(self, line: str) -> Any:
        """执行魔法命令。

        参数：
            line : 命令行（如 "run 10"）

        返回：
            命令结果
        """
        parts = line.strip().split()
        if not parts:
            return self._cmd_help([])

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd not in self.commands:
            print(f"未知命令: {cmd}。使用 %spepa help 查看可用命令。")
            return None

        return self.commands[cmd](args)

    def _cmd_run(self, args: List[str]) -> Dict[str, Any]:
        """运行 n 步。"""
        n = int(args[0]) if args else 10
        result = self.agent.run(n_steps=n)
        display_agent(self.agent)
        return result

    def _cmd_state(self, args: List[str]) -> None:
        """显示状态。"""
        verbose = "--verbose" in args or "-v" in args
        display_agent(self.agent, verbose=verbose)

    def _cmd_config(self, args: List[str]) -> None:
        """显示配置。"""
        display_config(self.agent.config)

    def _cmd_metrics(self, args: List[str]) -> None:
        """显示指标。"""
        metric_names = args if args else None
        display_metrics(self.agent, metric_names=metric_names)

    def _cmd_diagnose(self, args: List[str]) -> None:
        """诊断。"""
        display_diagnosis(self.agent)

    def _cmd_help(self, args: List[str]) -> None:
        """显示帮助。"""
        help_text = """
SPEPA 魔法命令:
  %spepa run <n>      运行 n 步（默认 10）
  %spepa state [-v]   显示 Agent 状态（-v 详细）
  %spepa config       显示配置
  %spepa metrics [m]  显示指标图表（m 为指标名）
  %spepa diagnose     诊断 Agent
  %spepa help         显示此帮助
"""
        print(help_text)
        return help_text


# ============================================================
# 注册与扩展入口
# ============================================================

_global_magics: Optional[SPEPAMagics] = None


def get_magics(agent: Optional[SPEPAAgent] = None) -> SPEPAMagics:
    """获取全局魔法命令实例。

    参数：
        agent : 可选的 Agent 实例（首次调用时设置）

    返回：
        SPEPAMagics 实例
    """
    global _global_magics
    if _global_magics is None or agent is not None:
        _global_magics = SPEPAMagics(agent=agent)
    return _global_magics


def register_magics(agent: Optional[SPEPAAgent] = None) -> bool:
    """注册 IPython 魔法命令。

    参数：
        agent : 可选的 Agent 实例

    返回：
        True 如果注册成功，False 如果 IPython 不可用
    """
    if not is_ipython_available():
        logger.info("IPython 不可用，跳过魔法命令注册")
        return False

    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False

        magics = get_magics(agent)

        def spepa_magic(line: str):
            return magics.execute(line)

        ip.register_magic_function(spepa_magic, magic_name="spepa")
        logger.info("SPEPA 魔法命令已注册（%spepa）")
        return True
    except Exception as e:
        logger.warning("注册魔法命令失败: %s", e)
        return False


def load_ipython_extension(ipython):
    """IPython 扩展入口点。

    使用方式：在 Notebook 中执行 %load_ext spepa.jupyter
    """
    register_magics()
    print("SPEPA Jupyter 扩展已加载。使用 %spepa help 查看可用命令。")


# ============================================================
# 便捷显示函数（不依赖 Jupyter 也能用）
# ============================================================

def show(agent: SPEPAAgent, what: str = "state") -> None:
    """便捷显示函数。

    参数：
        agent : SPEPAAgent 实例
        what : 要显示的内容（"state"/"config"/"metrics"/"diagnose"）
    """
    if what == "state":
        display_agent(agent)
    elif what == "config":
        display_config(agent.config)
    elif what == "metrics":
        display_metrics(agent)
    elif what == "diagnose":
        display_diagnosis(agent)
    else:
        print(f"未知显示类型: {what}。可选: state/config/metrics/diagnose")


def quickstart(config: Optional[SPEPAConfig] = None,
               n_steps: int = 10) -> SPEPAAgent:
    """快速开始：创建 Agent、运行、显示结果。

    参数：
        config : 配置（默认 minimal 预设）
        n_steps : 运行步数

    返回：
        创建的 SPEPAAgent 实例
    """
    config = config or get_preset_config("minimal")
    agent = SPEPAAgent(config)
    agent.run(n_steps=n_steps)
    display_agent(agent)
    return agent
