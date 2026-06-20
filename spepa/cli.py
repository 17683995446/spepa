"""SPEPA 命令行接口。

设计原则（第一性原理）：
- 零配置可用：不传参数也能运行
- 渐进式复杂度：简单任务一行命令，复杂任务通过选项配置
- 一致性：所有命令共享 --preset/--seed/--quiet 等通用选项

命令：
- run         : 基础在线运行 N 步
- autonomous  : 自主驱动运行 N 步
- perpetual   : 永续运行（异常自恢复）
- collaborate : 多Agent协作完成任务
- presets     : 列出可用预设配置
- info        : 显示系统信息与配置
"""
import sys
import json
import argparse
from typing import List, Optional

import numpy as np

from spepa.utils.config import (
    SPEPAConfig, get_default_config, get_preset_config, list_presets,
)
from spepa.utils.logger import get_logger
from spepa.agent import SPEPAAgent
from spepa.multiagent.collaboration import CollaborationMode
from spepa.dashboard import WebDashboard


def _build_config(args) -> SPEPAConfig:
    """根据命令行参数构建配置。

    优先级（从高到低）：
    1. 命令行参数（--seed, --state-dim, --action-dim）
    2. 配置文件（--config）
    3. 预设配置（--preset）
    4. 默认配置
    """
    # 1. 基础配置：配置文件 > 预设 > 默认
    config_file = getattr(args, 'config', None)
    if config_file:
        config = SPEPAConfig.load_from_file(config_file)
    elif args.preset:
        config = get_preset_config(args.preset)
    else:
        config = get_default_config()
    # 2. 命令行参数覆盖
    if args.seed is not None:
        config.seed = args.seed
    if args.state_dim is not None:
        config.state_dim = args.state_dim
    if args.action_dim is not None:
        config.action_dim = args.action_dim
    return config


def _print_result(result: dict, quiet: bool = False) -> None:
    """打印运行结果。"""
    if quiet:
        return
    # 精简输出，避免巨大字典刷屏
    summary_keys = [
        "total_steps", "total_time", "online_cycles", "offline_cycles",
        "goals_generated", "goals_achieved", "achievement_rate",
        "success_rate", "health_score", "mode",
    ]
    output = {}
    for key in summary_keys:
        if key in result:
            output[key] = result[key]
    # 添加指标摘要
    if "metrics_summary" in result:
        output["metrics"] = result["metrics_summary"]
    print(json.dumps(output, indent=2, default=str, ensure_ascii=False))


def _run_metrics_summary(agent: SPEPAAgent) -> dict:
    """获取 Agent 指标摘要。"""
    try:
        return agent.metrics.summary()
    except Exception:
        return {}


def cmd_run(args) -> int:
    """基础在线运行。"""
    config = _build_config(args)
    agent = SPEPAAgent(config)
    n_steps = args.steps or config.max_steps
    print(f"运行 {n_steps} 步（preset={args.preset or 'balanced'}, seed={config.seed}）...")
    result = agent.run(n_steps=n_steps)
    result["metrics_summary"] = _run_metrics_summary(agent)
    _print_result(result, args.quiet)
    return 0


def cmd_autonomous(args) -> int:
    """自主驱动运行。"""
    config = _build_config(args)
    agent = SPEPAAgent(config)
    n_steps = args.steps or 100
    print(f"自主驱动运行 {n_steps} 步...")
    result = agent.run_autonomous(n_steps=n_steps)
    result["metrics_summary"] = _run_metrics_summary(agent)
    _print_result(result, args.quiet)
    return 0


def cmd_perpetual(args) -> int:
    """永续运行。"""
    config = _build_config(args)
    agent = SPEPAAgent(config)
    max_cycles = args.cycles or 100
    print(f"永续运行（max_cycles={max_cycles}, auto_recover={not args.no_recover}）...")
    try:
        result = agent.run_perpetual(
            max_cycles=max_cycles,
            auto_recover=not args.no_recover,
        )
        result["metrics_summary"] = _run_metrics_summary(agent)
        _print_result(result, args.quiet)
    except KeyboardInterrupt:
        print("\n用户中断，停止永续运行。")
    return 0


def cmd_collaborate(args) -> int:
    """多Agent协作。"""
    config = _build_config(args)
    agent = SPEPAAgent(config)
    # 构建目标状态
    target_value = args.target if args.target is not None else 0.8
    target = np.ones(config.state_dim) * target_value
    mode_map = {
        "parallel": CollaborationMode.PARALLEL,
        "sequential": CollaborationMode.SEQUENTIAL,
        "competitive": CollaborationMode.COMPETITIVE,
    }
    mode = mode_map.get(args.mode, CollaborationMode.PARALLEL)
    print(f"多Agent协作（mode={args.mode}, target={target_value}）...")
    result = agent.collaborate(target, mode=mode, n_competitors=args.competitors)
    _print_result(result, args.quiet)
    return 0


def cmd_presets(args) -> int:
    """列出预设配置。"""
    presets = list_presets()
    print("可用预设配置：\n")
    for name, desc in presets.items():
        print(f"  {name:15s} - {desc}")
    print(f"\n使用方式: spepa run --preset <名称>")
    return 0


def cmd_tutorial(args) -> int:
    """运行交互式教程。"""
    from spepa.tutorial import list_tutorials, run_tutorial
    if args.tutorial_name is None:
        # 列出所有教程
        tutorials = list_tutorials()
        print("可用教程：\n")
        for name, desc in tutorials.items():
            print(f"  {name:15s} - {desc}")
        print(f"\n使用方式: spepa tutorial <名称>")
        print(f"非交互模式: spepa tutorial <名称> --no-interactive")
        return 0
    try:
        run_tutorial(args.tutorial_name, interactive=not args.no_interactive)
    except KeyError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_benchmark(args) -> int:
    """运行性能基准测试。"""
    from spepa.benchmark import compare_presets, format_benchmark_comparison
    presets = args.presets.split(",") if args.presets else None
    print(f"运行性能基准（steps={args.steps}, presets={presets or 'all'}）...\n")
    result = compare_presets(n_steps=args.steps, presets=presets)
    print(format_benchmark_comparison(result))
    return 0


def cmd_config_save(args) -> int:
    """保存配置到文件。"""
    config = _build_config(args)
    filepath = config.save_to_file(args.filepath)
    print(f"配置已保存到: {filepath}")
    print(f"  state_dim: {config.state_dim}, action_dim: {config.action_dim}")
    print(f"  max_steps: {config.max_steps}, seed: {config.seed}")
    print(f"\n使用方式: spepa run --config {filepath}")
    return 0


def cmd_config_show(args) -> int:
    """显示当前配置。"""
    config = _build_config(args)
    import json
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_info(args) -> int:
    """显示系统信息。"""
    import spepa
    config = _build_config(args)
    print(f"SPEPA v{spepa.__version__}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")
    print(f"\n当前配置:")
    print(f"  preset:        {args.preset or 'balanced'}")
    print(f"  state_dim:     {config.state_dim}")
    print(f"  action_dim:    {config.action_dim}")
    print(f"  obs_dim:       {config.obs_dim}")
    print(f"  max_steps:     {config.max_steps}")
    print(f"  seed:          {config.seed}")
    print(f"  cpu_max:       {config.resources.cpu_max_percent}%")
    print(f"  memory_max:    {config.resources.memory_max_percent}%")
    print(f"  exploration:   {config.exploration.exploration_ratio_min}-{config.exploration.exploration_ratio_max}")
    print(f"  safety_bound:  {config.safety_hard_boundary}")
    print(f"\n可用预设: {', '.join(list_presets().keys())}")
    return 0


def cmd_dashboard(args) -> int:
    """启动 Web Dashboard。"""
    config = _build_config(args)
    agent = SPEPAAgent(config)
    port = args.port or 8080
    host = args.host or "localhost"
    dashboard = WebDashboard(agent, host=host, port=port)
    print(f"SPEPA Web Dashboard 启动中...")
    print(f"访问 http://{host}:{port} 查看仪表盘")
    print(f"按 Ctrl+C 停止\n")
    try:
        dashboard.start(blocking=True)
    except KeyboardInterrupt:
        print("\n停止 Dashboard...")
        dashboard.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="spepa",
        description="SPEPA: Self-driven Perpetual Evolutionary Prediction-Correction Agent",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 通用选项函数
    def add_common_opts(sub):
        sub.add_argument("--config", type=str, default=None,
                         help="配置文件路径 (.json/.yaml)，优先于 --preset")
        sub.add_argument("--preset", type=str, default=None,
                         help="预设配置名称 (minimal/balanced/performance/conservative)")
        sub.add_argument("--seed", type=int, default=None,
                         help="随机种子")
        sub.add_argument("--state-dim", type=int, default=None,
                         help="状态维度")
        sub.add_argument("--action-dim", type=int, default=None,
                         help="动作维度")
        sub.add_argument("--quiet", "-q", action="store_true",
                         help="静默模式，不输出结果")

    # run 命令
    p_run = subparsers.add_parser("run", help="基础在线运行 N 步")
    p_run.add_argument("--steps", "-n", type=int, default=None,
                       help="运行步数（默认使用配置的 max_steps）")
    add_common_opts(p_run)
    p_run.set_defaults(func=cmd_run)

    # autonomous 命令
    p_auto = subparsers.add_parser("autonomous", help="自主驱动运行 N 步")
    p_auto.add_argument("--steps", "-n", type=int, default=None,
                        help="运行步数（默认 100）")
    add_common_opts(p_auto)
    p_auto.set_defaults(func=cmd_autonomous)

    # perpetual 命令
    p_perp = subparsers.add_parser("perpetual", help="永续运行（异常自恢复）")
    p_perp.add_argument("--cycles", "-c", type=int, default=None,
                        help="最大循环数（默认 100）")
    p_perp.add_argument("--no-recover", action="store_true",
                        help="禁用自动崩溃恢复")
    add_common_opts(p_perp)
    p_perp.set_defaults(func=cmd_perpetual)

    # collaborate 命令
    p_collab = subparsers.add_parser("collaborate", help="多Agent协作完成任务")
    p_collab.add_argument("--mode", "-m", type=str, default="parallel",
                          choices=["parallel", "sequential", "competitive"],
                          help="协作模式（默认 parallel）")
    p_collab.add_argument("--target", "-t", type=float, default=None,
                          help="目标状态值（0~1，默认 0.8）")
    p_collab.add_argument("--competitors", type=int, default=3,
                          help="竞争模式下的竞争者数量（默认 3）")
    add_common_opts(p_collab)
    p_collab.set_defaults(func=cmd_collaborate)

    # presets 命令
    p_presets = subparsers.add_parser("presets", help="列出可用预设配置")
    p_presets.set_defaults(func=cmd_presets)

    # tutorial 命令
    p_tutorial = subparsers.add_parser("tutorial", help="运行交互式教程")
    p_tutorial.add_argument("tutorial_name", type=str, nargs="?", default=None,
                            help="教程名称（省略则列出所有教程）")
    p_tutorial.add_argument("--no-interactive", action="store_true",
                            help="非交互模式（不暂停等待输入）")
    p_tutorial.set_defaults(func=cmd_tutorial)

    # benchmark 命令
    p_bench = subparsers.add_parser("benchmark", help="运行性能基准测试")
    p_bench.add_argument("--steps", "-n", type=int, default=50,
                         help="每个配置运行步数（默认 50）")
    p_bench.add_argument("--presets", "-p", type=str, default=None,
                         help="要对比的预设（逗号分隔，如 minimal,balanced）")
    p_bench.set_defaults(func=cmd_benchmark)

    # config 命令
    p_config = subparsers.add_parser("config", help="配置文件管理（保存/查看）")
    p_config_sub = p_config.add_subparsers(dest="config_action")
    # config save
    p_cfg_save = p_config_sub.add_parser("save", help="保存当前配置到文件")
    p_cfg_save.add_argument("filepath", type=str, help="保存路径 (.json/.yaml)")
    add_common_opts(p_cfg_save)
    p_cfg_save.set_defaults(func=cmd_config_save)
    # config show
    p_cfg_show = p_config_sub.add_parser("show", help="显示当前配置")
    add_common_opts(p_cfg_show)
    p_cfg_show.set_defaults(func=cmd_config_show)

    # info 命令
    p_info = subparsers.add_parser("info", help="显示系统信息与配置")
    add_common_opts(p_info)
    p_info.set_defaults(func=cmd_info)

    # dashboard 命令
    p_dash = subparsers.add_parser("dashboard", help="启动 Web Dashboard 交互式监控")
    p_dash.add_argument("--port", "-p", type=int, default=8080,
                        help="监听端口（默认 8080）")
    p_dash.add_argument("--host", type=str, default="localhost",
                        help="监听地址（默认 localhost）")
    add_common_opts(p_dash)
    p_dash.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI 主入口。

    参数：
    - argv: 命令行参数列表（None 时使用 sys.argv）

    返回：退出码（0=成功，1=错误）
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"运行失败: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
