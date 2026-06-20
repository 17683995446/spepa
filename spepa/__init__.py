"""SPEPA: Self-driven Perpetual Evolutionary Prediction-Correction Agent.

自驱永续进化预测修正 Agent 系统。

核心架构：
- 层级预测模型（多层预测编码 + 精度加权）
- 变分自由能计算（三种形式）
- 三大驱力引擎（安全/收益/能效 + 内稳态）
- 缺口检测器（信息缺口 = KL 散度，四级分类）
- 目标层次体系（L0-L4 + 冲突仲裁）
- 主动推理决策（期望自由能最小化）
- 元认知监控（信心校准）
- 双系统记忆（快速 + 慢速巩固）
- 双模式优化（微修正 + 结构性重构）
- 安全护栏（刚性边界）
- 自我模型（元认知自我指涉）

双闭环：
- 在线闭环：感知 -> 决策 -> 行动 -> 学习
- 离线闭环：低负载时复盘 -> 巩固 -> 重构

资源约束：CPU<=20%, 内存<=20%, 存储保留 Top30%, 探索比例>=30%
"""
from spepa.utils.config import (
    SPEPAConfig, get_default_config, get_preset_config, list_presets
)
from spepa.utils.logger import (
    get_logger, set_log_level, add_file_handler, remove_handlers,
    get_log_config, quiet_mode, verbose_mode, list_loggers,
)
from spepa.agent import SPEPAAgent
from spepa.events import EventBus, EventType, Event, ProgressReporter
from spepa.report import ReportGenerator, ASCIITextChart
from spepa.dashboard import WebDashboard
from spepa.environments import (
    Environment, SimpleEnvironment, StaticEnvironment,
    FunctionEnvironment, GymEnvironment,
    make_environment, list_environments,
)
from spepa.debug import (
    snapshot, format_snapshot, health_check,
    DebugProfiler, compare_snapshots, format_comparison,
)
from spepa.tutorial import (
    Tutorial, TutorialStep, list_tutorials, get_tutorial,
    run_tutorial, register_tutorial,
)
from spepa.benchmark import (
    BenchmarkRunner, BenchmarkResult,
    run_benchmark, compare_configs, compare_presets,
    format_benchmark_result, format_benchmark_comparison, format_average,
)
from spepa.plugins import (
    Plugin, PluginManager,
    LoggingPlugin, MetricPlugin, StateMonitorPlugin,
    list_builtin_plugins, create_plugin,
)
from spepa.diagnostics import (
    ErrorSuggestion, ErrorContext,
    friendly_error, diagnose, diagnose_config,
    safe_run, format_diagnosis,
)
from spepa.state_manager import (
    StateManager, save_state, load_state, resume_run,
)
from spepa.visualization import (
    ChartConfig, ASCIILineChart,
    generate_chart, generate_dashboard,
    plot_metric, plot_metrics, render_ascii_chart,
)
from spepa.config_advisor import (
    ConfigAdvice, ConfigDiff, ConfigWizard,
    validate_config, recommend_for_scenario, suggest_optimizations,
    diff_configs, list_scenarios,
    format_advice, format_diff,
)
from spepa.batch_scheduler import (
    BatchTask, BatchResult, BatchScheduler,
    run_batch, run_param_sweep, run_preset_comparison,
    format_batch_results, format_batch_summary,
)
from spepa.repl import REPLShell, Command, start_repl
from spepa.docs import (
    DocEntry, DocIndex,
    build_doc_index, search_docs, get_module_docs, get_topic_docs,
    list_topics, list_modules, format_doc, format_search_results, doc_stats,
)
from spepa.recipes import (
    Recipe, RecipeLibrary,
    get_recipe, list_recipes, run_recipe, format_recipe, format_recipe_list,
    add_recipe, list_recipe_tags, list_recipe_difficulties,
)
from spepa.helper import (
    HelpResult, SmartHelper,
    ask, format_help,
)
from spepa.progress import (
    ProgressInfo, ProgressTracker, ProgressBar, AgentProgressTracker,
    track_progress, track_agent_progress, format_progress,
)
from spepa.config_templates import (
    ConfigTemplate, TemplateLibrary,
    save_template, load_template, delete_template,
    list_templates, search_templates,
    export_template, import_template,
    format_template, format_template_list,
)
from spepa.jupyter import (
    is_jupyter, is_ipython_available,
    display_agent, display_config, display_metrics, display_diagnosis,
    SPEPAMagics, register_magics, load_ipython_extension,
    show, quickstart,
)

__version__ = "0.1.0"
__all__ = [
    "SPEPAAgent",
    "SPEPAConfig",
    "get_default_config",
    "get_preset_config",
    "list_presets",
    "get_logger",
    "set_log_level",
    "add_file_handler",
    "remove_handlers",
    "get_log_config",
    "quiet_mode",
    "verbose_mode",
    "list_loggers",
    "EventBus",
    "EventType",
    "Event",
    "ProgressReporter",
    "ReportGenerator",
    "ASCIITextChart",
    "WebDashboard",
    "Environment",
    "SimpleEnvironment",
    "StaticEnvironment",
    "FunctionEnvironment",
    "GymEnvironment",
    "make_environment",
    "list_environments",
    "snapshot",
    "format_snapshot",
    "health_check",
    "DebugProfiler",
    "compare_snapshots",
    "format_comparison",
    "Tutorial",
    "TutorialStep",
    "list_tutorials",
    "get_tutorial",
    "run_tutorial",
    "register_tutorial",
    "BenchmarkRunner",
    "BenchmarkResult",
    "run_benchmark",
    "compare_configs",
    "compare_presets",
    "format_benchmark_result",
    "format_benchmark_comparison",
    "format_average",
    "Plugin",
    "PluginManager",
    "LoggingPlugin",
    "MetricPlugin",
    "StateMonitorPlugin",
    "list_builtin_plugins",
    "create_plugin",
    "ErrorSuggestion",
    "ErrorContext",
    "friendly_error",
    "diagnose",
    "diagnose_config",
    "safe_run",
    "format_diagnosis",
    "StateManager",
    "save_state",
    "load_state",
    "resume_run",
    "ChartConfig",
    "ASCIILineChart",
    "generate_chart",
    "generate_dashboard",
    "plot_metric",
    "plot_metrics",
    "render_ascii_chart",
    "ConfigAdvice",
    "ConfigDiff",
    "ConfigWizard",
    "validate_config",
    "recommend_for_scenario",
    "suggest_optimizations",
    "diff_configs",
    "list_scenarios",
    "format_advice",
    "format_diff",
    "BatchTask",
    "BatchResult",
    "BatchScheduler",
    "run_batch",
    "run_param_sweep",
    "run_preset_comparison",
    "format_batch_results",
    "format_batch_summary",
    "REPLShell",
    "Command",
    "start_repl",
    "DocEntry",
    "DocIndex",
    "build_doc_index",
    "search_docs",
    "get_module_docs",
    "get_topic_docs",
    "list_topics",
    "list_modules",
    "format_doc",
    "format_search_results",
    "doc_stats",
    "Recipe",
    "RecipeLibrary",
    "get_recipe",
    "list_recipes",
    "run_recipe",
    "format_recipe",
    "format_recipe_list",
    "add_recipe",
    "list_recipe_tags",
    "list_recipe_difficulties",
    "HelpResult",
    "SmartHelper",
    "ask",
    "format_help",
    "ProgressInfo",
    "ProgressTracker",
    "ProgressBar",
    "AgentProgressTracker",
    "track_progress",
    "track_agent_progress",
    "format_progress",
    "ConfigTemplate",
    "TemplateLibrary",
    "save_template",
    "load_template",
    "delete_template",
    "list_templates",
    "search_templates",
    "export_template",
    "import_template",
    "format_template",
    "format_template_list",
    "is_jupyter",
    "is_ipython_available",
    "display_agent",
    "display_config",
    "display_metrics",
    "display_diagnosis",
    "SPEPAMagics",
    "register_magics",
    "load_ipython_extension",
    "show",
    "quickstart",
]
