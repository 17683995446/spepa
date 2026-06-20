"""SPEPA 错误诊断与友好提示系统。

设计原则（第一性原理）：
- 可操作：每个错误都附带具体的修复建议，而非抽象描述
- 上下文感知：根据错误类型和发生位置提供针对性建议
- 渐进式：简单错误简短提示，复杂错误详细诊断
- 零侵入：通过装饰器和上下文管理器包装，不修改核心逻辑

主要功能：
- friendly_error() : 将异常转为友好错误信息
- diagnose() : 深度诊断 Agent 状态问题
- diagnose_config() : 诊断配置问题
- safe_run() : 安全运行包装器，捕获并解释错误
- ErrorSuggestion : 错误建议数据结构
"""
import os
import sys
import traceback
import functools
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field

import numpy as np

from spepa.utils.logger import get_logger

logger = get_logger("spepa.diagnostics")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ErrorSuggestion:
    """错误建议。"""
    code: str          # 错误代码（如 "CONFIG_INVALID_DIM"）
    category: str      # 类别（config/runtime/environment/usage）
    message: str       # 错误信息
    cause: str         # 可能原因
    fix: str           # 修复建议
    example: str = ""  # 修复示例


# ============================================================
# 错误模式库
# ============================================================

# 已知错误模式：异常类型 + 关键词 → ErrorSuggestion
_ERROR_PATTERNS: List[Dict[str, Any]] = [
    {
        "keywords": ["state_dim", "action_dim", "obs_dim"],
        "category": "config",
        "code": "CONFIG_INVALID_DIM",
        "message": "配置维度参数无效",
        "cause": "state_dim/action_dim/obs_dim 必须为正整数",
        "fix": "检查配置中的维度参数，确保为正整数",
        "example": "config = get_default_config(); config.state_dim = 16",
    },
    {
        "keywords": ["memory", "capacity", "fast_memory"],
        "category": "config",
        "code": "CONFIG_MEMORY_INVALID",
        "message": "记忆容量配置无效",
        "cause": "记忆容量必须为正整数",
        "fix": "设置 fast_memory_capacity 和 slow_memory_capacity 为合理值（如 100-10000）",
        "example": "config.fast_memory_capacity = 200",
    },
    {
        "keywords": ["learning_rate", "lr"],
        "category": "config",
        "code": "CONFIG_LR_INVALID",
        "message": "学习率配置无效",
        "cause": "学习率应在 (0, 1] 范围内",
        "fix": "调整 learning_rate_micro 和 learning_rate_struct 到合理范围",
        "example": "config.learning_rate_micro = 0.01",
    },
    {
        "keywords": ["seed", "random"],
        "category": "config",
        "code": "CONFIG_SEED_INVALID",
        "message": "随机种子配置无效",
        "cause": "seed 应为非负整数",
        "fix": "设置 seed 为非负整数（0 表示使用系统随机）",
        "example": "config.seed = 42",
    },
    {
        "keywords": ["file", "not found", "no such file"],
        "category": "io",
        "code": "FILE_NOT_FOUND",
        "message": "文件未找到",
        "cause": "指定的文件路径不存在",
        "fix": "检查文件路径是否正确，或使用绝对路径",
        "example": "config = SPEPAConfig.load_from_file('/abs/path/to/config.json')",
    },
    {
        "keywords": ["permission", "denied"],
        "category": "io",
        "code": "PERMISSION_DENIED",
        "message": "权限不足",
        "cause": "没有文件读写权限",
        "fix": "检查文件权限或更换输出目录",
        "example": "chmod 644 output.log  # 或更换到有权限的目录",
    },
    {
        "keywords": ["import", "module", "numpy", "yaml"],
        "category": "dependency",
        "code": "MISSING_DEPENDENCY",
        "message": "缺少依赖库",
        "cause": "所需的第三方库未安装",
        "fix": "安装缺失的依赖库",
        "example": "pip install numpy pyyaml",
    },
    {
        "keywords": ["shape", "dimension", "broadcast", "size mismatch"],
        "category": "runtime",
        "code": "SHAPE_MISMATCH",
        "message": "数组形状不匹配",
        "cause": "观测/动作/状态的维度与配置不一致",
        "fix": "检查环境输出的维度是否与 config.state_dim/obs_dim 一致",
        "example": "env = SimpleEnvironment(state_dim=config.state_dim, action_dim=config.action_dim)",
    },
    {
        "keywords": ["nan", "inf", "finite", "overflow"],
        "category": "runtime",
        "code": "NUMERICAL_INSTABILITY",
        "message": "数值不稳定",
        "cause": "计算过程中出现 NaN/Inf，可能是学习率过大或输入值极端",
        "fix": "降低学习率，检查输入数据范围，或使用 conservative 预设",
        "example": "config = get_preset_config('conservative')",
    },
    {
        "keywords": ["timeout", "too slow"],
        "category": "runtime",
        "code": "PERFORMANCE_TIMEOUT",
        "message": "运行超时",
        "cause": "步数过多或配置过于复杂",
        "fix": "减少步数，使用 minimal 预设，或降低模型复杂度",
        "example": "agent.run(n_steps=10)  # 从小步数开始",
    },
    {
        "keywords": ["port", "address", "in use", "bind"],
        "category": "network",
        "code": "PORT_IN_USE",
        "message": "端口被占用",
        "cause": "Dashboard 或服务端口已被其他进程占用",
        "fix": "更换端口号或停止占用端口的进程",
        "example": "spepa dashboard --port 8081  # 换一个端口",
    },
    {
        "keywords": ["preset", "unknown", "invalid"],
        "category": "usage",
        "code": "UNKNOWN_PRESET",
        "message": "未知预设名称",
        "cause": "使用了不存在的预设名",
        "fix": "使用 list_presets() 查看可用预设",
        "example": "from spepa import list_presets; print(list_presets())",
    },
]


def _match_error(error: Exception) -> Optional[ErrorSuggestion]:
    """根据异常信息匹配已知错误模式。"""
    error_str = str(error).lower()
    error_type = type(error).__name__

    for pattern in _ERROR_PATTERNS:
        # 检查关键词是否出现在错误信息中
        for keyword in pattern["keywords"]:
            if keyword.lower() in error_str:
                return ErrorSuggestion(
                    code=pattern["code"],
                    category=pattern["category"],
                    message=pattern["message"],
                    cause=pattern["cause"],
                    fix=pattern["fix"],
                    example=pattern["example"],
                )
    return None


# ============================================================
# 公共 API
# ============================================================

def friendly_error(error: Exception, context: str = "") -> str:
    """将异常转换为友好的错误信息。

    参数：
        error : 异常对象
        context : 错误发生上下文（如 "运行 Agent"）

    返回：
        格式化的多行错误信息字符串
    """
    suggestion = _match_error(error)

    lines = []
    lines.append("=" * 60)
    lines.append("SPEPA 错误报告")
    lines.append("=" * 60)

    if context:
        lines.append(f"\n[上下文] {context}")

    lines.append(f"\n[错误类型] {type(error).__name__}")
    lines.append(f"[错误信息] {error}")

    if suggestion:
        lines.append(f"\n[诊断代码] {suggestion.code}")
        lines.append(f"[类别]     {suggestion.category}")
        lines.append(f"[原因]     {suggestion.cause}")
        lines.append(f"\n[修复建议] {suggestion.fix}")
        if suggestion.example:
            lines.append(f"[示例]     {suggestion.example}")
    else:
        lines.append("\n[修复建议] 未识别的错误类型，请查看以下堆栈跟踪")
        lines.append(f"\n[堆栈跟踪]")
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        # 只显示最后 5 行
        lines.extend(tb_lines[-5:])

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def diagnose(agent) -> Dict[str, Any]:
    """深度诊断 Agent 状态问题。

    参数：
        agent : SPEPAAgent 实例

    返回：
        {
            "healthy": bool,
            "issues": List[ErrorSuggestion],
            "warnings": List[str],
            "info": Dict[str, Any],
        }
    """
    issues: List[ErrorSuggestion] = []
    warnings: List[str] = []
    info: Dict[str, Any] = {}

    # 1. 检查状态向量
    state = agent.current_state
    if not np.all(np.isfinite(state)):
        issues.append(ErrorSuggestion(
            code="STATE_NON_FINITE",
            category="runtime",
            message="Agent 状态包含非有限值（NaN/Inf）",
            cause="环境输出或计算过程中产生 NaN/Inf",
            fix="重置 Agent 状态，检查环境输出，或降低学习率",
            example="agent.current_state = np.ones(config.state_dim) * 0.5",
        ))
    if np.allclose(state, 0.0):
        warnings.append("状态全为零，可能未正常初始化")
    state_norm = float(np.linalg.norm(state))
    info["state_norm"] = state_norm
    if state_norm > 10.0:
        warnings.append(f"状态范数过大 ({state_norm:.2f})，可能导致数值不稳定")

    # 2. 检查步数
    info["step_count"] = agent.step_count
    if agent.step_count == 0:
        warnings.append("Agent 尚未运行任何步数")

    # 3. 检查指标趋势
    try:
        summary = agent.metrics.summary()
        info["metric_count"] = len(summary)

        if "prediction_error" in summary:
            err = summary["prediction_error"]
            if isinstance(err, dict):
                last_err = err.get("last", 0)
                mean_err = err.get("mean", 0)
                if last_err > mean_err * 3 and mean_err > 0:
                    issues.append(ErrorSuggestion(
                        code="ERROR_SPIKE",
                        category="runtime",
                        message="预测误差突增",
                        cause=f"最近误差 ({last_err:.4f}) 远超均值 ({mean_err:.4f})",
                        fix="检查环境是否发生剧变，或降低学习率",
                        example="config.learning_rate_micro *= 0.5",
                    ))
                info["prediction_error_last"] = last_err

        if "confidence" in summary:
            conf = summary["confidence"]
            if isinstance(conf, dict):
                last_conf = conf.get("last", 1.0)
                if last_conf < 0.1:
                    issues.append(ErrorSuggestion(
                        code="LOW_CONFIDENCE",
                        category="runtime",
                        message="信心度过低",
                        cause=f"当前信心度 {last_conf:.4f} 低于 0.1",
                        fix="增加训练步数或调整学习率",
                        example="agent.run(n_steps=50)",
                    ))
                info["confidence_last"] = last_conf
    except Exception as e:
        warnings.append(f"无法获取指标摘要: {e}")

    # 4. 检查资源
    try:
        rm = agent.resource_monitor
        info["cpu_usage"] = rm.cpu_usage
        info["memory_usage"] = rm.memory_usage
        if rm.cpu_usage > 80:
            issues.append(ErrorSuggestion(
                code="HIGH_CPU",
                category="runtime",
                message="CPU 使用率过高",
                cause=f"当前 CPU 使用率 {rm.cpu_usage:.1f}%",
                fix="减少步数频率，使用 conservative 预设，或降低模型复杂度",
                example="config = get_preset_config('conservative')",
            ))
        if rm.memory_usage > 80:
            issues.append(ErrorSuggestion(
                code="HIGH_MEMORY",
                category="runtime",
                message="内存使用率过高",
                cause=f"当前内存使用率 {rm.memory_usage:.1f}%",
                fix="减少记忆容量，触发记忆巩固，或使用 minimal 预设",
                example="config.fast_memory_capacity = 50",
            ))
    except Exception as e:
        warnings.append(f"无法获取资源信息: {e}")

    # 5. 检查降级级别
    try:
        deg_level = agent.degradation_matrix.current_level.name
        info["degradation_level"] = deg_level
        if "CRITICAL" in deg_level:
            issues.append(ErrorSuggestion(
                code="DEGRADED_CRITICAL",
                category="runtime",
                message="系统处于临界降级状态",
                cause=f"降级级别: {deg_level}",
                fix="减少负载，等待资源恢复，或重启 Agent",
                example="agent = SPEPAAgent(get_preset_config('minimal'))",
            ))
    except Exception:
        pass

    # 6. 检查健康分数
    try:
        health = agent.get_health_score()
        info["health_score"] = float(health) if health is not None else 0.0
        if health is not None and health < 0.3:
            issues.append(ErrorSuggestion(
                code="LOW_HEALTH",
                category="runtime",
                message="Agent 健康分数过低",
                cause=f"健康分数 {health:.4f} 低于 0.3",
                fix="运行自修复: agent.self_repair_step()",
                example="agent.self_repair_step()",
            ))
    except Exception:
        pass

    return {
        "healthy": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "info": info,
    }


def diagnose_config(config) -> Dict[str, Any]:
    """诊断配置问题。

    参数：
        config : SPEPAConfig 实例

    返回：
        {
            "valid": bool,
            "issues": List[ErrorSuggestion],
            "warnings": List[str],
        }
    """
    issues: List[ErrorSuggestion] = []
    warnings: List[str] = []

    # 维度检查
    if config.state_dim <= 0:
        issues.append(ErrorSuggestion(
            code="CONFIG_INVALID_DIM",
            category="config",
            message="state_dim 必须为正整数",
            cause=f"当前 state_dim = {config.state_dim}",
            fix="设置 state_dim 为正整数（如 8/16/32）",
            example="config.state_dim = 16",
        ))
    if config.action_dim <= 0:
        issues.append(ErrorSuggestion(
            code="CONFIG_INVALID_DIM",
            category="config",
            message="action_dim 必须为正整数",
            cause=f"当前 action_dim = {config.action_dim}",
            fix="设置 action_dim 为正整数（如 4/8/16）",
            example="config.action_dim = 8",
        ))
    if config.obs_dim <= 0:
        issues.append(ErrorSuggestion(
            code="CONFIG_INVALID_DIM",
            category="config",
            message="obs_dim 必须为正整数",
            cause=f"当前 obs_dim = {config.obs_dim}",
            fix="设置 obs_dim 为正整数",
            example="config.obs_dim = 16",
        ))

    # 学习率检查
    if config.learning_rate_micro <= 0 or config.learning_rate_micro > 1:
        warnings.append(f"learning_rate_micro={config.learning_rate_micro} 超出推荐范围 (0, 1]")
    if config.learning_rate_struct <= 0 or config.learning_rate_struct > 1:
        warnings.append(f"learning_rate_struct={config.learning_rate_struct} 超出推荐范围 (0, 1]")

    # 记忆容量检查
    if config.fast_memory_capacity <= 0:
        issues.append(ErrorSuggestion(
            code="CONFIG_MEMORY_INVALID",
            category="config",
            message="fast_memory_capacity 必须为正整数",
            cause=f"当前值 = {config.fast_memory_capacity}",
            fix="设置 fast_memory_capacity 为正整数（推荐 100-1000）",
            example="config.fast_memory_capacity = 200",
        ))
    if config.slow_memory_capacity <= 0:
        issues.append(ErrorSuggestion(
            code="CONFIG_MEMORY_INVALID",
            category="config",
            message="slow_memory_capacity 必须为正整数",
            cause=f"当前值 = {config.slow_memory_capacity}",
            fix="设置 slow_memory_capacity 为正整数（推荐 500-5000）",
            example="config.slow_memory_capacity = 1000",
        ))

    # 种子检查
    if config.seed < 0:
        issues.append(ErrorSuggestion(
            code="CONFIG_SEED_INVALID",
            category="config",
            message="seed 必须为非负整数",
            cause=f"当前 seed = {config.seed}",
            fix="设置 seed 为非负整数",
            example="config.seed = 42",
        ))

    # 步数检查
    if config.max_steps <= 0:
        issues.append(ErrorSuggestion(
            code="CONFIG_INVALID_STEPS",
            category="config",
            message="max_steps 必须为正整数",
            cause=f"当前 max_steps = {config.max_steps}",
            fix="设置 max_steps 为正整数",
            example="config.max_steps = 1000",
        ))

    # 安全边界检查
    if config.safety_hard_boundary <= 0:
        warnings.append(f"safety_hard_boundary={config.safety_hard_boundary} 非正，安全约束可能无效")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }


def safe_run(func: Callable) -> Callable:
    """装饰器：安全运行函数，捕获异常并输出友好错误。

    参数：
        func : 要包装的函数

    返回：
        包装后的函数

    示例：
        @safe_run
        def my_operation(agent):
            agent.run(n_steps=100)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 尝试推断上下文
            context = func.__name__
            error_msg = friendly_error(e, context=context)
            print(error_msg, file=sys.stderr)
            return None
    return wrapper


class ErrorContext:
    """错误上下文管理器：捕获代码块中的异常并输出友好信息。

    使用方式：
        with ErrorContext("加载配置"):
            config = SPEPAConfig.load_from_file(filepath)
    """

    def __init__(self, context: str = "", reraise: bool = False):
        """初始化错误上下文。

        参数：
            context : 上下文描述
            reraise : 是否重新抛出异常（True=输出友好信息后仍抛出）
        """
        self.context = context
        self.reraise = reraise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            error_msg = friendly_error(exc_val, context=self.context)
            print(error_msg, file=sys.stderr)
            if not self.reraise:
                return True  # 抑制异常
        return False


def format_diagnosis(diagnosis: Dict[str, Any]) -> str:
    """格式化诊断结果为可读文本。

    参数：
        diagnosis : diagnose() 或 diagnose_config() 的返回值

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SPEPA 诊断报告")
    lines.append("=" * 60)

    if "healthy" in diagnosis:
        status = "健康" if diagnosis["healthy"] else "发现问题"
        lines.append(f"\n[状态] {status}")
    elif "valid" in diagnosis:
        status = "有效" if diagnosis["valid"] else "无效"
        lines.append(f"\n[配置状态] {status}")

    # 问题
    issues = diagnosis.get("issues", [])
    if issues:
        lines.append(f"\n[问题] ({len(issues)} 个)")
        for i, issue in enumerate(issues, 1):
            lines.append(f"\n  {i}. [{issue.code}] {issue.message}")
            lines.append(f"     原因: {issue.cause}")
            lines.append(f"     修复: {issue.fix}")
            if issue.example:
                lines.append(f"     示例: {issue.example}")

    # 警告
    warnings = diagnosis.get("warnings", [])
    if warnings:
        lines.append(f"\n[警告] ({len(warnings)} 个)")
        for w in warnings:
            lines.append(f"  - {w}")

    # 信息
    info = diagnosis.get("info", {})
    if info:
        lines.append(f"\n[系统信息]")
        for k, v in info.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

    if not issues and not warnings:
        lines.append("\n[结果] 未发现问题")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
