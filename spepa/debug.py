"""SPEPA 调试工具模块。

设计原则（第一性原理）：
- 一键快照：快速捕获 Agent 当前状态，便于诊断
- 可读格式：快照可格式化为人类可读文本
- 性能分析：记录各阶段耗时，定位瓶颈
- 零侵入：调试工具不修改 Agent 运行时状态

主要功能：
- snapshot(agent) : 捕获 Agent 完整状态快照
- format_snapshot(snapshot) : 格式化快照为可读文本
- health_check(agent) : 健康检查，返回诊断信息
- DebugProfiler : 性能分析器，记录各阶段耗时
- compare_snapshots(s1, s2) : 比较两个快照的差异
"""
import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

import numpy as np

from spepa.utils.logger import get_logger

logger = get_logger("spepa.debug")


# ============================================================
# 状态快照
# ============================================================

def snapshot(agent) -> Dict[str, Any]:
    """捕获 Agent 当前状态的完整快照。

    参数：
        agent : SPEPAAgent 实例

    返回：
        包含 Agent 所有关键状态的字典：
        - runtime : 运行时状态（步数、周期数等）
        - state : 当前状态向量统计
        - metrics : 指标摘要
        - health : 健康分数与告警
        - resources : 资源使用情况
        - degradation : 降级级别
        - memory : 记忆系统摘要
        - goals : 目标层次摘要
        - model : 预测模型摘要
    """
    snap = {
        "timestamp": time.time(),
        "runtime": {
            "step_count": agent.step_count,
            "online_cycles": agent.online_cycles,
            "offline_cycles": agent.offline_cycles,
            "running": agent.running,
        },
    }

    # 状态向量统计
    state = agent.current_state
    snap["state"] = {
        "dim": state.shape[0],
        "mean": float(np.mean(state)),
        "std": float(np.std(state)),
        "min": float(np.min(state)),
        "max": float(np.max(state)),
        "norm": float(np.linalg.norm(state)),
        "finite": bool(np.all(np.isfinite(state))),
    }

    # 指标摘要
    try:
        snap["metrics"] = agent.metrics.summary()
    except Exception as e:
        snap["metrics"] = {"error": str(e)}

    # 健康状态
    try:
        health_score = agent.get_health_score()
        alerts = agent.self_repair_system.monitor.get_alerts()
        trends = agent.self_repair_system.monitor.get_trends()
        snap["health"] = {
            "score": float(health_score) if health_score is not None else None,
            "n_alerts": len(alerts) if alerts else 0,
            "alerts": [
                {"type": a.alert_type, "severity": a.severity, "message": a.message}
                for a in (alerts or [])
            ],
            "trends": trends or {},
        }
    except Exception as e:
        snap["health"] = {"error": str(e)}

    # 资源使用
    try:
        rm = agent.resource_monitor
        snap["resources"] = {
            "cpu_usage": float(rm.cpu_usage),
            "memory_usage": float(rm.memory_usage),
            "within_limits": bool(rm.within_limits()),
            "should_throttle": bool(rm.should_throttle()),
        }
    except Exception as e:
        snap["resources"] = {"error": str(e)}

    # 降级级别
    try:
        deg = agent.degradation_matrix
        snap["degradation"] = {
            "level": deg.current_level.name,
            "params": {
                "n_policies": deg.get_current_params().n_policies,
                "learning_enabled": deg.get_current_params().learning_enabled,
                "offline_enabled": deg.get_current_params().offline_enabled,
            },
        }
    except Exception as e:
        snap["degradation"] = {"error": str(e)}

    # 记忆系统
    try:
        mem = agent.memory_system
        snap["memory"] = {
            "fast_size": len(mem.fast),
            "slow_size": len(mem.slow),
            "step_count": mem.step_count,
        }
    except Exception as e:
        snap["memory"] = {"error": str(e)}

    # 目标层次
    try:
        goals = agent.goal_hierarchy
        active = goals.active_goals()
        snap["goals"] = {
            "n_active": len(active),
            "active_goals": [
                {"level": g.level, "name": g.name, "progress": g.progress}
                for g in active
            ],
        }
    except Exception as e:
        snap["goals"] = {"error": str(e)}

    # 预测模型
    try:
        model = agent.predictive_model
        snap["model"] = {
            "n_layers": len(model.layers),
            "mu_norms": [float(np.linalg.norm(l.mu)) for l in model.layers],
            "var_means": [float(np.mean(l.var)) for l in model.layers],
        }
    except Exception as e:
        snap["model"] = {"error": str(e)}

    return snap


def format_snapshot(snap: Dict[str, Any]) -> str:
    """将快照格式化为人类可读的文本。

    参数：
        snap : snapshot() 返回的字典

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SPEPA Agent Snapshot")
    lines.append("=" * 60)

    # 运行时
    rt = snap.get("runtime", {})
    lines.append(f"\n[Runtime]")
    lines.append(f"  step_count:    {rt.get('step_count', '?')}")
    lines.append(f"  online_cycles: {rt.get('online_cycles', '?')}")
    lines.append(f"  offline_cycles:{rt.get('offline_cycles', '?')}")
    lines.append(f"  running:       {rt.get('running', '?')}")

    # 状态向量
    st = snap.get("state", {})
    lines.append(f"\n[State Vector]")
    lines.append(f"  dim:   {st.get('dim', '?')}")
    lines.append(f"  mean:  {st.get('mean', '?'):.4f}" if isinstance(st.get('mean'), (int, float)) else f"  mean:  {st.get('mean', '?')}")
    lines.append(f"  std:   {st.get('std', '?'):.4f}" if isinstance(st.get('std'), (int, float)) else f"  std:   {st.get('std', '?')}")
    lines.append(f"  range: [{st.get('min', '?')}, {st.get('max', '?')}]")
    lines.append(f"  norm:  {st.get('norm', '?'):.4f}" if isinstance(st.get('norm'), (int, float)) else f"  norm:  {st.get('norm', '?')}")
    lines.append(f"  finite: {st.get('finite', '?')}")

    # 健康
    hl = snap.get("health", {})
    lines.append(f"\n[Health]")
    score = hl.get("score")
    lines.append(f"  score:    {score:.4f}" if isinstance(score, (int, float)) else f"  score:    {score}")
    lines.append(f"  alerts:   {hl.get('n_alerts', 0)}")
    for a in hl.get("alerts", [])[:3]:  # 最多显示3条
        lines.append(f"    - [{a.get('type', '?')}] {a.get('message', '')}")
    trends = hl.get("trends", {})
    if trends:
        lines.append(f"  trends:")
        for metric, trend in trends.items():
            lines.append(f"    {metric}: {trend}")

    # 资源
    res = snap.get("resources", {})
    lines.append(f"\n[Resources]")
    lines.append(f"  cpu:    {res.get('cpu_usage', '?'):.1f}%" if isinstance(res.get('cpu_usage'), (int, float)) else f"  cpu:    {res.get('cpu_usage', '?')}")
    lines.append(f"  memory: {res.get('memory_usage', '?'):.1f}%" if isinstance(res.get('memory_usage'), (int, float)) else f"  memory: {res.get('memory_usage', '?')}")
    lines.append(f"  within_limits:  {res.get('within_limits', '?')}")
    lines.append(f"  should_throttle:{res.get('should_throttle', '?')}")

    # 降级
    deg = snap.get("degradation", {})
    lines.append(f"\n[Degradation]")
    lines.append(f"  level: {deg.get('level', '?')}")
    params = deg.get("params", {})
    lines.append(f"  n_policies:      {params.get('n_policies', '?')}")
    lines.append(f"  learning_enabled:{params.get('learning_enabled', '?')}")
    lines.append(f"  offline_enabled: {params.get('offline_enabled', '?')}")

    # 记忆
    mem = snap.get("memory", {})
    lines.append(f"\n[Memory]")
    lines.append(f"  fast: {mem.get('fast_size', '?')}")
    lines.append(f"  slow: {mem.get('slow_size', '?')}")
    lines.append(f"  steps:{mem.get('step_count', '?')}")

    # 目标
    goals = snap.get("goals", {})
    lines.append(f"\n[Goals]")
    lines.append(f"  active: {goals.get('n_active', '?')}")
    for g in goals.get("active_goals", [])[:5]:
        lines.append(f"    L{g.get('level', '?')} {g.get('name', '?')} (progress={g.get('progress', '?')})")

    # 模型
    model = snap.get("model", {})
    lines.append(f"\n[Predictive Model]")
    lines.append(f"  layers: {model.get('n_layers', '?')}")
    mu_norms = model.get("mu_norms", [])
    if mu_norms:
        lines.append(f"  mu_norms: [{', '.join(f'{x:.3f}' for x in mu_norms)}]")
    var_means = model.get("var_means", [])
    if var_means:
        lines.append(f"  var_means:[{', '.join(f'{x:.3f}' for x in var_means)}]")

    # 指标摘要（精简）
    metrics = snap.get("metrics", {})
    if metrics and "error" not in metrics:
        lines.append(f"\n[Metrics Summary]")
        for name in ["prediction_error", "confidence", "total_reward",
                      "exploration_ratio", "cycle_time"]:
            if name in metrics:
                m = metrics[name]
                if isinstance(m, dict):
                    lines.append(f"  {name:25s}: mean={m.get('mean', 0):.4f} last={m.get('last', 0):.4f}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ============================================================
# 健康检查
# ============================================================

def health_check(agent) -> Dict[str, Any]:
    """对 Agent 执行健康检查，返回诊断信息。

    参数：
        agent : SPEPAAgent 实例

    返回：
        {
            "healthy": bool,           # 是否健康
            "health_score": float,     # 健康分数 0~1
            "issues": List[str],       # 发现的问题
            "recommendations": List[str],  # 建议
        }
    """
    issues = []
    recommendations = []

    # 1. 健康分数
    try:
        score = agent.get_health_score()
        score = float(score) if score is not None else 0.5
    except Exception:
        score = 0.5
        issues.append("无法获取健康分数")
        recommendations.append("检查 self_repair_system 是否正常初始化")

    # 2. 状态向量检查
    state = agent.current_state
    if not np.all(np.isfinite(state)):
        issues.append("当前状态包含非有限值（NaN/Inf）")
        recommendations.append("重置 Agent 状态或检查环境输出")
    if np.allclose(state, 0.0):
        issues.append("状态全为零，可能未正常初始化或环境异常")
    if np.allclose(state, state[0]):
        issues.append("状态所有维度相同，缺乏多样性")

    # 3. 告警检查
    try:
        alerts = agent.self_repair_system.monitor.get_alerts()
        if alerts:
            for a in alerts:
                issues.append(f"告警: [{a.alert_type}] {a.message}")
            recommendations.append("运行 agent.self_repair_step() 触发自修复")
    except Exception:
        pass

    # 4. 资源检查
    try:
        rm = agent.resource_monitor
        if rm.cpu_usage > 80:
            issues.append(f"CPU 使用率过高: {rm.cpu_usage:.1f}%")
            recommendations.append("降低运行频率或使用 conservative 预设")
        if rm.memory_usage > 80:
            issues.append(f"内存使用率过高: {rm.memory_usage:.1f}%")
            recommendations.append("减少记忆容量或触发巩固")
    except Exception:
        pass

    # 5. 降级检查
    try:
        deg_level = agent.degradation_matrix.current_level.name
        if "CRITICAL" in deg_level or "SEVERE" in deg_level:
            issues.append(f"系统处于 {deg_level} 降级级别")
            recommendations.append("减少负载或等待资源恢复")
    except Exception:
        pass

    # 6. 指标趋势检查
    try:
        metrics_summary = agent.metrics.summary()
        if "prediction_error" in metrics_summary:
            err = metrics_summary["prediction_error"]
            if isinstance(err, dict) and err.get("last", 0) > err.get("mean", 0) * 2:
                issues.append("预测误差突增（最近值远超均值）")
                recommendations.append("检查环境是否发生剧变")
        if "confidence" in metrics_summary:
            conf = metrics_summary["confidence"]
            if isinstance(conf, dict) and conf.get("last", 1) < 0.2:
                issues.append("信心度过低")
                recommendations.append("增加学习步数或调整学习率")
    except Exception:
        pass

    healthy = len(issues) == 0 and score >= 0.5
    return {
        "healthy": healthy,
        "health_score": score,
        "issues": issues,
        "recommendations": recommendations,
    }


# ============================================================
# 性能分析器
# ============================================================

class DebugProfiler:
    """性能分析器：记录各阶段耗时。

    使用方式：
        profiler = DebugProfiler()
        with profiler.section("perception"):
            # 执行感知
            ...
        with profiler.section("decision"):
            # 执行决策
            ...
        print(profiler.report())

    或手动计时：
        profiler.start("step")
        # ...
        profiler.end("step")
    """

    def __init__(self):
        self._sections: Dict[str, List[float]] = {}
        self._active: Dict[str, float] = {}

    @contextmanager
    def section(self, name: str):
        """上下文管理器：自动计时一个代码段。

        参数：
            name : 段名称
        """
        self.start(name)
        try:
            yield
        finally:
            self.end(name)

    def start(self, name: str) -> None:
        """开始计时。

        参数：
            name : 段名称
        """
        self._active[name] = time.perf_counter()

    def end(self, name: str) -> float:
        """结束计时并返回耗时。

        参数：
            name : 段名称

        返回：
            本次耗时（秒）
        """
        if name not in self._active:
            return 0.0
        elapsed = time.perf_counter() - self._active.pop(name)
        if name not in self._sections:
            self._sections[name] = []
        self._sections[name].append(elapsed)
        return elapsed

    def reset(self) -> None:
        """重置所有记录。"""
        self._sections.clear()
        self._active.clear()

    def summary(self) -> Dict[str, Dict[str, float]]:
        """返回各段的统计摘要。

        返回：
            {section_name: {count, total, mean, min, max}}
        """
        result = {}
        for name, times in self._sections.items():
            if not times:
                continue
            result[name] = {
                "count": len(times),
                "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }
        return result

    def report(self) -> str:
        """生成可读的性能报告。

        返回：
            格式化的多行字符串
        """
        lines = []
        lines.append("=" * 50)
        lines.append("Performance Profile")
        lines.append("=" * 50)
        lines.append(f"{'Section':<25} {'Count':>6} {'Total':>10} {'Mean':>10}")
        lines.append("-" * 50)
        total_all = 0.0
        for name, stats in sorted(self.summary().items()):
            lines.append(
                f"{name:<25} {stats['count']:>6} "
                f"{stats['total']*1000:>8.1f}ms {stats['mean']*1000:>8.1f}ms"
            )
            total_all += stats["total"]
        lines.append("-" * 50)
        lines.append(f"{'TOTAL':<25} {'':>6} {total_all*1000:>8.1f}ms")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================
# 快照比较
# ============================================================

def compare_snapshots(s1: Dict[str, Any], s2: Dict[str, Any]) -> Dict[str, Any]:
    """比较两个快照的差异。

    参数：
        s1 : 前一个快照
        s2 : 后一个快照

    返回：
        {
            "time_delta": float,       # 时间差（秒）
            "step_delta": int,         # 步数差
            "state_delta": float,      # 状态向量范数差
            "health_delta": float,     # 健康分数变化
            "metrics_delta": dict,     # 关键指标变化
            "new_alerts": list,        # 新增告警
        }
    """
    # 时间差
    t1 = s1.get("timestamp", 0)
    t2 = s2.get("timestamp", 0)
    time_delta = t2 - t1 if t1 and t2 else 0.0

    # 步数差
    step1 = s1.get("runtime", {}).get("step_count", 0)
    step2 = s2.get("runtime", {}).get("step_count", 0)
    step_delta = step2 - step1

    # 状态变化（用范数差近似）
    norm1 = s1.get("state", {}).get("norm", 0)
    norm2 = s2.get("state", {}).get("norm", 0)
    state_delta = abs(norm2 - norm1)

    # 健康分数变化
    h1 = s1.get("health", {}).get("score", 0) or 0
    h2 = s2.get("health", {}).get("score", 0) or 0
    health_delta = h2 - h1

    # 关键指标变化
    metrics_delta = {}
    m1 = s1.get("metrics", {})
    m2 = s2.get("metrics", {})
    for key in ["prediction_error", "confidence", "total_reward"]:
        if key in m1 and key in m2:
            v1 = m1[key].get("last", 0) if isinstance(m1[key], dict) else 0
            v2 = m2[key].get("last", 0) if isinstance(m2[key], dict) else 0
            metrics_delta[key] = v2 - v1

    # 新增告警
    old_alerts = {(a.get("type"), a.get("message"))
                  for a in s1.get("health", {}).get("alerts", [])}
    new_alerts = []
    for a in s2.get("health", {}).get("alerts", []):
        key = (a.get("type"), a.get("message"))
        if key not in old_alerts:
            new_alerts.append(a)

    return {
        "time_delta": time_delta,
        "step_delta": step_delta,
        "state_delta": state_delta,
        "health_delta": health_delta,
        "metrics_delta": metrics_delta,
        "new_alerts": new_alerts,
    }


def format_comparison(comp: Dict[str, Any]) -> str:
    """格式化快照比较结果为可读文本。

    参数：
        comp : compare_snapshots() 返回的字典

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Snapshot Comparison")
    lines.append("=" * 50)
    lines.append(f"  time_delta:   {comp['time_delta']:.3f}s")
    lines.append(f"  step_delta:   {comp['step_delta']}")
    lines.append(f"  state_delta:  {comp['state_delta']:.4f}")
    lines.append(f"  health_delta: {comp['health_delta']:+.4f}")
    if comp["metrics_delta"]:
        lines.append(f"\n  [Metrics Delta]")
        for k, v in comp["metrics_delta"].items():
            lines.append(f"    {k}: {v:+.4f}")
    if comp["new_alerts"]:
        lines.append(f"\n  [New Alerts]")
        for a in comp["new_alerts"]:
            lines.append(f"    - [{a.get('type', '?')}] {a.get('message', '')}")
    lines.append("=" * 50)
    return "\n".join(lines)
