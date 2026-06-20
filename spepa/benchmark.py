"""SPEPA 性能基准与对比工具。

设计原则（第一性原理）：
- 受控条件：基准测试在固定种子、固定步数下运行，确保可复现
- 标准指标：统一收集运行时间、预测误差、信心度、奖励等关键指标
- 配置对比：相同条件下对比不同配置的性能表现
- 零侵入：基准测试不修改 Agent 内部逻辑

主要功能：
- BenchmarkRunner : 运行单个基准测试
- run_benchmark() : 便捷函数，运行单个基准
- compare_configs() : 对比多个配置的性能
- format_benchmark_result() : 格式化基准结果
- format_comparison() : 格式化对比结果
"""
import time
import statistics
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field

import numpy as np

from spepa.utils.logger import get_logger
from spepa.utils.config import SPEPAConfig, get_default_config, get_preset_config
from spepa.agent import SPEPAAgent

logger = get_logger("spepa.benchmark")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BenchmarkResult:
    """单个基准测试结果。"""
    name: str
    config_name: str
    n_steps: int
    seed: int
    total_time: float = 0.0
    steps_per_second: float = 0.0
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    health_score: float = 0.0
    state_norm: float = 0.0
    online_cycles: int = 0
    offline_cycles: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转为字典。"""
        return {
            "name": self.name,
            "config_name": self.config_name,
            "n_steps": self.n_steps,
            "seed": self.seed,
            "total_time": self.total_time,
            "steps_per_second": self.steps_per_second,
            "metrics": self.metrics,
            "health_score": self.health_score,
            "state_norm": self.state_norm,
            "online_cycles": self.online_cycles,
            "offline_cycles": self.offline_cycles,
            "error": self.error,
        }


# ============================================================
# 基准运行器
# ============================================================

class BenchmarkRunner:
    """性能基准运行器。

    使用方式：
        runner = BenchmarkRunner()
        result = runner.run("my_benchmark", config, n_steps=100)
        print(format_benchmark_result(result))

    多次运行取平均：
        results = runner.run_repeated("my_benchmark", config, n_steps=100, n_runs=3)
        avg = runner.average(results)
    """

    def __init__(self, quiet: bool = True):
        """初始化基准运行器。

        参数：
            quiet : 是否静默模式（不输出 Agent 运行日志）
        """
        self.quiet = quiet

    def run(self,
            name: str,
            config: SPEPAConfig,
            n_steps: int = 100,
            seed: Optional[int] = None) -> BenchmarkResult:
        """运行单个基准测试。

        参数：
            name : 基准测试名称
            config : SPEPA 配置
            n_steps : 运行步数
            seed : 随机种子（None 则使用 config.seed）

        返回：
            BenchmarkResult
        """
        if seed is not None:
            config.seed = seed
        result = BenchmarkResult(
            name=name,
            config_name=getattr(config, '_preset_name', 'custom'),
            n_steps=n_steps,
            seed=config.seed,
        )

        try:
            if self.quiet:
                # 临时降低日志级别
                from spepa.utils.logger import set_log_level
                import logging
                old_level = logging.getLogger("spepa").level
                set_log_level(logging.WARNING)

            # 创建并运行 Agent
            agent = SPEPAAgent(config)
            start_time = time.perf_counter()
            run_result = agent.run(n_steps=n_steps)
            elapsed = time.perf_counter() - start_time

            if self.quiet:
                set_log_level(old_level)

            # 收集结果
            result.total_time = elapsed
            result.steps_per_second = n_steps / elapsed if elapsed > 0 else 0
            result.online_cycles = run_result.get("online_cycles", 0)
            result.offline_cycles = run_result.get("offline_cycles", 0)

            # 收集指标摘要
            metrics_summary = agent.metrics.summary()
            for metric_name, stats in metrics_summary.items():
                if isinstance(stats, dict):
                    result.metrics[metric_name] = {
                        k: float(v) for k, v in stats.items()
                        if isinstance(v, (int, float))
                    }

            # 健康分数
            try:
                result.health_score = float(agent.get_health_score())
            except Exception:
                result.health_score = 0.0

            # 状态范数
            result.state_norm = float(np.linalg.norm(agent.current_state))

        except Exception as e:
            result.error = str(e)
            logger.error("Benchmark '%s' failed: %s", name, e)

        return result

    def run_repeated(self,
                     name: str,
                     config: SPEPAConfig,
                     n_steps: int = 100,
                     n_runs: int = 3,
                     seeds: Optional[List[int]] = None) -> List[BenchmarkResult]:
        """多次运行基准测试（使用不同种子）。

        参数：
            name : 基准测试名称
            config : SPEPA 配置
            n_steps : 每次运行步数
            n_runs : 运行次数
            seeds : 种子列表（None 则自动生成）

        返回：
            BenchmarkResult 列表
        """
        if seeds is None:
            seeds = [config.seed + i for i in range(n_runs)]

        results = []
        for i, seed in enumerate(seeds):
            run_name = f"{name}#{i}"
            result = self.run(run_name, config, n_steps=n_steps, seed=seed)
            results.append(result)
        return results

    @staticmethod
    def average(results: List[BenchmarkResult]) -> Dict[str, Any]:
        """计算多次运行的平均结果。

        参数：
            results : BenchmarkResult 列表

        返回：
            包含平均值的字典
        """
        if not results:
            return {}

        valid = [r for r in results if r.error is None]
        if not valid:
            return {"error": "all runs failed"}

        avg = {
            "name": results[0].name.split('#')[0],
            "n_runs": len(valid),
            "n_steps": valid[0].n_steps,
            "total_time_mean": statistics.mean(r.total_time for r in valid),
            "total_time_std": statistics.stdev(r.total_time for r in valid) if len(valid) > 1 else 0,
            "steps_per_second_mean": statistics.mean(r.steps_per_second for r in valid),
            "health_score_mean": statistics.mean(r.health_score for r in valid),
            "state_norm_mean": statistics.mean(r.state_norm for r in valid),
        }

        # 指标平均
        metric_names = set()
        for r in valid:
            metric_names.update(r.metrics.keys())

        avg["metrics"] = {}
        for metric_name in metric_names:
            values = []
            for r in valid:
                if metric_name in r.metrics and "mean" in r.metrics[metric_name]:
                    values.append(r.metrics[metric_name]["mean"])
            if values:
                avg["metrics"][metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                }

        return avg


# ============================================================
# 便捷函数
# ============================================================

def run_benchmark(name: str,
                  config: SPEPAConfig,
                  n_steps: int = 100,
                  seed: Optional[int] = None,
                  quiet: bool = True) -> BenchmarkResult:
    """便捷函数：运行单个基准测试。

    参数：
        name : 基准测试名称
        config : SPEPA 配置
        n_steps : 运行步数
        seed : 随机种子
        quiet : 静默模式

    返回：
        BenchmarkResult
    """
    runner = BenchmarkRunner(quiet=quiet)
    return runner.run(name, config, n_steps=n_steps, seed=seed)


def compare_configs(configs: Dict[str, SPEPAConfig],
                    n_steps: int = 100,
                    n_runs: int = 1,
                    quiet: bool = True) -> Dict[str, Any]:
    """对比多个配置的性能。

    参数：
        configs : {config_name: SPEPAConfig} 字典
        n_steps : 每次运行步数
        n_runs : 每个配置运行次数（>1 取平均）
        quiet : 静默模式

    返回：
        {
            "results": {config_name: BenchmarkResult 或 List[BenchmarkResult]},
            "summary": {config_name: avg_dict},
            "ranking": [(config_name, score), ...],  # 按综合得分排序
        }
    """
    runner = BenchmarkRunner(quiet=quiet)
    results = {}
    summaries = {}

    for name, config in configs.items():
        if n_runs > 1:
            run_results = runner.run_repeated(name, config, n_steps=n_steps, n_runs=n_runs)
            results[name] = run_results
            summaries[name] = BenchmarkRunner.average(run_results)
        else:
            result = runner.run(name, config, n_steps=n_steps)
            results[name] = result
            summaries[name] = result.to_dict()

    # 计算综合得分并排名
    ranking = []
    for name, summary in summaries.items():
        score = _compute_score(summary)
        ranking.append((name, score))
    ranking.sort(key=lambda x: x[1], reverse=True)

    return {
        "results": results,
        "summary": summaries,
        "ranking": ranking,
    }


def _compute_score(summary: Dict[str, Any]) -> float:
    """计算配置的综合得分（0~1，越高越好）。

    评分维度：
    - 速度（steps_per_second）：越快越好
    - 健康（health_score）：越高越好
    - 预测误差：越低越好
    - 信心度：越高越好
    """
    score = 0.0
    n_dims = 0

    # 速度（归一化到 0~1，假设 1000 steps/s 为满分）
    if "steps_per_second_mean" in summary:
        sps = summary["steps_per_second_mean"]
    elif "steps_per_second" in summary:
        sps = summary["steps_per_second"]
    else:
        sps = 0
    score += min(sps / 1000.0, 1.0)
    n_dims += 1

    # 健康分数
    if "health_score_mean" in summary:
        score += summary["health_score_mean"]
    elif "health_score" in summary:
        score += summary["health_score"]
    n_dims += 1

    # 预测误差（越低越好，归一化）
    metrics = summary.get("metrics", {})
    if "prediction_error" in metrics:
        err = metrics["prediction_error"].get("mean", 1.0)
        score += max(0, 1.0 - err)  # 误差 0 得满分，误差>=1 得 0
        n_dims += 1

    # 信心度（越高越好）
    if "confidence" in metrics:
        conf = metrics["confidence"].get("mean", 0.0)
        score += max(0, min(conf, 1.0))
        n_dims += 1

    return score / n_dims if n_dims > 0 else 0.0


def compare_presets(n_steps: int = 50,
                    presets: Optional[List[str]] = None,
                    quiet: bool = True) -> Dict[str, Any]:
    """对比所有预设配置的性能。

    参数：
        n_steps : 运行步数
        presets : 要对比的预设列表（None=所有预设）
        quiet : 静默模式

    返回：
        compare_configs() 的结果
    """
    if presets is None:
        presets = ["minimal", "balanced", "performance", "conservative"]

    configs = {}
    for name in presets:
        try:
            configs[name] = get_preset_config(name)
        except Exception as e:
            logger.warning("Failed to load preset '%s': %s", name, e)

    return compare_configs(configs, n_steps=n_steps, quiet=quiet)


# ============================================================
# 格式化函数
# ============================================================

def format_benchmark_result(result: BenchmarkResult) -> str:
    """格式化单个基准测试结果。

    参数：
        result : BenchmarkResult

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 50)
    lines.append(f"Benchmark: {result.name}")
    lines.append("=" * 50)
    lines.append(f"  config:       {result.config_name}")
    lines.append(f"  n_steps:      {result.n_steps}")
    lines.append(f"  seed:         {result.seed}")

    if result.error:
        lines.append(f"  ERROR: {result.error}")
        lines.append("=" * 50)
        return "\n".join(lines)

    lines.append(f"  total_time:   {result.total_time:.3f}s")
    lines.append(f"  steps/sec:    {result.steps_per_second:.1f}")
    lines.append(f"  online_cycles:{result.online_cycles}")
    lines.append(f"  offline_cycles:{result.offline_cycles}")
    lines.append(f"  health_score: {result.health_score:.4f}")
    lines.append(f"  state_norm:   {result.state_norm:.4f}")

    if result.metrics:
        lines.append(f"\n  [Metrics]")
        for name in ["prediction_error", "confidence", "total_reward",
                      "exploration_ratio", "cycle_time"]:
            if name in result.metrics:
                m = result.metrics[name]
                lines.append(f"    {name:25s}: mean={m.get('mean', 0):.4f} "
                             f"last={m.get('last', 0):.4f} "
                             f"std={m.get('std', 0):.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)


def format_benchmark_comparison(comp: Dict[str, Any]) -> str:
    """格式化配置对比结果。

    参数：
        comp : compare_configs() 或 compare_presets() 的结果

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Configuration Comparison")
    lines.append("=" * 70)

    ranking = comp.get("ranking", [])
    summary = comp.get("summary", {})

    # 排名表
    lines.append(f"\n{'Rank':<5} {'Config':<20} {'Score':>8} {'Steps/s':>10} "
                 f"{'Health':>8} {'PredErr':>10}")
    lines.append("-" * 70)
    for rank, (name, score) in enumerate(ranking, 1):
        s = summary.get(name, {})
        sps = s.get("steps_per_second_mean", s.get("steps_per_second", 0))
        health = s.get("health_score_mean", s.get("health_score", 0))
        metrics = s.get("metrics", {})
        err = metrics.get("prediction_error", {}).get("mean", 0)
        lines.append(f"{rank:<5} {name:<20} {score:>8.4f} {sps:>8.1f}/s "
                     f"{health:>8.4f} {err:>10.4f}")

    lines.append("-" * 70)
    lines.append("=" * 70)
    return "\n".join(lines)


def format_average(avg: Dict[str, Any]) -> str:
    """格式化多次运行的平均结果。

    参数：
        avg : BenchmarkRunner.average() 的结果

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 50)
    lines.append(f"Average over {avg.get('n_runs', 0)} runs: {avg.get('name', '?')}")
    lines.append("=" * 50)
    lines.append(f"  n_steps:          {avg.get('n_steps', '?')}")
    lines.append(f"  total_time:       {avg.get('total_time_mean', 0):.3f}s "
                 f"(±{avg.get('total_time_std', 0):.3f})")
    lines.append(f"  steps/sec:        {avg.get('steps_per_second_mean', 0):.1f}")
    lines.append(f"  health_score:     {avg.get('health_score_mean', 0):.4f}")
    lines.append(f"  state_norm:       {avg.get('state_norm_mean', 0):.4f}")

    metrics = avg.get("metrics", {})
    if metrics:
        lines.append(f"\n  [Metrics]")
        for name in ["prediction_error", "confidence", "total_reward"]:
            if name in metrics:
                m = metrics[name]
                lines.append(f"    {name:25s}: mean={m.get('mean', 0):.4f} "
                             f"(±{m.get('std', 0):.4f})")

    lines.append("=" * 50)
    return "\n".join(lines)
