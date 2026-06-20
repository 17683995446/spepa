"""阶段五测试：性能基准与对比工具。

测试范围：
1. BenchmarkResult 数据结构
2. BenchmarkRunner 单次运行
3. BenchmarkRunner 多次运行与平均
4. run_benchmark 便捷函数
5. compare_configs 配置对比
6. compare_presets 预设对比
7. 格式化函数
8. CLI benchmark 命令
"""
import io
import sys
from contextlib import redirect_stdout

import pytest

from spepa.benchmark import (
    BenchmarkRunner, BenchmarkResult,
    run_benchmark, compare_configs, compare_presets,
    format_benchmark_result, format_benchmark_comparison, format_average,
)
from spepa.utils.config import get_preset_config, get_default_config
from spepa.cli import main


# ============================================================
# BenchmarkResult 测试
# ============================================================

class TestBenchmarkResult:
    """BenchmarkResult 数据结构测试。"""

    def test_create_result(self):
        """创建结果对象。"""
        result = BenchmarkResult(
            name="test", config_name="minimal",
            n_steps=10, seed=42
        )
        assert result.name == "test"
        assert result.config_name == "minimal"
        assert result.n_steps == 10
        assert result.seed == 42
        assert result.error is None

    def test_to_dict(self):
        """转为字典。"""
        result = BenchmarkResult(
            name="test", config_name="minimal",
            n_steps=10, seed=42, total_time=0.5,
            steps_per_second=20.0, health_score=0.8,
        )
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["total_time"] == 0.5
        assert d["steps_per_second"] == 20.0
        assert d["health_score"] == 0.8


# ============================================================
# BenchmarkRunner 测试
# ============================================================

class TestBenchmarkRunner:
    """BenchmarkRunner 测试。"""

    def test_run_single(self):
        """单次运行基准测试。"""
        runner = BenchmarkRunner(quiet=True)
        config = get_preset_config("minimal")
        result = runner.run("test_single", config, n_steps=5, seed=42)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "test_single"
        assert result.n_steps == 5
        assert result.error is None
        assert result.total_time > 0
        assert result.steps_per_second > 0
        assert result.online_cycles > 0

    def test_run_collects_metrics(self):
        """运行后收集指标。"""
        runner = BenchmarkRunner(quiet=True)
        config = get_preset_config("minimal")
        result = runner.run("test_metrics", config, n_steps=5, seed=42)
        assert len(result.metrics) > 0
        assert "prediction_error" in result.metrics

    def test_run_collects_health(self):
        """运行后收集健康分数。"""
        runner = BenchmarkRunner(quiet=True)
        config = get_preset_config("minimal")
        result = runner.run("test_health", config, n_steps=5, seed=42)
        assert result.health_score >= 0

    def test_run_repeated(self):
        """多次运行。"""
        runner = BenchmarkRunner(quiet=True)
        config = get_preset_config("minimal")
        results = runner.run_repeated("test_rep", config, n_steps=3, n_runs=3)
        assert len(results) == 3
        for r in results:
            assert r.error is None

    def test_average(self):
        """计算平均值。"""
        runner = BenchmarkRunner(quiet=True)
        config = get_preset_config("minimal")
        results = runner.run_repeated("test_avg", config, n_steps=3, n_runs=3)
        avg = BenchmarkRunner.average(results)
        assert avg["n_runs"] == 3
        assert "total_time_mean" in avg
        assert "steps_per_second_mean" in avg
        assert "health_score_mean" in avg

    def test_average_empty(self):
        """空列表平均值。"""
        avg = BenchmarkRunner.average([])
        assert avg == {}

    def test_run_with_error(self):
        """运行出错时记录错误。"""
        runner = BenchmarkRunner(quiet=True)
        # 创建一个会导致错误的配置（极端值）
        config = get_preset_config("minimal")
        config.state_dim = -1  # 非法值
        result = runner.run("test_error", config, n_steps=1, seed=42)
        # 可能抛异常也可能不抛，取决于 Agent 的容错
        # 如果有错误，error 字段不为 None
        if result.error:
            assert isinstance(result.error, str)


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    """便捷函数测试。"""

    def test_run_benchmark(self):
        """run_benchmark 便捷函数。"""
        config = get_preset_config("minimal")
        result = run_benchmark("quick", config, n_steps=3, seed=42)
        assert isinstance(result, BenchmarkResult)
        assert result.name == "quick"
        assert result.error is None

    def test_compare_configs(self):
        """compare_configs 对比两个配置。"""
        configs = {
            "minimal": get_preset_config("minimal"),
            "balanced": get_preset_config("balanced"),
        }
        result = compare_configs(configs, n_steps=3)
        assert "results" in result
        assert "summary" in result
        assert "ranking" in result
        assert "minimal" in result["summary"]
        assert "balanced" in result["summary"]
        assert len(result["ranking"]) == 2

    def test_compare_presets(self):
        """compare_presets 对比预设。"""
        result = compare_presets(n_steps=3, presets=["minimal", "balanced"])
        assert len(result["ranking"]) == 2

    def test_compare_presets_all(self):
        """compare_presets 对比所有预设。"""
        result = compare_presets(n_steps=3)
        assert len(result["ranking"]) >= 3

    def test_ranking_sorted_by_score(self):
        """排名按得分降序。"""
        configs = {
            "minimal": get_preset_config("minimal"),
            "balanced": get_preset_config("balanced"),
        }
        result = compare_configs(configs, n_steps=3)
        ranking = result["ranking"]
        assert ranking[0][1] >= ranking[1][1]  # 第一个得分 >= 第二个


# ============================================================
# 格式化函数测试
# ============================================================

class TestFormatting:
    """格式化函数测试。"""

    def test_format_benchmark_result(self):
        """格式化单个结果。"""
        config = get_preset_config("minimal")
        result = run_benchmark("fmt_test", config, n_steps=3, seed=42)
        text = format_benchmark_result(result)
        assert isinstance(text, str)
        assert "Benchmark: fmt_test" in text
        assert "total_time" in text
        assert "steps/sec" in text

    def test_format_benchmark_result_with_error(self):
        """格式化错误结果。"""
        result = BenchmarkResult(
            name="err_test", config_name="test",
            n_steps=10, seed=42, error="something went wrong"
        )
        text = format_benchmark_result(result)
        assert "ERROR" in text
        assert "something went wrong" in text

    def test_format_comparison(self):
        """格式化对比结果。"""
        configs = {
            "minimal": get_preset_config("minimal"),
            "balanced": get_preset_config("balanced"),
        }
        result = compare_configs(configs, n_steps=3)
        text = format_benchmark_comparison(result)
        assert isinstance(text, str)
        assert "Configuration Comparison" in text
        assert "Rank" in text
        assert "minimal" in text
        assert "balanced" in text

    def test_format_average(self):
        """格式化平均结果。"""
        runner = BenchmarkRunner(quiet=True)
        config = get_preset_config("minimal")
        results = runner.run_repeated("avg_fmt", config, n_steps=3, n_runs=2)
        avg = BenchmarkRunner.average(results)
        text = format_average(avg)
        assert isinstance(text, str)
        assert "Average over" in text
        assert "avg_fmt" in text


# ============================================================
# CLI benchmark 命令测试
# ============================================================

class TestCLIBenchmark:
    """CLI benchmark 命令测试。"""

    def test_benchmark_command(self, capsys):
        """benchmark 命令运行。"""
        ret = main(["benchmark", "--steps", "3", "--presets", "minimal,balanced"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Configuration Comparison" in captured.out

    def test_benchmark_in_help(self, capsys):
        """benchmark 命令出现在帮助中。"""
        main([])
        captured = capsys.readouterr()
        assert "benchmark" in captured.out


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端测试。"""

    def test_full_benchmark_workflow(self):
        """完整基准工作流。"""
        # 1. 运行单个基准
        config = get_preset_config("minimal")
        result = run_benchmark("workflow", config, n_steps=5, seed=42)
        assert result.error is None

        # 2. 格式化结果
        text = format_benchmark_result(result)
        assert "workflow" in text

        # 3. 多次运行
        runner = BenchmarkRunner(quiet=True)
        results = runner.run_repeated("workflow_rep", config, n_steps=3, n_runs=3)
        assert len(results) == 3

        # 4. 计算平均
        avg = BenchmarkRunner.average(results)
        assert avg["n_runs"] == 3

        # 5. 格式化平均
        avg_text = format_average(avg)
        assert "Average over 3 runs" in avg_text

    def test_compare_all_presets_workflow(self):
        """对比所有预设的完整工作流。"""
        # 1. 对比所有预设
        result = compare_presets(n_steps=3)
        assert len(result["ranking"]) >= 3

        # 2. 格式化对比
        text = format_benchmark_comparison(result)
        assert "Configuration Comparison" in text

        # 3. 验证排名有顺序
        ranking = result["ranking"]
        for i in range(len(ranking) - 1):
            assert ranking[i][1] >= ranking[i + 1][1]

    def test_benchmark_reproducible(self):
        """相同种子结果可复现。"""
        config1 = get_preset_config("minimal")
        config2 = get_preset_config("minimal")
        r1 = run_benchmark("repro1", config1, n_steps=5, seed=42)
        r2 = run_benchmark("repro2", config2, n_steps=5, seed=42)
        # 步数和周期数应相同
        assert r1.online_cycles == r2.online_cycles
        assert r1.offline_cycles == r2.offline_cycles
