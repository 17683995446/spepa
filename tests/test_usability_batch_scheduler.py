"""SPEPA 批处理任务调度器测试。

测试范围：
- BatchTask / BatchResult 数据结构
- BatchScheduler（添加任务、拓扑排序、依赖检查、重试、进度、汇总）
- run_batch 便捷函数
- run_param_sweep 参数扫描
- run_preset_comparison 预设对比
- format_batch_results / format_batch_summary
- 端到端
"""
import pytest
from spepa.batch_scheduler import (
    BatchTask, BatchResult, BatchScheduler,
    run_batch, run_param_sweep, run_preset_comparison,
    format_batch_results, format_batch_summary,
)
from spepa.utils.config import SPEPAConfig, get_preset_config


# ============================================================
# BatchTask 数据结构
# ============================================================

class TestBatchTask:
    """BatchTask 数据结构测试。"""

    def test_create_task(self):
        config = get_preset_config("minimal")
        task = BatchTask(name="t1", config=config, n_steps=10)
        assert task.name == "t1"
        assert task.config is config
        assert task.n_steps == 10
        assert task.depends_on == []
        assert task.retries == 0
        assert task.tags == []

    def test_task_with_dependencies(self):
        config = get_preset_config("minimal")
        task = BatchTask(name="t2", config=config, n_steps=10, depends_on=["t1"])
        assert task.depends_on == ["t1"]

    def test_task_with_retries(self):
        config = get_preset_config("minimal")
        task = BatchTask(name="t3", config=config, n_steps=10, retries=3)
        assert task.retries == 3

    def test_task_with_tags(self):
        config = get_preset_config("minimal")
        task = BatchTask(name="t4", config=config, n_steps=10, tags=["exp", "lr"])
        assert task.tags == ["exp", "lr"]

    def test_task_default_n_steps(self):
        config = get_preset_config("minimal")
        task = BatchTask(name="t5", config=config)
        assert task.n_steps == 100


# ============================================================
# BatchResult 数据结构
# ============================================================

class TestBatchResult:
    """BatchResult 数据结构测试。"""

    def test_create_success_result(self):
        result = BatchResult(
            task_name="t1",
            success=True,
            metrics={"prediction_error": {"last": 0.1}},
            duration=1.5,
            attempts=1,
            steps_executed=10,
        )
        assert result.task_name == "t1"
        assert result.success is True
        assert result.error is None
        assert result.duration == 1.5
        assert result.attempts == 1

    def test_create_failure_result(self):
        result = BatchResult(
            task_name="t2",
            success=False,
            error="some error",
            attempts=3,
        )
        assert result.success is False
        assert result.error == "some error"
        assert result.attempts == 3

    def test_default_values(self):
        result = BatchResult(task_name="t3", success=True)
        assert result.metrics == {}
        assert result.duration == 0.0
        assert result.error is None
        assert result.attempts == 0
        assert result.steps_executed == 0


# ============================================================
# BatchScheduler - 任务管理
# ============================================================

class TestSchedulerTaskManagement:
    """调度器任务管理测试。"""

    def test_add_task(self):
        scheduler = BatchScheduler()
        task = BatchTask(name="t1", config=get_preset_config("minimal"), n_steps=5)
        scheduler.add_task(task)
        assert "t1" in scheduler.tasks

    def test_add_task_returns_self(self):
        scheduler = BatchScheduler()
        task = BatchTask(name="t1", config=get_preset_config("minimal"), n_steps=5)
        result = scheduler.add_task(task)
        assert result is scheduler

    def test_add_duplicate_task_raises(self):
        scheduler = BatchScheduler()
        task = BatchTask(name="t1", config=get_preset_config("minimal"), n_steps=5)
        scheduler.add_task(task)
        with pytest.raises(ValueError, match="已存在"):
            scheduler.add_task(task)

    def test_add_tasks(self):
        scheduler = BatchScheduler()
        tasks = [
            BatchTask(name=f"t{i}", config=get_preset_config("minimal"), n_steps=5)
            for i in range(3)
        ]
        scheduler.add_tasks(tasks)
        assert len(scheduler.tasks) == 3

    def test_clear(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="t1", config=get_preset_config("minimal"), n_steps=5))
        scheduler.clear()
        assert len(scheduler.tasks) == 0
        assert len(scheduler.results) == 0


# ============================================================
# BatchScheduler - 拓扑排序
# ============================================================

class TestTopologicalSort:
    """拓扑排序测试。"""

    def test_no_dependencies(self):
        scheduler = BatchScheduler()
        for name in ["c", "a", "b"]:
            scheduler.add_task(BatchTask(name=name, config=get_preset_config("minimal"), n_steps=2))
        order = scheduler._topological_sort()
        # 无依赖时按字母序
        assert order == ["a", "b", "c"]

    def test_with_dependencies(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="a", config=get_preset_config("minimal"), n_steps=2, depends_on=["b"]))
        scheduler.add_task(BatchTask(name="b", config=get_preset_config("minimal"), n_steps=2))
        order = scheduler._topological_sort()
        assert order.index("b") < order.index("a")

    def test_chain_dependencies(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="c", config=get_preset_config("minimal"), n_steps=2, depends_on=["b"]))
        scheduler.add_task(BatchTask(name="b", config=get_preset_config("minimal"), n_steps=2, depends_on=["a"]))
        scheduler.add_task(BatchTask(name="a", config=get_preset_config("minimal"), n_steps=2))
        order = scheduler._topological_sort()
        assert order == ["a", "b", "c"]

    def test_missing_dependency_raises(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="a", config=get_preset_config("minimal"), n_steps=2, depends_on=["nonexistent"]))
        with pytest.raises(ValueError, match="不存在的任务"):
            scheduler._topological_sort()

    def test_circular_dependency_raises(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="a", config=get_preset_config("minimal"), n_steps=2, depends_on=["b"]))
        scheduler.add_task(BatchTask(name="b", config=get_preset_config("minimal"), n_steps=2, depends_on=["a"]))
        with pytest.raises(ValueError, match="循环依赖"):
            scheduler._topological_sort()


# ============================================================
# BatchScheduler - 运行
# ============================================================

class TestSchedulerRun:
    """调度器运行测试。"""

    def test_run_single_task(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="t1", config=get_preset_config("minimal"), n_steps=5))
        results = scheduler.run()
        assert "t1" in results
        assert results["t1"].success is True
        assert results["t1"].steps_executed == 5

    def test_run_multiple_tasks(self):
        scheduler = BatchScheduler()
        for i in range(3):
            scheduler.add_task(BatchTask(name=f"t{i}", config=get_preset_config("minimal"), n_steps=3))
        results = scheduler.run()
        assert len(results) == 3
        for name in ["t0", "t1", "t2"]:
            assert results[name].success is True

    def test_run_with_dependencies(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="base", config=get_preset_config("minimal"), n_steps=3))
        scheduler.add_task(BatchTask(name="dependent", config=get_preset_config("minimal"), n_steps=3, depends_on=["base"]))
        results = scheduler.run()
        assert results["base"].success is True
        assert results["dependent"].success is True

    def test_dependency_failure_skips_dependent(self):
        scheduler = BatchScheduler()
        # 构造一个会失败的任务：使用非法配置
        bad_config = get_preset_config("minimal").merge({"state_dim": -1})
        scheduler.add_task(BatchTask(name="bad", config=bad_config, n_steps=3, retries=1))
        scheduler.add_task(BatchTask(name="dependent", config=get_preset_config("minimal"), n_steps=3, depends_on=["bad"]))
        results = scheduler.run()
        assert results["bad"].success is False
        assert results["dependent"].success is False
        assert "依赖" in (results["dependent"].error or "")

    def test_retry_on_failure(self):
        scheduler = BatchScheduler()
        bad_config = get_preset_config("minimal").merge({"state_dim": -1})
        scheduler.add_task(BatchTask(name="flaky", config=bad_config, n_steps=3, retries=2))
        results = scheduler.run()
        # 应该尝试 3 次（1 + 2 重试）
        assert results["flaky"].attempts == 3
        assert results["flaky"].success is False

    def test_progress_callback(self):
        progress_calls = []
        def on_progress(task_name, completed, total):
            progress_calls.append((task_name, completed, total))

        scheduler = BatchScheduler(on_progress=on_progress)
        for i in range(3):
            scheduler.add_task(BatchTask(name=f"t{i}", config=get_preset_config("minimal"), n_steps=2))
        scheduler.run()
        assert len(progress_calls) == 3
        assert progress_calls[-1][1] == 3  # completed
        assert progress_calls[-1][2] == 3  # total

    def test_progress_callback_doesnt_crash(self):
        # 回调抛异常不应影响主流程
        def bad_callback(task_name, completed, total):
            raise RuntimeError("callback error")

        scheduler = BatchScheduler(on_progress=bad_callback)
        scheduler.add_task(BatchTask(name="t1", config=get_preset_config("minimal"), n_steps=2))
        results = scheduler.run()
        assert results["t1"].success is True

    def test_results_contain_metrics(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="t1", config=get_preset_config("minimal"), n_steps=5))
        results = scheduler.run()
        assert isinstance(results["t1"].metrics, dict)

    def test_results_contain_duration(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="t1", config=get_preset_config("minimal"), n_steps=3))
        results = scheduler.run()
        assert results["t1"].duration > 0

    def test_summary(self):
        scheduler = BatchScheduler()
        for i in range(3):
            scheduler.add_task(BatchTask(name=f"t{i}", config=get_preset_config("minimal"), n_steps=2))
        scheduler.run()
        summary = scheduler.summary()
        assert summary["total"] == 3
        assert summary["success"] == 3
        assert summary["failed"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["total_duration"] > 0

    def test_get_successful_results(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="good", config=get_preset_config("minimal"), n_steps=2))
        bad_config = get_preset_config("minimal").merge({"state_dim": -1})
        scheduler.add_task(BatchTask(name="bad", config=bad_config, n_steps=2))
        scheduler.run()
        successful = scheduler.get_successful_results()
        assert "good" in successful
        assert "bad" not in successful

    def test_get_failed_results(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="good", config=get_preset_config("minimal"), n_steps=2))
        bad_config = get_preset_config("minimal").merge({"state_dim": -1})
        scheduler.add_task(BatchTask(name="bad", config=bad_config, n_steps=2))
        scheduler.run()
        failed = scheduler.get_failed_results()
        assert "bad" in failed
        assert "good" not in failed


# ============================================================
# run_batch 便捷函数
# ============================================================

class TestRunBatch:
    """run_batch 便捷函数测试。"""

    def test_run_batch_basic(self):
        tasks = [
            BatchTask(name=f"t{i}", config=get_preset_config("minimal"), n_steps=3)
            for i in range(3)
        ]
        results = run_batch(tasks)
        assert len(results) == 3
        for r in results.values():
            assert r.success is True

    def test_run_batch_with_progress(self):
        progress = []
        tasks = [
            BatchTask(name=f"t{i}", config=get_preset_config("minimal"), n_steps=2)
            for i in range(2)
        ]
        run_batch(tasks, on_progress=lambda n, c, t: progress.append((n, c, t)))
        assert len(progress) == 2

    def test_run_batch_empty(self):
        results = run_batch([])
        assert results == {}


# ============================================================
# run_param_sweep 参数扫描
# ============================================================

class TestParamSweep:
    """参数扫描测试。"""

    def test_sweep_learning_rate(self):
        base = get_preset_config("minimal")
        results = run_param_sweep(
            base_config=base,
            param_name="learning_rate_micro",
            values=[0.001, 0.01, 0.1],
            n_steps=3,
        )
        assert len(results) == 3
        for name in ["learning_rate_micro=0.001", "learning_rate_micro=0.01", "learning_rate_micro=0.1"]:
            assert name in results

    def test_sweep_max_steps(self):
        base = get_preset_config("minimal")
        results = run_param_sweep(
            base_config=base,
            param_name="max_steps",
            values=[10, 20],
            n_steps=3,
        )
        assert len(results) == 2

    def test_sweep_nested_param(self):
        base = get_preset_config("minimal")
        results = run_param_sweep(
            base_config=base,
            param_name="resources.cpu_max_percent",
            values=[10.0, 20.0],
            n_steps=3,
            nested=True,
        )
        assert len(results) == 2
        # 验证嵌套参数确实被修改
        for name, result in results.items():
            if result.success and result.config:
                assert result.config.resources.cpu_max_percent in [10.0, 20.0]

    def test_sweep_all_succeed(self):
        base = get_preset_config("minimal")
        results = run_param_sweep(
            base_config=base,
            param_name="seed",
            values=[1, 2, 3],
            n_steps=3,
        )
        for r in results.values():
            assert r.success is True

    def test_sweep_task_names_contain_param(self):
        base = get_preset_config("minimal")
        results = run_param_sweep(
            base_config=base,
            param_name="seed",
            values=[42, 100],
            n_steps=2,
        )
        for name in results.keys():
            assert "seed=" in name


# ============================================================
# run_preset_comparison 预设对比
# ============================================================

class TestPresetComparison:
    """预设对比测试。"""

    def test_compare_all_presets(self):
        results = run_preset_comparison(n_steps=3)
        # 应包含所有预设
        for preset in ["minimal", "balanced", "performance", "conservative"]:
            assert preset in results

    def test_compare_specific_presets(self):
        results = run_preset_comparison(presets=["minimal", "balanced"], n_steps=3)
        assert len(results) == 2
        assert "minimal" in results
        assert "balanced" in results

    def test_comparison_all_succeed(self):
        results = run_preset_comparison(presets=["minimal"], n_steps=3)
        for r in results.values():
            assert r.success is True

    def test_comparison_with_progress(self):
        progress = []
        run_preset_comparison(
            presets=["minimal"],
            n_steps=2,
            on_progress=lambda n, c, t: progress.append((n, c, t)),
        )
        assert len(progress) == 1


# ============================================================
# 格式化
# ============================================================

class TestFormatResults:
    """格式化测试。"""

    def test_format_batch_results_success(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="t1", config=get_preset_config("minimal"), n_steps=3))
        scheduler.run()
        text = format_batch_results(scheduler.results)
        assert "SPEPA 批处理结果报告" in text
        assert "汇总" in text
        assert "成功" in text

    def test_format_batch_results_with_failure(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="good", config=get_preset_config("minimal"), n_steps=2))
        bad_config = get_preset_config("minimal").merge({"state_dim": -1})
        scheduler.add_task(BatchTask(name="bad", config=bad_config, n_steps=2))
        scheduler.run()
        text = format_batch_results(scheduler.results)
        assert "失败" in text

    def test_format_batch_results_empty(self):
        text = format_batch_results({})
        assert "SPEPA 批处理结果报告" in text

    def test_format_batch_summary(self):
        summary = {
            "total": 5,
            "success": 4,
            "failed": 1,
            "total_duration": 12.5,
            "success_rate": 0.8,
        }
        text = format_batch_summary(summary)
        assert "任务总数" in text
        assert "成功" in text
        assert "80.0%" in text

    def test_format_contains_task_names(self):
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="my_task", config=get_preset_config("minimal"), n_steps=2))
        scheduler.run()
        text = format_batch_results(scheduler.results)
        assert "my_task" in text


# ============================================================
# 端到端
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_full_workflow(self):
        """完整工作流：创建任务 → 运行 → 汇总 → 格式化。"""
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="exp1", config=get_preset_config("minimal"), n_steps=3))
        scheduler.add_task(BatchTask(name="exp2", config=get_preset_config("minimal"), n_steps=3))
        results = scheduler.run()
        summary = scheduler.summary()
        text = format_batch_results(results)
        assert summary["success"] == 2
        assert "SPEPA 批处理结果报告" in text

    def test_dependency_chain_workflow(self):
        """依赖链工作流。"""
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="a", config=get_preset_config("minimal"), n_steps=2))
        scheduler.add_task(BatchTask(name="b", config=get_preset_config("minimal"), n_steps=2, depends_on=["a"]))
        scheduler.add_task(BatchTask(name="c", config=get_preset_config("minimal"), n_steps=2, depends_on=["b"]))
        results = scheduler.run()
        assert all(r.success for r in results.values())

    def test_param_sweep_then_format(self):
        """参数扫描 → 格式化。"""
        results = run_param_sweep(
            base_config=get_preset_config("minimal"),
            param_name="seed",
            values=[1, 2],
            n_steps=2,
        )
        text = format_batch_results(results)
        assert "seed=1" in text
        assert "seed=2" in text

    def test_preset_comparison_then_format(self):
        """预设对比 → 格式化。"""
        results = run_preset_comparison(presets=["minimal", "balanced"], n_steps=2)
        text = format_batch_results(results)
        assert "minimal" in text
        assert "balanced" in text

    def test_retry_then_success_in_summary(self):
        """重试后汇总正确。"""
        scheduler = BatchScheduler()
        bad_config = get_preset_config("minimal").merge({"state_dim": -1})
        scheduler.add_task(BatchTask(name="bad", config=bad_config, n_steps=2, retries=2))
        scheduler.add_task(BatchTask(name="good", config=get_preset_config("minimal"), n_steps=2))
        scheduler.run()
        summary = scheduler.summary()
        assert summary["total"] == 2
        assert summary["success"] == 1
        assert summary["failed"] == 1
