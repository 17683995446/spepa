"""阶段三测试：日志管理与调试工具。

测试范围：
1. 日志管理增强（set_log_level, add_file_handler, quiet/verbose_mode 等）
2. 状态快照（snapshot, format_snapshot）
3. 健康检查（health_check）
4. 性能分析器（DebugProfiler）
5. 快照比较（compare_snapshots, format_comparison）
"""
import os
import tempfile
import time
import logging
import numpy as np
import pytest

from spepa.utils.logger import (
    get_logger, set_log_level, add_file_handler, remove_handlers,
    get_log_config, quiet_mode, verbose_mode, list_loggers,
)
from spepa.debug import (
    snapshot, format_snapshot, health_check,
    DebugProfiler, compare_snapshots, format_comparison,
)
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_preset_config


# ============================================================
# 日志管理测试
# ============================================================

class TestLogLevelManagement:
    """日志级别管理测试。"""

    def test_set_log_level_by_string(self):
        """字符串级别设置。"""
        logger = get_logger("spepa.test_level")
        set_log_level("debug", "spepa.test_level")
        assert logger.level == logging.DEBUG
        set_log_level("warning", "spepa.test_level")
        assert logger.level == logging.WARNING

    def test_set_log_level_by_int(self):
        """数值级别设置。"""
        logger = get_logger("spepa.test_int")
        set_log_level(logging.ERROR, "spepa.test_int")
        assert logger.level == logging.ERROR

    def test_set_log_level_global(self):
        """全局设置所有 logger。"""
        get_logger("spepa.test_global1")
        get_logger("spepa.test_global2")
        set_log_level("error")
        # 所有已缓存的 logger 都应被设置
        assert get_logger("spepa.test_global1").level == logging.ERROR
        assert get_logger("spepa.test_global2").level == logging.ERROR
        # 恢复
        set_log_level("info")

    def test_quiet_mode(self):
        """静默模式。"""
        get_logger("spepa.test_quiet")
        quiet_mode()
        assert get_logger("spepa.test_quiet").level == logging.WARNING
        # 恢复
        set_log_level("info")

    def test_verbose_mode(self):
        """详细模式。"""
        get_logger("spepa.test_verbose")
        verbose_mode()
        assert get_logger("spepa.test_verbose").level == logging.DEBUG
        # 恢复
        set_log_level("info")

    def test_list_loggers(self):
        """列出所有 logger。"""
        get_logger("spepa.test_list")
        names = list_loggers()
        assert "spepa.test_list" in names


class TestFileHandler:
    """文件日志处理器测试。"""

    def test_add_file_handler(self):
        """添加文件处理器。"""
        logger = get_logger("spepa.test_file")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.log")
            handler = add_file_handler(filepath, "debug", "spepa.test_file")
            assert os.path.exists(filepath)
            # 写入日志
            logger.info("test message for file")
            handler.flush()
            # 验证文件内容
            with open(filepath, 'r') as f:
                content = f.read()
            assert "test message for file" in content
            # 清理
            remove_handlers("spepa.test_file", logging.FileHandler)

    def test_remove_file_handler(self):
        """移除文件处理器。"""
        logger = get_logger("spepa.test_remove")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.log")
            add_file_handler(filepath, "debug", "spepa.test_remove")
            # 确认有 FileHandler
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) >= 1
            # 移除
            remove_handlers("spepa.test_remove", logging.FileHandler)
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 0

    def test_get_log_config(self):
        """获取日志配置。"""
        get_logger("spepa.test_config")
        config = get_log_config()
        assert "spepa.test_config" in config
        assert "level" in config["spepa.test_config"]
        assert "handlers" in config["spepa.test_config"]


# ============================================================
# 状态快照测试
# ============================================================

class TestSnapshot:
    """状态快照测试。"""

    @pytest.fixture
    def agent(self):
        """创建一个运行了几步的 Agent。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        agent.run(n_steps=3)
        return agent

    def test_snapshot_returns_dict(self, agent):
        """snapshot 返回字典。"""
        snap = snapshot(agent)
        assert isinstance(snap, dict)

    def test_snapshot_contains_runtime(self, agent):
        """快照包含运行时信息。"""
        snap = snapshot(agent)
        assert "runtime" in snap
        assert snap["runtime"]["step_count"] >= 3
        assert snap["runtime"]["online_cycles"] >= 1

    def test_snapshot_contains_state(self, agent):
        """快照包含状态向量统计。"""
        snap = snapshot(agent)
        assert "state" in snap
        assert "dim" in snap["state"]
        assert "mean" in snap["state"]
        assert "norm" in snap["state"]
        assert snap["state"]["dim"] == agent.config.state_dim

    def test_snapshot_contains_metrics(self, agent):
        """快照包含指标摘要。"""
        snap = snapshot(agent)
        assert "metrics" in snap

    def test_snapshot_contains_health(self, agent):
        """快照包含健康信息。"""
        snap = snapshot(agent)
        assert "health" in snap
        assert "score" in snap["health"]

    def test_snapshot_contains_resources(self, agent):
        """快照包含资源信息。"""
        snap = snapshot(agent)
        assert "resources" in snap
        assert "cpu_usage" in snap["resources"]

    def test_snapshot_contains_degradation(self, agent):
        """快照包含降级信息。"""
        snap = snapshot(agent)
        assert "degradation" in snap
        assert "level" in snap["degradation"]

    def test_snapshot_contains_memory(self, agent):
        """快照包含记忆信息。"""
        snap = snapshot(agent)
        assert "memory" in snap
        assert "fast_size" in snap["memory"]

    def test_snapshot_contains_goals(self, agent):
        """快照包含目标信息。"""
        snap = snapshot(agent)
        assert "goals" in snap

    def test_snapshot_contains_model(self, agent):
        """快照包含模型信息。"""
        snap = snapshot(agent)
        assert "model" in snap
        assert "n_layers" in snap["model"]

    def test_snapshot_does_not_modify_agent(self, agent):
        """快照不修改 Agent 状态。"""
        state_before = agent.current_state.copy()
        step_before = agent.step_count
        snapshot(agent)
        assert np.allclose(agent.current_state, state_before)
        assert agent.step_count == step_before


class TestFormatSnapshot:
    """快照格式化测试。"""

    def test_format_snapshot_returns_string(self):
        """format_snapshot 返回字符串。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        agent.run(n_steps=2)
        snap = snapshot(agent)
        text = format_snapshot(snap)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_format_snapshot_contains_sections(self):
        """格式化文本包含各部分。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        agent.run(n_steps=2)
        snap = snapshot(agent)
        text = format_snapshot(snap)
        assert "[Runtime]" in text
        assert "[State Vector]" in text
        assert "[Health]" in text
        assert "[Resources]" in text
        assert "[Memory]" in text

    def test_format_snapshot_handles_empty_agent(self):
        """格式化未运行的 Agent 快照。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        snap = snapshot(agent)
        text = format_snapshot(snap)
        assert "step_count" in text


# ============================================================
# 健康检查测试
# ============================================================

class TestHealthCheck:
    """健康检查测试。"""

    def test_health_check_returns_dict(self):
        """health_check 返回字典。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        result = health_check(agent)
        assert isinstance(result, dict)
        assert "healthy" in result
        assert "health_score" in result
        assert "issues" in result
        assert "recommendations" in result

    def test_health_check_healthy_agent(self):
        """健康 Agent 的检查结果。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        agent.run(n_steps=5)
        result = health_check(agent)
        assert isinstance(result["healthy"], bool)
        assert isinstance(result["issues"], list)
        assert isinstance(result["recommendations"], list)

    def test_health_check_detects_zero_state(self):
        """检测全零状态。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        agent.current_state = np.zeros(config.state_dim)
        result = health_check(agent)
        # 应检测到状态全零
        assert any("零" in issue for issue in result["issues"])

    def test_health_check_detects_non_finite(self):
        """检测非有限值。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        agent.current_state = np.full(config.state_dim, np.nan)
        result = health_check(agent)
        assert any("非有限" in issue for issue in result["issues"])


# ============================================================
# 性能分析器测试
# ============================================================

class TestDebugProfiler:
    """性能分析器测试。"""

    def test_profiler_basic(self):
        """基本计时功能。"""
        profiler = DebugProfiler()
        profiler.start("task1")
        time.sleep(0.01)
        profiler.end("task1")
        summary = profiler.summary()
        assert "task1" in summary
        assert summary["task1"]["count"] == 1
        assert summary["task1"]["total"] > 0

    def test_profiler_context_manager(self):
        """上下文管理器用法。"""
        profiler = DebugProfiler()
        with profiler.section("task2"):
            time.sleep(0.01)
        summary = profiler.summary()
        assert "task2" in summary
        assert summary["task2"]["count"] == 1

    def test_profiler_multiple_calls(self):
        """多次调用同一 section。"""
        profiler = DebugProfiler()
        for _ in range(3):
            with profiler.section("repeat"):
                time.sleep(0.005)
        summary = profiler.summary()
        assert summary["repeat"]["count"] == 3

    def test_profiler_report(self):
        """报告生成。"""
        profiler = DebugProfiler()
        with profiler.section("fast"):
            pass
        with profiler.section("slow"):
            time.sleep(0.01)
        report = profiler.report()
        assert isinstance(report, str)
        assert "Performance Profile" in report
        assert "fast" in report
        assert "slow" in report

    def test_profiler_reset(self):
        """重置。"""
        profiler = DebugProfiler()
        with profiler.section("task"):
            pass
        assert len(profiler.summary()) > 0
        profiler.reset()
        assert len(profiler.summary()) == 0

    def test_profiler_end_without_start(self):
        """未 start 就 end 返回 0。"""
        profiler = DebugProfiler()
        elapsed = profiler.end("nonexistent")
        assert elapsed == 0.0


# ============================================================
# 快照比较测试
# ============================================================

class TestCompareSnapshots:
    """快照比较测试。"""

    def test_compare_returns_dict(self):
        """compare_snapshots 返回字典。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        s1 = snapshot(agent)
        agent.run(n_steps=2)
        s2 = snapshot(agent)
        comp = compare_snapshots(s1, s2)
        assert isinstance(comp, dict)
        assert "time_delta" in comp
        assert "step_delta" in comp
        assert "state_delta" in comp

    def test_compare_step_delta(self):
        """步数差正确。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        s1 = snapshot(agent)
        agent.run(n_steps=3)
        s2 = snapshot(agent)
        comp = compare_snapshots(s1, s2)
        assert comp["step_delta"] >= 3

    def test_compare_time_delta_positive(self):
        """时间差为正。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        s1 = snapshot(agent)
        time.sleep(0.05)
        s2 = snapshot(agent)
        comp = compare_snapshots(s1, s2)
        assert comp["time_delta"] > 0

    def test_format_comparison(self):
        """格式化比较结果。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        s1 = snapshot(agent)
        agent.run(n_steps=2)
        s2 = snapshot(agent)
        comp = compare_snapshots(s1, s2)
        text = format_comparison(comp)
        assert isinstance(text, str)
        assert "Snapshot Comparison" in text
        assert "step_delta" in text


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端工作流测试。"""

    def test_debug_workflow(self):
        """完整调试工作流：快照 -> 运行 -> 比较。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)

        # 1. 初始快照
        s1 = snapshot(agent)
        text1 = format_snapshot(s1)
        assert "step_count:    0" in text1

        # 2. 运行
        agent.run(n_steps=5)

        # 3. 第二次快照
        s2 = snapshot(agent)
        text2 = format_snapshot(s2)
        assert "step_count:    5" in text2 or "step_count:    6" in text2

        # 4. 比较
        comp = compare_snapshots(s1, s2)
        assert comp["step_delta"] >= 5

        # 5. 健康检查
        health = health_check(agent)
        assert "healthy" in health

    def test_profiler_with_agent(self):
        """性能分析器与 Agent 配合使用。"""
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        profiler = DebugProfiler()

        with profiler.section("run"):
            agent.run(n_steps=3)

        summary = profiler.summary()
        assert "run" in summary
        assert summary["run"]["total"] > 0

        report = profiler.report()
        assert "run" in report

    def test_log_management_workflow(self):
        """日志管理工作流。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 添加文件日志
            log_file = os.path.join(tmpdir, "spepa.log")
            add_file_handler(log_file, "info")

            # 2. 运行 Agent（产生日志）
            config = get_preset_config("minimal")
            agent = SPEPAAgent(config)
            agent.run(n_steps=2)

            # 3. 验证日志文件有内容
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
            assert len(content) > 0

            # 4. 清理
            remove_handlers(handler_type=logging.FileHandler)
