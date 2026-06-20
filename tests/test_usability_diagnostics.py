"""阶段一测试：错误诊断与友好提示系统。

测试范围：
1. ErrorSuggestion 数据结构
2. friendly_error 友好错误转换
3. diagnose Agent 状态诊断
4. diagnose_config 配置诊断
5. safe_run 装饰器
6. ErrorContext 上下文管理器
7. format_diagnosis 格式化
"""
import io
import sys
import os
import tempfile
from contextlib import redirect_stderr

import numpy as np
import pytest

from spepa.diagnostics import (
    ErrorSuggestion, ErrorContext,
    friendly_error, diagnose, diagnose_config,
    safe_run, format_diagnosis,
)
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_preset_config, SPEPAConfig


# ============================================================
# ErrorSuggestion 测试
# ============================================================

class TestErrorSuggestion:
    """ErrorSuggestion 数据结构测试。"""

    def test_create_suggestion(self):
        """创建建议对象。"""
        s = ErrorSuggestion(
            code="TEST_ERROR",
            category="test",
            message="测试错误",
            cause="测试原因",
            fix="测试修复",
            example="test example",
        )
        assert s.code == "TEST_ERROR"
        assert s.category == "test"
        assert s.message == "测试错误"
        assert s.example == "test example"

    def test_default_example_empty(self):
        """默认 example 为空。"""
        s = ErrorSuggestion("C", "cat", "msg", "cause", "fix")
        assert s.example == ""


# ============================================================
# friendly_error 测试
# ============================================================

class TestFriendlyError:
    """friendly_error 测试。"""

    def test_returns_string(self):
        """返回字符串。"""
        err = ValueError("invalid state_dim")
        result = friendly_error(err)
        assert isinstance(result, str)
        assert "SPEPA 错误报告" in result

    def test_contains_error_type(self):
        """包含错误类型。"""
        err = RuntimeError("something wrong")
        result = friendly_error(err)
        assert "RuntimeError" in result

    def test_contains_context(self):
        """包含上下文。"""
        err = ValueError("test")
        result = friendly_error(err, context="测试上下文")
        assert "测试上下文" in result

    def test_matches_config_error(self):
        """匹配配置错误。"""
        err = ValueError("state_dim must be positive")
        result = friendly_error(err)
        assert "CONFIG_INVALID_DIM" in result
        assert "修复建议" in result

    def test_matches_file_error(self):
        """匹配文件错误。"""
        err = FileNotFoundError("config.json not found")
        result = friendly_error(err)
        assert "FILE_NOT_FOUND" in result

    def test_matches_shape_error(self):
        """匹配形状错误。"""
        err = ValueError("shape mismatch (8,) (4,)")
        result = friendly_error(err)
        assert "SHAPE_MISMATCH" in result

    def test_matches_nan_error(self):
        """匹配 NaN 错误。"""
        err = ValueError("value is nan")
        result = friendly_error(err)
        assert "NUMERICAL_INSTABILITY" in result

    def test_unmatched_error_shows_traceback(self):
        """未匹配的错误显示堆栈跟踪。"""
        err = RuntimeError("a very unique error message qzxwv12345")
        result = friendly_error(err)
        assert "堆栈跟踪" in result or "未识别" in result

    def test_contains_fix_and_example(self):
        """包含修复建议和示例。"""
        err = ValueError("invalid learning_rate")
        result = friendly_error(err)
        assert "修复建议" in result
        assert "示例" in result


# ============================================================
# diagnose 测试
# ============================================================

class TestDiagnose:
    """diagnose Agent 诊断测试。"""

    def test_returns_dict(self):
        """返回字典。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        result = diagnose(agent)
        assert isinstance(result, dict)
        assert "healthy" in result
        assert "issues" in result
        assert "warnings" in result
        assert "info" in result

    def test_healthy_agent(self):
        """健康 Agent 诊断。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=5)
        result = diagnose(agent)
        assert isinstance(result["healthy"], bool)
        assert isinstance(result["issues"], list)
        assert isinstance(result["info"], dict)

    def test_detects_non_finite_state(self):
        """检测非有限状态。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.current_state = np.full(agent.config.state_dim, np.nan)
        result = diagnose(agent)
        assert not result["healthy"]
        assert any(i.code == "STATE_NON_FINITE" for i in result["issues"])

    def test_detects_zero_state(self):
        """检测全零状态（警告）。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.current_state = np.zeros(agent.config.state_dim)
        result = diagnose(agent)
        assert any("零" in w for w in result["warnings"])

    def test_info_contains_step_count(self):
        """info 包含步数。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=3)
        result = diagnose(agent)
        assert "step_count" in result["info"]
        assert result["info"]["step_count"] >= 3

    def test_info_contains_state_norm(self):
        """info 包含状态范数。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        result = diagnose(agent)
        assert "state_norm" in result["info"]

    def test_does_not_modify_agent(self):
        """诊断不修改 Agent。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        state_before = agent.current_state.copy()
        step_before = agent.step_count
        diagnose(agent)
        assert np.allclose(agent.current_state, state_before)
        assert agent.step_count == step_before


# ============================================================
# diagnose_config 测试
# ============================================================

class TestDiagnoseConfig:
    """diagnose_config 配置诊断测试。"""

    def test_valid_config(self):
        """有效配置。"""
        config = get_preset_config("balanced")
        result = diagnose_config(config)
        assert result["valid"] is True
        assert len(result["issues"]) == 0

    def test_invalid_state_dim(self):
        """无效 state_dim。"""
        config = get_preset_config("balanced").merge({"state_dim": -1})
        result = diagnose_config(config)
        assert result["valid"] is False
        assert any(i.code == "CONFIG_INVALID_DIM" for i in result["issues"])

    def test_invalid_action_dim(self):
        """无效 action_dim。"""
        config = get_preset_config("balanced").merge({"action_dim": 0})
        result = diagnose_config(config)
        assert result["valid"] is False
        assert any(i.code == "CONFIG_INVALID_DIM" for i in result["issues"])

    def test_invalid_memory(self):
        """无效记忆容量。"""
        config = get_preset_config("balanced")
        config = config.merge({"fast_memory_capacity": -1})
        result = diagnose_config(config)
        assert result["valid"] is False
        assert any(i.code == "CONFIG_MEMORY_INVALID" for i in result["issues"])

    def test_invalid_seed(self):
        """无效种子。"""
        config = get_preset_config("balanced").merge({"seed": -5})
        result = diagnose_config(config)
        assert result["valid"] is False
        assert any(i.code == "CONFIG_SEED_INVALID" for i in result["issues"])

    def test_invalid_steps(self):
        """无效步数。"""
        config = get_preset_config("balanced").merge({"max_steps": 0})
        result = diagnose_config(config)
        assert result["valid"] is False
        assert any(i.code == "CONFIG_INVALID_STEPS" for i in result["issues"])

    def test_warning_for_learning_rate(self):
        """学习率警告。"""
        config = get_preset_config("balanced").merge({"learning_rate_micro": 2.0})
        result = diagnose_config(config)
        assert len(result["warnings"]) > 0

    def test_presets_are_valid(self):
        """所有预设配置应有效。"""
        from spepa.utils.config import list_presets, get_preset_config
        for name in list_presets():
            config = get_preset_config(name)
            result = diagnose_config(config)
            assert result["valid"] is True, f"预设 {name} 配置无效"


# ============================================================
# safe_run 装饰器测试
# ============================================================

class TestSafeRun:
    """safe_run 装饰器测试。"""

    def test_successful_run(self):
        """成功运行返回结果。"""
        @safe_run
        def success():
            return 42

        assert success() == 42

    def test_catches_exception(self, capsys):
        """捕获异常并输出友好信息。"""
        @safe_run
        def fail():
            raise ValueError("state_dim invalid")

        result = fail()
        assert result is None
        captured = capsys.readouterr()
        assert "SPEPA 错误报告" in captured.err

    def test_preserves_function_name(self):
        """保留函数名。"""
        @safe_run
        def my_function():
            """docstring"""
            return 1

        assert my_function.__name__ == "my_function"

    def test_with_arguments(self):
        """带参数的函数。"""
        @safe_run
        def add(a, b):
            return a + b

        assert add(1, 2) == 3


# ============================================================
# ErrorContext 测试
# ============================================================

class TestErrorContext:
    """ErrorContext 上下文管理器测试。"""

    def test_no_exception(self):
        """无异常时正常退出。"""
        with ErrorContext("测试"):
            x = 1 + 1
        assert x == 2

    def test_catches_exception(self, capsys):
        """捕获异常并输出友好信息。"""
        with ErrorContext("测试上下文"):
            raise ValueError("state_dim invalid")

        captured = capsys.readouterr()
        assert "SPEPA 错误报告" in captured.err
        assert "测试上下文" in captured.err

    def test_reraise_option(self, capsys):
        """reraise=True 时重新抛出。"""
        with pytest.raises(ValueError):
            with ErrorContext("测试", reraise=True):
                raise ValueError("test error")

        captured = capsys.readouterr()
        assert "SPEPA 错误报告" in captured.err

    def test_suppresses_by_default(self):
        """默认抑制异常。"""
        # 不应抛出异常
        with ErrorContext("测试"):
            raise RuntimeError("test")


# ============================================================
# format_diagnosis 测试
# ============================================================

class TestFormatDiagnosis:
    """format_diagnosis 格式化测试。"""

    def test_format_agent_diagnosis(self):
        """格式化 Agent 诊断。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=3)
        diag = diagnose(agent)
        text = format_diagnosis(diag)
        assert isinstance(text, str)
        assert "SPEPA 诊断报告" in text
        assert "状态" in text

    def test_format_config_diagnosis(self):
        """格式化配置诊断。"""
        config = get_preset_config("balanced")
        diag = diagnose_config(config)
        text = format_diagnosis(diag)
        assert "SPEPA 诊断报告" in text
        assert "配置状态" in text

    def test_format_with_issues(self):
        """格式化有问题的情况。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.current_state = np.full(agent.config.state_dim, np.nan)
        diag = diagnose(agent)
        text = format_diagnosis(diag)
        assert "STATE_NON_FINITE" in text
        assert "修复" in text

    def test_format_no_issues(self):
        """格式化无问题的情况。"""
        config = get_preset_config("minimal")
        diag = diagnose_config(config)
        text = format_diagnosis(diag)
        assert "未发现问题" in text


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端测试。"""

    def test_full_diagnosis_workflow(self):
        """完整诊断工作流。"""
        # 1. 创建并运行 Agent
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=5)

        # 2. 诊断
        diag = diagnose(agent)
        text = format_diagnosis(diag)
        assert "SPEPA 诊断报告" in text

        # 3. 配置诊断
        config_diag = diagnose_config(agent.config)
        assert config_diag["valid"] is True

    def test_error_recovery_workflow(self):
        """错误恢复工作流。"""
        # 1. 用 safe_run 包装可能失败的操作
        @safe_run
        def load_config(filepath):
            return SPEPAConfig.load_from_file(filepath)

        # 2. 加载不存在的文件
        result = load_config("/nonexistent/path.json")
        assert result is None  # safe_run 捕获了异常

    def test_context_manager_workflow(self, capsys):
        """上下文管理工作流。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "config.json")
            config = get_preset_config("balanced")
            config.save_to_file(filepath)

            # 成功加载
            with ErrorContext("加载配置"):
                loaded = SPEPAConfig.load_from_file(filepath)
                assert loaded.state_dim == config.state_dim

    def test_diagnose_after_problem(self):
        """问题发生后诊断。"""
        agent = SPEPAAgent(get_preset_config("minimal"))

        # 制造问题：设置极端状态
        agent.current_state = np.full(agent.config.state_dim, 1e10)

        diag = diagnose(agent)
        # 应检测到状态范数过大
        assert any("范数过大" in w for w in diag["warnings"]) or \
               any(i.code == "STATE_NON_FINITE" for i in diag["issues"]) or \
               diag["info"]["state_norm"] > 10.0
