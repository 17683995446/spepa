"""SPEPA 易用性阶段一测试：包化与预设配置。

测试范围：
1. 预设配置：minimal/balanced/performance/conservative
2. 配置校验：validate() 方法
3. 友好错误提示：非法配置的清晰报错
4. list_presets：列出所有预设
5. 包导出：__init__.py 导出新函数
"""
import pytest
import numpy as np

from spepa.utils.config import (
    SPEPAConfig, get_default_config, get_preset_config, list_presets,
    ResourceLimits, ExplorationConfig, DriveWeights,
)
from spepa.agent import SPEPAAgent


# ============================================================
# 预设配置测试
# ============================================================

class TestPresetConfigs:
    """预设配置测试。"""

    def test_list_presets_returns_all(self):
        """list_presets 返回所有预设。"""
        presets = list_presets()
        assert "minimal" in presets
        assert "balanced" in presets
        assert "performance" in presets
        assert "conservative" in presets
        assert len(presets) == 4

    def test_get_preset_minimal(self):
        """minimal 预设正确。"""
        config = get_preset_config("minimal")
        assert config.state_dim == 8
        assert config.action_dim == 4
        assert config.max_steps == 200
        assert config.resources.cpu_max_percent == 10.0

    def test_get_preset_balanced(self):
        """balanced 预设正确（等同于默认）。"""
        config = get_preset_config("balanced")
        assert config.state_dim == 16
        assert config.action_dim == 8
        assert config.max_steps == 1000

    def test_get_preset_performance(self):
        """performance 预设正确。"""
        config = get_preset_config("performance")
        assert config.state_dim == 32
        assert config.action_dim == 16
        assert config.max_steps == 5000
        assert config.resources.cpu_max_percent == 80.0

    def test_get_preset_conservative(self):
        """conservative 预设正确。"""
        config = get_preset_config("conservative")
        assert config.safety_hard_boundary == 0.85
        assert config.drives.safety == 2.0
        assert config.max_steps == 500

    def test_get_preset_case_insensitive(self):
        """预设名称大小写不敏感。"""
        config = get_preset_config("MINIMAL")
        assert config.state_dim == 8
        config2 = get_preset_config(" Balanced ")
        assert config2.state_dim == 16

    def test_get_preset_unknown_raises(self):
        """未知预设抛出 ValueError。"""
        with pytest.raises(ValueError, match="未知预设配置"):
            get_preset_config("nonexistent")

    def test_get_preset_error_lists_available(self):
        """错误信息列出可用预设。"""
        try:
            get_preset_config("bad")
        except ValueError as e:
            assert "minimal" in str(e)
            assert "balanced" in str(e)
            assert "performance" in str(e)
            assert "conservative" in str(e)

    def test_preset_config_validates(self):
        """预设配置通过校验。"""
        for name in list_presets():
            config = get_preset_config(name)
            config.validate()  # 不应抛出异常

    def test_preset_config_creates_agent(self):
        """预设配置能创建 Agent。"""
        for name in list_presets():
            config = get_preset_config(name)
            agent = SPEPAAgent(config)
            assert agent.config.state_dim == config.state_dim


# ============================================================
# 配置校验测试
# ============================================================

class TestConfigValidation:
    """配置校验测试。"""

    def test_valid_config_passes(self):
        """合法配置通过校验。"""
        config = get_default_config()
        config.validate()  # 不应抛出异常

    def test_invalid_state_dim(self):
        """非法 state_dim 报错。"""
        config = SPEPAConfig(state_dim=0)
        with pytest.raises(ValueError, match="state_dim 必须为正整数"):
            config.validate()

    def test_invalid_action_dim(self):
        """非法 action_dim 报错。"""
        config = SPEPAConfig(action_dim=-1)
        with pytest.raises(ValueError, match="action_dim 必须为正整数"):
            config.validate()

    def test_invalid_learning_rate(self):
        """非法学习率报错。"""
        config = SPEPAConfig(learning_rate_micro=0)
        with pytest.raises(ValueError, match="learning_rate_micro 必须为正数"):
            config.validate()

    def test_invalid_safety_boundary(self):
        """非法安全边界报错。"""
        config = SPEPAConfig(safety_hard_boundary=1.5)
        with pytest.raises(ValueError, match="safety_hard_boundary"):
            config.validate()

    def test_invalid_exploration_range(self):
        """探索比例 min > max 报错。"""
        config = SPEPAConfig(
            exploration=ExplorationConfig(
                exploration_ratio_min=0.6,
                exploration_ratio_max=0.3,
            )
        )
        with pytest.raises(ValueError, match="不能大于"):
            config.validate()

    def test_invalid_threshold_order(self):
        """micro_threshold >= struct_threshold 报错。"""
        config = SPEPAConfig(
            micro_correction_threshold=1.5,
            structural_reconstruction_threshold=1.0,
        )
        with pytest.raises(ValueError, match="必须小于"):
            config.validate()

    def test_multiple_errors_reported(self):
        """多个错误同时报告。"""
        config = SPEPAConfig(
            state_dim=0,
            action_dim=-1,
            learning_rate_micro=0,
        )
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        msg = str(exc_info.value)
        assert "state_dim" in msg
        assert "action_dim" in msg
        assert "learning_rate_micro" in msg
        assert "3 个错误" in msg

    def test_invalid_cpu_percent(self):
        """非法 CPU 上限报错。"""
        config = SPEPAConfig(resources=ResourceLimits(cpu_max_percent=0))
        with pytest.raises(ValueError, match="cpu_max_percent"):
            config.validate()


# ============================================================
# 包导出测试
# ============================================================

class TestPackageExports:
    """包导出测试。"""

    def test_import_spepa(self):
        """能从 spepa 包导入所有公开 API。"""
        import spepa
        assert hasattr(spepa, "SPEPAAgent")
        assert hasattr(spepa, "SPEPAConfig")
        assert hasattr(spepa, "get_default_config")
        assert hasattr(spepa, "get_preset_config")
        assert hasattr(spepa, "list_presets")
        assert hasattr(spepa, "get_logger")

    def test_version(self):
        """版本号存在。"""
        import spepa
        assert hasattr(spepa, "__version__")
        assert spepa.__version__ == "0.1.0"

    def test_all_exports(self):
        """__all__ 包含所有导出。"""
        import spepa
        assert "SPEPAAgent" in spepa.__all__
        assert "SPEPAConfig" in spepa.__all__
        assert "get_preset_config" in spepa.__all__
        assert "list_presets" in spepa.__all__


# ============================================================
# 端到端易用性测试
# ============================================================

class TestUsabilityEndToEnd:
    """端到端易用性测试。"""

    def test_one_line_agent_creation(self):
        """一行代码创建 Agent。"""
        from spepa import SPEPAAgent
        agent = SPEPAAgent()
        assert agent is not None

    def test_preset_then_run(self):
        """预设配置后直接运行。"""
        from spepa import SPEPAAgent, get_preset_config
        config = get_preset_config("minimal")
        agent = SPEPAAgent(config)
        result = agent.run(n_steps=5)
        assert "total_steps" in result or "steps" in result

    def test_list_presets_before_use(self):
        """先列出预设再选择。"""
        from spepa import list_presets, get_preset_config
        presets = list_presets()
        # 用户可以查看描述后选择
        for name in presets:
            config = get_preset_config(name)
            assert isinstance(config, SPEPAConfig)
