"""SPEPA 易用性阶段一测试：配置文件持久化与导入导出。

测试范围：
1. to_dict/from_dict 往返一致性
2. save_to_file/load_from_file JSON 格式
3. save_to_file/load_from_file YAML 格式（如 PyYAML 可用）
4. merge 部分覆盖
5. CLI config save/show 命令
6. CLI --config 选项
"""
import os
import json
import tempfile
import pytest

from spepa.utils.config import (
    SPEPAConfig, get_default_config, get_preset_config,
    ResourceLimits, ExplorationConfig, DriveWeights,
)
from spepa.agent import SPEPAAgent
from spepa.cli import main


# ============================================================
# to_dict / from_dict 往返测试
# ============================================================

class TestConfigRoundTrip:
    """配置序列化往返测试。"""

    def test_to_dict_contains_all_fields(self):
        """to_dict 包含所有字段。"""
        config = get_default_config()
        d = config.to_dict()
        # 核心字段
        assert "state_dim" in d
        assert "action_dim" in d
        assert "obs_dim" in d
        assert "prediction_layers" in d
        assert "learning_rate_micro" in d
        assert "learning_rate_struct" in d
        assert "max_steps" in d
        assert "seed" in d
        assert "safety_hard_boundary" in d
        # 嵌套子配置
        assert "resources" in d
        assert "exploration" in d
        assert "drives" in d
        assert "cpu_max_percent" in d["resources"]
        assert "exploration_ratio_min" in d["exploration"]
        assert "safety" in d["drives"]

    def test_from_dict_reconstructs_config(self):
        """from_dict 重建配置。"""
        original = get_default_config()
        d = original.to_dict()
        reconstructed = SPEPAConfig.from_dict(d)
        assert reconstructed.state_dim == original.state_dim
        assert reconstructed.action_dim == original.action_dim
        assert reconstructed.seed == original.seed
        assert reconstructed.resources.cpu_max_percent == original.resources.cpu_max_percent
        assert reconstructed.exploration.exploration_ratio_min == original.exploration.exploration_ratio_min

    def test_round_trip_preserves_all_fields(self):
        """往返保留所有字段。"""
        original = get_preset_config("performance")
        d = original.to_dict()
        reconstructed = SPEPAConfig.from_dict(d)
        assert reconstructed.state_dim == original.state_dim
        assert reconstructed.prediction_layers == original.prediction_layers
        assert reconstructed.fast_memory_capacity == original.fast_memory_capacity
        assert reconstructed.learning_rate_micro == original.learning_rate_micro
        assert reconstructed.gap_level_thresholds == original.gap_level_thresholds

    def test_from_dict_partial_data(self):
        """from_dict 接受部分数据。"""
        config = SPEPAConfig.from_dict({"state_dim": 32, "seed": 100})
        assert config.state_dim == 32
        assert config.seed == 100
        # 未指定的字段使用默认值
        assert config.action_dim == 8

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict 忽略未知键。"""
        config = SPEPAConfig.from_dict({
            "state_dim": 16,
            "unknown_field": "should_be_ignored",
        })
        assert config.state_dim == 16

    def test_from_dict_handles_nested_partial(self):
        """from_dict 处理部分嵌套子配置。"""
        config = SPEPAConfig.from_dict({
            "resources": {"cpu_max_percent": 50.0}
        })
        assert config.resources.cpu_max_percent == 50.0
        # 未指定的子配置字段使用默认值
        assert config.resources.memory_max_percent == 20.0

    def test_gap_level_thresholds_list_to_tuple(self):
        """gap_level_thresholds 从 list 转回 tuple。"""
        d = {"gap_level_thresholds": [0.1, 0.2, 0.3, 0.4]}
        config = SPEPAConfig.from_dict(d)
        assert isinstance(config.gap_level_thresholds, tuple)
        assert config.gap_level_thresholds == (0.1, 0.2, 0.3, 0.4)


# ============================================================
# save_to_file / load_from_file 测试
# ============================================================

class TestConfigFileIO:
    """配置文件读写测试。"""

    def test_save_to_json_file(self):
        """保存为 JSON 文件。"""
        config = get_preset_config("minimal")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = config.save_to_file(os.path.join(tmpdir, "config.json"))
            assert filepath.endswith(".json")
            assert os.path.exists(filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert data["state_dim"] == 8

    def test_load_from_json_file(self):
        """从 JSON 文件加载。"""
        config = get_preset_config("minimal")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = config.save_to_file(os.path.join(tmpdir, "config.json"))
            loaded = SPEPAConfig.load_from_file(filepath)
            assert loaded.state_dim == config.state_dim
            assert loaded.action_dim == config.action_dim
            assert loaded.seed == config.seed

    def test_save_load_round_trip_json(self):
        """JSON 往返一致。"""
        original = get_preset_config("performance")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = original.save_to_file(os.path.join(tmpdir, "config.json"))
            loaded = SPEPAConfig.load_from_file(filepath)
            assert loaded.to_dict() == original.to_dict()

    def test_save_auto_adds_extension(self):
        """无扩展名时自动添加 .json。"""
        config = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = config.save_to_file(os.path.join(tmpdir, "config"))
            assert filepath.endswith(".json")

    def test_save_creates_directory(self):
        """保存时创建目录。"""
        config = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "config.json")
            filepath = config.save_to_file(nested)
            assert os.path.exists(filepath)

    def test_load_nonexistent_file_raises(self):
        """加载不存在的文件报错。"""
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            SPEPAConfig.load_from_file("/nonexistent/path/config.json")

    def test_save_yaml_if_available(self):
        """YAML 格式保存（如 PyYAML 可用）。"""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        config = get_preset_config("minimal")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = config.save_to_file(os.path.join(tmpdir, "config.yaml"))
            assert filepath.endswith(".yaml")
            assert os.path.exists(filepath)
            loaded = SPEPAConfig.load_from_file(filepath)
            assert loaded.state_dim == config.state_dim

    def test_loaded_config_validates(self):
        """加载的配置通过校验。"""
        config = get_preset_config("conservative")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = config.save_to_file(os.path.join(tmpdir, "config.json"))
            loaded = SPEPAConfig.load_from_file(filepath)
            loaded.validate()  # 不应抛出异常

    def test_loaded_config_creates_agent(self):
        """加载的配置能创建 Agent。"""
        config = get_preset_config("minimal")
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = config.save_to_file(os.path.join(tmpdir, "config.json"))
            loaded = SPEPAConfig.load_from_file(filepath)
            agent = SPEPAAgent(loaded)
            assert agent.config.state_dim == 8


# ============================================================
# merge 测试
# ============================================================

class TestConfigMerge:
    """配置合并测试。"""

    def test_merge_simple_override(self):
        """合并简单覆盖。"""
        config = get_default_config()
        new_config = config.merge({"state_dim": 32, "seed": 100})
        assert new_config.state_dim == 32
        assert new_config.seed == 100
        # 原配置不变
        assert config.state_dim == 16
        assert config.seed == 42

    def test_merge_nested_override(self):
        """合并嵌套覆盖。"""
        config = get_default_config()
        new_config = config.merge({
            "resources": {"cpu_max_percent": 50.0}
        })
        assert new_config.resources.cpu_max_percent == 50.0
        # 其他子配置字段保留
        assert new_config.resources.memory_max_percent == 20.0

    def test_merge_does_not_modify_original(self):
        """合并不修改原配置。"""
        config = get_default_config()
        original_dict = config.to_dict()
        config.merge({"state_dim": 32})
        assert config.to_dict() == original_dict

    def test_merge_multiple_fields(self):
        """合并多个字段。"""
        config = get_default_config()
        new_config = config.merge({
            "state_dim": 32,
            "action_dim": 16,
            "seed": 999,
            "exploration": {"exploration_ratio_min": 0.1},
        })
        assert new_config.state_dim == 32
        assert new_config.action_dim == 16
        assert new_config.seed == 999
        assert new_config.exploration.exploration_ratio_min == 0.1

    def test_merge_then_save_load(self):
        """合并后保存加载一致。"""
        config = get_default_config()
        merged = config.merge({"state_dim": 24, "seed": 77})
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = merged.save_to_file(os.path.join(tmpdir, "config.json"))
            loaded = SPEPAConfig.load_from_file(filepath)
            assert loaded.state_dim == 24
            assert loaded.seed == 77


# ============================================================
# CLI config 命令测试
# ============================================================

class TestCLIConfig:
    """CLI config 命令测试。"""

    def test_config_save_command(self, capsys):
        """config save 命令。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "my_config.json")
            ret = main(["config", "save", filepath, "--preset", "minimal"])
            assert ret == 0
            assert os.path.exists(filepath)
            captured = capsys.readouterr()
            assert "保存" in captured.out

    def test_config_show_command(self, capsys):
        """config show 命令。"""
        ret = main(["config", "show", "--preset", "minimal"])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["state_dim"] == 8

    def test_config_save_then_run_with_config(self, capsys):
        """保存配置后用 --config 运行。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "my_config.json")
            main(["config", "save", filepath, "--preset", "minimal"])
            ret = main(["run", "--config", filepath, "--steps", "3", "--quiet"])
            assert ret == 0

    def test_config_option_overrides_preset(self, capsys):
        """--config 优先于 --preset。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存 minimal 配置
            filepath = os.path.join(tmpdir, "my_config.json")
            main(["config", "save", filepath, "--preset", "minimal"])
            capsys.readouterr()  # 清空第一次调用的输出
            # 用 --config 加载（应使用 minimal 的 state_dim=8）
            ret = main(["config", "show", "--config", filepath])
            assert ret == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["state_dim"] == 8

    def test_config_command_in_help(self, capsys):
        """config 命令出现在帮助中。"""
        main([])
        captured = capsys.readouterr()
        assert "config" in captured.out


# ============================================================
# 端到端测试
# ============================================================

class TestConfigEndToEnd:
    """配置持久化端到端测试。"""

    def test_full_workflow(self):
        """完整工作流：创建→保存→加载→使用。"""
        # 1. 创建自定义配置
        config = get_default_config().merge({
            "state_dim": 12,
            "action_dim": 6,
            "seed": 12345,
        })
        # 2. 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = config.save_to_file(os.path.join(tmpdir, "custom.json"))
            # 3. 加载
            loaded = SPEPAConfig.load_from_file(filepath)
            # 4. 创建 Agent
            agent = SPEPAAgent(loaded)
            assert agent.config.state_dim == 12
            assert agent.config.action_dim == 6
            assert agent.config.seed == 12345
            # 5. 运行
            agent.run(n_steps=3)

    def test_preset_save_load(self):
        """预设配置保存加载。"""
        for preset_name in ["minimal", "balanced", "performance", "conservative"]:
            config = get_preset_config(preset_name)
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = config.save_to_file(
                    os.path.join(tmpdir, f"{preset_name}.json"))
                loaded = SPEPAConfig.load_from_file(filepath)
                assert loaded.to_dict() == config.to_dict()

    def test_config_sharing_workflow(self):
        """配置共享工作流：A 保存，B 加载。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # A 保存配置
            config_a = get_preset_config("performance")
            filepath = config_a.save_to_file(os.path.join(tmpdir, "shared.json"))
            # B 加载配置
            config_b = SPEPAConfig.load_from_file(filepath)
            # 两者一致
            assert config_b.state_dim == config_a.state_dim
            assert config_b.max_steps == config_a.max_steps
