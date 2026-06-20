"""SPEPA 易用性阶段二测试：CLI 命令行接口。

测试范围：
1. 参数解析：各命令的参数正确解析
2. 命令执行：run/autonomous/perpetual/collaborate/presets/info
3. 预设集成：--preset 选项
4. 错误处理：非法参数的友好报错
5. 输出格式：JSON 输出正确
"""
import sys
import json
import pytest
from io import StringIO

from spepa.cli import main, build_parser


class TestCLIParser:
    """CLI 参数解析测试。"""

    def test_no_command_shows_help(self, capsys):
        """无命令时显示帮助。"""
        ret = main([])
        assert ret == 0
        captured = capsys.readouterr()
        assert "可用命令" in captured.out or "usage" in captured.out.lower()

    def test_run_command_parsed(self):
        """run 命令参数解析。"""
        parser = build_parser()
        args = parser.parse_args(["run", "--steps", "10", "--seed", "123"])
        assert args.command == "run"
        assert args.steps == 10
        assert args.seed == 123

    def test_autonomous_command_parsed(self):
        """autonomous 命令参数解析。"""
        parser = build_parser()
        args = parser.parse_args(["autonomous", "-n", "50"])
        assert args.command == "autonomous"
        assert args.steps == 50

    def test_perpetual_command_parsed(self):
        """perpetual 命令参数解析。"""
        parser = build_parser()
        args = parser.parse_args(["perpetual", "--cycles", "5", "--no-recover"])
        assert args.command == "perpetual"
        assert args.cycles == 5
        assert args.no_recover is True

    def test_collaborate_command_parsed(self):
        """collaborate 命令参数解析。"""
        parser = build_parser()
        args = parser.parse_args([
            "collaborate", "--mode", "competitive",
            "--target", "0.9", "--competitors", "2"
        ])
        assert args.command == "collaborate"
        assert args.mode == "competitive"
        assert args.target == 0.9
        assert args.competitors == 2

    def test_preset_option_parsed(self):
        """--preset 选项解析。"""
        parser = build_parser()
        args = parser.parse_args(["run", "--preset", "minimal"])
        assert args.preset == "minimal"

    def test_quiet_option_parsed(self):
        """--quiet 选项解析。"""
        parser = build_parser()
        args = parser.parse_args(["run", "--quiet"])
        assert args.quiet is True


class TestCLICommands:
    """CLI 命令执行测试。"""

    def test_run_command(self, capsys):
        """run 命令执行。"""
        ret = main(["run", "--steps", "3", "--preset", "minimal", "--quiet"])
        assert ret == 0

    def test_run_command_with_output(self, capsys):
        """run 命令带输出。"""
        ret = main(["run", "--steps", "3", "--preset", "minimal"])
        assert ret == 0
        captured = capsys.readouterr()
        # 应有 JSON 输出
        assert "total_steps" in captured.out or "online_cycles" in captured.out

    def test_autonomous_command(self, capsys):
        """autonomous 命令执行。"""
        ret = main(["autonomous", "--steps", "5", "--preset", "minimal", "--quiet"])
        assert ret == 0

    def test_perpetual_command(self, capsys):
        """perpetual 命令执行。"""
        ret = main(["perpetual", "--cycles", "3", "--preset", "minimal", "--quiet"])
        assert ret == 0

    def test_collaborate_command(self, capsys):
        """collaborate 命令执行。"""
        ret = main([
            "collaborate", "--mode", "parallel",
            "--target", "0.7", "--preset", "minimal", "--quiet"
        ])
        assert ret == 0

    def test_collaborate_competitive(self, capsys):
        """collaborate competitive 模式。"""
        ret = main([
            "collaborate", "--mode", "competitive",
            "--competitors", "2", "--preset", "minimal", "--quiet"
        ])
        assert ret == 0

    def test_presets_command(self, capsys):
        """presets 命令列出预设。"""
        ret = main(["presets"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "minimal" in captured.out
        assert "balanced" in captured.out
        assert "performance" in captured.out
        assert "conservative" in captured.out

    def test_info_command(self, capsys):
        """info 命令显示系统信息。"""
        ret = main(["info", "--preset", "minimal"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "SPEPA" in captured.out
        assert "state_dim" in captured.out
        assert "minimal" in captured.out

    def test_info_default_preset(self, capsys):
        """info 默认 balanced。"""
        ret = main(["info"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "balanced" in captured.out


class TestCLIErrorHandling:
    """CLI 错误处理测试。"""

    def test_invalid_preset_returns_error(self, capsys):
        """非法预设返回错误码。"""
        ret = main(["run", "--preset", "nonexistent", "--steps", "1"])
        assert ret == 1
        captured = capsys.readouterr()
        assert "错误" in captured.err or "未知预设" in captured.err

    def test_invalid_mode_choice(self):
        """非法模式选择被 argparse 拒绝。"""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["collaborate", "--mode", "invalid"])


class TestCLIUsability:
    """CLI 易用性测试。"""

    def test_minimal_invocation(self, capsys):
        """最小调用：spepa presets。"""
        ret = main(["presets"])
        assert ret == 0

    def test_help_shows_commands(self, capsys):
        """帮助显示所有命令。"""
        ret = main([])
        assert ret == 0
        captured = capsys.readouterr()
        for cmd in ["run", "autonomous", "perpetual", "collaborate", "presets", "info"]:
            assert cmd in captured.out

    def test_seed_override(self, capsys):
        """--seed 覆盖配置种子。"""
        ret = main(["run", "--steps", "2", "--preset", "minimal", "--seed", "99", "--quiet"])
        assert ret == 0

    def test_state_dim_override(self, capsys):
        """--state-dim 覆盖维度。"""
        ret = main([
            "run", "--steps", "2", "--preset", "minimal",
            "--state-dim", "4", "--quiet"
        ])
        assert ret == 0
