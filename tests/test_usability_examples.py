"""SPEPA 易用性阶段五测试：示例与文档。

测试范围：
1. 示例脚本可运行
2. 示例输出正确
3. README 中的代码片段有效
"""
import os
import sys
import subprocess
import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_example(filename):
    """运行示例脚本，返回 (returncode, stdout)。"""
    filepath = os.path.join(EXAMPLES_DIR, filename)
    result = subprocess.run(
        [sys.executable, filepath],
        capture_output=True, text=True, timeout=60,
        cwd=PROJECT_ROOT,
    )
    return result.returncode, result.stdout + result.stderr


class TestExamples:
    """示例脚本测试。"""

    def test_quickstart_runs(self):
        """quickstart.py 可运行。"""
        code, output = _run_example("quickstart.py")
        assert code == 0, f"quickstart failed: {output}"
        assert "SPEPA" in output
        assert "运行结果" in output

    def test_quickstart_shows_metrics(self):
        """quickstart 显示指标。"""
        code, output = _run_example("quickstart.py")
        assert code == 0
        assert "预测误差" in output or "prediction_error" in output

    def test_autonomous_runs(self):
        """autonomous.py 可运行。"""
        code, output = _run_example("autonomous.py")
        assert code == 0, f"autonomous failed: {output}"
        assert "自主驱动" in output

    def test_autonomous_shows_goals(self):
        """autonomous 显示目标统计。"""
        code, output = _run_example("autonomous.py")
        assert code == 0
        assert "生成目标" in output or "达成目标" in output

    def test_collaboration_runs(self):
        """collaboration.py 可运行。"""
        code, output = _run_example("collaboration.py")
        assert code == 0, f"collaboration failed: {output}"
        assert "多Agent协作" in output

    def test_collaboration_shows_all_modes(self):
        """collaboration 显示三种模式。"""
        code, output = _run_example("collaboration.py")
        assert code == 0
        assert "PARALLEL" in output or "并行" in output
        assert "SEQUENTIAL" in output or "串行" in output
        assert "COMPETITIVE" in output or "竞争" in output

    def test_events_and_reports_runs(self):
        """events_and_reports.py 可运行。"""
        code, output = _run_example("events_and_reports.py")
        assert code == 0, f"events_and_reports failed: {output}"
        assert "事件回调" in output or "STEP_END" in output

    def test_events_and_reports_exports(self):
        """events_and_reports 导出报告。"""
        code, output = _run_example("events_and_reports.py")
        assert code == 0
        assert "JSON" in output or "json" in output
        assert "HTML" in output or "html" in output
        assert "CSV" in output or "csv" in output


class TestReadmeCodeSnippets:
    """README 代码片段有效性测试。"""

    def test_three_line_startup(self):
        """README 中的3行启动代码有效。"""
        from spepa import SPEPAAgent, get_preset_config
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=5)

    def test_preset_listing(self):
        """README 中的预设列出代码有效。"""
        from spepa import list_presets, get_preset_config
        presets = list_presets()
        assert len(presets) == 4
        config = get_preset_config("performance")
        assert config.state_dim == 32

    def test_event_callback_pattern(self):
        """README 中的事件回调模式有效。"""
        from spepa import SPEPAAgent, EventType, get_preset_config
        agent = SPEPAAgent(get_preset_config("minimal"))
        received = []
        agent.on(EventType.STEP_END, lambda e: received.append(e.data["step"]))
        agent.run(n_steps=3)
        assert len(received) == 3

    def test_report_export_pattern(self):
        """README 中的报告导出模式有效。"""
        from spepa import SPEPAAgent, get_preset_config
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=3)
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.generate_report("html", os.path.join(tmpdir, "report"))
            agent.generate_report("json", os.path.join(tmpdir, "report"))
            agent.generate_report("csv", os.path.join(tmpdir, "report"))

    def test_collaboration_pattern(self):
        """README 中的协作模式有效。"""
        from spepa import SPEPAAgent, get_preset_config
        from spepa.multiagent.collaboration import CollaborationMode
        import numpy as np
        agent = SPEPAAgent(get_preset_config("minimal"))
        target = np.ones(agent.config.state_dim) * 0.7
        agent.collaborate(target, mode=CollaborationMode.PARALLEL)


class TestDocumentationExists:
    """文档存在性测试。"""

    def test_readme_exists(self):
        """README.md 存在。"""
        readme = os.path.join(PROJECT_ROOT, "README.md")
        assert os.path.exists(readme)

    def test_readme_has_quickstart(self):
        """README 包含快速开始。"""
        with open(os.path.join(PROJECT_ROOT, "README.md"), encoding="utf-8") as f:
            content = f.read()
        assert "快速开始" in content or "快速" in content

    def test_readme_has_presets(self):
        """README 包含预设配置。"""
        with open(os.path.join(PROJECT_ROOT, "README.md"), encoding="utf-8") as f:
            content = f.read()
        assert "minimal" in content
        assert "balanced" in content
        assert "performance" in content

    def test_readme_has_cli(self):
        """README 包含 CLI 使用。"""
        with open(os.path.join(PROJECT_ROOT, "README.md"), encoding="utf-8") as f:
            content = f.read()
        assert "spepa" in content
        assert "run" in content

    def test_readme_has_api_reference(self):
        """README 包含 API 速查。"""
        with open(os.path.join(PROJECT_ROOT, "README.md"), encoding="utf-8") as f:
            content = f.read()
        assert "API" in content or "速查" in content

    def test_examples_directory_exists(self):
        """examples/ 目录存在。"""
        assert os.path.isdir(EXAMPLES_DIR)

    def test_all_examples_exist(self):
        """所有示例文件存在。"""
        for name in ["quickstart.py", "autonomous.py", "collaboration.py",
                      "events_and_reports.py"]:
            assert os.path.isfile(os.path.join(EXAMPLES_DIR, name)), \
                f"缺少示例: {name}"
