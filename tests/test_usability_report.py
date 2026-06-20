"""SPEPA 易用性阶段四测试：可视化与报告导出。

测试范围：
1. ASCIITextChart：ASCII 图表生成
2. ReportGenerator：JSON/HTML/CSV 报告生成
3. Agent 集成：generate_report 方法
4. 文件保存：报告保存到文件
"""
import os
import json
import pytest
import tempfile

from spepa.report import ReportGenerator, ASCIITextChart
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_default_config


# ============================================================
# ASCIITextChart 测试
# ============================================================

class TestASCIITextChart:
    """ASCII 图表测试。"""

    def test_line_chart_basic(self):
        """基本折线图。"""
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        chart = ASCIITextChart.line_chart(values, title="测试")
        assert "测试" in chart
        assert "*" in chart

    def test_line_chart_empty(self):
        """空数据。"""
        chart = ASCIITextChart.line_chart([], title="空")
        assert "无数据" in chart

    def test_line_chart_constant(self):
        """常量数据。"""
        values = [0.5, 0.5, 0.5, 0.5]
        chart = ASCIITextChart.line_chart(values, title="常量")
        assert "常量" in chart

    def test_line_chart_large_dataset(self):
        """大数据集采样。"""
        values = [float(i) / 100 for i in range(200)]
        chart = ASCIITextChart.line_chart(values, width=30, title="大数据")
        assert "大数据" in chart
        assert "200" in chart  # 步数显示

    def test_sparkline_basic(self):
        """基本 sparkline。"""
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        spark = ASCIITextChart.sparkline(values)
        assert len(spark) == 5
        assert all(c in "▁▂▃▄▅▆▇█" for c in spark)

    def test_sparkline_empty(self):
        """空 sparkline。"""
        assert ASCIITextChart.sparkline([]) == ""

    def test_sparkline_constant(self):
        """常量 sparkline。"""
        spark = ASCIITextChart.sparkline([0.5, 0.5, 0.5])
        assert all(c == "█" for c in spark)

    def test_bar_chart(self):
        """进度条。"""
        bar = ASCIITextChart.bar_chart("指标", 0.5, max_val=1.0, width=10)
        assert "指标" in bar
        assert "█" in bar
        assert "0.5000" in bar

    def test_bar_chart_overflow(self):
        """超出最大值的进度条。"""
        bar = ASCIITextChart.bar_chart("溢出", 1.5, max_val=1.0, width=10)
        assert "█" in bar


# ============================================================
# ReportGenerator 测试
# ============================================================

class TestReportGenerator:
    """报告生成器测试。"""

    def test_initialization(self):
        """正确初始化。"""
        gen = ReportGenerator()
        assert gen is not None

    def test_generate_json_report(self):
        """生成 JSON 报告。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        report_str = gen = ReportGenerator().generate_json_report(agent)
        report = json.loads(report_str)
        assert "timestamp" in report
        assert "overview" in report
        assert "config" in report
        assert "metrics_summary" in report
        assert report["overview"]["total_steps"] == 3

    def test_generate_html_report(self):
        """生成 HTML 报告。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        html = ReportGenerator().generate_html_report(agent)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "SPEPA" in html
        assert "运行概览" in html
        assert "系统配置" in html

    def test_generate_csv_report(self):
        """生成 CSV 报告。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        csv = ReportGenerator().generate_csv_report(agent)
        lines = csv.strip().split('\n')
        assert "metric_name,mean,std,min,max,last,count" in lines[0]
        assert len(lines) > 1

    def test_json_report_with_run_summary(self):
        """带 run_summary 的 JSON 报告。"""
        agent = SPEPAAgent(get_default_config())
        summary = agent.run(n_steps=3)
        report_str = ReportGenerator().generate_json_report(agent, summary)
        report = json.loads(report_str)
        assert "total_time" in report["overview"]

    def test_html_report_contains_charts(self):
        """HTML 报告包含图表。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=5)
        html = ReportGenerator().generate_html_report(agent)
        assert "chart" in html.lower() or "趋势" in html

    def test_html_report_contains_health(self):
        """HTML 报告包含健康状态。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        html = ReportGenerator().generate_html_report(agent)
        assert "健康" in html

    def test_save_report_json(self):
        """保存 JSON 报告到文件。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = ReportGenerator().save_report(
                agent, os.path.join(tmpdir, "report"), "json")
            assert filepath.endswith(".json")
            assert os.path.exists(filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert "overview" in data

    def test_save_report_html(self):
        """保存 HTML 报告到文件。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = ReportGenerator().save_report(
                agent, os.path.join(tmpdir, "report"), "html")
            assert filepath.endswith(".html")
            assert os.path.exists(filepath)

    def test_save_report_csv(self):
        """保存 CSV 报告到文件。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = ReportGenerator().save_report(
                agent, os.path.join(tmpdir, "report"), "csv")
            assert filepath.endswith(".csv")
            assert os.path.exists(filepath)

    def test_save_report_creates_directory(self):
        """保存报告时创建目录。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "dir", "report")
            filepath = ReportGenerator().save_report(
                agent, nested, "json")
            assert os.path.exists(filepath)

    def test_save_report_invalid_format(self):
        """非法格式报错。"""
        agent = SPEPAAgent(get_default_config())
        with pytest.raises(ValueError, match="不支持"):
            ReportGenerator().save_report(agent, "/tmp/test", "xml")


# ============================================================
# Agent 集成测试
# ============================================================

class TestAgentReportIntegration:
    """Agent 报告集成测试。"""

    def test_agent_generate_report_json(self):
        """Agent.generate_report JSON。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        report = agent.generate_report("json")
        data = json.loads(report)
        assert "overview" in data

    def test_agent_generate_report_html(self):
        """Agent.generate_report HTML。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        html = agent.generate_report("html")
        assert "<html" in html

    def test_agent_generate_report_csv(self):
        """Agent.generate_report CSV。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        csv = agent.generate_report("csv")
        assert "metric_name" in csv

    def test_agent_generate_report_to_file(self):
        """Agent.generate_report 保存到文件。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = agent.generate_report("json", os.path.join(tmpdir, "r"))
            assert os.path.exists(filepath)

    def test_agent_generate_report_invalid_format(self):
        """非法格式报错。"""
        agent = SPEPAAgent(get_default_config())
        with pytest.raises(ValueError, match="不支持"):
            agent.generate_report("xml")


# ============================================================
# 端到端测试
# ============================================================

class TestReportEndToEnd:
    """报告端到端测试。"""

    def test_full_run_then_report(self):
        """完整运行后生成报告。"""
        agent = SPEPAAgent(get_default_config())
        summary = agent.run(n_steps=10)
        # JSON
        json_report = agent.generate_report("json", run_summary=summary)
        data = json.loads(json_report)
        assert data["overview"]["total_steps"] == 10
        # HTML
        html_report = agent.generate_report("html", run_summary=summary)
        assert "10" in html_report

    def test_report_after_autonomous_run(self):
        """自主运行后生成报告。"""
        agent = SPEPAAgent(get_default_config())
        agent.run_autonomous(n_steps=10)
        report = agent.generate_report("json")
        data = json.loads(report)
        assert data["overview"]["total_steps"] > 0

    def test_report_contains_all_sections(self):
        """报告包含所有部分。"""
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=5)
        html = agent.generate_report("html")
        assert "运行概览" in html
        assert "系统配置" in html
        assert "指标摘要" in html or "指标" in html

    def test_csv_importable(self):
        """CSV 可被解析。"""
        import csv as csv_mod
        import io
        agent = SPEPAAgent(get_default_config())
        agent.run(n_steps=3)
        csv_str = agent.generate_report("csv")
        reader = csv_mod.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) > 0
        assert "metric_name" in rows[0]
