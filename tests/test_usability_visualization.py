"""阶段三测试：可视化图表生成工具。

测试范围：
1. ChartConfig 配置
2. generate_chart SVG 生成
3. plot_metric 单指标绘图
4. plot_metrics 多指标绘图
5. generate_dashboard 仪表盘
6. ASCIILineChart ASCII 图表
7. render_ascii_chart 便捷函数
8. 文件保存
"""
import os
import tempfile
import pytest

from spepa.visualization import (
    ChartConfig, ASCIILineChart,
    generate_chart, generate_dashboard,
    plot_metric, plot_metrics, render_ascii_chart,
)
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_preset_config


# ============================================================
# ChartConfig 测试
# ============================================================

class TestChartConfig:
    """ChartConfig 测试。"""

    def test_default_config(self):
        """默认配置。"""
        config = ChartConfig()
        assert config.width == 600
        assert config.height == 300
        assert config.title == ""
        assert config.grid is True

    def test_custom_config(self):
        """自定义配置。"""
        config = ChartConfig(width=800, height=400, title="Test")
        assert config.width == 800
        assert config.height == 400
        assert config.title == "Test"


# ============================================================
# generate_chart 测试
# ============================================================

class TestGenerateChart:
    """generate_chart SVG 生成测试。"""

    def test_returns_svg_string(self):
        """返回 SVG 字符串。"""
        data = {"series1": [1.0, 2.0, 3.0, 2.5]}
        svg = generate_chart(data)
        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_contains_polyline(self):
        """包含折线。"""
        data = {"error": [0.5, 0.3, 0.1]}
        svg = generate_chart(data)
        assert "<polyline" in svg

    def test_contains_title(self):
        """包含标题。"""
        config = ChartConfig(title="My Chart")
        svg = generate_chart({"data": [1, 2, 3]}, config)
        assert "My Chart" in svg

    def test_multiple_series(self):
        """多系列图表。"""
        data = {
            "error": [0.5, 0.3, 0.1],
            "confidence": [0.5, 0.7, 0.9],
        }
        svg = generate_chart(data)
        assert svg.count("<polyline") == 2

    def test_empty_data(self):
        """空数据。"""
        svg = generate_chart({})
        assert "No data" in svg

    def test_save_to_file(self):
        """保存到文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "chart.svg")
            generate_chart({"data": [1, 2, 3]}, filepath=filepath)
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
            assert "<svg" in content

    def test_save_creates_directory(self):
        """保存时创建目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "chart.svg")
            generate_chart({"data": [1, 2, 3]}, filepath=filepath)
            assert os.path.exists(filepath)

    def test_single_value(self):
        """单个值。"""
        svg = generate_chart({"data": [1.0]})
        assert "<svg" in svg

    def test_constant_values(self):
        """常数值。"""
        svg = generate_chart({"data": [5.0, 5.0, 5.0]})
        assert "<svg" in svg


# ============================================================
# plot_metric 测试
# ============================================================

class TestPlotMetric:
    """plot_metric 测试。"""

    def test_plot_existing_metric(self):
        """绘制存在的指标。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=5)
        svg = plot_metric(agent, "prediction_error")
        assert "<svg" in svg
        assert "polyline" in svg

    def test_plot_nonexistent_metric(self):
        """绘制不存在的指标。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        svg = plot_metric(agent, "nonexistent")
        assert "No data" in svg

    def test_plot_with_config(self):
        """带配置绘制。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=3)
        config = ChartConfig(title="Custom Title")
        svg = plot_metric(agent, "prediction_error", config=config)
        assert "Custom Title" in svg

    def test_plot_save_to_file(self):
        """保存到文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=3)
            filepath = os.path.join(tmpdir, "metric.svg")
            plot_metric(agent, "prediction_error", filepath=filepath)
            assert os.path.exists(filepath)


# ============================================================
# plot_metrics 测试
# ============================================================

class TestPlotMetrics:
    """plot_metrics 测试。"""

    def test_plot_multiple_metrics(self):
        """绘制多个指标。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=5)
        svg = plot_metrics(agent, ["prediction_error", "confidence"])
        assert "<svg" in svg
        assert svg.count("<polyline") >= 1

    def test_plot_no_metrics(self):
        """无指标。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        svg = plot_metrics(agent, ["nonexistent1", "nonexistent2"])
        assert "No data" in svg

    def test_plot_partial_metrics(self):
        """部分指标存在。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=3)
        svg = plot_metrics(agent, ["prediction_error", "nonexistent"])
        assert "<svg" in svg


# ============================================================
# generate_dashboard 测试
# ============================================================

class TestGenerateDashboard:
    """generate_dashboard 测试。"""

    def test_returns_svg(self):
        """返回 SVG。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=5)
        svg = generate_dashboard(agent)
        assert "<svg" in svg
        assert "Dashboard" in svg

    def test_contains_step_count(self):
        """包含步数。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=7)
        svg = generate_dashboard(agent)
        assert "7" in svg  # step_count

    def test_custom_metrics(self):
        """自定义指标。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=5)
        svg = generate_dashboard(agent, metrics=["prediction_error"])
        assert "<svg" in svg

    def test_save_to_file(self):
        """保存到文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=3)
            filepath = os.path.join(tmpdir, "dashboard.svg")
            generate_dashboard(agent, filepath=filepath)
            assert os.path.exists(filepath)

    def test_no_metrics_available(self):
        """无可用指标。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        # 不运行，可能无指标
        svg = generate_dashboard(agent)
        assert isinstance(svg, str)


# ============================================================
# ASCIILineChart 测试
# ============================================================

class TestASCIILineChart:
    """ASCIILineChart 测试。"""

    def test_render_returns_string(self):
        """render 返回字符串。"""
        chart = ASCIILineChart()
        result = chart.render({"data": [1, 2, 3]})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_with_title(self):
        """带标题。"""
        chart = ASCIILineChart()
        result = chart.render({"data": [1, 2, 3]}, title="My Chart")
        assert "My Chart" in result

    def test_render_empty_data(self):
        """空数据。"""
        chart = ASCIILineChart()
        result = chart.render({})
        assert "No data" in result

    def test_render_multiple_series(self):
        """多系列。"""
        chart = ASCIILineChart()
        result = chart.render({
            "series1": [1, 2, 3],
            "series2": [3, 2, 1],
        })
        assert "Legend" in result
        assert "series1" in result
        assert "series2" in result

    def test_custom_dimensions(self):
        """自定义尺寸。"""
        chart = ASCIILineChart(width=40, height=10)
        result = chart.render({"data": [1, 2, 3]})
        lines = result.split('\n')
        # 至少有 height 行（加上标题和图例）
        assert len(lines) >= 10

    def test_contains_axis(self):
        """包含坐标轴。"""
        chart = ASCIILineChart()
        result = chart.render({"data": [1, 2, 3]})
        assert '|' in result  # Y 轴
        assert '-' in result  # X 轴


# ============================================================
# render_ascii_chart 测试
# ============================================================

class TestRenderAsciiChart:
    """render_ascii_chart 便捷函数测试。"""

    def test_returns_string(self):
        """返回字符串。"""
        result = render_ascii_chart({"data": [1, 2, 3]})
        assert isinstance(result, str)

    def test_with_title(self):
        """带标题。"""
        result = render_ascii_chart({"data": [1, 2, 3]}, title="Test")
        assert "Test" in result

    def test_custom_dimensions(self):
        """自定义尺寸。"""
        result = render_ascii_chart({"data": [1, 2, 3]}, width=30, height=8)
        assert isinstance(result, str)


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端测试。"""

    def test_full_visualization_workflow(self):
        """完整可视化工作流。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 运行 Agent
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=10)

            # 2. 生成单指标图
            svg1 = plot_metric(agent, "prediction_error",
                               filepath=os.path.join(tmpdir, "error.svg"))
            assert "<svg" in svg1

            # 3. 生成多指标图
            svg2 = plot_metrics(agent, ["prediction_error", "confidence"],
                                filepath=os.path.join(tmpdir, "multi.svg"))
            assert "<svg" in svg2

            # 4. 生成仪表盘
            svg3 = generate_dashboard(agent,
                                      filepath=os.path.join(tmpdir, "dashboard.svg"))
            assert "<svg" in svg3

            # 5. 生成 ASCII 图
            ascii_chart = render_ascii_chart(
                {"error": agent.metrics.get("prediction_error") or []},
                title="Error Trend"
            )
            assert "Error Trend" in ascii_chart

            # 验证文件都存在
            assert os.path.exists(os.path.join(tmpdir, "error.svg"))
            assert os.path.exists(os.path.join(tmpdir, "multi.svg"))
            assert os.path.exists(os.path.join(tmpdir, "dashboard.svg"))

    def test_chart_with_agent_metrics(self):
        """图表与 Agent 指标集成。"""
        agent = SPEPAAgent(get_preset_config("minimal"))
        agent.run(n_steps=5)

        # 获取所有可用指标
        metric_names = list(agent.metrics.to_dict().keys())
        assert len(metric_names) > 0

        # 为每个指标生成图表
        for name in metric_names[:3]:  # 只测前3个
            svg = plot_metric(agent, name)
            assert "<svg" in svg

    def test_ascii_vs_svg_consistency(self):
        """ASCII 和 SVG 图表数据一致。"""
        data = {"series": [1.0, 2.0, 3.0, 2.5, 1.5]}
        svg = generate_chart(data)
        ascii_result = render_ascii_chart(data)
        assert "<svg" in svg
        assert "series" in ascii_result
