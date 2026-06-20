"""SPEPA 可视化图表生成工具。

设计原则（第一性原理）：
- 零依赖：仅使用 Python 标准库生成 SVG 图表
- 自动化：从 Agent 指标自动生成时间序列图
- 多格式：支持 SVG（矢量）和 ASCII（终端）两种输出
- 可组合：多个图表可组合为仪表盘

主要功能：
- generate_chart() : 生成单个 SVG 图表
- generate_dashboard() : 生成多图仪表盘
- plot_metric() : 绘制单个指标的时间序列
- plot_metrics() : 绘制多个指标的对比图
- ASCIILineChart : ASCII 折线图（终端友好）
"""
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

from spepa.utils.logger import get_logger

logger = get_logger("spepa.visualization")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ChartConfig:
    """图表配置。"""
    width: int = 600
    height: int = 300
    margin: int = 40
    title: str = ""
    x_label: str = "Step"
    y_label: str = "Value"
    color: str = "#2196F3"
    grid: bool = True
    line_width: float = 2.0


# ============================================================
# SVG 图表生成
# ============================================================

def _svg_escape(text: str) -> str:
    """转义 SVG 文本中的特殊字符。"""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))


def _scale_values(values: List[float],
                  target_min: float, target_max: float
                  ) -> List[float]:
    """将值列表缩放到目标范围。"""
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [target_min + (target_max - target_min) / 2] * len(values)
    return [
        target_min + (v - vmin) / (vmax - vmin) * (target_max - target_min)
        for v in values
    ]


def generate_chart(data: Dict[str, List[float]],
                   config: Optional[ChartConfig] = None,
                   filepath: Optional[str] = None) -> str:
    """生成 SVG 折线图。

    参数：
        data : {series_name: values} 字典，每个系列一条线
        config : 图表配置（None=默认）
        filepath : 保存路径（None=不保存，仅返回 SVG 字符串）

    返回：
        SVG 字符串

    示例：
        >>> svg = generate_chart({"error": [0.5, 0.3, 0.1]}, filepath="chart.svg")
    """
    if config is None:
        config = ChartConfig()

    if not data:
        return _empty_svg(config)

    # 颜色循环
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    w, h = config.width, config.height
    m = config.margin
    plot_w = w - 2 * m
    plot_h = h - 2 * m

    # 计算全局 Y 范围
    all_values = []
    for vals in data.values():
        all_values.extend(vals)
    if not all_values:
        return _empty_svg(config)

    y_min, y_max = min(all_values), max(all_values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    # 计算最大 X 长度
    max_len = max(len(v) for v in data.values())
    if max_len == 0:
        return _empty_svg(config)

    # 构建 SVG
    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">')

    # 背景
    lines.append(f'<rect width="{w}" height="{h}" fill="white"/>')

    # 标题
    if config.title:
        lines.append(f'<text x="{w/2}" y="20" text-anchor="middle" '
                     f'font-size="14" font-weight="bold" fill="#333">'
                     f'{_svg_escape(config.title)}</text>')

    # 网格
    if config.grid:
        for i in range(5):
            y = m + i * plot_h / 4
            lines.append(f'<line x1="{m}" y1="{y}" x2="{w-m}" y2="{y}" '
                        f'stroke="#e0e0e0" stroke-width="1"/>')
        for i in range(5):
            x = m + i * plot_w / 4
            lines.append(f'<line x1="{x}" y1="{m}" x2="{x}" y2="{h-m}" '
                        f'stroke="#e0e0e0" stroke-width="1"/>')

    # 坐标轴
    lines.append(f'<line x1="{m}" y1="{h-m}" x2="{w-m}" y2="{h-m}" '
                f'stroke="#333" stroke-width="1"/>')
    lines.append(f'<line x1="{m}" y1="{m}" x2="{m}" y2="{h-m}" '
                f'stroke="#333" stroke-width="1"/>')

    # Y 轴标签
    for i in range(5):
        y = m + i * plot_h / 4
        val = y_max - (y_max - y_min) * i / 4
        lines.append(f'<text x="{m-5}" y="{y+4}" text-anchor="end" '
                    f'font-size="10" fill="#666">{val:.3g}</text>')

    # X 轴标签
    for i in range(5):
        x = m + i * plot_w / 4
        val = int(max_len * i / 4)
        lines.append(f'<text x="{x}" y="{h-m+15}" text-anchor="middle" '
                    f'font-size="10" fill="#666">{val}</text>')

    # 轴标题
    if config.x_label:
        lines.append(f'<text x="{w/2}" y="{h-5}" text-anchor="middle" '
                    f'font-size="11" fill="#333">{_svg_escape(config.x_label)}</text>')
    if config.y_label:
        lines.append(f'<text x="15" y="{h/2}" text-anchor="middle" '
                    f'font-size="11" fill="#333" transform="rotate(-90 15 {h/2})">'
                    f'{_svg_escape(config.y_label)}</text>')

    # 数据线
    for idx, (name, values) in enumerate(data.items()):
        color = colors[idx % len(colors)]
        if not values:
            continue
        # 缩放 Y 值到绘图区域
        scaled_y = [
            (h - m) - (v - y_min) / (y_max - y_min) * plot_h
            for v in values
        ]
        # 缩放 X 值
        scaled_x = [
            m + i / max(max_len - 1, 1) * plot_w
            for i in range(len(values))
        ]
        # 生成路径
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(scaled_x, scaled_y))
        lines.append(f'<polyline points="{points}" fill="none" '
                    f'stroke="{color}" stroke-width="{config.line_width}"/>')

        # 图例
        legend_x = w - m - 80
        legend_y = m + 15 + idx * 18
        lines.append(f'<rect x="{legend_x}" y="{legend_y-10}" width="10" height="10" '
                    f'fill="{color}"/>')
        lines.append(f'<text x="{legend_x+15}" y="{legend_y-1}" '
                    f'font-size="10" fill="#333">{_svg_escape(name)}</text>')

    lines.append('</svg>')
    svg = "\n".join(lines)

    # 保存到文件
    if filepath:
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg)
        logger.info("Chart saved to %s", filepath)

    return svg


def _empty_svg(config: ChartConfig) -> str:
    """生成空图表。"""
    w, h = config.width, config.height
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
            f'<rect width="{w}" height="{h}" fill="white"/>'
            f'<text x="{w/2}" y="{h/2}" text-anchor="middle" fill="#999">'
            f'No data</text></svg>')


# ============================================================
# 从 Agent 指标生成图表
# ============================================================

def plot_metric(agent, metric_name: str,
                config: Optional[ChartConfig] = None,
                filepath: Optional[str] = None) -> str:
    """绘制单个指标的时间序列图。

    参数：
        agent : SPEPAAgent 实例
        metric_name : 指标名称（如 "prediction_error"）
        config : 图表配置
        filepath : 保存路径

    返回：
        SVG 字符串
    """
    values = agent.metrics.get(metric_name)
    if values is None:
        logger.warning("Metric '%s' not found", metric_name)
        return _empty_svg(config or ChartConfig())

    if config is None:
        config = ChartConfig(title=metric_name, y_label=metric_name)

    return generate_chart({metric_name: values}, config, filepath)


def plot_metrics(agent, metric_names: List[str],
                 config: Optional[ChartConfig] = None,
                 filepath: Optional[str] = None) -> str:
    """绘制多个指标的对比图。

    参数：
        agent : SPEPAAgent 实例
        metric_names : 指标名称列表
        config : 图表配置
        filepath : 保存路径

    返回：
        SVG 字符串
    """
    data = {}
    for name in metric_names:
        values = agent.metrics.get(name)
        if values:
            data[name] = values

    if not data:
        logger.warning("No metrics found: %s", metric_names)
        return _empty_svg(config or ChartConfig())

    if config is None:
        config = ChartConfig(title="Metrics Comparison", y_label="Value")

    return generate_chart(data, config, filepath)


def generate_dashboard(agent, filepath: Optional[str] = None,
                       metrics: Optional[List[str]] = None) -> str:
    """生成多图仪表盘。

    参数：
        agent : SPEPAAgent 实例
        filepath : 保存路径
        metrics : 要绘制的指标列表（None=默认选择）

    返回：
        SVG 字符串
    """
    if metrics is None:
        # 默认选择关键指标
        available = set(agent.metrics.to_dict().keys())
        preferred = ["prediction_error", "confidence", "total_reward",
                     "exploration_ratio", "cpu_usage", "cycle_time"]
        metrics = [m for m in preferred if m in available]
        if not metrics:
            metrics = list(available)[:4]

    if not metrics:
        return _empty_svg(ChartConfig())

    # 每行2个图表
    n = len(metrics)
    cols = 2
    rows = (n + cols - 1) // cols
    chart_w, chart_h = 400, 250
    padding = 20
    total_w = cols * chart_w + (cols + 1) * padding
    total_h = rows * chart_h + (rows + 1) * padding + 40  # +40 for title

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'width="{total_w}" height="{total_h}" '
                 f'viewBox="0 0 {total_w} {total_h}">')
    lines.append(f'<rect width="{total_w}" height="{total_h}" fill="#f5f5f5"/>')

    # 标题
    lines.append(f'<text x="{total_w/2}" y="25" text-anchor="middle" '
                 f'font-size="16" font-weight="bold" fill="#333">'
                 f'SPEPA Dashboard (step={agent.step_count})</text>')

    # 各子图
    for idx, metric_name in enumerate(metrics):
        row = idx // cols
        col = idx % cols
        x_offset = padding + col * (chart_w + padding)
        y_offset = 40 + padding + row * (chart_h + padding)

        values = agent.metrics.get(metric_name)
        if not values:
            continue

        # 生成子图 SVG
        sub_config = ChartConfig(
            width=chart_w, height=chart_h,
            title=metric_name, y_label=metric_name,
        )
        sub_svg = generate_chart({metric_name: values}, sub_config)

        # 提取子图内容（去掉外层 svg 标签）
        sub_content = sub_svg
        sub_content = sub_content.split('>', 1)[1] if '>' in sub_content else sub_content
        sub_content = sub_content.rsplit('</svg>', 1)[0]

        # 平移子图
        lines.append(f'<g transform="translate({x_offset},{y_offset})">')
        lines.append(sub_content)
        lines.append('</g>')

    lines.append('</svg>')
    svg = "\n".join(lines)

    if filepath:
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg)
        logger.info("Dashboard saved to %s", filepath)

    return svg


# ============================================================
# ASCII 折线图（终端友好）
# ============================================================

class ASCIILineChart:
    """ASCII 折线图生成器（终端友好）。

    使用方式：
        chart = ASCIILineChart(width=60, height=15)
        text = chart.render({"error": [0.5, 0.3, 0.1]})
        print(text)
    """

    def __init__(self, width: int = 60, height: int = 15):
        """初始化 ASCII 图表。

        参数：
            width : 图表宽度（字符数）
            height : 图表高度（行数）
        """
        self.width = max(20, width)
        self.height = max(8, height)

    def render(self, data: Dict[str, List[float]],
               title: str = "") -> str:
        """渲染 ASCII 图表。

        参数：
            data : {series_name: values} 字典
            title : 图表标题

        返回：
            ASCII 图表字符串
        """
        if not data:
            return "[No data]"

        lines = []
        if title:
            lines.append(title)
            lines.append("=" * self.width)

        # 计算全局范围
        all_values = []
        max_len = 0
        for vals in data.values():
            all_values.extend(vals)
            max_len = max(max_len, len(vals))

        if not all_values or max_len == 0:
            lines.append("[No data]")
            return "\n".join(lines)

        y_min, y_max = min(all_values), max(all_values)
        if y_min == y_max:
            y_min -= 1
            y_max += 1

        # 采样点到字符位置
        symbols = ['*', '+', 'o', 'x', '#']
        colors_code = [''] * len(data)  # 简单符号区分

        # 构建网格
        grid = [[' '] * self.width for _ in range(self.height)]

        # 绘制坐标轴
        for i in range(self.height):
            grid[i][0] = '|'
        for j in range(self.width):
            grid[self.height - 1][j] = '-'
        grid[self.height - 1][0] = '+'

        # 绘制数据
        for idx, (name, values) in enumerate(data.items()):
            symbol = symbols[idx % len(symbols)]
            for i, v in enumerate(values):
                x = int(i / max(max_len - 1, 1) * (self.width - 2)) + 1
                y = int((y_max - v) / (y_max - y_min) * (self.height - 2))
                y = min(max(y, 0), self.height - 2)
                grid[y][x] = symbol

        # Y 轴标签
        lines.append(f"{y_max:.4g} |")
        for row in grid[1:-1]:
            lines.append("      |" + "".join(row[1:]))
        lines.append(f"{y_min:.4g} |" + "-" * (self.width - 7))
        lines.append(" " * 7 + "0" + " " * (self.width - 10) + f"{max_len}")

        # 图例
        legend_parts = []
        for idx, name in enumerate(data.keys()):
            symbol = symbols[idx % len(symbols)]
            legend_parts.append(f"{symbol} {name}")
        lines.append("  Legend: " + "  ".join(legend_parts))

        return "\n".join(lines)


def render_ascii_chart(data: Dict[str, List[float]],
                       width: int = 60, height: int = 15,
                       title: str = "") -> str:
    """便捷函数：渲染 ASCII 图表。

    参数：
        data : {series_name: values} 字典
        width : 宽度
        height : 高度
        title : 标题

    返回：
        ASCII 图表字符串
    """
    chart = ASCIILineChart(width=width, height=height)
    return chart.render(data, title=title)
