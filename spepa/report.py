"""SPEPA 报告生成与可视化。

设计原则（第一性原理）：
- 自描述原理：报告应包含足够上下文，无需额外文档即可理解
- 零依赖原理：核心报告功能不依赖第三方库（matplotlib 可选）
- 多格式原理：支持 JSON（机器可读）/ HTML（人可读）/ CSV（表格）

报告内容：
1. 系统配置摘要
2. 运行统计（步数、周期、耗时）
3. 指标趋势（ASCII 图表 + 数值摘要）
4. 健康状态
5. 事件日志（可选）

可视化：
- ASCII 折线图：不依赖 matplotlib，适合终端输出
- matplotlib 图表：可选，当 matplotlib 可用时生成 PNG
"""
import json
import time
import os
from typing import Dict, Any, List, Optional

import numpy as np

from spepa.utils.logger import LoggedMixin


class ASCIITextChart:
    """ASCII 文本图表生成器（零依赖）。"""

    @staticmethod
    def line_chart(values: List[float], width: int = 50, height: int = 10,
                   title: str = "") -> str:
        """生成 ASCII 折线图。

        参数：
        - values: 数值列表
        - width: 图表宽度（字符数）
        - height: 图表高度（行数）
        - title: 图表标题

        返回：ASCII 图表字符串
        """
        if not values:
            return f"{title}: (无数据)"
        values = [float(v) for v in values if np.isfinite(v)]
        if not values:
            return f"{title}: (无有效数据)"
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val
        if val_range < 1e-10:
            val_range = 1.0
        # 采样到 width 个点
        n = len(values)
        if n > width:
            indices = np.linspace(0, n - 1, width).astype(int)
            sampled = [values[i] for i in indices]
        else:
            sampled = values
        # 生成网格
        grid = [[' ' for _ in range(len(sampled))] for _ in range(height)]
        for col, val in enumerate(sampled):
            # 归一化到 [0, height-1]
            normalized = (val - min_val) / val_range
            row = int((height - 1) * (1 - normalized))
            row = max(0, min(height - 1, row))
            grid[row][col] = '*'
            # 填充下方
            for r in range(row + 1, height):
                if grid[r][col] == ' ':
                    grid[r][col] = '.'
        # 构建输出
        lines = []
        if title:
            lines.append(title)
        lines.append(f"  {max_val:.4f} |" + ''.join(grid[0]))
        for i in range(1, height - 1):
            lines.append(f"         |" + ''.join(grid[i]))
        lines.append(f"  {min_val:.4f} |" + ''.join(grid[-1]))
        lines.append(f"         +" + '-' * len(sampled))
        lines.append(f"         步数: {n}")
        return '\n'.join(lines)

    @staticmethod
    def sparkline(values: List[float]) -> str:
        """生成单行迷你图（sparkline）。

        使用 Unicode 块字符：▁▂▃▄▅▆▇█
        """
        if not values:
            return ""
        values = [float(v) for v in values if np.isfinite(v)]
        if not values:
            return ""
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val
        if val_range < 1e-10:
            return "█" * len(values)
        blocks = "▁▂▃▄▅▆▇█"
        result = []
        for v in values:
            normalized = (v - min_val) / val_range
            idx = min(7, max(0, int(normalized * 7.999)))
            result.append(blocks[idx])
        return ''.join(result)

    @staticmethod
    def bar_chart(label: str, value: float, max_val: float = 1.0,
                  width: int = 30) -> str:
        """生成单行进度条。"""
        if max_val <= 0:
            max_val = 1.0
        ratio = max(0.0, min(1.0, value / max_val))
        filled = int(ratio * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"{label:20s} |{bar}| {value:.4f}"


class ReportGenerator(LoggedMixin):
    """报告生成器：将 Agent 运行结果导出为多种格式。

    支持格式：
    - JSON：完整结构化数据，机器可读
    - HTML：带样式的可视化报告，人可读
    - CSV：指标表格，适合导入 Excel
    """

    def __init__(self):
        self._set_log_name("spepa.report")
        self.log.debug("ReportGenerator initialized")

    def generate_json_report(self, agent, run_summary: Optional[Dict] = None) -> str:
        """生成 JSON 格式报告。

        参数：
        - agent: SPEPAAgent 实例
        - run_summary: run() 返回的摘要（可选）

        返回：JSON 字符串
        """
        report = self._collect_report_data(agent, run_summary)
        return json.dumps(report, indent=2, default=str, ensure_ascii=False)

    def generate_html_report(self, agent, run_summary: Optional[Dict] = None) -> str:
        """生成 HTML 格式报告。

        参数：
        - agent: SPEPAAgent 实例
        - run_summary: run() 返回的摘要（可选）

        返回：HTML 字符串
        """
        data = self._collect_report_data(agent, run_summary)
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='zh-CN'>",
            "<head>",
            "<meta charset='UTF-8'>",
            f"<title>SPEPA 运行报告 - {data['timestamp']}</title>",
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; "
            "background: #f5f5f5; color: #333; }",
            ".container { max-width: 900px; margin: 0 auto; background: white; "
            "padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }",
            "h2 { color: #34495e; margin-top: 30px; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }",
            "th { background: #3498db; color: white; }",
            "tr:nth-child(even) { background: #f9f9f9; }",
            ".metric-card { display: inline-block; background: #ecf0f1; "
            "padding: 15px; margin: 5px; border-radius: 5px; min-width: 120px; }",
            ".metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }",
            ".metric-label { font-size: 12px; color: #7f8c8d; }",
            ".health-good { color: #27ae60; }",
            ".health-warn { color: #f39c12; }",
            ".health-bad { color: #e74c3c; }",
            ".chart { background: #1e1e1e; color: #0f0; padding: 15px; "
            "font-family: monospace; font-size: 12px; border-radius: 5px; "
            "overflow-x: auto; white-space: pre; }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
            f"<h1>SPEPA 运行报告</h1>",
            f"<p>生成时间: {data['timestamp']}</p>",
        ]
        # 概览卡片
        overview = data.get("overview", {})
        html_parts.append("<h2>运行概览</h2>")
        for key, val in overview.items():
            label_map = {
                "total_steps": "总步数",
                "online_cycles": "在线周期",
                "offline_cycles": "离线周期",
                "total_time": "总耗时(s)",
                "health_score": "健康评分",
            }
            label = label_map.get(key, key)
            html_parts.append(
                f"<div class='metric-card'>"
                f"<div class='metric-value'>{val}</div>"
                f"<div class='metric-label'>{label}</div>"
                f"</div>"
            )
        # 配置
        html_parts.append("<h2>系统配置</h2>")
        config = data.get("config", {})
        html_parts.append("<table>")
        html_parts.append("<tr><th>参数</th><th>值</th></tr>")
        for key, val in config.items():
            html_parts.append(f"<tr><td>{key}</td><td>{val}</td></tr>")
        html_parts.append("</table>")
        # 指标摘要
        html_parts.append("<h2>指标摘要</h2>")
        metrics = data.get("metrics_summary", {})
        if metrics:
            html_parts.append("<table>")
            html_parts.append("<tr><th>指标</th><th>均值</th><th>标准差</th>"
                              "<th>最小值</th><th>最大值</th><th>最后值</th><th>趋势</th></tr>")
            for name, stats in metrics.items():
                if isinstance(stats, dict):
                    sparkline = ASCIITextChart.sparkline(
                        stats.get("history", []))
                    html_parts.append(
                        f"<tr><td>{name}</td>"
                        f"<td>{stats.get('mean', 0):.4f}</td>"
                        f"<td>{stats.get('std', 0):.4f}</td>"
                        f"<td>{stats.get('min', 0):.4f}</td>"
                        f"<td>{stats.get('max', 0):.4f}</td>"
                        f"<td>{stats.get('last', 0):.4f}</td>"
                        f"<td style='font-family:monospace'>{sparkline}</td></tr>"
                    )
            html_parts.append("</table>")
        # 指标图表
        html_parts.append("<h2>指标趋势图</h2>")
        chart_data = data.get("charts", {})
        for name, chart_text in chart_data.items():
            html_parts.append(f"<h3>{name}</h3>")
            html_parts.append(f"<div class='chart'>{chart_text}</div>")
        # 健康状态
        health = data.get("health", {})
        if health:
            html_parts.append("<h2>健康状态</h2>")
            score = health.get("health_score", 1.0)
            css_class = "health-good" if score > 0.7 else (
                "health-warn" if score > 0.3 else "health-bad")
            html_parts.append(
                f"<p>健康评分: <span class='{css_class}'>{score:.2f}</span></p>"
            )
            alerts = health.get("alerts", [])
            if alerts:
                html_parts.append("<h3>当前告警</h3>")
                html_parts.append("<table>")
                html_parts.append("<tr><th>类型</th><th>指标</th>"
                                  "<th>严重度</th><th>描述</th></tr>")
                for alert in alerts:
                    html_parts.append(
                        f"<tr><td>{alert.get('anomaly_type', '')}</td>"
                        f"<td>{alert.get('metric_name', '')}</td>"
                        f"<td>{alert.get('severity', 0):.2f}</td>"
                        f"<td>{alert.get('description', '')}</td></tr>"
                    )
                html_parts.append("</table>")
        html_parts.append("</div></body></html>")
        return '\n'.join(html_parts)

    def generate_csv_report(self, agent, run_summary: Optional[Dict] = None) -> str:
        """生成 CSV 格式指标报告。

        返回：CSV 字符串
        """
        data = self._collect_report_data(agent, run_summary)
        metrics = data.get("metrics_summary", {})
        lines = ["metric_name,mean,std,min,max,last,count"]
        for name, stats in metrics.items():
            if isinstance(stats, dict):
                lines.append(
                    f"{name},{stats.get('mean', 0):.6f},"
                    f"{stats.get('std', 0):.6f},"
                    f"{stats.get('min', 0):.6f},"
                    f"{stats.get('max', 0):.6f},"
                    f"{stats.get('last', 0):.6f},"
                    f"{stats.get('count', 0)}"
                )
        return '\n'.join(lines)

    def save_report(self, agent, filepath: str,
                    format: str = "json",
                    run_summary: Optional[Dict] = None) -> str:
        """保存报告到文件。

        参数：
        - agent: SPEPAAgent 实例
        - filepath: 文件路径（不含扩展名，自动添加）
        - format: 格式 ("json"/"html"/"csv")
        - run_summary: run() 返回的摘要

        返回：实际保存的文件路径
        """
        ext_map = {"json": ".json", "html": ".html", "csv": ".csv"}
        ext = ext_map.get(format.lower(), ".json")
        if not filepath.endswith(ext):
            filepath = filepath + ext
        if format.lower() == "json":
            content = self.generate_json_report(agent, run_summary)
        elif format.lower() == "html":
            content = self.generate_html_report(agent, run_summary)
        elif format.lower() == "csv":
            content = self.generate_csv_report(agent, run_summary)
        else:
            raise ValueError(f"不支持的格式: {format}。可用: json, html, csv")
        # 确保目录存在
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        self.log.info("Report saved to %s (%s, %d bytes)",
                      filepath, format, len(content))
        return filepath

    def _collect_report_data(self, agent,
                             run_summary: Optional[Dict] = None) -> Dict[str, Any]:
        """收集报告所需的全部数据。"""
        config = agent.config
        # 概览
        overview = {
            "total_steps": agent.step_count,
            "online_cycles": agent.online_cycles,
            "offline_cycles": agent.offline_cycles,
            "health_score": round(agent.get_health_score(), 4),
        }
        if run_summary:
            overview["total_time"] = round(run_summary.get("total_time", 0), 2)
        # 配置摘要
        config_summary = {
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "obs_dim": config.obs_dim,
            "max_steps": config.max_steps,
            "seed": config.seed,
            "prediction_layers": config.prediction_layers,
            "learning_rate_micro": config.learning_rate_micro,
            "safety_hard_boundary": config.safety_hard_boundary,
            "cpu_max_percent": config.resources.cpu_max_percent,
            "memory_max_percent": config.resources.memory_max_percent,
        }
        # 指标摘要（含历史用于 sparkline）
        metrics_summary = {}
        try:
            raw_summary = agent.metrics.summary()
            for name, stats in raw_summary.items():
                if isinstance(stats, dict):
                    history = agent.metrics.get_recent(name, n=20)
                    metrics_summary[name] = {
                        **stats,
                        "history": history,
                    }
                else:
                    metrics_summary[name] = {"value": stats, "history": []}
        except Exception:
            pass
        # 图表
        charts = {}
        for name in ["prediction_error", "confidence", "total_reward"]:
            history = agent.metrics.get_recent(name, n=50)
            if history:
                charts[name] = ASCIITextChart.line_chart(
                    history, title=f"{name} 趋势")
        # 健康状态
        health = {}
        try:
            monitor = agent.self_repair_system.monitor
            health = {
                "health_score": monitor.get_health_score(),
                "alerts": [a.to_dict() for a in monitor.get_alerts()],
                "trends": monitor.get_trends(),
            }
        except Exception:
            pass
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "0.1.0",
            "overview": overview,
            "config": config_summary,
            "metrics_summary": metrics_summary,
            "charts": charts,
            "health": health,
        }
