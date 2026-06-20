"""SPEPA 交互式 Web Dashboard。

设计原则（第一性原理）：
- 零依赖原理：核心功能基于 Python 标准库 http.server，无需安装 Flask
- 实时性原理：Dashboard 自动刷新，展示 Agent 最新状态
- 控制性原理：不仅可观察，还可通过 Web 控制 Agent 运行

功能：
- GET /          : Dashboard 主页（HTML，自动刷新）
- GET /api/state : Agent 完整状态（JSON）
- GET /api/metrics : 指标摘要（JSON）
- GET /api/health : 健康状态（JSON）
- POST /api/run  : 启动运行（参数: steps）
- POST /api/stop : 停止运行
- GET /api/report/:下载报告（参数: format=json/html/csv）

使用方式：
    from spepa import SPEPAAgent, WebDashboard
    agent = SPEPAAgent()
    dashboard = WebDashboard(agent, port=8080)
    dashboard.start()  # 在后台线程启动
    # 浏览器访问 http://localhost:8080
"""
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

from spepa.utils.logger import LoggedMixin


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang='zh-CN'>
<head>
<meta charset='UTF-8'>
<meta http-equiv='refresh' content='2'>
<title>SPEPA Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
h1 { color: #0f3460; background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px; }
.card { background: #16213e; padding: 20px; border-radius: 8px; border: 1px solid #0f3460; }
.card h3 { color: #e94560; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; }
.metric { font-size: 28px; font-weight: bold; color: #0f3460; }
.metric-label { font-size: 12px; color: #888; margin-top: 5px; }
.health-good { color: #27ae60; } .health-warn { color: #f39c12; } .health-bad { color: #e74c3c; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px; text-align: left; border-bottom: 1px solid #0f3460; }
th { color: #e94560; font-size: 12px; text-transform: uppercase; }
.controls { margin: 20px 0; }
button { background: #e94560; color: white; border: none; padding: 10px 20px;
         border-radius: 5px; cursor: pointer; margin-right: 10px; font-size: 14px; }
button:hover { background: #c81e45; }
input { background: #0f3460; color: white; border: 1px solid #e94560; padding: 8px;
        border-radius: 5px; width: 80px; }
pre { background: #0d1117; padding: 15px; border-radius: 5px; overflow-x: auto;
      font-size: 12px; color: #0f0; }
.section { background: #16213e; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
.section h2 { color: #e94560; margin-bottom: 15px; }
</style>
</head>
<body>
<h1>SPEPA Dashboard</h1>
<div class='controls'>
  <input type='number' id='steps' value='10' min='1' max='1000'>
  <button onclick='runAgent()'>运行</button>
  <button onclick='stopAgent()'>停止</button>
  <button onclick='downloadReport("json")'>JSON报告</button>
  <button onclick='downloadReport("html")'>HTML报告</button>
  <button onclick='downloadReport("csv")'>CSV报告</button>
</div>
<div class='grid' id='overview'></div>
<div class='grid' id='health'></div>
<div class='section'><h2>指标摘要</h2><div id='metrics'></div></div>
<div class='section'><h2>系统状态</h2><pre id='state'></pre></div>
<script>
async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}
async function update() {
  try {
    const state = await fetchJSON('/api/state');
    const health = await fetchJSON('/api/health');
    const metrics = await fetchJSON('/api/metrics');
    // 概览
    const ov = state;
    const overviewHTML = [
      ['步数', ov.step_count],
      ['在线周期', ov.online_cycles],
      ['离线周期', ov.offline_cycles],
      ['探索比例', (ov.exploration_ratio||0).toFixed(3)],
    ].map(([k,v]) => `<div class='card'><h3>${k}</h3><div class='metric'>${v}</div></div>`).join('');
    document.getElementById('overview').innerHTML = overviewHTML;
    // 健康
    const hs = health.health_score || 1.0;
    const cls = hs > 0.7 ? 'health-good' : (hs > 0.3 ? 'health-warn' : 'health-bad');
    const alertHTML = (health.alerts||[]).map(a =>
      `<tr><td>${a.anomaly_type}</td><td>${a.metric_name}</td><td>${a.severity.toFixed(2)}</td><td>${a.description}</td></tr>`
    ).join('');
    document.getElementById('health').innerHTML = `
      <div class='card'><h3>健康评分</h3><div class='metric ${cls}'>${hs.toFixed(2)}</div></div>
      <div class='card'><h3>异常总数</h3><div class='metric'>${health.total_anomalies||0}</div></div>
      <div class='card'><h3>当前告警</h3><div class='metric'>${(health.alerts||[]).length}</div></div>`;
    // 指标
    let mHTML = '<table><tr><th>指标</th><th>均值</th><th>最小</th><th>最大</th><th>最后</th></tr>';
    for (const [k,v] of Object.entries(metrics)) {
      if (typeof v === 'object') {
        mHTML += `<tr><td>${k}</td><td>${(v.mean||0).toFixed(4)}</td><td>${(v.min||0).toFixed(4)}</td><td>${(v.max||0).toFixed(4)}</td><td>${(v.last||0).toFixed(4)}</td></tr>`;
      }
    }
    mHTML += '</table>';
    document.getElementById('metrics').innerHTML = mHTML;
    // 状态
    document.getElementById('state').textContent = JSON.stringify(state, null, 2).substring(0, 2000);
  } catch(e) {
    console.error(e);
  }
}
async function runAgent() {
  const steps = document.getElementById('steps').value;
  await fetch('/api/run?steps=' + steps, {method: 'POST'});
  update();
}
async function stopAgent() {
  await fetch('/api/stop', {method: 'POST'});
  update();
}
function downloadReport(fmt) {
  window.open('/api/report?format=' + fmt);
}
update();
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """Dashboard HTTP 请求处理器。"""

    # 类级引用，由 WebDashboard 设置
    dashboard = None

    def log_message(self, format, *args):
        """静默 HTTP 日志（除非 DEBUG）。"""
        pass

    def do_GET(self):
        """处理 GET 请求。"""
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/':
            self._send_html(DASHBOARD_HTML)
        elif path == '/api/state':
            state = self.dashboard.agent.get_state()
            self._send_json(state)
        elif path == '/api/metrics':
            metrics = self.dashboard.agent.metrics.summary()
            self._send_json(metrics)
        elif path == '/api/health':
            monitor = self.dashboard.agent.self_repair_system.monitor
            health = {
                "health_score": monitor.get_health_score(),
                "total_anomalies": monitor.total_anomalies,
                "alerts": [a.to_dict() for a in monitor.get_alerts()],
                "trends": monitor.get_trends(),
            }
            self._send_json(health)
        elif path == '/api/report':
            qs = parse_qs(parsed.query)
            fmt = qs.get('format', ['json'])[0]
            try:
                report = self.dashboard.agent.generate_report(fmt)
                if fmt == 'html':
                    self._send_html(report)
                elif fmt == 'csv':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/csv')
                    self.send_header('Content-Disposition',
                                     'attachment; filename="spepa_report.csv"')
                    self.end_headers()
                    self.wfile.write(report.encode('utf-8'))
                else:
                    self._send_json_str(report)
            except Exception as e:
                self._send_error(str(e))
        else:
            self._send_error("Not found", 404)

    def do_POST(self):
        """处理 POST 请求。"""
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == '/api/run':
            steps = int(qs.get('steps', ['10'])[0])
            self.dashboard.start_run(steps)
            self._send_json({"status": "running", "steps": steps})
        elif path == '/api/stop':
            self.dashboard.stop_run()
            self._send_json({"status": "stopped"})
        else:
            self._send_error("Not found", 404)

    def _send_json(self, data):
        """发送 JSON 响应。"""
        self._send_json_str(json.dumps(data, default=str, ensure_ascii=False))

    def _send_json_str(self, json_str):
        """发送 JSON 字符串。"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json_str.encode('utf-8'))

    def _send_html(self, html):
        """发送 HTML 响应。"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _send_error(self, message, code=400):
        """发送错误响应。"""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode('utf-8'))


class WebDashboard(LoggedMixin):
    """SPEPA Web Dashboard：交互式 Web 监控界面。

    使用方式：
        agent = SPEPAAgent()
        dashboard = WebDashboard(agent, port=8080)
        dashboard.start()  # 后台启动
        # 浏览器访问 http://localhost:8080
        dashboard.stop()   # 停止
    """

    def __init__(self, agent, host: str = "localhost", port: int = 8080):
        """初始化 Dashboard。

        参数：
        - agent: SPEPAAgent 实例
        - host: 监听地址
        - port: 监听端口
        """
        self._set_log_name("spepa.dashboard")
        self.agent = agent
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._run_thread: Optional[threading.Thread] = None
        self._should_stop = False
        # 设置 handler 的 dashboard 引用
        DashboardHandler.dashboard = self
        self.log.info("WebDashboard initialized (http://%s:%d)", host, port)

    def _create_server(self) -> HTTPServer:
        """创建 HTTP 服务器（允许端口复用）。"""
        server = HTTPServer((self.host, self.port), DashboardHandler)
        server.allow_reuse_address = True
        return server

    def start(self, blocking: bool = False) -> None:
        """启动 Dashboard 服务器。

        参数：
        - blocking: True 时阻塞当前线程；False 时在后台线程启动
        """
        if self._running:
            self.log.warning("Dashboard already running")
            return
        self._server = self._create_server()
        self._running = True
        if blocking:
            self.log.info("Dashboard running at http://%s:%d (blocking)",
                          self.host, self.port)
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True)
            self._thread.start()
            self.log.info("Dashboard started at http://%s:%d (background)",
                          self.host, self.port)

    def stop(self) -> None:
        """停止 Dashboard 服务器。"""
        self._should_stop = True
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self._run_thread:
            self._run_thread.join(timeout=5)
            self._run_thread = None
        self._running = False
        self.log.info("Dashboard stopped")

    def start_run(self, steps: int) -> None:
        """在后台线程启动 Agent 运行。"""
        if self._run_thread and self._run_thread.is_alive():
            self.log.warning("Agent already running")
            return
        self._should_stop = False
        self._run_thread = threading.Thread(
            target=self._run_agent, args=(steps,), daemon=True)
        self._run_thread.start()
        self.log.info("Agent run started: %d steps", steps)

    def stop_run(self) -> None:
        """停止 Agent 运行。"""
        self._should_stop = True
        self.log.info("Agent run stop requested")

    def _run_agent(self, steps: int) -> None:
        """在后台运行 Agent。"""
        try:
            for i in range(steps):
                if self._should_stop:
                    break
                self.agent.online_cycle()
        except Exception as e:
            self.log.error("Agent run error: %s", e)

    def is_running(self) -> bool:
        """Dashboard 是否在运行。"""
        return self._running

    def get_url(self) -> str:
        """获取 Dashboard URL。"""
        return f"http://{self.host}:{self.port}"

    def get_state(self) -> Dict[str, Any]:
        """获取 Dashboard 状态。"""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "url": self.get_url(),
            "agent_running": (
                self._run_thread is not None and self._run_thread.is_alive()
            ),
        }
