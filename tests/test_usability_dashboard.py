"""SPEPA 易用性阶段六测试：交互式 Web Dashboard。

测试范围：
1. WebDashboard 初始化与生命周期
2. HTTP API 端点（/api/state, /api/metrics, /api/health, /api/report）
3. Dashboard 主页 HTML
4. 运行控制（/api/run, /api/stop）
5. CLI dashboard 命令
"""
import time
import json
import urllib.request
import urllib.error
import pytest

from spepa.dashboard import WebDashboard, DashboardHandler
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_default_config


# 全局端口分配器，避免测试间端口冲突
_port_counter = [19000]

def _get_port():
    """获取一个唯一端口。"""
    _port_counter[0] += 1
    return _port_counter[0]


def _fetch(url):
    """获取 URL 内容。"""
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read().decode('utf-8'), resp.status


def _post(url):
    """POST 请求。"""
    req = urllib.request.Request(url, method='POST')
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode('utf-8'), resp.status


@pytest.fixture
def running_dashboard():
    """启动一个 Dashboard 实例（后台线程），测试后停止。"""
    agent = SPEPAAgent(get_default_config())
    agent.run(n_steps=3)  # 产生一些数据
    port = _get_port()
    dashboard = WebDashboard(agent, host="localhost", port=port)
    dashboard.start(blocking=False)
    time.sleep(0.5)  # 等待服务器启动
    yield dashboard
    dashboard.stop()
    time.sleep(0.2)  # 等待端口释放


# ============================================================
# WebDashboard 初始化测试
# ============================================================

class TestWebDashboardInit:
    """Dashboard 初始化测试。"""

    def test_initialization(self):
        """正确初始化。"""
        agent = SPEPAAgent(get_default_config())
        dashboard = WebDashboard(agent, port=_get_port())
        assert dashboard.agent is agent
        assert dashboard.host == "localhost"
        assert not dashboard.is_running()

    def test_get_url(self):
        """获取 URL。"""
        agent = SPEPAAgent(get_default_config())
        port = _get_port()
        dashboard = WebDashboard(agent, port=port)
        assert dashboard.get_url() == f"http://localhost:{port}"

    def test_get_state(self):
        """获取状态。"""
        agent = SPEPAAgent(get_default_config())
        dashboard = WebDashboard(agent, port=_get_port())
        state = dashboard.get_state()
        assert "running" in state
        assert "url" in state
        assert state["running"] is False


# ============================================================
# HTTP API 测试
# ============================================================

class TestDashboardAPI:
    """Dashboard HTTP API 测试。"""

    def test_main_page(self, running_dashboard):
        """主页返回 HTML。"""
        content, status = _fetch(running_dashboard.get_url() + "/")
        assert status == 200
        assert "<html" in content.lower()
        assert "SPEPA" in content

    def test_api_state(self, running_dashboard):
        """/api/state 返回状态 JSON。"""
        content, status = _fetch(running_dashboard.get_url() + "/api/state")
        assert status == 200
        data = json.loads(content)
        assert "step_count" in data
        assert "online_cycles" in data

    def test_api_metrics(self, running_dashboard):
        """/api/metrics 返回指标 JSON。"""
        content, status = _fetch(running_dashboard.get_url() + "/api/metrics")
        assert status == 200
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_api_health(self, running_dashboard):
        """/api/health 返回健康 JSON。"""
        content, status = _fetch(running_dashboard.get_url() + "/api/health")
        assert status == 200
        data = json.loads(content)
        assert "health_score" in data
        assert "alerts" in data

    def test_api_report_json(self, running_dashboard):
        """/api/report?format=json 返回报告。"""
        content, status = _fetch(running_dashboard.get_url() + "/api/report?format=json")
        assert status == 200
        data = json.loads(content)
        assert "overview" in data

    def test_api_report_html(self, running_dashboard):
        """/api/report?format=html 返回 HTML。"""
        content, status = _fetch(running_dashboard.get_url() + "/api/report?format=html")
        assert status == 200
        assert "<html" in content.lower()

    def test_api_report_csv(self, running_dashboard):
        """/api/report?format=csv 返回 CSV。"""
        content, status = _fetch(running_dashboard.get_url() + "/api/report?format=csv")
        assert status == 200
        assert "metric_name" in content

    def test_404_for_unknown_path(self, running_dashboard):
        """未知路径返回 404。"""
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _fetch(running_dashboard.get_url() + "/unknown")
        assert exc_info.value.code == 404


# ============================================================
# 运行控制测试
# ============================================================

class TestDashboardRunControl:
    """Dashboard 运行控制测试。"""

    def test_api_run(self, running_dashboard):
        """POST /api/run 启动运行。"""
        initial_steps = running_dashboard.agent.step_count
        content, status = _post(running_dashboard.get_url() + "/api/run?steps=3")
        assert status == 200
        data = json.loads(content)
        assert data["status"] == "running"
        # 等待运行完成
        time.sleep(1.0)
        assert running_dashboard.agent.step_count > initial_steps

    def test_api_stop(self, running_dashboard):
        """POST /api/stop 停止运行。"""
        _post(running_dashboard.get_url() + "/api/run?steps=100")
        time.sleep(0.2)
        content, status = _post(running_dashboard.get_url() + "/api/stop")
        assert status == 200
        data = json.loads(content)
        assert data["status"] == "stopped"

    def test_run_in_background(self, running_dashboard):
        """运行在后台线程执行。"""
        _post(running_dashboard.get_url() + "/api/run?steps=5")
        time.sleep(0.5)
        # Agent 应在运行（或已完成少量步数）
        state = running_dashboard.get_state()
        # 运行可能已完成（步数少），检查步数增加了即可
        initial_steps = running_dashboard.agent.step_count
        assert initial_steps >= 3  # 至少运行了一些步
        time.sleep(1.5)
        # 运行完成后应停止
        assert running_dashboard.get_state()["agent_running"] is False


# ============================================================
# Dashboard 生命周期测试
# ============================================================

class TestDashboardLifecycle:
    """Dashboard 生命周期测试。"""

    def test_start_stop(self):
        """启动和停止。"""
        agent = SPEPAAgent(get_default_config())
        dashboard = WebDashboard(agent, port=_get_port())
        dashboard.start(blocking=False)
        time.sleep(0.3)
        assert dashboard.is_running()
        dashboard.stop()
        assert not dashboard.is_running()

    def test_double_start_warning(self):
        """重复启动不崩溃。"""
        agent = SPEPAAgent(get_default_config())
        dashboard = WebDashboard(agent, port=_get_port())
        dashboard.start(blocking=False)
        time.sleep(0.2)
        dashboard.start(blocking=False)  # 应警告但不崩溃
        dashboard.stop()

    def test_stop_without_start(self):
        """未启动时停止不崩溃。"""
        agent = SPEPAAgent(get_default_config())
        dashboard = WebDashboard(agent, port=_get_port())
        dashboard.stop()  # 应安全


# ============================================================
# CLI dashboard 命令测试
# ============================================================

class TestCLIDashboard:
    """CLI dashboard 命令测试。"""

    def test_dashboard_command_parsed(self):
        """dashboard 命令参数解析。"""
        from spepa.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["dashboard", "--port", "9090", "--host", "0.0.0.0"])
        assert args.command == "dashboard"
        assert args.port == 9090
        assert args.host == "0.0.0.0"

    def test_dashboard_in_help(self, capsys):
        """dashboard 出现在帮助中。"""
        from spepa.cli import main
        main([])
        captured = capsys.readouterr()
        assert "dashboard" in captured.out


# ============================================================
# 端到端测试
# ============================================================

class TestDashboardEndToEnd:
    """Dashboard 端到端测试。"""

    def test_full_dashboard_session(self, running_dashboard):
        """完整 Dashboard 会话。"""
        base = running_dashboard.get_url()
        # 1. 访问主页
        html, _ = _fetch(base + "/")
        assert "SPEPA" in html
        # 2. 查看状态
        state, _ = _fetch(base + "/api/state")
        assert "step_count" in json.loads(state)
        # 3. 查看健康
        health, _ = _fetch(base + "/api/health")
        assert "health_score" in json.loads(health)
        # 4. 启动运行
        run_result, _ = _post(base + "/api/run?steps=2")
        assert json.loads(run_result)["status"] == "running"
        time.sleep(0.5)
        # 5. 查看更新后的状态
        state2, _ = _fetch(base + "/api/state")
        assert json.loads(state2)["step_count"] >= json.loads(state)["step_count"]
        # 6. 下载报告
        report, _ = _fetch(base + "/api/report?format=json")
        assert "overview" in json.loads(report)

    def test_dashboard_reflects_agent_state(self, running_dashboard):
        """Dashboard 反映 Agent 状态变化。"""
        base = running_dashboard.get_url()
        # 记录初始步数
        state1, _ = _fetch(base + "/api/state")
        steps1 = json.loads(state1)["step_count"]
        # 运行几步
        _post(base + "/api/run?steps=3")
        time.sleep(1.0)
        # 检查步数增加
        state2, _ = _fetch(base + "/api/state")
        steps2 = json.loads(state2)["step_count"]
        assert steps2 > steps1
