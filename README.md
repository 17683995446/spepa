# SPEPA

**Self-driven Perpetual Evolutionary Prediction-Correction Agent**

自驱永续进化预测修正 Agent 系统。

## 快速开始

### 安装

```bash
pip install -e .
```

### 3 行代码启动

```python
from spepa import SPEPAAgent, get_preset_config

agent = SPEPAAgent(get_preset_config("minimal"))
agent.run(n_steps=20)
```

### 命令行使用

```bash
# 列出预设配置
spepa presets

# 基础运行
spepa run --steps 100 --preset balanced

# 自主驱动运行
spepa autonomous --steps 50 --preset minimal

# 永续运行（异常自恢复）
spepa perpetual --cycles 100

# 多Agent协作
spepa collaborate --mode parallel --target 0.8

# 查看系统信息
spepa info
```

## 预设配置

| 预设 | 适用场景 | state_dim | max_steps | CPU上限 |
|------|---------|-----------|-----------|--------|
| `minimal` | 嵌入式/测试 | 8 | 200 | 10% |
| `balanced` | 通用场景（默认） | 16 | 1000 | 20% |
| `performance` | 服务器/研究 | 32 | 5000 | 80% |
| `conservative` | 安全优先 | 16 | 500 | 15% |

```python
from spepa import list_presets, get_preset_config

# 查看所有预设
print(list_presets())

# 选择预设
config = get_preset_config("performance")
```

## 核心功能

### 基础运行

```python
agent = SPEPAAgent()
summary = agent.run(n_steps=100)
print(f"步数: {summary['total_steps']}, 耗时: {summary['total_time']:.2f}s")
```

### 自主驱动

Agent 自主生成目标、入队、执行、达成。

```python
# 完全自主运行
agent.run_autonomous(n_steps=100)

# 添加外部目标
import numpy as np
agent.add_autonomous_goal(np.ones(16) * 0.8, priority=10.0)
```

### 多Agent协作

三种协作模式：并行分解、串行依赖、竞争择优。

```python
from spepa.multiagent.collaboration import CollaborationMode

# 并行协作
agent.collaborate(target, mode=CollaborationMode.PARALLEL)

# 串行协作
agent.collaborate(target, mode=CollaborationMode.SEQUENTIAL)

# 竞争协作
agent.collaborate(target, mode=CollaborationMode.COMPETITIVE, n_competitors=3)
```

### 事件回调

运行时监控 Agent 内部事件。

```python
from spepa import EventType

# 每步结束时打印误差
agent.on(EventType.STEP_END, lambda e: print(f"Step {e.data['step']}: err={e.data['error']:.4f}"))

# 目标达成时通知
agent.on(EventType.GOAL_ACHIEVED, lambda e: print(f"目标达成: {e.data['goal_id']}"))

agent.run(n_steps=100)
```

### 进度报告

自动打印运行进度。

```python
from spepa import ProgressReporter

reporter = ProgressReporter(agent, interval=10)
reporter.start(100)
agent.run(n_steps=100)
reporter.finish({"total_steps": 100})
```

### 报告导出

导出 JSON / HTML / CSV 格式运行报告。

```python
# 导出 HTML 报告
agent.run(n_steps=100)
agent.generate_report("html", "report.html")

# 导出 JSON
agent.generate_report("json", "report.json")

# 导出 CSV（指标表格）
agent.generate_report("csv", "metrics.csv")
```

### 永续运行

异常自恢复，永不停止。

```python
agent.run_perpetual(max_cycles=1000, auto_recover=True)
```

### 状态持久化

```python
# 保存
agent.save_state("./checkpoint")

# 恢复
agent.load_state("./checkpoint")
```

### 元认知自监控与自我修复

```python
# 执行一轮自监控与修复
result = agent.self_repair_step()
print(f"健康评分: {agent.get_health_score():.2f}")

# 运行中持续自监控
for i in range(100):
    agent.online_cycle()
    if i % 10 == 0:
        agent.self_repair_step()
```

## 示例代码

| 文件 | 内容 |
|------|------|
| `examples/quickstart.py` | 最简入门示例 |
| `examples/autonomous.py` | 自主驱动与永续运行 |
| `examples/collaboration.py` | 多Agent协作三种模式 |
| `examples/events_and_reports.py` | 事件回调与报告导出 |

运行示例：

```bash
python examples/quickstart.py
python examples/autonomous.py
python examples/collaboration.py
python examples/events_and_reports.py
```

## API 速查

### SPEPAAgent 主要方法

| 方法 | 说明 |
|------|------|
| `run(n_steps)` | 基础在线运行 |
| `run_autonomous(n_steps)` | 自主驱动运行 |
| `run_perpetual(max_cycles)` | 永续运行 |
| `collaborate(target, mode)` | 多Agent协作 |
| `run_long_task(target, n_stages)` | 长任务驱动 |
| `online_cycle()` | 单步在线闭环 |
| `step()` | 单步（在线+条件离线） |
| `add_autonomous_goal(target)` | 添加外部目标 |
| `on(event_type, callback)` | 注册事件回调 |
| `generate_report(format, filepath)` | 生成报告 |
| `self_repair_step()` | 自监控与修复 |
| `get_health_score()` | 获取健康评分 |
| `save_state(dir)` / `load_state(dir)` | 状态持久化 |
| `get_state()` | 获取完整状态 |
| `reset()` | 重置 Agent |

### EventType 事件类型

| 事件 | 触发时机 |
|------|---------|
| `STEP_START` | 每步开始 |
| `STEP_END` | 每步结束（含指标） |
| `CYCLE_COMPLETE` | run() 完成 |
| `GOAL_GENERATED` | 自主目标生成 |
| `GOAL_ACHIEVED` | 自主目标达成 |
| `ANOMALY` | 异常检测 |
| `REPAIR` | 自我修复执行 |
| `ERROR` | 运行错误 |

## 系统架构

```
SPEPA Agent
├── 核心引擎
│   ├── 层级预测模型（预测编码）
│   ├── 自由能计算（变分推断）
│   ├── 三大驱力引擎（安全/收益/能效）
│   ├── 缺口检测器（信息缺口 = KL散度）
│   ├── 目标层次体系（L0-L4）
│   ├── 主动推理决策
│   ├── 元认知监控（信心校准）
│   ├── 双系统记忆（快速+慢速巩固）
│   ├── 三档优化器（micro/hebbian/structural）
│   └── 安全护栏
├── 自主驱动
│   ├── 目标生成器（4来源）
│   ├── 任务队列（优先级+去重+老化）
│   └── 自主执行器
├── 多Agent协作
│   ├── 通信总线
│   ├── 经验回传
│   └── 协作编排器（并行/串行/竞争）
├── 永续运行
│   ├── 永续运行器（异常自恢复）
│   ├── 状态持久化（检查点+崩溃恢复）
│   └── 资源自调节（PID控制+背压+降级）
└── 元认知自修复
    ├── 自我监控（5种异常检测）
    ├── 根因诊断
    └── 分级修复（L1软/L2中/L3硬）
```

## 配置参数

```python
from spepa import SPEPAConfig

config = SPEPAConfig(
    state_dim=16,           # 状态维度
    action_dim=8,           # 动作维度
    obs_dim=16,             # 观测维度
    max_steps=1000,         # 最大步数
    seed=42,                # 随机种子
    learning_rate_micro=0.01,
    safety_hard_boundary=0.95,
)
```

## 测试

```bash
pytest tests/ -q
```

## 许可证

MIT
