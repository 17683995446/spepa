"""SPEPA 事件回调与报告导出示例。

演示如何使用事件系统监控运行过程，以及如何导出运行报告。

运行方式：
    python examples/events_and_reports.py
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spepa import SPEPAAgent, get_preset_config, EventType, ProgressReporter


def main():
    config = get_preset_config("minimal")
    agent = SPEPAAgent(config)

    print("=== SPEPA 事件回调与报告导出示例 ===\n")

    # 1. 注册事件回调
    print("1. 注册事件回调...")
    step_count = [0]
    error_history = []

    def on_step_end(event):
        step_count[0] += 1
        error = event.data.get("error", 0)
        error_history.append(error)
        if step_count[0] % 5 == 0:
            print(f"   Step {event.data['step']}: error={error:.4f}, "
                  f"confidence={event.data.get('confidence', 0):.3f}")

    agent.on(EventType.STEP_END, on_step_end)
    print(f"   已注册 STEP_END 回调\n")

    # 2. 使用 ProgressReporter 自动打印进度
    print("2. 启动 ProgressReporter...")
    reporter = ProgressReporter(agent, interval=10)

    # 3. 运行
    print("3. 运行 20 步...\n")
    reporter.start(20)
    summary = agent.run(n_steps=20)
    reporter.finish({"total_steps": 20, "goals_achieved": 0})

    # 4. 导出报告
    print("\n4. 导出报告...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON 报告
        json_path = agent.generate_report("json", os.path.join(tmpdir, "report"))
        print(f"   JSON 报告: {json_path}")

        # HTML 报告
        html_path = agent.generate_report("html", os.path.join(tmpdir, "report"))
        print(f"   HTML 报告: {html_path}")

        # CSV 报告
        csv_path = agent.generate_report("csv", os.path.join(tmpdir, "report"))
        print(f"   CSV 报告:  {csv_path}")

    # 5. 打印指标摘要
    print("\n5. 指标摘要:")
    metrics = summary.get("metrics", {})
    for name in ["prediction_error", "confidence", "cycle_time"]:
        if name in metrics:
            m = metrics[name]
            print(f"   {name:20s}: mean={m['mean']:.4f}, "
                  f"min={m['min']:.4f}, max={m['max']:.4f}")

    print(f"\n=== 完成！共收到 {step_count[0]} 个 STEP_END 事件 ===")


if __name__ == "__main__":
    main()
