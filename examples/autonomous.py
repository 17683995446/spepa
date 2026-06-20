"""SPEPA 自主驱动示例。

演示 Agent 如何自主生成目标、执行任务、达成目标。

运行方式：
    python examples/autonomous.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spepa import SPEPAAgent, get_preset_config, EventType


def main():
    config = get_preset_config("minimal")
    agent = SPEPAAgent(config)

    # 注册事件回调：目标达成时通知
    goals_achieved = []
    agent.on(EventType.GOAL_ACHIEVED, lambda e: goals_achieved.append(e.data))

    print("=== SPEPA 自主驱动示例 ===\n")

    # 方式1：完全自主运行（Agent 自己生成目标）
    print("1. 自主运行 30 步（Agent 自行生成目标）...")
    summary = agent.run_autonomous(n_steps=30)
    print(f"   生成目标: {summary.get('goals_generated', 0)}")
    print(f"   达成目标: {summary.get('goals_achieved', 0)}")
    print(f"   达成率:   {summary.get('achievement_rate', 0):.1%}")

    # 方式2：添加外部目标
    print("\n2. 添加外部目标并执行...")
    target = np.ones(config.state_dim) * 0.8
    agent.add_autonomous_goal(target, priority=10.0, description="用户指定目标")
    for _ in range(10):
        agent.autonomous_step()
    print(f"   累计达成目标: {agent.autonomous_driver.goals_achieved}")

    # 方式3：永续运行（异常自恢复）
    print("\n3. 永续运行 5 个周期（演示异常自恢复）...")
    perp_summary = agent.run_perpetual(max_cycles=5)
    print(f"   完成周期: {perp_summary.get('cycles_completed', 0)}")
    print(f"   恢复次数: {perp_summary.get('recoveries', 0)}")

    print(f"\n=== 完成！共达成 {len(goals_achieved)} 个目标 ===")


if __name__ == "__main__":
    main()
