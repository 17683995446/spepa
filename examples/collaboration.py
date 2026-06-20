"""SPEPA 多Agent协作示例。

演示三种协作模式：并行分解、串行依赖、竞争择优。

运行方式：
    python examples/collaboration.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spepa import SPEPAAgent, get_preset_config
from spepa.multiagent.collaboration import CollaborationMode


def main():
    config = get_preset_config("minimal")
    agent = SPEPAAgent(config)

    print("=== SPEPA 多Agent协作示例 ===\n")

    # 1. 并行协作：按维度分解，子Agent并行处理
    print("1. 并行协作 (PARALLEL)...")
    target = np.ones(config.state_dim) * 0.7
    result = agent.collaborate(target, mode=CollaborationMode.PARALLEL)
    print(f"   成功率:     {result['success_rate']:.1%}")
    print(f"   子Agent数: {result['total_count']}")
    print(f"   平均误差:   {result['avg_error']:.4f}")

    # 2. 串行协作：按阶段分解，前一个输出是后一个输入
    print("\n2. 串行协作 (SEQUENTIAL)...")
    target = np.ones(config.state_dim) * 0.9
    result = agent.collaborate(target, mode=CollaborationMode.SEQUENTIAL)
    print(f"   成功率:     {result['success_rate']:.1%}")
    print(f"   阶段数:     {result['total_count']}")

    # 3. 竞争协作：多个Agent同时尝试，选最优
    print("\n3. 竞争协作 (COMPETITIVE)...")
    target = np.ones(config.state_dim) * 0.5
    result = agent.collaborate(
        target, mode=CollaborationMode.COMPETITIVE, n_competitors=3
    )
    print(f"   成功率:     {result['success_rate']:.1%}")
    print(f"   竞争者数:   {result['total_count']}")
    print(f"   最优误差:   {result.get('best_error', 0):.4f}")

    # 查看协作统计
    state = agent.collaboration_orchestrator.get_state()
    print(f"\n=== 协作统计 ===")
    print(f"总协作次数: {state['total_collaborations']}")
    for mode, count in state['mode_stats'].items():
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()
