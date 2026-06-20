"""SPEPA 快速入门示例。

最简使用方式：3 行代码启动 Agent 并运行。

运行方式：
    python examples/quickstart.py
"""
import sys
import os

# 将项目根目录加入路径（开发模式下）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spepa import SPEPAAgent, get_preset_config


def main():
    # 1. 选择预设配置（minimal/balanced/performance/conservative）
    config = get_preset_config("minimal")

    # 2. 创建 Agent
    agent = SPEPAAgent(config)

    # 3. 运行 20 步
    print("=== SPEPA 快速入门 ===")
    print(f"配置: minimal (state_dim={config.state_dim})")
    print(f"运行 20 步...\n")

    summary = agent.run(n_steps=20)

    # 4. 查看结果
    print(f"\n=== 运行结果 ===")
    print(f"总步数:     {summary['total_steps']}")
    print(f"在线周期:   {summary['online_cycles']}")
    print(f"离线周期:   {summary['offline_cycles']}")
    print(f"总耗时:     {summary['total_time']:.2f}s")

    # 5. 查看关键指标
    metrics = summary.get("metrics", {})
    if "prediction_error" in metrics:
        err = metrics["prediction_error"]
        print(f"\n预测误差:   mean={err['mean']:.4f}, last={err['last']:.4f}")
    if "confidence" in metrics:
        conf = metrics["confidence"]
        print(f"信心:       mean={conf['mean']:.4f}, last={conf['last']:.4f}")

    # 6. 查看健康评分
    print(f"\n健康评分:   {agent.get_health_score():.2f}")

    print("\n=== 完成！===")
    print("提示: 运行 'spepa presets' 查看所有预设配置")
    print("提示: 运行 'spepa run --preset performance --steps 100' 尝试高性能模式")


if __name__ == "__main__":
    main()
