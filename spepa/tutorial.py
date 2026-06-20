"""SPEPA 交互式教程系统。

设计原则（第一性原理）：
- 分步引导：每个教程由多个步骤组成，逐步讲解概念并演示
- 实操演示：每步包含可执行的代码演示，用户能看到真实运行结果
- 非交互兼容：教程可在非交互模式下自动运行（用于测试/CI）
- 零依赖：仅使用 Python 标准库

教程结构：
    Tutorial
      ├── title       : 教程标题
      ├── description : 教程描述
      └── steps       : 步骤列表
            ├── title       : 步骤标题
            ├── concept     : 概念讲解
            ├── demo        : 演示函数（返回演示结果字符串）
            └── tip         : 提示信息

内置教程：
    - "quickstart" : 5分钟快速入门
    - "config"     : 配置系统详解
    - "environment": 环境适配器使用
    - "debug"      : 调试工具使用
"""
import sys
import time
from typing import Callable, Dict, Any, List, Optional

from spepa.utils.logger import get_logger

logger = get_logger("spepa.tutorial")


# ============================================================
# 数据结构
# ============================================================

class TutorialStep:
    """教程的一个步骤。"""

    def __init__(self,
                 title: str,
                 concept: str,
                 demo: Optional[Callable[[], str]] = None,
                 tip: str = ""):
        """初始化教程步骤。

        参数：
            title : 步骤标题
            concept : 概念讲解（支持多行）
            demo : 演示函数，返回演示结果字符串（None 表示无演示）
            tip : 提示信息
        """
        self.title = title
        self.concept = concept
        self.demo = demo
        self.tip = tip

    def run_demo(self) -> str:
        """运行演示，返回结果字符串。"""
        if self.demo is None:
            return ""
        try:
            return self.demo()
        except Exception as e:
            return f"[演示出错] {e}"


class Tutorial:
    """一个完整的教程。"""

    def __init__(self,
                 name: str,
                 title: str,
                 description: str,
                 steps: List[TutorialStep]):
        """初始化教程。

        参数：
            name : 教程唯一标识
            title : 教程标题
            description : 教程描述
            steps : 步骤列表
        """
        self.name = name
        self.title = title
        self.description = description
        self.steps = steps

    def run(self, interactive: bool = True, pause_fn: Optional[Callable] = None) -> None:
        """运行教程。

        参数：
            interactive : 是否交互模式（每步暂停等待用户确认）
            pause_fn : 暂停函数（None 则使用 input()）
        """
        print(f"\n{'=' * 60}")
        print(f"  教程: {self.title}")
        print(f"  {self.description}")
        print(f"{'=' * 60}\n")

        for i, step in enumerate(self.steps, 1):
            print(f"\n{'─' * 60}")
            print(f"  步骤 {i}/{len(self.steps)}: {step.title}")
            print(f"{'─' * 60}\n")

            # 概念讲解
            print(step.concept)

            # 演示
            if step.demo is not None:
                print("\n[演示]")
                result = step.run_demo()
                if result:
                    print(result)

            # 提示
            if step.tip:
                print(f"\n[提示] {step.tip}")

            # 交互暂停
            if interactive:
                if pause_fn is None:
                    try:
                        input("\n按 Enter 继续...")
                    except EOFError:
                        # 非交互环境，自动继续
                        print("(非交互环境，自动继续)")
                        time.sleep(0.1)
                else:
                    pause_fn()

        print(f"\n{'=' * 60}")
        print(f"  教程完成: {self.title}")
        print(f"  共 {len(self.steps)} 个步骤")
        print(f"{'=' * 60}\n")


# ============================================================
# 内置教程
# ============================================================

def _demo_quickstart_1() -> str:
    """快速入门步骤1演示：创建 Agent。"""
    from spepa import SPEPAAgent, get_preset_config
    config = get_preset_config("minimal")
    agent = SPEPAAgent(config)
    return (f"已创建 Agent:\n"
            f"  state_dim = {config.state_dim}\n"
            f"  action_dim = {config.action_dim}\n"
            f"  max_steps = {config.max_steps}")


def _demo_quickstart_2() -> str:
    """快速入门步骤2演示：运行 Agent。"""
    from spepa import SPEPAAgent, get_preset_config
    config = get_preset_config("minimal")
    agent = SPEPAAgent(config)
    result = agent.run(n_steps=5)
    return (f"运行 5 步完成:\n"
            f"  total_steps = {result['total_steps']}\n"
            f"  online_cycles = {result['online_cycles']}\n"
            f"  offline_cycles = {result['offline_cycles']}\n"
            f"  total_time = {result['total_time']:.3f}s")


def _demo_quickstart_3() -> str:
    """快速入门步骤3演示：查看指标。"""
    from spepa import SPEPAAgent, get_preset_config
    config = get_preset_config("minimal")
    agent = SPEPAAgent(config)
    agent.run(n_steps=5)
    summary = agent.metrics.summary()
    lines = ["指标摘要:"]
    for key in ["prediction_error", "confidence", "exploration_ratio"]:
        if key in summary:
            m = summary[key]
            lines.append(f"  {key}: mean={m['mean']:.4f}, last={m['last']:.4f}")
    return "\n".join(lines)


def _demo_quickstart_4() -> str:
    """快速入门步骤4演示：使用预设。"""
    from spepa import list_presets
    presets = list_presets()
    lines = ["可用预设:"]
    for name, desc in presets.items():
        lines.append(f"  {name:15s} - {desc}")
    return "\n".join(lines)


def _demo_quickstart_5() -> str:
    """快速入门步骤5演示：快照。"""
    from spepa import SPEPAAgent, snapshot, get_preset_config
    config = get_preset_config("minimal")
    agent = SPEPAAgent(config)
    agent.run(n_steps=3)
    snap = snapshot(agent)
    return (f"Agent 快照:\n"
            f"  step_count = {snap['runtime']['step_count']}\n"
            f"  health_score = {snap['health']['score']}\n"
            f"  state_norm = {snap['state']['norm']:.4f}")


def _demo_config_1() -> str:
    """配置教程步骤1演示：默认配置。"""
    from spepa import get_default_config
    config = get_default_config()
    return (f"默认配置:\n"
            f"  state_dim = {config.state_dim}\n"
            f"  action_dim = {config.action_dim}\n"
            f"  learning_rate_micro = {config.learning_rate_micro}\n"
            f"  safety_hard_boundary = {config.safety_hard_boundary}")


def _demo_config_2() -> str:
    """配置教程步骤2演示：保存与加载。"""
    import os
    import tempfile
    from spepa import get_default_config
    config = get_default_config()
    config.seed = 42
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "my_config.json")
        config.save_to_file(filepath)
        from spepa import SPEPAConfig
        loaded = SPEPAConfig.load_from_file(filepath)
        return (f"配置已保存并重新加载:\n"
                f"  原始 seed = {config.seed}\n"
                f"  加载 seed = {loaded.seed}\n"
                f"  一致: {config.seed == loaded.seed}")


def _demo_config_3() -> str:
    """配置教程步骤3演示：merge 部分覆盖。"""
    from spepa import get_default_config
    config = get_default_config()
    new_config = config.merge({"state_dim": 32, "seed": 99})
    return (f"合并配置:\n"
            f"  原 state_dim = {config.state_dim}, 新 = {new_config.state_dim}\n"
            f"  原 seed = {config.seed}, 新 = {new_config.seed}\n"
            f"  原配置未变: {config.state_dim == 16}")


def _demo_env_1() -> str:
    """环境教程步骤1演示：创建环境。"""
    from spepa import SimpleEnvironment
    env = SimpleEnvironment(state_dim=8, action_dim=4, seed=42)
    obs = env.reset()
    return (f"已创建 SimpleEnvironment:\n"
            f"  observation_dim = {env.observation_dim}\n"
            f"  action_dim = {env.action_dim}\n"
            f"  初始观测 = {obs[:4]}...")


def _demo_env_2() -> str:
    """环境教程步骤2演示：注入 Agent。"""
    import numpy as np
    from spepa import SPEPAAgent, SimpleEnvironment, get_default_config
    env = SimpleEnvironment(state_dim=16, action_dim=8, seed=42)
    config = get_default_config()
    agent = SPEPAAgent(config, environment=env)
    result = agent.run(n_steps=3)
    return (f"Agent + 环境运行 3 步:\n"
            f"  total_steps = {result['total_steps']}\n"
            f"  state_norm = {np.linalg.norm(agent.current_state):.4f}")


def _demo_debug_1() -> str:
    """调试教程步骤1演示：快照。"""
    from spepa import SPEPAAgent, snapshot, format_snapshot, get_preset_config
    config = get_preset_config("minimal")
    agent = SPEPAAgent(config)
    agent.run(n_steps=3)
    snap = snapshot(agent)
    text = format_snapshot(snap)
    # 只返回前几行
    lines = text.split('\n')[:10]
    return '\n'.join(lines) + "\n  ..."


def _demo_debug_2() -> str:
    """调试教程步骤2演示：健康检查。"""
    from spepa import SPEPAAgent, health_check, get_preset_config
    config = get_preset_config("minimal")
    agent = SPEPAAgent(config)
    agent.run(n_steps=3)
    result = health_check(agent)
    return (f"健康检查:\n"
            f"  healthy = {result['healthy']}\n"
            f"  health_score = {result['health_score']:.4f}\n"
            f"  issues = {len(result['issues'])} 个\n"
            f"  recommendations = {len(result['recommendations'])} 条")


def _build_tutorials() -> Dict[str, Tutorial]:
    """构建所有内置教程。"""
    tutorials = {}

    # 1. 快速入门教程
    tutorials["quickstart"] = Tutorial(
        name="quickstart",
        title="SPEPA 5分钟快速入门",
        description="从零开始，5步掌握 SPEPA 的基本用法",
        steps=[
            TutorialStep(
                title="创建你的第一个 Agent",
                concept=(
                    "SPEPA Agent 是系统的核心。创建 Agent 时需要一个配置对象，\n"
                    "配置定义了状态维度、动作维度、学习率等参数。\n\n"
                    "最简单的方式是使用预设配置：\n"
                    "  - minimal    : 最小资源，适合测试\n"
                    "  - balanced   : 平衡配置（默认）\n"
                    "  - performance: 高性能\n"
                    "  - conservative: 保守安全"
                ),
                demo=_demo_quickstart_1,
                tip="预设配置让你无需理解所有参数即可开始使用 SPEPA",
            ),
            TutorialStep(
                title="运行 Agent",
                concept=(
                    "Agent 通过 run() 方法运行。每一步包含：\n"
                    "  1. 感知：接收观测，检测预测缺口\n"
                    "  2. 决策：基于主动推理选择动作\n"
                    "  3. 行动：执行动作，影响环境\n"
                    "  4. 学习：更新预测模型\n\n"
                    "运行结果包含步数、周期数、耗时等统计信息。"
                ),
                demo=_demo_quickstart_2,
                tip="低负载时 Agent 会自动触发离线闭环（记忆巩固）",
            ),
            TutorialStep(
                title="查看运行指标",
                concept=(
                    "Agent 内置指标追踪器，记录每步的关键指标：\n"
                    "  - prediction_error : 预测误差\n"
                    "  - confidence       : 信心度\n"
                    "  - exploration_ratio: 探索比例\n\n"
                    "通过 agent.metrics.summary() 获取统计摘要。"
                ),
                demo=_demo_quickstart_3,
                tip="指标趋势是判断 Agent 学习效果的关键",
            ),
            TutorialStep(
                title="选择合适的预设",
                concept=(
                    "不同场景需要不同配置：\n"
                    "  - 嵌入式设备 → minimal\n"
                    "  - 通用场景   → balanced\n"
                    "  - 服务器/研究 → performance\n"
                    "  - 安全关键   → conservative\n\n"
                    "使用 list_presets() 查看所有预设。"
                ),
                demo=_demo_quickstart_4,
                tip="预设只是起点，你可以基于预设进一步自定义",
            ),
            TutorialStep(
                title="使用调试快照",
                concept=(
                    "snapshot() 函数可以一键捕获 Agent 的完整状态：\n"
                    "  - 运行时状态（步数、周期数）\n"
                    "  - 状态向量统计\n"
                    "  - 健康分数与告警\n"
                    "  - 资源使用情况\n"
                    "  - 记忆、目标、模型状态\n\n"
                    "format_snapshot() 将快照格式化为可读文本。"
                ),
                demo=_demo_quickstart_5,
                tip="在运行前后各拍一次快照，可以对比 Agent 的变化",
            ),
        ],
    )

    # 2. 配置教程
    tutorials["config"] = Tutorial(
        name="config",
        title="配置系统详解",
        description="掌握 SPEPA 的配置管理：默认值、预设、持久化、合并",
        steps=[
            TutorialStep(
                title="默认配置",
                concept=(
                    "get_default_config() 返回平衡的默认配置。\n"
                    "配置包含 20+ 参数，分为：\n"
                    "  - 维度参数（state_dim, action_dim, obs_dim）\n"
                    "  - 学习参数（learning_rate_micro/struct）\n"
                    "  - 记忆参数（fast/slow_memory_capacity）\n"
                    "  - 安全参数（safety_hard_boundary）\n"
                    "  - 资源限制（resources 子配置）\n"
                    "  - 探索配置（exploration 子配置）\n"
                    "  - 驱力权重（drives 子配置）"
                ),
                demo=_demo_config_1,
                tip="所有参数都有合理默认值，零配置即可运行",
            ),
            TutorialStep(
                title="保存与加载配置",
                concept=(
                    "配置可以持久化到 JSON 或 YAML 文件：\n"
                    "  config.save_to_file('my.json')  → 保存\n"
                    "  SPEPAConfig.load_from_file('my.json') → 加载\n\n"
                    "加载时会自动校验配置合法性。"
                ),
                demo=_demo_config_2,
                tip="配置文件可以版本控制，方便团队共享",
            ),
            TutorialStep(
                title="合并配置（部分覆盖）",
                concept=(
                    "merge() 方法允许部分覆盖配置，返回新配置：\n"
                    "  new = config.merge({'state_dim': 32, 'seed': 99})\n\n"
                    "特点：\n"
                    "  - 不修改原配置（不可变模式）\n"
                    "  - 支持嵌套字典合并\n"
                    "  - 适合基于预设微调"
                ),
                demo=_demo_config_3,
                tip="merge 是创建自定义配置的推荐方式",
            ),
        ],
    )

    # 3. 环境教程
    tutorials["environment"] = Tutorial(
        name="environment",
        title="环境适配器使用",
        description="学习如何使用标准环境接口和内置适配器",
        steps=[
            TutorialStep(
                title="创建环境",
                concept=(
                    "SPEPA 提供标准 Environment 接口：\n"
                    "  - reset() → 返回初始观测\n"
                    "  - step(action) → 返回 (obs, reward, done, info)\n\n"
                    "内置环境：\n"
                    "  - SimpleEnvironment : 线性动力学+噪声\n"
                    "  - StaticEnvironment : 固定观测（测试用）\n"
                    "  - FunctionEnvironment : 包装现有回调\n"
                    "  - GymEnvironment : 适配 OpenAI Gym"
                ),
                demo=_demo_env_1,
                tip="SimpleEnvironment 是学习和测试的最佳起点",
            ),
            TutorialStep(
                title="将环境注入 Agent",
                concept=(
                    "Agent 接受可选的 environment 参数：\n"
                    "  agent = SPEPAAgent(config, environment=env)\n\n"
                    "注入后，Agent 会自动使用环境的 step() 方法，\n"
                    "无需手动传递 env_step_fn 回调。"
                ),
                demo=_demo_env_2,
                tip="不传 environment 时，Agent 使用内置默认环境",
            ),
        ],
    )

    # 4. 调试教程
    tutorials["debug"] = Tutorial(
        name="debug",
        title="调试工具使用",
        description="掌握快照、健康检查、性能分析等调试工具",
        steps=[
            TutorialStep(
                title="状态快照",
                concept=(
                    "snapshot(agent) 捕获 Agent 完整状态：\n"
                    "  - runtime : 运行时信息\n"
                    "  - state   : 状态向量统计\n"
                    "  - health  : 健康分数与告警\n"
                    "  - resources : 资源使用\n"
                    "  - memory  : 记忆系统\n"
                    "  - goals   : 目标层次\n"
                    "  - model   : 预测模型\n\n"
                    "format_snapshot() 格式化为可读文本。"
                ),
                demo=_demo_debug_1,
                tip="快照是只读的，不会影响 Agent 运行",
            ),
            TutorialStep(
                title="健康检查",
                concept=(
                    "health_check(agent) 执行全面诊断：\n"
                    "  - 检查状态向量是否正常\n"
                    "  - 检查是否有告警\n"
                    "  - 检查资源使用\n"
                    "  - 检查降级级别\n"
                    "  - 检查指标趋势\n\n"
                    "返回 issues（问题列表）和 recommendations（建议列表）。"
                ),
                demo=_demo_debug_2,
                tip="定期运行健康检查可以及早发现问题",
            ),
        ],
    )

    return tutorials


# 教程注册表
_TUTORIALS: Optional[Dict[str, Tutorial]] = None


def _get_tutorials() -> Dict[str, Tutorial]:
    """获取教程注册表（懒加载）。"""
    global _TUTORIALS
    if _TUTORIALS is None:
        _TUTORIALS = _build_tutorials()
    return _TUTORIALS


# ============================================================
# 公共 API
# ============================================================

def list_tutorials() -> Dict[str, str]:
    """列出所有可用教程。

    返回：
        {tutorial_name: "title - description"}
    """
    tutorials = _get_tutorials()
    return {name: f"{t.title} - {t.description}" for name, t in tutorials.items()}


def get_tutorial(name: str) -> Tutorial:
    """获取指定教程。

    参数：
        name : 教程名称

    返回：
        Tutorial 实例

    异常：
        KeyError : 教程不存在
    """
    tutorials = _get_tutorials()
    if name not in tutorials:
        raise KeyError(
            f"教程 '{name}' 不存在。可用: {', '.join(tutorials.keys())}"
        )
    return tutorials[name]


def run_tutorial(name: str, interactive: bool = True,
                 pause_fn: Optional[Callable] = None) -> None:
    """运行指定教程。

    参数：
        name : 教程名称
        interactive : 是否交互模式（每步暂停）
        pause_fn : 自定义暂停函数

    示例：
        >>> run_tutorial("quickstart")  # 交互模式
        >>> run_tutorial("quickstart", interactive=False)  # 非交互
    """
    tutorial = get_tutorial(name)
    tutorial.run(interactive=interactive, pause_fn=pause_fn)


def register_tutorial(tutorial: Tutorial) -> None:
    """注册自定义教程。

    参数：
        tutorial : Tutorial 实例
    """
    tutorials = _get_tutorials()
    tutorials[tutorial.name] = tutorial
    logger.info("Tutorial registered: %s", tutorial.name)
