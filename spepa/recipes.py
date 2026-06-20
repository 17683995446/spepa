"""SPEPA 示例库（Recipes）。

设计原则（第一性原理）：
- 可运行：每个示例都是可直接执行的代码，而非伪代码
- 可复制：一键复制代码，降低上手成本
- 可分类：按难度/主题/场景组织，便于查找
- 可扩展：用户可注册自定义示例

与 tutorial 模块的区别：
- tutorial : 引导式学习（步骤化交互教程）
- recipes  : 参考式代码片段（复制即用的代码示例）

主要功能：
- Recipe          : 示例数据结构
- RecipeLibrary   : 示例库（内置 + 自定义）
- get_recipe()    : 获取示例
- list_recipes()  : 列出所有示例
- run_recipe()    : 运行示例代码
- format_recipe() : 格式化示例为可读文本
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from spepa.utils.logger import get_logger

logger = get_logger("spepa.recipes")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Recipe:
    """代码示例。

    一个示例 = 标题 + 描述 + 代码 + 元信息。
    """
    name: str                  # 唯一标识（如 "quick_start"）
    title: str                 # 标题（如 "快速开始"）
    description: str           # 描述
    code: str                  # 可执行代码
    difficulty: str = "beginner"  # beginner / intermediate / advanced
    tags: List[str] = field(default_factory=list)  # 标签
    outputs: List[str] = field(default_factory=list)  # 预期输出说明


# ============================================================
# 内置示例
# ============================================================

def _builtin_recipes() -> List[Recipe]:
    """返回内置示例列表。"""
    return [
        Recipe(
            name="quick_start",
            title="快速开始",
            description="最小化示例：创建 Agent 并运行 10 步",
            difficulty="beginner",
            tags=["basic", "agent"],
            code="""from spepa import SPEPAAgent, get_preset_config

# 使用 minimal 预设（资源占用最小）
config = get_preset_config("minimal")
agent = SPEPAAgent(config)

# 运行 10 步
result = agent.run(n_steps=10)
print(f"完成 {result['total_steps']} 步，耗时 {result['total_time']:.2f}s")
""",
            outputs=["完成 10 步，耗时 0.0Xs"],
        ),
        Recipe(
            name="custom_config",
            title="自定义配置",
            description="通过 merge 方法自定义配置参数",
            difficulty="beginner",
            tags=["config"],
            code="""from spepa import SPEPAAgent, get_preset_config

# 基于 balanced 预设，覆盖部分参数
config = get_preset_config("balanced").merge({
    "state_dim": 32,
    "learning_rate_micro": 0.02,
    "max_steps": 200,
    "seed": 42,
})
agent = SPEPAAgent(config)
agent.run(n_steps=50)
""",
        ),
        Recipe(
            name="save_and_load",
            title="保存与加载状态",
            description="保存 Agent 状态到检查点，后续加载恢复",
            difficulty="intermediate",
            tags=["persistence"],
            code="""from spepa import SPEPAAgent, get_preset_config, save_state, load_state

config = get_preset_config("minimal")
agent = SPEPAAgent(config)
agent.run(n_steps=20)

# 保存
save_state(agent, force=True)
print(f"已保存，步数: {agent.step_count}")

# 加载（新 Agent）
agent2 = SPEPAAgent(config)
loaded = load_state(agent2)
print(f"加载: {loaded}, 步数: {agent2.step_count}")
""",
        ),
        Recipe(
            name="diagnose_agent",
            title="诊断 Agent 状态",
            description="运行后诊断 Agent 健康状况",
            difficulty="intermediate",
            tags=["debugging"],
            code="""from spepa import SPEPAAgent, get_preset_config, diagnose, format_diagnosis

agent = SPEPAAgent(get_preset_config("minimal"))
agent.run(n_steps=30)

result = diagnose(agent)
print(format_diagnosis(result))
print(f"健康: {result['healthy']}, 问题数: {len(result['issues'])}")
""",
        ),
        Recipe(
            name="visualization",
            title="生成可视化图表",
            description="运行后生成 ASCII 图表和 SVG 图表",
            difficulty="intermediate",
            tags=["visualization"],
            code="""from spepa import SPEPAAgent, get_preset_config, render_ascii_chart, plot_metric

agent = SPEPAAgent(get_preset_config("minimal"))
agent.run(n_steps=50)

# 终端 ASCII 图表
print(render_ascii_chart(agent, metric_name="prediction_error"))

# 保存 SVG 图表到文件
plot_metric(agent, "prediction_error", filepath="error_chart.svg")
""",
        ),
        Recipe(
            name="batch_comparison",
            title="批量对比预设",
            description="运行多个预设配置并对比结果",
            difficulty="advanced",
            tags=["batch", "benchmarking"],
            code="""from spepa import run_preset_comparison, format_batch_results

# 运行所有预设，每个 30 步
results = run_preset_comparison(n_steps=30)
print(format_batch_results(results))
""",
        ),
        Recipe(
            name="param_sweep",
            title="参数扫描",
            description="扫描学习率参数，观察对收敛的影响",
            difficulty="advanced",
            tags=["batch", "experiment"],
            code="""from spepa import get_preset_config, run_param_sweep, format_batch_results

base = get_preset_config("minimal")
results = run_param_sweep(
    base_config=base,
    param_name="learning_rate_micro",
    values=[0.001, 0.01, 0.05, 0.1],
    n_steps=30,
)
print(format_batch_results(results))
""",
        ),
        Recipe(
            name="config_advisor",
            title="配置建议",
            description="验证配置合理性并获取优化建议",
            difficulty="intermediate",
            tags=["config"],
            code="""from spepa import get_preset_config, validate_config, format_advice

config = get_preset_config("balanced")
result = validate_config(config)
print(format_advice(result))
print(f"健康分: {result['score']:.1f}/100")
""",
        ),
        Recipe(
            name="repl_usage",
            title="REPL 交互式使用",
            description="通过 REPL Shell 交互式探索 Agent",
            difficulty="beginner",
            tags=["repl"],
            code="""from spepa import start_repl, get_preset_config

# 启动交互式 Shell
# 可用命令: help, run, step, state, metrics, diagnose, chart, exit
start_repl(config=get_preset_config("minimal"))
""",
        ),
        Recipe(
            name="auto_save",
            title="自动保存",
            description="使用 auto_save 上下文管理器定期保存",
            difficulty="advanced",
            tags=["persistence"],
            code="""from spepa import SPEPAAgent, get_preset_config, StateManager

agent = SPEPAAgent(get_preset_config("minimal"))
mgr = StateManager(agent)

# 运行时每 10 步自动保存
with mgr.auto_save(interval=10):
    agent.run(n_steps=50)

print(f"运行完成，步数: {agent.step_count}")
""",
        ),
        Recipe(
            name="error_handling",
            title="错误处理",
            description="使用 safe_run 和 ErrorContext 优雅处理错误",
            difficulty="intermediate",
            tags=["error"],
            code="""from spepa import safe_run, ErrorContext

@safe_run
def risky_operation():
    raise FileNotFoundError("config.json 不存在")

# 不会抛异常，而是输出友好错误信息
result = risky_operation()
print(f"结果: {result}")

# 使用上下文管理器
with ErrorContext("加载配置"):
    # 这里的异常会被捕获并友好输出
    pass
""",
        ),
        Recipe(
            name="custom_environment",
            title="自定义环境",
            description="使用 FunctionEnvironment 创建自定义环境",
            difficulty="advanced",
            tags=["environment"],
            code="""from spepa import SPEPAAgent, get_preset_config, FunctionEnvironment
import numpy as np

# 定义环境步进函数
def step_fn(action):
    state = np.random.rand(8)
    reward = float(np.sum(action))
    return state, reward

env = FunctionEnvironment(step_fn=step_fn, state_dim=8)
config = get_preset_config("minimal")
agent = SPEPAAgent(config)
agent.run(n_steps=20, env_step_fn=env.step)
""",
        ),
    ]


# ============================================================
# 示例库
# ============================================================

class RecipeLibrary:
    """示例库。

    使用方式：
        lib = RecipeLibrary()
        recipe = lib.get("quick_start")
        print(recipe.code)
        lib.run("quick_start")
    """

    def __init__(self):
        self._recipes: Dict[str, Recipe] = {}
        self._load_builtins()

    def _load_builtins(self) -> None:
        """加载内置示例。"""
        for recipe in _builtin_recipes():
            self._recipes[recipe.name] = recipe

    def add(self, recipe: Recipe) -> None:
        """添加自定义示例。

        参数：
            recipe : Recipe 实例

        抛出：
            ValueError 当示例名重复时
        """
        if recipe.name in self._recipes:
            raise ValueError(f"示例 '{recipe.name}' 已存在")
        self._recipes[recipe.name] = recipe

    def get(self, name: str) -> Optional[Recipe]:
        """获取示例。

        参数：
            name : 示例名

        返回：
            Recipe 实例，不存在则返回 None
        """
        return self._recipes.get(name)

    def list(self,
             difficulty: Optional[str] = None,
             tag: Optional[str] = None) -> List[Recipe]:
        """列出示例（可按难度/标签筛选）。

        参数：
            difficulty : 按难度筛选（beginner/intermediate/advanced）
            tag : 按标签筛选

        返回：
            Recipe 列表
        """
        result = list(self._recipes.values())
        if difficulty:
            result = [r for r in result if r.difficulty == difficulty]
        if tag:
            result = [r for r in result if tag in r.tags]
        return sorted(result, key=lambda r: r.name)

    def list_names(self) -> List[str]:
        """列出所有示例名。"""
        return sorted(self._recipes.keys())

    def list_difficulties(self) -> List[str]:
        """列出所有难度级别。"""
        return sorted(set(r.difficulty for r in self._recipes.values()))

    def list_tags(self) -> List[str]:
        """列出所有标签。"""
        tags = set()
        for r in self._recipes.values():
            tags.update(r.tags)
        return sorted(tags)

    def run(self, name: str, globals_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """运行示例代码。

        参数：
            name : 示例名
            globals_dict : 自定义全局命名空间（默认使用新命名空间）

        返回：
            {"success": bool, "error": Optional[str]}

        抛出：
            ValueError 当示例不存在时
        """
        recipe = self.get(name)
        if recipe is None:
            raise ValueError(f"示例 '{name}' 不存在。可用: {self.list_names()}")

        if globals_dict is None:
            globals_dict = {"__name__": "__recipe__", "__builtins__": __builtins__}

        try:
            exec(recipe.code, globals_dict)
            return {"success": True, "error": None}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def format(self, name: str, with_code: bool = True) -> str:
        """格式化示例为可读文本。

        参数：
            name : 示例名
            with_code : 是否包含代码

        返回：
            格式化的字符串
        """
        recipe = self.get(name)
        if recipe is None:
            return f"未找到示例: {name}"

        lines = []
        lines.append("=" * 60)
        lines.append(f"示例: {recipe.title}")
        lines.append("=" * 60)
        lines.append(f"\n名称: {recipe.name}")
        lines.append(f"难度: {recipe.difficulty}")
        if recipe.tags:
            lines.append(f"标签: {', '.join(recipe.tags)}")
        lines.append(f"\n描述: {recipe.description}")
        if recipe.outputs:
            lines.append(f"\n预期输出:")
            for out in recipe.outputs:
                lines.append(f"  - {out}")
        if with_code:
            lines.append(f"\n代码:")
            lines.append("```python")
            lines.append(recipe.code)
            lines.append("```")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def format_list(self, recipes: Optional[List[Recipe]] = None) -> str:
        """格式化示例列表为可读文本。

        参数：
            recipes : 要列出的示例（默认全部）

        返回：
            格式化的字符串
        """
        if recipes is None:
            recipes = self.list()
        lines = []
        lines.append("=" * 60)
        lines.append(f"SPEPA 示例库（共 {len(recipes)} 个）")
        lines.append("=" * 60)
        for r in recipes:
            tags_str = f" [{', '.join(r.tags)}]" if r.tags else ""
            lines.append(f"\n  {r.name:<25} ({r.difficulty}){tags_str}")
            lines.append(f"    {r.title}: {r.description}")
        lines.append("\n" + "=" * 60)
        lines.append("提示: 使用 get_recipe(name) 查看详情，run_recipe(name) 运行")
        return "\n".join(lines)


# ============================================================
# 模块级便捷 API
# ============================================================

_global_library: Optional[RecipeLibrary] = None


def _get_library() -> RecipeLibrary:
    """获取全局示例库。"""
    global _global_library
    if _global_library is None:
        _global_library = RecipeLibrary()
    return _global_library


def get_recipe(name: str) -> Optional[Recipe]:
    """获取示例。"""
    return _get_library().get(name)


def list_recipes(difficulty: Optional[str] = None,
                 tag: Optional[str] = None) -> List[Recipe]:
    """列出示例（可筛选）。"""
    return _get_library().list(difficulty=difficulty, tag=tag)


def run_recipe(name: str, globals_dict: Optional[Dict] = None) -> Dict[str, Any]:
    """运行示例代码。"""
    return _get_library().run(name, globals_dict=globals_dict)


def format_recipe(name: str, with_code: bool = True) -> str:
    """格式化示例为可读文本。"""
    return _get_library().format(name, with_code=with_code)


def format_recipe_list(difficulty: Optional[str] = None,
                       tag: Optional[str] = None) -> str:
    """格式化示例列表。"""
    recipes = list_recipes(difficulty=difficulty, tag=tag)
    return _get_library().format_list(recipes)


def add_recipe(recipe: Recipe) -> None:
    """添加自定义示例到全局库。"""
    _get_library().add(recipe)


def list_recipe_tags() -> List[str]:
    """列出所有示例标签。"""
    return _get_library().list_tags()


def list_recipe_difficulties() -> List[str]:
    """列出所有难度级别。"""
    return _get_library().list_difficulties()
