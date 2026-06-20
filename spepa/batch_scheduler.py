"""SPEPA 批处理任务调度器。

设计原则（第一性原理）：
- 批量化：一次性运行多个配置/任务，自动收集结果
- 依赖感知：支持任务间依赖，按拓扑排序执行
- 容错：支持失败重试，单个任务失败不影响其他独立任务
- 可观测：进度回调 + 结果汇总，全程可见
- 隔离：每个任务创建独立 Agent，避免状态污染

与 benchmark 模块的区别：
- benchmark : 专注"性能测量"（多次运行取平均，测时间/内存）
- batch_scheduler : 专注"任务编排"（依赖、重试、批量对比）

主要功能：
- BatchTask        : 批处理任务数据结构
- BatchResult      : 任务结果数据结构
- BatchScheduler   : 批处理调度器（依赖排序 + 重试 + 进度）
- run_batch()      : 一键运行批处理
- run_param_sweep() : 参数扫描
- run_preset_comparison() : 预设对比
- format_batch_results() : 格式化结果
"""
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field

from spepa.utils.config import SPEPAConfig, get_preset_config, list_presets
from spepa.utils.logger import get_logger
from spepa.agent import SPEPAAgent
from spepa.diagnostics import friendly_error

logger = get_logger("spepa.batch_scheduler")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BatchTask:
    """批处理任务。

    一个任务 = 一份配置 + 运行步数 + 依赖 + 重试策略。
    """
    name: str                              # 任务名（唯一标识）
    config: SPEPAConfig                    # 使用的配置
    n_steps: int = 100                     # 运行步数
    depends_on: List[str] = field(default_factory=list)  # 依赖的任务名
    retries: int = 0                       # 失败重试次数
    tags: List[str] = field(default_factory=list)        # 标签（用于分组/筛选）
    env_step_fn: Optional[Callable] = None  # 自定义环境步进函数


@dataclass
class BatchResult:
    """任务运行结果。"""
    task_name: str                         # 任务名
    success: bool                          # 是否成功
    config: Optional[SPEPAConfig] = None   # 使用的配置
    metrics: Dict[str, Any] = field(default_factory=dict)  # 指标摘要
    duration: float = 0.0                  # 运行时长（秒）
    error: Optional[str] = None            # 错误信息
    attempts: int = 0                      # 尝试次数
    steps_executed: int = 0                # 实际执行步数


# ============================================================
# 批处理调度器
# ============================================================

class BatchScheduler:
    """批处理调度器。

    使用方式：
        scheduler = BatchScheduler()
        scheduler.add_task(BatchTask(name="exp1", config=config1, n_steps=50))
        scheduler.add_task(BatchTask(name="exp2", config=config2, n_steps=50))
        results = scheduler.run()
        print(format_batch_results(results))
    """

    def __init__(self, on_progress: Optional[Callable[[str, int, int], None]] = None):
        """初始化调度器。

        参数：
            on_progress : 进度回调 (task_name, completed, total)
        """
        self.tasks: Dict[str, BatchTask] = {}
        self.results: Dict[str, BatchResult] = {}
        self.on_progress = on_progress
        self._completed_count = 0

    def add_task(self, task: BatchTask) -> "BatchScheduler":
        """添加单个任务。

        参数：
            task : BatchTask 实例

        返回：
            self（支持链式调用）

        抛出：
            ValueError 当任务名重复时
        """
        if task.name in self.tasks:
            raise ValueError(f"任务名 '{task.name}' 已存在")
        self.tasks[task.name] = task
        return self

    def add_tasks(self, tasks: List[BatchTask]) -> "BatchScheduler":
        """批量添加任务。

        参数：
            tasks : BatchTask 列表

        返回：
            self
        """
        for task in tasks:
            self.add_task(task)
        return self

    def run(self) -> Dict[str, BatchResult]:
        """运行所有任务（按依赖拓扑排序）。

        返回：
            {task_name: BatchResult}
        """
        self.results = {}
        self._completed_count = 0
        order = self._topological_sort()
        total = len(order)

        logger.info("BatchScheduler: 开始运行 %d 个任务", total)

        for task_name in order:
            task = self.tasks[task_name]
            # 检查依赖是否成功
            deps_ok = self._check_dependencies(task)
            if not deps_ok:
                failed_deps = [d for d in task.depends_on
                               if d in self.results and not self.results[d].success]
                self.results[task_name] = BatchResult(
                    task_name=task_name,
                    success=False,
                    error=f"依赖任务失败: {failed_deps}",
                )
                self._completed_count += 1
                self._notify_progress(task_name, total)
                continue

            # 运行任务（含重试）
            result = self._run_task(task)
            self.results[task_name] = result
            self._completed_count += 1
            self._notify_progress(task_name, total)

        logger.info("BatchScheduler: 完成，成功 %d / %d",
                    sum(1 for r in self.results.values() if r.success), total)
        return self.results

    def _topological_sort(self) -> List[str]:
        """拓扑排序任务（按依赖关系）。

        返回：
            排序后的任务名列表

        抛出：
            ValueError 当存在循环依赖或缺失依赖时
        """
        # 检查依赖是否存在
        for task in self.tasks.values():
            for dep in task.depends_on:
                if dep not in self.tasks:
                    raise ValueError(
                        f"任务 '{task.name}' 依赖不存在的任务 '{dep}'"
                    )

        # Kahn 算法
        in_degree: Dict[str, int] = {name: 0 for name in self.tasks}
        for task in self.tasks.values():
            for dep in task.depends_on:
                in_degree[task.name] += 1

        # 入度为 0 的节点
        queue = [name for name, deg in in_degree.items() if deg == 0]
        queue.sort()  # 确保确定性
        result: List[str] = []

        while queue:
            name = queue.pop(0)
            result.append(name)
            # 找到依赖 name 的任务，入度减 1
            for task in self.tasks.values():
                if name in task.depends_on:
                    in_degree[task.name] -= 1
                    if in_degree[task.name] == 0:
                        # 插入并保持排序
                        queue.append(task.name)
                        queue.sort()

        if len(result) != len(self.tasks):
            # 存在循环依赖
            remaining = [n for n in self.tasks if n not in result]
            raise ValueError(
                f"检测到循环依赖，涉及任务: {remaining}"
            )

        return result

    def _check_dependencies(self, task: BatchTask) -> bool:
        """检查任务的所有依赖是否成功完成。"""
        for dep in task.depends_on:
            if dep not in self.results:
                return False
            if not self.results[dep].success:
                return False
        return True

    def _run_task(self, task: BatchTask) -> BatchResult:
        """运行单个任务（含重试）。

        参数：
            task : BatchTask 实例

        返回：
            BatchResult
        """
        max_attempts = task.retries + 1
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            logger.info("运行任务 '%s' (尝试 %d/%d)", task.name, attempt, max_attempts)
            try:
                # 每次创建独立 Agent，避免状态污染
                agent = SPEPAAgent(task.config)
                start_time = time.time()
                run_result = agent.run(n_steps=task.n_steps,
                                       env_step_fn=task.env_step_fn)
                duration = time.time() - start_time

                return BatchResult(
                    task_name=task.name,
                    success=True,
                    config=task.config,
                    metrics=run_result.get("metrics", {}),
                    duration=duration,
                    attempts=attempt,
                    steps_executed=run_result.get("total_steps", task.n_steps),
                )
            except Exception as e:
                last_error = friendly_error(e, context=f"批处理任务 '{task.name}'")
                logger.warning("任务 '%s' 尝试 %d 失败: %s", task.name, attempt, e)

        return BatchResult(
            task_name=task.name,
            success=False,
            config=task.config,
            error=last_error,
            attempts=max_attempts,
        )

    def _notify_progress(self, task_name: str, total: int):
        """通知进度。"""
        if self.on_progress:
            try:
                self.on_progress(task_name, self._completed_count, total)
            except Exception:
                pass  # 回调失败不影响主流程

    def summary(self) -> Dict[str, Any]:
        """返回运行汇总。"""
        results = list(self.results.values())
        success_count = sum(1 for r in results if r.success)
        total_duration = sum(r.duration for r in results)
        return {
            "total": len(results),
            "success": success_count,
            "failed": len(results) - success_count,
            "total_duration": total_duration,
            "success_rate": success_count / len(results) if results else 0.0,
        }

    def get_successful_results(self) -> Dict[str, BatchResult]:
        """获取所有成功的结果。"""
        return {name: r for name, r in self.results.items() if r.success}

    def get_failed_results(self) -> Dict[str, BatchResult]:
        """获取所有失败的结果。"""
        return {name: r for name, r in self.results.items() if not r.success}

    def clear(self) -> None:
        """清空所有任务和结果。"""
        self.tasks = {}
        self.results = {}
        self._completed_count = 0


# ============================================================
# 便捷函数
# ============================================================

def run_batch(tasks: List[BatchTask],
              on_progress: Optional[Callable[[str, int, int], None]] = None
              ) -> Dict[str, BatchResult]:
    """一键运行批处理任务。

    参数：
        tasks : BatchTask 列表
        on_progress : 进度回调

    返回：
        {task_name: BatchResult}
    """
    scheduler = BatchScheduler(on_progress=on_progress)
    scheduler.add_tasks(tasks)
    return scheduler.run()


def run_param_sweep(base_config: SPEPAConfig,
                    param_name: str,
                    values: List[Any],
                    n_steps: int = 50,
                    nested: bool = False,
                    on_progress: Optional[Callable[[str, int, int], None]] = None
                    ) -> Dict[str, BatchResult]:
    """参数扫描：固定其他参数，变化单个参数运行多个任务。

    参数：
        base_config : 基础配置
        param_name : 要扫描的参数名（如 "learning_rate_micro"）
        values : 参数值列表
        n_steps : 每个任务运行步数
        nested : 参数是否为嵌套（如 "resources.cpu_max_percent"）
        on_progress : 进度回调

    返回：
        {task_name: BatchResult}，task_name 格式为 "{param_name}={value}"
    """
    tasks: List[BatchTask] = []
    for value in values:
        if nested:
            # 解析嵌套路径，如 "resources.cpu_max_percent"
            parts = param_name.split(".")
            override: Dict[str, Any] = {}
            current = override
            for part in parts[:-1]:
                current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            override = {param_name: value}

        config = base_config.merge(override)
        task_name = f"{param_name}={value}"
        tasks.append(BatchTask(
            name=task_name,
            config=config,
            n_steps=n_steps,
            tags=["param_sweep", param_name],
        ))

    return run_batch(tasks, on_progress=on_progress)


def run_preset_comparison(presets: Optional[List[str]] = None,
                          n_steps: int = 50,
                          on_progress: Optional[Callable[[str, int, int], None]] = None
                          ) -> Dict[str, BatchResult]:
    """预设对比：运行多个预设配置并对比结果。

    参数：
        presets : 预设名列表（默认全部预设）
        n_steps : 每个任务运行步数
        on_progress : 进度回调

    返回：
        {task_name: BatchResult}，task_name 为预设名
    """
    if presets is None:
        presets = list(list_presets().keys())

    tasks: List[BatchTask] = []
    for preset in presets:
        config = get_preset_config(preset)
        tasks.append(BatchTask(
            name=preset,
            config=config,
            n_steps=n_steps,
            tags=["preset_comparison"],
        ))

    return run_batch(tasks, on_progress=on_progress)


# ============================================================
# 格式化
# ============================================================

def format_batch_results(results: Dict[str, BatchResult]) -> str:
    """格式化批处理结果为可读文本。

    参数：
        results : BatchScheduler.run() 的返回值

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SPEPA 批处理结果报告")
    lines.append("=" * 70)

    total = len(results)
    success = sum(1 for r in results.values() if r.success)
    failed = total - success
    total_duration = sum(r.duration for r in results.values())

    lines.append(f"\n[汇总] 共 {total} 个任务，成功 {success}，失败 {failed}")
    lines.append(f"[总耗时] {total_duration:.2f}s")
    if total > 0:
        lines.append(f"[成功率] {success / total * 100:.1f}%")

    # 成功任务明细
    successful = {name: r for name, r in results.items() if r.success}
    if successful:
        lines.append(f"\n[成功任务] ({len(successful)} 个)")
        lines.append(f"{'任务名':<30} {'步数':<8} {'耗时':<10} {'尝试':<6}")
        lines.append("-" * 60)
        for name, r in successful.items():
            lines.append(f"{name:<30} {r.steps_executed:<8} {r.duration:<10.2f} {r.attempts:<6}")

    # 失败任务明细
    failed_tasks = {name: r for name, r in results.items() if not r.success}
    if failed_tasks:
        lines.append(f"\n[失败任务] ({len(failed_tasks)} 个)")
        for name, r in failed_tasks.items():
            lines.append(f"  - {name}: {r.error or '未知错误'}")

    # 指标对比（如果有 prediction_error）
    metrics_with_error = {
        name: r for name, r in successful.items()
        if "prediction_error" in r.metrics
    }
    if metrics_with_error:
        lines.append(f"\n[预测误差对比]")
        for name, r in metrics_with_error.items():
            err = r.metrics["prediction_error"]
            if isinstance(err, dict):
                last_err = err.get("last", 0)
                mean_err = err.get("mean", 0)
                lines.append(f"  {name:<30} last={last_err:.4f} mean={mean_err:.4f}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def format_batch_summary(summary: Dict[str, Any]) -> str:
    """格式化批处理汇总信息。

    参数：
        summary : BatchScheduler.summary() 的返回值

    返回：
        格式化的字符串
    """
    lines = []
    lines.append(f"任务总数: {summary['total']}")
    lines.append(f"成功: {summary['success']}")
    lines.append(f"失败: {summary['failed']}")
    lines.append(f"总耗时: {summary['total_duration']:.2f}s")
    lines.append(f"成功率: {summary['success_rate'] * 100:.1f}%")
    return "\n".join(lines)
