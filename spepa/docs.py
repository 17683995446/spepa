"""SPEPA 统一文档系统。

设计原则（第一性原理）：
- 自描述：文档从代码 docstring 自动提取，避免文档与代码脱节
- 可发现：按主题/模块/类别组织，支持全文搜索
- 渐进式：概览 → 模块详情 → 函数详情，逐层深入
- 零依赖：纯 Python 实现，不依赖 Sphinx 等外部工具

与 tutorial 模块的区别：
- tutorial : 引导式学习（步骤化教程）
- docs     : 参考式查阅（API 索引 + 搜索）

主要功能：
- DocEntry        : 文档条目数据结构
- DocIndex        : 文档索引（自动从包构建）
- build_doc_index() : 构建文档索引
- search_docs()   : 全文搜索
- get_module_docs() : 获取模块文档
- format_doc()    : 格式化文档为可读文本
- list_topics()   : 列出所有主题
"""
import inspect
import importlib
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from spepa.utils.logger import get_logger

logger = get_logger("spepa.docs")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class DocEntry:
    """文档条目。

    一个条目对应一个可文档化的对象（模块/类/函数/方法）。
    """
    name: str              # 完整名（如 "spepa.SPEPAAgent.run"）
    short_name: str        # 短名（如 "run"）
    kind: str              # "module" / "class" / "function" / "method"
    module: str            # 所属模块（如 "spepa.agent"）
    docstring: str = ""    # docstring
    signature: str = ""    # 函数签名（仅 function/method）
    topic: str = ""        # 主题分类（如 "core" / "config" / "io"）
    examples: List[str] = field(default_factory=list)  # 示例代码


# ============================================================
# 主题映射
# ============================================================

# 模块 → 主题分类
# 主题比模块更抽象，便于用户按"用途"而非"代码位置"查找
_MODULE_TOPICS: Dict[str, str] = {
    "spepa": "overview",
    "spepa.agent": "core",
    "spepa.utils.config": "config",
    "spepa.utils.logger": "config",
    "spepa.utils.metrics": "config",
    "spepa.environments": "environment",
    "spepa.events": "events",
    "spepa.report": "output",
    "spepa.dashboard": "output",
    "spepa.debug": "debugging",
    "spepa.tutorial": "learning",
    "spepa.benchmark": "benchmarking",
    "spepa.plugins": "extending",
    "spepa.diagnostics": "debugging",
    "spepa.state_manager": "persistence",
    "spepa.persistence.serializer": "persistence",
    "spepa.persistence.checkpoint": "persistence",
    "spepa.persistence.recovery": "persistence",
    "spepa.visualization": "output",
    "spepa.config_advisor": "config",
    "spepa.batch_scheduler": "workflow",
    "spepa.repl": "workflow",
    "spepa.cli": "workflow",
}

# 主题 → 描述
_TOPIC_DESCRIPTIONS: Dict[str, str] = {
    "overview": "总览：包级入口与版本信息",
    "core": "核心：Agent 运行与状态管理",
    "config": "配置：参数设置与调优建议",
    "environment": "环境：观测/动作/奖励接口",
    "events": "事件：事件总线与进度报告",
    "output": "输出：报告、图表、仪表盘",
    "debugging": "调试：诊断、快照、健康检查",
    "learning": "学习：教程与引导",
    "benchmarking": "基准：性能测量与对比",
    "extending": "扩展：插件系统",
    "persistence": "持久化：状态保存与恢复",
    "workflow": "工作流：批处理、REPL、CLI",
}


# ============================================================
# 文档索引
# ============================================================

class DocIndex:
    """文档索引。

    使用方式：
        index = DocIndex()
        index.build()
        entries = index.search("run")
        text = index.format_entry(entries[0])
    """

    def __init__(self):
        self.entries: List[DocEntry] = []
        self._by_name: Dict[str, DocEntry] = {}
        self._by_module: Dict[str, List[DocEntry]] = {}
        self._by_topic: Dict[str, List[DocEntry]] = {}
        self._built = False

    def build(self, package_name: str = "spepa") -> "DocIndex":
        """从包构建文档索引。

        参数：
            package_name : 要索引的包名

        返回：
            self（支持链式调用）
        """
        self.entries = []
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.error("无法导入包 %s: %s", package_name, e)
            return self

        # 索引包本身
        self._index_object(package, package_name, "module")

        # 索引包的 __all__ 导出
        all_names = getattr(package, "__all__", [])
        for name in all_names:
            obj = getattr(package, name, None)
            if obj is None:
                continue
            full_name = f"{package_name}.{name}"
            kind = self._classify_object(obj)
            self._index_object(obj, full_name, kind)

        # 索引主要子模块（顶层 .py 文件）
        submodules = [
            "spepa.agent", "spepa.environments", "spepa.events",
            "spepa.report", "spepa.dashboard", "spepa.debug",
            "spepa.tutorial", "spepa.benchmark", "spepa.plugins",
            "spepa.diagnostics", "spepa.state_manager", "spepa.visualization",
            "spepa.config_advisor", "spepa.batch_scheduler", "spepa.repl",
            "spepa.cli",
        ]
        for mod_name in submodules:
            try:
                mod = importlib.import_module(mod_name)
                self._index_object(mod, mod_name, "module")
            except ImportError:
                pass

        # 构建索引
        self._rebuild_indices()
        self._built = True
        logger.info("文档索引构建完成: %d 个条目", len(self.entries))
        return self

    def _classify_object(self, obj: Any) -> str:
        """分类对象类型。"""
        if inspect.ismodule(obj):
            return "module"
        if inspect.isclass(obj):
            return "class"
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            return "function"
        return "other"

    def _index_object(self, obj: Any, full_name: str, kind: str) -> None:
        """索引单个对象。"""
        if kind == "other":
            return

        short_name = full_name.rsplit(".", 1)[-1]
        module = getattr(obj, "__module__", full_name) or full_name
        if kind == "module":
            module = full_name
        docstring = inspect.getdoc(obj) or ""

        signature = ""
        if kind in ("function", "method"):
            try:
                signature = str(inspect.signature(obj))
            except (ValueError, TypeError):
                signature = "(...)"

        topic = _MODULE_TOPICS.get(module, "")
        if not topic and "." in module:
            # 尝试匹配父模块
            parent = module.rsplit(".", 1)[0]
            topic = _MODULE_TOPICS.get(parent, "")

        entry = DocEntry(
            name=full_name,
            short_name=short_name,
            kind=kind,
            module=module,
            docstring=docstring,
            signature=signature,
            topic=topic,
        )
        self.entries.append(entry)

        # 如果是类，索引其公开方法
        if kind == "class" and inspect.isclass(obj):
            for mname, mobj in inspect.getmembers(obj, predicate=inspect.isfunction):
                if mname.startswith("_"):
                    continue
                m_full_name = f"{full_name}.{mname}"
                m_sig = ""
                try:
                    m_sig = str(inspect.signature(mobj))
                except (ValueError, TypeError):
                    m_sig = "(...)"
                self.entries.append(DocEntry(
                    name=m_full_name,
                    short_name=mname,
                    kind="method",
                    module=module,
                    docstring=inspect.getdoc(mobj) or "",
                    signature=m_sig,
                    topic=topic,
                ))

    def _rebuild_indices(self) -> None:
        """重建查找索引。"""
        self._by_name = {e.name: e for e in self.entries}
        self._by_module = {}
        self._by_topic = {}
        for e in self.entries:
            self._by_module.setdefault(e.module, []).append(e)
            if e.topic:
                self._by_topic.setdefault(e.topic, []).append(e)

    # --------------------------------------------------------
    # 查询
    # --------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> List[DocEntry]:
        """全文搜索。

        参数：
            query : 搜索关键词
            limit : 最多返回条目数

        返回：
            匹配的 DocEntry 列表（按相关度排序）
        """
        if not self._built:
            self.build()
        query_lower = query.lower()
        scored: List[tuple] = []  # (score, entry)

        for entry in self.entries:
            score = 0
            # 名称匹配（权重最高）
            if query_lower in entry.short_name.lower():
                score += 10
                if entry.short_name.lower() == query_lower:
                    score += 20  # 精确匹配加分
            # 模块匹配
            if query_lower in entry.module.lower():
                score += 3
            # docstring 匹配
            if query_lower in entry.docstring.lower():
                score += 5
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [e for _, e in scored[:limit]]

    def get_by_name(self, name: str) -> Optional[DocEntry]:
        """按完整名获取条目。"""
        if not self._built:
            self.build()
        return self._by_name.get(name)

    def get_module_docs(self, module: str) -> List[DocEntry]:
        """获取指定模块的所有文档条目。"""
        if not self._built:
            self.build()
        return self._by_module.get(module, [])

    def get_topic_docs(self, topic: str) -> List[DocEntry]:
        """获取指定主题的所有文档条目。"""
        if not self._built:
            self.build()
        return self._by_topic.get(topic, [])

    def list_topics(self) -> Dict[str, str]:
        """列出所有主题及其描述。"""
        return dict(_TOPIC_DESCRIPTIONS)

    def list_modules(self) -> List[str]:
        """列出所有已索引的模块。"""
        if not self._built:
            self.build()
        return sorted(self._by_module.keys())

    def stats(self) -> Dict[str, int]:
        """返回索引统计信息。"""
        if not self._built:
            self.build()
        return {
            "total_entries": len(self.entries),
            "modules": len(self._by_module),
            "topics": len(self._by_topic),
            "classes": sum(1 for e in self.entries if e.kind == "class"),
            "functions": sum(1 for e in self.entries if e.kind == "function"),
            "methods": sum(1 for e in self.entries if e.kind == "method"),
        }

    # --------------------------------------------------------
    # 格式化
    # --------------------------------------------------------

    def format_entry(self, entry: DocEntry, verbose: bool = False) -> str:
        """格式化单个条目为可读文本。

        参数：
            entry : DocEntry 实例
            verbose : 是否显示完整 docstring

        返回：
            格式化的字符串
        """
        lines = []
        kind_label = {"module": "模块", "class": "类",
                      "function": "函数", "method": "方法"}.get(entry.kind, entry.kind)
        lines.append(f"[{kind_label}] {entry.name}")
        if entry.signature:
            lines.append(f"签名: {entry.short_name}{entry.signature}")
        if entry.topic:
            lines.append(f"主题: {entry.topic}")
        lines.append(f"模块: {entry.module}")
        if entry.docstring:
            if verbose:
                lines.append("")
                lines.append(entry.docstring)
            else:
                # 只显示第一段
                first_para = entry.docstring.split("\n\n")[0]
                lines.append(f"说明: {first_para}")
        else:
            lines.append("说明: (无文档)")
        return "\n".join(lines)

    def format_topic_overview(self, topic: str) -> str:
        """格式化主题概览。"""
        entries = self.get_topic_docs(topic)
        desc = _TOPIC_DESCRIPTIONS.get(topic, "")
        lines = [f"主题: {topic}"]
        if desc:
            lines.append(f"描述: {desc}")
        lines.append(f"条目数: {len(entries)}")
        lines.append("")
        lines.append("包含:")
        for e in entries[:30]:  # 限制数量
            kind_label = {"module": "模块", "class": "类",
                          "function": "函数", "method": "方法"}.get(e.kind, e.kind)
            lines.append(f"  [{kind_label}] {e.name}")
        if len(entries) > 30:
            lines.append(f"  ... 还有 {len(entries) - 30} 个条目")
        return "\n".join(lines)

    def format_search_results(self, query: str, entries: List[DocEntry]) -> str:
        """格式化搜索结果。"""
        lines = [f"搜索 '{query}' - 找到 {len(entries)} 个结果"]
        lines.append("=" * 50)
        for i, e in enumerate(entries, 1):
            kind_label = {"module": "模块", "class": "类",
                          "function": "函数", "method": "方法"}.get(e.kind, e.kind)
            lines.append(f"\n{i}. [{kind_label}] {e.name}")
            if e.signature:
                lines.append(f"   {e.short_name}{e.signature}")
            if e.docstring:
                first_line = e.docstring.split("\n")[0]
                lines.append(f"   {first_line}")
        return "\n".join(lines)


# ============================================================
# 模块级便捷 API
# ============================================================

# 全局索引（惰性构建）
_global_index: Optional[DocIndex] = None


def _get_index() -> DocIndex:
    """获取全局文档索引（惰性构建）。"""
    global _global_index
    if _global_index is None:
        _global_index = DocIndex().build()
    return _global_index


def build_doc_index() -> DocIndex:
    """构建并返回文档索引。"""
    return _get_index()


def search_docs(query: str, limit: int = 20) -> List[DocEntry]:
    """全文搜索文档。

    参数：
        query : 搜索关键词
        limit : 最多返回条目数

    返回：
        匹配的 DocEntry 列表
    """
    return _get_index().search(query, limit=limit)


def get_module_docs(module: str) -> List[DocEntry]:
    """获取指定模块的文档条目。"""
    return _get_index().get_module_docs(module)


def get_topic_docs(topic: str) -> List[DocEntry]:
    """获取指定主题的文档条目。"""
    return _get_index().get_topic_docs(topic)


def list_topics() -> Dict[str, str]:
    """列出所有文档主题。"""
    return _get_index().list_topics()


def list_modules() -> List[str]:
    """列出所有已索引的模块。"""
    return _get_index().list_modules()


def format_doc(name_or_entry: Any, verbose: bool = False) -> str:
    """格式化文档条目为可读文本。

    参数：
        name_or_entry : 条目名（str）或 DocEntry 实例
        verbose : 是否显示完整 docstring

    返回：
        格式化的字符串
    """
    index = _get_index()
    if isinstance(name_or_entry, DocEntry):
        entry = name_or_entry
    else:
        entry = index.get_by_name(name_or_entry)
        if entry is None:
            # 尝试搜索
            results = index.search(name_or_entry, limit=1)
            if results:
                entry = results[0]
            else:
                return f"未找到文档: {name_or_entry}"
    return index.format_entry(entry, verbose=verbose)


def format_search_results(query: str, limit: int = 20) -> str:
    """搜索并格式化结果。"""
    index = _get_index()
    results = index.search(query, limit=limit)
    return index.format_search_results(query, results)


def doc_stats() -> Dict[str, int]:
    """返回文档索引统计信息。"""
    return _get_index().stats()
