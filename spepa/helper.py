"""SPEPA 智能帮助助手。

设计原则（第一性原理）：
- 意图理解：从用户自然语言查询推断意图，而非要求精确关键词
- 资源整合：统一检索文档、示例、教程、预设等多类资源
- 上下文感知：根据查询类型返回最相关的帮助
- 渐进式：先给简短答案，再提供深入链接

与 docs 模块的区别：
- docs   : 精确的 API 文档检索（按名称/模块/主题）
- helper : 模糊的自然语言帮助查询（"怎么保存状态？" → 推荐相关 API + 示例 + 教程）

主要功能：
- HelpResult       : 帮助结果数据结构
- SmartHelper      : 智能帮助助手
- ask()            : 提问并获取帮助
- format_help()    : 格式化帮助结果
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from spepa.utils.logger import get_logger
from spepa.docs import search_docs, DocEntry
from spepa.recipes import list_recipes, Recipe

logger = get_logger("spepa.helper")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class HelpResult:
    """帮助查询结果。"""
    query: str                              # 用户查询
    intent: str                             # 推断的意图（如 "how_to" / "api_lookup" / "troubleshoot"）
    answer: str                             # 直接答案
    related_docs: List[DocEntry] = field(default_factory=list)
    related_recipes: List[Recipe] = field(default_factory=list)
    related_tutorials: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)


# ============================================================
# 意图识别规则
# ============================================================

# 意图 → 关键词模式
# 第一性原理：用户的查询方式有限，按模式匹配即可覆盖大多数场景
# 注意：顺序很重要，更具体的意图应排在前面（troubleshoot 优先于 how_to）
_INTENT_PATTERNS: Dict[str, List[str]] = {
    "troubleshoot": [
        "错误", "报错", "失败", "问题", "异常", "不工作", "崩溃",
        "error", "fail", "problem", "issue", "crash", "bug",
    ],
    "start": [
        "开始", "入门", "快速", "启动", "start", "begin",
        "quick", "hello", "第一个",
    ],
    "concept": [
        "是什么", "什么是", "概念", "原理", "区别", "含义",
        "what is", "concept", "meaning", "difference",
    ],
    "config": [
        "配置", "参数设置", "预设", "config", "preset",
        "调优", "优化",
    ],
    "how_to": [
        "怎么", "如何", "怎样", "如何才能", "how to", "how do i",
        "怎样做", "怎么做", "方法", "步骤",
    ],
    "api_lookup": [
        "api", "函数", "方法", "类", "参数", "签名", "接口",
        "function", "class", "method", "parameter",
    ],
}


def _detect_intent(query: str) -> str:
    """推断查询意图。

    参数：
        query : 用户查询字符串

    返回：
        意图标签（如 "how_to"）
    """
    query_lower = query.lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in query_lower:
                return intent
    return "general"


# ============================================================
# 关键词提取
# ============================================================

# 停用词（中文 + 英文）
_STOP_WORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "那",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "how", "what", "when", "where", "why", "which", "who",
    "do", "does", "did", "can", "could", "should", "would",
    "i", "you", "he", "she", "it", "we", "they",
}


def _extract_keywords(query: str) -> List[str]:
    """从查询中提取关键词。

    参数：
        query : 用户查询

    返回：
        关键词列表
    """
    # 移除意图词，保留实义词
    keywords = []
    # 简单分词：按空格 + 常见分隔符
    import re
    tokens = re.split(r"[\s,，。.!！?？;；:：、/\\]+", query)
    for token in tokens:
        token = token.strip().lower()
        if not token:
            continue
        if token in _STOP_WORDS:
            continue
        # 跳过纯数字和过短的词
        if len(token) < 2 and not token.isalpha():
            continue
        keywords.append(token)
    return keywords


# ============================================================
# 智能帮助助手
# ============================================================

class SmartHelper:
    """智能帮助助手。

    使用方式：
        helper = SmartHelper()
        result = helper.ask("怎么保存 Agent 状态？")
        print(format_help(result))
    """

    def __init__(self):
        self._conversation_history: List[str] = []

    def ask(self, query: str) -> HelpResult:
        """提问并获取帮助。

        参数：
            query : 用户查询（自然语言）

        返回：
            HelpResult 实例
        """
        self._conversation_history.append(query)
        intent = _detect_intent(query)
        keywords = _extract_keywords(query)

        # 根据意图生成答案
        answer = self._generate_answer(query, intent, keywords)

        # 检索相关资源
        related_docs = self._search_related_docs(query, keywords)
        related_recipes = self._search_related_recipes(query, keywords)
        related_tutorials = self._get_related_tutorials(intent, keywords)
        tips = self._get_tips(intent, keywords)

        return HelpResult(
            query=query,
            intent=intent,
            answer=answer,
            related_docs=related_docs,
            related_recipes=related_recipes,
            related_tutorials=related_tutorials,
            tips=tips,
        )

    def _generate_answer(self, query: str, intent: str,
                         keywords: List[str]) -> str:
        """根据意图生成直接答案。"""
        if intent == "start":
            return (
                "快速开始：\n"
                "1. 安装：pip install spepa\n"
                "2. 创建 Agent：from spepa import SPEPAAgent, get_preset_config; agent = SPEPAAgent(get_preset_config('minimal'))\n"
                "3. 运行：agent.run(n_steps=10)\n"
                "提示：查看 'quick_start' 示例获取完整代码。"
            )
        if intent == "how_to":
            # 根据 keywords 推断具体操作
            kw_str = " ".join(keywords)
            if any(k in kw_str for k in ["保存", "save", "持久化", "检查点"]):
                return (
                    "保存 Agent 状态：\n"
                    "from spepa import save_state, load_state\n"
                    "save_state(agent, force=True)  # 保存\n"
                    "load_state(agent)  # 加载\n"
                    "或使用 StateManager：from spepa import StateManager; mgr = StateManager(agent)"
                )
            if any(k in kw_str for k in ["可视化", "图表", "chart", "plot", "图"]):
                return (
                    "生成可视化图表：\n"
                    "from spepa import render_ascii_chart, plot_metric\n"
                    "render_ascii_chart(agent, 'prediction_error')  # 终端图表\n"
                    "plot_metric(agent, 'prediction_error', filepath='chart.svg')  # SVG 文件"
                )
            if any(k in kw_str for k in ["诊断", "diagnose", "健康", "问题"]):
                return (
                    "诊断 Agent 状态：\n"
                    "from spepa import diagnose, format_diagnosis\n"
                    "result = diagnose(agent)\n"
                    "print(format_diagnosis(result))"
                )
            if any(k in kw_str for k in ["配置", "config", "参数", "设置"]):
                return (
                    "配置 Agent：\n"
                    "from spepa import get_preset_config, validate_config\n"
                    "config = get_preset_config('balanced')\n"
                    "result = validate_config(config)  # 验证合理性\n"
                    "print(result['summary'])"
                )
            if any(k in kw_str for k in ["批处理", "batch", "批量", "对比"]):
                return (
                    "批处理任务：\n"
                    "from spepa import run_preset_comparison, run_param_sweep\n"
                    "results = run_preset_comparison(n_steps=30)  # 预设对比\n"
                    "results = run_param_sweep(config, 'learning_rate_micro', [0.01, 0.05])  # 参数扫描"
                )
            return (
                "要完成此操作，请查看下方相关文档和示例。\n"
                "提示：使用 search_docs('关键词') 搜索 API 文档。"
            )
        if intent == "troubleshoot":
            return (
                "排查问题：\n"
                "1. 诊断 Agent：from spepa import diagnose; print(diagnose(agent))\n"
                "2. 检查配置：from spepa import validate_config; validate_config(config)\n"
                "3. 查看友好错误：from spepa import friendly_error; print(friendly_error(error))\n"
                "4. 使用 safe_run 装饰器自动捕获错误：from spepa import safe_run"
            )
        if intent == "api_lookup":
            if keywords:
                return f"查找 API '{' '.join(keywords)}'，请查看下方相关文档。"
            return "请指定要查找的 API 名称，或使用 search_docs() 搜索。"
        if intent == "concept":
            return (
                "SPEPA 是自驱永续进化预测修正 Agent 系统。\n"
                "核心概念：\n"
                "- 预测编码：多层预测模型 + 精度加权\n"
                "- 自由能：变分自由能最小化驱动学习\n"
                "- 三大驱力：安全/收益/能效\n"
                "- 缺口检测：信息缺口 = KL 散度\n"
                "查看下方文档了解详情。"
            )
        if intent == "config":
            return (
                "配置相关：\n"
                "- 预设：get_preset_config('minimal'/'balanced'/'performance'/'conservative')\n"
                "- 验证：validate_config(config)\n"
                "- 建议：suggest_optimizations(config)\n"
                "- 场景推荐：recommend_for_scenario('embedded')"
            )
        return (
            "我理解您的问题。请查看下方相关资源，\n"
            "或使用 search_docs() 搜索文档，list_recipes() 查看示例。"
        )

    def _search_related_docs(self, query: str,
                             keywords: List[str]) -> List[DocEntry]:
        """搜索相关文档。"""
        # 用关键词搜索，合并结果
        all_results: List[DocEntry] = []
        seen_names = set()
        for kw in keywords[:3]:  # 最多用前 3 个关键词
            results = search_docs(kw, limit=5)
            for r in results:
                if r.name not in seen_names:
                    all_results.append(r)
                    seen_names.add(r.name)
        return all_results[:5]  # 最多 5 个

    def _search_related_recipes(self, query: str,
                                keywords: List[str]) -> List[Recipe]:
        """搜索相关示例。"""
        all_recipes = list_recipes()
        scored: List[tuple] = []
        kw_str = " ".join(keywords).lower()
        for recipe in all_recipes:
            score = 0
            # 名称匹配
            if any(kw in recipe.name.lower() for kw in keywords):
                score += 10
            # 标签匹配
            for tag in recipe.tags:
                if tag.lower() in kw_str or any(kw in tag.lower() for kw in keywords):
                    score += 5
            # 描述匹配
            desc_lower = recipe.description.lower()
            for kw in keywords:
                if kw in desc_lower:
                    score += 3
            if score > 0:
                scored.append((score, recipe))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:3]]

    def _get_related_tutorials(self, intent: str,
                               keywords: List[str]) -> List[str]:
        """获取相关教程。"""
        # 映射意图到教程
        tutorial_map = {
            "start": ["快速入门教程"],
            "how_to": ["基本操作教程"],
            "config": ["配置调优教程"],
            "troubleshoot": ["调试技巧教程"],
        }
        return tutorial_map.get(intent, [])

    def _get_tips(self, intent: str, keywords: List[str]) -> List[str]:
        """获取提示。"""
        tips_map = {
            "start": [
                "从 minimal 预设开始，资源占用最小",
                "先运行少量步数（如 10 步）验证环境",
            ],
            "how_to": [
                "查看相关示例：list_recipes()",
                "搜索 API：search_docs('关键词')",
            ],
            "troubleshoot": [
                "使用 diagnose(agent) 诊断问题",
                "使用 friendly_error(error) 获取友好错误信息",
                "检查配置合法性：validate_config(config)",
            ],
            "config": [
                "使用 validate_config() 检查配置合理性",
                "使用 suggest_optimizations() 获取优化建议",
                "使用 recommend_for_scenario() 按场景推荐",
            ],
            "api_lookup": [
                "使用 format_doc(name) 查看完整文档",
                "使用 search_docs(query) 全文搜索",
            ],
            "concept": [
                "查看 'spepa' 模块文档了解整体架构",
                "运行教程：run_tutorial('quickstart')",
            ],
        }
        return tips_map.get(intent, ["使用 search_docs() 搜索文档"])

    def history(self) -> List[str]:
        """返回查询历史。"""
        return list(self._conversation_history)

    def clear_history(self) -> None:
        """清空查询历史。"""
        self._conversation_history = []


# ============================================================
# 模块级便捷 API
# ============================================================

_global_helper: Optional[SmartHelper] = None


def _get_helper() -> SmartHelper:
    """获取全局帮助助手。"""
    global _global_helper
    if _global_helper is None:
        _global_helper = SmartHelper()
    return _global_helper


def ask(query: str) -> HelpResult:
    """提问并获取帮助。

    参数：
        query : 用户查询（自然语言）

    返回：
        HelpResult 实例
    """
    return _get_helper().ask(query)


def format_help(result: HelpResult) -> str:
    """格式化帮助结果为可读文本。

    参数：
        result : HelpResult 实例

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SPEPA 智能帮助")
    lines.append("=" * 60)
    lines.append(f"\n[问题] {result.query}")
    lines.append(f"[意图] {result.intent}")

    lines.append(f"\n[回答]")
    lines.append(result.answer)

    if result.related_docs:
        lines.append(f"\n[相关文档] ({len(result.related_docs)} 个)")
        for doc in result.related_docs:
            kind_label = {"module": "模块", "class": "类",
                          "function": "函数", "method": "方法"}.get(doc.kind, doc.kind)
            lines.append(f"  [{kind_label}] {doc.name}")
            if doc.docstring:
                first_line = doc.docstring.split("\n")[0]
                lines.append(f"    {first_line}")

    if result.related_recipes:
        lines.append(f"\n[相关示例] ({len(result.related_recipes)} 个)")
        for recipe in result.related_recipes:
            lines.append(f"  {recipe.name}: {recipe.title}")
            lines.append(f"    {recipe.description}")

    if result.related_tutorials:
        lines.append(f"\n[相关教程]")
        for t in result.related_tutorials:
            lines.append(f"  - {t}")

    if result.tips:
        lines.append(f"\n[提示]")
        for tip in result.tips:
            lines.append(f"  - {tip}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
