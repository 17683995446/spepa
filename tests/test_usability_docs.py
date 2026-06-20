"""SPEPA 统一文档系统测试。

测试范围：
- DocEntry 数据结构
- DocIndex 构建、查询、格式化
- 便捷函数（build_doc_index/search_docs/get_module_docs 等）
- 端到端
"""
import pytest
from spepa.docs import (
    DocEntry, DocIndex,
    build_doc_index, search_docs, get_module_docs, get_topic_docs,
    list_topics, list_modules, format_doc, format_search_results, doc_stats,
)


# ============================================================
# DocEntry 数据结构
# ============================================================

class TestDocEntry:
    """DocEntry 数据结构测试。"""

    def test_create_entry(self):
        entry = DocEntry(
            name="spepa.test",
            short_name="test",
            kind="function",
            module="spepa.test",
            docstring="测试函数",
        )
        assert entry.name == "spepa.test"
        assert entry.short_name == "test"
        assert entry.kind == "function"
        assert entry.docstring == "测试函数"
        assert entry.signature == ""
        assert entry.topic == ""
        assert entry.examples == []

    def test_entry_with_all_fields(self):
        entry = DocEntry(
            name="spepa.foo",
            short_name="foo",
            kind="class",
            module="spepa.foo",
            docstring="Foo 类",
            signature="(x, y)",
            topic="core",
            examples=["foo(1, 2)"],
        )
        assert entry.signature == "(x, y)"
        assert entry.topic == "core"
        assert entry.examples == ["foo(1, 2)"]


# ============================================================
# DocIndex 构建
# ============================================================

class TestDocIndexBuild:
    """DocIndex 构建测试。"""

    def test_build_returns_self(self):
        index = DocIndex()
        result = index.build()
        assert result is index

    def test_build_creates_entries(self):
        index = DocIndex().build()
        assert len(index.entries) > 0

    def test_build_includes_known_modules(self):
        index = DocIndex().build()
        modules = index.list_modules()
        assert "spepa.agent" in modules
        assert "spepa.diagnostics" in modules

    def test_build_includes_classes(self):
        index = DocIndex().build()
        classes = [e for e in index.entries if e.kind == "class"]
        assert len(classes) > 0
        class_names = [e.name for e in classes]
        assert "spepa.SPEPAAgent" in class_names

    def test_build_includes_functions(self):
        index = DocIndex().build()
        functions = [e for e in index.entries if e.kind == "function"]
        assert len(functions) > 0

    def test_build_includes_methods(self):
        index = DocIndex().build()
        methods = [e for e in index.entries if e.kind == "method"]
        assert len(methods) > 0

    def test_build_assigns_topics(self):
        index = DocIndex().build()
        entries_with_topic = [e for e in index.entries if e.topic]
        assert len(entries_with_topic) > 0

    def test_build_idempotent(self):
        """多次构建不累积。"""
        index = DocIndex()
        index.build()
        count1 = len(index.entries)
        index.build()
        count2 = len(index.entries)
        assert count1 == count2


# ============================================================
# DocIndex 查询
# ============================================================

class TestDocIndexSearch:
    """DocIndex 搜索测试。"""

    def test_search_returns_list(self):
        index = DocIndex().build()
        results = index.search("run")
        assert isinstance(results, list)

    def test_search_finds_matches(self):
        index = DocIndex().build()
        results = index.search("run")
        assert len(results) > 0

    def test_search_no_match(self):
        index = DocIndex().build()
        results = index.search("zzz_nonexistent_zzz")
        assert len(results) == 0

    def test_search_respects_limit(self):
        index = DocIndex().build()
        results = index.search("a", limit=5)
        assert len(results) <= 5

    def test_search_exact_name_high_priority(self):
        index = DocIndex().build()
        results = index.search("SPEPAAgent")
        assert len(results) > 0
        # 精确匹配应排在前面
        assert any(e.short_name == "SPEPAAgent" for e in results[:3])

    def test_search_case_insensitive(self):
        index = DocIndex().build()
        results_lower = index.search("run")
        results_upper = index.search("RUN")
        # 两者应找到相同数量的结果（顺序可能不同）
        assert len(results_lower) == len(results_upper)

    def test_get_by_name_existing(self):
        index = DocIndex().build()
        entry = index.get_by_name("spepa.SPEPAAgent")
        assert entry is not None
        assert entry.kind == "class"

    def test_get_by_name_nonexistent(self):
        index = DocIndex().build()
        entry = index.get_by_name("spepa.nonexistent")
        assert entry is None

    def test_get_module_docs(self):
        index = DocIndex().build()
        entries = index.get_module_docs("spepa.agent")
        assert len(entries) > 0

    def test_get_module_docs_nonexistent(self):
        index = DocIndex().build()
        entries = index.get_module_docs("spepa.nonexistent")
        assert entries == []

    def test_get_topic_docs(self):
        index = DocIndex().build()
        entries = index.get_topic_docs("core")
        assert len(entries) > 0

    def test_get_topic_docs_nonexistent(self):
        index = DocIndex().build()
        entries = index.get_topic_docs("nonexistent_topic")
        assert entries == []

    def test_list_topics(self):
        index = DocIndex().build()
        topics = index.list_topics()
        assert isinstance(topics, dict)
        assert "core" in topics
        assert "config" in topics

    def test_list_modules(self):
        index = DocIndex().build()
        modules = index.list_modules()
        assert isinstance(modules, list)
        assert len(modules) > 0

    def test_stats(self):
        index = DocIndex().build()
        stats = index.stats()
        assert "total_entries" in stats
        assert "modules" in stats
        assert "topics" in stats
        assert "classes" in stats
        assert "functions" in stats
        assert "methods" in stats
        assert stats["total_entries"] > 0


# ============================================================
# DocIndex 格式化
# ============================================================

class TestDocIndexFormat:
    """DocIndex 格式化测试。"""

    def test_format_entry(self):
        index = DocIndex().build()
        entry = index.get_by_name("spepa.SPEPAAgent")
        text = index.format_entry(entry)
        assert "spepa.SPEPAAgent" in text
        assert "类" in text

    def test_format_entry_verbose(self):
        index = DocIndex().build()
        entry = index.get_by_name("spepa.SPEPAAgent")
        text_short = index.format_entry(entry, verbose=False)
        text_verbose = index.format_entry(entry, verbose=True)
        assert len(text_verbose) >= len(text_short)

    def test_format_entry_function(self):
        index = DocIndex().build()
        # 找一个函数
        funcs = [e for e in index.entries if e.kind == "function"]
        if funcs:
            text = index.format_entry(funcs[0])
            assert "函数" in text

    def test_format_topic_overview(self):
        index = DocIndex().build()
        text = index.format_topic_overview("core")
        assert "core" in text
        assert "条目数" in text

    def test_format_search_results(self):
        index = DocIndex().build()
        results = index.search("run", limit=3)
        text = index.format_search_results("run", results)
        assert "搜索" in text
        assert "run" in text

    def test_format_search_results_empty(self):
        index = DocIndex().build()
        text = index.format_search_results("zzz", [])
        assert "0 个结果" in text


# ============================================================
# 便捷函数
# ============================================================

class TestConvenienceFunctions:
    """便捷函数测试。"""

    def test_build_doc_index(self):
        index = build_doc_index()
        assert isinstance(index, DocIndex)
        assert len(index.entries) > 0

    def test_search_docs(self):
        results = search_docs("agent")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_docs_with_limit(self):
        results = search_docs("a", limit=3)
        assert len(results) <= 3

    def test_get_module_docs(self):
        entries = get_module_docs("spepa.agent")
        assert len(entries) > 0

    def test_get_topic_docs(self):
        entries = get_topic_docs("core")
        assert len(entries) > 0

    def test_list_topics(self):
        topics = list_topics()
        assert isinstance(topics, dict)
        assert len(topics) > 0

    def test_list_modules(self):
        modules = list_modules()
        assert isinstance(modules, list)
        assert len(modules) > 0

    def test_format_doc_by_name(self):
        text = format_doc("spepa.SPEPAAgent")
        assert "spepa.SPEPAAgent" in text

    def test_format_doc_by_entry(self):
        entry = DocEntry(name="test", short_name="test", kind="function",
                         module="test", docstring="测试")
        text = format_doc(entry)
        assert "test" in text

    def test_format_doc_nonexistent(self):
        text = format_doc("spepa.nonexistent_xyz")
        assert "未找到" in text

    def test_format_doc_verbose(self):
        text = format_doc("spepa.SPEPAAgent", verbose=True)
        assert isinstance(text, str)

    def test_format_search_results(self):
        text = format_search_results("run")
        assert "搜索" in text

    def test_doc_stats(self):
        stats = doc_stats()
        assert stats["total_entries"] > 0


# ============================================================
# 端到端
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_full_workflow(self):
        """完整工作流：构建 → 搜索 → 格式化。"""
        index = build_doc_index()
        results = index.search("config")
        assert len(results) > 0
        text = index.format_entry(results[0])
        assert isinstance(text, str)

    def test_topic_browse_workflow(self):
        """主题浏览工作流。"""
        topics = list_topics()
        for topic_name in topics:
            entries = get_topic_docs(topic_name)
            # 每个主题至少应有一些条目（或为空也合法）
            assert isinstance(entries, list)

    def test_module_browse_workflow(self):
        """模块浏览工作流。"""
        modules = list_modules()
        for mod in modules[:3]:  # 测试前 3 个
            entries = get_module_docs(mod)
            assert len(entries) > 0

    def test_search_then_format(self):
        """搜索 → 格式化。"""
        results = search_docs("run", limit=5)
        for entry in results:
            text = format_doc(entry)
            assert isinstance(text, str)

    def test_stats_consistency(self):
        """统计信息一致性。"""
        stats = doc_stats()
        index = build_doc_index()
        assert stats["total_entries"] == len(index.entries)
        assert stats["classes"] == sum(1 for e in index.entries if e.kind == "class")
        assert stats["functions"] == sum(1 for e in index.entries if e.kind == "function")
        assert stats["methods"] == sum(1 for e in index.entries if e.kind == "method")
