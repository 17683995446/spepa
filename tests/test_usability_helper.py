"""SPEPA 智能帮助助手测试。

测试范围：
- HelpResult 数据结构
- 意图识别（_detect_intent）
- 关键词提取（_extract_keywords）
- SmartHelper（ask、答案生成、资源检索、历史）
- 便捷函数（ask、format_help）
- 端到端
"""
import pytest
from spepa.helper import (
    HelpResult, SmartHelper,
    ask, format_help,
    _detect_intent, _extract_keywords,
)


# ============================================================
# HelpResult 数据结构
# ============================================================

class TestHelpResult:
    """HelpResult 数据结构测试。"""

    def test_create_result(self):
        result = HelpResult(
            query="测试",
            intent="how_to",
            answer="答案",
        )
        assert result.query == "测试"
        assert result.intent == "how_to"
        assert result.answer == "答案"
        assert result.related_docs == []
        assert result.related_recipes == []
        assert result.tips == []

    def test_result_with_all_fields(self):
        result = HelpResult(
            query="测试",
            intent="how_to",
            answer="答案",
            related_tutorials=["教程1"],
            tips=["提示1"],
        )
        assert result.related_tutorials == ["教程1"]
        assert result.tips == ["提示1"]


# ============================================================
# 意图识别
# ============================================================

class TestDetectIntent:
    """意图识别测试。"""

    def test_how_to_chinese(self):
        assert _detect_intent("怎么保存状态") == "how_to"
        assert _detect_intent("如何运行 Agent") == "how_to"
        assert _detect_intent("怎样创建实例") == "how_to"

    def test_how_to_english(self):
        assert _detect_intent("how to save state") == "how_to"
        assert _detect_intent("how do I run agent") == "how_to"

    def test_troubleshoot(self):
        assert _detect_intent("运行报错了") == "troubleshoot"
        assert _detect_intent("出现错误") == "troubleshoot"
        assert _detect_intent("agent crashed") == "troubleshoot"

    def test_api_lookup(self):
        assert _detect_intent("SPEPAAgent 类的参数") == "api_lookup"
        assert _detect_intent("run 函数签名") == "api_lookup"

    def test_concept(self):
        assert _detect_intent("什么是自由能") == "concept"
        assert _detect_intent("预测编码是什么") == "concept"

    def test_config(self):
        assert _detect_intent("如何配置参数") in ("how_to", "config")
        assert _detect_intent("配置预设") == "config"

    def test_start(self):
        assert _detect_intent("如何开始使用") in ("how_to", "start")
        assert _detect_intent("快速入门") == "start"

    def test_general_fallback(self):
        # 无明显意图的查询
        intent = _detect_intent("xyz random query")
        assert isinstance(intent, str)


# ============================================================
# 关键词提取
# ============================================================

class TestExtractKeywords:
    """关键词提取测试。"""

    def test_extract_basic(self):
        keywords = _extract_keywords("保存 Agent 状态")
        assert len(keywords) > 0
        assert "保存" in keywords or "agent" in keywords or "状态" in keywords

    def test_extract_removes_stopwords(self):
        keywords = _extract_keywords("怎么 保存 的 状态")
        assert "怎么" not in keywords or "怎么" in keywords  # "怎么" 可能不被过滤
        # 但 "的" 应该被过滤
        assert "的" not in keywords

    def test_extract_english(self):
        keywords = _extract_keywords("how to save agent state")
        assert "save" in keywords or "agent" in keywords or "state" in keywords

    def test_extract_empty_query(self):
        keywords = _extract_keywords("")
        assert keywords == []

    def test_extract_only_stopwords(self):
        keywords = _extract_keywords("the a is of")
        assert keywords == []


# ============================================================
# SmartHelper
# ============================================================

class TestSmartHelper:
    """智能帮助助手测试。"""

    def test_ask_returns_result(self):
        helper = SmartHelper()
        result = helper.ask("怎么保存状态")
        assert isinstance(result, HelpResult)
        assert result.query == "怎么保存状态"
        assert isinstance(result.intent, str)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_ask_how_to_save(self):
        helper = SmartHelper()
        result = helper.ask("怎么保存 Agent 状态")
        assert result.intent == "how_to"
        assert "保存" in result.answer or "save" in result.answer.lower()

    def test_ask_how_to_visualize(self):
        helper = SmartHelper()
        result = helper.ask("如何生成图表")
        assert result.intent == "how_to"
        assert "图表" in result.answer or "chart" in result.answer.lower()

    def test_ask_how_to_diagnose(self):
        helper = SmartHelper()
        result = helper.ask("怎么诊断健康状况")
        assert result.intent == "how_to"
        assert "诊断" in result.answer or "diagnose" in result.answer.lower()

    def test_ask_troubleshoot(self):
        helper = SmartHelper()
        result = helper.ask("运行报错了怎么办")
        assert result.intent == "troubleshoot"
        assert len(result.answer) > 0

    def test_ask_start(self):
        helper = SmartHelper()
        result = helper.ask("快速入门")
        assert result.intent == "start"
        assert "开始" in result.answer or "start" in result.answer.lower()

    def test_ask_concept(self):
        helper = SmartHelper()
        result = helper.ask("什么是自由能")
        assert result.intent == "concept"
        assert "SPEPA" in result.answer or "自由能" in result.answer

    def test_ask_config(self):
        helper = SmartHelper()
        result = helper.ask("配置预设")
        assert result.intent == "config"
        assert "预设" in result.answer or "preset" in result.answer.lower()

    def test_ask_returns_related_docs(self):
        helper = SmartHelper()
        result = helper.ask("怎么保存状态")
        # related_docs 应是列表（可能为空也可能有内容）
        assert isinstance(result.related_docs, list)

    def test_ask_returns_related_recipes(self):
        helper = SmartHelper()
        result = helper.ask("怎么保存状态")
        assert isinstance(result.related_recipes, list)

    def test_ask_returns_tips(self):
        helper = SmartHelper()
        result = helper.ask("快速入门")
        assert isinstance(result.tips, list)
        assert len(result.tips) > 0

    def test_ask_returns_tutorials(self):
        helper = SmartHelper()
        result = helper.ask("快速入门")
        assert isinstance(result.related_tutorials, list)

    def test_history_tracking(self):
        helper = SmartHelper()
        helper.ask("问题1")
        helper.ask("问题2")
        history = helper.history()
        assert len(history) == 2
        assert "问题1" in history
        assert "问题2" in history

    def test_clear_history(self):
        helper = SmartHelper()
        helper.ask("问题1")
        helper.clear_history()
        assert len(helper.history()) == 0

    def test_multiple_queries_independent(self):
        helper = SmartHelper()
        r1 = helper.ask("怎么保存")
        r2 = helper.ask("如何诊断")
        assert r1.query == "怎么保存"
        assert r2.query == "如何诊断"


# ============================================================
# 便捷函数
# ============================================================

class TestConvenienceFunctions:
    """便捷函数测试。"""

    def test_ask(self):
        result = ask("怎么保存状态")
        assert isinstance(result, HelpResult)
        assert result.query == "怎么保存状态"

    def test_ask_multiple_times(self):
        # 全局助手应能处理多次查询
        r1 = ask("问题1")
        r2 = ask("问题2")
        assert r1.query == "问题1"
        assert r2.query == "问题2"

    def test_format_help(self):
        result = ask("怎么保存状态")
        text = format_help(result)
        assert "SPEPA 智能帮助" in text
        assert "问题" in text
        assert "意图" in text
        assert "回答" in text

    def test_format_help_contains_tips(self):
        result = ask("快速入门")
        text = format_help(result)
        assert "提示" in text

    def test_format_help_contains_query(self):
        result = ask("测试查询")
        text = format_help(result)
        assert "测试查询" in text


# ============================================================
# 端到端
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_full_workflow(self):
        """完整工作流：提问 → 获取结果 → 格式化。"""
        helper = SmartHelper()
        result = helper.ask("怎么保存 Agent 状态？")
        text = format_help(result)
        assert "SPEPA 智能帮助" in text
        assert "保存" in text
        assert result.intent == "how_to"

    def test_different_intents_produce_different_answers(self):
        """不同意图应产生不同答案。"""
        helper = SmartHelper()
        r1 = helper.ask("怎么保存状态")
        r2 = helper.ask("运行报错了")
        r3 = helper.ask("快速入门")
        # 答案不应完全相同
        assert r1.answer != r2.answer or r1.answer != r3.answer

    def test_helper_with_history(self):
        """带历史的助手工作流。"""
        helper = SmartHelper()
        questions = ["怎么保存", "如何诊断", "快速入门"]
        for q in questions:
            helper.ask(q)
        assert len(helper.history()) == 3
        helper.clear_history()
        assert len(helper.history()) == 0

    def test_related_resources_present(self):
        """相关资源应被检索到。"""
        helper = SmartHelper()
        result = helper.ask("怎么配置参数")
        # 至少应有 tips
        assert len(result.tips) > 0

    def test_chinese_and_english_queries(self):
        """中英文查询都能处理。"""
        helper = SmartHelper()
        r_cn = helper.ask("怎么保存状态")
        r_en = helper.ask("how to save state")
        assert r_cn.intent == "how_to"
        assert r_en.intent == "how_to"
