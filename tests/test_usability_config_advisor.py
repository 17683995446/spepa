"""SPEPA 配置验证与智能建议系统测试。

测试范围：
- ConfigAdvice 数据结构
- list_scenarios / recommend_for_scenario（场景化推荐）
- validate_config（合法性 + 合理性 + 评分）
- suggest_optimizations（静态 + 动态 with metrics_history）
- diff_configs（配置对比）
- ConfigWizard（交互式向导）
- format_advice / format_diff
- 端到端
"""
import pytest
from spepa.config_advisor import (
    ConfigAdvice, ConfigDiff, ConfigWizard,
    validate_config, recommend_for_scenario, suggest_optimizations,
    diff_configs, list_scenarios,
    format_advice, format_diff,
)
from spepa.utils.config import SPEPAConfig, get_preset_config


# ============================================================
# ConfigAdvice 数据结构
# ============================================================

class TestConfigAdvice:
    """ConfigAdvice 数据结构测试。"""

    def test_create_advice(self):
        advice = ConfigAdvice(
            severity="warning",
            category="performance",
            field="learning_rate_micro",
            current=0.1,
            recommended=0.05,
            reason="学习率偏高",
            impact="可能导致震荡",
        )
        assert advice.severity == "warning"
        assert advice.category == "performance"
        assert advice.field == "learning_rate_micro"
        assert advice.current == 0.1
        assert advice.recommended == 0.05
        assert advice.reason == "学习率偏高"
        assert advice.impact == "可能导致震荡"

    def test_advice_default_impact(self):
        advice = ConfigAdvice(
            severity="info",
            category="memory",
            field="fast_memory_capacity",
            current=100,
            recommended=500,
            reason="容量过小",
        )
        assert advice.impact == ""


# ============================================================
# list_scenarios
# ============================================================

class TestListScenarios:
    """list_scenarios 测试。"""

    def test_returns_dict(self):
        scenarios = list_scenarios()
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0

    def test_contains_common_scenarios(self):
        scenarios = list_scenarios()
        for name in ["embedded", "testing", "research", "production"]:
            assert name in scenarios
            assert isinstance(scenarios[name], str)
            assert len(scenarios[name]) > 0


# ============================================================
# recommend_for_scenario
# ============================================================

class TestRecommendForScenario:
    """场景化推荐测试。"""

    def test_embedded_scenario(self):
        config = recommend_for_scenario("embedded")
        assert isinstance(config, SPEPAConfig)
        # embedded 应该资源受限
        assert config.resources.cpu_max_percent <= 10
        assert config.resources.memory_max_percent <= 10

    def test_testing_scenario(self):
        config = recommend_for_scenario("testing")
        assert isinstance(config, SPEPAConfig)
        assert config.max_steps <= 100
        assert config.seed == 42

    def test_research_scenario(self):
        config = recommend_for_scenario("research")
        assert isinstance(config, SPEPAConfig)
        # research 应该高维度
        assert config.state_dim >= 32

    def test_production_scenario(self):
        config = recommend_for_scenario("production")
        assert isinstance(config, SPEPAConfig)
        # production 应该保守
        assert config.safety_hard_boundary <= 0.95

    def test_realtime_scenario(self):
        config = recommend_for_scenario("realtime")
        assert isinstance(config, SPEPAConfig)
        # realtime 应该小步数
        assert config.max_steps <= 50

    def test_educational_scenario(self):
        config = recommend_for_scenario("educational")
        assert isinstance(config, SPEPAConfig)
        assert config.max_steps <= 500

    def test_case_insensitive(self):
        config_lower = recommend_for_scenario("embedded")
        config_upper = recommend_for_scenario("EMBEDDED")
        assert config_lower.to_dict() == config_upper.to_dict()

    def test_strips_whitespace(self):
        config1 = recommend_for_scenario("embedded")
        config2 = recommend_for_scenario("  embedded  ")
        assert config1.to_dict() == config2.to_dict()

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="未知场景"):
            recommend_for_scenario("nonexistent")

    def test_with_overrides(self):
        config = recommend_for_scenario("balanced", overrides={"state_dim": 32})
        assert config.state_dim == 32

    def test_overrides_take_priority(self):
        # 场景覆盖 + 用户覆盖，用户优先
        config = recommend_for_scenario("testing", overrides={"max_steps": 999})
        assert config.max_steps == 999

    def test_nested_overrides(self):
        config = recommend_for_scenario("balanced", overrides={
            "resources": {"cpu_max_percent": 50.0}
        })
        assert config.resources.cpu_max_percent == 50.0

    def test_returned_config_is_valid(self):
        # 所有场景返回的配置都应通过 validate
        for name in list_scenarios():
            config = recommend_for_scenario(name)
            config.validate()  # 不抛异常即可


# ============================================================
# validate_config
# ============================================================

class TestValidateConfig:
    """配置合理性验证测试。"""

    def test_valid_config(self):
        config = get_preset_config("balanced")
        result = validate_config(config)
        assert result["valid"] is True
        assert isinstance(result["advice"], list)
        assert isinstance(result["score"], float)
        assert 0 <= result["score"] <= 100
        assert isinstance(result["summary"], str)

    def test_invalid_config_state_dim(self):
        config = get_preset_config("balanced").merge({"state_dim": -1})
        result = validate_config(config)
        assert result["valid"] is False
        assert len(result["base_issues"]) > 0

    def test_invalid_config_action_dim(self):
        config = get_preset_config("balanced").merge({"action_dim": 0})
        result = validate_config(config)
        assert result["valid"] is False

    def test_invalid_config_memory(self):
        config = get_preset_config("balanced").merge({"fast_memory_capacity": -1})
        result = validate_config(config)
        assert result["valid"] is False

    def test_high_lr_with_many_steps_warning(self):
        # 学习率大 + 步数多 → 警告
        config = get_preset_config("balanced").merge({
            "learning_rate_micro": 0.1,
            "max_steps": 5000,
        })
        result = validate_config(config)
        advice_fields = [a.field for a in result["advice"]]
        assert any("learning_rate_micro" in f for f in advice_fields)

    def test_low_lr_with_few_steps_warning(self):
        # 学习率小 + 步数少 → 警告
        config = get_preset_config("balanced").merge({
            "learning_rate_micro": 0.0001,
            "max_steps": 100,
        })
        result = validate_config(config)
        advice_fields = [a.field for a in result["advice"]]
        assert any("learning_rate_micro" in f for f in advice_fields)

    def test_large_state_dim_small_memory_warning(self):
        config = get_preset_config("balanced").merge({
            "state_dim": 32,
            "fast_memory_capacity": 100,
        })
        result = validate_config(config)
        advice_fields = [a.field for a in result["advice"]]
        assert any("fast_memory_capacity" in f for f in advice_fields)

    def test_exploration_decay_geq_one_warning(self):
        config = get_preset_config("balanced").merge({
            "exploration": {"decay": 1.0}
        })
        result = validate_config(config)
        advice_fields = [a.field for a in result["advice"]]
        assert any("decay" in f for f in advice_fields)

    def test_non_monotonic_thresholds_critical(self):
        # gap_level_thresholds 非递增 → critical
        config = get_preset_config("balanced").merge({
            "gap_level_thresholds": [0.2, 0.5, 0.3, 2.0]
        })
        result = validate_config(config)
        critical = [a for a in result["advice"] if a.severity == "critical"]
        assert any("gap_level_thresholds" in a.field for a in critical)

    def test_micro_geq_struct_critical(self):
        config = get_preset_config("balanced").merge({
            "micro_correction_threshold": 1.5,
            "structural_reconstruction_threshold": 1.0,
        })
        result = validate_config(config)
        critical = [a for a in result["advice"] if a.severity == "critical"]
        assert any("micro_correction" in a.field for a in critical)

    def test_high_complexity_low_cpu_warning(self):
        # 复杂度高 + CPU 上限低 → 警告
        config = get_preset_config("balanced").merge({
            "state_dim": 32,
            "action_dim": 16,
            "prediction_layers": 5,
            "resources": {"cpu_max_percent": 10.0},
        })
        result = validate_config(config)
        advice_fields = [a.field for a in result["advice"]]
        assert any("cpu_max_percent" in f for f in advice_fields)

    def test_score_decreases_with_issues(self):
        good_config = get_preset_config("balanced")
        bad_config = get_preset_config("balanced").merge({"state_dim": -1})
        good_result = validate_config(good_config)
        bad_result = validate_config(bad_config)
        assert good_result["score"] > bad_result["score"]

    def test_score_in_range(self):
        config = get_preset_config("balanced")
        result = validate_config(config)
        assert 0.0 <= result["score"] <= 100.0

    def test_summary_for_valid_config(self):
        config = get_preset_config("balanced")
        result = validate_config(config)
        # 合法配置的总结应包含"合法"或"合理"
        assert "合法" in result["summary"] or "合理" in result["summary"]

    def test_summary_for_invalid_config(self):
        config = get_preset_config("balanced").merge({"state_dim": -1})
        result = validate_config(config)
        assert "非法" in result["summary"] or "错误" in result["summary"]

    def test_summary_with_critical(self):
        config = get_preset_config("balanced").merge({
            "gap_level_thresholds": [0.5, 0.2, 1.0, 2.0]
        })
        result = validate_config(config)
        assert "严重" in result["summary"] or "critical" in result["summary"].lower()


# ============================================================
# suggest_optimizations
# ============================================================

class TestSuggestOptimizations:
    """优化建议测试。"""

    def test_static_low_lr_suggestion(self):
        config = get_preset_config("balanced").merge({"learning_rate_micro": 0.001})
        advice = suggest_optimizations(config)
        fields = [a.field for a in advice]
        assert any("learning_rate_micro" in f for f in fields)

    def test_static_high_lr_suggestion(self):
        config = get_preset_config("balanced").merge({"learning_rate_micro": 0.2})
        advice = suggest_optimizations(config)
        fields = [a.field for a in advice]
        assert any("learning_rate_micro" in f for f in fields)

    def test_no_metrics_returns_static_only(self):
        config = get_preset_config("balanced")
        advice = suggest_optimizations(config, metrics_history=None)
        # 无历史数据，只有静态建议（可能为空或少量）
        assert isinstance(advice, list)

    def test_metrics_error_not_converging(self):
        # 误差不下降 → 建议增大学习率
        config = get_preset_config("balanced")
        # 前5步均值 0.5，后5步均值 0.48（几乎未下降）
        errors = [0.5, 0.5, 0.5, 0.5, 0.5, 0.48, 0.48, 0.48, 0.48, 0.48]
        advice = suggest_optimizations(config, {"prediction_error": errors})
        fields = [a.field for a in advice]
        assert any("learning_rate_micro" in f for f in fields)

    def test_metrics_error_converged_well(self):
        # 误差大幅下降 → 建议减小学习率精调
        config = get_preset_config("balanced")
        errors = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1]
        advice = suggest_optimizations(config, {"prediction_error": errors})
        fields = [a.field for a in advice]
        assert any("learning_rate_micro" in f for f in fields)

    def test_metrics_low_confidence(self):
        config = get_preset_config("balanced")
        confs = [0.1, 0.1, 0.1, 0.1, 0.1]
        advice = suggest_optimizations(config, {"confidence": confs})
        fields = [a.field for a in advice]
        assert any("max_steps" in f for f in fields)

    def test_metrics_oscillating_error(self):
        # 误差震荡剧烈
        config = get_preset_config("balanced")
        errors = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
        advice = suggest_optimizations(config, {"prediction_error": errors})
        fields = [a.field for a in advice]
        assert any("learning_rate_micro" in f for f in fields)

    def test_metrics_short_history_ignored(self):
        # 不足 10 步的历史不触发动态建议
        config = get_preset_config("balanced")
        errors = [0.5, 0.5, 0.5]  # 只有 3 步
        advice = suggest_optimizations(config, {"prediction_error": errors})
        # 不应包含基于历史的建议（但仍可能有静态建议）
        for a in advice:
            assert "未下降" not in a.reason
            assert "大幅下降" not in a.reason

    def test_returns_list_of_config_advice(self):
        config = get_preset_config("balanced")
        advice = suggest_optimizations(config)
        for a in advice:
            assert isinstance(a, ConfigAdvice)


# ============================================================
# diff_configs
# ============================================================

class TestDiffConfigs:
    """配置对比测试。"""

    def test_identical_configs(self):
        config_a = get_preset_config("balanced")
        config_b = get_preset_config("balanced")
        result = diff_configs(config_a, config_b)
        assert result["identical"] is True
        assert len(result["diffs"]) == 0
        assert "相同" in result["summary"]

    def test_different_configs(self):
        config_a = get_preset_config("minimal")
        config_b = get_preset_config("performance")
        result = diff_configs(config_a, config_b)
        assert result["identical"] is False
        assert len(result["diffs"]) > 0
        assert "差异" in result["summary"]

    def test_diff_contains_state_dim(self):
        config_a = get_preset_config("minimal")  # state_dim=8
        config_b = get_preset_config("performance")  # state_dim=32
        result = diff_configs(config_a, config_b)
        fields = [d.field for d in result["diffs"]]
        assert any("state_dim" in f for f in fields)

    def test_diff_contains_nested_field(self):
        config_a = get_preset_config("minimal")
        config_b = get_preset_config("performance")
        result = diff_configs(config_a, config_b)
        fields = [d.field for d in result["diffs"]]
        # 嵌套字段应被扁平化（如 resources.cpu_max_percent）
        assert any("resources" in f for f in fields)

    def test_diff_values_correct(self):
        config_a = get_preset_config("minimal")
        config_b = get_preset_config("performance")
        result = diff_configs(config_a, config_b)
        for d in result["diffs"]:
            if "state_dim" in d.field:
                assert d.value_a == 8
                assert d.value_b == 32

    def test_diff_has_impact_description(self):
        config_a = get_preset_config("minimal")
        config_b = get_preset_config("performance")
        result = diff_configs(config_a, config_b)
        # 至少有一些差异项带有影响描述
        impacts = [d.impact for d in result["diffs"] if d.impact]
        assert len(impacts) > 0

    def test_diff_returns_config_diff_objects(self):
        config_a = get_preset_config("minimal")
        config_b = get_preset_config("performance")
        result = diff_configs(config_a, config_b)
        for d in result["diffs"]:
            assert isinstance(d, ConfigDiff)

    def test_single_field_diff(self):
        config_a = get_preset_config("balanced")
        config_b = config_a.merge({"seed": 999})
        result = diff_configs(config_a, config_b)
        assert not result["identical"]
        seed_diffs = [d for d in result["diffs"] if "seed" in d.field]
        assert len(seed_diffs) == 1
        assert seed_diffs[0].value_a == 42
        assert seed_diffs[0].value_b == 999


# ============================================================
# ConfigWizard
# ============================================================

class TestConfigWizard:
    """配置向导测试。"""

    def test_create_wizard(self):
        wizard = ConfigWizard()
        assert len(wizard.questions) > 0

    def test_questions_have_required_fields(self):
        wizard = ConfigWizard()
        for q in wizard.questions:
            assert "key" in q
            assert "question" in q
            assert "options" in q
            assert "default" in q
            assert len(q["options"]) > 0

    def test_ask_questions_with_injected_input(self):
        wizard = ConfigWizard()
        # 模拟用户全部使用默认值（输入空字符串）
        answers = wizard.ask_questions(input_func=lambda _: "")
        for q in wizard.questions:
            assert answers[q["key"]] == q["default"]

    def test_ask_questions_with_custom_answers(self):
        wizard = ConfigWizard()
        inputs = iter(["embedded", "8", "100", "42"])
        answers = wizard.ask_questions(input_func=lambda _: next(inputs))
        assert answers["scenario"] == "embedded"
        assert answers["state_dim"] == 8
        assert answers["max_steps"] == 100
        assert answers["seed"] == 42

    def test_ask_questions_invalid_input_uses_default(self):
        wizard = ConfigWizard()
        # 输入无效值，应回退到默认
        answers = wizard.ask_questions(input_func=lambda _: "invalid_xyz")
        for q in wizard.questions:
            assert answers[q["key"]] == q["default"]

    def test_build_config_from_answers(self):
        wizard = ConfigWizard()
        answers = {
            "scenario": "balanced",
            "state_dim": 32,
            "max_steps": 500,
            "seed": 42,
        }
        config = wizard.build_config(answers)
        assert isinstance(config, SPEPAConfig)
        assert config.state_dim == 32
        assert config.max_steps == 500

    def test_build_config_obs_dim_follows_state_dim(self):
        wizard = ConfigWizard()
        answers = {
            "scenario": "balanced",
            "state_dim": 32,
            "max_steps": 1000,
            "seed": 42,
        }
        config = wizard.build_config(answers)
        assert config.obs_dim == 32

    def test_build_config_validates(self):
        wizard = ConfigWizard()
        answers = {
            "scenario": "balanced",
            "state_dim": 16,
            "max_steps": 1000,
            "seed": 42,
        }
        config = wizard.build_config(answers)
        # 不抛异常即可
        config.validate()

    def test_run_full_workflow(self):
        wizard = ConfigWizard()
        inputs = iter(["testing", "8", "100", "42"])
        config = wizard.run(input_func=lambda _: next(inputs))
        assert isinstance(config, SPEPAConfig)
        assert config.max_steps == 100

    def test_build_config_with_partial_answers(self):
        wizard = ConfigWizard()
        # 只提供部分答案
        answers = {"scenario": "balanced"}
        config = wizard.build_config(answers)
        assert isinstance(config, SPEPAConfig)


# ============================================================
# format_advice
# ============================================================

class TestFormatAdvice:
    """格式化建议测试。"""

    def test_format_valid_config(self):
        config = get_preset_config("balanced")
        result = validate_config(config)
        text = format_advice(result)
        assert isinstance(text, str)
        assert "SPEPA 配置建议报告" in text
        assert "健康分" in text

    def test_format_invalid_config(self):
        config = get_preset_config("balanced").merge({"state_dim": -1})
        result = validate_config(config)
        text = format_advice(result)
        assert "未通过" in text

    def test_format_contains_summary(self):
        config = get_preset_config("balanced")
        result = validate_config(config)
        text = format_advice(result)
        assert "总结" in text

    def test_format_contains_severity_sections(self):
        config = get_preset_config("balanced").merge({
            "gap_level_thresholds": [0.5, 0.2, 1.0, 2.0]
        })
        result = validate_config(config)
        text = format_advice(result)
        assert "严重问题" in text

    def test_format_no_advice(self):
        # 构造一个无建议的结果
        result = {
            "valid": True,
            "advice": [],
            "score": 100.0,
            "summary": "配置合理",
            "base_issues": [],
            "base_warnings": [],
        }
        text = format_advice(result)
        assert "无优化建议" in text


# ============================================================
# format_diff
# ============================================================

class TestFormatDiff:
    """格式化配置对比测试。"""

    def test_format_identical(self):
        config_a = get_preset_config("balanced")
        config_b = get_preset_config("balanced")
        result = diff_configs(config_a, config_b)
        text = format_diff(result)
        assert "SPEPA 配置对比报告" in text
        assert "相同" in text

    def test_format_different(self):
        config_a = get_preset_config("minimal")
        config_b = get_preset_config("performance")
        result = diff_configs(config_a, config_b)
        text = format_diff(result)
        assert "差异" in text

    def test_format_contains_field_names(self):
        config_a = get_preset_config("minimal")
        config_b = get_preset_config("performance")
        result = diff_configs(config_a, config_b)
        text = format_diff(result)
        assert "state_dim" in text

    def test_format_contains_impact(self):
        config_a = get_preset_config("minimal")
        config_b = get_preset_config("performance")
        result = diff_configs(config_a, config_b)
        text = format_diff(result)
        # 应包含至少一个影响描述关键词
        assert "影响" in text or "改变" in text


# ============================================================
# 端到端
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_full_workflow_validate_then_optimize(self):
        """完整工作流：验证 → 优化建议。"""
        config = get_preset_config("balanced")
        # 1. 验证
        result = validate_config(config)
        assert result["valid"] is True
        # 2. 优化建议
        advice = suggest_optimizations(config)
        assert isinstance(advice, list)
        # 3. 格式化
        text = format_advice(result)
        assert "SPEPA 配置建议报告" in text

    def test_scenario_recommend_then_validate(self):
        """场景推荐 → 验证。"""
        for scenario in list_scenarios():
            config = recommend_for_scenario(scenario)
            result = validate_config(config)
            assert result["valid"] is True

    def test_wizard_to_config_to_validate(self):
        """向导 → 配置 → 验证。"""
        wizard = ConfigWizard()
        inputs = iter(["research", "32", "5000", "42"])
        config = wizard.run(input_func=lambda _: next(inputs))
        result = validate_config(config)
        assert result["valid"] is True

    def test_compare_two_scenarios(self):
        """对比两个场景的配置。"""
        config_a = recommend_for_scenario("embedded")
        config_b = recommend_for_scenario("research")
        result = diff_configs(config_a, config_b)
        assert not result["identical"]
        text = format_diff(result)
        assert "差异" in text

    def test_optimize_with_simulated_metrics(self):
        """模拟运行指标 → 优化建议。"""
        config = get_preset_config("balanced")
        # 模拟不收敛的误差曲线
        metrics = {
            "prediction_error": [0.8, 0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.75],
            "confidence": [0.2, 0.2, 0.2, 0.2, 0.2],
        }
        advice = suggest_optimizations(config, metrics)
        assert len(advice) > 0
        # 应有针对学习率或步数的建议
        fields = [a.field for a in advice]
        assert any("learning_rate" in f or "max_steps" in f for f in fields)

    def test_advice_chain_validate_optimize_format(self):
        """链式调用：验证 + 优化 + 格式化。"""
        config = get_preset_config("balanced").merge({"learning_rate_micro": 0.2})
        validate_result = validate_config(config)
        optimize_advice = suggest_optimizations(config)
        text = format_advice(validate_result)
        assert isinstance(text, str)
        assert isinstance(optimize_advice, list)
