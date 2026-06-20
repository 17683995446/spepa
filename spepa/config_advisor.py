"""SPEPA 配置验证与智能建议系统。

设计原则（第一性原理）：
- 场景驱动：根据用户场景推荐配置，而非让用户手动调参
- 影响可量化：每个建议都说明预期影响（性能/稳定性/内存）
- 渐进式：从合法性 → 合理性 → 优化建议，逐层深入
- 可解释：每个推荐参数都附带理由
- 不重复：合法性检查复用 diagnostics.diagnose_config，本模块聚焦"合理性"与"建议"

与 diagnostics.diagnose_config 的分工：
- diagnose_config : 检查配置是否"合法"（能否运行）
- validate_config  : 检查配置是否"合理"（运行得好不好）
- suggest_optimizations : 给出"如何更好"的建议

主要功能：
- ConfigAdvice          : 配置建议数据结构
- validate_config()     : 深度合理性验证（返回建议列表 + 评分）
- recommend_for_scenario() : 根据场景推荐配置
- suggest_optimizations()  : 基于运行历史给出优化建议
- diff_configs()        : 对比两份配置的差异与影响
- ConfigWizard          : 交互式配置向导
- format_advice()       : 格式化建议为可读文本
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from spepa.utils.config import (
    SPEPAConfig, get_preset_config, list_presets,
)
from spepa.utils.logger import get_logger
from spepa.diagnostics import diagnose_config, ErrorSuggestion

logger = get_logger("spepa.config_advisor")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ConfigAdvice:
    """配置建议。

    与 ErrorSuggestion 的区别：
    - ErrorSuggestion 描述"错误"（必须修复）
    - ConfigAdvice 描述"建议"（可选优化）
    """
    severity: str        # "info" / "warning" / "critical"
    category: str        # performance / stability / memory / safety / exploration
    field: str           # 涉及的配置字段（如 "learning_rate_micro"）
    current: Any         # 当前值
    recommended: Any     # 推荐值
    reason: str          # 推荐理由
    impact: str = ""     # 预期影响（如 "提升收敛速度约 2 倍"）


# ============================================================
# 场景定义
# ============================================================

# 场景 → 推荐预设 + 覆盖参数
# 场景比预设更细化：预设是"配置模板"，场景是"使用情境"
_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "description": "通用平衡场景：适合大多数情况，无特殊偏好",
        "preset": "balanced",
        "overrides": {},
    },
    "embedded": {
        "description": "嵌入式/边缘设备：资源极度受限，优先低功耗",
        "preset": "minimal",
        "overrides": {
            "resources": {"cpu_max_percent": 5.0, "memory_max_percent": 5.0},
            "max_steps": 100,
            "fast_memory_capacity": 100,
            "slow_memory_capacity": 30,
        },
    },
    "testing": {
        "description": "单元测试/CI：快速运行，确定性优先",
        "preset": "minimal",
        "overrides": {
            "seed": 42,
            "max_steps": 50,
        },
    },
    "research": {
        "description": "学术研究：高维度、长步数，可接受高资源占用",
        "preset": "performance",
        "overrides": {
            "exploration": {
                "exploration_ratio_min": 0.20,
                "exploration_ratio_max": 0.40,
            },
        },
    },
    "production": {
        "description": "生产环境：稳定性优先，适度资源占用",
        "preset": "conservative",
        "overrides": {
            "safety_hard_boundary": 0.90,
            "max_steps": 2000,
        },
    },
    "realtime": {
        "description": "实时控制：低延迟优先，小步数高频次",
        "preset": "minimal",
        "overrides": {
            "state_dim": 16,
            "action_dim": 8,
            "max_steps": 20,
            "learning_rate_micro": 0.05,
            "consolidation_interval": 10,
        },
    },
    "educational": {
        "description": "教学演示：易理解，参数温和",
        "preset": "balanced",
        "overrides": {
            "max_steps": 200,
            "seed": 42,
        },
    },
}


def list_scenarios() -> Dict[str, str]:
    """列出所有可用场景及其描述。"""
    return {name: info["description"] for name, info in _SCENARIOS.items()}


# ============================================================
# 合理性验证
# ============================================================

def validate_config(config: SPEPAConfig) -> Dict[str, Any]:
    """深度配置合理性验证。

    与 diagnose_config 的区别：
    - diagnose_config 只检查"能否运行"（合法性）
    - validate_config 检查"运行得好不好"（合理性），返回优化建议

    参数：
        config : SPEPAConfig 实例

    返回：
        {
            "valid": bool,           # 是否合法（复用 diagnose_config）
            "advice": List[ConfigAdvice],  # 合理性建议
            "score": float,          # 配置健康分（0-100）
            "summary": str,          # 一句话总结
        }
    """
    # 1. 复用合法性检查
    base = diagnose_config(config)
    advice: List[ConfigAdvice] = []

    # 2. 合理性检查（不报错，只给建议）

    # 2.1 学习率与步数搭配
    # 第一性原理：学习率大 → 收敛快但易震荡 → 步数应少；学习率小 → 需更多步数
    lr_micro = config.learning_rate_micro
    max_steps = config.max_steps
    if lr_micro > 0.05 and max_steps > 2000:
        advice.append(ConfigAdvice(
            severity="warning",
            category="stability",
            field="learning_rate_micro / max_steps",
            current=f"lr={lr_micro}, steps={max_steps}",
            recommended="lr<=0.05 或 steps<=2000",
            reason="学习率较大且步数过多，可能导致后期震荡",
            impact="可能降低最终收敛精度",
        ))
    elif lr_micro < 0.001 and max_steps < 500:
        advice.append(ConfigAdvice(
            severity="warning",
            category="performance",
            field="learning_rate_micro / max_steps",
            current=f"lr={lr_micro}, steps={max_steps}",
            recommended="lr>=0.001 或 steps>=500",
            reason="学习率过小且步数不足，可能无法收敛",
            impact="模型可能欠训练",
        ))

    # 2.2 记忆容量与状态维度搭配
    # 第一性原理：状态维度大 → 每条记忆占用大 → 容量应相应增大
    state_dim = config.state_dim
    fast_cap = config.fast_memory_capacity
    if state_dim >= 32 and fast_cap < 500:
        advice.append(ConfigAdvice(
            severity="warning",
            category="memory",
            field="fast_memory_capacity",
            current=fast_cap,
            recommended=">= 500",
            reason=f"状态维度 {state_dim} 较大，记忆容量过小可能导致样本不足",
            impact="影响学习多样性，可能过拟合",
        ))
    if config.slow_memory_capacity > fast_cap * 5:
        advice.append(ConfigAdvice(
            severity="info",
            category="memory",
            field="slow_memory_capacity",
            current=config.slow_memory_capacity,
            recommended=f"<= {fast_cap * 5}",
            reason="慢速记忆远大于快速记忆，巩固效率可能低下",
            impact="部分慢速记忆可能长期未被巩固",
        ))

    # 2.3 探索比例范围合理性
    exp = config.exploration
    if exp.exploration_ratio_max - exp.exploration_ratio_min < 0.1:
        advice.append(ConfigAdvice(
            severity="info",
            category="exploration",
            field="exploration_ratio_max - exploration_ratio_min",
            current=f"[{exp.exploration_ratio_min}, {exp.exploration_ratio_max}]",
            recommended="范围 >= 0.1",
            reason="探索比例范围过窄，自适应空间有限",
            impact="探索能力受限，可能陷入局部最优",
        ))
    if exp.decay >= 1.0:
        advice.append(ConfigAdvice(
            severity="warning",
            category="exploration",
            field="exploration.decay",
            current=exp.decay,
            recommended="< 1.0（如 0.995）",
            reason="衰减率 >= 1.0 表示探索比例不衰减或增长",
            impact="后期仍保持高探索比例，影响利用效率",
        ))

    # 2.4 资源约束与配置复杂度匹配
    res = config.resources
    complexity = config.state_dim * config.action_dim * config.prediction_layers
    if complexity > 1000 and res.cpu_max_percent < 20:
        advice.append(ConfigAdvice(
            severity="warning",
            category="performance",
            field="resources.cpu_max_percent",
            current=res.cpu_max_percent,
            recommended=">= 20",
            reason=f"配置复杂度 {complexity} 较高，但 CPU 上限 {res.cpu_max_percent}% 过低",
            impact="可能触发频繁降级，影响训练效果",
        ))
    if complexity > 5000 and res.memory_max_percent < 30:
        advice.append(ConfigAdvice(
            severity="warning",
            category="memory",
            field="resources.memory_max_percent",
            current=res.memory_max_percent,
            recommended=">= 30",
            reason=f"配置复杂度 {complexity} 很高，内存上限过低",
            impact="可能因内存不足触发记忆淘汰",
        ))

    # 2.5 安全边界与驱力权重匹配
    # 第一性原理：safety 权重高 → 应配合较严格的安全边界
    drives = config.drives
    if drives.safety > 1.5 and config.safety_hard_boundary > 0.95:
        advice.append(ConfigAdvice(
            severity="info",
            category="safety",
            field="safety_hard_boundary",
            current=config.safety_hard_boundary,
            recommended="<= 0.90",
            reason="safety 权重较高但安全边界宽松，行为可能不一致",
            impact="安全驱力权重与边界不匹配",
        ))

    # 2.6 巩固间隔与步数匹配
    consol = config.consolidation_interval
    if consol > max_steps // 2 and max_steps > 0:
        advice.append(ConfigAdvice(
            severity="info",
            category="performance",
            field="consolidation_interval",
            current=consol,
            recommended=f"<= {max_steps // 2}",
            reason="巩固间隔超过步数一半，可能只触发一次巩固",
            impact="记忆巩固不充分",
        ))

    # 2.7 阈值单调性（gap_level_thresholds 应递增）
    thresholds = config.gap_level_thresholds
    if len(thresholds) >= 2:
        for i in range(len(thresholds) - 1):
            if thresholds[i] >= thresholds[i + 1]:
                advice.append(ConfigAdvice(
                    severity="critical",
                    category="stability",
                    field="gap_level_thresholds",
                    current=list(thresholds),
                    recommended="严格递增的阈值序列",
                    reason=f"阈值在位置 {i} 非递增，缺口分级将失效",
                    impact="缺口检测逻辑混乱，降级策略异常",
                ))
                break

    # 2.8 微修正/重构阈值关系
    if config.micro_correction_threshold >= config.structural_reconstruction_threshold:
        advice.append(ConfigAdvice(
            severity="critical",
            category="stability",
            field="micro_correction_threshold / structural_reconstruction_threshold",
            current=f"micro={config.micro_correction_threshold}, struct={config.structural_reconstruction_threshold}",
            recommended="micro < struct",
            reason="微修正阈值应小于结构性重构阈值，否则优化器选择逻辑失效",
            impact="优化器无法正确选择修正策略",
        ))

    # 3. 计算健康分
    score = _compute_config_score(config, base, advice)

    # 4. 总结
    critical_count = sum(1 for a in advice if a.severity == "critical")
    warning_count = sum(1 for a in advice if a.severity == "warning")
    if not base["valid"]:
        summary = f"配置非法（{len(base['issues'])} 个错误），无法运行"
    elif critical_count > 0:
        summary = f"配置合法但有 {critical_count} 个严重问题，建议修复后再运行"
    elif warning_count > 0:
        summary = f"配置合法但有 {warning_count} 个警告，建议优化"
    else:
        summary = "配置合理，可以运行"

    return {
        "valid": base["valid"],
        "advice": advice,
        "score": score,
        "summary": summary,
        "base_issues": base["issues"],
        "base_warnings": base["warnings"],
    }


def _compute_config_score(config: SPEPAConfig,
                          base: Dict[str, Any],
                          advice: List[ConfigAdvice]) -> float:
    """计算配置健康分（0-100）。

    评分维度：
    - 合法性（40 分）：无错误得满分，每个错误扣 10 分
    - 合理性（60 分）：无 critical 得 30 分，无 warning 得 20 分，无 info 得 10 分
    """
    score = 100.0
    # 合法性扣分
    score -= len(base.get("issues", [])) * 10
    # 合理性扣分
    for a in advice:
        if a.severity == "critical":
            score -= 15
        elif a.severity == "warning":
            score -= 5
        elif a.severity == "info":
            score -= 1
    return max(0.0, min(100.0, score))


# ============================================================
# 场景化推荐
# ============================================================

def recommend_for_scenario(scenario: str,
                           overrides: Optional[Dict[str, Any]] = None) -> SPEPAConfig:
    """根据场景推荐配置。

    与 get_preset_config 的区别：
    - get_preset_config 返回固定预设
    - recommend_for_scenario 根据使用情境选择预设并应用场景化覆盖

    参数：
        scenario : 场景名称（见 list_scenarios()）
        overrides : 额外的配置覆盖（优先级最高）

    返回：
        SPEPAConfig 实例

    抛出：
        ValueError 当场景名称不存在时
    """
    scenario = scenario.lower().strip()
    if scenario not in _SCENARIOS:
        available = ", ".join(sorted(_SCENARIOS.keys()))
        raise ValueError(
            f"未知场景 '{scenario}'。可用场景: {available}。"
            f"提示：使用 list_scenarios() 查看所有场景。"
        )
    info = _SCENARIOS[scenario]
    config = get_preset_config(info["preset"])
    # 应用场景覆盖
    if info["overrides"]:
        config = config.merge(info["overrides"])
    # 应用用户自定义覆盖（优先级最高）
    if overrides:
        config = config.merge(overrides)
    config.validate()
    logger.info("Config recommended for scenario '%s' (preset=%s)",
                scenario, info["preset"])
    return config


# ============================================================
# 优化建议
# ============================================================

def suggest_optimizations(config: SPEPAConfig,
                          metrics_history: Optional[Dict[str, List[float]]] = None
                          ) -> List[ConfigAdvice]:
    """给出配置优化建议。

    参数：
        config : SPEPAConfig 实例
        metrics_history : 可选的运行指标历史（如 {"prediction_error": [...]}）
                         若提供，将基于实际运行数据给出更精准的建议

    返回：
        List[ConfigAdvice]
    """
    advice: List[ConfigAdvice] = []

    # 1. 基于配置本身的静态建议
    # 学习率偏低
    if config.learning_rate_micro < 0.005:
        advice.append(ConfigAdvice(
            severity="info",
            category="performance",
            field="learning_rate_micro",
            current=config.learning_rate_micro,
            recommended=0.01,
            reason="学习率偏低，收敛可能较慢",
            impact="提升收敛速度约 2 倍（视任务而定）",
        ))
    # 学习率偏高
    elif config.learning_rate_micro > 0.1:
        advice.append(ConfigAdvice(
            severity="warning",
            category="stability",
            field="learning_rate_micro",
            current=config.learning_rate_micro,
            recommended=0.05,
            reason="学习率偏高，可能导致震荡",
            impact="提升稳定性，可能略降收敛速度",
        ))

    # 2. 基于运行历史的动态建议
    if metrics_history:
        # 2.1 误差不收敛
        if "prediction_error" in metrics_history:
            errors = metrics_history["prediction_error"]
            if len(errors) >= 10:
                first_avg = sum(errors[:5]) / 5
                last_avg = sum(errors[-5:]) / 5
                if last_avg >= first_avg * 0.9 and first_avg > 0:
                    # 误差未明显下降
                    advice.append(ConfigAdvice(
                        severity="warning",
                        category="performance",
                        field="learning_rate_micro",
                        current=config.learning_rate_micro,
                        recommended=config.learning_rate_micro * 2,
                        reason=f"预测误差未下降（前5步均值 {first_avg:.4f} → 后5步均值 {last_avg:.4f}）",
                        impact="增大学习率可能加速收敛",
                    ))
                elif last_avg < first_avg * 0.3 and first_avg > 0:
                    # 误差大幅下降，可考虑减小学习率精调
                    advice.append(ConfigAdvice(
                        severity="info",
                        category="performance",
                        field="learning_rate_micro",
                        current=config.learning_rate_micro,
                        recommended=config.learning_rate_micro * 0.5,
                        reason="预测误差已大幅下降，可减小学习率精调",
                        impact="提升最终收敛精度",
                    ))

        # 2.2 信心度低
        if "confidence" in metrics_history:
            confs = metrics_history["confidence"]
            if len(confs) >= 5:
                recent_conf = sum(confs[-5:]) / 5
                if recent_conf < 0.3:
                    advice.append(ConfigAdvice(
                        severity="warning",
                        category="stability",
                        field="max_steps",
                        current=config.max_steps,
                        recommended=config.max_steps * 2,
                        reason=f"近期信心度 {recent_conf:.4f} 偏低，训练不足",
                        impact="增加步数可提升信心度",
                    ))

        # 2.3 误差震荡
        if "prediction_error" in metrics_history:
            errors = metrics_history["prediction_error"]
            if len(errors) >= 10:
                # 计算最近 10 步的标准差
                recent = errors[-10:]
                mean_err = sum(recent) / len(recent)
                variance = sum((e - mean_err) ** 2 for e in recent) / len(recent)
                std_err = variance ** 0.5
                if mean_err > 0 and std_err / mean_err > 0.5:
                    advice.append(ConfigAdvice(
                        severity="warning",
                        category="stability",
                        field="learning_rate_micro",
                        current=config.learning_rate_micro,
                        recommended=config.learning_rate_micro * 0.5,
                        reason=f"误差震荡剧烈（std/mean={std_err/mean_err:.2f}）",
                        impact="降低学习率可稳定训练",
                    ))

    return advice


# ============================================================
# 配置对比
# ============================================================

@dataclass
class ConfigDiff:
    """配置差异项。"""
    field: str           # 字段名
    value_a: Any         # 配置 A 的值
    value_b: Any         # 配置 B 的值
    impact: str = ""     # 差异的预期影响


def diff_configs(config_a: SPEPAConfig,
                 config_b: SPEPAConfig) -> Dict[str, Any]:
    """对比两份配置的差异与影响。

    参数：
        config_a : 配置 A
        config_b : 配置 B

    返回：
        {
            "diffs": List[ConfigDiff],
            "identical": bool,
            "summary": str,
        }
    """
    dict_a = config_a.to_dict()
    dict_b = config_b.to_dict()
    diffs: List[ConfigDiff] = []

    def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """扁平化嵌套字典。"""
        items: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(_flatten(v, key))
            else:
                items[key] = v
        return items

    flat_a = _flatten(dict_a)
    flat_b = _flatten(dict_b)
    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))

    for key in all_keys:
        va = flat_a.get(key, "<missing>")
        vb = flat_b.get(key, "<missing>")
        if va != vb:
            impact = _describe_diff_impact(key, va, vb)
            diffs.append(ConfigDiff(
                field=key,
                value_a=va,
                value_b=vb,
                impact=impact,
            ))

    return {
        "diffs": diffs,
        "identical": len(diffs) == 0,
        "summary": (
            "配置完全相同" if not diffs
            else f"共有 {len(diffs)} 处差异"
        ),
    }


def _describe_diff_impact(field_name: str, value_a: Any, value_b: Any) -> str:
    """描述单个配置差异的影响。"""
    impacts = {
        "state_dim": "影响状态空间维度，改变模型容量",
        "action_dim": "影响动作空间维度，改变决策复杂度",
        "obs_dim": "影响观测维度，需与环境匹配",
        "prediction_layers": "影响预测模型深度，改变计算量与表达能力",
        "learning_rate_micro": "影响微修正速度，改变收敛性与稳定性",
        "learning_rate_struct": "影响结构性重构速度",
        "fast_memory_capacity": "影响短期记忆样本数",
        "slow_memory_capacity": "影响长期记忆容量",
        "max_steps": "影响训练总步数",
        "seed": "影响随机性，改变可复现性",
        "safety_hard_boundary": "影响安全约束严格程度",
        "consolidation_interval": "影响记忆巩固频率",
    }
    # 支持嵌套字段名（如 resources.cpu_max_percent）
    for key, desc in impacts.items():
        if field_name.endswith(key):
            return desc
    return "配置值不同"


# ============================================================
# 配置向导
# ============================================================

class ConfigWizard:
    """交互式配置向导。

    通过一系列问题引导用户生成合适的配置。

    使用方式：
        wizard = ConfigWizard()
        answers = wizard.ask_questions(input_func=input)
        config = wizard.build_config(answers)
    """

    def __init__(self):
        self.questions = self._default_questions()

    def _default_questions(self) -> List[Dict[str, Any]]:
        """默认问题列表。"""
        return [
            {
                "key": "scenario",
                "question": "您的使用场景是？",
                "options": list(_SCENARIOS.keys()),
                "default": "balanced",
            },
            {
                "key": "state_dim",
                "question": "状态维度（8/16/32，未知可选 16）？",
                "options": [8, 16, 32],
                "default": 16,
            },
            {
                "key": "max_steps",
                "question": "训练步数（100/500/1000/5000）？",
                "options": [100, 500, 1000, 5000],
                "default": 1000,
            },
            {
                "key": "seed",
                "question": "随机种子（0=随机，42=固定）？",
                "options": [0, 42],
                "default": 42,
            },
        ]

    def ask_questions(self,
                      input_func: Optional[Callable[[str], str]] = None
                      ) -> Dict[str, Any]:
        """提出问题并收集答案。

        参数：
            input_func : 输入函数（默认使用内置 input，测试时可注入）

        返回：
            答案字典 {question_key: answer}
        """
        if input_func is None:
            input_func = input

        answers: Dict[str, Any] = {}
        for q in self.questions:
            options_str = "/".join(str(o) for o in q["options"])
            prompt = f"{q['question']} [{options_str}] (默认 {q['default']}): "
            raw = input_func(prompt).strip()
            if not raw:
                answers[q["key"]] = q["default"]
            else:
                # 尝试匹配选项
                matched = None
                for opt in q["options"]:
                    if str(opt) == raw or str(opt).lower() == raw.lower():
                        matched = opt
                        break
                answers[q["key"]] = matched if matched is not None else q["default"]
        return answers

    def build_config(self, answers: Dict[str, Any]) -> SPEPAConfig:
        """根据答案构建配置。

        参数：
            answers : ask_questions() 返回的答案字典

        返回：
            SPEPAConfig 实例
        """
        scenario = answers.get("scenario", "balanced")
        config = recommend_for_scenario(scenario)
        # 应用用户指定的覆盖
        overrides: Dict[str, Any] = {}
        if "state_dim" in answers:
            overrides["state_dim"] = answers["state_dim"]
            overrides["obs_dim"] = answers["state_dim"]  # obs_dim 跟随 state_dim
        if "max_steps" in answers:
            overrides["max_steps"] = answers["max_steps"]
        if "seed" in answers:
            overrides["seed"] = answers["seed"]
        if overrides:
            config = config.merge(overrides)
        config.validate()
        return config

    def run(self, input_func: Optional[Callable[[str], str]] = None
            ) -> SPEPAConfig:
        """一键运行：提问 + 构建配置。"""
        answers = self.ask_questions(input_func)
        return self.build_config(answers)


# ============================================================
# 格式化
# ============================================================

def format_advice(result: Dict[str, Any]) -> str:
    """格式化配置建议结果为可读文本。

    参数：
        result : validate_config() 的返回值

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SPEPA 配置建议报告")
    lines.append("=" * 60)

    # 合法性
    valid = result.get("valid", True)
    lines.append(f"\n[合法性] {'通过' if valid else '未通过'}")

    # 健康分
    score = result.get("score", 0.0)
    lines.append(f"[健康分] {score:.1f} / 100")

    # 总结
    lines.append(f"[总结]   {result.get('summary', '')}")

    # 基础问题（来自 diagnose_config）
    base_issues = result.get("base_issues", [])
    if base_issues:
        lines.append(f"\n[错误] ({len(base_issues)} 个)")
        for i, issue in enumerate(base_issues, 1):
            lines.append(f"  {i}. [{issue.code}] {issue.message}")
            lines.append(f"     修复: {issue.fix}")

    # 建议列表
    advice = result.get("advice", [])
    if advice:
        # 按严重程度分组
        critical = [a for a in advice if a.severity == "critical"]
        warnings = [a for a in advice if a.severity == "warning"]
        infos = [a for a in advice if a.severity == "info"]

        if critical:
            lines.append(f"\n[严重问题] ({len(critical)} 个)")
            for i, a in enumerate(critical, 1):
                lines.append(f"  {i}. [{a.category}] {a.field}")
                lines.append(f"     当前: {a.current}")
                lines.append(f"     推荐: {a.recommended}")
                lines.append(f"     原因: {a.reason}")
                if a.impact:
                    lines.append(f"     影响: {a.impact}")

        if warnings:
            lines.append(f"\n[警告] ({len(warnings)} 个)")
            for i, a in enumerate(warnings, 1):
                lines.append(f"  {i}. [{a.category}] {a.field}")
                lines.append(f"     当前: {a.current}")
                lines.append(f"     推荐: {a.recommended}")
                lines.append(f"     原因: {a.reason}")
                if a.impact:
                    lines.append(f"     影响: {a.impact}")

        if infos:
            lines.append(f"\n[提示] ({len(infos)} 个)")
            for i, a in enumerate(infos, 1):
                lines.append(f"  {i}. [{a.category}] {a.field}")
                lines.append(f"     当前: {a.current}")
                lines.append(f"     推荐: {a.recommended}")
                lines.append(f"     原因: {a.reason}")
                if a.impact:
                    lines.append(f"     影响: {a.impact}")
    else:
        lines.append("\n[建议] 无优化建议，配置合理")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def format_diff(diff_result: Dict[str, Any]) -> str:
    """格式化配置对比结果为可读文本。

    参数：
        diff_result : diff_configs() 的返回值

    返回：
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SPEPA 配置对比报告")
    lines.append("=" * 60)

    lines.append(f"\n[总结] {diff_result['summary']}")

    diffs = diff_result.get("diffs", [])
    if diffs:
        lines.append(f"\n[差异明细] ({len(diffs)} 处)")
        lines.append(f"{'字段':<40} {'配置A':<20} {'配置B':<20}")
        lines.append("-" * 80)
        for d in diffs:
            va = str(d.value_a)[:18]
            vb = str(d.value_b)[:18]
            lines.append(f"{d.field:<40} {va:<20} {vb:<20}")
            if d.impact:
                lines.append(f"  → {d.impact}")
    else:
        lines.append("\n[结果] 两份配置完全相同")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
