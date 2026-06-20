"""SPEPA 系统配置。

资源约束：
- CPU <= 20%
- 内存 <= 20%
- 存储保留 Top 30%
- 探索比例 >= 30%（即 70/30 利用/探索）

预设配置（get_preset_config）：
- "minimal"  : 最小资源占用，适合嵌入式/测试
- "balanced" : 平衡配置，适合通用场景（默认）
- "performance" : 高性能配置，适合服务器/研究
- "conservative" : 保守配置，强调安全与稳定
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from spepa.utils.logger import get_logger


@dataclass
class ResourceLimits:
    """资源约束配置。"""
    cpu_max_percent: float = 20.0          # CPU 使用率上限
    memory_max_percent: float = 20.0       # 内存使用率上限
    storage_keep_top_ratio: float = 0.30   # 存储仅保留 Top 30%


@dataclass
class ExplorationConfig:
    """探索/利用配置。"""
    exploration_ratio_min: float = 0.30    # 探索比例下限
    exploration_ratio_max: float = 0.50    # 探索比例上限
    initial_exploration_ratio: float = 0.30
    decay: float = 0.995                   # 探索比例衰减


@dataclass
class DriveWeights:
    """三大驱力权重。"""
    safety: float = 1.0      # Reward_safe
    gain: float = 1.0        # Reward_gain
    efficiency: float = 0.5  # Reward_eff


@dataclass
class SPEPAConfig:
    """SPEPA 主配置。"""
    # 资源约束
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    # 探索利用
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    # 驱力权重
    drives: DriveWeights = field(default_factory=DriveWeights)

    # 状态维度
    state_dim: int = 16
    action_dim: int = 8
    obs_dim: int = 16

    # 预测模型层级
    prediction_layers: int = 3

    # 自由能相关
    free_energy_threshold: float = 1.0
    gap_kl_threshold: float = 0.5

    # 缺口四级阈值
    gap_level_thresholds: tuple = (0.2, 0.5, 1.0, 2.0)

    # 学习率
    learning_rate_micro: float = 0.01   # 微修正学习率
    learning_rate_struct: float = 0.001  # 结构性重构学习率

    # 记忆系统
    fast_memory_capacity: int = 1000
    slow_memory_capacity: int = 200
    consolidation_interval: int = 50    # 每 N 步触发一次巩固

    # 离线复盘
    offline_load_threshold: float = 0.5  # 负载低于此值触发离线复盘
    offline_replay_batch: int = 16

    # 安全
    safety_hard_boundary: float = 0.95  # 超过硬边界则强制中止

    # 元认知
    confidence_calibration_window: int = 50

    # 优化器
    micro_correction_threshold: float = 0.3  # 误差小于该值用微修正
    structural_reconstruction_threshold: float = 1.0  # 误差大于该值用结构性重构

    # 运行时
    max_steps: int = 1000
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """导出完整配置为字典（包含所有字段，用于持久化）。"""
        return {
            "resources": {
                "cpu_max_percent": self.resources.cpu_max_percent,
                "memory_max_percent": self.resources.memory_max_percent,
                "storage_keep_top_ratio": self.resources.storage_keep_top_ratio,
            },
            "exploration": {
                "exploration_ratio_min": self.exploration.exploration_ratio_min,
                "exploration_ratio_max": self.exploration.exploration_ratio_max,
                "initial_exploration_ratio": self.exploration.initial_exploration_ratio,
                "decay": self.exploration.decay,
            },
            "drives": {
                "safety": self.drives.safety,
                "gain": self.drives.gain,
                "efficiency": self.drives.efficiency,
            },
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "obs_dim": self.obs_dim,
            "prediction_layers": self.prediction_layers,
            "free_energy_threshold": self.free_energy_threshold,
            "gap_kl_threshold": self.gap_kl_threshold,
            "gap_level_thresholds": list(self.gap_level_thresholds),
            "learning_rate_micro": self.learning_rate_micro,
            "learning_rate_struct": self.learning_rate_struct,
            "fast_memory_capacity": self.fast_memory_capacity,
            "slow_memory_capacity": self.slow_memory_capacity,
            "consolidation_interval": self.consolidation_interval,
            "offline_load_threshold": self.offline_load_threshold,
            "offline_replay_batch": self.offline_replay_batch,
            "safety_hard_boundary": self.safety_hard_boundary,
            "confidence_calibration_window": self.confidence_calibration_window,
            "micro_correction_threshold": self.micro_correction_threshold,
            "structural_reconstruction_threshold": self.structural_reconstruction_threshold,
            "max_steps": self.max_steps,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SPEPAConfig":
        """从字典重建配置（与 to_dict 互逆）。

        参数：
        - data: to_dict() 返回的字典，或部分字段字典

        返回：SPEPAConfig 实例

        示例：
            config = SPEPAConfig.from_dict(loaded_data)
        """
        data = dict(data)  # 浅拷贝避免修改原字典
        # 处理嵌套子配置
        resources_data = data.pop("resources", {})
        exploration_data = data.pop("exploration", {})
        drives_data = data.pop("drives", {})
        # gap_level_thresholds 需要转回 tuple
        if "gap_level_thresholds" in data:
            glt = data["gap_level_thresholds"]
            if isinstance(glt, list):
                data["gap_level_thresholds"] = tuple(glt)
        # 构建子配置对象
        resources = ResourceLimits(**resources_data) if resources_data else ResourceLimits()
        exploration = ExplorationConfig(**exploration_data) if exploration_data else ExplorationConfig()
        drives = DriveWeights(**drives_data) if drives_data else DriveWeights()
        # 过滤掉不属于 SPEPAConfig 的键
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(
            resources=resources,
            exploration=exploration,
            drives=drives,
            **filtered,
        )

    def save_to_file(self, filepath: str) -> str:
        """保存配置到文件（JSON 或 YAML 格式，根据扩展名自动判断）。

        参数：
        - filepath: 文件路径（.json 或 .yaml/.yml）

        返回：实际保存的文件路径

        示例：
            config.save_to_file("my_config.json")
            config.save_to_file("my_config.yaml")
        """
        import os
        ext = os.path.splitext(filepath)[1].lower()
        data = self.to_dict()
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "保存 YAML 需要 PyYAML 库。请运行: pip install pyyaml"
                )
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            # 默认 JSON
            if not ext:
                filepath = filepath + ".json"
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath

    @classmethod
    def load_from_file(cls, filepath: str) -> "SPEPAConfig":
        """从文件加载配置（JSON 或 YAML 格式，根据扩展名自动判断）。

        参数：
        - filepath: 文件路径（.json 或 .yaml/.yml）

        返回：SPEPAConfig 实例（已校验）

        示例：
            config = SPEPAConfig.load_from_file("my_config.json")
        """
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "加载 YAML 需要 PyYAML 库。请运行: pip install pyyaml"
                )
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        config = cls.from_dict(data)
        config.validate()
        return config

    def merge(self, overrides: Dict[str, Any]) -> "SPEPAConfig":
        """合并部分配置覆盖，返回新配置（不修改原配置）。

        参数：
        - overrides: 部分字段字典，支持嵌套（如 {"resources": {"cpu_max_percent": 50}}）

        返回：合并后的新 SPEPAConfig 实例

        示例：
            config = get_default_config()
            new_config = config.merge({"state_dim": 32, "seed": 100})
        """
        base = self.to_dict()
        # 深度合并嵌套字典
        for key, val in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                base[key] = {**base[key], **val}
            else:
                base[key] = val
        return SPEPAConfig.from_dict(base)

    def validate(self) -> None:
        """校验配置合法性，不合法时抛出 ValueError 并给出友好提示。

        第一性原理：在系统启动前尽早发现配置错误，避免运行时崩溃。
        """
        errors = []
        if self.state_dim <= 0:
            errors.append(f"state_dim 必须为正整数，当前为 {self.state_dim}")
        if self.action_dim <= 0:
            errors.append(f"action_dim 必须为正整数，当前为 {self.action_dim}")
        if self.obs_dim <= 0:
            errors.append(f"obs_dim 必须为正整数，当前为 {self.obs_dim}")
        if self.prediction_layers <= 0:
            errors.append(f"prediction_layers 必须为正整数，当前为 {self.prediction_layers}")
        if self.learning_rate_micro <= 0:
            errors.append(f"learning_rate_micro 必须为正数，当前为 {self.learning_rate_micro}")
        if self.learning_rate_struct <= 0:
            errors.append(f"learning_rate_struct 必须为正数，当前为 {self.learning_rate_struct}")
        if not (0.0 < self.safety_hard_boundary <= 1.0):
            errors.append(f"safety_hard_boundary 必须在 (0, 1] 范围内，当前为 {self.safety_hard_boundary}")
        if not (0.0 <= self.exploration.exploration_ratio_min <= 1.0):
            errors.append(f"exploration_ratio_min 必须在 [0, 1] 范围内，当前为 {self.exploration.exploration_ratio_min}")
        if self.exploration.exploration_ratio_min > self.exploration.exploration_ratio_max:
            errors.append(
                f"exploration_ratio_min({self.exploration.exploration_ratio_min}) "
                f"不能大于 exploration_ratio_max({self.exploration.exploration_ratio_max})"
            )
        if self.micro_correction_threshold >= self.structural_reconstruction_threshold:
            errors.append(
                f"micro_correction_threshold({self.micro_correction_threshold}) "
                f"必须小于 structural_reconstruction_threshold({self.structural_reconstruction_threshold})"
            )
        if self.resources.cpu_max_percent <= 0:
            errors.append(f"cpu_max_percent 必须为正数，当前为 {self.resources.cpu_max_percent}")
        if self.resources.memory_max_percent <= 0:
            errors.append(f"memory_max_percent 必须为正数，当前为 {self.resources.memory_max_percent}")
        if errors:
            msg = "SPEPA 配置校验失败，共 {} 个错误:\n".format(len(errors))
            msg += "\n".join("  - " + e for e in errors)
            raise ValueError(msg)


# ============================================================
# 预设配置
# ============================================================

def _minimal_config() -> SPEPAConfig:
    """最小资源占用配置：适合嵌入式/测试。"""
    return SPEPAConfig(
        state_dim=8,
        action_dim=4,
        obs_dim=8,
        prediction_layers=2,
        fast_memory_capacity=200,
        slow_memory_capacity=50,
        max_steps=200,
        resources=ResourceLimits(cpu_max_percent=10.0, memory_max_percent=10.0),
        exploration=ExplorationConfig(
            exploration_ratio_min=0.20,
            exploration_ratio_max=0.40,
            initial_exploration_ratio=0.20,
        ),
    )


def _balanced_config() -> SPEPAConfig:
    """平衡配置：适合通用场景（默认）。"""
    return SPEPAConfig()


def _performance_config() -> SPEPAConfig:
    """高性能配置：适合服务器/研究。"""
    return SPEPAConfig(
        state_dim=32,
        action_dim=16,
        obs_dim=32,
        prediction_layers=5,
        fast_memory_capacity=5000,
        slow_memory_capacity=1000,
        consolidation_interval=30,
        offline_replay_batch=32,
        max_steps=5000,
        resources=ResourceLimits(cpu_max_percent=80.0, memory_max_percent=80.0),
        exploration=ExplorationConfig(
            exploration_ratio_min=0.10,
            exploration_ratio_max=0.30,
            initial_exploration_ratio=0.10,
        ),
    )


def _conservative_config() -> SPEPAConfig:
    """保守配置：强调安全与稳定。"""
    return SPEPAConfig(
        state_dim=16,
        action_dim=8,
        obs_dim=16,
        prediction_layers=3,
        learning_rate_micro=0.005,
        learning_rate_struct=0.0005,
        safety_hard_boundary=0.85,
        max_steps=500,
        resources=ResourceLimits(cpu_max_percent=15.0, memory_max_percent=15.0),
        exploration=ExplorationConfig(
            exploration_ratio_min=0.40,
            exploration_ratio_max=0.60,
            initial_exploration_ratio=0.40,
        ),
        drives=DriveWeights(safety=2.0, gain=0.5, efficiency=1.0),
    )


# 预设配置注册表
_PRESETS: Dict[str, Any] = {
    "minimal": _minimal_config,
    "balanced": _balanced_config,
    "performance": _performance_config,
    "conservative": _conservative_config,
}


def get_preset_config(preset: str = "balanced") -> SPEPAConfig:
    """获取预设配置。

    参数：
    - preset: 预设名称，可选 "minimal"/"balanced"/"performance"/"conservative"

    返回：对应的 SPEPAConfig 实例（已校验）

    抛出：ValueError 当 preset 名称不存在时
    """
    logger = get_logger("spepa.config")
    preset = preset.lower().strip()
    if preset not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(
            f"未知预设配置 '{preset}'。可用预设: {available}。"
            f"提示：使用 get_default_config() 获取默认配置。"
        )
    config = _PRESETS[preset]()
    config.validate()
    logger.info("Preset config loaded: %s (state_dim=%d, max_steps=%d)",
                preset, config.state_dim, config.max_steps)
    return config


def list_presets() -> Dict[str, str]:
    """列出所有可用预设配置及其描述。"""
    return {
        "minimal": "最小资源占用，适合嵌入式/测试（state_dim=8, max_steps=200）",
        "balanced": "平衡配置，适合通用场景（默认，state_dim=16, max_steps=1000）",
        "performance": "高性能配置，适合服务器/研究（state_dim=32, max_steps=5000）",
        "conservative": "保守配置，强调安全与稳定（safety=2.0, max_steps=500）",
    }


# 默认全局配置
DEFAULT_CONFIG = SPEPAConfig()


def get_default_config() -> SPEPAConfig:
    """返回默认配置（等同于 get_preset_config("balanced")）。"""
    logger = get_logger("spepa.config")
    logger.info("Default config loaded: cpu<=%s%%, mem<=%s%%, explore>=%s%%",
                DEFAULT_CONFIG.resources.cpu_max_percent,
                DEFAULT_CONFIG.resources.memory_max_percent,
                DEFAULT_CONFIG.exploration.exploration_ratio_min)
    return DEFAULT_CONFIG
