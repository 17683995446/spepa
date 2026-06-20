"""SPEPA 技能系统：技能抽象与技能库。

设计原则（第一性原理 - 神经可塑性）：
- 技能 = 稳定的状态-动作映射模式（重复成功的动作模式被固化）
- 技能效果模型 = 预测执行技能后的状态变化
- 技能可复用、可组合、可遗忘（基于使用频率和成功率）

技能生命周期：
1. 探索：Agent 尝试不同动作
2. 提取：成功模式被 SkillLearner 提取为技能
3. 固化：重复使用且成功率高的技能被固化
4. 复用：相似状态下直接调用技能（跳过探索）
5. 遗忘：长期未使用或成功率低的技能被遗忘
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.utils.metrics import cosine_similarity


@dataclass
class Skill:
    """技能：稳定的状态-动作映射模式。

    属性：
    - name: 技能名称
    - trigger_state: 触发该技能的典型状态（原型）
    - action: 执行的动作
    - expected_effect: 预期的状态变化
    - success_count: 成功次数
    - failure_count: 失败次数
    - use_count: 总使用次数
    - last_used: 最后使用时间戳
    - consolidated: 是否已固化
    """
    name: str
    trigger_state: np.ndarray
    action: np.ndarray
    expected_effect: np.ndarray
    success_count: int = 0
    failure_count: int = 0
    use_count: int = 0
    last_used: float = 0.0
    consolidated: bool = False

    @property
    def success_rate(self) -> float:
        """成功率。"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    @property
    def score(self) -> float:
        """技能得分（用于排序和遗忘）。"""
        # 成功率 * 使用频率 * 固化加成
        frequency = np.log1p(self.use_count)
        consolidation_bonus = 1.5 if self.consolidated else 1.0
        return self.success_rate * frequency * consolidation_bonus

    def matches(self, state: np.ndarray, threshold: float = 0.7) -> float:
        """状态与技能触发状态的相似度。"""
        sim = cosine_similarity(state, self.trigger_state)
        return sim

    def record_outcome(self, success: bool, timestamp: float = 0.0) -> None:
        """记录技能执行结果。"""
        self.use_count += 1
        self.last_used = timestamp
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "success_rate": self.success_rate,
            "use_count": self.use_count,
            "score": self.score,
            "consolidated": self.consolidated,
            "trigger_norm": float(np.linalg.norm(self.trigger_state)),
        }


class SkillLibrary(LoggedMixin):
    """技能库：管理所有已学技能。

    功能：
    - 存储技能
    - 按状态检索匹配技能
    - 固化高频高成功率技能
    - 遗忘低频低成功率技能
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.skills.skill")
        self.config = config
        self.dim = config.state_dim
        self.skills: Dict[str, Skill] = {}
        # 固化阈值：使用次数 >= 5 且成功率 >= 0.7
        self.consolidate_min_uses = 5
        self.consolidate_min_success_rate = 0.7
        # 遗忘阈值：使用次数 < 3 且成功率 < 0.3
        self.forget_min_uses = 3
        self.forget_max_success_rate = 0.3
        # 技能库容量上限
        self.max_skills = 50
        self.log.info("SkillLibrary initialized (max_skills=%d)", self.max_skills)

    def add(self, skill: Skill) -> None:
        """添加技能（若同名则合并）。"""
        if skill.name in self.skills:
            existing = self.skills[skill.name]
            # 合并：加权平均触发状态和动作
            total = existing.use_count + skill.use_count + 1
            w1 = existing.use_count / total
            w2 = skill.use_count / total
            existing.trigger_state = w1 * existing.trigger_state + w2 * skill.trigger_state
            existing.action = w1 * existing.action + w2 * skill.action
            existing.expected_effect = w1 * existing.expected_effect + w2 * skill.expected_effect
            existing.success_count += skill.success_count
            existing.failure_count += skill.failure_count
            existing.use_count += skill.use_count
            self.log.debug("Skill '%s' merged (total_uses=%d)", skill.name, existing.use_count)
        else:
            self.skills[skill.name] = skill
            self.log.debug("Skill '%s' added (uses=%d)", skill.name, skill.use_count)
        # 容量超限时遗忘低分技能
        if len(self.skills) > self.max_skills:
            self._forget_low_score()

    def retrieve(self, state: np.ndarray, threshold: float = 0.7,
                 top_k: int = 3) -> List[Skill]:
        """检索匹配状态的技能（按相似度排序）。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        if state.shape[0] != self.dim:
            state = np.resize(state, self.dim)
        if not np.all(np.isfinite(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        scored = []
        for skill in self.skills.values():
            sim = skill.matches(state, threshold)
            if sim >= threshold:
                # 综合得分 = 相似度 * 技能得分
                combined = sim * (0.5 + skill.score)
                scored.append((combined, skill))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top_k]]

    def consolidate(self) -> int:
        """固化高频高成功率技能。返回固化数量。"""
        count = 0
        for skill in self.skills.values():
            if (not skill.consolidated and
                skill.use_count >= self.consolidate_min_uses and
                skill.success_rate >= self.consolidate_min_success_rate):
                skill.consolidated = True
                count += 1
        if count > 0:
            self.log.info("Consolidated %d skills", count)
        return count

    def _forget_low_score(self) -> int:
        """遗忘低分技能。返回遗忘数量。"""
        to_forget = []
        for name, skill in self.skills.items():
            if (skill.use_count < self.forget_min_uses and
                skill.success_rate < self.forget_max_success_rate and
                not skill.consolidated):
                to_forget.append(name)
        for name in to_forget:
            del self.skills[name]
        if to_forget:
            self.log.info("Forgot %d low-score skills", len(to_forget))
        return len(to_forget)

    def get_best_skill(self) -> Optional[Skill]:
        """返回得分最高的技能。"""
        if not self.skills:
            return None
        return max(self.skills.values(), key=lambda s: s.score)

    def get_state(self) -> Dict[str, Any]:
        return {
            "n_skills": len(self.skills),
            "n_consolidated": sum(1 for s in self.skills.values() if s.consolidated),
            "avg_success_rate": (
                np.mean([s.success_rate for s in self.skills.values()])
                if self.skills else 0.0
            ),
            "total_uses": sum(s.use_count for s in self.skills.values()),
        }
