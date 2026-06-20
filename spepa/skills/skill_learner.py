"""SPEPA 技能学习器：从经验中提取和增强技能。

设计原则（第一性原理 - 神经可塑性）：
- 重复成功的动作模式被提取为技能（Hebbian 巩固）
- 技能效果通过预测误差评估（预测编码）
- 技能可迁移到相似状态（泛化）

工作流程：
1. 记录每次 (state, action, effect, success) 经验
2. 当相似状态下的动作模式重复出现且成功时，提取为技能
3. 技能效果模型 = 动作到效果的平均映射
4. 定期固化高频高成功率技能
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.utils.metrics import cosine_similarity
from spepa.skills.skill import Skill, SkillLibrary


class SkillLearner(LoggedMixin):
    """技能学习器：从经验中提取技能。

    维护一个经验缓冲区，检测重复的成功模式并提取为技能。
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.skills.skill_learner")
        self.config = config
        self.dim = config.state_dim
        self.action_dim = config.action_dim
        self.library = SkillLibrary(config)
        # 经验缓冲区：(state, action, effect, success, timestamp)
        self.experience_buffer: deque = deque(maxlen=200)
        # 模式检测：相似状态-动作聚类的中心
        self.pattern_centers: List[Dict[str, Any]] = []
        # 提取阈值：同一模式重复出现 >= 3 次才提取
        self.extraction_threshold = 3
        # 相似度阈值：状态相似度 >= 0.8 视为同一模式
        self.similarity_threshold = 0.8
        # 技能计数器（用于命名）
        self._skill_counter = 0
        self.log.info("SkillLearner initialized (buffer_size=200)")

    def record_experience(self, state: np.ndarray, action: np.ndarray,
                          effect: np.ndarray, success: bool,
                          timestamp: float = 0.0) -> None:
        """记录一次经验。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        action = np.asarray(action, dtype=np.float64).flatten()
        effect = np.asarray(effect, dtype=np.float64).flatten()
        # 维度对齐
        if state.shape[0] != self.dim:
            state = np.resize(state, self.dim)
        if action.shape[0] != self.action_dim:
            action = np.resize(action, self.action_dim)
        if effect.shape[0] != self.dim:
            effect = np.resize(effect, self.dim)
        # 清洗 NaN/Inf
        for arr in (state, action, effect):
            if not np.all(np.isfinite(arr)):
                arr[:] = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        self.experience_buffer.append({
            "state": state,
            "action": action,
            "effect": effect,
            "success": success,
            "timestamp": timestamp,
        })
        self.log.debug("Experience recorded (success=%s, buffer_size=%d)",
                       success, len(self.experience_buffer))

    def extract_skills(self) -> List[Skill]:
        """从经验缓冲区提取技能。

        算法：
        1. 只考虑成功的经验
        2. 按状态相似度聚类
        3. 同一聚类中动作相似度高的模式提取为技能
        4. 技能效果 = 该聚类中效果的平均
        """
        if len(self.experience_buffer) < self.extraction_threshold:
            return []
        # 只考虑成功的经验
        successes = [e for e in self.experience_buffer if e["success"]]
        if len(successes) < self.extraction_threshold:
            return []
        # 聚类：基于状态相似度
        clusters: List[List[Dict[str, Any]]] = []
        for exp in successes:
            assigned = False
            for cluster in clusters:
                # 与聚类中心比较
                center_state = cluster[0]["state"]
                sim = cosine_similarity(exp["state"], center_state)
                if sim >= self.similarity_threshold:
                    # 检查动作也相似
                    center_action = cluster[0]["action"]
                    action_sim = cosine_similarity(exp["action"], center_action)
                    if action_sim >= 0.7:
                        cluster.append(exp)
                        assigned = True
                        break
            if not assigned:
                clusters.append([exp])
        # 提取技能：聚类大小 >= extraction_threshold
        new_skills = []
        for cluster in clusters:
            if len(cluster) >= self.extraction_threshold:
                skill = self._create_skill_from_cluster(cluster)
                if skill is not None:
                    self.library.add(skill)
                    new_skills.append(skill)
        if new_skills:
            self.log.info("Extracted %d new skills (total=%d)",
                          len(new_skills), len(self.library.skills))
        return new_skills

    def _create_skill_from_cluster(self, cluster: List[Dict[str, Any]]) -> Optional[Skill]:
        """从聚类创建技能。"""
        if not cluster:
            return None
        # 平均状态、动作、效果
        states = np.array([e["state"] for e in cluster])
        actions = np.array([e["action"] for e in cluster])
        effects = np.array([e["effect"] for e in cluster])
        trigger_state = np.mean(states, axis=0)
        action = np.mean(actions, axis=0)
        expected_effect = np.mean(effects, axis=0)
        # 清洗
        trigger_state = np.nan_to_num(trigger_state, nan=0.0, posinf=1.0, neginf=0.0)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        expected_effect = np.nan_to_num(expected_effect, nan=0.0, posinf=1.0, neginf=0.0)
        # 命名
        self._skill_counter += 1
        name = f"skill_{self._skill_counter:04d}"
        skill = Skill(
            name=name,
            trigger_state=trigger_state,
            action=action,
            expected_effect=expected_effect,
            success_count=len(cluster),
            use_count=len(cluster),
        )
        self.log.debug("Created skill '%s' from cluster of size %d", name, len(cluster))
        return skill

    def apply_skill(self, state: np.ndarray) -> Optional[Skill]:
        """根据当前状态检索并返回最佳匹配技能。"""
        skills = self.library.retrieve(state, threshold=0.7, top_k=1)
        if skills:
            self.log.debug("Skill '%s' matched (score=%.4f)",
                          skills[0].name, skills[0].score)
            return skills[0]
        return None

    def update_skill_outcome(self, skill: Skill, success: bool,
                             timestamp: float = 0.0) -> None:
        """更新技能执行结果。"""
        skill.record_outcome(success, timestamp)
        self.log.debug("Skill '%s' outcome updated (success=%s, rate=%.2f)",
                       skill.name, success, skill.success_rate)

    def consolidate(self) -> int:
        """固化技能。"""
        return self.library.consolidate()

    def get_state(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self.experience_buffer),
            "library": self.library.get_state(),
            "patterns_detected": len(self.pattern_centers),
        }
