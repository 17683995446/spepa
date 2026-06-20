"""SPEPA 状态序列化器：将 Agent 状态序列化/反序列化。

设计原则（第一性原理）：
- 持久性原理：学习成果应能跨越进程生命周期保存
- 完整性原理：序列化应捕获所有影响行为的状态
- 幂等性原理：序列化→反序列化→序列化应产生相同结果

序列化范围：
- 预测模型：各层 mu/var/W/b
- 记忆系统：快速记忆项 + 慢速记忆项
- 技能库：所有技能
- 运行时状态：current_state、step_count、exploration_ratio
- 优化器：自适应学习率、模式历史
- 元认知：性能历史
"""
import json
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


class StateSerializer(LoggedMixin):
    """状态序列化器：将 Agent 状态转换为可持久化的字典。

    使用 JSON 兼容格式（numpy 数组转为列表）。
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.persistence.serializer")
        self.config = config
        self.log.info("StateSerializer initialized")

    def serialize_agent(self, agent: Any) -> Dict[str, Any]:
        """序列化整个 Agent 状态。"""
        state = {
            "version": 1,
            "config": self._serialize_config(agent.config),
            "runtime": self._serialize_runtime(agent),
            "predictive_model": self._serialize_predictive_model(agent.predictive_model),
            "memory_system": self._serialize_memory(agent.memory_system),
            "skill_library": self._serialize_skills(agent.skill_learner.library),
            "optimizer": self._serialize_optimizer(agent.optimizer),
            "drive_engine": self._serialize_drive_engine(agent.drive_engine),
            "goal_hierarchy": self._serialize_goal_hierarchy(agent.goal_hierarchy),
            "metacognition": self._serialize_metacognition(agent.metacognition),
        }
        self.log.debug("Agent state serialized (step=%d)", agent.step_count)
        return state

    def deserialize_agent(self, agent: Any, state: Dict[str, Any]) -> bool:
        """将序列化状态恢复到 Agent。"""
        try:
            self._deserialize_runtime(agent, state.get("runtime", {}))
            self._deserialize_predictive_model(agent.predictive_model,
                                               state.get("predictive_model", {}))
            self._deserialize_memory(agent.memory_system,
                                     state.get("memory_system", {}))
            self._deserialize_skills(agent.skill_learner.library,
                                     state.get("skill_library", {}))
            self._deserialize_optimizer(agent.optimizer,
                                        state.get("optimizer", {}))
            self._deserialize_drive_engine(agent.drive_engine,
                                           state.get("drive_engine", {}))
            self._deserialize_goal_hierarchy(agent.goal_hierarchy,
                                             state.get("goal_hierarchy", {}))
            self._deserialize_metacognition(agent.metacognition,
                                            state.get("metacognition", {}))
            self.log.info("Agent state restored (step=%d)", agent.step_count)
            return True
        except Exception as e:
            self.log.error("Deserialize failed: %s", e)
            return False

    # ============================================================
    # 序列化方法
    # ============================================================

    def _serialize_config(self, config: SPEPAConfig) -> Dict[str, Any]:
        """序列化配置（只保存关键参数）。"""
        return {
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "obs_dim": config.obs_dim,
            "prediction_layers": config.prediction_layers,
            "seed": config.seed,
            "max_steps": config.max_steps,
        }

    def _serialize_runtime(self, agent: Any) -> Dict[str, Any]:
        """序列化运行时状态。"""
        return {
            "current_state": agent.current_state.tolist(),
            "step_count": agent.step_count,
            "online_cycles": agent.online_cycles,
            "offline_cycles": agent.offline_cycles,
            "exploration_ratio": agent.decision.exploration_ratio,
        }

    def _serialize_predictive_model(self, model: Any) -> Dict[str, Any]:
        """序列化预测模型。"""
        layers = []
        for layer in model.layers:
            layers.append({
                "mu": layer.mu.tolist(),
                "var": layer.var.tolist(),
                "W": layer.W.tolist(),
                "b": layer.b.tolist(),
            })
        return {"layers": layers}

    def _serialize_memory(self, memory_system: Any) -> Dict[str, Any]:
        """序列化记忆系统。"""
        def serialize_items(items):
            return [{
                "key": it.key.tolist(),
                "value": it.value if isinstance(it.value, (str, int, float, bool)) else str(it.value),
                "importance": it.importance,
                "timestamp": it.timestamp,
                "access_count": it.access_count,
                "consolidated": getattr(it, "consolidated", False),
            } for it in items]
        return {
            "fast_items": serialize_items(memory_system.fast.items),
            "slow_items": serialize_items(memory_system.slow.items),
        }

    def _serialize_skills(self, library: Any) -> Dict[str, Any]:
        """序列化技能库。"""
        skills = []
        for name, skill in library.skills.items():
            skills.append({
                "name": skill.name,
                "trigger_state": skill.trigger_state.tolist(),
                "action": skill.action.tolist(),
                "expected_effect": skill.expected_effect.tolist(),
                "success_count": skill.success_count,
                "failure_count": skill.failure_count,
                "use_count": skill.use_count,
                "last_used": skill.last_used,
                "consolidated": skill.consolidated,
            })
        return {"skills": skills}

    def _serialize_optimizer(self, optimizer: Any) -> Dict[str, Any]:
        """序列化优化器状态。"""
        return {
            "adaptive_lr": optimizer._adaptive_lr,
            "high_error_streak": optimizer._high_error_streak,
            "mode_history": optimizer.mode_history[-100:],  # 只保留最近100条
        }

    def _serialize_drive_engine(self, drive_engine: Any) -> Dict[str, Any]:
        """序列化驱力引擎。"""
        return {
            "homeostasis_values": drive_engine.homeostasis.values.tolist(),
            "homeostasis_target": drive_engine.homeostasis.target.tolist(),
            "last_rewards": dict(drive_engine.last_rewards),
        }

    def _serialize_goal_hierarchy(self, goal_hierarchy: Any) -> Dict[str, Any]:
        """序列化目标层次。"""
        goals = {}
        for level, goal_list in goal_hierarchy.goals.items():
            goals[str(level)] = [{
                "level": g.level,
                "name": g.name,
                "target": g.target.tolist(),
                "priority": g.priority,
                "rigid": g.rigid,
                "active": g.active,
                "progress": g.progress,
            } for g in goal_list]
        return {"goals": goals}

    def _serialize_metacognition(self, metacognition: Any) -> Dict[str, Any]:
        """序列化元认知。"""
        cal = metacognition.calibrator
        return {
            "current_confidence": metacognition.current_confidence,
            "meta_control_signal": metacognition.meta_control_signal,
            "calibration_offset": cal.calibration_offset,
            "confidences": list(cal.confidences),
            "accuracies": list(cal.accuracies),
        }

    # ============================================================
    # 反序列化方法
    # ============================================================

    def _deserialize_runtime(self, agent: Any, state: Dict[str, Any]) -> None:
        """恢复运行时状态。"""
        if "current_state" in state:
            agent.current_state = np.asarray(state["current_state"], dtype=np.float64)
        if "step_count" in state:
            agent.step_count = int(state["step_count"])
        if "online_cycles" in state:
            agent.online_cycles = int(state["online_cycles"])
        if "offline_cycles" in state:
            agent.offline_cycles = int(state["offline_cycles"])
        if "exploration_ratio" in state:
            agent.decision.exploration_ratio = float(state["exploration_ratio"])

    def _deserialize_predictive_model(self, model: Any, state: Dict[str, Any]) -> None:
        """恢复预测模型。"""
        layers = state.get("layers", [])
        for i, layer_state in enumerate(layers):
            if i < len(model.layers):
                layer = model.layers[i]
                layer.mu = np.asarray(layer_state["mu"], dtype=np.float64)
                layer.var = np.asarray(layer_state["var"], dtype=np.float64)
                layer.W = np.asarray(layer_state["W"], dtype=np.float64)
                layer.b = np.asarray(layer_state["b"], dtype=np.float64)

    def _deserialize_memory(self, memory_system: Any, state: Dict[str, Any]) -> None:
        """恢复记忆系统。"""
        from spepa.core.memory_system import MemoryItem
        # 恢复快速记忆
        memory_system.fast.items = []
        for item_state in state.get("fast_items", []):
            item = MemoryItem(
                key=np.asarray(item_state["key"], dtype=np.float64),
                value=item_state["value"],
                importance=item_state["importance"],
                timestamp=item_state["timestamp"],
            )
            item.access_count = item_state.get("access_count", 0)
            item.consolidated = item_state.get("consolidated", False)
            memory_system.fast.items.append(item)
        # 恢复慢速记忆
        memory_system.slow.items = []
        for item_state in state.get("slow_items", []):
            item = MemoryItem(
                key=np.asarray(item_state["key"], dtype=np.float64),
                value=item_state["value"],
                importance=item_state["importance"],
                timestamp=item_state["timestamp"],
            )
            item.access_count = item_state.get("access_count", 0)
            item.consolidated = item_state.get("consolidated", True)
            memory_system.slow.items.append(item)

    def _deserialize_skills(self, library: Any, state: Dict[str, Any]) -> None:
        """恢复技能库。"""
        from spepa.skills.skill import Skill
        library.skills = {}
        for skill_state in state.get("skills", []):
            skill = Skill(
                name=skill_state["name"],
                trigger_state=np.asarray(skill_state["trigger_state"], dtype=np.float64),
                action=np.asarray(skill_state["action"], dtype=np.float64),
                expected_effect=np.asarray(skill_state["expected_effect"], dtype=np.float64),
                success_count=skill_state.get("success_count", 0),
                failure_count=skill_state.get("failure_count", 0),
                use_count=skill_state.get("use_count", 0),
                last_used=skill_state.get("last_used", 0.0),
                consolidated=skill_state.get("consolidated", False),
            )
            library.skills[skill.name] = skill

    def _deserialize_optimizer(self, optimizer: Any, state: Dict[str, Any]) -> None:
        """恢复优化器状态。"""
        if "adaptive_lr" in state:
            optimizer._adaptive_lr = float(state["adaptive_lr"])
        if "high_error_streak" in state:
            optimizer._high_error_streak = int(state["high_error_streak"])
        if "mode_history" in state:
            optimizer.mode_history = list(state["mode_history"])

    def _deserialize_drive_engine(self, drive_engine: Any, state: Dict[str, Any]) -> None:
        """恢复驱力引擎。"""
        if "homeostasis_values" in state:
            drive_engine.homeostasis.values = np.asarray(
                state["homeostasis_values"], dtype=np.float64)
        if "homeostasis_target" in state:
            drive_engine.homeostasis.target = np.asarray(
                state["homeostasis_target"], dtype=np.float64)
        if "last_rewards" in state:
            drive_engine.last_rewards = dict(state["last_rewards"])

    def _deserialize_goal_hierarchy(self, goal_hierarchy: Any, state: Dict[str, Any]) -> None:
        """恢复目标层次。"""
        from spepa.core.goal_hierarchy import Goal
        goals_state = state.get("goals", {})
        # 清空现有目标
        for level in goal_hierarchy.goals:
            goal_hierarchy.goals[level] = []
        for level_str, goal_list in goals_state.items():
            level = int(level_str)
            for g_state in goal_list:
                goal = Goal(
                    level=g_state["level"],
                    name=g_state["name"],
                    target=np.asarray(g_state["target"], dtype=np.float64),
                    priority=g_state.get("priority", 1.0),
                    rigid=g_state.get("rigid", False),
                )
                goal.active = g_state.get("active", True)
                goal.progress = g_state.get("progress", 0.0)
                goal_hierarchy.goals[level].append(goal)

    def _deserialize_metacognition(self, metacognition: Any, state: Dict[str, Any]) -> None:
        """恢复元认知。"""
        from collections import deque
        if "current_confidence" in state:
            metacognition.current_confidence = float(state["current_confidence"])
        if "meta_control_signal" in state:
            metacognition.meta_control_signal = float(state["meta_control_signal"])
        cal = metacognition.calibrator
        if "calibration_offset" in state:
            cal.calibration_offset = float(state["calibration_offset"])
        if "confidences" in state:
            cal.confidences = deque(state["confidences"], maxlen=cal.confidences.maxlen)
        if "accuracies" in state:
            cal.accuracies = deque(state["accuracies"], maxlen=cal.accuracies.maxlen)
