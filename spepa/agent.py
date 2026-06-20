"""SPEPA 主 Agent：整合所有模块，实现在线 + 离线双闭环。

SPEPA = Self-driven Perpetual Evolutionary Prediction-Correction Agent

在线闭环：感知 -> 决策 -> 行动 -> 学习
离线闭环：低负载时触发离线复盘 -> 巩固记忆 -> 重构模型
"""
import time
import numpy as np
from typing import Dict, Any, Optional, Callable, List, TYPE_CHECKING

from spepa.utils.logger import LoggedMixin, get_logger
from spepa.utils.config import SPEPAConfig, get_default_config
from spepa.utils.metrics import MetricsTracker
from spepa.core.state_bus import StateBus
from spepa.core.predictive_model import PredictiveModel
from spepa.core.free_energy import FreeEnergyCalculator
from spepa.core.drive_engine import DriveEngine
from spepa.core.gap_detector import GapDetector
from spepa.core.goal_hierarchy import GoalHierarchy
from spepa.core.active_inference import ActiveInference
from spepa.core.metacognition import Metacognition
from spepa.core.memory_system import MemorySystem
from spepa.core.optimizer import Optimizer
from spepa.core.safety_guard import SafetyGuard
from spepa.core.self_model import SelfModel
from spepa.modules.perception import Perception
from spepa.modules.decision import Decision
from spepa.modules.action import Action
from spepa.modules.learning import Learning
from spepa.modules.offline_replay import OfflineReplay
from spepa.skills.skill_learner import SkillLearner
from spepa.multiagent.agent_pool import AgentPool
from spepa.multiagent.task_allocator import TaskAllocator, Coordinator
from spepa.multiagent.collaboration import CollaborationOrchestrator, CollaborationMode
from spepa.longtask.task_driver import LongTaskDriver
from spepa.perpetual.runner import PerpetualRunner
from spepa.persistence.serializer import StateSerializer
from spepa.persistence.checkpoint import CheckpointManager
from spepa.persistence.recovery import CrashRecovery

if TYPE_CHECKING:
    from spepa.environments import Environment
from spepa.resource.controller import ResourceController, ThrottleSignal
from spepa.resource.backpressure import Backpressure, BackpressureLevel
from spepa.resource.degradation import DegradationMatrix, DegradationLevel
from spepa.autonomous.driver import AutonomousDriver
from spepa.autonomous.goal_generator import GoalGenerator, AutonomousGoal, GoalSource
from spepa.core.self_repair import MetacognitiveSelfRepair
from spepa.events import EventBus, EventType, Event


class ResourceMonitor:
    """资源监控器：集成 PID 控制器的自适应资源调节。

    保留原有接口（cpu_usage/memory_usage/throttle_active/update/current_load/
    within_limits/should_throttle）以兼容现有代码，
    内部委托给 ResourceController 实现 PID 调节。
    """

    def __init__(self, config: SPEPAConfig):
        self.config = config
        self.controller = ResourceController(config)
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.rng = np.random.default_rng(config.seed)
        self.throttle_active = False  # 降级标志（兼容旧接口）
        # 当前节流信号
        self.current_signal: ThrottleSignal = ThrottleSignal()

    def update(self, load_factor: float = 0.5) -> Dict[str, float]:
        """根据负载因子更新资源使用，并通过 PID 控制器计算节流信号。"""
        # 模拟：基础负载 + 随机波动
        self.cpu_usage = float(np.clip(load_factor * 15 + self.rng.normal(0, 2), 0, 100))
        self.memory_usage = float(np.clip(load_factor * 12 + self.rng.normal(0, 1.5), 0, 100))
        # 通过 PID 控制器计算节流信号
        self.current_signal = self.controller.update(self.cpu_usage, self.memory_usage)
        # 兼容旧接口
        self.throttle_active = not self.within_limits()
        return {"cpu": self.cpu_usage, "memory": self.memory_usage}

    def current_load(self) -> float:
        """返回当前负载水平（0~1）。"""
        return float(max(self.cpu_usage, self.memory_usage) / 100.0)

    def within_limits(self) -> bool:
        """是否在资源约束内。"""
        return (self.cpu_usage <= self.config.resources.cpu_max_percent and
                self.memory_usage <= self.config.resources.memory_max_percent)

    def should_throttle(self) -> bool:
        """是否应降级运行（资源超限时减少计算量）。"""
        return self.throttle_active

    def get_throttle_signal(self) -> ThrottleSignal:
        """获取当前节流信号（新接口）。"""
        return self.current_signal


class SPEPAAgent(LoggedMixin):
    """SPEPA 主 Agent。

    整合所有核心模块与功能模块，实现在线 + 离线双闭环。
    """

    def __init__(self, config: Optional[SPEPAConfig] = None,
                 environment: Optional["Environment"] = None):
        self._set_log_name("spepa.agent")
        self.config = config or get_default_config()
        self.log.info("=== SPEPA Agent initializing ===")

        # 0. 外部环境注入（可选）
        self._external_env = environment
        if environment is not None:
            self.log.info("External environment injected: %s (obs_dim=%d, action_dim=%d)",
                          type(environment).__name__,
                          environment.observation_dim, environment.action_dim)

        # 1. 基础设施
        self.bus = StateBus()
        self.metrics = MetricsTracker()
        self.event_bus = EventBus()

        # 2. 核心模块
        self.predictive_model = PredictiveModel(self.config)
        self.fe_calculator = FreeEnergyCalculator(self.config)
        self.drive_engine = DriveEngine(self.config)
        self.gap_detector = GapDetector(self.config)
        self.goal_hierarchy = GoalHierarchy(self.config)
        self.metacognition = Metacognition(self.config)
        self.memory_system = MemorySystem(self.config)
        self.optimizer = Optimizer(self.config)
        self.safety_guard = SafetyGuard(self.config)
        self.self_model = SelfModel(self.config)
        self.active_inference = ActiveInference(self.config, self.fe_calculator)

        # 3. 功能模块
        self.perception = Perception(self.config, self.predictive_model,
                                     self.gap_detector, self.memory_system, self.bus)
        self.decision = Decision(self.config, self.active_inference, self.fe_calculator,
                                 self.goal_hierarchy, self.drive_engine,
                                 self.metacognition, self.safety_guard, self.bus)
        self.action = Action(self.config, self.safety_guard, self.bus)
        self.learning = Learning(self.config, self.predictive_model, self.optimizer,
                                 self.metacognition, self.self_model, self.bus)
        self.offline_replay = OfflineReplay(self.config, self.predictive_model,
                                            self.memory_system, self.optimizer,
                                            self.metacognition, self.self_model, self.bus)

        # 4. 资源监控与自调节
        self.resource_monitor = ResourceMonitor(self.config)
        self.backpressure = Backpressure(self.config)
        self.degradation_matrix = DegradationMatrix()

        # 5. 高级能力（应用场景：多Agent协作 + 技能增强 + 长任务驱动）
        self.skill_learner = SkillLearner(self.config)
        self.agent_pool = AgentPool(self.config)
        self.task_allocator = TaskAllocator(self.config, self.agent_pool)
        self.coordinator = Coordinator(self.config, self.agent_pool)
        self.collaboration_orchestrator = CollaborationOrchestrator(self.config)
        self.long_task_driver = LongTaskDriver(self.config)

        # 6. 自主驱动（目标生成 + 任务队列 + 自主执行）
        self.autonomous_driver = AutonomousDriver(self.config)

        # 7. 元认知自监控与自我修复
        self.self_repair_system = MetacognitiveSelfRepair(self.config)

        # 6. 运行时状态
        self.current_state = np.ones(self.config.state_dim) * 0.5
        self.step_count = 0
        self.online_cycles = 0
        self.offline_cycles = 0
        self.running = False

        # 7. 内部预测函数（用于主动推理）
        self._rng = np.random.default_rng(self.config.seed)

        self.log.info("=== SPEPA Agent initialized (state_dim=%d action_dim=%d) ===",
                      self.config.state_dim, self.config.action_dim)

    def _model_predict_fn(self, state: np.ndarray, action: np.ndarray):
        """内部世界模型：预测下一状态与方差。"""
        state = np.asarray(state, dtype=np.float64).flatten()
        action = np.asarray(action, dtype=np.float64).flatten()
        if state.shape[0] != self.config.state_dim:
            state = np.resize(state, self.config.state_dim)
        if action.shape[0] != self.config.state_dim:
            action = np.resize(action, self.config.state_dim)
        # 简单线性动力学 + 模型预测
        model_pred = self.predictive_model.predict(state)
        next_state = 0.7 * state + 0.2 * action + 0.1 * model_pred
        next_state = np.clip(next_state, 0.0, 1.0)
        var = np.ones_like(next_state) * 0.1
        return next_state, var

    def _env_step_fn(self, action: np.ndarray) -> np.ndarray:
        """环境步进函数：有外部环境则用外部环境，否则用默认动力学。"""
        if self._external_env is not None:
            # 使用注入的外部环境
            obs, _reward, _done, _info = self._external_env.step(action)
            return obs
        # 默认环境：简单线性动力学
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape[0] != self.config.state_dim:
            action = np.resize(action, self.config.state_dim)
        # 环境动力学：状态受动作影响 + 噪声
        noise = self._rng.normal(0, 0.05, self.config.state_dim)
        next_state = self.current_state + 0.3 * action + noise
        next_state = np.clip(next_state, 0.0, 1.0)
        return next_state

    def online_cycle(self, observation: Optional[np.ndarray] = None,
                     env_step_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Dict[str, Any]:
        """在线闭环：感知 -> 决策 -> 行动 -> 学习。"""
        self.log.info("--- Online cycle #%d start ---", self.online_cycles)
        cycle_start = time.time()

        # 安全检查：确保 current_state 数值有限
        if not np.all(np.isfinite(self.current_state)):
            self.log.warning("current_state contained non-finite values, resetting to default")
            self.current_state = np.ones(self.config.state_dim) * 0.5

        # 生成或使用观测
        if observation is None:
            observation = self.current_state.copy()
        observation = np.asarray(observation, dtype=np.float64).flatten()
        if observation.shape[0] != self.config.obs_dim:
            observation = np.resize(observation, self.config.obs_dim)
        # 清洗 NaN/Inf，确保所有下游模块收到干净的观测
        if not np.all(np.isfinite(observation)):
            bad = ~np.isfinite(observation)
            observation[bad] = 0.0
            self.log.warning("Observation contained non-finite values, sanitized")

        # 1. 感知
        perc_result = self.perception.perceive(observation)
        gap_kl = perc_result["error"]
        self.metrics.record("prediction_error", gap_kl)

        # 2. 计算信心
        uncertainty = float(np.mean(self.predictive_model.layers[-1].var))
        confidence = self.metacognition.compute_confidence(gap_kl, uncertainty)
        self.metrics.record("confidence", confidence)

        # 2.5 获取当前降级参数（基于上一周期的资源测量）
        deg_params = self.degradation_matrix.get_current_params()

        # 3. 决策（根据降级级别调整策略数量）
        n_policies = deg_params.n_policies
        if self.resource_monitor.should_throttle():
            # 兼容旧逻辑：资源超限时进一步减少
            n_policies = min(n_policies, 4)
            self.log.info("Resource throttle active, reducing policies to %d", n_policies)
        dec_result = self.decision.decide(self.current_state,
                                          self.gap_detector.last_gap,
                                          self._model_predict_fn,
                                          n_policies=n_policies)
        self.metrics.record("expected_free_energy", dec_result["decision"]["expected_free_energy"])
        self.metrics.record("exploration_ratio", self.decision.exploration_ratio)

        # 4. 行动
        env_fn = env_step_fn if env_step_fn is not None else self._env_step_fn
        act_result = self.action.execute(dec_result["decision"]["action"], env_fn)
        if act_result["effect"] is not None:
            new_state = np.asarray(act_result["effect"], dtype=np.float64).flatten()
            # 对齐到 state_dim
            if new_state.shape[0] != self.config.state_dim:
                new_state = np.resize(new_state, self.config.state_dim)
            # 再次确保数值有限
            if not np.all(np.isfinite(new_state)):
                bad = ~np.isfinite(new_state)
                new_state[bad] = 0.0
                self.log.warning("New state contained non-finite values, sanitized")
            self.current_state = new_state

        # 5. 学习（根据降级级别决定是否学习）
        learn_result = {"post_error": gap_kl, "skipped": False}
        if deg_params.learning_enabled and (
                self.step_count % deg_params.learning_interval == 0):
            learn_result = self.learning.learn(observation, gap_kl, confidence)
        else:
            learn_result["skipped"] = True
        self.metrics.record("post_error", learn_result["post_error"])

        # 6. 驱力更新（reward 计算是纯函数，不修改状态）
        goal_target = np.asarray(
            dec_result["arbitration"].get("dominant_target") or self.current_state,
            dtype=np.float64).flatten()
        if goal_target.shape[0] != self.config.state_dim:
            goal_target = np.resize(goal_target, self.config.state_dim)
        achieved = 1.0 - float(np.tanh(gap_kl))
        action_for_reward = np.asarray(dec_result["action"], dtype=np.float64).flatten()
        total_reward = self.drive_engine.total_reward(
            self.current_state,
            action_for_reward,
            goal_target, achieved,
        )
        self.metrics.record("total_reward", total_reward)

        # 6.5 显式更新内稳态（基于执行的动作，因果清晰）
        # 这是对 reward 计算的解耦：reward 只评估，不修改；内稳态由动作显式驱动
        self.drive_engine.update_homeostasis(action_for_reward)
        self.metrics.record("homeostasis_deviation", self.drive_engine.homeostatic_drive())

        # 7. 记忆步进（始终调用以保持计数器准确，巩固由 consolidation_interval 控制）
        self.memory_system.step()

        # 7.5 技能学习：记录经验并尝试提取技能（根据降级级别决定是否执行）
        if deg_params.skill_learning and act_result.get("effect") is not None:
            effect = np.asarray(act_result["effect"], dtype=np.float64).flatten()
            if effect.shape[0] != self.config.state_dim:
                effect = np.resize(effect, self.config.state_dim)
            # 判断成功：误差下降或 reward 为正
            success = (gap_kl < 0.5) or (total_reward > 0)
            self.skill_learner.record_experience(
                state=self.current_state,
                action=action_for_reward,
                effect=effect,
                success=success,
                timestamp=float(self.step_count),
            )
            # 每 20 步尝试提取技能
            if self.online_cycles % 20 == 0:
                new_skills = self.skill_learner.extract_skills()
                if new_skills:
                    self.metrics.record("new_skills", len(new_skills))
                    self.log.info("Extracted %d new skills", len(new_skills))
            # 每 50 步固化技能
            if self.online_cycles % 50 == 0:
                consolidated = self.skill_learner.consolidate()
                if consolidated > 0:
                    self.log.info("Consolidated %d skills", consolidated)
            self.metrics.record("skill_count", len(self.skill_learner.library.skills))

        # 8. 资源监控与降级更新
        load = 0.5 + 0.3 * float(np.tanh(gap_kl))
        resources = self.resource_monitor.update(load)
        self.metrics.record("cpu_usage", resources["cpu"])
        self.metrics.record("memory_usage", resources["memory"])
        # 根据节流信号更新降级矩阵（供下一周期使用）
        throttle_signal = self.resource_monitor.get_throttle_signal()
        self.degradation_matrix.update_from_throttle(throttle_signal.throttle)
        # 背压：高误差时向决策组件发背压信号
        if gap_kl > 2.0:
            self.backpressure.signal("perception", BackpressureLevel.HIGH,
                                     reason="high_prediction_error")
        elif gap_kl > 1.0:
            self.backpressure.signal("perception", BackpressureLevel.MEDIUM,
                                     reason="moderate_prediction_error")
        else:
            self.backpressure.clear_signal("perception")
        self.metrics.record("degradation_level", self.degradation_matrix.current_level.value)
        self.metrics.record("backpressure_level", self.backpressure.get_pressure().value)

        self.online_cycles += 1
        self.step_count += 1
        cycle_time = time.time() - cycle_start
        self.metrics.record("cycle_time", cycle_time)

        result = {
            "cycle": self.online_cycles,
            "perception": perc_result,
            "decision": dec_result,
            "action": act_result,
            "learning": learn_result,
            "reward": total_reward,
            "resources": resources,
            "confidence": confidence,
            "cycle_time": cycle_time,
        }
        self.bus.publish("online_cycle", result)
        self.log.info("--- Online cycle #%d done (err=%.4f reward=%.4f conf=%.4f time=%.3fs) ---",
                      self.online_cycles, gap_kl, total_reward, confidence, cycle_time)
        # 发布事件
        self.event_bus.publish(Event(EventType.STEP_END, {
            "step": self.step_count,
            "cycle": self.online_cycles,
            "error": float(gap_kl),
            "reward": float(total_reward),
            "confidence": float(confidence),
            "cycle_time": cycle_time,
        }))
        return result

    def offline_cycle(self) -> Dict[str, Any]:
        """离线闭环：复盘 -> 巩固 -> 重构。"""
        self.log.info("=== Offline cycle #%d start ===", self.offline_cycles)
        cycle_start = time.time()

        load = self.resource_monitor.current_load()
        if not self.offline_replay.should_trigger(load):
            self.log.info("Offline cycle skipped: load=%.4f too high", load)
            return {"skipped": True, "load": load}

        # 离线复盘
        replay_result = self.offline_replay.replay()
        self.metrics.record("offline_replayed", replay_result.get("replayed", 0))
        self.metrics.record("offline_pre_error", replay_result.get("avg_pre_error", 0.0))
        self.metrics.record("offline_post_error", replay_result.get("avg_post_error", 0.0))

        # 自我指涉检查
        self_check = self.self_model.self_referential_check()
        self.metrics.record("self_consistency", self_check["consistency"])

        self.offline_cycles += 1
        cycle_time = time.time() - cycle_start
        self.metrics.record("offline_cycle_time", cycle_time)

        result = {
            "cycle": self.offline_cycles,
            "replay": replay_result,
            "self_check": self_check,
            "cycle_time": cycle_time,
        }
        self.bus.publish("offline_cycle", result)
        self.log.info("=== Offline cycle #%d done (replayed=%d err %.4f->%.4f time=%.3fs) ===",
                      self.offline_cycles, replay_result.get("replayed", 0),
                      replay_result.get("avg_pre_error", 0.0),
                      replay_result.get("avg_post_error", 0.0), cycle_time)
        return result

    def step(self, observation: Optional[np.ndarray] = None,
             env_step_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Dict[str, Any]:
        """单步：在线闭环 + 条件触发离线闭环。"""
        online_result = self.online_cycle(observation, env_step_fn)

        offline_result = None
        load = self.resource_monitor.current_load()
        if self.offline_replay.should_trigger(load):
            offline_result = self.offline_cycle()

        return {"online": online_result, "offline": offline_result}

    def run(self, n_steps: Optional[int] = None,
            env_step_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Dict[str, Any]:
        """运行 n_steps 步。"""
        n_steps = n_steps or self.config.max_steps
        self.running = True
        self.log.info(">>> SPEPA Agent run start: %d steps <<<", n_steps)
        start_time = time.time()

        for i in range(n_steps):
            self.step(env_step_fn=env_step_fn)
            # 资源约束检查
            if not self.resource_monitor.within_limits():
                self.log.warning("Resource limit exceeded at step %d (cpu=%.1f%% mem=%.1f%%)",
                                 i, self.resource_monitor.cpu_usage,
                                 self.resource_monitor.memory_usage)

        total_time = time.time() - start_time
        self.running = False
        summary = {
            "total_steps": n_steps,
            "online_cycles": self.online_cycles,
            "offline_cycles": self.offline_cycles,
            "total_time": total_time,
            "metrics": self.metrics.summary(),
        }
        self.log.info(">>> SPEPA Agent run done: %d online / %d offline cycles in %.2fs <<<",
                      self.online_cycles, self.offline_cycles, total_time)
        self.event_bus.publish(Event(EventType.CYCLE_COMPLETE, summary))
        return summary

    def get_state(self) -> Dict[str, Any]:
        """返回系统完整状态。"""
        return {
            "step_count": self.step_count,
            "online_cycles": self.online_cycles,
            "offline_cycles": self.offline_cycles,
            "current_state": self.current_state.tolist(),
            "exploration_ratio": self.decision.exploration_ratio,
            "resources": {
                "cpu": self.resource_monitor.cpu_usage,
                "memory": self.resource_monitor.memory_usage,
                "within_limits": self.resource_monitor.within_limits(),
            },
            "resource_controller": self.resource_monitor.controller.get_state(),
            "backpressure": self.backpressure.get_state(),
            "degradation": self.degradation_matrix.get_state(),
            "predictive_model": self.predictive_model.get_state(),
            "drive_engine": self.drive_engine.get_state(),
            "gap_detector": self.gap_detector.get_state(),
            "goal_hierarchy": self.goal_hierarchy.get_state(),
            "metacognition": self.metacognition.get_state(),
            "memory_system": self.memory_system.get_state(),
            "optimizer": self.optimizer.get_state(),
            "safety_guard": self.safety_guard.get_state(),
            "self_model": self.self_model.get_state(),
            "skill_learner": self.skill_learner.get_state(),
            "agent_pool": self.agent_pool.get_state(),
            "long_task_driver": self.long_task_driver.get_state(),
            "autonomous_driver": self.autonomous_driver.get_state(),
            "collaboration_orchestrator": self.collaboration_orchestrator.get_state(),
            "self_repair_system": self.self_repair_system.get_state(),
            "event_bus": self.event_bus.get_state(),
        }

    # ============================================================
    # 自主驱动API：目标生成 + 任务队列 + 自主执行
    # ============================================================

    def autonomous_step(self, env_step_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """自主驱动一步：生成目标 → 入队 → 执行任务。

        这是"自驱动"的核心入口。调用此方法后，Agent 会：
        1. 定期自主生成目标（从内驱力、好奇心、技能缺口等）
        2. 将目标入队任务队列
        3. 从队列取出最高优先级任务执行
        4. 评估任务完成情况

        参数：
        - env_step_fn: 环境步进函数

        返回：执行结果字典
        """
        return self.autonomous_driver.tick(self, self.step_count, env_step_fn)

    def run_autonomous(self, n_steps: int = 100,
                       env_step_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """自主运行 N 步。

        参数：
        - n_steps: 运行步数
        - env_step_fn: 环境步进函数
        """
        self.log.info(">>> SPEPA Agent autonomous run: %d steps <<<", n_steps)
        start_time = time.time()
        results = []
        for i in range(n_steps):
            result = self.autonomous_step(env_step_fn)
            results.append(result)
        total_time = time.time() - start_time
        summary = {
            "total_steps": n_steps,
            "total_time": total_time,
            "autonomous_state": self.autonomous_driver.get_state(),
            "goals_generated": self.autonomous_driver.goals_generated,
            "goals_achieved": self.autonomous_driver.goals_achieved,
            "goals_failed": self.autonomous_driver.goals_failed,
            "achievement_rate": (
                self.autonomous_driver.goals_achieved /
                max(1, self.autonomous_driver.goals_achieved +
                    self.autonomous_driver.goals_failed)
            ),
        }
        self.log.info(">>> Autonomous run done: %d goals generated, %d achieved <<<",
                      summary["goals_generated"], summary["goals_achieved"])
        return summary

    def add_autonomous_goal(self, target_state: np.ndarray,
                            priority: float = 5.0,
                            description: str = "external") -> bool:
        """添加外部自主目标。"""
        return self.autonomous_driver.add_external_goal(
            target_state, priority, description
        )

    # ============================================================
    # 元认知自监控与自我修复API
    # ============================================================

    def self_repair_step(self) -> Dict[str, Any]:
        """执行一轮元认知自监控与自我修复。

        监控→诊断→修复闭环：
        1. 记录当前性能指标
        2. 定期分析趋势和异常
        3. 如有异常，诊断根因
        4. 根据诊断执行分级修复

        返回：执行结果字典
        """
        return self.self_repair_system.step(self, self.step_count)

    def get_health_score(self) -> float:
        """获取当前健康评分（0~1）。"""
        return self.self_repair_system.monitor.get_health_score()

    # ============================================================
    # 事件回调API
    # ============================================================

    def on(self, event_type: EventType, callback: Callable) -> None:
        """注册事件回调。

        参数：
        - event_type: 事件类型（EventType.STEP_END 等）
        - callback: 回调函数，接收 Event 对象

        示例：
            agent.on(EventType.STEP_END, lambda e: print(f"Step {e.data['step']}"))
        """
        self.event_bus.subscribe(event_type, callback)

    def generate_report(self, format: str = "json",
                        filepath: Optional[str] = None,
                        run_summary: Optional[Dict] = None) -> str:
        """生成运行报告。

        参数：
        - format: 格式 ("json"/"html"/"csv")
        - filepath: 保存路径（None 则返回字符串）
        - run_summary: run() 返回的摘要

        返回：报告字符串（filepath 为 None 时）或保存路径

        示例：
            agent.run(n_steps=100)
            agent.generate_report("html", "report.html")
        """
        from spepa.report import ReportGenerator
        gen = ReportGenerator()
        if filepath:
            return gen.save_report(self, filepath, format, run_summary)
        if format.lower() == "json":
            return gen.generate_json_report(self, run_summary)
        elif format.lower() == "html":
            return gen.generate_html_report(self, run_summary)
        elif format.lower() == "csv":
            return gen.generate_csv_report(self, run_summary)
        else:
            raise ValueError(f"不支持的格式: {format}。可用: json, html, csv")

    # ============================================================
    # 高级能力API：多Agent协作 + 技能增强 + 长任务驱动
    # ============================================================

    def delegate_task(self, target_state: np.ndarray,
                      env_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """委派任务给子Agent（多Agent协作）。

        将目标状态分解为子任务，派生子Agent并行处理，汇总结果。
        """
        self.log.info("=== Delegating task to sub-agents ===")
        target_state = np.asarray(target_state, dtype=np.float64).flatten()
        if target_state.shape[0] != self.config.state_dim:
            target_state = np.resize(target_state, self.config.state_dim)
        # 分解并分配
        assignments = self.task_allocator.decompose_and_allocate(
            self.current_state, target_state, description="delegated_task"
        )
        # 协调执行
        result = self.coordinator.coordinate(env_fn)
        # 合并结果
        merged_state = self.coordinator.merge_results(
            result.get("results", []), self.current_state
        )
        self.log.info("Task delegated: %d/%d succeeded, merged state norm=%.4f",
                      result["success_count"], result["total_count"],
                      float(np.linalg.norm(merged_state)))
        return {
            "coordination": result,
            "merged_state": merged_state.tolist(),
            "pool_state": self.agent_pool.get_state(),
        }

    def collaborate(self, target_state: np.ndarray,
                    mode: CollaborationMode = CollaborationMode.PARALLEL,
                    env_fn: Optional[Callable] = None,
                    n_competitors: int = 3) -> Dict[str, Any]:
        """多Agent协作完成任务（深化版多Agent协作）。

        支持三种协作模式：
        - PARALLEL: 按维度分解，子Agent并行处理
        - SEQUENTIAL: 按阶段分解，串行依赖执行
        - COMPETITIVE: 多Agent竞争，择优选取

        协作完成后，子Agent经验自动回传到主Agent技能库。

        参数：
        - target_state: 目标状态
        - mode: 协作模式
        - env_fn: 环境步进函数
        - n_competitors: 竞争模式下的竞争者数量

        返回：协作结果字典
        """
        self.log.info("=== Collaborating (mode=%s) ===", mode.value)
        result = self.collaboration_orchestrator.collaborate(
            initial_state=self.current_state,
            target_state=target_state,
            mode=mode,
            env_fn=env_fn,
            skill_library=self.skill_learner.library,
            n_competitors=n_competitors,
        )
        # 更新当前状态为协作结果
        final_state = result.get("final_state")
        if final_state is not None:
            final_state = np.asarray(final_state, dtype=np.float64).flatten()
            if final_state.shape[0] == self.config.state_dim and np.all(np.isfinite(final_state)):
                self.current_state = final_state
        self.log.info("Collaboration done: success_rate=%.2f, skills_feedback=%d",
                      result.get("success_rate", 0.0),
                      result.get("skills_feedback", 0))
        return result

    def learn_skill_from_experience(self) -> List[Dict[str, Any]]:
        """从经验中学习技能（技能增强）。"""
        new_skills = self.skill_learner.extract_skills()
        consolidated = self.skill_learner.consolidate()
        self.log.info("Skill learning: %d new, %d consolidated",
                      len(new_skills), consolidated)
        return [s.get_state() for s in new_skills]

    def apply_learned_skill(self, state: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """应用已学技能（技能复用）。"""
        state = state if state is not None else self.current_state
        skill = self.skill_learner.apply_skill(state)
        if skill is None:
            return None
        return {
            "skill_name": skill.name,
            "action": skill.action.tolist(),
            "expected_effect": skill.expected_effect.tolist(),
            "success_rate": skill.success_rate,
            "consolidated": skill.consolidated,
        }

    def run_long_task(self, target_state: np.ndarray,
                      n_stages: int = 3,
                      env_fn: Optional[Callable] = None,
                      max_total_steps: int = 1000) -> Dict[str, Any]:
        """自主驱动长任务（长任务驱动）。

        将长任务分解为多阶段子任务，自主推进执行，支持中断恢复。
        """
        self.log.info("=== Starting long task (n_stages=%d) ===", n_stages)
        target_state = np.asarray(target_state, dtype=np.float64).flatten()
        if target_state.shape[0] != self.config.state_dim:
            target_state = np.resize(target_state, self.config.state_dim)
        # 规划
        self.long_task_driver.plan(
            initial_state=self.current_state,
            final_target=target_state,
            description="agent_long_task",
            n_stages=n_stages,
        )
        # 自主驱动执行
        result = self.long_task_driver.run(env_fn=env_fn, max_total_steps=max_total_steps)
        # 更新当前状态
        if result.get("final_state"):
            self.current_state = np.asarray(result["final_state"], dtype=np.float64)
        self.log.info("Long task finished: progress=%.2f, steps=%d",
                      result["progress"], result["total_steps"])
        return result

    def save_long_task_checkpoint(self) -> Dict[str, Any]:
        """保存长任务检查点（中断恢复）。"""
        return self.long_task_driver.save_checkpoint()

    def resume_long_task(self, checkpoint: Dict[str, Any],
                         env_fn: Optional[Callable] = None,
                         max_total_steps: int = 1000) -> Dict[str, Any]:
        """从检查点恢复长任务（中断恢复）。"""
        success = self.long_task_driver.restore_checkpoint(checkpoint)
        if not success:
            return {"resumed": False, "error": "checkpoint restore failed"}
        result = self.long_task_driver.run(env_fn=env_fn, max_total_steps=max_total_steps)
        if result.get("final_state"):
            self.current_state = np.asarray(result["final_state"], dtype=np.float64)
        return result

    # ============================================================
    # 永续运行API：永远运行不停止
    # ============================================================

    def run_perpetual(self, max_cycles: Optional[int] = None,
                      env_step_fn: Optional[Callable] = None,
                      checkpoint_dir: Optional[str] = None,
                      auto_recover: bool = True) -> Dict[str, Any]:
        """永续运行：自主循环驱动，异常自恢复，永不停止。

        参数：
        - max_cycles: 最大周期数（None=无限运行）
        - env_step_fn: 环境步进函数
        - checkpoint_dir: 检查点目录（None=不持久化）
        - auto_recover: 是否自动从检查点恢复
        """
        # 创建工厂函数（用于硬重启时重建Agent）
        config = self.config
        def agent_factory():
            return SPEPAAgent(config)
        runner = PerpetualRunner(self.config, agent_factory)
        runner.agent = self  # 复用当前Agent
        return runner.start(max_cycles=max_cycles, env_step_fn=env_step_fn,
                            checkpoint_dir=checkpoint_dir, auto_recover=auto_recover)

    # ============================================================
    # 持久化API：状态保存与恢复
    # ============================================================

    def save_state(self, checkpoint_dir: str) -> Optional[Dict[str, Any]]:
        """保存当前状态到检查点目录。

        参数：
        - checkpoint_dir: 检查点目录

        返回：保存信息字典（失败返回None）
        """
        mgr = CheckpointManager(self.config, checkpoint_dir)
        info = mgr.save(self)
        if info is None:
            return None
        return {
            "version": info.version,
            "step_count": info.step_count,
            "size_bytes": info.size_bytes,
            "timestamp": info.timestamp,
        }

    def load_state(self, checkpoint_dir: str,
                   version: Optional[int] = None) -> bool:
        """从检查点目录恢复状态。

        参数：
        - checkpoint_dir: 检查点目录
        - version: 指定版本号（None=最新）

        返回：是否恢复成功
        """
        mgr = CheckpointManager(self.config, checkpoint_dir)
        if version is None:
            return mgr.load_latest(self)
        return mgr.load(self, version=version)

    def create_checkpoint_manager(self, checkpoint_dir: str) -> CheckpointManager:
        """创建检查点管理器（用于长期持有）。"""
        return CheckpointManager(self.config, checkpoint_dir)

    def create_recovery(self, checkpoint_dir: str) -> CrashRecovery:
        """创建崩溃恢复器（用于长期持有）。"""
        return CrashRecovery(self.config, checkpoint_dir)

    def serialize_state(self) -> Dict[str, Any]:
        """序列化当前状态为字典（不写入磁盘）。"""
        serializer = StateSerializer(self.config)
        return serializer.serialize_agent(self)

    def deserialize_state(self, state: Dict[str, Any]) -> bool:
        """从字典反序列化状态（不读取磁盘）。"""
        serializer = StateSerializer(self.config)
        return serializer.deserialize_agent(self, state)

    def reset(self) -> None:
        """重置 Agent 到初始状态。"""
        self.log.info("Resetting SPEPA Agent...")
        self.current_state = np.ones(self.config.state_dim) * 0.5
        self.step_count = 0
        self.online_cycles = 0
        self.offline_cycles = 0
        self.metrics.reset()
        self.bus.clear()
        # 重置内稳态到目标值
        self.drive_engine.reset_homeostasis()
        # 重置资源调节
        self.resource_monitor.controller.reset()
        self.backpressure.reset()
        self.degradation_matrix.reset()
        # 重置自主驱动
        self.autonomous_driver.reset()
        # 重置协作编排器
        self.collaboration_orchestrator.reset()
        # 重置自监控与修复系统
        self.self_repair_system.reset()
        # 重置事件总线
        self.event_bus.reset()
        self.log.info("SPEPA Agent reset complete")
