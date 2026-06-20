"""SPEPA 协作编排器：多协作模式的任务编排。

设计原则（第一性原理）：
- 模式适配原理：不同任务适合不同协作模式
  - 可分解任务 → 并行（PARALLEL）
  - 有依赖任务 → 串行（SEQUENTIAL）
  - 不确定任务 → 竞争择优（COMPETITIVE）
- 经验回传原理：协作完成后，子Agent经验回传主Agent
- 通信原理：协作过程中通过 MessageBus 共享进展

协作模式：
1. PARALLEL: 按维度分解，子Agent并行处理，合并结果（现有模式）
2. SEQUENTIAL: 按阶段分解，前一个的输出是后一个的输入
3. COMPETITIVE: 多个子Agent同时尝试同一任务，选最优结果
"""
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.multiagent.agent_pool import AgentPool, SubAgent, SubTask
from spepa.multiagent.task_allocator import TaskAllocator, Coordinator
from spepa.multiagent.communication import MessageBus, Message, MessageType
from spepa.multiagent.experience import ExperienceFeedback


class CollaborationMode(Enum):
    """协作模式。"""
    PARALLEL = "parallel"        # 并行分解
    SEQUENTIAL = "sequential"    # 串行依赖
    COMPETITIVE = "competitive"  # 竞争择优


class CollaborationOrchestrator(LoggedMixin):
    """协作编排器：根据任务特征选择协作模式并执行。

    集成 AgentPool、TaskAllocator、Coordinator、MessageBus、ExperienceFeedback，
    提供统一的多Agent协作接口。
    """

    def __init__(self, config: SPEPAConfig):
        self._set_log_name("spepa.multiagent.collaboration")
        self.config = config
        self.dim = config.state_dim
        # 组件
        self.pool = AgentPool(config)
        self.allocator = TaskAllocator(config, self.pool)
        self.coordinator = Coordinator(config, self.pool)
        self.message_bus = MessageBus()
        self.experience_feedback = ExperienceFeedback(config)
        # 统计
        self.total_collaborations = 0
        self.mode_stats: Dict[str, int] = {m.value: 0 for m in CollaborationMode}
        self.log.info("CollaborationOrchestrator initialized")

    def collaborate(self, initial_state: np.ndarray,
                    target_state: np.ndarray,
                    mode: CollaborationMode = CollaborationMode.PARALLEL,
                    env_fn: Optional[Callable] = None,
                    skill_library=None,
                    n_competitors: int = 3) -> Dict[str, Any]:
        """执行一次多Agent协作。

        参数：
        - initial_state: 初始状态
        - target_state: 目标状态
        - mode: 协作模式
        - env_fn: 环境步进函数
        - skill_library: 主Agent技能库（用于经验回传）
        - n_competitors: 竞争模式下的竞争者数量

        返回：协作结果字典
        """
        initial_state = np.asarray(initial_state, dtype=np.float64).flatten()
        target_state = np.asarray(target_state, dtype=np.float64).flatten()
        if initial_state.shape[0] != self.dim:
            initial_state = np.resize(initial_state, self.dim)
        if target_state.shape[0] != self.dim:
            target_state = np.resize(target_state, self.dim)
        # 清洗
        initial_state = np.nan_to_num(initial_state, nan=0.5, posinf=1.0, neginf=0.0)
        target_state = np.nan_to_num(target_state, nan=0.5, posinf=1.0, neginf=0.0)

        self.total_collaborations += 1
        self.mode_stats[mode.value] += 1
        self.log.info("Collaboration #%d start (mode=%s, dist=%.3f)",
                      self.total_collaborations, mode.value,
                      float(np.linalg.norm(target_state - initial_state)))

        # 注册主Agent到通信总线
        self.message_bus.register("main_agent")

        # 按模式执行
        if mode == CollaborationMode.PARALLEL:
            result = self._run_parallel(initial_state, target_state, env_fn)
        elif mode == CollaborationMode.SEQUENTIAL:
            result = self._run_sequential(initial_state, target_state, env_fn)
        elif mode == CollaborationMode.COMPETITIVE:
            result = self._run_competitive(
                initial_state, target_state, env_fn, n_competitors)
        else:
            result = self._run_parallel(initial_state, target_state, env_fn)

        # 经验回传
        if skill_library is not None:
            skills_feedback = self._feedback_experience(skill_library)
            result["skills_feedback"] = skills_feedback

        # 清理通信总线
        self.message_bus.unregister("main_agent")

        result["mode"] = mode.value
        result["collaboration_id"] = self.total_collaborations
        self.log.info("Collaboration #%d done (mode=%s, success_rate=%.2f)",
                      self.total_collaborations, mode.value,
                      result.get("success_rate", 0.0))
        return result

    def _run_parallel(self, initial_state: np.ndarray,
                      target_state: np.ndarray,
                      env_fn: Optional[Callable]) -> Dict[str, Any]:
        """并行协作：按维度分解，并行执行，合并结果。"""
        # 分解并分配
        assignments = self.allocator.decompose_and_allocate(
            initial_state, target_state, description="parallel_task")
        # 注册子Agent到通信总线
        for task_id, agent in assignments.items():
            if agent is not None:
                self.message_bus.register(agent.agent_id)
        # 协调执行
        summary = self.coordinator.coordinate(env_fn)
        # 合并结果
        merged = self.coordinator.merge_results(
            summary["results"], initial_state)
        # 广播结果
        self.message_bus.publish(Message(
            sender_id="main_agent",
            msg_type=MessageType.RESULT_REPORT,
            content={"merged_state": merged.tolist(),
                     "success_rate": summary["success_rate"]},
        ))
        return {
            "final_state": merged,
            "success_rate": summary["success_rate"],
            "success_count": summary["success_count"],
            "total_count": summary["total_count"],
            "avg_error": summary["avg_error"],
            "total_steps": summary["total_steps"],
            "results": summary["results"],
        }

    def _run_sequential(self, initial_state: np.ndarray,
                        target_state: np.ndarray,
                        env_fn: Optional[Callable]) -> Dict[str, Any]:
        """串行协作：按阶段分解，前一个的输出是后一个的输入。

        策略：将总距离分成多个阶段，每个阶段目标是中间点。
        """
        total_dist = float(np.linalg.norm(target_state - initial_state))
        n_stages = min(4, max(2, int(total_dist / 0.3) + 1))
        if total_dist < 0.1:
            n_stages = 1
        results = []
        current_state = initial_state.copy()
        total_steps = 0
        success_count = 0
        for stage in range(n_stages):
            # 阶段目标 = 线性插值的中间点
            progress = (stage + 1) / n_stages
            stage_target = initial_state + progress * (target_state - initial_state)
            # 创建子任务
            task = SubTask(
                task_id=f"seq_stage_{stage}",
                description=f"Sequential stage {stage}/{n_stages}",
                initial_state=current_state.copy(),
                target_state=stage_target,
                priority=1.0,
                max_steps=20,
            )
            agent = self.pool.spawn(task)
            if agent is None:
                self.log.warning("Pool full at stage %d, skipping", stage)
                continue
            self.message_bus.register(agent.agent_id)
            # 执行该阶段
            run_result = agent.run(env_fn)
            results.append(run_result)
            total_steps += run_result.get("steps", 0)
            if run_result.get("success", False):
                success_count += 1
            # 更新当前状态为该阶段的最终状态
            final_state = run_result.get("final_state")
            if final_state is not None:
                current_state = np.asarray(final_state, dtype=np.float64).flatten()
            # 发送进度更新
            self.message_bus.publish(Message(
                sender_id=agent.agent_id,
                msg_type=MessageType.PROGRESS_UPDATE,
                content={"stage": stage, "total_stages": n_stages,
                         "current_error": run_result.get("final_error", 0.0)},
            ))
        return {
            "final_state": current_state,
            "success_rate": success_count / n_stages if n_stages > 0 else 0.0,
            "success_count": success_count,
            "total_count": n_stages,
            "avg_error": float(np.mean([r.get("final_error", 0.0) for r in results]))
            if results else 0.0,
            "total_steps": total_steps,
            "results": results,
        }

    def _run_competitive(self, initial_state: np.ndarray,
                         target_state: np.ndarray,
                         env_fn: Optional[Callable],
                         n_competitors: int) -> Dict[str, Any]:
        """竞争协作：多个子Agent同时尝试同一任务，选最优结果。

        策略：派生 n 个子Agent处理同一任务，选误差最小的结果。
        """
        n_competitors = max(2, min(n_competitors, self.pool.max_concurrent))
        competitors = []
        for i in range(n_competitors):
            task = SubTask(
                task_id=f"comp_agent_{i}",
                description=f"Competitive attempt {i}",
                initial_state=initial_state.copy(),
                target_state=target_state.copy(),
                priority=1.0,
                max_steps=30,
            )
            agent = self.pool.spawn(task)
            if agent is None:
                break
            self.message_bus.register(agent.agent_id)
            competitors.append(agent)
        if not competitors:
            return {
                "final_state": initial_state.copy(),
                "success_rate": 0.0,
                "success_count": 0,
                "total_count": 0,
                "avg_error": float(np.linalg.norm(target_state - initial_state)),
                "total_steps": 0,
                "results": [],
            }
        # 并行执行所有竞争者
        all_results = self.pool.run_all(env_fn)
        # 选最优结果（误差最小）
        best_result = min(all_results, key=lambda r: r.get("final_error", float('inf')))
        best_state = np.asarray(best_result.get("final_state", initial_state),
                                dtype=np.float64).flatten()
        success_count = sum(1 for r in all_results if r.get("success", False))
        # 广播获胜者
        self.message_bus.publish(Message(
            sender_id="main_agent",
            msg_type=MessageType.RESULT_REPORT,
            content={"winner_steps": best_result.get("steps", 0),
                     "winner_error": best_result.get("final_error", 0.0)},
        ))
        return {
            "final_state": best_state,
            "success_rate": success_count / len(competitors),
            "success_count": success_count,
            "total_count": len(competitors),
            "avg_error": float(np.mean([r.get("final_error", 0.0) for r in all_results])),
            "total_steps": sum(r.get("steps", 0) for r in all_results),
            "results": all_results,
            "best_error": best_result.get("final_error", 0.0),
        }

    def _feedback_experience(self, skill_library) -> int:
        """从已完成的子Agent回传经验到技能库。"""
        total_merged = 0
        for record in self.pool.completed_agents:
            agent_id = record.get("agent_id")
            if agent_id is None:
                continue
            # 从记录重建简易轨迹信息
            # 注意：completed_agents 只存摘要，不存完整 agent
            # 这里用摘要信息做简单回传
            pass
        # 从活跃池中处理（如果还有的话）
        for agent in list(self.pool.active_agents.values()):
            total_merged += self.experience_feedback.process_agent(
                agent, skill_library)
        return total_merged

    def get_state(self) -> Dict[str, Any]:
        return {
            "total_collaborations": self.total_collaborations,
            "mode_stats": dict(self.mode_stats),
            "pool_state": self.pool.get_state(),
            "message_bus_state": self.message_bus.get_state(),
            "experience_state": self.experience_feedback.get_state(),
        }

    def reset(self) -> None:
        """重置编排器。"""
        self.pool = AgentPool(self.config)
        self.allocator = TaskAllocator(self.config, self.pool)
        self.coordinator = Coordinator(self.config, self.pool)
        self.message_bus.reset()
        self.experience_feedback.reset()
        self.total_collaborations = 0
        self.mode_stats = {m.value: 0 for m in CollaborationMode}
        self.log.info("CollaborationOrchestrator reset")
