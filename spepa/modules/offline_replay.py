"""SPEPA 离线复盘模块（低负载时触发）。

在系统负载较低时，回放记忆，巩固学习，重构模型。
"""
import numpy as np
from typing import Dict, Any, List, Optional

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.core.predictive_model import PredictiveModel
from spepa.core.memory_system import MemorySystem
from spepa.core.optimizer import Optimizer
from spepa.core.metacognition import Metacognition
from spepa.core.self_model import SelfModel
from spepa.core.state_bus import StateBus


class OfflineReplay(LoggedMixin):
    """离线复盘模块。"""

    def __init__(self, config: SPEPAConfig, model: PredictiveModel,
                 memory: MemorySystem, optimizer: Optimizer,
                 metacog: Metacognition, self_model: SelfModel,
                 state_bus: StateBus):
        self._set_log_name("spepa.modules.offline_replay")
        self.config = config
        self.model = model
        self.memory = memory
        self.optimizer = optimizer
        self.metacog = metacog
        self.self_model = self_model
        self.bus = state_bus
        self.replay_count = 0
        self.last_replay_results: List[Dict[str, Any]] = []
        self.log.info("OfflineReplay module initialized")

    def should_trigger(self, current_load: float) -> bool:
        """是否应触发离线复盘（负载低于阈值）。"""
        trigger = current_load < self.config.offline_load_threshold
        self.log.debug("Offline replay trigger check: load=%.4f threshold=%.4f -> %s",
                       current_load, self.config.offline_load_threshold, trigger)
        return trigger

    def replay(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """执行一次离线复盘。"""
        batch_size = batch_size or self.config.offline_replay_batch
        memories = self.memory.replay(n=batch_size)
        if not memories:
            self.log.info("Offline replay skipped: no memories")
            return {"replayed": 0, "improvements": []}

        improvements: List[Dict[str, Any]] = []
        total_pre_err = 0.0
        total_post_err = 0.0

        for mem in memories:
            obs = mem.key
            # 重新预测
            pre_err = self.model.total_prediction_error(obs)
            total_pre_err += pre_err
            # 用原始误差驱动学习
            original_err = mem.value.get("error", pre_err) if isinstance(mem.value, dict) else pre_err
            opt_result = self.optimizer.optimize(self.model, obs, original_err)
            post_err = self.model.total_prediction_error(obs)
            total_post_err += post_err
            improvements.append({
                "pre_error": pre_err,
                "post_error": post_err,
                "mode": opt_result["mode"],
            })

        # 巩固记忆
        consolidated = self.memory.consolidate()

        # 元认知校准
        if improvements:
            avg_pre = total_pre_err / len(improvements)
            avg_post = total_post_err / len(improvements)
            self.metacog.record_outcome(0.5, avg_post)

        self.replay_count += 1
        result = {
            "replayed": len(memories),
            "consolidated": consolidated,
            "avg_pre_error": float(total_pre_err / max(len(improvements), 1)),
            "avg_post_error": float(total_post_err / max(len(improvements), 1)),
            "improvements": improvements,
            "replay_count": self.replay_count,
        }
        self.last_replay_results = improvements
        self.bus.set_state("last_offline_replay", result)
        self.bus.publish("offline_replay", result)
        self.log.info("Offline replay #%d: replayed=%d consolidated=%d err %.4f->%.4f",
                      self.replay_count, len(memories), consolidated,
                      result["avg_pre_error"], result["avg_post_error"])
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "replay_count": self.replay_count,
            "last_replay_size": len(self.last_replay_results),
        }
