"""SPEPA 崩溃恢复机制：启动恢复、断点续跑。

设计原则（第一性原理）：
- 可恢复性原理：任何检查点都应能完整恢复系统状态
- 断点续跑原理：崩溃后从最后成功的检查点继续运行
- 渐进恢复原理：优先尝试最新检查点，失败则回退到更早版本
"""
import os
import time
from typing import Dict, Any, Optional, Callable

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.persistence.checkpoint import CheckpointManager


class CrashRecovery(LoggedMixin):
    """崩溃恢复机制。

    功能：
    - 启动恢复：启动时自动加载最新检查点
    - 断点续跑：从最后成功的检查点继续运行
    - 渐进回退：最新检查点失败时回退到更早版本
    - 恢复验证：恢复后验证状态完整性
    """

    def __init__(self, config: SPEPAConfig, checkpoint_dir: str = "checkpoints"):
        self._set_log_name("spepa.persistence.recovery")
        self.config = config
        self.checkpoint_mgr = CheckpointManager(config, checkpoint_dir)
        # 恢复统计
        self.recovery_attempts = 0
        self.recovery_successes = 0
        self.recovery_failures = 0
        self.last_recovery_time = 0.0
        self.log.info("CrashRecovery initialized (dir=%s)", checkpoint_dir)

    def recover_on_startup(self, agent: Any) -> Dict[str, Any]:
        """启动恢复：尝试加载最新检查点。

        渐进回退策略：
        1. 尝试最新版本
        2. 失败则尝试次新版本
        3. 全部失败则返回未恢复
        """
        self.recovery_attempts += 1
        checkpoints = self.checkpoint_mgr.list_checkpoints()
        if not checkpoints:
            self.log.info("No checkpoints found, starting fresh")
            return {"recovered": False, "reason": "no_checkpoints"}
        # 按版本号降序尝试
        checkpoints.sort(key=lambda c: -c["version"])
        for cp in checkpoints:
            version = cp["version"]
            self.log.info("Attempting recovery from version %d (step=%d)",
                          version, cp["step_count"])
            success = self.checkpoint_mgr.load(agent, version=version)
            if success:
                # 验证恢复后的状态
                if self._verify_agent_state(agent):
                    self.recovery_successes += 1
                    self.last_recovery_time = time.time()
                    self.log.info("Recovery successful from version %d", version)
                    return {
                        "recovered": True,
                        "version": version,
                        "step_count": agent.step_count,
                    }
                else:
                    self.log.warning("Recovery from version %d failed verification", version)
            else:
                self.log.warning("Recovery from version %d failed", version)
        self.recovery_failures += 1
        self.log.warning("All recovery attempts failed, starting fresh")
        return {"recovered": False, "reason": "all_attempts_failed"}

    def _verify_agent_state(self, agent: Any) -> bool:
        """验证恢复后的 Agent 状态完整性。"""
        import numpy as np
        try:
            # 检查 current_state
            if not np.all(np.isfinite(agent.current_state)):
                self.log.error("Verification failed: current_state non-finite")
                return False
            # 检查预测模型
            for layer in agent.predictive_model.layers:
                if not np.all(np.isfinite(layer.mu)):
                    return False
                if not np.all(np.isfinite(layer.W)):
                    return False
                if not np.all(np.isfinite(layer.var)):
                    return False
            # 检查 step_count 合理性
            if agent.step_count < 0:
                return False
            return True
        except Exception as e:
            self.log.error("Verification error: %s", e)
            return False

    def periodic_checkpoint(self, agent: Any, interval: int = 50,
                            force: bool = False) -> Optional[Dict[str, Any]]:
        """周期性保存检查点。

        参数：
        - interval: 保存间隔（周期数）
        - force: 强制保存（忽略间隔）
        """
        if not force and agent.step_count % interval != 0:
            return None
        if not force and agent.step_count == 0:
            return None
        info = self.checkpoint_mgr.save(agent)
        if info is not None:
            return {
                "saved": True,
                "version": info.version,
                "step_count": info.step_count,
                "size_bytes": info.size_bytes,
            }
        return {"saved": False}

    def get_state(self) -> Dict[str, Any]:
        return {
            "recovery_attempts": self.recovery_attempts,
            "recovery_successes": self.recovery_successes,
            "recovery_failures": self.recovery_failures,
            "last_recovery_time": self.last_recovery_time,
            "checkpoint_state": self.checkpoint_mgr.get_state(),
        }
