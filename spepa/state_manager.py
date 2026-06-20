"""SPEPA 状态管理与断点续传（高级 API）。

设计原则（第一性原理）：
- 简单易用：一行代码保存/恢复 Agent 状态
- 自动化：运行时自动定期保存检查点
- 透明恢复：从检查点恢复后可无缝继续运行
- 零侵入：基于现有 persistence 模块封装，不修改核心逻辑

主要功能：
- StateManager : 高级状态管理器
- save_state() / load_state() : 便捷函数
- resume_run() : 从检查点恢复并继续运行
- list_checkpoints() : 列出可用检查点
"""
import os
import time
from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager

from spepa.utils.logger import get_logger
from spepa.utils.config import SPEPAConfig
from spepa.persistence.checkpoint import CheckpointManager
from spepa.persistence.recovery import CrashRecovery

logger = get_logger("spepa.state_manager")


class StateManager:
    """高级状态管理器：封装检查点与恢复功能。

    使用方式：
        # 基本用法
        manager = StateManager(agent, checkpoint_dir="checkpoints")
        manager.save()                    # 保存当前状态
        agent.run(n_steps=100)
        manager.save()                    # 再次保存

        # 自动保存
        with manager.auto_save(interval=50):
            agent.run(n_steps=500)

        # 恢复并继续
        manager.load()                    # 加载最新检查点
        agent.run(n_steps=100)            # 从检查点继续

        # 从崩溃恢复
        manager.recover()                 # 启动时恢复
    """

    def __init__(self, agent, checkpoint_dir: str = "checkpoints"):
        """初始化状态管理器。

        参数：
            agent : SPEPAAgent 实例
            checkpoint_dir : 检查点存储目录
        """
        self.agent = agent
        self.config: SPEPAConfig = agent.config
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_mgr = CheckpointManager(self.config, checkpoint_dir)
        self.recovery = CrashRecovery(self.config, checkpoint_dir)
        self._auto_save_active = False

    def save(self, force: bool = True) -> Optional[Dict[str, Any]]:
        """保存当前 Agent 状态到检查点。

        参数：
            force : 是否强制保存（忽略间隔检查）

        返回：
            检查点信息字典，失败返回 None
        """
        info = self.checkpoint_mgr.save(self.agent)
        if info:
            logger.info("State saved: version=%d, step=%d",
                        info.version, info.step_count)
            return {
                "version": info.version,
                "timestamp": info.timestamp,
                "step_count": info.step_count,
                "file_path": info.file_path,
            }
        return None

    def load(self, version: Optional[int] = None) -> bool:
        """从检查点加载状态到 Agent。

        参数：
            version : 指定版本号（None=最新）

        返回：
            是否加载成功
        """
        success = self.checkpoint_mgr.load(self.agent, version=version)
        if success:
            logger.info("State loaded: step=%d", self.agent.step_count)
        else:
            logger.warning("Failed to load state")
        return success

    def recover(self) -> Dict[str, Any]:
        """启动时从崩溃恢复（自动选择最佳检查点）。

        返回：
            {"recovered": bool, "version": int, "step_count": int}
        """
        result = self.recovery.recover_on_startup(self.agent)
        if result.get("recovered"):
            logger.info("Recovered from crash: step=%d", self.agent.step_count)
        return result

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有可用检查点。

        返回：
            检查点信息列表（按版本降序）
        """
        checkpoints = self.checkpoint_mgr.list_checkpoints()
        # 按版本降序排列
        checkpoints.sort(key=lambda x: x.get("version", 0), reverse=True)
        return checkpoints

    def delete_checkpoint(self, version: int) -> bool:
        """删除指定版本的检查点。

        参数：
            version : 版本号

        返回：
            是否删除成功
        """
        return self.checkpoint_mgr.delete_checkpoint(version)

    def clear_all(self) -> int:
        """清空所有检查点。

        返回：
            删除的检查点数量
        """
        return self.checkpoint_mgr.clear_all()

    def get_info(self) -> Dict[str, Any]:
        """获取状态管理器信息。

        返回：
            包含检查点数量、最新检查点等信息的字典
        """
        latest = self.checkpoint_mgr.get_latest_info()
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "total_checkpoints": len(self.checkpoint_mgr.list_checkpoints()),
            "latest_version": latest.version if latest else None,
            "latest_step_count": latest.step_count if latest else None,
            "latest_timestamp": latest.timestamp if latest else None,
            "auto_save_active": self._auto_save_active,
        }

    @contextmanager
    def auto_save(self, interval: int = 50):
        """上下文管理器：自动定期保存检查点。

        参数：
            interval : 保存间隔（步数）

        使用方式：
            with manager.auto_save(interval=50):
                agent.run(n_steps=500)
        """
        self._auto_save_active = True
        original_step = self.agent.step_count
        try:
            # 订阅事件总线以实现自动保存
            from spepa.events import EventType, Event

            def on_step_end(event: Event):
                step = event.data.get("step", 0)
                if step > 0 and step % interval == 0:
                    self.save(force=True)

            self.agent.event_bus.subscribe(EventType.STEP_END, on_step_end)
            logger.info("Auto-save enabled (interval=%d)", interval)
            yield
        finally:
            self.agent.event_bus.unsubscribe(EventType.STEP_END, on_step_end)
            self._auto_save_active = False
            # 退出时保存最终状态
            if self.agent.step_count > original_step:
                self.save(force=True)
            logger.info("Auto-save disabled")

    def resume_run(self, n_steps: int = 100,
                   save_on_exit: bool = True) -> Dict[str, Any]:
        """从检查点恢复并继续运行。

        参数：
            n_steps : 继续运行的步数
            save_on_exit : 运行结束后是否保存

        返回：
            运行结果字典
        """
        # 1. 尝试恢复
        recovery_result = self.recover()
        recovered = recovery_result.get("recovered", False)
        start_step = self.agent.step_count

        logger.info("Resume run: start_step=%d, n_steps=%d, recovered=%s",
                    start_step, n_steps, recovered)

        # 2. 继续运行
        run_result = self.agent.run(n_steps=n_steps)

        # 3. 保存
        if save_on_exit:
            self.save(force=True)

        # 4. 汇总结果
        return {
            "recovered": recovered,
            "start_step": start_step,
            "end_step": self.agent.step_count,
            "steps_executed": self.agent.step_count - start_step,
            "run_result": run_result,
        }


# ============================================================
# 便捷函数
# ============================================================

def save_state(agent, checkpoint_dir: str = "checkpoints") -> Optional[Dict[str, Any]]:
    """便捷函数：保存 Agent 状态。

    参数：
        agent : SPEPAAgent 实例
        checkpoint_dir : 检查点目录

    返回：
        检查点信息字典
    """
    manager = StateManager(agent, checkpoint_dir)
    return manager.save()


def load_state(agent, checkpoint_dir: str = "checkpoints",
               version: Optional[int] = None) -> bool:
    """便捷函数：加载 Agent 状态。

    参数：
        agent : SPEPAAgent 实例
        checkpoint_dir : 检查点目录
        version : 版本号（None=最新）

    返回：
        是否加载成功
    """
    manager = StateManager(agent, checkpoint_dir)
    return manager.load(version=version)


def resume_run(agent, n_steps: int = 100,
               checkpoint_dir: str = "checkpoints") -> Dict[str, Any]:
    """便捷函数：从检查点恢复并继续运行。

    参数：
        agent : SPEPAAgent 实例
        n_steps : 继续运行步数
        checkpoint_dir : 检查点目录

    返回：
        运行结果字典
    """
    manager = StateManager(agent, checkpoint_dir)
    return manager.resume_run(n_steps=n_steps)


def list_checkpoints(checkpoint_dir: str = "checkpoints",
                     config: Optional[SPEPAConfig] = None) -> List[Dict[str, Any]]:
    """便捷函数：列出检查点。

    参数：
        checkpoint_dir : 检查点目录
        config : 配置（None=默认配置）

    返回：
        检查点信息列表
    """
    if config is None:
        from spepa.utils.config import get_default_config
        config = get_default_config()
    manager = CheckpointManager(config, checkpoint_dir)
    return manager.list_checkpoints()
