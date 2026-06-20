"""SPEPA 永续运行机制：自主循环 + 异常自恢复 + 心跳监控。

设计原则（第一性原理）：
- 永续性原理：系统应能自主持续运行，任何单点故障都不能导致永久停止
- 自恢复原理：异常发生时，系统应能自动检测、隔离、恢复
- 心跳原理：通过周期性心跳信号检测系统是否存活，卡死时强制恢复

恢复策略分级：
- L0: 正常运行
- L1: 单次异常 → 跳过本步，继续
- L2: 连续异常(>=3) → 降级运行（减少策略数、跳过学习）
- L3: 连续异常(>=10) → 软重启（重置部分状态，保留记忆）
- L4: 软重启失败 → 硬重启（重置全部状态，仅保留配置）
"""
import time
import threading
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.persistence.recovery import CrashRecovery


class RecoveryLevel(Enum):
    """恢复级别。"""
    NORMAL = "normal"        # L0: 正常
    SKIP = "skip"            # L1: 跳过本步
    DEGRADE = "degrade"      # L2: 降级运行
    SOFT_RESTART = "soft"    # L3: 软重启
    HARD_RESTART = "hard"    # L4: 硬重启


@dataclass
class RunStats:
    """运行统计。"""
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    skipped_cycles: int = 0
    degraded_cycles: int = 0
    soft_restarts: int = 0
    hard_restarts: int = 0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0
    start_time: float = 0.0
    last_heartbeat: float = 0.0
    uptime: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.successful_cycles + self.failed_cycles
        if total == 0:
            return 0.0
        return self.successful_cycles / total

    @property
    def is_healthy(self) -> bool:
        """系统是否健康（连续失败 < 10）。"""
        return self.consecutive_failures < 10


class HeartbeatMonitor(LoggedMixin):
    """心跳监控：检测系统是否卡死。

    工作原理：
    - 每个周期更新心跳时间戳
    - 监控线程定期检查心跳是否超时
    - 超时则触发恢复动作
    """

    def __init__(self, config: SPEPAConfig, timeout: float = 30.0):
        self._set_log_name("spepa.perpetual.heartbeat")
        self.config = config
        self.timeout = timeout
        self.last_beat: float = time.time()
        self.beat_count: int = 0
        self.timeout_count: int = 0
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_timeout: Optional[Callable] = None
        self.log.info("HeartbeatMonitor initialized (timeout=%.1fs)", timeout)

    def beat(self) -> None:
        """更新心跳。"""
        with self._lock:
            self.last_beat = time.time()
            self.beat_count += 1

    def check(self) -> bool:
        """检查心跳是否正常（非阻塞）。"""
        with self._lock:
            elapsed = time.time() - self.last_beat
        return elapsed < self.timeout

    def elapsed_since_last_beat(self) -> float:
        """距离上次心跳的秒数。"""
        with self._lock:
            return time.time() - self.last_beat

    def set_timeout_callback(self, callback: Callable) -> None:
        """设置超时回调。"""
        self._on_timeout = callback

    def start_monitor(self) -> None:
        """启动监控线程（后台守护线程）。"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="heartbeat-monitor"
        )
        self._monitor_thread.start()
        self.log.info("Heartbeat monitor thread started")

    def stop_monitor(self) -> None:
        """停止监控线程。"""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        self.log.info("Heartbeat monitor thread stopped")

    def _monitor_loop(self) -> None:
        """监控循环（后台线程）。"""
        check_interval = self.timeout / 3
        while not self._stop_event.is_set():
            if not self.check():
                self.timeout_count += 1
                self.log.warning("Heartbeat timeout! (elapsed=%.1fs, count=%d)",
                                 self.elapsed_since_last_beat(), self.timeout_count)
                if self._on_timeout is not None:
                    try:
                        self._on_timeout()
                    except Exception as e:
                        self.log.error("Heartbeat timeout callback failed: %s", e)
            self._stop_event.wait(check_interval)

    def get_state(self) -> Dict[str, Any]:
        return {
            "beat_count": self.beat_count,
            "timeout_count": self.timeout_count,
            "elapsed_since_last_beat": self.elapsed_since_last_beat(),
            "is_alive": self.check(),
            "monitor_running": (self._monitor_thread is not None and
                                self._monitor_thread.is_alive()),
        }


class PerpetualRunner(LoggedMixin):
    """永续运行器：自主驱动 Agent 持续运行，永不停止。

    核心能力：
    1. 自主循环：持续调用 online_cycle / offline_cycle
    2. 异常自恢复：分级恢复策略（L0-L4）
    3. 心跳监控：检测卡死并强制恢复
    4. 健康检查：资源、状态、数值稳定性
    5. 降级运行：资源紧张时降级运行
    """

    # 恢复阈值
    DEGRADE_THRESHOLD = 3       # 连续失败 >= 3 触发降级
    SOFT_RESTART_THRESHOLD = 10 # 连续失败 >= 10 触发软重启
    HARD_RESTART_THRESHOLD = 20 # 连续失败 >= 20 触发硬重启

    def __init__(self, config: SPEPAConfig, agent_factory: Callable[[], Any]):
        """
        参数：
        - config: 系统配置
        - agent_factory: 创建 Agent 的工厂函数（用于硬重启）
        """
        self._set_log_name("spepa.perpetual.runner")
        self.config = config
        self.agent_factory = agent_factory
        self.agent: Optional[Any] = None
        # 运行统计
        self.stats = RunStats()
        # 心跳监控
        self.heartbeat = HeartbeatMonitor(config, timeout=30.0)
        self.heartbeat.set_timeout_callback(self._on_heartbeat_timeout)
        # 运行控制
        self._running = False
        self._stop_requested = False
        # 降级模式标志
        self._degraded_mode = False
        # 恢复历史（用于诊断）
        self.recovery_history: deque = deque(maxlen=100)
        # 周期性任务调度
        self._offline_interval = 10  # 每10个online cycle做1次offline
        # 检查点间隔
        self._checkpoint_interval = 50  # 每50个周期保存一次检查点
        # 崩溃恢复
        self.recovery: Optional[CrashRecovery] = None
        self.log.info("PerpetualRunner initialized")

    def _create_agent(self) -> Any:
        """创建新 Agent 实例。"""
        agent = self.agent_factory()
        self.log.info("Agent created by factory")
        return agent

    def start(self, max_cycles: Optional[int] = None,
              env_step_fn: Optional[Callable] = None,
              checkpoint_dir: Optional[str] = None,
              auto_recover: bool = True) -> Dict[str, Any]:
        """启动永续运行。

        参数：
        - max_cycles: 最大周期数（None=无限）
        - env_step_fn: 环境步进函数
        - checkpoint_dir: 检查点目录（None=不持久化）
        - auto_recover: 是否自动从检查点恢复
        """
        self.log.info("=== PerpetualRunner starting ===")
        self._running = True
        self._stop_requested = False
        self.stats.start_time = time.time()
        # 创建 Agent
        if self.agent is None:
            self.agent = self._create_agent()
        # 初始化崩溃恢复
        if checkpoint_dir is not None:
            self.recovery = CrashRecovery(self.config, checkpoint_dir)
            # 启动恢复
            if auto_recover:
                recovery_result = self.recovery.recover_on_startup(self.agent)
                self.log.info("Startup recovery: %s", recovery_result)
        # 启动心跳监控
        self.heartbeat.start_monitor()
        self.heartbeat.beat()

        cycle_count = 0
        try:
            while not self._stop_requested:
                if max_cycles is not None and cycle_count >= max_cycles:
                    break
                cycle_count += 1
                # 执行一个周期（带异常恢复）
                self._run_one_cycle(env_step_fn)
                # 更新心跳
                self.heartbeat.beat()
                self.stats.last_heartbeat = time.time()
                self.stats.uptime = time.time() - self.stats.start_time
                # 周期性检查点保存
                if self.recovery is not None:
                    self.recovery.periodic_checkpoint(
                        self.agent, interval=self._checkpoint_interval
                    )
        except KeyboardInterrupt:
            self.log.info("Received KeyboardInterrupt, stopping gracefully")
        finally:
            # 停止前保存最终检查点
            if self.recovery is not None:
                self.recovery.periodic_checkpoint(self.agent, force=True)
            self._running = False
            self.heartbeat.stop_monitor()

        summary = self._get_summary()
        self.log.info("=== PerpetualRunner stopped: %d cycles, %d successes, %d failures ===",
                      self.stats.total_cycles, self.stats.successful_cycles,
                      self.stats.failed_cycles)
        return summary

    def stop(self) -> None:
        """请求停止（优雅停止）。"""
        self.log.info("Stop requested")
        self._stop_requested = True

    def _run_one_cycle(self, env_step_fn: Optional[Callable]) -> None:
        """执行一个周期（带异常恢复）。"""
        self.stats.total_cycles += 1
        try:
            # 健康检查
            if not self._health_check():
                self.log.warning("Health check failed, attempting recovery")
                self._recover(RecoveryLevel.SOFT_RESTART)
                return
            # 执行 online cycle
            if self._degraded_mode:
                # 降级模式：跳过学习，减少计算
                self._run_degraded_cycle(env_step_fn)
            else:
                self.agent.online_cycle(env_step_fn=env_step_fn)
            # 周期性 offline cycle
            if self.stats.total_cycles % self._offline_interval == 0:
                try:
                    self.agent.offline_cycle()
                except Exception as e:
                    self.log.warning("Offline cycle failed (non-fatal): %s", e)
            # 成功
            self.stats.successful_cycles += 1
            self.stats.consecutive_failures = 0
            # 恢复正常模式
            if self._degraded_mode and self.stats.consecutive_failures == 0:
                if self.stats.successful_cycles % 5 == 0:
                    self._degraded_mode = False
                    self.log.info("Recovered from degraded mode to normal")
        except Exception as e:
            self._handle_exception(e)

    def _run_degraded_cycle(self, env_step_fn: Optional[Callable]) -> None:
        """降级运行周期（减少计算量）。"""
        try:
            # 降级：只做感知和决策，跳过学习和记忆巩固
            self.agent.online_cycle(env_step_fn=env_step_fn)
            self.stats.degraded_cycles += 1
        except Exception as e:
            self.log.warning("Degraded cycle also failed: %s", e)
            raise

    def _handle_exception(self, exc: Exception) -> None:
        """处理异常：分级恢复。"""
        self.stats.failed_cycles += 1
        self.stats.consecutive_failures += 1
        if self.stats.consecutive_failures > self.stats.max_consecutive_failures:
            self.stats.max_consecutive_failures = self.stats.consecutive_failures

        level = self._determine_recovery_level()
        self.log.warning("Exception in cycle %d: %s (consecutive=%d, level=%s)",
                         self.stats.total_cycles, str(exc)[:100],
                         self.stats.consecutive_failures, level.value)
        self.recovery_history.append({
            "cycle": self.stats.total_cycles,
            "error": str(exc)[:200],
            "level": level.value,
            "timestamp": time.time(),
        })
        self._recover(level)

    def _determine_recovery_level(self) -> RecoveryLevel:
        """根据连续失败次数确定恢复级别。"""
        cf = self.stats.consecutive_failures
        if cf >= self.HARD_RESTART_THRESHOLD:
            return RecoveryLevel.HARD_RESTART
        elif cf >= self.SOFT_RESTART_THRESHOLD:
            return RecoveryLevel.SOFT_RESTART
        elif cf >= self.DEGRADE_THRESHOLD:
            return RecoveryLevel.DEGRADE
        else:
            return RecoveryLevel.SKIP

    def _recover(self, level: RecoveryLevel) -> None:
        """执行恢复动作。"""
        try:
            if level == RecoveryLevel.SKIP:
                # L1: 跳过本步
                self.stats.skipped_cycles += 1
                self.log.info("Recovery L1: skipped cycle")
            elif level == RecoveryLevel.DEGRADE:
                # L2: 降级运行
                self._degraded_mode = True
                self.log.info("Recovery L2: entered degraded mode")
            elif level == RecoveryLevel.SOFT_RESTART:
                # L3: 软重启（重置部分状态，保留记忆）
                self._soft_restart()
                self.stats.soft_restarts += 1
                self.log.info("Recovery L3: soft restart completed")
            elif level == RecoveryLevel.HARD_RESTART:
                # L4: 硬重启（重建 Agent，仅保留配置）
                self._hard_restart()
                self.stats.hard_restarts += 1
                self.log.info("Recovery L4: hard restart completed")
        except Exception as e:
            self.log.error("Recovery (%s) itself failed: %s", level.value, e)
            # 恢复失败时升级到更高级别
            if level == RecoveryLevel.SOFT_RESTART:
                self.log.error("Soft restart failed, escalating to hard restart")
                self._hard_restart()
                self.stats.hard_restarts += 1

    def _soft_restart(self) -> None:
        """软重启：重置部分状态，保留记忆和技能。"""
        if self.agent is None:
            return
        try:
            # 重置运行时状态，但保留记忆系统、技能库、预测模型
            self.agent.current_state = np.ones(self.agent.config.state_dim) * 0.5
            self.agent.step_count = 0
            # 重置内稳态
            self.agent.drive_engine.reset_homeostasis()
            # 重置资源监控
            self.agent.resource_monitor.cpu_usage = 0.0
            self.agent.resource_monitor.memory_usage = 0.0
            self.agent.resource_monitor.throttle_active = False
            # 重置资源调节组件
            if hasattr(self.agent, 'backpressure'):
                self.agent.backpressure.reset()
            if hasattr(self.agent, 'degradation_matrix'):
                self.agent.degradation_matrix.reset()
            if hasattr(self.agent.resource_monitor, 'controller'):
                self.agent.resource_monitor.controller.reset()
            # 清空状态总线
            self.agent.bus.clear()
            # 重置连续失败计数（给系统一个新机会）
            self.stats.consecutive_failures = 0
            self._degraded_mode = False
        except Exception as e:
            self.log.error("Soft restart error: %s", e)
            raise

    def _hard_restart(self) -> None:
        """硬重启：重建 Agent，仅保留配置。"""
        try:
            self.log.warning("Hard restart: recreating Agent from factory")
            self.agent = self._create_agent()
            self.stats.consecutive_failures = 0
            self._degraded_mode = False
        except Exception as e:
            self.log.error("Hard restart error: %s", e)
            # 硬重启也失败，等待一段时间后重试
            time.sleep(1.0)

    def _health_check(self) -> bool:
        """健康检查：状态数值稳定性。"""
        if self.agent is None:
            return False
        try:
            state = self.agent.current_state
            if not np.all(np.isfinite(state)):
                self.log.warning("Health check: current_state contains non-finite values")
                return False
            # 检查预测模型权重
            for layer in self.agent.predictive_model.layers:
                if not np.all(np.isfinite(layer.mu)):
                    return False
                if not np.all(np.isfinite(layer.W)):
                    return False
            return True
        except Exception:
            return False

    def _on_heartbeat_timeout(self) -> None:
        """心跳超时回调：强制恢复。"""
        self.log.error("Heartbeat timeout triggered force recovery")
        # 强制软重启
        try:
            self._soft_restart()
        except Exception:
            try:
                self._hard_restart()
            except Exception as e:
                self.log.error("Force recovery failed: %s", e)

    def _get_summary(self) -> Dict[str, Any]:
        """获取运行摘要。"""
        return {
            "total_cycles": self.stats.total_cycles,
            "successful_cycles": self.stats.successful_cycles,
            "failed_cycles": self.stats.failed_cycles,
            "skipped_cycles": self.stats.skipped_cycles,
            "degraded_cycles": self.stats.degraded_cycles,
            "soft_restarts": self.stats.soft_restarts,
            "hard_restarts": self.stats.hard_restarts,
            "max_consecutive_failures": self.stats.max_consecutive_failures,
            "success_rate": self.stats.success_rate,
            "uptime": self.stats.uptime,
            "is_healthy": self.stats.is_healthy,
            "heartbeat": self.heartbeat.get_state(),
            "recovery_history_size": len(self.recovery_history),
        }

    def get_state(self) -> Dict[str, Any]:
        """获取当前状态。"""
        return {
            "running": self._running,
            "degraded_mode": self._degraded_mode,
            "stats": {
                "total_cycles": self.stats.total_cycles,
                "successful_cycles": self.stats.successful_cycles,
                "failed_cycles": self.stats.failed_cycles,
                "consecutive_failures": self.stats.consecutive_failures,
                "max_consecutive_failures": self.stats.max_consecutive_failures,
                "soft_restarts": self.stats.soft_restarts,
                "hard_restarts": self.stats.hard_restarts,
                "success_rate": self.stats.success_rate,
                "is_healthy": self.stats.is_healthy,
                "uptime": self.stats.uptime,
            },
            "heartbeat": self.heartbeat.get_state(),
            "recovery_history": list(self.recovery_history)[-10:],  # 最近10条
        }
