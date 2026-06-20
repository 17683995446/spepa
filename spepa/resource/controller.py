"""SPEPA 资源控制器：PID 式自适应资源调节。

设计原则（第一性原理）：
- 守恒原理：系统消耗的资源不应超过可用资源
- 稳态调节原理：资源使用应围绕目标值振荡，不超调、不振荡发散
- 比例-积分-微分（PID）控制：用误差的比例、积分、微分三项组合控制
- 因果性原理：调节信号由实际资源使用驱动，而非预设阈值

PID 控制器公式：
    u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de(t)/dt
其中 e(t) = setpoint - measurement（误差）

在 SPEPA 中：
- setpoint = 资源上限（cpu_max, memory_max）
- measurement = 实际资源使用率
- e(t) > 0 表示资源有余量，可以全速运行
- e(t) < 0 表示资源超限，需要降级
- u(t) 映射到 throttle_signal ∈ [0, 1]（0=全速，1=最大降级）
"""
import time
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig


@dataclass
class ThrottleSignal:
    """节流信号：传递给各模块的降级指令。"""
    throttle: float = 0.0  # 0.0=全速, 1.0=最大降级
    n_policies: int = 8    # 策略数量
    skip_learning: bool = False
    skip_offline: bool = False
    skip_memory_consolidation: bool = False
    learning_interval: int = 1  # 每N步学习一次
    memory_interval: int = 1    # 每N步记忆步进一次

    @property
    def is_throttled(self) -> bool:
        return self.throttle > 0.01


class ResourceController(LoggedMixin):
    """PID 式资源控制器。

    根据实际 CPU/内存使用率与目标上限的误差，
    通过 PID 控制计算节流信号，驱动系统降级或恢复。
    """

    def __init__(self, config: SPEPAConfig,
                 kp: float = 2.0, ki: float = 0.5, kd: float = 0.3):
        """
        参数：
        - config: 系统配置（含资源上限）
        - kp: 比例增益（对当前误差的响应强度）
        - ki: 积分增益（消除稳态误差）
        - kd: 微分增益（抑制超调）
        """
        self._set_log_name("spepa.resource.controller")
        self.config = config
        # PID 参数
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # 目标值（资源上限）
        self.cpu_target = config.resources.cpu_max_percent
        self.memory_target = config.resources.memory_max_percent
        # PID 状态
        self._integral_cpu = 0.0
        self._integral_mem = 0.0
        self._prev_error_cpu = 0.0
        self._prev_error_mem = 0.0
        # 当前测量值
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        # 当前节流信号
        self.current_signal = ThrottleSignal()
        # 历史记录（用于分析）
        self._history: deque = deque(maxlen=100)
        # 时间戳
        self._last_update = time.time()
        self.log.info("ResourceController initialized (targets: cpu<=%.1f%% mem<=%.1f%%)",
                      self.cpu_target, self.memory_target)

    def update(self, cpu_usage: float, memory_usage: float) -> ThrottleSignal:
        """更新资源使用测量值，计算节流信号。

        参数：
        - cpu_usage: 当前 CPU 使用率（0-100）
        - memory_usage: 当前内存使用率（0-100）
        """
        self.cpu_usage = float(np.clip(cpu_usage, 0.0, 100.0))
        self.memory_usage = float(np.clip(memory_usage, 0.0, 100.0))
        now = time.time()
        dt = max(now - self._last_update, 1e-6)
        self._last_update = now
        # 计算误差（正=有余量，负=超限）
        error_cpu = self.cpu_target - self.cpu_usage
        error_mem = self.memory_target - self.memory_usage
        # 取最严格的误差（资源最紧张的那个）
        error = min(error_cpu, error_mem)
        # PID 三项
        # 比例项
        p_term = self.kp * error
        # 积分项（带抗饱和：限制积分范围）
        self._integral_cpu += error_cpu * dt
        self._integral_mem += error_mem * dt
        # 积分抗饱和
        max_integral = 50.0  # 积分上限
        self._integral_cpu = float(np.clip(self._integral_cpu, -max_integral, max_integral))
        self._integral_mem = float(np.clip(self._integral_mem, -max_integral, max_integral))
        integral = min(self._integral_cpu, self._integral_mem)
        i_term = self.ki * integral
        # 微分项
        d_error_cpu = (error_cpu - self._prev_error_cpu) / dt
        d_error_mem = (error_mem - self._prev_error_mem) / dt
        d_error = min(d_error_cpu, d_error_mem)
        d_term = self.kd * d_error
        self._prev_error_cpu = error_cpu
        self._prev_error_mem = error_mem
        # PID 输出（正值=有余量，负值=超限）
        pid_output = p_term + i_term + d_term
        # 映射到节流信号 [0, 1]
        # pid_output 越负（超限越严重），throttle 越大
        throttle = float(np.clip(-pid_output / 20.0, 0.0, 1.0))
        # 生成节流信号
        self.current_signal = self._compute_signal(throttle)
        # 记录历史
        self._history.append({
            "cpu": self.cpu_usage,
            "memory": self.memory_usage,
            "error": error,
            "throttle": throttle,
            "timestamp": now,
        })
        if throttle > 0.1:
            self.log.debug("Resource throttle: %.2f (cpu=%.1f%% mem=%.1f%% err=%.1f)",
                           throttle, self.cpu_usage, self.memory_usage, error)
        return self.current_signal

    def _compute_signal(self, throttle: float) -> ThrottleSignal:
        """根据节流程度计算具体的降级参数。"""
        signal = ThrottleSignal()
        signal.throttle = throttle
        # 策略数量：8 -> 4 -> 2
        if throttle < 0.25:
            signal.n_policies = 8
        elif throttle < 0.5:
            signal.n_policies = 4
        elif throttle < 0.75:
            signal.n_policies = 2
        else:
            signal.n_policies = 1
        # 学习间隔：1 -> 2 -> 5
        if throttle < 0.3:
            signal.learning_interval = 1
            signal.skip_learning = False
        elif throttle < 0.6:
            signal.learning_interval = 2
            signal.skip_learning = False
        elif throttle < 0.85:
            signal.learning_interval = 5
            signal.skip_learning = False
        else:
            signal.skip_learning = True
        # 离线复盘
        signal.skip_offline = throttle >= 0.5
        # 记忆巩固
        if throttle >= 0.7:
            signal.skip_memory_consolidation = True
            signal.memory_interval = 10
        elif throttle >= 0.4:
            signal.memory_interval = 3
        else:
            signal.memory_interval = 1
        return signal

    def reset(self) -> None:
        """重置控制器状态。"""
        self._integral_cpu = 0.0
        self._integral_mem = 0.0
        self._prev_error_cpu = 0.0
        self._prev_error_mem = 0.0
        self.current_signal = ThrottleSignal()
        self._history.clear()
        self.log.info("ResourceController reset")

    def get_state(self) -> Dict[str, Any]:
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "cpu_target": self.cpu_target,
            "memory_target": self.memory_target,
            "throttle": self.current_signal.throttle,
            "n_policies": self.current_signal.n_policies,
            "skip_learning": self.current_signal.skip_learning,
            "skip_offline": self.current_signal.skip_offline,
            "integral_cpu": self._integral_cpu,
            "integral_mem": self._integral_mem,
            "history_size": len(self._history),
        }

    def get_history(self, n: int = 20) -> list:
        """获取最近 n 条历史记录。"""
        return list(self._history)[-n:]
