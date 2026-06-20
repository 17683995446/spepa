"""SPEPA 全局日志系统。

每个方法执行后都打印日志，便于追踪系统行为。

日志管理功能：
- get_logger() : 获取带统一格式的 logger
- set_log_level() : 全局或按模块设置日志级别
- add_file_handler() : 添加文件输出
- remove_handlers() : 清理处理器
- get_log_config() : 获取当前日志配置
- quiet_mode() / verbose_mode() : 快捷切换
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List


_LOGGER_CACHE = {}


def get_logger(name: str = "spepa", level: int = logging.INFO) -> logging.Logger:
    """获取（或创建）一个带统一格式的 logger。

    每次调用都会返回同一个 logger 实例，避免重复添加 handler。
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 避免重复 handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.propagate = False
    _LOGGER_CACHE[name] = logger
    logger.info("Logger initialized: name=%s level=%s", name, logging.getLevelName(level))
    return logger


class LoggedMixin:
    """为所有模块提供统一的日志能力。

    子类可直接使用 self.log.info / self.log.debug 等。
    """

    @property
    def log(self) -> logging.Logger:
        if not hasattr(self, "_log_name") or self._log_name is None:
            self._log_name = "spepa." + self.__class__.__name__
        return get_logger(self._log_name)

    def _set_log_name(self, name: str) -> None:
        self._log_name = name
        self.log.debug("Log name set to %s", name)


def log_call(method):
    """装饰器：在方法执行前后打印日志。"""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        logger = getattr(self, "log", None) or get_logger("spepa")
        logger.debug(">> %s.%s called", self.__class__.__name__, method.__name__)
        result = method(self, *args, **kwargs)
        logger.debug("<< %s.%s returned", self.__class__.__name__, method.__name__)
        return result

    return wrapper


def now_str() -> str:
    """返回当前时间字符串。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_dir(path: str) -> str:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================
# 日志管理增强功能
# ============================================================

# 级别名称到数值的映射（方便用户使用字符串）
_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def _parse_level(level) -> int:
    """将级别名称或数值转为 logging 级别数值。"""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return _LEVEL_MAP.get(level.lower(), logging.INFO)
    return logging.INFO


def set_log_level(level, logger_name: Optional[str] = None) -> None:
    """设置日志级别。

    参数：
        level : 日志级别，可为字符串("debug"/"info"/"warning"/"error")或数值
        logger_name : 指定 logger 名称（None=设置所有已缓存的 logger）

    示例：
        >>> set_log_level("debug")  # 所有 logger 设为 DEBUG
        >>> set_log_level("warning", "spepa.agent")  # 仅 Agent logger
    """
    level_val = _parse_level(level)
    if logger_name is not None:
        # 设置指定 logger
        logger = _LOGGER_CACHE.get(logger_name) or logging.getLogger(logger_name)
        logger.setLevel(level_val)
        # 同步更新 handler 级别
        for h in logger.handlers:
            h.setLevel(level_val)
    else:
        # 设置所有已缓存的 logger
        for name, logger in _LOGGER_CACHE.items():
            logger.setLevel(level_val)
            for h in logger.handlers:
                h.setLevel(level_val)
        # 同时设置根 spepa logger
        root = logging.getLogger("spepa")
        root.setLevel(level_val)


def add_file_handler(filepath: str, level=None,
                     logger_name: Optional[str] = None) -> logging.FileHandler:
    """添加文件日志处理器。

    参数：
        filepath : 日志文件路径
        level : 文件处理器级别（None=继承 logger 级别）
        logger_name : 指定 logger（None=所有已缓存的 logger）

    返回：
        FileHandler 实例

    示例：
        >>> add_file_handler("spepa.log", "debug")
    """
    ensure_dir(os.path.dirname(filepath)) if os.path.dirname(filepath) else None
    handler = logging.FileHandler(filepath, encoding='utf-8')
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    if level is not None:
        handler.setLevel(_parse_level(level))

    if logger_name is not None:
        logger = _LOGGER_CACHE.get(logger_name) or logging.getLogger(logger_name)
        logger.addHandler(handler)
    else:
        for logger in _LOGGER_CACHE.values():
            logger.addHandler(handler)
    return handler


def remove_handlers(logger_name: Optional[str] = None,
                    handler_type: Optional[type] = None) -> None:
    """移除日志处理器。

    参数：
        logger_name : 指定 logger（None=所有已缓存的 logger）
        handler_type : 指定处理器类型（None=所有类型，如 logging.FileHandler）
    """
    target_loggers = []
    if logger_name is not None:
        target_loggers.append(_LOGGER_CACHE.get(logger_name) or logging.getLogger(logger_name))
    else:
        target_loggers = list(_LOGGER_CACHE.values())

    for logger in target_loggers:
        if handler_type is not None:
            logger.handlers = [h for h in logger.handlers if not isinstance(h, handler_type)]
        else:
            logger.handlers.clear()


def get_log_config() -> Dict[str, Any]:
    """获取当前所有 logger 的配置信息。

    返回：
        {logger_name: {level, handlers: [{type, level}]}}
    """
    config = {}
    for name, logger in _LOGGER_CACHE.items():
        config[name] = {
            "level": logging.getLevelName(logger.level),
            "handlers": [
                {
                    "type": type(h).__name__,
                    "level": logging.getLevelName(h.level) if hasattr(h, 'level') else None,
                }
                for h in logger.handlers
            ],
        }
    return config


def quiet_mode() -> None:
    """静默模式：仅输出 WARNING 及以上级别的日志。"""
    set_log_level(logging.WARNING)


def verbose_mode() -> None:
    """详细模式：输出 DEBUG 及以上级别的日志。"""
    set_log_level(logging.DEBUG)


def list_loggers() -> List[str]:
    """列出所有已创建的 logger 名称。"""
    return sorted(_LOGGER_CACHE.keys())
