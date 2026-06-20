"""SPEPA 检查点管理器：增量保存、版本管理、原子写入。

设计原则（第一性原理）：
- 原子性原理：检查点写入要么完全成功，要么完全失败（不留半成品）
- 版本管理：保留最近K个版本，支持回滚
- 完整性校验：每个检查点带校验和验证
- 清理策略：自动清理过期检查点
"""
import os
import json
import time
import hashlib
import tempfile
import shutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from spepa.utils.logger import LoggedMixin
from spepa.utils.config import SPEPAConfig
from spepa.persistence.serializer import StateSerializer


@dataclass
class CheckpointInfo:
    """检查点元信息。"""
    version: int
    timestamp: float
    step_count: int
    file_path: str
    checksum: str
    size_bytes: int


class CheckpointManager(LoggedMixin):
    """检查点管理器。

    功能：
    - 保存检查点（原子写入）
    - 加载检查点
    - 版本管理（保留最近K个）
    - 完整性校验
    - 自动清理
    """

    def __init__(self, config: SPEPAConfig, checkpoint_dir: str = "checkpoints"):
        self._set_log_name("spepa.persistence.checkpoint")
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.serializer = StateSerializer(config)
        # 保留的版本数
        self.max_versions = 5
        # 检查点索引
        self._index: List[CheckpointInfo] = []
        # 当前版本号
        self._version_counter = 0
        # 确保目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # 加载已有索引
        self._load_index()
        self.log.info("CheckpointManager initialized (dir=%s, versions=%d)",
                      self.checkpoint_dir, len(self._index))

    def save(self, agent: Any) -> Optional[CheckpointInfo]:
        """保存检查点（原子写入）。"""
        try:
            # 序列化状态
            state = self.serializer.serialize_agent(agent)
            state_json = json.dumps(state, ensure_ascii=False, default=str)
            # 计算校验和
            checksum = hashlib.md5(state_json.encode("utf-8")).hexdigest()
            # 版本号
            self._version_counter += 1
            version = self._version_counter
            # 文件路径
            filename = f"checkpoint_{version:06d}_{int(time.time())}.json"
            file_path = os.path.join(self.checkpoint_dir, filename)
            # 原子写入：先写临时文件，再重命名
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self.checkpoint_dir, suffix=".tmp", prefix="ckpt_"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.write(state_json)
                # 重命名（原子操作）
                os.rename(tmp_path, file_path)
            except Exception:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
            # 记录检查点信息
            size_bytes = os.path.getsize(file_path)
            info = CheckpointInfo(
                version=version,
                timestamp=time.time(),
                step_count=agent.step_count,
                file_path=file_path,
                checksum=checksum,
                size_bytes=size_bytes,
            )
            self._index.append(info)
            # 清理旧版本
            self._cleanup_old_versions()
            # 保存索引
            self._save_index()
            self.log.info("Checkpoint saved: version=%d, step=%d, size=%d bytes",
                          version, agent.step_count, size_bytes)
            return info
        except Exception as e:
            self.log.error("Save checkpoint failed: %s", e)
            return None

    def load(self, agent: Any, version: Optional[int] = None) -> bool:
        """加载检查点。

        参数：
        - version: 指定版本号（None=加载最新）
        """
        if not self._index:
            self.log.warning("No checkpoints to load")
            return False
        # 选择检查点
        if version is None:
            info = self._index[-1]  # 最新
        else:
            info = None
            for ci in self._index:
                if ci.version == version:
                    info = ci
                    break
            if info is None:
                self.log.warning("Checkpoint version %d not found", version)
                return False
        # 读取文件
        try:
            with open(info.file_path, "r", encoding="utf-8") as f:
                state_json = f.read()
            # 校验和验证
            actual_checksum = hashlib.md5(state_json.encode("utf-8")).hexdigest()
            if actual_checksum != info.checksum:
                self.log.error("Checksum mismatch: expected=%s, actual=%s",
                               info.checksum, actual_checksum)
                return False
            # 反序列化
            state = json.loads(state_json)
            success = self.serializer.deserialize_agent(agent, state)
            if success:
                self.log.info("Checkpoint loaded: version=%d, step=%d",
                              info.version, agent.step_count)
            return success
        except Exception as e:
            self.log.error("Load checkpoint failed: %s", e)
            return False

    def load_latest(self, agent: Any) -> bool:
        """加载最新检查点。"""
        return self.load(agent, version=None)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点。"""
        return [{
            "version": ci.version,
            "timestamp": ci.timestamp,
            "step_count": ci.step_count,
            "size_bytes": ci.size_bytes,
            "checksum": ci.checksum[:8],
        } for ci in self._index]

    def get_latest_info(self) -> Optional[CheckpointInfo]:
        """获取最新检查点信息。"""
        if not self._index:
            return None
        return self._index[-1]

    def delete_checkpoint(self, version: int) -> bool:
        """删除指定版本检查点。"""
        for i, ci in enumerate(self._index):
            if ci.version == version:
                try:
                    if os.path.exists(ci.file_path):
                        os.remove(ci.file_path)
                    self._index.pop(i)
                    self._save_index()
                    self.log.info("Checkpoint version %d deleted", version)
                    return True
                except Exception as e:
                    self.log.error("Delete checkpoint failed: %s", e)
                    return False
        return False

    def clear_all(self) -> int:
        """清空所有检查点。返回删除数量。"""
        count = 0
        for ci in list(self._index):
            try:
                if os.path.exists(ci.file_path):
                    os.remove(ci.file_path)
                count += 1
            except Exception:
                pass
        self._index = []
        self._save_index()
        self.log.info("Cleared all checkpoints (%d deleted)", count)
        return count

    def _cleanup_old_versions(self) -> int:
        """清理旧版本（保留最近 max_versions 个）。"""
        if len(self._index) <= self.max_versions:
            return 0
        to_remove = len(self._index) - self.max_versions
        removed = 0
        for _ in range(to_remove):
            ci = self._index.pop(0)  # 删除最旧的
            try:
                if os.path.exists(ci.file_path):
                    os.remove(ci.file_path)
                removed += 1
            except Exception:
                pass
        if removed > 0:
            self.log.debug("Cleaned up %d old checkpoints", removed)
        return removed

    def _save_index(self) -> None:
        """保存索引文件。"""
        index_path = os.path.join(self.checkpoint_dir, "_index.json")
        index_data = [{
            "version": ci.version,
            "timestamp": ci.timestamp,
            "step_count": ci.step_count,
            "file_path": ci.file_path,
            "checksum": ci.checksum,
            "size_bytes": ci.size_bytes,
        } for ci in self._index]
        # 原子写入
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False)
            os.rename(tmp_path, index_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _load_index(self) -> None:
        """加载索引文件。"""
        index_path = os.path.join(self.checkpoint_dir, "_index.json")
        if not os.path.exists(index_path):
            return
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            self._index = []
            max_version = 0
            for item in index_data:
                # 只记录文件仍存在的检查点
                if os.path.exists(item["file_path"]):
                    info = CheckpointInfo(
                        version=item["version"],
                        timestamp=item["timestamp"],
                        step_count=item["step_count"],
                        file_path=item["file_path"],
                        checksum=item["checksum"],
                        size_bytes=item["size_bytes"],
                    )
                    self._index.append(info)
                    if info.version > max_version:
                        max_version = info.version
            self._version_counter = max_version
            self.log.info("Loaded %d checkpoints from index", len(self._index))
        except Exception as e:
            self.log.warning("Load index failed: %s", e)
            self._index = []

    def get_state(self) -> Dict[str, Any]:
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "version_count": len(self._index),
            "max_versions": self.max_versions,
            "latest_version": (self._index[-1].version if self._index else 0),
            "latest_step": (self._index[-1].step_count if self._index else 0),
        }
