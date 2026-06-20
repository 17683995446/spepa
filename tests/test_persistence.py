"""SPEPA 阶段二测试：状态持久化与崩溃恢复。

测试范围：
1. StateSerializer：序列化/反序列化的完整性与一致性
2. CheckpointManager：原子写入、版本管理、校验和、清理
3. CrashRecovery：启动恢复、渐进回退、周期性检查点
4. SPEPAAgent 持久化API：save_state/load_state/serialize_state
5. PerpetualRunner 集成：检查点目录透传、启动恢复
6. 端到端：崩溃后断点续跑
"""
import os
import json
import time
import tempfile
import shutil
import numpy as np
import pytest
from collections import deque

from spepa.utils.config import get_default_config, SPEPAConfig
from spepa.agent import SPEPAAgent
from spepa.persistence.serializer import StateSerializer
from spepa.persistence.checkpoint import CheckpointManager, CheckpointInfo
from spepa.persistence.recovery import CrashRecovery


# ============================================================
# 辅助函数
# ============================================================

def make_agent(n_cycles=3):
    """创建并运行若干周期的 Agent。"""
    config = get_default_config()
    agent = SPEPAAgent(config)
    for _ in range(n_cycles):
        agent.online_cycle()
    return agent


# ============================================================
# StateSerializer 测试
# ============================================================

class TestStateSerializer:
    """状态序列化器测试。"""

    def test_serializer_initialization(self):
        """序列化器能正确初始化。"""
        config = get_default_config()
        serializer = StateSerializer(config)
        assert serializer.config is config

    def test_serialize_agent_returns_dict(self):
        """serialize_agent 返回字典。"""
        agent = make_agent(2)
        config = agent.config
        serializer = StateSerializer(config)
        state = serializer.serialize_agent(agent)
        assert isinstance(state, dict)

    def test_serialized_state_has_all_keys(self):
        """序列化状态包含所有必需的键。"""
        agent = make_agent(1)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        required_keys = {
            "version", "config", "runtime", "predictive_model",
            "memory_system", "skill_library", "optimizer",
            "drive_engine", "goal_hierarchy", "metacognition",
        }
        assert required_keys.issubset(set(state.keys()))

    def test_serialized_state_is_json_compatible(self):
        """序列化状态可转为 JSON（无 numpy 对象）。"""
        agent = make_agent(2)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        # 应能成功 dump 为 JSON
        json_str = json.dumps(state, ensure_ascii=False, default=str)
        assert isinstance(json_str, str)
        # 应能成功 load 回来
        loaded = json.loads(json_str)
        assert loaded["version"] == state["version"]

    def test_serialize_predictive_model_layers(self):
        """预测模型各层 mu/var/W/b 被正确序列化。"""
        agent = make_agent(1)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        layers = state["predictive_model"]["layers"]
        assert len(layers) == len(agent.predictive_model.layers)
        for layer_state in layers:
            assert "mu" in layer_state
            assert "var" in layer_state
            assert "W" in layer_state
            assert "b" in layer_state

    def test_serialize_runtime_state(self):
        """运行时状态被正确序列化。"""
        agent = make_agent(3)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        runtime = state["runtime"]
        assert runtime["step_count"] == agent.step_count
        assert runtime["online_cycles"] == agent.online_cycles
        assert len(runtime["current_state"]) == agent.config.state_dim

    def test_serialize_drive_engine(self):
        """驱力引擎内稳态被正确序列化。"""
        agent = make_agent(2)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        de_state = state["drive_engine"]
        assert "homeostasis_values" in de_state
        assert "homeostasis_target" in de_state
        assert len(de_state["homeostasis_values"]) == agent.config.state_dim

    def test_serialize_goal_hierarchy(self):
        """目标层次被正确序列化。"""
        agent = make_agent(1)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        gh_state = state["goal_hierarchy"]
        assert "goals" in gh_state
        # 应有5个层级
        assert len(gh_state["goals"]) == 5

    def test_serialize_metacognition(self):
        """元认知被正确序列化。"""
        agent = make_agent(2)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        mc_state = state["metacognition"]
        assert "current_confidence" in mc_state
        assert "meta_control_signal" in mc_state
        assert "calibration_offset" in mc_state

    def test_deserialize_agent_restores_step_count(self):
        """反序列化恢复 step_count。"""
        agent = make_agent(5)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        agent2 = SPEPAAgent(agent.config)
        ok = serializer.deserialize_agent(agent2, state)
        assert ok is True
        assert agent2.step_count == agent.step_count

    def test_deserialize_agent_restores_current_state(self):
        """反序列化恢复 current_state。"""
        agent = make_agent(3)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        agent2 = SPEPAAgent(agent.config)
        serializer.deserialize_agent(agent2, state)
        assert np.allclose(agent.current_state, agent2.current_state)

    def test_deserialize_agent_restores_predictive_model(self):
        """反序列化恢复预测模型权重。"""
        agent = make_agent(3)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        agent2 = SPEPAAgent(agent.config)
        serializer.deserialize_agent(agent2, state)
        for l1, l2 in zip(agent.predictive_model.layers,
                          agent2.predictive_model.layers):
            assert np.allclose(l1.mu, l2.mu)
            assert np.allclose(l1.W, l2.W)
            assert np.allclose(l1.var, l2.var)
            assert np.allclose(l1.b, l2.b)

    def test_deserialize_agent_restores_homeostasis(self):
        """反序列化恢复内稳态值。"""
        agent = make_agent(3)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        agent2 = SPEPAAgent(agent.config)
        serializer.deserialize_agent(agent2, state)
        assert np.allclose(agent.drive_engine.homeostasis.values,
                           agent2.drive_engine.homeostasis.values)

    def test_deserialize_agent_restores_goals(self):
        """反序列化恢复目标层次。"""
        agent = make_agent(2)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        agent2 = SPEPAAgent(agent.config)
        serializer.deserialize_agent(agent2, state)
        for level in range(5):
            goals1 = agent.goal_hierarchy.goals[level]
            goals2 = agent2.goal_hierarchy.goals[level]
            assert len(goals1) == len(goals2)
            for g1, g2 in zip(goals1, goals2):
                assert g1.name == g2.name
                assert g1.level == g2.level
                assert np.allclose(g1.target, g2.target)

    def test_deserialize_agent_restores_metacognition(self):
        """反序列化恢复元认知状态。"""
        agent = make_agent(3)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        agent2 = SPEPAAgent(agent.config)
        serializer.deserialize_agent(agent2, state)
        assert abs(agent.metacognition.current_confidence -
                   agent2.metacognition.current_confidence) < 1e-9
        assert abs(agent.metacognition.meta_control_signal -
                   agent2.metacognition.meta_control_signal) < 1e-9

    def test_deserialize_agent_restores_optimizer(self):
        """反序列化恢复优化器状态。"""
        agent = make_agent(3)
        serializer = StateSerializer(agent.config)
        state = serializer.serialize_agent(agent)
        agent2 = SPEPAAgent(agent.config)
        serializer.deserialize_agent(agent2, state)
        assert abs(agent.optimizer._adaptive_lr -
                   agent2.optimizer._adaptive_lr) < 1e-9
        assert agent.optimizer._high_error_streak == agent2.optimizer._high_error_streak

    def test_roundtrip_idempotent(self):
        """序列化→反序列化→序列化应产生相同结果（幂等性）。"""
        agent = make_agent(3)
        serializer = StateSerializer(agent.config)
        state1 = serializer.serialize_agent(agent)
        agent2 = SPEPAAgent(agent.config)
        serializer.deserialize_agent(agent2, state1)
        state2 = serializer.serialize_agent(agent2)
        # 比较 JSON 字符串（忽略可能差异的运行时时间戳等）
        assert state1["runtime"]["step_count"] == state2["runtime"]["step_count"]
        assert np.allclose(state1["runtime"]["current_state"],
                           state2["runtime"]["current_state"])
        assert state1["predictive_model"] == state2["predictive_model"]

    def test_deserialize_handles_empty_state(self):
        """反序列化空状态不崩溃。"""
        agent = SPEPAAgent(get_default_config())
        serializer = StateSerializer(agent.config)
        ok = serializer.deserialize_agent(agent, {})
        assert ok is True

    def test_deserialize_returns_false_on_error(self):
        """反序列化出错时返回 False。"""
        agent = SPEPAAgent(get_default_config())
        serializer = StateSerializer(agent.config)
        # 传入无效状态（运行时部分会尝试访问不存在的属性）
        bad_state = {"runtime": {"current_state": "not_a_list"}}
        ok = serializer.deserialize_agent(agent, bad_state)
        assert ok is False


# ============================================================
# CheckpointManager 测试
# ============================================================

class TestCheckpointManager:
    """检查点管理器测试。"""

    def test_checkpoint_manager_initialization(self):
        """检查点管理器能正确初始化并创建目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "checkpoints")
            mgr = CheckpointManager(get_default_config(), ckpt_dir)
            assert os.path.isdir(ckpt_dir)
            assert mgr.max_versions == 5

    def test_save_creates_checkpoint_file(self):
        """save 创建检查点文件。"""
        agent = make_agent(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            info = mgr.save(agent)
            assert info is not None
            assert os.path.exists(info.file_path)
            assert info.version == 1
            assert info.step_count == agent.step_count
            assert info.size_bytes > 0

    def test_save_returns_checkpoint_info(self):
        """save 返回 CheckpointInfo 对象。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            info = mgr.save(agent)
            assert isinstance(info, CheckpointInfo)
            assert info.checksum  # 非空校验和

    def test_load_restores_state(self):
        """load 恢复 Agent 状态。"""
        agent = make_agent(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            mgr.save(agent)
            agent2 = SPEPAAgent(agent.config)
            ok = mgr.load_latest(agent2)
            assert ok is True
            assert agent2.step_count == agent.step_count
            assert np.allclose(agent.current_state, agent2.current_state)

    def test_load_specific_version(self):
        """load 指定版本。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            info1 = mgr.save(agent)
            agent.online_cycle()
            mgr.save(agent)
            # 加载第一个版本
            agent2 = SPEPAAgent(agent.config)
            ok = mgr.load(agent2, version=info1.version)
            assert ok is True
            assert agent2.step_count == 1

    def test_load_nonexistent_version_returns_false(self):
        """加载不存在的版本返回 False。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            mgr.save(agent)
            agent2 = SPEPAAgent(agent.config)
            ok = mgr.load(agent2, version=999)
            assert ok is False

    def test_load_with_no_checkpoints_returns_false(self):
        """无检查点时 load 返回 False。"""
        agent = SPEPAAgent(get_default_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            ok = mgr.load_latest(agent)
            assert ok is False

    def test_list_checkpoints(self):
        """list_checkpoints 返回所有检查点。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            mgr.save(agent)
            agent.online_cycle()
            mgr.save(agent)
            cps = mgr.list_checkpoints()
            assert len(cps) == 2
            for cp in cps:
                assert "version" in cp
                assert "timestamp" in cp
                assert "step_count" in cp

    def test_delete_checkpoint(self):
        """delete_checkpoint 删除指定版本。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            info = mgr.save(agent)
            assert len(mgr.list_checkpoints()) == 1
            ok = mgr.delete_checkpoint(info.version)
            assert ok is True
            assert len(mgr.list_checkpoints()) == 0

    def test_delete_nonexistent_checkpoint(self):
        """删除不存在的版本返回 False。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(get_default_config(), tmpdir)
            ok = mgr.delete_checkpoint(999)
            assert ok is False

    def test_clear_all(self):
        """clear_all 清空所有检查点。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            mgr.save(agent)
            agent.online_cycle()
            mgr.save(agent)
            count = mgr.clear_all()
            assert count == 2
            assert len(mgr.list_checkpoints()) == 0

    def test_version_management(self):
        """版本号递增。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            v1 = mgr.save(agent).version
            v2 = mgr.save(agent).version
            v3 = mgr.save(agent).version
            assert v1 == 1
            assert v2 == 2
            assert v3 == 3

    def test_cleanup_old_versions(self):
        """超过 max_versions 时清理旧版本。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            mgr.max_versions = 3
            for i in range(5):
                mgr.save(agent)
            cps = mgr.list_checkpoints()
            assert len(cps) == 3
            # 应保留最近3个（版本3,4,5）
            versions = sorted([cp["version"] for cp in cps])
            assert versions == [3, 4, 5]

    def test_checksum_verification(self):
        """校验和验证：篡改文件后加载失败。"""
        agent = make_agent(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            info = mgr.save(agent)
            # 篡改检查点文件
            with open(info.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            tampered = content.replace('"step_count": 2', '"step_count": 999')
            with open(info.file_path, "w", encoding="utf-8") as f:
                f.write(tampered)
            agent2 = SPEPAAgent(agent.config)
            ok = mgr.load_latest(agent2)
            assert ok is False  # 校验和不匹配

    def test_index_persistence(self):
        """索引文件持久化：重新创建管理器后能加载已有检查点。"""
        agent = make_agent(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr1 = CheckpointManager(agent.config, tmpdir)
            mgr1.save(agent)
            agent.online_cycle()
            mgr1.save(agent)
            # 重新创建管理器
            mgr2 = CheckpointManager(agent.config, tmpdir)
            cps = mgr2.list_checkpoints()
            assert len(cps) == 2
            # 版本号应继续递增
            info = mgr2.save(agent)
            assert info.version == 3

    def test_atomic_write_no_temp_files_left(self):
        """原子写入后不残留临时文件。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            mgr.save(agent)
            files = os.listdir(tmpdir)
            tmp_files = [f for f in files if f.endswith(".tmp")]
            assert len(tmp_files) == 0

    def test_get_state(self):
        """get_state 返回管理器状态。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(agent.config, tmpdir)
            mgr.save(agent)
            state = mgr.get_state()
            assert state["version_count"] == 1
            assert state["latest_version"] == 1
            assert state["latest_step"] == agent.step_count


# ============================================================
# CrashRecovery 测试
# ============================================================

class TestCrashRecovery:
    """崩溃恢复机制测试。"""

    def test_recovery_initialization(self):
        """恢复器能正确初始化。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(get_default_config(), tmpdir)
            assert recovery.recovery_attempts == 0
            assert recovery.recovery_successes == 0

    def test_recover_with_no_checkpoints(self):
        """无检查点时返回未恢复。"""
        agent = SPEPAAgent(get_default_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(get_default_config(), tmpdir)
            result = recovery.recover_on_startup(agent)
            assert result["recovered"] is False
            assert result["reason"] == "no_checkpoints"

    def test_recover_from_latest_checkpoint(self):
        """从最新检查点恢复。"""
        agent = make_agent(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            recovery.periodic_checkpoint(agent, force=True)
            agent2 = SPEPAAgent(agent.config)
            result = recovery.recover_on_startup(agent2)
            assert result["recovered"] is True
            assert result["step_count"] == agent.step_count
            assert np.allclose(agent.current_state, agent2.current_state)

    def test_progressive_fallback(self):
        """渐进回退：最新检查点损坏时回退到次新。"""
        agent = make_agent(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            info1 = recovery.checkpoint_mgr.save(agent)
            agent.online_cycle()
            info2 = recovery.checkpoint_mgr.save(agent)
            # 篡改最新检查点
            with open(info2.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            with open(info2.file_path, "w", encoding="utf-8") as f:
                f.write(content + "tampered")
            # 应能从次新检查点恢复
            agent2 = SPEPAAgent(agent.config)
            result = recovery.recover_on_startup(agent2)
            assert result["recovered"] is True
            assert result["version"] == info1.version

    def test_all_recovery_attempts_failed(self):
        """所有恢复尝试失败时返回未恢复。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            info = recovery.checkpoint_mgr.save(agent)
            # 篡改检查点使校验和失败
            with open(info.file_path, "w", encoding="utf-8") as f:
                f.write("corrupted")
            agent2 = SPEPAAgent(agent.config)
            result = recovery.recover_on_startup(agent2)
            assert result["recovered"] is False

    def test_periodic_checkpoint_respects_interval(self):
        """periodic_checkpoint 遵守间隔。"""
        agent = make_agent(1)  # step_count=1
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            # step_count=1, interval=50, 不应保存
            result = recovery.periodic_checkpoint(agent, interval=50)
            assert result is None

    def test_periodic_checkpoint_force(self):
        """periodic_checkpoint force=True 强制保存。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            result = recovery.periodic_checkpoint(agent, force=True)
            assert result is not None
            assert result["saved"] is True

    def test_periodic_checkpoint_at_interval(self):
        """周期性检查点在间隔点保存。"""
        agent = make_agent(50)  # step_count=50
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            result = recovery.periodic_checkpoint(agent, interval=50)
            assert result is not None
            assert result["saved"] is True

    def test_periodic_checkpoint_skips_step_zero(self):
        """step_count=0 时不保存（非强制）。"""
        agent = SPEPAAgent(get_default_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            result = recovery.periodic_checkpoint(agent, interval=1)
            assert result is None

    def test_verify_agent_state_valid(self):
        """验证有效 Agent 状态。"""
        agent = make_agent(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            assert recovery._verify_agent_state(agent) is True

    def test_verify_agent_state_non_finite(self):
        """验证含非有限值的状态失败。"""
        agent = make_agent(1)
        agent.current_state[0] = np.nan
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            assert recovery._verify_agent_state(agent) is False

    def test_recovery_stats_updated(self):
        """恢复统计被更新。"""
        agent = make_agent(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(agent.config, tmpdir)
            recovery.periodic_checkpoint(agent, force=True)
            agent2 = SPEPAAgent(agent.config)
            recovery.recover_on_startup(agent2)
            assert recovery.recovery_attempts == 1
            assert recovery.recovery_successes == 1

    def test_get_state(self):
        """get_state 返回恢复器状态。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = CrashRecovery(get_default_config(), tmpdir)
            state = recovery.get_state()
            assert "recovery_attempts" in state
            assert "checkpoint_state" in state


# ============================================================
# SPEPAAgent 持久化API 测试
# ============================================================

class TestAgentPersistenceAPI:
    """Agent 持久化 API 测试。"""

    def test_save_state_returns_info(self):
        """save_state 返回保存信息。"""
        agent = make_agent(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            info = agent.save_state(tmpdir)
            assert info is not None
            assert info["step_count"] == agent.step_count
            assert info["version"] == 1

    def test_load_state_restores(self):
        """load_state 恢复状态。"""
        agent = make_agent(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save_state(tmpdir)
            agent2 = SPEPAAgent(agent.config)
            ok = agent2.load_state(tmpdir)
            assert ok is True
            assert agent2.step_count == agent.step_count

    def test_load_state_specific_version(self):
        """load_state 加载指定版本。"""
        agent = make_agent(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save_state(tmpdir)
            agent.online_cycle()
            agent.save_state(tmpdir)
            ok = agent.load_state(tmpdir, version=1)
            assert ok is True
            assert agent.step_count == 1

    def test_create_checkpoint_manager(self):
        """create_checkpoint_manager 返回管理器。"""
        agent = SPEPAAgent(get_default_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = agent.create_checkpoint_manager(tmpdir)
            assert isinstance(mgr, CheckpointManager)

    def test_create_recovery(self):
        """create_recovery 返回恢复器。"""
        agent = SPEPAAgent(get_default_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery = agent.create_recovery(tmpdir)
            assert isinstance(recovery, CrashRecovery)

    def test_serialize_state_returns_dict(self):
        """serialize_state 返回字典。"""
        agent = make_agent(2)
        state = agent.serialize_state()
        assert isinstance(state, dict)
        assert "predictive_model" in state

    def test_deserialize_state_restores(self):
        """deserialize_state 恢复状态。"""
        agent = make_agent(3)
        state = agent.serialize_state()
        agent2 = SPEPAAgent(agent.config)
        ok = agent2.deserialize_state(state)
        assert ok is True
        assert agent2.step_count == agent.step_count

    def test_save_load_roundtrip(self):
        """保存→加载完整往返。"""
        agent = make_agent(5)
        original_state = agent.current_state.copy()
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save_state(tmpdir)
            agent.reset()
            assert agent.step_count == 0
            ok = agent.load_state(tmpdir)
            assert ok is True
            assert agent.step_count == 5
            assert np.allclose(agent.current_state, original_state)


# ============================================================
# PerpetualRunner 集成测试
# ============================================================

class TestPerpetualRunnerIntegration:
    """PerpetualRunner 与持久化集成测试。"""

    def test_run_with_checkpoint_dir(self):
        """带 checkpoint_dir 运行时保存检查点。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = agent.run_perpetual(
                max_cycles=60,  # 超过 checkpoint_interval=50
                checkpoint_dir=tmpdir,
                auto_recover=False,
            )
            assert summary["total_cycles"] == 60
            # 应有检查点文件
            files = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint_")]
            assert len(files) >= 1

    def test_run_with_auto_recover(self):
        """auto_recover=True 时启动恢复。"""
        config = get_default_config()
        # 第一次运行：保存检查点
        with tempfile.TemporaryDirectory() as tmpdir:
            agent1 = SPEPAAgent(config)
            agent1.run_perpetual(
                max_cycles=55,
                checkpoint_dir=tmpdir,
                auto_recover=True,
            )
            # 第二次运行：应能从检查点恢复
            agent2 = SPEPAAgent(config)
            initial_step = agent2.step_count
            agent2.run_perpetual(
                max_cycles=5,
                checkpoint_dir=tmpdir,
                auto_recover=True,
            )
            # 恢复后 step_count 应大于初始值
            assert agent2.step_count > initial_step

    def test_run_without_checkpoint_dir(self):
        """不带 checkpoint_dir 时不持久化。"""
        config = get_default_config()
        agent = SPEPAAgent(config)
        summary = agent.run_perpetual(max_cycles=5)
        assert summary["total_cycles"] == 5


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端场景测试。"""

    def test_crash_recovery_scenario(self):
        """崩溃恢复场景：运行→保存→"崩溃"→重启→恢复→继续。"""
        config = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            # 阶段1：运行并保存
            agent1 = SPEPAAgent(config)
            for _ in range(10):
                agent1.online_cycle()
            agent1.save_state(tmpdir)
            saved_state = agent1.current_state.copy()
            saved_step = agent1.step_count

            # 阶段2：模拟崩溃（丢弃 agent1）
            del agent1

            # 阶段3：重启并恢复
            agent2 = SPEPAAgent(config)
            recovery = agent2.create_recovery(tmpdir)
            result = recovery.recover_on_startup(agent2)
            assert result["recovered"] is True
            assert agent2.step_count == saved_step
            assert np.allclose(agent2.current_state, saved_state)

            # 阶段4：继续运行
            for _ in range(5):
                agent2.online_cycle()
            assert agent2.step_count == saved_step + 5

    def test_multiple_checkpoints_versioning(self):
        """多版本检查点管理。"""
        config = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(config, tmpdir)
            agent = SPEPAAgent(config)
            # 保存多个版本
            step_counts = []
            for i in range(7):
                agent.online_cycle()
                mgr.save(agent)
                step_counts.append(agent.step_count)
            # 应只保留最近5个
            cps = mgr.list_checkpoints()
            assert len(cps) == 5
            # 最早两个应被清理
            versions = sorted([cp["version"] for cp in cps])
            assert versions == [3, 4, 5, 6, 7]

    def test_learning_progress_preserved(self):
        """学习成果（预测模型权重）被保留。"""
        config = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            agent1 = SPEPAAgent(config)
            for _ in range(5):
                agent1.online_cycle()
            # 记录学习后的权重
            original_W = [l.W.copy() for l in agent1.predictive_model.layers]
            original_mu = [l.mu.copy() for l in agent1.predictive_model.layers]
            agent1.save_state(tmpdir)

            # 恢复
            agent2 = SPEPAAgent(config)
            agent2.load_state(tmpdir)
            for i, layer in enumerate(agent2.predictive_model.layers):
                assert np.allclose(layer.W, original_W[i])
                assert np.allclose(layer.mu, original_mu[i])

    def test_skill_library_preserved(self):
        """技能库被保留。"""
        config = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            agent1 = SPEPAAgent(config)
            # 手动添加一些技能
            from spepa.skills.skill import Skill
            skill = Skill(
                name="test_skill",
                trigger_state=np.ones(config.state_dim) * 0.5,
                action=np.ones(config.action_dim) * 0.3,
                expected_effect=np.ones(config.state_dim) * 0.1,
                success_count=5,
                failure_count=1,
            )
            agent1.skill_learner.library.add(skill)
            agent1.save_state(tmpdir)

            agent2 = SPEPAAgent(config)
            agent2.load_state(tmpdir)
            assert "test_skill" in agent2.skill_learner.library.skills
            restored = agent2.skill_learner.library.skills["test_skill"]
            assert restored.success_count == 5
            assert restored.failure_count == 1

    def test_memory_preserved(self):
        """记忆系统被保留。"""
        config = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            agent1 = SPEPAAgent(config)
            # 运行若干周期产生记忆
            for _ in range(3):
                agent1.online_cycle()
            fast_count = len(agent1.memory_system.fast.items)
            agent1.save_state(tmpdir)

            agent2 = SPEPAAgent(config)
            agent2.load_state(tmpdir)
            assert len(agent2.memory_system.fast.items) == fast_count
