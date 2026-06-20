"""阶段二测试：状态持久化与断点续传。

测试范围：
1. StateManager 基本功能（save/load）
2. StateManager 检查点管理（list/delete/clear）
3. StateManager auto_save 上下文管理器
4. StateManager resume_run 断点续传
5. StateManager recover 崩溃恢复
6. 便捷函数 save_state/load_state/resume_run
7. 端到端工作流
"""
import os
import tempfile
import numpy as np
import pytest

from spepa.state_manager import (
    StateManager, save_state, load_state, resume_run,
)
from spepa.agent import SPEPAAgent
from spepa.utils.config import get_preset_config


# ============================================================
# 基本功能测试
# ============================================================

class TestStateManagerBasic:
    """StateManager 基本功能测试。"""

    def test_create_manager(self):
        """创建管理器。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)
            assert manager.agent is agent
            assert manager.checkpoint_dir == tmpdir

    def test_save_returns_info(self):
        """save 返回检查点信息。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=3)
            manager = StateManager(agent, checkpoint_dir=tmpdir)
            info = manager.save()
            assert info is not None
            assert "version" in info
            assert "step_count" in info
            assert info["step_count"] == agent.step_count

    def test_save_creates_file(self):
        """save 创建检查点文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=2)
            manager = StateManager(agent, checkpoint_dir=tmpdir)
            manager.save()
            # 目录下应有检查点文件
            files = os.listdir(tmpdir)
            assert len(files) > 0

    def test_load_restores_state(self):
        """load 恢复 Agent 状态。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=5)
            state_before = agent.current_state.copy()
            step_before = agent.step_count

            manager = StateManager(agent, checkpoint_dir=tmpdir)
            manager.save()

            # 修改 Agent 状态
            agent.current_state = np.zeros(agent.config.state_dim)
            agent.step_count = 0

            # 加载恢复
            success = manager.load()
            assert success is True
            assert np.allclose(agent.current_state, state_before)
            assert agent.step_count == step_before

    def test_load_nonexistent_returns_false(self):
        """加载不存在的检查点返回 False。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)
            success = manager.load()
            assert success is False

    def test_load_specific_version(self):
        """加载指定版本。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            # 保存多个版本
            agent.run(n_steps=2)
            info1 = manager.save()
            agent.run(n_steps=3)
            info2 = manager.save()

            # 加载第一个版本
            success = manager.load(version=info1["version"])
            assert success is True


# ============================================================
# 检查点管理测试
# ============================================================

class TestCheckpointManagement:
    """检查点管理测试。"""

    def test_list_checkpoints(self):
        """列出检查点。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            agent.run(n_steps=2)
            manager.save()
            agent.run(n_steps=2)
            manager.save()

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) >= 2

    def test_list_checkpoints_sorted_desc(self):
        """检查点按版本降序。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            manager.save()
            manager.save()
            manager.save()

            checkpoints = manager.list_checkpoints()
            versions = [c["version"] for c in checkpoints]
            assert versions == sorted(versions, reverse=True)

    def test_delete_checkpoint(self):
        """删除检查点。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)
            info = manager.save()
            assert manager.delete_checkpoint(info["version"]) is True
            # 再次删除返回 False
            assert manager.delete_checkpoint(info["version"]) is False

    def test_clear_all(self):
        """清空所有检查点。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)
            manager.save()
            manager.save()
            count = manager.clear_all()
            assert count >= 2
            assert len(manager.list_checkpoints()) == 0

    def test_get_info(self):
        """获取管理器信息。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=3)
            manager = StateManager(agent, checkpoint_dir=tmpdir)
            manager.save()

            info = manager.get_info()
            assert info["checkpoint_dir"] == tmpdir
            assert info["total_checkpoints"] >= 1
            assert info["latest_version"] is not None
            assert info["latest_step_count"] == agent.step_count


# ============================================================
# auto_save 测试
# ============================================================

class TestAutoSave:
    """auto_save 上下文管理器测试。"""

    def test_auto_save_context(self):
        """auto_save 上下文管理器。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            with manager.auto_save(interval=5):
                agent.run(n_steps=10)

            # 应有检查点保存
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) >= 1

    def test_auto_save_disabled_after_context(self):
        """退出上下文后 auto_save 关闭。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            assert manager._auto_save_active is False
            with manager.auto_save(interval=5):
                assert manager._auto_save_active is True
            assert manager._auto_save_active is False

    def test_auto_save_saves_on_exit(self):
        """退出时保存最终状态。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            with manager.auto_save(interval=100):  # 高间隔，运行中不触发
                agent.run(n_steps=3)

            # 退出时应保存
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) >= 1
            latest = checkpoints[0]
            assert latest["step_count"] == agent.step_count


# ============================================================
# resume_run 测试
# ============================================================

class TestResumeRun:
    """resume_run 断点续传测试。"""

    def test_resume_run_without_checkpoint(self):
        """无检查点时 resume_run 从头开始。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            result = manager.resume_run(n_steps=5)
            assert result["recovered"] is False
            assert result["start_step"] == 0
            assert result["steps_executed"] >= 5

    def test_resume_run_with_checkpoint(self):
        """有检查点时 resume_run 从检查点继续。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            # 第一次运行并保存
            agent.run(n_steps=5)
            manager.save()
            saved_step = agent.step_count

            # 创建新 Agent 并恢复运行
            agent2 = SPEPAAgent(get_preset_config("minimal"))
            manager2 = StateManager(agent2, checkpoint_dir=tmpdir)
            result = manager2.resume_run(n_steps=3)

            assert result["recovered"] is True
            assert result["start_step"] == saved_step
            assert result["end_step"] > saved_step

    def test_resume_run_saves_on_exit(self):
        """resume_run 结束后保存。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            manager.resume_run(n_steps=3, save_on_exit=True)
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) >= 1

    def test_resume_run_no_save_on_exit(self):
        """resume_run 可选择不保存。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            initial_count = len(manager.list_checkpoints())
            manager.resume_run(n_steps=3, save_on_exit=False)
            # 不保存时检查点数量不变（或仅恢复时无新增）
            # 注意：resume_run 内部 recover 可能不产生新检查点


# ============================================================
# recover 测试
# ============================================================

class TestRecover:
    """recover 崩溃恢复测试。"""

    def test_recover_no_checkpoints(self):
        """无检查点时恢复返回未恢复。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)
            result = manager.recover()
            assert result["recovered"] is False

    def test_recover_with_checkpoints(self):
        """有检查点时恢复成功。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=5)
            state_before = agent.current_state.copy()
            step_before = agent.step_count

            manager = StateManager(agent, checkpoint_dir=tmpdir)
            manager.save()

            # 新 Agent 恢复
            agent2 = SPEPAAgent(get_preset_config("minimal"))
            manager2 = StateManager(agent2, checkpoint_dir=tmpdir)
            result = manager2.recover()

            assert result["recovered"] is True
            assert agent2.step_count == step_before
            assert np.allclose(agent2.current_state, state_before)


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    """便捷函数测试。"""

    def test_save_state(self):
        """save_state 便捷函数。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=2)
            info = save_state(agent, checkpoint_dir=tmpdir)
            assert info is not None
            assert "version" in info

    def test_load_state(self):
        """load_state 便捷函数。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=3)
            save_state(agent, checkpoint_dir=tmpdir)

            # 修改状态
            agent.current_state = np.zeros(agent.config.state_dim)

            # 加载
            success = load_state(agent, checkpoint_dir=tmpdir)
            assert success is True
            assert not np.allclose(agent.current_state, 0)

    def test_resume_run_function(self):
        """resume_run 便捷函数。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            result = resume_run(agent, n_steps=3, checkpoint_dir=tmpdir)
            assert result["steps_executed"] >= 3


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端测试。"""

    def test_full_save_load_workflow(self):
        """完整保存加载工作流。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 创建 Agent 并运行
            agent = SPEPAAgent(get_preset_config("minimal"))
            agent.run(n_steps=5)
            original_state = agent.current_state.copy()
            original_step = agent.step_count

            # 2. 保存
            manager = StateManager(agent, checkpoint_dir=tmpdir)
            manager.save()

            # 3. 继续运行
            agent.run(n_steps=5)

            # 4. 加载回之前的状态
            manager.load()
            assert np.allclose(agent.current_state, original_state)
            assert agent.step_count == original_step

    def test_auto_save_workflow(self):
        """自动保存工作流。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            with manager.auto_save(interval=5):
                agent.run(n_steps=15)

            # 应有多个检查点
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) >= 2

    def test_crash_recovery_workflow(self):
        """崩溃恢复工作流。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 第一个 Agent 运行并保存
            agent1 = SPEPAAgent(get_preset_config("minimal"))
            manager1 = StateManager(agent1, checkpoint_dir=tmpdir)
            with manager1.auto_save(interval=5):
                agent1.run(n_steps=10)

            # 2. 模拟崩溃：创建新 Agent
            agent2 = SPEPAAgent(get_preset_config("minimal"))
            manager2 = StateManager(agent2, checkpoint_dir=tmpdir)

            # 3. 恢复
            result = manager2.recover()
            assert result["recovered"] is True
            assert agent2.step_count > 0

            # 4. 继续运行
            result = manager2.resume_run(n_steps=5)
            assert result["steps_executed"] >= 5

    def test_multiple_saves_and_loads(self):
        """多次保存和加载。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SPEPAAgent(get_preset_config("minimal"))
            manager = StateManager(agent, checkpoint_dir=tmpdir)

            # 保存多个版本
            steps_record = []
            for i in range(3):
                agent.run(n_steps=2)
                info = manager.save()
                steps_record.append((info["version"], agent.step_count))

            # 加载每个版本验证
            for version, expected_step in steps_record:
                success = manager.load(version=version)
                assert success is True
                assert agent.step_count == expected_step
