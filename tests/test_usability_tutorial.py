"""阶段四测试：交互式教程模式。

测试范围：
1. TutorialStep 数据结构
2. Tutorial 数据结构与运行
3. 内置教程（quickstart/config/environment/debug）
4. 教程注册与查询
5. CLI tutorial 命令
6. 非交互模式运行
"""
import io
import sys
from contextlib import redirect_stdout

import pytest

from spepa.tutorial import (
    Tutorial, TutorialStep, list_tutorials, get_tutorial,
    run_tutorial, register_tutorial,
)
from spepa.cli import main


# ============================================================
# TutorialStep 测试
# ============================================================

class TestTutorialStep:
    """TutorialStep 数据结构测试。"""

    def test_create_step(self):
        """创建步骤。"""
        step = TutorialStep("标题", "概念讲解")
        assert step.title == "标题"
        assert step.concept == "概念讲解"
        assert step.demo is None
        assert step.tip == ""

    def test_create_step_with_demo(self):
        """创建带演示的步骤。"""
        def demo():
            return "演示结果"

        step = TutorialStep("标题", "概念", demo=demo, tip="提示")
        assert step.demo is not None
        assert step.tip == "提示"
        assert step.run_demo() == "演示结果"

    def test_run_demo_without_demo(self):
        """无演示函数时返回空字符串。"""
        step = TutorialStep("标题", "概念")
        assert step.run_demo() == ""

    def test_run_demo_handles_exception(self):
        """演示函数抛异常时返回错误信息。"""
        def bad_demo():
            raise RuntimeError("boom")

        step = TutorialStep("标题", "概念", demo=bad_demo)
        result = step.run_demo()
        assert "演示出错" in result
        assert "boom" in result


# ============================================================
# Tutorial 测试
# ============================================================

class TestTutorial:
    """Tutorial 数据结构测试。"""

    def test_create_tutorial(self):
        """创建教程。"""
        steps = [TutorialStep("步骤1", "概念1")]
        tutorial = Tutorial("test", "测试教程", "描述", steps)
        assert tutorial.name == "test"
        assert tutorial.title == "测试教程"
        assert tutorial.description == "描述"
        assert len(tutorial.steps) == 1

    def test_run_non_interactive(self):
        """非交互模式运行教程。"""
        steps = [
            TutorialStep("步骤1", "概念1", demo=lambda: "结果1"),
            TutorialStep("步骤2", "概念2", demo=lambda: "结果2"),
        ]
        tutorial = Tutorial("test", "测试", "描述", steps)
        output = io.StringIO()
        with redirect_stdout(output):
            tutorial.run(interactive=False)
        text = output.getvalue()
        assert "测试" in text
        assert "步骤1" in text
        assert "步骤2" in text
        assert "结果1" in text
        assert "结果2" in text
        assert "教程完成" in text

    def test_run_interactive_with_pause_fn(self):
        """交互模式使用自定义暂停函数。"""
        pause_count = [0]

        def pause():
            pause_count[0] += 1

        steps = [
            TutorialStep("步骤1", "概念1"),
            TutorialStep("步骤2", "概念2"),
        ]
        tutorial = Tutorial("test", "测试", "描述", steps)
        output = io.StringIO()
        with redirect_stdout(output):
            tutorial.run(interactive=True, pause_fn=pause)
        assert pause_count[0] == 2  # 每步暂停一次

    def test_run_interactive_eof_continues(self):
        """交互模式遇到 EOF 自动继续。"""
        steps = [TutorialStep("步骤1", "概念1")]
        tutorial = Tutorial("test", "测试", "描述", steps)
        # 模拟 input() 抛 EOFError
        old_input = __builtins__.get('input', input) if hasattr(__builtins__, 'get') else input
        try:
            import builtins
            builtins.input = lambda *a: (_ for _ in ()).throw(EOFError())
            output = io.StringIO()
            with redirect_stdout(output):
                tutorial.run(interactive=True)
            text = output.getvalue()
            assert "教程完成" in text
        finally:
            builtins.input = old_input


# ============================================================
# 内置教程测试
# ============================================================

class TestBuiltinTutorials:
    """内置教程测试。"""

    def test_list_tutorials(self):
        """列出所有教程。"""
        tutorials = list_tutorials()
        assert "quickstart" in tutorials
        assert "config" in tutorials
        assert "environment" in tutorials
        assert "debug" in tutorials

    def test_get_tutorial(self):
        """获取指定教程。"""
        tutorial = get_tutorial("quickstart")
        assert tutorial.name == "quickstart"
        assert len(tutorial.steps) >= 3

    def test_get_nonexistent_tutorial(self):
        """获取不存在的教程抛 KeyError。"""
        with pytest.raises(KeyError, match="不存在"):
            get_tutorial("nonexistent")

    def test_quickstart_tutorial_steps(self):
        """快速入门教程有 5 个步骤。"""
        tutorial = get_tutorial("quickstart")
        assert len(tutorial.steps) == 5
        assert "创建" in tutorial.steps[0].title or "Agent" in tutorial.steps[0].title

    def test_config_tutorial_steps(self):
        """配置教程有 3 个步骤。"""
        tutorial = get_tutorial("config")
        assert len(tutorial.steps) == 3

    def test_environment_tutorial_steps(self):
        """环境教程有 2 个步骤。"""
        tutorial = get_tutorial("environment")
        assert len(tutorial.steps) == 2

    def test_debug_tutorial_steps(self):
        """调试教程有 2 个步骤。"""
        tutorial = get_tutorial("debug")
        assert len(tutorial.steps) == 2

    def test_all_steps_have_concepts(self):
        """所有步骤都有概念讲解。"""
        tutorials = list_tutorials()
        for name in tutorials:
            tutorial = get_tutorial(name)
            for step in tutorial.steps:
                assert len(step.concept) > 0, f"{name}.{step.title} 缺少概念讲解"

    def test_all_demos_runnable(self):
        """所有演示函数都能正常运行。"""
        tutorials = list_tutorials()
        for name in tutorials:
            tutorial = get_tutorial(name)
            for step in tutorial.steps:
                if step.demo is not None:
                    result = step.run_demo()
                    assert isinstance(result, str), f"{name}.{step.title} 演示未返回字符串"
                    assert len(result) > 0, f"{name}.{step.title} 演示返回空字符串"


# ============================================================
# 教程注册测试
# ============================================================

class TestTutorialRegistration:
    """教程注册测试。"""

    def test_register_custom_tutorial(self):
        """注册自定义教程。"""
        steps = [TutorialStep("自定义步骤", "自定义概念")]
        tutorial = Tutorial("custom", "自定义教程", "自定义描述", steps)
        register_tutorial(tutorial)
        assert "custom" in list_tutorials()
        retrieved = get_tutorial("custom")
        assert retrieved.title == "自定义教程"

    def test_run_registered_tutorial(self):
        """运行已注册的教程。"""
        steps = [TutorialStep("步骤", "概念", demo=lambda: "OK")]
        tutorial = Tutorial("runtest", "运行测试", "描述", steps)
        register_tutorial(tutorial)
        output = io.StringIO()
        with redirect_stdout(output):
            run_tutorial("runtest", interactive=False)
        text = output.getvalue()
        assert "运行测试" in text
        assert "OK" in text


# ============================================================
# CLI tutorial 命令测试
# ============================================================

class TestCLITutorial:
    """CLI tutorial 命令测试。"""

    def test_tutorial_list(self, capsys):
        """列出所有教程。"""
        ret = main(["tutorial"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "可用教程" in captured.out
        assert "quickstart" in captured.out

    def test_tutorial_run_non_interactive(self, capsys):
        """非交互模式运行教程。"""
        ret = main(["tutorial", "quickstart", "--no-interactive"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "SPEPA" in captured.out
        assert "教程完成" in captured.out

    def test_tutorial_nonexistent(self, capsys):
        """运行不存在的教程返回错误。"""
        ret = main(["tutorial", "nonexistent", "--no-interactive"])
        assert ret == 1

    def test_tutorial_in_help(self, capsys):
        """tutorial 命令出现在帮助中。"""
        main([])
        captured = capsys.readouterr()
        assert "tutorial" in captured.out


# ============================================================
# 端到端测试
# ============================================================

class TestEndToEnd:
    """端到端测试。"""

    def test_run_all_tutorials_non_interactive(self):
        """非交互模式运行所有内置教程。"""
        tutorials = list_tutorials()
        for name in tutorials:
            output = io.StringIO()
            with redirect_stdout(output):
                run_tutorial(name, interactive=False)
            text = output.getvalue()
            assert "教程完成" in text, f"教程 {name} 未正常完成"

    def test_tutorial_demo_produces_real_output(self):
        """教程演示产生真实的运行结果。"""
        tutorial = get_tutorial("quickstart")
        # 步骤2的演示应该运行 Agent 并返回步数
        step2 = tutorial.steps[1]
        result = step2.run_demo()
        assert "total_steps" in result
        assert "5" in result  # 运行了5步
