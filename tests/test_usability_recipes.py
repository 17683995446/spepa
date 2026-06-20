"""SPEPA 示例库（Recipes）测试。

测试范围：
- Recipe 数据结构
- RecipeLibrary（加载、添加、查询、运行、格式化）
- 便捷函数（get_recipe/list_recipes/run_recipe 等）
- 端到端
"""
import pytest
from spepa.recipes import (
    Recipe, RecipeLibrary,
    get_recipe, list_recipes, run_recipe, format_recipe, format_recipe_list,
    add_recipe, list_recipe_tags, list_recipe_difficulties,
)


# ============================================================
# Recipe 数据结构
# ============================================================

class TestRecipe:
    """Recipe 数据结构测试。"""

    def test_create_recipe(self):
        recipe = Recipe(
            name="test",
            title="测试示例",
            description="测试用",
            code="print('hello')",
        )
        assert recipe.name == "test"
        assert recipe.title == "测试示例"
        assert recipe.code == "print('hello')"
        assert recipe.difficulty == "beginner"
        assert recipe.tags == []
        assert recipe.outputs == []

    def test_recipe_with_all_fields(self):
        recipe = Recipe(
            name="advanced_test",
            title="高级测试",
            description="高级测试示例",
            code="x = 1",
            difficulty="advanced",
            tags=["math", "test"],
            outputs=["1"],
        )
        assert recipe.difficulty == "advanced"
        assert recipe.tags == ["math", "test"]
        assert recipe.outputs == ["1"]


# ============================================================
# RecipeLibrary - 基础功能
# ============================================================

class TestLibraryBasic:
    """示例库基础功能测试。"""

    def test_create_library(self):
        lib = RecipeLibrary()
        assert len(lib.list()) > 0

    def test_library_has_builtins(self):
        lib = RecipeLibrary()
        names = lib.list_names()
        assert "quick_start" in names
        assert "custom_config" in names

    def test_get_existing_recipe(self):
        lib = RecipeLibrary()
        recipe = lib.get("quick_start")
        assert recipe is not None
        assert recipe.name == "quick_start"

    def test_get_nonexistent_recipe(self):
        lib = RecipeLibrary()
        recipe = lib.get("nonexistent_xyz")
        assert recipe is None

    def test_add_custom_recipe(self):
        lib = RecipeLibrary()
        recipe = Recipe(name="custom", title="自定义", description="自定义", code="x=1")
        lib.add(recipe)
        assert lib.get("custom") is not None

    def test_add_duplicate_raises(self):
        lib = RecipeLibrary()
        recipe = Recipe(name="quick_start", title="重复", description="重复", code="x=1")
        with pytest.raises(ValueError, match="已存在"):
            lib.add(recipe)


# ============================================================
# RecipeLibrary - 查询与筛选
# ============================================================

class TestLibraryQuery:
    """示例库查询测试。"""

    def test_list_all(self):
        lib = RecipeLibrary()
        recipes = lib.list()
        assert len(recipes) > 0

    def test_list_by_difficulty(self):
        lib = RecipeLibrary()
        beginners = lib.list(difficulty="beginner")
        for r in beginners:
            assert r.difficulty == "beginner"

    def test_list_by_tag(self):
        lib = RecipeLibrary()
        tagged = lib.list(tag="config")
        for r in tagged:
            assert "config" in r.tags

    def test_list_sorted_by_name(self):
        lib = RecipeLibrary()
        recipes = lib.list()
        names = [r.name for r in recipes]
        assert names == sorted(names)

    def test_list_names(self):
        lib = RecipeLibrary()
        names = lib.list_names()
        assert isinstance(names, list)
        assert "quick_start" in names
        assert names == sorted(names)

    def test_list_difficulties(self):
        lib = RecipeLibrary()
        difficulties = lib.list_difficulties()
        assert "beginner" in difficulties
        assert "intermediate" in difficulties
        assert "advanced" in difficulties

    def test_list_tags(self):
        lib = RecipeLibrary()
        tags = lib.list_tags()
        assert isinstance(tags, list)
        assert len(tags) > 0
        assert tags == sorted(tags)


# ============================================================
# RecipeLibrary - 运行
# ============================================================

class TestLibraryRun:
    """示例库运行测试。"""

    def test_run_simple_recipe(self):
        lib = RecipeLibrary()
        # quick_start 是最简单的可运行示例
        result = lib.run("quick_start")
        assert result["success"] is True
        assert result["error"] is None

    def test_run_nonexistent_raises(self):
        lib = RecipeLibrary()
        with pytest.raises(ValueError, match="不存在"):
            lib.run("nonexistent_xyz")

    def test_run_with_custom_globals(self):
        lib = RecipeLibrary()
        globals_dict = {"__builtins__": __builtins__}
        result = lib.run("quick_start", globals_dict=globals_dict)
        assert result["success"] is True

    def test_run_failing_code(self):
        """运行会失败的代码应返回错误信息而非抛异常。"""
        lib = RecipeLibrary()
        lib.add(Recipe(
            name="fail_test",
            title="失败测试",
            description="故意失败",
            code="raise ValueError('故意出错')",
        ))
        result = lib.run("fail_test")
        assert result["success"] is False
        assert "故意出错" in result["error"]

    def test_run_code_defines_variables(self):
        """运行的代码可以在 globals 中定义变量。"""
        lib = RecipeLibrary()
        lib.add(Recipe(
            name="var_test",
            title="变量测试",
            description="定义变量",
            code="x = 42\ny = 'hello'",
        ))
        globals_dict = {"__builtins__": __builtins__}
        lib.run("var_test", globals_dict=globals_dict)
        assert globals_dict["x"] == 42
        assert globals_dict["y"] == "hello"


# ============================================================
# RecipeLibrary - 格式化
# ============================================================

class TestLibraryFormat:
    """示例库格式化测试。"""

    def test_format_recipe(self):
        lib = RecipeLibrary()
        text = lib.format("quick_start")
        assert "示例" in text
        assert "快速开始" in text
        assert "代码" in text
        assert "```python" in text

    def test_format_recipe_without_code(self):
        lib = RecipeLibrary()
        text = lib.format("quick_start", with_code=False)
        assert "快速开始" in text
        assert "```python" not in text

    def test_format_nonexistent_recipe(self):
        lib = RecipeLibrary()
        text = lib.format("nonexistent")
        assert "未找到" in text

    def test_format_list_all(self):
        lib = RecipeLibrary()
        text = lib.format_list()
        assert "SPEPA 示例库" in text
        assert "quick_start" in text

    def test_format_list_contains_count(self):
        lib = RecipeLibrary()
        text = lib.format_list()
        assert "共" in text
        assert "个" in text

    def test_format_list_contains_hint(self):
        lib = RecipeLibrary()
        text = lib.format_list()
        assert "提示" in text

    def test_format_list_subset(self):
        lib = RecipeLibrary()
        recipes = lib.list(difficulty="beginner")
        text = lib.format_list(recipes)
        assert "SPEPA 示例库" in text


# ============================================================
# 便捷函数
# ============================================================

class TestConvenienceFunctions:
    """便捷函数测试。"""

    def test_get_recipe(self):
        recipe = get_recipe("quick_start")
        assert recipe is not None
        assert recipe.name == "quick_start"

    def test_get_recipe_nonexistent(self):
        recipe = get_recipe("nonexistent")
        assert recipe is None

    def test_list_recipes(self):
        recipes = list_recipes()
        assert len(recipes) > 0

    def test_list_recipes_by_difficulty(self):
        recipes = list_recipes(difficulty="beginner")
        for r in recipes:
            assert r.difficulty == "beginner"

    def test_list_recipes_by_tag(self):
        recipes = list_recipes(tag="config")
        for r in recipes:
            assert "config" in r.tags

    def test_run_recipe(self):
        result = run_recipe("quick_start")
        assert result["success"] is True

    def test_format_recipe(self):
        text = format_recipe("quick_start")
        assert "快速开始" in text

    def test_format_recipe_without_code(self):
        text = format_recipe("quick_start", with_code=False)
        assert "```python" not in text

    def test_format_recipe_list(self):
        text = format_recipe_list()
        assert "SPEPA 示例库" in text

    def test_add_recipe(self):
        # 注意：全局库是单例，添加后其他测试也能看到
        # 使用独特名称避免冲突
        recipe = Recipe(name="test_convenience_add", title="测试", description="测试", code="x=1")
        add_recipe(recipe)
        assert get_recipe("test_convenience_add") is not None

    def test_list_recipe_tags(self):
        tags = list_recipe_tags()
        assert isinstance(tags, list)
        assert len(tags) > 0

    def test_list_recipe_difficulties(self):
        difficulties = list_recipe_difficulties()
        assert "beginner" in difficulties


# ============================================================
# 端到端
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_browse_and_run(self):
        """浏览 → 查看详情 → 运行。"""
        recipes = list_recipes()
        assert len(recipes) > 0
        recipe = recipes[0]
        text = format_recipe(recipe.name)
        assert recipe.title in text
        result = run_recipe(recipe.name)
        # 不要求所有示例都能运行（有些需要特定环境），但应返回结果
        assert "success" in result

    def test_filter_by_difficulty_workflow(self):
        """按难度筛选工作流。"""
        for diff in list_recipe_difficulties():
            recipes = list_recipes(difficulty=diff)
            for r in recipes:
                assert r.difficulty == diff

    def test_filter_by_tag_workflow(self):
        """按标签筛选工作流。"""
        for tag in list_recipe_tags():
            recipes = list_recipes(tag=tag)
            for r in recipes:
                assert tag in r.tags

    def test_custom_recipe_lifecycle(self):
        """自定义示例生命周期：添加 → 查询 → 运行 → 格式化。"""
        lib = RecipeLibrary()
        recipe = Recipe(
            name="lifecycle_test",
            title="生命周期测试",
            description="测试完整生命周期",
            code="result = 1 + 1",
            difficulty="beginner",
            tags=["test"],
        )
        lib.add(recipe)
        # 查询
        assert lib.get("lifecycle_test") is not None
        # 运行
        globals_dict = {"__builtins__": __builtins__}
        run_result = lib.run("lifecycle_test", globals_dict=globals_dict)
        assert run_result["success"] is True
        assert globals_dict["result"] == 2
        # 格式化
        text = lib.format("lifecycle_test")
        assert "生命周期测试" in text

    def test_quick_start_runs_successfully(self):
        """快速开始示例应能成功运行。"""
        result = run_recipe("quick_start")
        assert result["success"] is True

    def test_format_list_shows_all_builtins(self):
        """格式化列表应显示所有内置示例。"""
        text = format_recipe_list()
        for name in ["quick_start", "custom_config", "save_and_load"]:
            assert name in text
