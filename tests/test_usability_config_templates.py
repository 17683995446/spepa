"""SPEPA 配置模板管理测试。

测试范围：
- ConfigTemplate 数据结构（to_dict/from_dict）
- TemplateLibrary（CRUD、查询、导入导出、磁盘持久化）
- 便捷函数
- 格式化
- 端到端
"""
import os
import json
import tempfile
import pytest
from spepa.config_templates import (
    ConfigTemplate, TemplateLibrary,
    save_template, load_template, delete_template,
    list_templates, search_templates,
    export_template, import_template,
    format_template, format_template_list,
)
from spepa.utils.config import SPEPAConfig, get_preset_config


# ============================================================
# ConfigTemplate 数据结构
# ============================================================

class TestConfigTemplate:
    """ConfigTemplate 数据结构测试。"""

    def test_create_template(self):
        config = get_preset_config("minimal")
        template = ConfigTemplate(name="test", config=config, description="测试")
        assert template.name == "test"
        assert template.config is config
        assert template.description == "测试"
        assert template.category == "custom"
        assert template.tags == []
        assert template.version == "1.0"

    def test_template_with_all_fields(self):
        config = get_preset_config("balanced")
        template = ConfigTemplate(
            name="full",
            config=config,
            description="完整模板",
            category="experiment",
            tags=["tag1", "tag2"],
            author="tester",
            version="2.0",
        )
        assert template.category == "experiment"
        assert template.tags == ["tag1", "tag2"]
        assert template.author == "tester"
        assert template.version == "2.0"

    def test_to_dict(self):
        config = get_preset_config("minimal")
        template = ConfigTemplate(name="test", config=config, description="测试", tags=["a"])
        d = template.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "测试"
        assert d["tags"] == ["a"]
        assert "config" in d
        assert d["config"]["state_dim"] == config.state_dim

    def test_from_dict(self):
        config = get_preset_config("minimal")
        template = ConfigTemplate(name="test", config=config, description="测试", tags=["a"])
        d = template.to_dict()
        restored = ConfigTemplate.from_dict(d)
        assert restored.name == "test"
        assert restored.description == "测试"
        assert restored.tags == ["a"]
        assert restored.config.state_dim == config.state_dim

    def test_to_dict_from_dict_roundtrip(self):
        config = get_preset_config("balanced")
        template = ConfigTemplate(
            name="roundtrip", config=config, description="往返",
            category="test", tags=["x", "y"], author="me", version="3.0",
        )
        restored = ConfigTemplate.from_dict(template.to_dict())
        assert restored.name == template.name
        assert restored.description == template.description
        assert restored.category == template.category
        assert restored.tags == template.tags
        assert restored.author == template.author
        assert restored.version == template.version
        assert restored.config.to_dict() == template.config.to_dict()


# ============================================================
# TemplateLibrary - 内存模式
# ============================================================

class TestLibraryMemory:
    """内存模式模板库测试。"""

    def test_create_memory_library(self):
        lib = TemplateLibrary()
        assert lib.storage_dir is None
        assert lib.count() == 0

    def test_save_and_load(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        template = ConfigTemplate(name="t1", config=config, description="测试")
        lib.save(template)
        loaded = lib.load("t1")
        assert loaded is not None
        assert loaded.name == "t1"

    def test_save_duplicate_raises(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        lib.save(ConfigTemplate(name="t1", config=config))
        with pytest.raises(ValueError, match="已存在"):
            lib.save(ConfigTemplate(name="t1", config=config))

    def test_save_overwrite(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        lib.save(ConfigTemplate(name="t1", config=config, description="原"))
        lib.save(ConfigTemplate(name="t1", config=config, description="新"), overwrite=True)
        loaded = lib.load("t1")
        assert loaded.description == "新"

    def test_load_nonexistent(self):
        lib = TemplateLibrary()
        assert lib.load("nonexistent") is None

    def test_delete(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        lib.save(ConfigTemplate(name="t1", config=config))
        assert lib.delete("t1") is True
        assert lib.load("t1") is None

    def test_delete_nonexistent(self):
        lib = TemplateLibrary()
        assert lib.delete("nonexistent") is False

    def test_exists(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        lib.save(ConfigTemplate(name="t1", config=config))
        assert lib.exists("t1") is True
        assert lib.exists("t2") is False

    def test_count(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        lib.save(ConfigTemplate(name="t1", config=config))
        lib.save(ConfigTemplate(name="t2", config=config))
        assert lib.count() == 2


# ============================================================
# TemplateLibrary - 查询
# ============================================================

class TestLibraryQuery:
    """模板库查询测试。"""

    def setup_method(self):
        """每个测试创建独立库。"""
        self.lib = TemplateLibrary()
        self.config = get_preset_config("minimal")
        self.lib.save(ConfigTemplate(name="exp1", config=self.config, category="experiment", tags=["lr"]))
        self.lib.save(ConfigTemplate(name="exp2", config=self.config, category="experiment", tags=["seed"]))
        self.lib.save(ConfigTemplate(name="prod1", config=self.config, category="production", tags=["stable"]))

    def test_list_all(self):
        templates = self.lib.list()
        assert len(templates) == 3

    def test_list_by_category(self):
        experiments = self.lib.list(category="experiment")
        assert len(experiments) == 2
        for t in experiments:
            assert t.category == "experiment"

    def test_list_by_tag(self):
        tagged = self.lib.list(tag="lr")
        assert len(tagged) == 1
        assert tagged[0].name == "exp1"

    def test_list_sorted_by_name(self):
        templates = self.lib.list()
        names = [t.name for t in templates]
        assert names == sorted(names)

    def test_list_names(self):
        names = self.lib.list_names()
        assert "exp1" in names
        assert "exp2" in names
        assert "prod1" in names
        assert names == sorted(names)

    def test_list_categories(self):
        categories = self.lib.list_categories()
        assert "experiment" in categories
        assert "production" in categories

    def test_list_tags(self):
        tags = self.lib.list_tags()
        assert "lr" in tags
        assert "seed" in tags
        assert "stable" in tags

    def test_search_by_name(self):
        results = self.lib.search("exp")
        assert len(results) >= 2  # exp1, exp2

    def test_search_by_description(self):
        config = get_preset_config("minimal")
        self.lib.save(ConfigTemplate(name="desc_test", config=config, description="特殊描述关键词"))
        results = self.lib.search("特殊描述")
        assert any(t.name == "desc_test" for t in results)

    def test_search_by_tag(self):
        results = self.lib.search("lr")
        assert any(t.name == "exp1" for t in results)

    def test_search_no_match(self):
        results = self.lib.search("zzz_nonexistent")
        assert len(results) == 0

    def test_search_sorted_by_score(self):
        # 精确名称匹配应排在前面
        config = get_preset_config("minimal")
        self.lib.save(ConfigTemplate(name="run_test", config=config, description="run"))
        self.lib.save(ConfigTemplate(name="other", config=config, description="run something"))
        results = self.lib.search("run")
        assert results[0].name == "run_test"


# ============================================================
# TemplateLibrary - 磁盘持久化
# ============================================================

class TestLibraryDisk:
    """磁盘持久化测试。"""

    def test_save_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            lib = TemplateLibrary(storage_dir=d)
            config = get_preset_config("minimal")
            lib.save(ConfigTemplate(name="disk1", config=config, description="磁盘测试"))
            # 文件应存在
            filepath = os.path.join(d, "disk1.json")
            assert os.path.exists(filepath)

    def test_load_from_disk_on_init(self):
        with tempfile.TemporaryDirectory() as d:
            # 先保存
            lib1 = TemplateLibrary(storage_dir=d)
            config = get_preset_config("minimal")
            lib1.save(ConfigTemplate(name="persist_test", config=config, description="持久化"))
            # 新实例应能加载
            lib2 = TemplateLibrary(storage_dir=d)
            loaded = lib2.load("persist_test")
            assert loaded is not None
            assert loaded.description == "持久化"

    def test_delete_from_disk(self):
        with tempfile.TemporaryDirectory() as d:
            lib = TemplateLibrary(storage_dir=d)
            config = get_preset_config("minimal")
            lib.save(ConfigTemplate(name="to_delete", config=config))
            filepath = os.path.join(d, "to_delete.json")
            assert os.path.exists(filepath)
            lib.delete("to_delete")
            assert not os.path.exists(filepath)

    def test_disk_and_memory_consistent(self):
        with tempfile.TemporaryDirectory() as d:
            lib = TemplateLibrary(storage_dir=d)
            config = get_preset_config("minimal")
            lib.save(ConfigTemplate(name="t1", config=config))
            lib.save(ConfigTemplate(name="t2", config=config))
            # 重新加载
            lib2 = TemplateLibrary(storage_dir=d)
            assert lib2.count() == 2
            assert lib2.list_names() == ["t1", "t2"]


# ============================================================
# TemplateLibrary - 导入导出
# ============================================================

class TestLibraryImportExport:
    """导入导出测试。"""

    def test_export_template(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        lib.save(ConfigTemplate(name="export_test", config=config, description="导出"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        try:
            lib.export_template("export_test", filepath)
            assert os.path.exists(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)
            assert data["name"] == "export_test"
        finally:
            os.unlink(filepath)

    def test_export_nonexistent_raises(self):
        lib = TemplateLibrary()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        try:
            with pytest.raises(ValueError, match="不存在"):
                lib.export_template("nonexistent", filepath)
        finally:
            os.unlink(filepath)

    def test_import_template(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        template = ConfigTemplate(name="import_test", config=config, description="导入")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(template.to_dict(), f)
            filepath = f.name
        try:
            imported = lib.import_template(filepath)
            assert imported.name == "import_test"
            assert lib.exists("import_test")
        finally:
            os.unlink(filepath)

    def test_import_nonexistent_file_raises(self):
        lib = TemplateLibrary()
        with pytest.raises(ValueError, match="文件不存在"):
            lib.import_template("/nonexistent/path.json")

    def test_import_overwrite(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        lib.save(ConfigTemplate(name="t1", config=config, description="原"))
        template = ConfigTemplate(name="t1", config=config, description="新")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(template.to_dict(), f)
            filepath = f.name
        try:
            lib.import_template(filepath, overwrite=True)
            assert lib.load("t1").description == "新"
        finally:
            os.unlink(filepath)

    def test_export_all(self):
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        lib.save(ConfigTemplate(name="t1", config=config))
        lib.save(ConfigTemplate(name="t2", config=config))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        try:
            lib.export_all(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)
            assert "templates" in data
            assert len(data["templates"]) == 2
        finally:
            os.unlink(filepath)

    def test_import_all(self):
        # 准备导出文件
        lib1 = TemplateLibrary()
        config = get_preset_config("minimal")
        lib1.save(ConfigTemplate(name="t1", config=config))
        lib1.save(ConfigTemplate(name="t2", config=config))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        lib1.export_all(filepath)
        try:
            # 导入到新库
            lib2 = TemplateLibrary()
            count = lib2.import_all(filepath)
            assert count == 2
            assert lib2.count() == 2
        finally:
            os.unlink(filepath)


# ============================================================
# 便捷函数
# ============================================================

class TestConvenienceFunctions:
    """便捷函数测试。"""

    def test_save_template(self):
        config = get_preset_config("minimal")
        template = save_template("conv_test", config, description="便捷测试", tags=["test"])
        assert template.name == "conv_test"
        assert template.description == "便捷测试"
        assert template.tags == ["test"]

    def test_load_template(self):
        config = get_preset_config("minimal")
        save_template("conv_load", config)
        loaded = load_template("conv_load")
        assert loaded is not None
        assert loaded.name == "conv_load"

    def test_delete_template(self):
        config = get_preset_config("minimal")
        save_template("conv_delete", config)
        assert delete_template("conv_delete") is True
        assert load_template("conv_delete") is None

    def test_list_templates(self):
        config = get_preset_config("minimal")
        save_template("conv_list", config, category="test_cat")
        templates = list_templates(category="test_cat")
        assert any(t.name == "conv_list" for t in templates)

    def test_search_templates(self):
        config = get_preset_config("minimal")
        save_template("conv_search", config, description="搜索测试")
        results = search_templates("搜索")
        assert any(t.name == "conv_search" for t in results)

    def test_export_import_template(self):
        config = get_preset_config("minimal")
        save_template("conv_export", config)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        try:
            export_template("conv_export", filepath)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_format_template_list(self):
        text = format_template_list()
        assert "SPEPA 配置模板库" in text

    def test_format_template(self):
        config = get_preset_config("minimal")
        save_template("conv_format", config, description="格式化测试")
        text = format_template("conv_format")
        assert "conv_format" in text
        assert "格式化测试" in text

    def test_format_nonexistent_template(self):
        text = format_template("nonexistent_xyz")
        assert "未找到" in text


# ============================================================
# 端到端
# ============================================================

class TestEndToEnd:
    """端到端集成测试。"""

    def test_full_lifecycle(self):
        """完整生命周期：保存 → 加载 → 修改 → 删除。"""
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        # 保存
        lib.save(ConfigTemplate(name="life", config=config, description="生命周期"))
        assert lib.exists("life")
        # 加载
        loaded = lib.load("life")
        assert loaded.description == "生命周期"
        # 覆盖
        lib.save(ConfigTemplate(name="life", config=config, description="更新"), overwrite=True)
        assert lib.load("life").description == "更新"
        # 删除
        lib.delete("life")
        assert not lib.exists("life")

    def test_export_import_roundtrip(self):
        """导出导入往返测试。"""
        lib1 = TemplateLibrary()
        config = get_preset_config("balanced")
        lib1.save(ConfigTemplate(
            name="roundtrip", config=config, description="往返",
            category="test", tags=["a", "b"], author="tester",
        ))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        try:
            lib1.export_template("roundtrip", filepath)
            lib2 = TemplateLibrary()
            imported = lib2.import_template(filepath)
            assert imported.name == "roundtrip"
            assert imported.description == "往返"
            assert imported.category == "test"
            assert imported.tags == ["a", "b"]
            assert imported.author == "tester"
            assert imported.config.to_dict() == config.to_dict()
        finally:
            os.unlink(filepath)

    def test_disk_persistence_workflow(self):
        """磁盘持久化工作流。"""
        with tempfile.TemporaryDirectory() as d:
            # 会话 1：保存
            lib1 = TemplateLibrary(storage_dir=d)
            config = get_preset_config("minimal")
            lib1.save(ConfigTemplate(name="persist1", config=config, description="持久化1"))
            lib1.save(ConfigTemplate(name="persist2", config=config, description="持久化2"))
            # 会话 2：加载
            lib2 = TemplateLibrary(storage_dir=d)
            assert lib2.count() == 2
            assert lib2.load("persist1").description == "持久化1"
            # 会话 2：删除一个
            lib2.delete("persist1")
            # 会话 3：验证
            lib3 = TemplateLibrary(storage_dir=d)
            assert lib3.count() == 1
            assert lib3.exists("persist2")
            assert not lib3.exists("persist1")

    def test_search_and_filter_workflow(self):
        """搜索与筛选工作流。"""
        lib = TemplateLibrary()
        config = get_preset_config("minimal")
        lib.save(ConfigTemplate(name="exp_lr", config=config, category="experiment", tags=["lr", "test"]))
        lib.save(ConfigTemplate(name="exp_seed", config=config, category="experiment", tags=["seed"]))
        lib.save(ConfigTemplate(name="prod_stable", config=config, category="production", tags=["stable"]))
        # 按分类
        experiments = lib.list(category="experiment")
        assert len(experiments) == 2
        # 按标签
        lr_templates = lib.list(tag="lr")
        assert len(lr_templates) == 1
        # 搜索
        results = lib.search("exp")
        assert len(results) == 2
        # 格式化
        text = lib.format_list(experiments)
        assert "exp_lr" in text
        assert "exp_seed" in text

    def test_template_with_custom_config(self):
        """自定义配置模板测试。"""
        lib = TemplateLibrary()
        # 自定义配置
        config = get_preset_config("balanced").merge({
            "state_dim": 32,
            "learning_rate_micro": 0.02,
            "max_steps": 500,
        })
        lib.save(ConfigTemplate(
            name="custom_exp", config=config, description="自定义实验",
            category="experiment", tags=["custom"],
        ))
        loaded = lib.load("custom_exp")
        assert loaded.config.state_dim == 32
        assert loaded.config.learning_rate_micro == 0.02
        assert loaded.config.max_steps == 500
