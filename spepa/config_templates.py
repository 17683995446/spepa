"""SPEPA 配置模板管理。

设计原则（第一性原理）：
- 可复用：用户调好的配置可保存为模板，下次直接加载
- 可分享：模板可导出为文件，分享给他人
- 可组织：模板可分类、打标签、搜索
- 可版本化：模板带元信息（创建时间、描述、来源）

与 config_advisor 的区别：
- config_advisor : 给出"建议配置"（系统推荐）
- config_templates : 管理"用户配置"（用户自定义保存）

与 SPEPAConfig.save_to_file 的区别：
- save_to_file : 保存单份配置到文件（底层 API）
- config_templates : 管理多份配置的集合（带元信息、分类、搜索）

主要功能：
- ConfigTemplate    : 配置模板（配置 + 元信息）
- TemplateLibrary   : 模板库（保存/加载/搜索/导出）
- save_template()   : 保存当前配置为模板
- load_template()   : 加载模板
- list_templates()  : 列出模板
- export_template() : 导出模板到文件
- import_template() : 从文件导入模板
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from spepa.utils.config import SPEPAConfig, get_preset_config
from spepa.utils.logger import get_logger

logger = get_logger("spepa.config_templates")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ConfigTemplate:
    """配置模板。

    一个模板 = 配置 + 元信息。
    """
    name: str                          # 模板名（唯一标识）
    config: SPEPAConfig                # 配置内容
    description: str = ""              # 描述
    category: str = "custom"           # 分类（如 "experiment" / "production" / "custom"）
    tags: List[str] = field(default_factory=list)  # 标签
    created_at: float = field(default_factory=time.time)  # 创建时间戳
    author: str = ""                   # 作者
    version: str = "1.0"               # 版本

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化的字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "author": self.author,
            "version": self.version,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigTemplate":
        """从字典重建模板。"""
        config = SPEPAConfig.from_dict(data.get("config", {}))
        return cls(
            name=data["name"],
            config=config,
            description=data.get("description", ""),
            category=data.get("category", "custom"),
            tags=data.get("tags", []),
            created_at=data.get("created_at", time.time()),
            author=data.get("author", ""),
            version=data.get("version", "1.0"),
        )


# ============================================================
# 模板库
# ============================================================

class TemplateLibrary:
    """配置模板库。

    使用方式：
        lib = TemplateLibrary(storage_dir="~/.spepa/templates")
        lib.save(ConfigTemplate(name="my_exp", config=config, description="我的实验"))
        template = lib.load("my_exp")
        config = template.config
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """初始化模板库。

        参数：
            storage_dir : 存储目录（None 表示仅内存，不持久化）
        """
        self.storage_dir = os.path.expanduser(storage_dir) if storage_dir else None
        self._templates: Dict[str, ConfigTemplate] = {}
        if self.storage_dir:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """从磁盘加载所有模板。"""
        if not self.storage_dir or not os.path.isdir(self.storage_dir):
            return
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    template = ConfigTemplate.from_dict(data)
                    self._templates[template.name] = template
                except Exception as e:
                    logger.warning("加载模板 %s 失败: %s", filename, e)

    def _save_to_disk(self, template: ConfigTemplate) -> None:
        """保存单个模板到磁盘。"""
        if not self.storage_dir:
            return
        os.makedirs(self.storage_dir, exist_ok=True)
        filepath = os.path.join(self.storage_dir, f"{template.name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(template.to_dict(), f, indent=2, ensure_ascii=False)

    def _delete_from_disk(self, name: str) -> None:
        """从磁盘删除模板。"""
        if not self.storage_dir:
            return
        filepath = os.path.join(self.storage_dir, f"{name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)

    # --------------------------------------------------------
    # CRUD
    # --------------------------------------------------------

    def save(self, template: ConfigTemplate, overwrite: bool = False) -> None:
        """保存模板。

        参数：
            template : ConfigTemplate 实例
            overwrite : 是否覆盖同名模板

        抛出：
            ValueError 当模板已存在且 overwrite=False
        """
        if template.name in self._templates and not overwrite:
            raise ValueError(
                f"模板 '{template.name}' 已存在。使用 overwrite=True 覆盖。"
            )
        self._templates[template.name] = template
        self._save_to_disk(template)
        logger.info("模板 '%s' 已保存", template.name)

    def load(self, name: str) -> Optional[ConfigTemplate]:
        """加载模板。

        参数：
            name : 模板名

        返回：
            ConfigTemplate 实例，不存在则返回 None
        """
        return self._templates.get(name)

    def delete(self, name: str) -> bool:
        """删除模板。

        参数：
            name : 模板名

        返回：
            是否删除成功
        """
        if name not in self._templates:
            return False
        del self._templates[name]
        self._delete_from_disk(name)
        logger.info("模板 '%s' 已删除", name)
        return True

    def exists(self, name: str) -> bool:
        """检查模板是否存在。"""
        return name in self._templates

    # --------------------------------------------------------
    # 查询
    # --------------------------------------------------------

    def list(self,
             category: Optional[str] = None,
             tag: Optional[str] = None) -> List[ConfigTemplate]:
        """列出模板（可按分类/标签筛选）。

        参数：
            category : 按分类筛选
            tag : 按标签筛选

        返回：
            ConfigTemplate 列表（按名称排序）
        """
        result = list(self._templates.values())
        if category:
            result = [t for t in result if t.category == category]
        if tag:
            result = [t for t in result if tag in t.tags]
        return sorted(result, key=lambda t: t.name)

    def list_names(self) -> List[str]:
        """列出所有模板名。"""
        return sorted(self._templates.keys())

    def list_categories(self) -> List[str]:
        """列出所有分类。"""
        return sorted(set(t.category for t in self._templates.values()))

    def list_tags(self) -> List[str]:
        """列出所有标签。"""
        tags = set()
        for t in self._templates.values():
            tags.update(t.tags)
        return sorted(tags)

    def search(self, query: str) -> List[ConfigTemplate]:
        """搜索模板（按名称/描述/标签匹配）。

        参数：
            query : 搜索关键词

        返回：
            匹配的 ConfigTemplate 列表
        """
        query_lower = query.lower()
        scored: List[tuple] = []
        for t in self._templates.values():
            score = 0
            if query_lower in t.name.lower():
                score += 10
            if query_lower in t.description.lower():
                score += 5
            for tag in t.tags:
                if query_lower in tag.lower():
                    score += 3
            if score > 0:
                scored.append((score, t))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [t for _, t in scored]

    def count(self) -> int:
        """返回模板数量。"""
        return len(self._templates)

    # --------------------------------------------------------
    # 导入导出
    # --------------------------------------------------------

    def export_template(self, name: str, filepath: str) -> str:
        """导出模板到文件。

        参数：
            name : 模板名
            filepath : 目标文件路径

        返回：
            实际保存的文件路径

        抛出：
            ValueError 当模板不存在时
        """
        template = self.load(name)
        if template is None:
            raise ValueError(f"模板 '{name}' 不存在")
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(template.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("模板 '%s' 已导出到 %s", name, filepath)
        return filepath

    def import_template(self, filepath: str, overwrite: bool = False) -> ConfigTemplate:
        """从文件导入模板。

        参数：
            filepath : 源文件路径
            overwrite : 是否覆盖同名模板

        返回：
            导入的 ConfigTemplate

        抛出：
            ValueError 当文件不存在或模板已存在且 overwrite=False
        """
        if not os.path.exists(filepath):
            raise ValueError(f"文件不存在: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        template = ConfigTemplate.from_dict(data)
        self.save(template, overwrite=overwrite)
        logger.info("模板 '%s' 已从 %s 导入", template.name, filepath)
        return template

    def export_all(self, filepath: str) -> str:
        """导出所有模板到单个文件。

        参数：
            filepath : 目标文件路径

        返回：
            实际保存的文件路径
        """
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        data = {
            "version": "1.0",
            "exported_at": time.time(),
            "templates": [t.to_dict() for t in self._templates.values()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filepath

    def import_all(self, filepath: str, overwrite: bool = False) -> int:
        """从文件批量导入模板。

        参数：
            filepath : 源文件路径
            overwrite : 是否覆盖同名模板

        返回：
            导入的模板数量
        """
        if not os.path.exists(filepath):
            raise ValueError(f"文件不存在: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        templates_data = data.get("templates", [])
        count = 0
        for t_data in templates_data:
            try:
                template = ConfigTemplate.from_dict(t_data)
                if overwrite or template.name not in self._templates:
                    self.save(template, overwrite=True)
                    count += 1
            except Exception as e:
                logger.warning("导入模板失败: %s", e)
        return count

    # --------------------------------------------------------
    # 格式化
    # --------------------------------------------------------

    def format_list(self, templates: Optional[List[ConfigTemplate]] = None) -> str:
        """格式化模板列表。"""
        if templates is None:
            templates = self.list()
        lines = []
        lines.append("=" * 60)
        lines.append(f"SPEPA 配置模板库（共 {len(templates)} 个）")
        lines.append("=" * 60)
        if not templates:
            lines.append("\n（空）")
        else:
            for t in templates:
                tags_str = f" [{', '.join(t.tags)}]" if t.tags else ""
                lines.append(f"\n  {t.name:<25} ({t.category}){tags_str}")
                if t.description:
                    lines.append(f"    {t.description}")
                lines.append(f"    创建: {time.strftime('%Y-%m-%d %H:%M', time.localtime(t.created_at))}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def format_template(self, name: str) -> str:
        """格式化单个模板详情。"""
        template = self.load(name)
        if template is None:
            return f"未找到模板: {name}"
        lines = []
        lines.append("=" * 60)
        lines.append(f"模板: {template.name}")
        lines.append("=" * 60)
        lines.append(f"描述: {template.description or '(无)'}")
        lines.append(f"分类: {template.category}")
        if template.tags:
            lines.append(f"标签: {', '.join(template.tags)}")
        lines.append(f"作者: {template.author or '(未知)'}")
        lines.append(f"版本: {template.version}")
        lines.append(f"创建: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(template.created_at))}")
        lines.append(f"\n配置:")
        config_dict = template.config.to_dict()
        for k, v in config_dict.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    lines.append(f"    {kk}: {vv}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ============================================================
# 模块级便捷 API
# ============================================================

_global_library: Optional[TemplateLibrary] = None


def _get_library() -> TemplateLibrary:
    """获取全局模板库（仅内存）。"""
    global _global_library
    if _global_library is None:
        _global_library = TemplateLibrary()
    return _global_library


def save_template(name: str,
                  config: SPEPAConfig,
                  description: str = "",
                  category: str = "custom",
                  tags: Optional[List[str]] = None,
                  overwrite: bool = False) -> ConfigTemplate:
    """保存配置为模板。

    参数：
        name : 模板名
        config : SPEPAConfig 实例
        description : 描述
        category : 分类
        tags : 标签列表
        overwrite : 是否覆盖同名模板

    返回：
        创建的 ConfigTemplate
    """
    template = ConfigTemplate(
        name=name,
        config=config,
        description=description,
        category=category,
        tags=tags or [],
    )
    _get_library().save(template, overwrite=overwrite)
    return template


def load_template(name: str) -> Optional[ConfigTemplate]:
    """加载模板。"""
    return _get_library().load(name)


def delete_template(name: str) -> bool:
    """删除模板。"""
    return _get_library().delete(name)


def list_templates(category: Optional[str] = None,
                   tag: Optional[str] = None) -> List[ConfigTemplate]:
    """列出模板。"""
    return _get_library().list(category=category, tag=tag)


def search_templates(query: str) -> List[ConfigTemplate]:
    """搜索模板。"""
    return _get_library().search(query)


def export_template(name: str, filepath: str) -> str:
    """导出模板到文件。"""
    return _get_library().export_template(name, filepath)


def import_template(filepath: str, overwrite: bool = False) -> ConfigTemplate:
    """从文件导入模板。"""
    return _get_library().import_template(filepath, overwrite=overwrite)


def format_template_list(category: Optional[str] = None,
                         tag: Optional[str] = None) -> str:
    """格式化模板列表。"""
    templates = list_templates(category=category, tag=tag)
    return _get_library().format_list(templates)


def format_template(name: str) -> str:
    """格式化单个模板详情。"""
    return _get_library().format_template(name)
