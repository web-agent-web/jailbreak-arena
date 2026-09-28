"""防御方法注册表。

新增防御只需：在子模块定义类并用 ``@register_defender("name")`` 装饰，
并在 ``defenses/__init__.py`` 中 import 该模块即可生效。arena 按名字装配，
无需修改 arena 代码 —— 这正是攻防解耦的关键机制。
"""
from __future__ import annotations

from typing import Callable

from ..logging import logger
from .base import BaseDefender

DEFENSE_REGISTRY: dict[str, type[BaseDefender]] = {}


def register_defense(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.name = name
        if name in DEFENSE_REGISTRY:
            logger.warning(f"防御方法 '{name}' 已注册，将被覆盖。")
        DEFENSE_REGISTRY[name] = cls
        return cls

    return deco


def get_defender(name: str, aux_model=None, **kwargs) -> BaseDefender:
    """按名字构造防御实例。``aux_model`` 为辅助 LLM（可选）。"""
    if name not in DEFENSE_REGISTRY:
        raise ValueError(
            f"未知防御方法: {name}，可选: {list_defenses()}"
        )
    return DEFENSE_REGISTRY[name](aux_model=aux_model, **kwargs)


def list_defenses() -> list[str]:
    return sorted(DEFENSE_REGISTRY.keys())
