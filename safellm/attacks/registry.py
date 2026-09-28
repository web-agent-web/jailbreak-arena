"""攻击方法注册表。

新增攻击只需：定义类并用 ``@register_attack("name")`` 装饰，并在
``attacks/__init__.py`` 中 import 该模块即可生效。arena 按名字装配，无需修改
arena 代码 -- 与防御注册表共同构成攻防解耦的关键机制。
"""
from __future__ import annotations

from typing import Callable

from ..logging import logger
from .base import BaseAttacker

ATTACK_REGISTRY: dict[str, type[BaseAttacker]] = {}


def register_attack(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.name = name
        if name in ATTACK_REGISTRY:
            logger.warning(f"攻击方法 '{name}' 已注册，将被覆盖。")
        ATTACK_REGISTRY[name] = cls
        return cls

    return deco


def get_attacker(name: str, attack_model=None, **kwargs) -> BaseAttacker:
    """按名字构造攻击实例。``attack_model`` 为辅助攻击 LLM（可选）。"""
    if name not in ATTACK_REGISTRY:
        raise ValueError(
            f"未知攻击方法: {name}，可选: {list_attacks()}"
        )
    return ATTACK_REGISTRY[name](attack_model=attack_model, **kwargs)


def list_attacks() -> list[str]:
    return sorted(ATTACK_REGISTRY.keys())
