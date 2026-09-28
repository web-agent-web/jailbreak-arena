"""防御算法层。

防御方实现赛题 §6.6 的 ``defend(env, request) -> {"response": str}`` 接口。
基类 ``BaseDefender`` 采用 AISafetyLab 的三段式防御架构（适配自
``defense/inference_defense/defend_chat.py``）：预处理 -> 过程内生成 -> 后处理。
所有防御通过注册表装配，与攻击方法完全解耦。
"""
# 导入各防御模块以触发 @register_defense 注册。
from . import (
    none_defense,
    self_reminder,
    goal_prioritization,
    paraphrase,
    self_exam,
    refusal_filter,
)
from .base import BaseDefender, SORRY_RESPONSE
from .registry import (
    DEFENSE_REGISTRY,
    get_defender,
    list_defenses,
    register_defense,
)

__all__ = [
    "BaseDefender",
    "SORRY_RESPONSE",
    "DEFENSE_REGISTRY",
    "get_defender",
    "list_defenses",
    "register_defense",
]
