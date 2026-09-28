"""攻击方法层。

攻击方实现赛题 §5.5 的 ``attack(env, task) -> {"messages": [...]}`` 接口。
攻击方只能通过 ``AttackEnv.query`` 获得目标模型文本回答（§2.1 黑盒条件），
可在查询预算内迭代改进候选提示词，最终返回不超过 8 轮、末尾为 user 的 messages。
所有攻击通过注册表装配，与防御算法完全解耦。
"""
from . import templates  # noqa: F401  触发模板加载
from .base import BaseAttacker, is_refusal, COMPLIANCE_HINTS
from .registry import ATTACK_REGISTRY, get_attacker, list_attacks, register_attack

# 导入各攻击模块以触发注册。
from . import direct, template, cipher, multilingual, pair  # noqa: F401

__all__ = [
    "BaseAttacker", "is_refusal", "COMPLIANCE_HINTS",
    "ATTACK_REGISTRY", "get_attacker", "list_attacks", "register_attack",
]
