"""Direct 攻击：直接发送 goal（基线对照）。

不调用 env.query，单轮提交，用于衡量目标模型在无防御下的固有安全性。
"""
from .base import BaseAttacker
from .registry import register_attack


@register_attack("direct")
class DirectAttack(BaseAttacker):
    def attack(self, env, task):
        goal = task["goal"]
        return self.finalize([self.build_user_message(goal)])
