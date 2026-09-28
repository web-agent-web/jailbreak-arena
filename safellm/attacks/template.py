"""Template 攻击：角色扮演 / DAN / 指令覆盖模板。

适配自 AISafetyLab ``initialization/init_templates.json`` 中的 default/AutoDAN/
Jailbroken 模板族。在查询预算内依次尝试模板包裹 goal，用 env.query 探测目标
回复，挑选首个非拒绝（疑似成功）的模板作为最终 messages；若全部被拒绝则返回
首个模板（保留攻击意图）。这是黑盒、无需辅助 LLM 的模板攻击。
"""
from ..logging import logger
from .base import BaseAttacker, is_refusal
from .registry import register_attack
from .templates import seed_templates, format_template


@register_attack("template")
class TemplateAttack(BaseAttacker):
    def __init__(self, attack_model=None, max_queries: int = 20,
                 max_turns: int = 8, try_count: int = 5, **kwargs):
        super().__init__(attack_model=attack_model, max_queries=max_queries,
                         max_turns=max_turns, **kwargs)
        self.try_count = try_count

    def attack(self, env, task):
        goal = task["goal"]
        templates = seed_templates()[: max(self.try_count, 1)]
        best_msg = None
        for tmpl in templates:
            prompt = format_template(tmpl, goal)
            messages = [self.build_user_message(prompt)]
            resp = self._safe_query(env, messages)
            if resp and not is_refusal(resp):
                # 疑似成功，采用此模板。
                return self.finalize(messages)
            if best_msg is None:
                best_msg = messages
            if env.query_count >= self.max_queries:
                logger.debug("[template] 查询预算耗尽。")
                break
        # 全部被拒，返回首个模板尝试（仍提交攻击）。
        return self.finalize(best_msg or [self.build_user_message(goal)])
