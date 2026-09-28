"""Goal Prioritization 防御（预处理）。

适配自 AISafetyLab ``preprocess_defenders/goal_prioritization.py``。
论文: Defending LLMs Against Jailbreaking Attacks Through Goal Prioritization。
显式要求模型将安全目标置于帮助性目标之上。
"""
from .base import BaseDefender
from .registry import register_defense
from .texts import GOAL_PRIORITIZATION_TEXT


@register_defense("goal_prioritization")
class GoalPrioritizationDefense(BaseDefender):
    def __init__(self, aux_model=None, prompt_text: str = GOAL_PRIORITIZATION_TEXT, **kwargs):
        super().__init__(aux_model=aux_model, **kwargs)
        self.prompt_text = prompt_text

    def preprocess(self, messages):
        if not messages:
            return messages, False
        messages = [dict(m) for m in messages]
        last = messages[-1]
        if last.get("role") == "user":
            last["content"] = self.prompt_text.replace(
                "{input_text}", last.get("content", "")
            )
        return messages, False
