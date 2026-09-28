"""Self-Reminder 防御（预处理）。

适配自 AISafetyLab ``preprocess_defenders/self_reminder.py``。
论文: Defending ChatGPT against jailbreak attack via self-reminders。
在用户输入外包裹安全提醒，强化模型的安全职责。
"""
from .base import BaseDefender
from .registry import register_defense
from .texts import SELF_REMINDER_TEXT


@register_defense("self_reminder")
class SelfReminderDefense(BaseDefender):
    def __init__(self, aux_model=None, reminder_text: str = SELF_REMINDER_TEXT, **kwargs):
        super().__init__(aux_model=aux_model, **kwargs)
        self.reminder_text = reminder_text

    def preprocess(self, messages):
        if not messages:
            return messages, False
        messages = [dict(m) for m in messages]
        last = messages[-1]
        if last.get("role") == "user":
            last["content"] = self.reminder_text.replace(
                "{input_text}", last.get("content", "")
            )
        return messages, False
