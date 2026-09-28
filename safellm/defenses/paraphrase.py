"""Paraphrase 防御（预处理）。

适配自 AISafetyLab ``preprocess_defenders/paraphrasing.py``。
用辅助 LLM 将用户输入改写为朴素直接形式，消解角色扮演/指令覆盖/编码等
提示注入手法，再交给目标模型处理。辅助 LLM 走标准 OpenAI 接口。
"""
from ..logging import logger
from .base import BaseDefender
from .registry import register_defense
from .texts import PARAPHRASE_INSTRUCTION


@register_defense("paraphrase")
class ParaphraseDefense(BaseDefender):
    def __init__(self, aux_model=None, instruction: str = PARAPHRASE_INSTRUCTION,
                 **kwargs):
        super().__init__(aux_model=aux_model, **kwargs)
        self.instruction = instruction

    def preprocess(self, messages):
        if not messages or self.aux_model is None:
            # 无辅助模型则不改写（降级为透传）。
            return messages, False
        messages = [dict(m) for m in messages]
        last = messages[-1]
        if last.get("role") != "user":
            return messages, False
        prompt = self.instruction.replace("{input_text}", last.get("content", ""))
        try:
            rewritten = self.aux_model.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=512,
            ).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[paraphrase] 改写失败，保留原文: {e}")
            return messages, False
        if rewritten:
            last["content"] = rewritten
        return messages, False
