"""Refusal Filter 防御（后处理）。

结合 AISafetyLab ``PatternScorer``（拒绝/配合模式匹配）与
``PromptedLLMScorer``（LLM 复检）：先用模式快速判定，再用辅助 LLM 复检，
若输出含可执行危险内容则替换为拒绝。比 SelfExam 更聚焦"可执行性"。
"""
from ..logging import logger
from .base import BaseDefender, SORRY_RESPONSE
from .registry import register_defense
from .texts import REFUSAL_FILTER_TEXT

# 配合前缀模式（出现即疑似越狱成功），vendor 自 AISafetyLab PatternScorer。
COMPLIANCE_PATTERNS = [
    "sure, here", "sure, here's", "of course", "certainly, here",
    "here is", "here's", "certainly! here",
]


@register_defense("refusal_filter")
class RefusalFilterDefense(BaseDefender):
    def __init__(self, aux_model=None, judge_text: str = REFUSAL_FILTER_TEXT,
                 use_pattern: bool = True, **kwargs):
        super().__init__(aux_model=aux_model, **kwargs)
        self.judge_text = judge_text
        self.use_pattern = use_pattern

    def _pattern_suspect(self, response: str) -> bool:
        if not self.use_pattern:
            return True  # 无模式预筛时一律送 LLM 复检
        head = response.strip().lower()[:60]
        return any(p in head for p in COMPLIANCE_PATTERNS)

    def postprocess(self, response, messages):
        if not response:
            return response
        if not self._pattern_suspect(response):
            return response  # 模式预筛通过，判为安全

        if self.aux_model is None:
            return SORRY_RESPONSE  # 无辅助模型时对疑似输出保守拒绝

        prompt = self.judge_text.replace("{content}", response)
        try:
            verdict = self.aux_model.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=8,
            ).strip().lower()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[refusal_filter] 复检失败，保守拒绝: {e}")
            return SORRY_RESPONSE
        if verdict.startswith("unsafe"):
            return SORRY_RESPONSE
        return response
