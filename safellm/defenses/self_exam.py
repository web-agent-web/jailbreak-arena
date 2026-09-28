"""Self-Examination 防御（后处理）。

适配自 AISafetyLab ``postprocess_defenders/self_evaluation.py``。
论文: LLM Self Defense: By Self Examination。
用辅助 LLM 对目标模型输出自审，若判定有害则替换为拒绝。
"""
from ..logging import logger
from .base import BaseDefender, SORRY_RESPONSE
from .registry import register_defense
from .texts import SELF_EXAM_TEXT


@register_defense("self_exam")
class SelfExamDefense(BaseDefender):
    def __init__(self, aux_model=None, judge_text: str = SELF_EXAM_TEXT, **kwargs):
        super().__init__(aux_model=aux_model, **kwargs)
        self.judge_text = judge_text

    def postprocess(self, response, messages):
        if not response:
            return response
        if self.aux_model is None:
            # 无辅助模型时降级为简单关键词启发式。
            return response if not _heuristic_unsafe(response) else SORRY_RESPONSE
        prompt = self.judge_text.replace("{content}", response)
        try:
            verdict = self.aux_model.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=8,
            ).strip().lower()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[self_exam] 自审失败，保留原输出: {e}")
            return response
        if verdict.startswith("yes"):
            return SORRY_RESPONSE
        return response


def _heuristic_unsafe(text: str) -> bool:
    """无辅助模型时的启发式：响应以配合前缀起手视为可疑。"""
    head = text.strip().lower()[:40]
    for p in ("sure, here", "of course", "certainly, here", "here is", "here's"):
        if head.startswith(p):
            return True
    return False
