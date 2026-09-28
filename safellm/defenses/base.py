"""防御基类：三段式防御，适配自 AISafetyLab ``defend_chat.chat``。

数据流：
    request.messages
        -> preprocess(messages)  [可改写/拒绝，对应 PreprocessDefender]
        -> intraprocess(env, messages)  [调用目标模型生成，对应 IntraprocessDefender]
        -> postprocess(response, messages)  [审查/替换，对应 PostprocessDefender]
        -> {"response": ...}

防御方仅通过 ``env``（DefenseEnv）访问固定系统提示词保护下的目标模型，
辅助 LLM（改写/裁判）通过 ``aux_model`` 访问，二者均走标准 OpenAI 接口。
防御方不知道攻击方身份/原始 goal/风险类别（赛题 §6.4）。
"""
from __future__ import annotations

from abc import ABC
from typing import Any

from ..logging import logger

SORRY_RESPONSE = "I'm sorry, but I cannot provide assistance or information on that topic."


class BaseDefender(ABC):
    """所有防御方法的基类。子类按需覆盖 preprocess/intraprocess/postprocess。"""

    name: str = "base"

    def __init__(self, aux_model=None, **kwargs):
        # aux_model: 辅助 LLM（改写/裁判），OpenAI 兼容；可选。
        self.aux_model = aux_model
        self.kwargs = kwargs

    # ---- 三段式钩子，默认为 no-op ----
    def preprocess(self, messages: list[dict]) -> tuple[list[dict], bool]:
        """预处理输入消息。返回 (改写后消息, 是否拒绝)。"""
        return messages, False

    def intraprocess(self, env, messages: list[dict]) -> str:
        """过程内生成：调用目标模型。默认直接调用 env.query。"""
        return env.query(messages)["response"]

    def postprocess(self, response: str, messages: list[dict]) -> str:
        """后处理：审查/修正模型输出。"""
        return response

    # ---- 赛题 §6.6 提交接口 ----
    def defend(self, env, request: dict) -> dict[str, Any]:
        messages = [dict(m) for m in request["messages"]]

        messages, reject = self.preprocess(messages)
        if reject:
            logger.debug(f"[{self.name}] 预处理阶段拒绝。")
            return {"response": SORRY_RESPONSE}

        response = self.intraprocess(env, messages)
        response = self.postprocess(response, messages)
        return {"response": response}
