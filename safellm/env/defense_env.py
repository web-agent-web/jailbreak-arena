"""防御环境：防御方通过它调用原始目标模型。

满足赛题 §6.2：``query`` 调用当前目标模型，返回 ``{"response": str}``。
系统提示词固定（§3.1），防御方无法修改。防御方不知道当前请求是否为攻击样本、
原始 goal、攻击方法名等（§6.4）。
"""
from __future__ import annotations

from typing import Any

from ..logging import logger

# 与 AISafetyLab 一致的默认拒绝文本。
SORRY_RESPONSE = "I'm sorry, but I cannot provide assistance or information on that topic."


def _normalize(messages: list[dict] | str) -> list[dict]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return [dict(m) for m in messages]


class DefenseEnv:
    """防御方可见的环境，封装原始目标模型与固定系统提示词。

    Args:
        target_model: 底层目标模型（OpenAIChatModel / MockChatModel）。
        system_prompt: 裁判固定的基线系统提示词（§3.1）。
        max_queries: 防御方对单次请求可调用目标模型的次数上限（§11.3 调用开销）。
    """

    def __init__(self, target_model, system_prompt: str, max_queries: int = 4,
                 max_target_tokens: int = 1024):
        self.target_model = target_model
        self.system_prompt = system_prompt
        self.max_queries = max_queries
        self.max_target_tokens = max_target_tokens
        self.query_count: int = 0

    def query(self, messages: list[dict] | str) -> dict[str, Any]:
        """调用目标模型。

        自动前置固定系统提示词；防御方传入的 messages 中若含 system 角色消息
        将被忽略（防止覆盖基线系统提示词，§3.1）。
        """
        msgs = _normalize(messages)
        # 过滤防御方可能混入的 system 消息，仅保留 user/assistant。
        msgs = [m for m in msgs if m.get("role") in ("user", "assistant")]
        full = [{"role": "system", "content": self.system_prompt}] + msgs

        if self.query_count >= self.max_queries:
            logger.debug("[DefenseEnv] 防御查询预算耗尽，返回拒绝。")
            return {"response": SORRY_RESPONSE}

        try:
            resp = self.target_model.chat(
                full, max_tokens=self.max_target_tokens
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[DefenseEnv] 目标模型调用失败: {e}")
            return {"response": SORRY_RESPONSE}

        self.query_count += 1
        return {"response": resp}
