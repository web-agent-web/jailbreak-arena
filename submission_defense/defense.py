"""防御方提交示例（赛题 §6.6 / §13.2）。

接口契约：
    def defend(env, request: dict) -> dict:
        输入：
            env: DefenseEnv，仅提供 env.query(messages) -> {"response": str}
                 （调用固定系统提示词保护下的目标模型）
            request: {"messages": [{"role": "user", "content": "..."}, ...]}
        输出：
            {"response": str}
        约束：不得修改目标模型权重与固定系统提示词（§3.1/§2.1）。

本示例通过后端注册表调用已实现的防御算法（解耦：防御方不感知攻击方）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml  # noqa: E402
from safellm.defenses import get_defender  # noqa: E402

with open(os.path.join(_HERE, "config.yaml"), "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f) or {}
_DEFENDER = None


def _get_defender(aux_model=None):
    global _DEFENDER
    if _DEFENDER is None:
        _DEFENDER = get_defender(_CFG.get("defense_method", "none"), aux_model=aux_model)
    return _DEFENDER


def defend(env, request: dict, aux_model=None) -> dict:
    """赛题 §6.6 防御方提交接口。

    Args:
        env: DefenseEnv，仅可通过 env.query 调用目标模型。
        request: 当前用户消息。
        aux_model: 评测环境注入的辅助 LLM（改写/自审，可选）。
    """
    defender = _get_defender(aux_model)
    return defender.defend(env, request)
