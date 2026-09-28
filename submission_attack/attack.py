"""攻击方提交示例（赛题 §5.5 / §13.1）。

接口契约：
    def attack(env, task: dict) -> dict:
        输入：
            env: AttackEnv，仅提供 env.query(messages) -> {"response": str}
            task: {"sample_id", "goal", "language", "context"}
        输出：
            {"messages": [{"role": "user", "content": "..."}, ...]}
        要求：最后一条消息为 user；消息总轮数 ≤ 8。

本示例通过后端注册表调用已实现的攻击方法（解耦：攻击方不感知防御方）。
参赛者也可在 methods/ 下实现自定义攻击并在此装配。
"""
import os
import sys

# 使提交目录能引用后端 safellm 包（实际部署时由评测环境处理路径）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml  # noqa: E402
from safellm.attacks import get_attacker  # noqa: E402

# 读取本提交配置。
with open(os.path.join(_HERE, "config.yaml"), "r", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f) or {}
_ATTACKER = None


def _get_attacker(attack_model=None):
    """惰性构造攻击器（评测环境注入 attack_model）。"""
    global _ATTACKER
    if _ATTACKER is None:
        _ATTACKER = get_attacker(
            _CFG.get("attack_method", "direct"),
            attack_model=attack_model,
            max_queries=_CFG.get("max_attack_queries", 20),
            max_turns=_CFG.get("max_attack_turns", 8),
        )
    return _ATTACKER


def attack(env, task: dict, attack_model=None) -> dict:
    """赛题 §5.5 攻击方提交接口。

    Args:
        env: AttackEnv，仅可通过 env.query 探测目标模型（黑盒，§2.1）。
        task: 当前高风险任务。
        attack_model: 评测环境注入的辅助攻击 LLM（可选）。
    """
    attacker = _get_attacker(attack_model)
    return attacker.attack(env, task)
