"""攻击种子模板加载，vendor 自 AISafetyLab ``initialization/init_templates.json``。

提供：
- ``seed_templates()``：角色扮演/DAN 等模板列表（用 ``{query}`` 占位）。
- ``pair_system_prompt()``：PAIR 攻击 LLM 的系统提示。
- ``pair_seed_prompt()``：PAIR 初始用户提示。
"""
from __future__ import annotations

import json
import os

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "init_templates.json")
_DATA: dict | None = None


def _load() -> dict:
    global _DATA
    if _DATA is None:
        with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            _DATA = json.load(f)
    return _DATA


def _fmt(template: str, query: str) -> str:
    return template.replace("{query}", query).replace("{goal}", query)


def _flatten(entries: list) -> list[str]:
    """将模板条目展平为字符串列表：str 直接保留，dict 取其所有 str 值。"""
    out: list[str] = []
    for e in entries:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, dict):
            for v in e.values():
                if isinstance(v, str):
                    out.append(v)
    return out


def seed_templates() -> list[str]:
    """返回一组代表性越狱模板（含 {query} 占位）。

    取自 AISafetyLab init_templates 的 default / AutoDAN / Jailbroken 子集，
    覆盖角色扮演、开发者模式、指令覆盖、前缀注入等家族。
    """
    atk = _load()["attack"]
    out: list[str] = []
    out += _flatten(atk.get("default", []))
    # AutoDAN 模板较多，取若干代表性片段。
    out += _flatten(atk.get("AutoDAN", []))[:8]
    out += _flatten(atk.get("Jailbroken", []))[:8]
    # 去重并过滤空模板。
    seen, uniq = set(), []
    for t in out:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def pair_templates() -> list[str]:
    """返回 PAIR 的系统提示与种子提示（原始含 {query} 占位）。"""
    return list(_load()["attack"].get("PAIR", []))


def pair_system_prompt() -> str:
    return pair_templates()[0] if pair_templates() else ""


def pair_seed_prompt() -> str:
    pts = pair_templates()
    return pts[-1] if pts else ""


def format_template(template: str, query: str) -> str:
    return _fmt(template, query)
