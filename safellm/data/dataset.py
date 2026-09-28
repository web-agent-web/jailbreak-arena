"""本地数据集加载。

读取 ``data/harmful/*.jsonl`` 与 ``data/benign/*.jsonl``，返回 ``Example`` 列表。
高风险样本统一为赛题 §4.1 schema：
    {"sample_id", "goal", "risk_category", "language", "context", "target"}
正常样本统一为：
    {"sample_id", "prompt", "expected", "language"}
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from ..logging import logger
from .example import Example


def _read_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _take(items: list, limit: int | None) -> list:
    if limit is None:
        return items
    return items[:limit]


def load_harmful_dataset(data_dir: str, names: Iterable[str],
                         limit: int | None = None) -> list[Example]:
    """加载高风险任务集。

    Args:
        data_dir: 项目数据根目录（含 ``harmful/`` 子目录）。
        names: 数据集名列表，如 ``["advbench", "harmbench", "jbb"]``。
        limit: 每个数据集最多取多少条（便于快速调试）。
    """
    examples: list[Example] = []
    harmful_dir = os.path.join(data_dir, "harmful")
    for name in names:
        path = os.path.join(harmful_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"高风险数据集文件不存在: {path}（请先运行 download_data.py）")
            continue
        for d in _take(_read_jsonl(path), limit):
            examples.append(Example(**d))
    logger.info(f"加载高风险样本 {len(examples)} 条（来自 {list(names)}）")
    return examples


def load_benign_dataset(data_dir: str, names: Iterable[str],
                        limit: int | None = None) -> list[Example]:
    """加载正常任务集（用于正常能力保持率/误拒绝率评测，§4.2/§8）。"""
    examples: list[Example] = []
    benign_dir = os.path.join(data_dir, "benign")
    for name in names:
        path = os.path.join(benign_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"正常数据集文件不存在: {path}（请先运行 download_data.py）")
            continue
        for d in _take(_read_jsonl(path), limit):
            examples.append(Example(**d))
    logger.info(f"加载正常样本 {len(examples)} 条（来自 {list(names)}）")
    return examples


def load_dataset(data_dir: str, kind: str, names: Iterable[str],
                 limit: int | None = None) -> list[Example]:
    """统一入口：``kind`` 为 ``"harmful"`` 或 ``"benign"``。"""
    if kind == "harmful":
        return load_harmful_dataset(data_dir, names, limit)
    return load_benign_dataset(data_dir, names, limit)
