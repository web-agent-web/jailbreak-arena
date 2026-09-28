"""从 HuggingFace 下载 AISafetyLab 数据集并归一化为赛题 schema 写入本地。

依赖 ``datasets`` 库。下载源：``thu-coai/AISafetyLab_Datasets``。

归一化目标 schema（赛题 §4.1）：
- 高风险: {"sample_id","goal","risk_category","language","context","target"}
- 正常:   {"sample_id","prompt","expected","language"}
"""
from __future__ import annotations

import json
import os

from ..logging import logger

HF_REPO = "thu-coai/AISafetyLab_Datasets"

# (HF配置名, split, 输出文件名) 列表。输出文件名统一小写，与 config 数据集清单一致。
HARMFUL_DATASETS = [
    ("advbench", "test", "advbench"),
    ("harmbench", "standard", "harmbench"),
    ("JBB", "test", "jbb"),
]
BENIGN_DATASETS = [
    ("xstest", "safe", "xstest"),
]


def _write_jsonl(path: str, items: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    logger.info(f"写入 {len(items)} 条 -> {path}")


def _normalize_harmful(name: str, row: dict, idx: int) -> dict:
    goal = row.get("query") or row.get("goal") or ""
    target = row.get("target") or ""
    category = row.get("Category") or row.get("category") or "general"
    language = "zh" if any("一" <= ch <= "鿿" for ch in goal) else "en"
    return {
        "sample_id": f"{name}-{idx:04d}",
        "goal": goal,
        "risk_category": category,
        "language": language,
        "context": None,
        "target": target,
    }


def _normalize_benign(name: str, row: dict, idx: int) -> dict:
    prompt = row.get("prompt") or row.get("query") or ""
    language = "zh" if any("一" <= ch <= "鿿" for ch in prompt) else "en"
    sid = row.get("id") or row.get("id_v1") or f"{name}-{idx:04d}"
    return {
        "sample_id": f"{name}-{sid}",
        "prompt": prompt,
        "expected": "safe",  # 正常任务集期望模型正常作答而非拒绝
        "language": language,
    }


def download_all(data_dir: str, harmful: list[tuple[str, str, str]] | None = None,
                 benign: list[tuple[str, str, str]] | None = None) -> None:
    """下载并归一化全部数据集到 ``data_dir`` 下的 harmful/ 与 benign/。

    每项为 ``(hf_name, split, out_name)``：``hf_name`` 为 HuggingFace 配置名，
    ``out_name`` 为输出文件名（不含扩展名）。
    """
    from datasets import load_dataset

    harmful = harmful or HARMFUL_DATASETS
    benign = benign or BENIGN_DATASETS

    for hf_name, split, out_name in harmful:
        try:
            ds = load_dataset(HF_REPO, hf_name, split=split)
        except Exception as e:  # noqa: BLE001
            logger.error(f"下载 {hf_name}/{split} 失败: {e}")
            continue
        items = [_normalize_harmful(out_name, dict(r), i) for i, r in enumerate(ds)]
        _write_jsonl(os.path.join(data_dir, "harmful", f"{out_name}.jsonl"), items)

    for hf_name, split, out_name in benign:
        try:
            ds = load_dataset(HF_REPO, hf_name, split=split)
        except Exception as e:  # noqa: BLE001
            logger.error(f"下载 {hf_name}/{split} 失败: {e}")
            continue
        items = [_normalize_benign(out_name, dict(r), i) for i, r in enumerate(ds)]
        _write_jsonl(os.path.join(data_dir, "benign", f"{out_name}.jsonl"), items)

    logger.info("数据集下载与归一化完成。")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="下载并归一化 AISafetyLab 数据集到本地")
    p.add_argument("--data_dir", default="data", help="数据根目录")
    args = p.parse_args()
    download_all(args.data_dir)
