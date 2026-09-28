"""运行全对阵矩阵并输出报告（赛题 §9/§12）。

用法：
    python scripts/run_arena.py --config config.yaml
    python scripts/run_arena.py --mock          # 强制 Mock 模式快速跑通
    python scripts/run_arena.py --attacks direct,template --defenses none,self_reminder
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safellm.runner import Arena
from safellm.utils import ConfigManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--mock", action="store_true", help="强制 Mock 模式")
    p.add_argument("--attacks", default=None, help="逗号分隔的攻击方法子集")
    p.add_argument("--defenses", default=None, help="逗号分隔的防御方法子集")
    p.add_argument("--harmful_samples", type=int, default=None)
    p.add_argument("--benign_samples", type=int, default=None)
    args = p.parse_args()

    cm = ConfigManager(args.config)
    cfg = cm.config
    if args.mock:
        cfg["mock_mode"] = True
    if args.attacks:
        cfg["attacks"] = [a.strip() for a in args.attacks.split(",")]
    if args.defenses:
        cfg["defenses"] = [d.strip() for d in args.defenses.split(",")]
    if args.harmful_samples is not None:
        cfg["data"]["harmful_samples"] = args.harmful_samples
    if args.benign_samples is not None:
        cfg["data"]["benign_samples"] = args.benign_samples

    arena = Arena(cfg)
    report = arena.run_all()

    print("\n========== 攻击排行榜 ==========")
    for i, e in enumerate(report["leaderboard"]["attack"], 1):
        print(f"  {i}. {e['attack']:<14} CoreASR={e['core_asr']:.2%}  avg={e['avg_asr']:.2%}")
    print("\n========== 防御排行榜 ==========")
    for i, e in enumerate(report["leaderboard"]["defense"], 1):
        print(f"  {i}. {e['defense']:<18} ResidualASR={e['residual_asr']:.2%}  UR={e['ur']:.2%}")
    print(f"\n报告已输出至: {cfg['output_dir']}/")


if __name__ == "__main__":
    main()
