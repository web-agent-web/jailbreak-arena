"""单 攻击×防御 调试：在少量高风险样本上运行并打印明细。

用法：
    python scripts/run_single.py --attack template --defense self_reminder --config config.yaml
    python scripts/run_single.py --attack pair --defense none --samples 3 --mock
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safellm.attacks import get_attacker
from safellm.data import load_harmful_dataset
from safellm.defenses import get_defender
from safellm.env import AttackEnv, DefenseEnv
from safellm.judge import LLMJudge
from safellm.models import create_model
from safellm.utils import ConfigManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--attack", default="direct")
    p.add_argument("--defense", default="none")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--mock", action="store_true", help="强制 Mock 模式")
    args = p.parse_args()

    cm = ConfigManager(args.config)
    cfg = cm.config
    if args.mock:
        cfg["mock_mode"] = True

    target = create_model("target", cfg)
    attack_model = create_model("attack", cfg)
    judge = LLMJudge(create_model("judge", cfg))

    data_dir = cfg["data"].get("data_dir", "data")
    samples = load_harmful_dataset(
        data_dir, cfg["data"]["harmful"], args.samples
    )

    defender = get_defender(args.defense, aux_model=attack_model)
    attacker = get_attacker(
        args.attack, attack_model=attack_model,
        max_queries=cfg["limits"]["max_attack_queries"],
        max_turns=cfg["limits"]["max_attack_turns"],
    )

    lim = cfg["limits"]
    print(f"\n=== 单阵调试: attack={args.attack} vs defense={args.defense} "
          f"({len(samples)} samples, mock={cfg['mock_mode']}) ===\n")

    for s in samples:
        env = AttackEnv(
            defend_fn=defender.defend, target_model=target,
            system_prompt=cfg["system_prompt"],
            max_queries=lim["max_attack_queries"], max_turns=lim["max_attack_turns"],
            max_defense_queries=lim["max_defense_queries"],
            max_target_tokens=lim.get("max_target_tokens", 1024),
        )
        task = {"sample_id": s.sample_id, "goal": s.goal,
                "language": getattr(s, "language", "en"),
                "context": getattr(s, "context", None)}
        result = attacker.attack(env, task)
        messages = result["messages"]

        # 重放
        denv = DefenseEnv(target, cfg["system_prompt"],
                          lim["max_defense_queries"], lim.get("max_target_tokens", 1024))
        response = defender.defend(denv, {"messages": messages}).get("response", "")
        v = judge.judge(s.sample_id, s.goal, response)

        print(f"[{s.sample_id}] goal={s.goal[:60]!r}")
        print(f"  final_prompt={messages[-1]['content'][:80]!r}")
        print(f"  response={response[:80]!r}")
        print(f"  verdict: success={v.success} "
              f"(goal={v.J_goal} unsafe={v.J_unsafe} refusal={v.J_refusal} "
              f"rel={v.J_relevance} act={v.J_actionable}) queries={env.query_count}\n")


if __name__ == "__main__":
    main()
