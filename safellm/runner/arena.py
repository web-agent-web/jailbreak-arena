"""全对阵编排器（赛题 §9/§12）。

将攻击方法与防御方法解耦装配：每个 (攻击, 防御) 对在每条高风险样本上独立运行，
攻击方只见 AttackEnv、防御方只见 DefenseEnv。最终重放（§5.6）经防御保护的
目标模型得到正式回复，由裁判评分，输出明细与汇总报告。
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any

from ..attacks import get_attacker, list_attacks
from ..data import load_benign_dataset, load_harmful_dataset
from ..defenses import get_defender, list_defenses
from ..env import AttackEnv, DefenseEnv
from ..judge import LLMJudge
from ..logging import logger
from ..models import create_model
from .metrics import aggregate_report


class Arena:
    """攻防全对阵编排器。"""

    def __init__(self, config: dict):
        self.config = config
        lim = config["limits"]
        self.system_prompt = config["system_prompt"]
        self.max_attack_queries = lim["max_attack_queries"]
        self.max_attack_turns = lim["max_attack_turns"]
        self.max_defense_queries = lim["max_defense_queries"]
        self.max_target_tokens = lim.get("max_target_tokens", 1024)

        # 模型：target / attack（兼防御辅助）/ judge（独立）
        self.target_model = create_model("target", config)
        self.attack_model = create_model("attack", config)
        self.judge_model = create_model("judge", config)
        self.judge = LLMJudge(self.judge_model)

        # 数据
        data_cfg = config["data"]
        data_dir = data_cfg.get("data_dir", "data")
        self.harmful = load_harmful_dataset(
            data_dir, data_cfg["harmful"], data_cfg.get("harmful_samples"),
        )
        self.benign = load_benign_dataset(
            data_dir, data_cfg["benign"], data_cfg.get("benign_samples"),
        )

        self.attacks: list[str] = config["attacks"]
        self.defenses: list[str] = config["defenses"]
        self.baseline_attacks: list[str] = config.get("phase1_baseline_attacks", ["direct"])
        self.baseline_defenses: list[str] = config.get("phase1_baseline_defenses", ["none"])
        self.output_dir = config.get("output_dir", "results")

    # ---- 构造攻防组件 ----
    def _build_defender(self, defense_name: str):
        # 防御辅助 LLM 使用 attack_model（与 judge 独立，避免自审自判）。
        return get_defender(defense_name, aux_model=self.attack_model)

    def _build_attacker(self, attack_name: str):
        return get_attacker(
            attack_name, attack_model=self.attack_model,
            max_queries=self.max_attack_queries, max_turns=self.max_attack_turns,
        )

    def _build_attack_env(self, defend_fn) -> AttackEnv:
        return AttackEnv(
            defend_fn=defend_fn,
            target_model=self.target_model,
            system_prompt=self.system_prompt,
            max_queries=self.max_attack_queries,
            max_turns=self.max_attack_turns,
            max_defense_queries=self.max_defense_queries,
            max_target_tokens=self.max_target_tokens,
        )

    def _build_defense_env(self) -> DefenseEnv:
        return DefenseEnv(
            target_model=self.target_model,
            system_prompt=self.system_prompt,
            max_queries=self.max_defense_queries,
            max_target_tokens=self.max_target_tokens,
        )

    def _replay(self, defender, messages: list[dict]) -> str:
        """赛题 §5.6 最终重放：将最终 messages 经防御保护的目标模型执行。"""
        env = self._build_defense_env()
        try:
            result = defender.defend(env, {"messages": messages})
            return result.get("response", "")
        except Exception as e:  # noqa: BLE001
            logger.error(f"重放失败: {e}")
            return ""

    # ---- 单样本攻防 ----
    def run_sample(self, attack_name: str, defense_name: str, sample) -> dict:
        defender = self._build_defender(defense_name)
        attacker = self._build_attacker(attack_name)
        env = self._build_attack_env(defender.defend)

        task = {
            "sample_id": sample.sample_id,
            "goal": sample.goal,
            "language": getattr(sample, "language", "en"),
            "context": getattr(sample, "context", None),
        }
        t0 = time.time()
        try:
            result = attacker.attack(env, task)
            messages = result.get("messages", [])
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{attack_name} vs {defense_name}] sample {sample.sample_id} 攻击异常: {e}")
            messages = [{"role": "user", "content": task["goal"]}]

        # 最终重放 + 裁判
        response = self._replay(defender, messages)
        verdict = self.judge.judge(sample.sample_id, task["goal"], response)

        rec = verdict.to_dict()
        rec.update({
            "attack": attack_name,
            "defense": defense_name,
            "final_messages": messages,
            "attack_queries": env.query_count,
            "elapsed": round(time.time() - t0, 2),
        })
        return rec

    # ---- 单防御正常能力评测 ----
    def run_capability(self, defense_name: str) -> list[dict]:
        defender = self._build_defender(defense_name)
        records = []
        for sample in self.benign:
            env = self._build_defense_env()
            request = {"messages": [{"role": "user", "content": sample.prompt}]}
            try:
                resp = defender.defend(env, request).get("response", "")
            except Exception as e:  # noqa: BLE001
                logger.error(f"[{defense_name}] 能力评测异常 {sample.sample_id}: {e}")
                resp = ""
            cap = self.judge.judge_capability(sample.sample_id, sample.prompt, resp)
            records.append({
                "defense": defense_name,
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "response": resp,
                "appropriate": cap["appropriate"],
                "over_refusal": cap["over_refusal"],
            })
        return records

    # ---- 全对阵 ----
    def run_all(self, detail_path: str | None = None) -> dict:
        """运行全部 攻击×防御×高风险样本 + 防御×正常样本，输出报告。"""
        os.makedirs(self.output_dir, exist_ok=True)
        detail_path = detail_path or os.path.join(self.output_dir, "detail.jsonl")
        cap_records: list[dict] = []
        detail_records: list[dict] = []

        # 第一阶段（§9.1）：基线池 + 第二阶段（§9.2）：全对阵，合并为全矩阵。
        try:
            from tqdm import tqdm
            pair_iter = tqdm(
                [(a, d) for a in self.attacks for d in self.defenses],
                desc="攻防对阵", unit="pair",
            )
        except Exception:  # noqa: BLE001
            pair_iter = [(a, d) for a in self.attacks for d in self.defenses]

        with open(detail_path, "w", encoding="utf-8") as fout:
            for attack_name, defense_name in pair_iter:
                for sample in self.harmful:
                    rec = self.run_sample(attack_name, defense_name, sample)
                    detail_records.append(rec)
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logger.info(
                    f"ASR[{attack_name} vs {defense_name}] = "
                    f"{_asr(detail_records, attack_name, defense_name):.2%}"
                )

        # 正常能力评测
        for defense_name in self.defenses:
            cap_records.extend(self.run_capability(defense_name))
        with open(os.path.join(self.output_dir, "capability.jsonl"), "w",
                  encoding="utf-8") as f:
            for r in cap_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 汇总报告
        report = aggregate_report(
            detail_records, cap_records,
            self.attacks, self.defenses,
            self.baseline_attacks, self.baseline_defenses,
        )
        self._write_reports(report)
        return report

    # ---- 报告输出（§12） ----
    def _write_reports(self, report: dict) -> None:
        # 汇总 JSON
        with open(os.path.join(self.output_dir, "report.json"), "w",
                  encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 攻防热力矩阵 CSV（行=攻击，列=防御）
        csv_path = os.path.join(self.output_dir, "asr_heatmap.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["attack \\ defense"] + self.defenses)
            for a in self.attacks:
                row = report["asr_matrix"].get(a, {})
                w.writerow([a] + [f"{row.get(d, 0.0):.3f}" for d in self.defenses])

        # 公开榜单 Markdown（§12.3）
        md = ["# 攻防对抗公开榜单\n"]
        md.append("## 攻击总排行榜\n")
        md.append("| 排名 | 攻击方法 | CoreASR | 平均ASR |")
        md.append("|---|---|---|---|")
        for i, e in enumerate(report["leaderboard"]["attack"], 1):
            md.append(f"| {i} | {e['attack']} | {e['core_asr']:.2%} | {e['avg_asr']:.2%} |")
        md.append("\n## 防御总排行榜\n")
        md.append("| 排名 | 防御方法 | ResidualASR(越低越好) | UR(越高越好) |")
        md.append("|---|---|---|---|")
        for i, e in enumerate(report["leaderboard"]["defense"], 1):
            md.append(f"| {i} | {e['defense']} | {e['residual_asr']:.2%} | {e['ur']:.2%} |")
        with open(os.path.join(self.output_dir, "leaderboard.md"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        logger.info(f"报告已输出至 {self.output_dir}/ (report.json, asr_heatmap.csv, "
                    f"leaderboard.md, detail.jsonl, capability.jsonl)")


def _asr(records: list[dict], attack: str, defense: str) -> float:
    vals = [r["success"] for r in records if r["attack"] == attack and r["defense"] == defense]
    return sum(vals) / len(vals) if vals else 0.0
