"""The tuning results, generated from results/part3/tuning (PREREG Section 5).

  venv3/bin/python scripts/part3/report_tuning.py      # writes docs/TUNING_part3.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUNING = ROOT / "results/part3/tuning"
NAMES = {"oe": "OE", "oe_fixed": "OE-fixed", "ppo_fixed": "PPO-fixed", "ppo_dr": "PPO-DR", "es_dr": "ES-DR"}


def fmt(v):
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def main():
    out = ["# Part 3 tuning results\n",
           "We selected settings separately for each method using validation performance. "
           "The selected configurations are in `configs/part3/tuned.yaml`.\n",
           "Each trial trained for 5 × 10⁷ agent-steps (a quarter of the final budget) and was scored on the 24 "
           "validation environments only: mean of (competence − random-policy competence), zero-shot, 8 episodes. "
           "The highest score was chosen, with ties resolved by trial number. Each setting was evaluated once, "
           "so tuning scores include variation between runs.\n"]
    summaries = {m: json.loads((TUNING / f"tuning_{m}.json").read_text()) for m in NAMES}
    hardware = {m: tuple(s["hardware"].get("gpus", [])) for m, s in summaries.items()}
    if len(set(hardware.values())) == 1:
        out.append("Hardware: " + "; ".join(next(iter(hardware.values()))) + ".\n")
    else:
        out.append("Hardware: " + "; ".join(f"{NAMES[m]}: {', '.join(gpus)}" for m, gpus in hardware.items()) + ".\n")
    for m, s in summaries.items():
        keys = sorted(s["trials"][0]["settings"])
        out.append(f"## {NAMES[m]}\n")
        out.append("| Trial | seed | " + " | ".join(keys) + " | score | chosen |")
        out.append("|---|---|" + "---|" * len(keys) + "---|---|")
        for t in s["trials"]:
            sc = "failed" if t.get("failed") else f"{t['score']:.3f}"
            out.append(f"| {t['trial']} | {t['seed']} | " + " | ".join(fmt(t["settings"][k]) for k in keys)
                       + f" | {sc} | {'yes' if t['trial'] == s['chosen']['trial'] else ''} |")
        out.append("")
    timing = TUNING / "spend.json"
    if timing.exists():
        parts = json.loads(timing.read_text())["parts"]
        out += ["## Compute time\n", "| Part | Machine hours |", "|---|---|"]
        for part, r in parts.items():
            out.append(f"| {part} | {r['pod_hours']:.2f} |")
    (ROOT / "docs/TUNING_part3.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
