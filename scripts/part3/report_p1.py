"""Write docs/PILOT_part3_P1.md from the saved stage P1 results (no numbers typed by hand).

Reads results/part3/pilot/<part>/ (each machine's logs and run folders, copied off the pods),
results/part3/pilot/merged/pilot_summary.json (pilot.py --merge over the parts) and
results/part3/pilot/spend.json.

  python3 scripts/part3/report_p1.py
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/part3/pilot"
PARTS = {"oe": ["oe"], "oefixed": ["oe_fixed"], "es_ppo": ["es_dr", "ppo_fixed", "ppo_dr"]}
METHODS = ["oe", "oe_fixed", "ppo_fixed", "ppo_dr", "es_dr"]
NAMES = {"oe": "OE", "oe_fixed": "OE-fixed", "ppo_fixed": "PPO-fixed", "ppo_dr": "PPO-DR", "es_dr": "ES-DR"}
F = "%Y-%m-%dT%H:%M:%SZ"


def part_of(method):
    return next(p for p, ms in PARTS.items() if method in ms)


def status_times(part):
    t = {}
    for ln in (R / part / "logs/status.txt").read_text().splitlines():
        ts, ev = ln.split(" ", 1)
        t.setdefault(ev, ts)
    return t


def minutes(a, b):
    return (datetime.strptime(b, F) - datetime.strptime(a, F)).total_seconds() / 60


def progress(method, seed, repeat=False):
    d = R / part_of(method) / "results" / f"{method}_seed{seed}{'_repeat' if repeat else ''}"
    return [json.loads(x) for x in (d / "progress.jsonl").read_text().splitlines() if x.strip()]


def tenths(lines, key_fn, budget):
    """Mean of key_fn over the chunks ending in each tenth of the budget (by agent_steps_total)."""
    out = []
    for k in range(10):
        lo, hi = budget * k / 10, budget * (k + 1) / 10
        sel = [key_fn(ln) for ln in lines if lo < ln["agent_steps_total"] <= hi or (k == 9 and ln["agent_steps_total"] > hi)]
        sel = [v for v in sel if v is not None]
        out.append(statistics.fmean(sel) if sel else float("nan"))
    return out


def fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def main():
    s = json.loads((R / "merged/pilot_summary.json").read_text())
    spend = json.loads((R / "spend.json").read_text())
    M = s["methods"]
    seeds = s["seeds"]
    budget = s["budget"]
    out = []
    w = out.append
    w("# Part 3 pilot, stage P1: results\n")
    w("This pilot measured training throughput, competence, adaptation and population dynamics before the final experiment.\n")
    w(f"Settings: training budget {budget:,} agent-steps; adaptation budget {s['adapt_budget']:,} agent-steps per world "
      f"with 10 checkpoints; seeds {', '.join(map(str, seeds))} (and {seeds[0]} again for the repeat check); "
      f"{s['n_val_envs']} validation worlds; {2 * s['n_shift_envs']} pilot stand-in worlds ({s['n_shift_envs']} from "
      f"the T2 ranges, then the same number from the T3 ranges). Summaries use {len(seeds)} runs per method.\n")

    w("## Machines and compute time\n")
    w("| Part | Methods | Pod created → set up | Set up → all done | Machine time |")
    w("|---|---|---|---|---|")
    for part, ms in PARTS.items():
        t = status_times(part)
        sp = spend["parts"][part]
        w(f"| {part} | {', '.join(NAMES[m] for m in ms)} | {minutes(sp['created_at'], t['setup_ok']):.1f} min | "
          f"{minutes(t['setup_ok'], t['all_done']):.1f} min | {sp['pod_hours']:.2f} h |")
    w("")

    w("## Repeatability on the GPU\n")
    w("Seed " + str(seeds[0]) + " was trained twice on the same machine. \"Same\" means the two trained policies are bit-identical.\n")
    w("| " + " | ".join(NAMES[m] for m in METHODS) + " |")
    w("|" + "---|" * len(METHODS))
    w("| " + " | ".join("same" if M[m]["repeat_bit_identical"] else "**different**" for m in METHODS) + " |\n")

    w("## Training: time and speed\n")
    w("Wall time of each 2 × 10⁸ agent-step training run (including compiling in the first run of each machine) and "
      "living agent-steps per second over the whole run.\n")
    w("| Method | " + " | ".join(f"seed {x}: minutes" for x in seeds) + " | agent-steps per second (mean) |")
    w("|---|" + "---|" * (len(seeds) + 1))
    for m in METHODS:
        runs = M[m]["runs"]
        w(f"| {NAMES[m]} | " + " | ".join(f"{r['train']['wall_seconds'] / 60:.1f}" for r in runs) +
          f" | {statistics.fmean(r['agent_steps_per_second'] for r in runs):,.0f} |")
    w("")

    w("## Zero-shot competence\n")
    w("Food per agent in 500-step episodes (16 agents), mean over runs; the random policy on the same episodes in brackets.\n")
    n_val = s["n_val_envs"]
    w("| Method | " + " | ".join(f"validation {j + 1}" for j in range(n_val)) + " | pilot stand-ins (mean of all) |")
    w("|---|" + "---|" * (n_val + 1))
    for m in METHODS:
        runs = M[m]["runs"]
        cells = []
        for j in range(n_val):
            c = statistics.fmean(r["validation"][j]["competence"] for r in runs)
            rnd = statistics.fmean(r["validation"][j]["random_competence"] for r in runs)
            cells.append(f"{c:.2f} ({rnd:.2f})")
        sz = statistics.fmean(row["zero_shot"]["competence"] for r in runs for row in r["shift"])
        srnd = statistics.fmean(row["zero_shot"]["random_competence"] for r in runs for row in r["shift"])
        w(f"| {NAMES[m]} | " + " | ".join(cells) + f" | {sz:.2f} ({srnd:.2f}) |")
    w("")
    w("Persistence (ecological mode, 32 agents of the policy, 5,000 steps, no floor): runs that went extinct, out of all "
      "(runs × worlds).\n")
    w("| Method | validation worlds | pilot stand-ins |")
    w("|---|---|---|")
    for m in METHODS:
        runs = M[m]["runs"]
        ve = sum(v["persistence"]["extinct"] for r in runs for v in r["validation"])
        vn = sum(1 for r in runs for _ in r["validation"])
        se = sum(row["zero_shot"]["persistence"]["extinct"] for r in runs for row in r["shift"])
        sn = sum(1 for r in runs for _ in r["shift"])
        w(f"| {NAMES[m]} | {ve} of {vn} | {se} of {sn} |")
    w("")

    w("## Adaptation on the pilot stand-in worlds\n")
    w("Competence before adaptation (point 0) and after the whole adaptation budget (point 10), mean over runs and "
      "worlds; and the wall time of one adaptation.\n")
    w("| Method | point 0 | point 10 | change | minutes per adaptation (mean) |")
    w("|---|---|---|---|---|")
    for m in METHODS:
        rows = [row for r in M[m]["runs"] for row in r["shift"]]
        p0 = statistics.fmean(row["adaptation"]["curve"][0]["competence"] for row in rows)
        p10 = statistics.fmean(row["adaptation"]["curve"][-1]["competence"] for row in rows)
        wall = statistics.fmean(row["adaptation"]["wall_seconds"] for row in rows) / 60
        w(f"| {NAMES[m]} | {p0:.2f} | {p10:.2f} | {p10 - p0:+.2f} | {wall:.1f} |")
    w("")
    w("**Primary measure (as in the pilot):** normalized adaptation AUC per world, where 0 is the random policy and 1 "
      "the best competence any method reached on that world in this pilot; per run, the mean over the "
      f"{2 * s['n_shift_envs']} worlds.\n")
    w("| Method | " + " | ".join(f"seed {x}" for x in seeds) + " | mean | SD between runs |")
    w("|---|" + "---|" * (len(seeds) + 2))
    for m in METHODS:
        per_run = [float(np.nanmean(v)) for v in M[m]["normalized_auc"]]
        w(f"| {NAMES[m]} | " + " | ".join(f"{v:.3f}" for v in per_run) +
          f" | {statistics.fmean(per_run):.3f} | {statistics.stdev(per_run):.3f} |")
    w("")
    if "power" in s:
        p = s["power"]
        w(f"**Power (OE vs PPO-DR; {p['note']}):** per-run SD OE {p['per_run_sd']['oe']:.3f}, PPO-DR "
          f"{p['per_run_sd']['ppo_dr']:.3f}. Runs per method for an expected 95% CI half-width of 0.025: "
          f"{p['runs_needed']['0.025']}; of 0.05: {p['runs_needed']['0.05']} (`pilot.runs_needed`, at least 10).\n")

    w("## OE populations during training\n")
    w("Per tenth of the training budget, mean over seeds " + ", ".join(map(str, seeds)) + ": living agents, births "
      "by the agents themselves and floor births (copies added to keep 16 alive) per 1,000 steps, and food per "
      "living agent-step.\n")
    for m in ("oe", "oe_fixed"):
        w(f"**{NAMES[m]}**\n")
        w("| | " + " | ".join(f"{k + 1}/10" for k in range(10)) + " |")
        w("|---|" + "---|" * 10)
        series = {"living agents": lambda ln: ln["mean_alive"],
                  "own births per 1,000 steps": lambda ln: 1000 * ln["births"] / ln["steps"],
                  "floor births per 1,000 steps": lambda ln: 1000 * ln["floor_births"] / ln["steps"],
                  "food per agent-step": lambda ln: ln["food"] / max(ln["agent_steps"], 1)}
        for label, fn in series.items():
            vals = np.nanmean([tenths(progress(m, x), fn, budget) for x in seeds], axis=0)
            w(f"| {label} | " + " | ".join(fmt(v, 3 if 'food' in label else 1) for v in vals) + " |")
        w("")

    w("## Training signal over time\n")
    w("Per tenth of the training budget, mean over seeds. These are the methods' own training numbers (in their "
      "training worlds), not competence on held-out worlds; they show whether training is still rising at 2 × 10⁸.\n")
    signals = {"oe": ("food per agent-step", lambda ln: ln["food"] / max(ln["agent_steps"], 1)),
               "oe_fixed": ("food per agent-step", lambda ln: ln["food"] / max(ln["agent_steps"], 1)),
               "ppo_fixed": ("food per living agent-step", lambda ln: ln["food_per_valid_step"]),
               "ppo_dr": ("food per living agent-step", lambda ln: ln["food_per_valid_step"]),
               "es_dr": ("mean candidate score", lambda ln: ln["mean_score"])}
    w("| Method | measure | " + " | ".join(f"{k + 1}/10" for k in range(10)) + " |")
    w("|---|---|" + "---|" * 10)
    for m in METHODS:
        label, fn = signals[m]
        vals = np.nanmean([tenths(progress(m, x), fn, budget) for x in seeds], axis=0)
        w(f"| {NAMES[m]} | {label} | " + " | ".join(fmt(v, 3) for v in vals) + " |")
    w("")
    (ROOT / "docs/PILOT_part3_P1.md").write_text("\n".join(out))
    print("wrote docs/PILOT_part3_P1.md")


if __name__ == "__main__":
    main()
