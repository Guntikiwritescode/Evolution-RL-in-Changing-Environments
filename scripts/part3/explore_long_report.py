"""Generate the exploratory longer-training report and its diagnostic figure.

Exploratory: decided on 4 Nov 2026 after the final results. The reading rule below was written and
committed on 4 Nov 2026 while the machines were training, before any of their curves existed.

Reading rule (fixed before the results):
  For each run, gain = mean validation score at tenths 9-10 (9e8 and 1e9 agent-steps)
                       minus mean at tenths 1-2 (1e8 and 2e8 agent-steps).
  A gain above GAIN_CLEAR = 4.3 points is "clear": twice the noise SD of that difference (2.17),
  estimated from the 10 final OE runs' curves (SD of tenth-to-tenth changes 3.07 / sqrt(2) per point).
  "more training helped" if all runs have a clear gain; "did not help" if none has;
  otherwise "mixed". Three runs: no test, no interval; described, not tested.

  .venv-validation/bin/python scripts/part3/explore_long_report.py
"""
from __future__ import annotations

import argparse
import json
import statistics as S
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "results/part3/explore_long"
FINAL = ROOT / "results/part3/final"
SEEDS = (5000, 5001, 5002)
GAIN_CLEAR = 4.3
NOISE_SD = 2.17


def gain(scores: list[float]) -> float:
    """Mean of tenths 9-10 minus mean of tenths 1-2 (scores are the 10 tenths in order)."""
    if len(scores) != 10:
        raise ValueError("need the 10 tenths")
    return (scores[8] + scores[9]) / 2 - (scores[0] + scores[1]) / 2


def reading(gains: list[float]) -> str:
    clear = [g > GAIN_CLEAR for g in gains]
    if all(clear):
        return "more training helped"
    if not any(clear):
        return "did not help"
    return "mixed"


def curve(run: Path) -> list[dict]:
    return json.load(open(run / "curves.json"))["points"]


def progress(run: Path) -> list[dict]:
    return [json.loads(x) for x in open(run / "progress.jsonl") if x.strip()]


def first_difference(a: list[dict], b: list[dict]):
    strip = lambda d: {k: v for k, v in d.items() if k != "wall_seconds"}   # noqa: E731
    for i, (x, y) in enumerate(zip(a, b)):
        if strip(x) != strip(y):
            return i + 1, strip(x), strip(y)
    return None


def per_tenth(lines: list[dict], budget: int, key: str) -> list[float]:
    """Mean of a progress field over the chunks inside each tenth of the budget."""
    out = []
    for k in range(10):
        lo, hi = budget * k / 10, budget * (k + 1) / 10
        v = [ln[key] for ln in lines if lo < ln["agent_steps_total"] <= hi]
        out.append(S.fmean(v) if v else float("nan"))
    return out


def figure(new: dict, final: dict, refs: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for s, pts in final.items():
        ax.plot([p["agent_steps"] / 1e8 for p in pts], [p["score"] for p in pts], color="0.75", lw=0.8,
                label="final OE runs (10), budget 2e8" if s == min(final) else None)
    for s, pts in new.items():
        ax.plot([p["agent_steps"] / 1e8 for p in pts], [p["score"] for p in pts], marker="o", lw=1.6,
                label=f"longer run, seed {s}")
    for (name, v), ls in zip(refs.items(), ("--", ":", "-.")):
        ax.axhline(v, ls=ls, lw=0.9, color="k", label=f"{name}, mean at 2e8 ({v:.1f})")
    ax.set_xlabel("training agent-steps (x 1e8)")
    ax.set_ylabel("validation score (competence - random)")
    ax.set_title("OE trained 5x longer (exploratory, 3 runs)")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 30)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.30, 0.66))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/EXPLORE_part3_long.md"))
    a = ap.parse_args()
    budget = 1_000_000_000
    runs = {s: NEW / f"oe_seed{s}" for s in SEEDS}
    missing = [s for s, r in runs.items() if not (r / "curves.json").exists() or not (r / "done.json").exists()]
    if missing:
        raise SystemExit(f"runs without done.json and curves.json: {missing}")
    new = {s: curve(r) for s, r in runs.items()}
    final = {s: curve(FINAL / f"oe_seed{s}") for s in range(5000, 5010)}
    refs = {name: S.fmean(curve(FINAL / f"{key}_seed{s}")[-1]["score"] for s in range(5000, 5010))
            for name, key in (("PPO-fixed", "ppo_fixed"), ("PPO-DR", "ppo_dr"), ("ES-DR", "es_dr"))}
    gains = {s: gain([p["score"] for p in pts]) for s, pts in new.items()}
    final_gains = [gain([p["score"] for p in pts]) for pts in final.values()]
    verdict = reading(list(gains.values()))
    fig = ROOT / "figures/part3/explore_long_curves.png"
    figure(new, final, refs, fig)

    L = ["# Part 3: longer OE training", "",
         "This exploratory experiment was designed on 4 November 2026, after the core results. "
         "It evaluated whether a larger training budget improved OE validation performance.", "",
         "## Design", "",
         f"We trained {len(runs)} OE runs for {budget:,} agent-steps, five times the core budget, using the selected "
         "OE settings. Each tenth of training was evaluated on 24 validation environments with eight episodes per "
         "environment. Evaluation was limited to validation environments.", ""]
    hardware = sorted({g for r in runs.values() for g in json.load(open(r / "run.json"))["hardware"].get("gpus", [])})
    L += ["Hardware: " + "; ".join(hardware) + ".", "",
          "Gain was defined as the mean validation score at 9e8 and 1e9 agent-steps minus the mean at 1e8 and 2e8. "
          f"The prespecified descriptive rule counted gains above {GAIN_CLEAR} points as clear: all runs above the "
          'threshold gave "more training helped", none gave "did not help", and other outcomes gave "mixed".', "",
          "## Results", "",
          "| Seed | " + " | ".join(f"{k}e8" for k in range(1, 11)) + " | gain | clear gain? |",
          "|---|" + "---|" * 12]
    for s, pts in new.items():
        L.append(f"| {s} | " + " | ".join(f"{p['score']:.2f}" for p in pts)
                 + f" | {gains[s]:+.2f} | {'yes' if gains[s] > GAIN_CLEAR else 'no'} |")
    mean_pts = [S.fmean(new[s][k]["score"] for s in SEEDS) for k in range(10)]
    L.append("| mean | " + " | ".join(f"{v:.2f}" for v in mean_pts) + f" | {S.fmean(gains.values()):+.2f} | |")
    L += ["", f"**Descriptive result: {verdict}.** "
          f"{sum(g > GAIN_CLEAR for g in gains.values())} of {len(gains)} runs exceeded the {GAIN_CLEAR}-point threshold. "
          "This small experiment does not estimate the size of a possible benefit precisely.", "",
          "At the core budget of 2e8 agent-steps, mean validation scores were "
          + "; ".join(f"{k} {v:.2f}" for k, v in refs.items())
          + f"; OE {S.fmean(p[-1]['score'] for p in final.values()):.2f} "
          f"({len(final)} runs, range {min(p[-1]['score'] for p in final.values()):.2f}–{max(p[-1]['score'] for p in final.values()):.2f}).",
          "At 2e7 agent-steps, the corresponding scores were "
          + "; ".join(f"{name} {S.fmean(curve(FINAL / f'{key}_seed{s}')[0]['score'] for s in range(5000, 5010)):.2f}"
                      for name, key in (("PPO-fixed", "ppo_fixed"), ("PPO-DR", "ppo_dr"), ("ES-DR", "es_dr")))
          + ". PPO and ES were not evaluated at 1e9 agent-steps.", "",
          "## Threshold sensitivity", "",
          f"The {GAIN_CLEAR}-point threshold was based on an estimated noise SD of {NOISE_SD}, assuming independent "
          "scores across checkpoints. A post hoc calculation of the same gain statistic on the core OE curves "
          f"gave mean {S.fmean(final_gains):+.2f}, sample SD {S.stdev(final_gains):.2f} and range "
          f"{min(final_gains):+.2f} to {max(final_gains):+.2f}; "
          f"{sum(abs(x) > GAIN_CLEAR for x in final_gains)} of {len(final_gains)} gains exceeded ±{GAIN_CLEAR}. "
          f"The threshold was therefore {GAIN_CLEAR / S.stdev(final_gains):.1f} SD of this observed spread.", "",
          "![Validation scores over training](../figures/publication/fig08_long_training.png)", ""]

    differences = {s: first_difference(progress(r), progress(FINAL / f"oe_seed{s}")) for s, r in runs.items()}
    L += ["## Comparison with the core runs", "",
          "The longer runs restarted training with the same seeds and settings on separate machines. "
          f"{sum(d is not None for d in differences.values())} of {len(runs)} differed from their core counterpart "
          "during the shared training interval.", "",
          "At 1e8 and 2e8 agent-steps, the core OE means were "
          f"{S.fmean(p[4]['score'] for p in final.values()):.2f} (range {min(p[4]['score'] for p in final.values()):.2f}–"
          f"{max(p[4]['score'] for p in final.values()):.2f}) and {S.fmean(p[9]['score'] for p in final.values()):.2f} "
          f"(range {min(p[9]['score'] for p in final.values()):.2f}–{max(p[9]['score'] for p in final.values()):.2f}); "
          f"the longer-run means were {S.fmean(new[s][0]['score'] for s in SEEDS):.2f} and {S.fmean(new[s][1]['score'] for s in SEEDS):.2f}.",
          ""]

    L += ["## Population during training (mean per tenth, from progress.jsonl)", "",
          "| Seed | measure | " + " | ".join(f"{k}/10" for k in range(1, 11)) + " |", "|---|---|" + "---|" * 10]
    for s, r in runs.items():
        lines = progress(r)
        alive = per_tenth(lines, budget, "mean_alive")
        floor = [fb / (fb + b) if (fb + b) else float("nan") for fb, b in
                 zip(per_tenth(lines, budget, "floor_births"), per_tenth(lines, budget, "births"))]
        L.append(f"| {s} | agents alive | " + " | ".join(f"{v:.1f}" for v in alive) + " |")
        L.append(f"| {s} | floor share of new agents | " + " | ".join(f"{v:.2f}" for v in floor) + " |")
    walls = {s: json.load(open(r / "done.json"))["wall_seconds"] for s, r in runs.items()}
    L += ["", "Training wall time: " + "; ".join(f"seed {s} {w / 3600:.2f} h" for s, w in walls.items()) + "."]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out} and {fig}; reading: {verdict}; gains {gains}")


if __name__ == "__main__":
    main()
