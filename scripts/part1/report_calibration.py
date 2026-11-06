"""Calibration report: every target (paper vs. rebuild) for the selected setting,
the top settings, and plots of the rebuild's ET training curves against the
paper's digitized curves.

Usage: python scripts/part1/report_calibration.py
Writes calibration/REPORT.md and figures/part1/calibration_*.png.
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAL = os.path.join(ROOT, "calibration")
FIG = os.path.join(ROOT, "figures", "part1")
BLUE, ORANGE, GREEN, PINK, GRAY = "#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#555555"


def main():
    sel = json.load(open(os.path.join(CAL, "selected.json")))
    targets = json.load(open(os.path.join(CAL, "targets.json")))
    dig = json.load(open(os.path.join(CAL, "digitized_paper_figures.json")))["figures"]
    log = pd.read_csv(os.path.join(CAL, "log.csv"))
    best = sel["selected_setting"]
    run = json.load(open(os.path.join(CAL, "runs", f"{best}__{sel['selected_from_stage']}.json")))
    curves = {k: np.array(v, dtype=float) for k, v in run["mean_curves"].items()}
    rows = run["score"]["targets"]
    os.makedirs(FIG, exist_ok=True)

    def paper(fig, c):
        d = dig[fig]["curves"][c]
        g = np.array(sorted(int(k) for k in d))
        return g, np.array([d[str(k)] for k in g])

    gens = np.arange(100)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axs[0, 0]
    g, v = paper("fig1_et_fitness", "et")
    ax.plot(g, v, color=GRAY, lw=1.5, ls="--", label="paper (digitized Fig. 1)")
    ax.plot(gens, curves["fitness"], color=BLUE, lw=1.8, label="rebuild (mean of 30 seeds)")
    ax.set(title="ET fitness (food per blob)", xlabel="generation")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    g, v = paper("fig2_et_population", "et")
    ax.plot(g, v, color=GRAY, lw=1.5, ls="--", label="paper (digitized Fig. 2)")
    ax.plot(gens, curves["population"], color=BLUE, lw=1.8, label="rebuild")
    ax.set(title="ET population", xlabel="generation")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    for t, col in (("speed", BLUE), ("sense", GREEN), ("size", ORANGE), ("reach", PINK)):
        g, v = paper("fig3_et_traits", t)
        sel_g = g <= 99
        ax.plot(g[sel_g], v[sel_g], color=col, lw=1.2, ls="--")
        ax.plot(gens, curves["mean_" + t], color=col, lw=1.8, label=t)
    ax.set(title="ET mean traits (dashed = paper Fig. 3)", xlabel="generation")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    g, v = paper("fig5_et_shannon", "et")
    ax.plot(g, v, color=GRAY, lw=1.5, ls="--", label="paper H (Fig. 5)")
    ax.plot(gens, curves["H_exact4"], color=BLUE, lw=1.8, label="rebuild H (exact types)")
    g, v = paper("fig4_et_cv2", "et")
    ax.plot(g, v, color=GRAY, lw=1.2, ls=":", label="paper CV² (Fig. 4)")
    ax.plot(gens, curves["cv2_4_samp"], color=GREEN, lw=1.8, label="rebuild CV² (4 traits)")
    ax.set(title="ET diversity", xlabel="generation")
    ax.legend(fontsize=8)
    fig.suptitle(f"Calibration: selected setting {best} vs. the paper (ET, Environment A)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "calibration_selected_vs_paper.png"), dpi=150)
    plt.close(fig)

    # report
    tdf = pd.DataFrame(rows)
    lines = ["# Calibration report (Part 1, ET training phase)", "",
             f"Selected setting: **{best}** (stage {sel['selected_from_stage']}), total score {sel['score']:.3f}. "
             f"Targets passed: {run['score']['n_pass']} of {run['score']['n_targets']}.", "",
             "Rules: `calibration/targets.json` (committed before any calibration run). "
             "Every setting tried: `calibration/log.csv`. Seeds 1000-1029.", "",
             "## Group scores (mean of squared error in tolerance units; 1.0 = on the edge of tolerance)", "",
             "| group | score |", "|---|---|"]
    for k, v in run["score"]["groups"].items():
        lines.append(f"| {k} | {v:.3f} |")
    lines += ["", "## Top 10 settings", "",
              "| id | stage | total | fitness | population | survival | traits | diversity | pass |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in log.nsmallest(10, "total_score").iterrows():
        lines.append(f"| {r['id']} | {r['stage']} | {r['total_score']:.3f} | {r['score_fitness']:.2f} | "
                     f"{r['score_population']:.2f} | {r['score_survival']:.3f} | {r['score_traits']:.2f} | "
                     f"{r['score_diversity']:.2f} | {r['n_pass']}/{r['n_targets']} |")
    lines += ["", f"All {len(log)} settings tried are in `calibration/log.csv`.", "",
              "## Every target for the selected setting", "",
              "| target | group | paper | rebuild | tolerance | error / tolerance | pass |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['group']} | {r['paper']:.4g} | {r['sim']:.4g} | {r['tolerance']:.3g} | "
                     f"{r['error_in_tol']:+.2f} | {'yes' if r['pass'] else '**no**'} |")
    lines += ["", "## Metric definitions chosen in stage 2", "",
              "| metric | chosen | rebuild | paper | error / tolerance | all candidates (mean) |", "|---|---|---|---|---|---|"]
    for name, d in sel["definitions"].items():
        c = d["candidates"][d["chosen"]]
        allc = "; ".join(f"{k}: {v['mean']:.3g}" for k, v in d["candidates"].items())
        lines.append(f"| {name} | {d['chosen']} | {c['mean']:.4g} | {d['paper']} | {c['error_in_tol']:+.2f} | {allc} |")
    lines += ["", "![selected vs paper](../figures/part1/calibration_selected_vs_paper.png)", ""]
    with open(os.path.join(CAL, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote calibration/REPORT.md and figures/part1/calibration_selected_vs_paper.png")
    print(tdf[~tdf["pass"]][["id", "paper", "sim", "tolerance", "error_in_tol"]].to_string(index=False))


if __name__ == "__main__":
    main()
