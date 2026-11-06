"""Part 1: rebuild the paper's Figures 1-11 from the final run, with the paper's values marked.

Solid line + band = rebuild (mean +/- 1 SD across runs). Dashed line = the paper, read from
its figure by pixel (ET training curves: calibration/digitized_paper_figures.json, used as
calibration targets; everything else: results/part1/paper_heldout_digitized.json, held out).
Every number drawn comes from saved files; nothing is typed by hand.

Usage: python scripts/part1/make_figures.py --name final
Writes figures/part1/fig01_*.png ... fig11_*.png
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evosim import metrics  # noqa: E402
from compare_paper import fmt, load  # noqa: E402

ET_C, GS_C = "#0173B2", "#DE8F05"  # the paper's own colors (validated: CVD dE 26, normal dE 34)
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
TRAIN, TRANSFER = 100, 50
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
                     "savefig.dpi": 160, "figure.dpi": 100})


def band(ax, x, st, color, label, n_label, floor=None):
    """Mean line and +/- 1 SD band. floor: lowest possible value (e.g. 0 blobs); the band is cut there."""
    with np.errstate(invalid="ignore"):
        m = np.nanmean(st, axis=0)
        sd = np.nanstd(st, axis=0, ddof=1) if st.shape[0] > 1 else np.zeros_like(m)
    ok = np.sum(~np.isnan(st), axis=0) >= 2
    lo = m - sd if floor is None else np.maximum(m - sd, floor)
    ax.fill_between(x[ok], lo[ok], (m + sd)[ok], color=color, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=color, lw=2, label=f"{label} rebuild ({n_label})")
    return m


def paper_line(ax, d, color, label, skip=()):
    """The paper's line as read from its figure. Generations left out (skip) leave a gap."""
    g = sorted(int(k) for k in d if int(k) not in skip)
    xs, ys = [], []
    for i, k in enumerate(g):
        if i and k != g[i - 1] + 1:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(k)
        ys.append(d[str(k)])
    ax.plot(xs, ys, color=color, lw=1.6, ls=(0, (5, 3)), label=f"{label} paper (read from figure)")


def finish(fig, path, note=None):
    if note:  # below everything else; bbox_inches="tight" grows the canvas to fit it
        fig.text(0.01, -0.01, note, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="final")
    args = ap.parse_args()
    folder = os.path.join(ROOT, "results", "part1", args.name)
    outdir = os.path.join(ROOT, "figures", "part1")
    os.makedirs(outdir, exist_ok=True)
    comp = json.load(open(os.path.join(folder, "comparison.json")))
    held = json.load(open(os.path.join(ROOT, "results", "part1", "paper_heldout_digitized.json")))["figures"]
    cal = json.load(open(os.path.join(ROOT, "calibration", "digitized_paper_figures.json")))["figures"]
    paper = yaml.safe_load(open(os.path.join(ROOT, "docs", "paper_numbers_part1.yaml")))
    runs = {c: load(folder, c) for c in ("ET", "GS")}
    ser = {c: [metrics.series(a) for a in runs[c]] for c in runs}
    n = {c: len(runs[c]) for c in runs}

    def stack(c, key, zero=False):
        st = np.stack([s[key] for s in ser[c]])
        return np.nan_to_num(st, nan=0.0) if zero else st

    def bad(fig, curve):
        d = cal[fig]
        return set(d.get("interpolated_gens", {}).get(curve, [])) | set(d.get("extrapolated_gens", {}).get(curve, []))

    gs_alive = np.sum(stack("GS", "population") > 0, axis=0)
    ext = comp["extinction"]["GS"]
    gs_note = (f"GS rebuild: all {ext['extinct_runs']} of {n['GS']} runs died out during training "
               f"(first generation with no blobs: earliest {ext['first_zero_min']}, median "
               f"{fmt(ext['first_zero_median'], 1)}, latest {ext['first_zero_max']}). The GS line averages only runs "
               f"still alive, so late points rest on few runs (lower panel).")
    xt = np.arange(TRAIN)
    xb = np.arange(TRANSFER)

    # ---- Fig 1: fitness over training (+ GS runs alive)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    band(a1, xt, stack("ET", "fitness")[:, :TRAIN], ET_C, "ET", f"n={n['ET']}")
    band(a1, xt, stack("GS", "fitness")[:, :TRAIN], GS_C, "GS", "runs still alive")
    paper_line(a1, cal["fig1_et_fitness"]["curves"]["et"], ET_C, "ET", skip=bad("fig1_et_fitness", "et"))
    paper_line(a1, held["fig1_gs_fitness_train"]["curves"]["gs"], GS_C, "GS")
    a1.set_ylabel("Mean food eaten per blob")
    a1.set_title("Fig. 1 rebuilt: mean fitness over training", loc="left", color=INK)
    a1.legend(fontsize=8, ncol=2, loc="lower right")
    a2.step(xt, gs_alive[:TRAIN], where="mid", color=GS_C, lw=1.6)
    a2.set_ylabel("GS runs alive")
    a2.set_ylim(0, n["GS"] + 1)
    a2.set_xlabel("Generation (Environment A)")
    finish(fig, os.path.join(outdir, "fig01_fitness_training.png"),
           "ET paper curve = calibration target. GS paper curve = held out. Gaps in a dashed line = points the "
           "reader filled in or extended, left out. " + gs_note)

    # ---- Fig 2: population over training
    fig, ax = plt.subplots(figsize=(8, 4.2))
    band(ax, xt, stack("ET", "population", True)[:, :TRAIN], ET_C, "ET", f"n={n['ET']}", floor=0)
    band(ax, xt, stack("GS", "population", True)[:, :TRAIN], GS_C, "GS", f"n={n['GS']}, extinct runs count as 0",
         floor=0)
    paper_line(ax, cal["fig2_et_population"]["curves"]["et"], ET_C, "ET", skip=bad("fig2_et_population", "et"))
    paper_line(ax, held["fig2_gs_population_train"]["curves"]["gs"], GS_C, "GS", skip={0})
    ax.set_ylabel("Population size")
    ax.set_xlabel("Generation (Environment A)")
    ax.set_title("Fig. 2 rebuilt: population over training", loc="left", color=INK)
    ax.legend(fontsize=8, loc="center right")
    finish(fig, os.path.join(outdir, "fig02_population_training.png"),
           "Bands are mean +/- 1 SD, cut at 0 blobs. "
           f"Paper GS generation 0 not drawn: it reads {fmt(held['fig2_gs_population_train']['curves']['gs']['0'], 1)} "
           "on the steep edge, but is 50 by design (50 clones). "
           "Gaps in a dashed line = points the reader filled in or extended, left out.")

    # ---- Fig 3: ET traits over training and transfer (small multiples)
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    x150 = np.arange(TRAIN + TRANSFER)
    for ax, tr in zip(axs.ravel(), ("speed", "size", "sense", "reach")):
        band(ax, x150, stack("ET", f"mean_{tr}"), ET_C, "ET", f"n={n['ET']}")
        paper_line(ax, cal["fig3_et_traits"]["curves"][tr], INK, "ET", skip=bad("fig3_et_traits", tr))
        ax.axvline(TRAIN, color=MUTED, lw=1)
        ax.set_title(f"mean {tr}", loc="left", fontsize=10, color=INK)
    axs[0, 0].legend(fontsize=7, loc="lower right")
    for ax in axs[1]:
        ax.set_xlabel("Generation (0-99 A, 100-149 B)")
    fig.suptitle("Fig. 3 rebuilt: ET mean traits (paper's lines dashed)", x=0.01, ha="left", color=INK)
    finish(fig, os.path.join(outdir, "fig03_traits.png"),
           "Calibration targets (calibration/targets.json): "
           + "; ".join(f"{k} at generations {', '.join(str(gg) for gg, _ in vv)}"
                       for k, vv in comp["extra"]["trait_target_gens"].items())
           + " (generation-99 speed, size and sense from the paper's text). Generations 100-149 were never targets. "
           "Gaps in a dashed line = points the reader filled in or extended, left out.")

    # ---- Fig 4 and 5: diversity over training
    for num, key, cal_fig, title, ylab, fname in (
            (4, "cv2_4_samp", "fig4_et_cv2", "trait diversity (CV2 sum, 4 traits)", "CV2 sum", "fig04_cv2_training.png"),
            (5, "H_exact4", "fig5_et_shannon", "Shannon diversity (H)", "Shannon H", "fig05_shannon_training.png")):
        fig, ax = plt.subplots(figsize=(8, 4))
        band(ax, xt, stack("ET", key)[:, :TRAIN], ET_C, "ET", f"n={n['ET']}", floor=0)
        band(ax, xt, stack("GS", key)[:, :TRAIN], GS_C, "GS", "runs still alive", floor=0)
        paper_line(ax, cal[cal_fig]["curves"]["et"], ET_C, "ET", skip=bad(cal_fig, "et"))
        ax.set_ylabel(ylab)
        ax.set_xlabel("Generation (Environment A)")
        ax.set_title(f"Fig. {num} rebuilt: {title} over training", loc="left", color=INK)
        ax.legend(fontsize=8, loc="upper left" if num == 4 else "center right")
        finish(fig, os.path.join(outdir, fname), "ET paper curve = calibration target. GS is 0 by design (identical clones). "
               "Bands are mean +/- 1 SD, cut at 0. Gaps in a dashed line = points the reader filled in or extended, "
               "left out.")

    # ---- Fig 6, 7, 9: transfer curves
    no_gs = f"No GS run reached transfer: all {n['GS']} died out during training."
    for num, key, hfig, title, ylab, zero, fname in (
            (6, "fitness", "fig6_fitness_transfer", "fitness during transfer", "Mean food eaten per blob", False,
             "fig06_fitness_transfer.png"),
            (7, "population", "fig7_population_transfer", "population during transfer", "Population size", True,
             "fig07_population_transfer.png"),
            (9, "cv2_4_samp", "fig9_cv2_transfer", "trait diversity (CV2 sum) during transfer", "CV2 sum", False,
             "fig09_cv2_transfer.png")):
        fig, ax = plt.subplots(figsize=(8, 4))
        band(ax, xb, stack("ET", key, zero)[:, TRAIN:], ET_C, "ET", f"n={n['ET']}", floor=0)
        paper_line(ax, held[hfig]["curves"]["et"], ET_C, "ET")
        paper_line(ax, held[hfig]["curves"]["gs"], GS_C, "GS")
        if num == 9:  # bottom right would sit on the paper's GS line; top left is empty
            ax.text(0.01, 0.97, no_gs, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK)
        else:
            ax.text(0.99, 0.02 if num != 7 else 0.08, no_gs, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color=INK)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Transfer generation (Environment B)")
        ax.set_title(f"Fig. {num} rebuilt: {title}", loc="left", color=INK)
        ax.legend(fontsize=8, loc="upper right")
        finish(fig, os.path.join(outdir, fname), "Both paper curves held out. Paper GS line stops where the paper's does. "
               "Bands are mean +/- 1 SD, cut at 0.")

    # ---- Fig 8: generalization gap
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = {r["key"]: r for r in paper["fig11"]}
    gap_rows = {(f["condition"], f["window"]): f for f in comp["fig8"]}
    labels, xs, k = [], [], 0
    for cond, col in (("ET", ET_C), ("GS", GS_C)):
        p_tab = rows["gap"][cond.lower()]
        b = held["fig8_gap_bars"][cond.lower()]
        bars = [("paper table", p_tab[0], p_tab[1], True), ("paper chart", b["bar_top"], b.get("error_bar_half_length"), True)]
        for w, wl in (("gap_last10_first10", "rebuild\ntransfer 0-9"), ("gap_last10_first5", "rebuild\ntransfer 0-4")):
            f = gap_rows[(cond, w)]
            bars.append((wl, f["rebuilt"], f["rebuilt_sd"], False))
        for lab, v, e, is_paper in bars:
            if v is None:
                ax.text(k, 0.005, f"no value\n(n=0)", ha="center", va="bottom", fontsize=8, color=INK)
            else:
                ax.bar(k, v, width=0.7, color="white" if is_paper else col, edgecolor=col, hatch="///" if is_paper else None,
                       linewidth=1.5)
                if e is not None:
                    ax.errorbar(k, v, yerr=e, color=INK, capsize=4, lw=1)
                ax.text(k + 0.38, v, fmt(v), ha="left", va="bottom", fontsize=8, color=INK)
            labels.append(f"{cond}\n{lab}")
            xs.append(k)
            k += 1
        k += 0.6
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Fitness drop (late training minus early transfer)")
    ax.set_title("Fig. 8 rebuilt: generalization gap", loc="left", color=INK)
    finish(fig, os.path.join(outdir, "fig08_gap.png"),
           "Hatched = paper (table value +/- SD; chart bar +/- its error bar, read from the figure). "
           "Solid = rebuild, mean +/- SD; late training = last 10 training generations. All held out.")

    # ---- Fig 10: compute
    fig, ax = plt.subplots(figsize=(7, 4))
    tx = {t["id"]: t for t in comp["text"]}
    srch = comp["grid"]["search_creature_steps"]
    ax.bar(0, tx["compute_et"]["value"], 0.7, color="white", edgecolor=ET_C, hatch="///", lw=1.5)
    ax.bar(1, tx["compute_et"]["rebuilt"], 0.7, color=ET_C)
    ax.bar(2.6, tx["compute_gs"]["value"], 0.7, color="white", edgecolor=GS_C, hatch="///", lw=1.5)
    ax.bar(3.6, srch, 0.7, color=GS_C, alpha=0.55, label="rebuild GS: search")
    ax.bar(3.6, tx["compute_gs"]["rebuilt"] - srch, 0.7, bottom=srch, color=GS_C, label="rebuild GS: deployed runs")
    for x, v in ((0, tx["compute_et"]["value"]), (1, tx["compute_et"]["rebuilt"]), (2.6, tx["compute_gs"]["value"]),
                 (3.6, tx["compute_gs"]["rebuilt"])):
        ax.text(x, v, f"{v / 1e6:.1f}M", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(3.6 + 0.4, srch, f"search {srch / 1e6:.1f}M\n+ runs {(tx['compute_gs']['rebuilt'] - srch) / 1e6:.2f}M",
            ha="left", va="center", fontsize=8, color=INK)
    ax.set_xticks([0, 1, 2.6, 3.6])
    ax.set_xticklabels(["ET paper", "ET rebuild", "GS paper", "GS rebuild"])
    ax.set_ylabel("Simulation steps, all 30 runs (millions)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Fig. 10 rebuilt: compute", loc="left", color=INK)
    finish(fig, os.path.join(outdir, "fig10_compute.png"),
           "The rebuild counts creature-steps (one blob moving one tick), which is guess K1 for what the paper "
           "means by a step. ET compute is held out. GS compute is NOT an independent check: the paper's GS number "
           "was used to set the search length at check-in 1.")

    # ---- Fig 11: the statistics table, paper vs rebuild
    head = ["Metric", "Status ET / GS", "ET paper", "ET rebuild (n)", "GS paper", "GS rebuild (n)",
            "t paper / rebuild", "d paper / rebuild", "Within? ET / GS"]
    short = {"target": "target", "defn": "picked defn", "held_out": "held out", "setting": "set a guess"}

    def verdict(g):
        w = {"replicates": "yes", "misses": "no", "no value": "no value"}[g["verdict"]]
        if g.get("posthoc_verdict") and g["posthoc_verdict"] != g["verdict"]:
            w += "*"
        return w
    def cell(g):
        if g["n"] == 0:
            return "no value (0)"
        return f"{fmt(g['mean'])} ± {fmt(g['sd'])} ({g['n']})"

    cells = []
    for r in comp["fig11"]:
        p = r["paper"]
        st = r["stats"]
        cells.append([r["row"], f"{short[r['et']['status']]} / {short[r['gs']['status']]}",
                      f"{fmt(p['et'][0])} ± {fmt(p['et'][1])}",
                      cell(r["et"]),
                      f"{fmt(p['gs'][0])} ± {fmt(p['gs'][1])}",
                      cell(r["gs"]),
                      f"{fmt(p['t'])} / {fmt(st['t'])}", f"{fmt(p['d'])} / {fmt(st['d'])}",
                      f"{verdict(r['et'])} / {verdict(r['gs'])}"])
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=head, loc="center", cellLoc="center", colLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.auto_set_column_width(col=list(range(len(head))))  # each column as wide as its longest text
    tb.scale(1, 1.45)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_text_props(color=INK, weight="bold")
    ax.set_title("Fig. 11 rebuilt: statistics table, paper vs rebuild (mean ± SD across runs; n = runs with a value)",
                 loc="left", color=INK, fontsize=10)
    finish(fig, os.path.join(outdir, "fig11_table.png"),
           "Within? = the rule fixed before the run, as written: within the paper's SD, or 10% where no SD is printed; "
           "SD 0.00 means an exact match (calibration-target rows use calibration's own 10% rule instead). "
           "* = a reading chosen after seeing the data (within the printed precision, 0.005) gives the other answer. "
           "'no value' = no run had a value (every GS run died out in training). GS 'yes' on CV2 and H is zero by "
           f"design. GS final fitness rests on {comp['fig11'][0]['gs']['n']} runs. Details in docs/REPLICATION.md.")


if __name__ == "__main__":
    main()
