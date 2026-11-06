"""Part 2 figures (PREREG_part2.md Section 18).

Usage: python scripts/part2/make_figures_part2.py --run final
Reads results/part2/<run>/analysis/{confirmatory,exploratory}.json and series.npz;
writes figures/part2/<run>/*.png. Every number drawn comes from those files.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUPS = ("ET", "ET_freezeB", "ET_cloneB", "GS_food_frozen", "GS_pop_frozen", "GS_pop_mut", "SV_frozen", "ORACLE_B")
COLORS = dict(zip(GROUPS, ("#1f77b4", "#6baed6", "#9ecae1", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#7f7f7f")))


def save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    adir = os.path.join(ROOT, "results", "part2", args.run, "analysis")
    outdir = os.path.join(ROOT, "figures", "part2", args.run)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(adir, "exploratory.json")) as f:
        ex = json.load(f)
    conf = None
    if os.path.exists(os.path.join(adir, "confirmatory.json")):
        with open(os.path.join(adir, "confirmatory.json")) as f:
            conf = json.load(f)
    se = np.load(os.path.join(adir, "series.npz"))
    worlds = sorted({k.split("__")[0] for k in se.files})

    # 1. population over time (A then B), per arm
    for w in worlds:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for g in GROUPS:
            k = f"{w}__{g}__population_mean"
            if k not in se.files:
                continue
            y = se[k]
            x = np.arange(len(y)) if g != "ORACLE_B" else np.arange(len(y)) + (len(se[f"{w}__ET__population_mean"]) - len(y))
            ax.plot(x, y, color=COLORS[g], label=g, lw=1.4)
        n_a = len(se[f"{w}__ET__population_mean"]) - 50
        ax.axvline(n_a - 0.5, color="k", lw=0.8, ls=":")
        ax.set_xlabel("generation (A, then B)")
        ax.set_ylabel("mean population (dead runs count 0)")
        ax.set_title(f"Population over time ({w} B)")
        ax.legend(fontsize=7, ncol=2)
        save(fig, outdir, f"population_{w}.png")

    # 2. survival curves in B
    for arm, r in ex["secondary"].items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for g, curve in r["survival_curve"].items():
            ax.step(np.arange(len(curve)), curve, where="post", color=COLORS.get(g, "k"), label=g)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("B generation")
        ax.set_ylabel("share of runs with blobs")
        ax.set_title(f"Survival in B ({arm})")
        ax.legend(fontsize=7, ncol=2)
        save(fig, outdir, f"survival_{arm}.png")

    # 3. food per creature in B (S4), with the warning
    for arm, r in ex["secondary"].items():
        gs = [g for g in GROUPS if g in r["S4"] and r["S4"][g]["mean"] is not None]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(gs)), [r["S4"][g]["mean"] for g in gs], color=[COLORS[g] for g in gs])
        ax.set_xticks(range(len(gs)), gs, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("food per blob, B gens 40-49")
        ax.set_title(f"Food per blob in B ({arm})")
        fig.text(0.01, 0.01, "Warning: food per blob rises when a population shrinks, so it can't stand alone.",
                 fontsize=7)
        save(fig, outdir, f"food_per_blob_{arm}.png")

    # 4. ET traits over time, 5. diversity over time
    for w in worlds:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        for ax, t in zip(axes.flat, ("speed", "size", "sense", "reach")):
            for g in ("ET", "GS_pop_mut", "SV_frozen"):
                k = f"{w}__{g}__{t}_mean"
                if k in se.files:
                    ax.plot(se[k], color=COLORS[g], label=g, lw=1.2)
            ax.set_title(f"mean {t}")
            ax.set_xlabel("generation")
        axes.flat[0].legend(fontsize=7)
        save(fig, outdir, f"traits_{w}.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for g in ("ET", "ET_freezeB", "GS_pop_mut", "SV_frozen"):
            for ax, m, lab in ((axes[0], "cv2", "CV² (4 traits, sample)"), (axes[1], "shannon", "Shannon H (exact designs)")):
                k = f"{w}__{g}__{m}_mean"
                if k in se.files:
                    ax.plot(se[k], color=COLORS[g], label=g, lw=1.2)
                ax.set_title(lab)
                ax.set_xlabel("generation")
        axes[0].legend(fontsize=7)
        save(fig, outdir, f"diversity_{w}.png")

    # 6. compute cost
    for arm, rows in ex["compute_cost"].items():
        gs = [g for g in GROUPS if g in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(gs)), [rows[g]["mean"] for g in gs], color=[COLORS[g] for g in gs])
        ax.set_yscale("log")
        ax.set_xticks(range(len(gs)), gs, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("creature-steps per run, incl. search (guess K1)")
        ax.set_title(f"Compute cost ({arm})")
        save(fig, outdir, f"compute_{arm}.png")

    # 7. difficulty curve (exploratory): P1 by map side
    sides = []
    for v, d in ex["suite"].get("variants", {}).items():
        if v.startswith("side"):
            sides.append((int(v[4:]), d["groups"]))
    if sides and conf is not None:
        main_p1 = {g: conf["U1_U4"]["main"][g]["mean_P1"] for g in conf["U1_U4"]["main"]}
        pts = [(300, {g: {"P1": main_p1[g]} for g in main_p1})] + sides
        with open(os.path.join(ROOT, "results", "part2", args.run, "run.json")) as f:
            side3 = json.load(f)["provenance"]["config"]["part2"].get("arm3_side")
        if side3 is not None and "arm3" in conf.get("U1_U4", {}):
            pts.append((int(float(side3)), {g: {"P1": conf["U1_U4"]["arm3"][g]["mean_P1"]}
                                            for g in conf["U1_U4"]["arm3"]}))
        pts.sort(key=lambda p: p[0])
        fig, ax = plt.subplots(figsize=(7, 4))
        for g in GROUPS[:-1]:
            xs = [p for p, d in pts if g in d]
            ys = [d[g]["P1"] for p, d in pts if g in d]
            ax.plot(xs, ys, marker="o", color=COLORS[g], label=g)
        ax.set_xlabel("B map side (exploratory suite; 300 = main B)")
        ax.set_ylabel("mean P1")
        ax.set_title("Difficulty curve (exploratory)")
        ax.legend(fontsize=7, ncol=2)
        save(fig, outdir, "difficulty_curve.png")

    # 8. landscape check
    if ex.get("landscape"):
        ws = sorted(ex["landscape"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(ws)), [ex["landscape"][w]["mean_tau_b_all"] for w in ws], color="#4c72b0")
        ax.set_xticks(range(len(ws)), ws, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Kendall tau-b, A score vs B score")
        ax.set_title("How much each B rewards the same designs as A (exploratory)")
        save(fig, outdir, "landscape_tau.png")
    print("figures in", os.path.relpath(outdir, ROOT))


if __name__ == "__main__":
    main()
