"""Addendum A analysis (PREREG_part3.md Section 16): ES-fixed against ES-DR, and the variety effect
for ES against OE. Every number from saved results; the core analysis is not changed.

Reads
  <final>/analysis.json and the core run folders es_dr, oe, oe_fixed (seeds 5000-5009)
  <addendum>/es_fixed_seed*/ (the addendum's runs), <addendum>/es_dr_seed5000, 5001 (check reruns),
  <addendum>/tuning/tuning_es_fixed.json, <addendum>/tuned_addendum.yaml
Writes
  <addendum>/analysis.json, figures/part3/addendum_variety.png, docs/RESULTS_part3_addendum.md

  venv3/bin/python scripts/part3/analyze_addendum.py --final results/part3/final --addendum results/part3/addendum
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from evo2 import analysis as A
from evo2 import metrics
from evo2.runio import ROOT, write_json

_spec = importlib.util.spec_from_file_location("analyze", Path(__file__).resolve().parent / "analyze.py")
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

SEEDS = list(range(5000, 5010))
CHECK_SEEDS = [5000, 5001]          # ES-DR reruns on the addendum's machine and commit (PREREG 16.3)
RANDOM_STOP = 1e-3                   # PREREG 16.4: random competence must match the core's this closely
NAMES = {"oe": "OE", "oe_fixed": "OE-fixed", "es_dr": "ES-DR", "es_fixed": "ES-fixed"}


def finished(d: Path) -> bool:
    """Trained, scored on the validation curve and evaluated on the test split."""
    return all((d / f).is_file() for f in ("done.json", "curves.json", "eval_test.json")) and not (d / "failed.json").exists()


def load_addendum_runs(add: Path) -> tuple[list[Path], list[dict]]:
    """ES-fixed seeds 5000-5009. A seed that failed (failed.json), crashed, or did not finish training,
    curves and evaluation is left out and reported (PREREG 16.3)."""
    got, excluded = [], []
    for s in SEEDS:
        d = add / f"es_fixed_seed{s}"
        if (d / "failed.json").exists():
            excluded.append({"seed": s, "reason": "failed: " + str(json.loads((d / "failed.json").read_text()).get("reason"))})
        elif finished(d):
            got.append(d)
        else:
            excluded.append({"seed": s, "reason": "crashed or did not finish" if d.exists() else "never ran"})
    return got, excluded


def check_tuned(add: Path, runs: list[dict]) -> dict:
    """Every ES-fixed run used the settings its own tuning chose (PREREG 16.2)."""
    summary = json.loads((add / "tuning" / "tuning_es_fixed.json").read_text())
    tuned = yaml.safe_load((add / "tuned_addendum.yaml").read_text())
    if summary.get("quick") or len(summary["trials"]) != 8:
        raise SystemExit("ES-fixed tuning is not the full 8-trial tuning")
    if tuned["settings"]["es_fixed"] != summary["chosen"]["settings"]:
        raise SystemExit("tuned_addendum.yaml does not hold the tuning's choice")
    for r in runs:
        mc = r["methods_config"]
        if mc["es_fixed"]["settings"] != summary["chosen"]["settings"] or "tuned_from_addendum" not in mc:
            raise SystemExit(f"es_fixed seed {r['seed']} did not train with the tuned settings")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="results/part3/final")
    ap.add_argument("--addendum", default="results/part3/addendum")
    ap.add_argument("--reps", type=int, default=A.REPS)
    ap.add_argument("--out-doc", default=str(ROOT / "docs/RESULTS_part3_addendum.md"))
    ap.add_argument("--out-fig", default=str(ROOT / "figures/part3/addendum_variety.png"))
    a = ap.parse_args()
    res = run(Path(a.final), Path(a.addendum), a.reps, Path(a.out_doc), Path(a.out_fig))
    print(json.dumps({k: res[k] for k in ("verdicts_given", "a1", "readings") if k in res}, indent=1, default=str))


def run(final: Path, add: Path, reps: int, out_doc: Path, out_fig: Path) -> dict:
    corej = json.loads((final / "analysis.json").read_text())
    best, random = np.asarray(corej["best"], float), np.asarray(corej["random"], float)
    valid = A.valid_envs(best, random)

    # core runs, read and checked by the core analysis's own reader
    data = {m: [core.read_eval(final / f"{m}_seed{s}") for s in corej["runs"][m]] for m in ("es_dr", "oe", "oe_fixed")}
    es_dirs, excluded = load_addendum_runs(add)
    if not es_dirs:
        raise SystemExit("no ES-fixed run finished")
    data["es_fixed"] = [core.read_eval(d) for d in es_dirs]
    for r in data["es_fixed"]:
        if r["method"] != "es_fixed":
            raise SystemExit(f"seed {r['seed']}: not an es_fixed run")
    summary = check_tuned(add, data["es_fixed"])
    checks = [core.read_eval(add / f"es_dr_seed{s}") for s in CHECK_SEEDS if finished(add / f"es_dr_seed{s}")]
    missing_checks = [s for s in CHECK_SEEDS if not finished(add / f"es_dr_seed{s}")]
    one = {(r["train_commit"], r["commit"]) for r in data["es_fixed"]}
    if len(one) != 1 or len({c for pair in one for c in pair}) != 1:
        raise SystemExit(f"the ES-fixed runs were not all trained and evaluated from one commit: {sorted(one)}")

    # the same evaluation: random competence as in the core analysis (PREREG 16.4)
    rand_diff = max(float(np.max(np.abs(r["random"] - random))) for r in data["es_fixed"] + checks)
    if rand_diff > RANDOM_STOP:
        raise SystemExit(f"random competence differs from the core analysis by {rand_diff}: stop and record a protocol amendment")
    commits = sorted({r["commit"] for r in data["es_fixed"] + checks} | {r["train_commit"] for r in data["es_fixed"] + checks})

    curves = {m: np.stack([r["curve"] for r in rs]) for m, rs in data.items()}
    for m, rs in data.items():
        if not np.array_equal(curves[m][:, :, 0], np.stack([r["zero_shot"] for r in rs])):
            raise SystemExit(f"{m}: adaptation point 0 is not the zero-shot competence")
    norm = {m: A.normalized_curves(c, random, best, valid) for m, c in curves.items()}
    primary = {m: A.auc(n) for m, n in norm.items()}
    measures = {"primary": primary, "zero_shot": {m: n[:, :, 0] for m, n in norm.items()}}
    t23 = A.env_index("T2", "T3")
    prim23 = {m: A.select(primary[m], t23) for m in data}

    verdicts = len(excluded) <= A.ADDENDUM_LEFT_OUT_MAX
    res = {"core_analysis": str(final / "analysis.json"), "commits": commits, "bootstrap_reps": reps,
           "verdicts_given": verdicts,
           "runs": {m: [r["seed"] for r in data[m]] for m in data}, "excluded_es_fixed": excluded,
           "random_max_difference_from_core": rand_diff,
           "tuning": {"chosen": summary["chosen"],
                      "trials": [{"trial": t["trial"], "seed": t["seed"], "settings": t["settings"], "score": t["score"]}
                                 for t in summary["trials"]]}}
    if verdicts:
        res["a1"] = A.a1(prim23, reps)
        res["family"] = A.compare_family(measures, A.ADDENDUM_FAMILY, reps)
        res["a4"] = A.a4(prim23, reps)
        res["readings"] = A.addendum_readings(res["a1"]["verdict"], {r["name"]: r for r in res["family"]}["A3"]["result"])
    res["iqm_primary"] = metrics.iqm_with_ci(prim23, reps, A.STATS_SEED)

    # descriptive
    zs = {m: np.stack([r["zero_shot"] for r in data[m]]) for m in data}
    res["descriptive"] = {m: {"primary_mean_by_set": {s: float(np.nanmean(primary[m][:, A.env_index(s)])) for s in A.SETS},
                              "zero_shot_raw_mean_by_set": {s: float(zs[m][:, A.env_index(s)].mean()) for s in A.SETS},
                              "zero_shot_norm_mean_by_set": {s: float(np.nanmean(measures["zero_shot"][m][:, A.env_index(s)]))
                                                             for s in A.SETS},
                              "train_wall_hours_mean": float(np.mean([r["done"]["wall_seconds"] for r in data[m]]) / 3600)}
                          for m in data}
    orig = {r["seed"]: r for r in data["es_dr"]}
    rows = []
    for c in checks:
        cn = A.auc(A.normalized_curves(c["curve"][None], random, best, valid))
        on = A.auc(A.normalized_curves(orig[c["seed"]]["curve"][None], random, best, valid))
        rows.append({"seed": c["seed"], "rerun": float(np.nanmean(A.select(cn, t23))),
                     "original": float(np.nanmean(A.select(on, t23)))})
    orig_means = A.per_run_mean(prim23["es_dr"])
    lo, hi = float(orig_means.min()), float(orig_means.max())
    res["es_dr_check"] = {"rows": rows, "missing": missing_checks, "original_range": [lo, hi],
                          "tolerance": A.RERUN_TOLERANCE, "flag": A.rerun_flag([r["rerun"] for r in rows], lo, hi),
                          "hardware": checks[0]["hardware"]["gpus"] if checks else []}
    res["hardware"] = sorted({g for r in data["es_fixed"] for g in r["hardware"]["gpus"]})
    log = add / "test_access_log.jsonl"
    res["test_access_log"] = [json.loads(x) for x in log.read_text().splitlines() if x.strip()] if log.exists() else None
    spend = add / "spend.json"
    res["spend"] = json.loads(spend.read_text()) if spend.exists() else None
    write_json(add / "analysis.json", res)
    figure(prim23, out_fig)
    report(res, out_doc, out_fig)
    return res


def figure(prim23: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    order = ("es_fixed", "es_dr", "oe_fixed", "oe")
    for i, m in enumerate(order):
        v = A.per_run_mean(prim23[m])
        ax.scatter(np.full(v.shape, i) + np.linspace(-0.12, 0.12, v.size), v, s=14)
        ax.hlines(metrics.iqm(prim23[m]), i - 0.25, i + 0.25, colors="k")
    ax.set_xticks(range(len(order)), ["ES-fixed\n(one world)", "ES-DR\n(varied worlds)", "OE-fixed\n(one world)",
                                      "OE\n(varied worlds)"])
    ax.set_ylabel("primary metric, T2 and T3\n(dots: runs; line: IQM)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def f3(x):
    return f"{x:.3f}"


def report(res: dict, path: Path, fig: Path):
    chk = res["es_dr_check"]
    flag = (" **Flag (PREREG 16.3): an ES-DR check rerun lies more than 0.01 outside the core ES-DR range, so A1 and "
            "A2 may be shifted by machine or code.**") if chk["flag"] else ""
    L = ["# Part 3, Addendum A: ES-fixed (pre-registered after the core results)", "",
         "Generated by `scripts/part3/analyze_addendum.py` from saved results. Do not edit by hand. "
         "Pre-registration: `docs/PREREG_part3.md` Section 16 (written and committed before any ES-fixed run). "
         "ES-fixed is normalized with the core analysis's saved random-policy and best-score anchors.", "",
         f"- Commits of the addendum's runs and evaluations: {', '.join(c[:7] for c in res['commits'])}; "
         f"hardware: {', '.join(res['hardware'])}.",
         f"- ES-fixed seeds analysed: {res['runs']['es_fixed']}; left out: "
         + ("; ".join(f"{x['seed']} ({x['reason']})" for x in res["excluded_es_fixed"]) or "none")
         + ". ES-DR, OE and OE-fixed: the core final runs (10 each).",
         f"- Random policy's competence: largest difference from the core analysis {res['random_max_difference_from_core']:.2e}.",
         f"- Bootstrap: {res['bootstrap_reps']:,} resamples of whole runs, seed 4800.", ""]
    if not res["verdicts_given"]:
        L += ["**No verdicts:** more than 2 of the 10 ES-fixed seeds were left out (PREREG 16.3). The runs that "
              "finished are described below.", ""]
    else:
        a1r, a4r, rd = res["a1"], res["a4"], res["readings"]
        fam = {r["name"]: r for r in res["family"]}
        L += ["## Results", "",
              'A3 compares T1 zero-shot competence; A1 compares T2/T3 adaptation AUC. The ES-DR and ES-fixed configurations were tuned separately, so these comparisons do not isolate environmental variation.', "",
              "## A1: ES-DR vs ES-fixed, adaptation on T2 and T3", "",
              f"**Verdict: {a1r['verdict']}.** D = IQM(ES-DR) − IQM(ES-fixed) = {f3(a1r['D'])}, 95% CI "
              f"[{f3(a1r['ci'][0])}, {f3(a1r['ci'][1])}] (margin {a1r['margin']}). P(ES-DR > ES-fixed) on per-run "
              f"means: {f3(a1r['p_es_dr_greater'])}. Historical planned prediction motivated by Milano et al. (2019): ES-DR > ES-fixed.", ""]
    L += ["| Method | IQM of the primary metric (T2, T3) | 95% CI |", "|---|---|---|"]
    for m in ("es_fixed", "es_dr", "oe_fixed", "oe"):
        v, lo, hi = res["iqm_primary"][m]
        L.append(f"| {NAMES[m]} | {f3(v)} | [{f3(lo)}, {f3(hi)}] |")
    if res["verdicts_given"]:
        L += ["", "## A2 and A3 (one Holm family, α = 0.05)", "",
              "| | X vs Y | Measure | Sets | P(X > Y) | p | p (Holm) | IQM difference [CI] | Predicted | Result |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["family"]:
            L.append(f"| {r['name']} | {NAMES[r['x']]} vs {NAMES[r['y']]} | {r['measure']} | {', '.join(r['sets'])} | "
                     f"{f3(r['p_x_greater'])} | {r['p']:.4g} | {r['p_holm']:.4g} | {f3(r['iqm_difference'])} "
                     f"[{f3(r['ci'][0])}, {f3(r['ci'][1])}] | X {r['predicted']} Y | {r['result']} |")
        a4r = res["a4"]
        L += ["", "A3 measures zero-shot competence on the fixed T1 suite, without further adaptation. "
              "Task, algorithm, environment schedule, tuning and normalization differ from Milano et al.'s study.", "",
              "## A4 (descriptive): the variety effect for ES and for OE (primary metric, T2 and T3)", "",
              f"Variety effect for ES (ES-DR − ES-fixed): {f3(a4r['variety_es']['value'])} "
              f"[{f3(a4r['variety_es']['ci'][0])}, {f3(a4r['variety_es']['ci'][1])}]; for OE (OE − OE-fixed): "
              f"{f3(a4r['variety_oe']['value'])} [{f3(a4r['variety_oe']['ci'][0])}, {f3(a4r['variety_oe']['ci'][1])}]; "
              f"difference (ES minus OE): {f3(a4r['interaction']['value'])} [{f3(a4r['interaction']['ci'][0])}, "
              f"{f3(a4r['interaction']['ci'][1])}] ({a4r['verdict']}). Not a verdict: the OE effect was known "
              "before this run (PREREG 16.4)."]
    L += ["", f"![per-run primary metric]({Path('../figures/part3') / fig.name})", "",
          "## Descriptive (not tested)", "",
          "| Method | primary T1 / T2 / T3 | zero-shot, raw competence T1 / T2 / T3 | zero-shot, normalized T1 / T2 / T3 | training hours per run |",
          "|---|---|---|---|---|"]
    for m in ("es_fixed", "es_dr", "oe_fixed", "oe"):
        d = res["descriptive"][m]
        prim = " / ".join(f3(d["primary_mean_by_set"][s]) for s in A.SETS)
        raw = " / ".join(f"{d['zero_shot_raw_mean_by_set'][s]:.2f}" for s in A.SETS)
        nrm = " / ".join(f3(d["zero_shot_norm_mean_by_set"][s]) for s in A.SETS)
        L.append(f"| {NAMES[m]} | {prim} | {raw} | {nrm} | {d['train_wall_hours_mean']:.3f} |")
    rows = "; ".join(f"seed {r['seed']}: {f3(r['rerun'])} (core run {f3(r['original'])})" for r in chk["rows"]) or "none finished"
    L += ["", f"**ES-DR check reruns** (same seeds, the addendum's commit and machine): {rows}"
          + (f"; missing: {chk['missing']}" if chk["missing"] else "")
          + f". Range of the 10 core ES-DR runs: [{f3(chk['original_range'][0])}, {f3(chk['original_range'][1])}]; "
          f"flag if a rerun is more than {chk['tolerance']} outside it: {'yes' if chk['flag'] else 'no'}. "
          "Runs on different machines are not bit-identical, so this checks the level, not identity.", "",
          "## ES-fixed tuning (validation environments only)", "",
          "| Trial | seed | settings | score |", "|---|---|---|---|"]
    for t in res["tuning"]["trials"]:
        st = ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in sorted(t["settings"].items()))
        L.append(f"| {t['trial']} | {t['seed']} | {st} | {t['score']:.3f}{' (chosen)' if t['trial'] == res['tuning']['chosen']['trial'] else ''} |")
    log = res["test_access_log"]
    L += ["", "## Record", "",
          f"- Test-access log of this stage: {len(log)} entr{'y' if len(log) == 1 else 'ies'}." if log is not None
          else "- Test-access log of this stage: not found.",
          f"- Spend of this stage: ${res['spend']['total']:.2f} (`spend.json`)." if res["spend"] else "- Spend: not recorded yet."]
    path.write_text("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
