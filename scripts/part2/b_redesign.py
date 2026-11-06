"""World B redesign (1 Nov 2026, after the pilot): a B that calls for a different design.

The pilot showed that in every B tried, the best design found in A was about as
good as the best design found in B (check U2 fired). The A winner is a small,
fast, cheap design (speed 40, size 3.7), and every B tried also rewarded small,
fast, cheap foragers. Less energy per generation in B makes fast movement too
costly, which should favor a different design.

This script only runs clones (no group is compared), on seeds set aside for it.

Step 1, probe (--probe): seeds 2870-2871. A stage-1 landscape and the B stage-1
  landscape for each energy level; reports where the top A designs rank in B.
Step 2, rule (--rule): for each candidate energy, the full A search, the full
  B search and the U2 runs (Section 3, Section 8), on the main B's streams.
  Chosen: the highest energy (smallest change from the main B) where the B
  winner beats the A winner by at least 5 blobs on the U2 sets, and the B
  winner's mean U2 score is at least 20. Exact arithmetic. If none qualifies, stop without selecting a setting. Writes configs/part2/arm4.yaml.
  - Round 1 (--round 1): seeds 2860-2869, levels 100-40, MEAN gap over seeds
    (Section 9.3). No level qualified.
  - Round 2 (--round 2, added on 2 Nov 2026): seeds 2872-2881, levels
    30, 25, 20, 15, MEDIAN gap over seeds (Section 17). Chose 25.
  Each round's levels are fixed here (ROUND_LEVELS) so a rerun can't use others.

Usage: python scripts/part2/b_redesign.py --probe | --rule --round 1|2 [--quick]
"""
import argparse
import json
import os
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evosim import config, run2, search2, streams  # noqa: E402

ARM3 = os.path.join(ROOT, "configs", "part2", "arm3.yaml")
PROBE_SEEDS = (2870, 2871)
RULE_SEEDS = tuple(range(2860, 2870))
PROBE_ENERGIES = (500, 400, 300, 250, 200, 150, 100)  # probe 1; probes 2-4 (--probe-energies) go lower
ROUND_LEVELS = {1: (100, 90, 80, 70, 60, 50, 40), 2: (30, 25, 20, 15)}  # as recorded in Section 17


def probe_one(args):
    seed, cfg, energies = args
    designs = search2.merged_designs(cfg)
    n = len(designs)
    with streams.context("search_A"):
        a1 = search2.stage1(designs, cfg["env_A"], cfg["world"], seed, "search_A1", (), cfg)
        tie = streams.rng(seed, "tiebreak_A").random(n)
    o = search2.order(a1["pop_exact"], a1["worst"], tie)
    top = max(a1["pop_exact"])
    tied = [i for i, v in enumerate(a1["pop_exact"]) if v == top]
    out = {"seed": seed, "A_top5": [int(i) for i in o[:5]], "A_tied_at_top": len(tied), "levels": {},
           "arrays": {"A_stage1_pop": a1["pop_score"].tolist()}}
    for e in energies:
        name = "main" if e == 500 else f"energy{e}"
        env, wc = run2.b_world(cfg, name)
        with streams.context(f"landscape:{name}"):
            b = search2.landscape_stage1(cfg, seed, run2.world_id(name), env, wc)
        sb = b["pop_score"]
        ranks = sps.rankdata(sb)
        out["levels"][str(e)] = {
            "best_B": float(sb.max()), "n_at_best": int(np.sum(sb == sb.max())),
            "A_top5_B_scores": [float(sb[i]) for i in out["A_top5"]],
            "A_top1_B_percentile": float((ranks[out["A_top5"][0]] - 1) / (n - 1)),
            "tau_b": float(sps.kendalltau(a1["pop_score"], sb, variant="b").statistic),
            # Many designs tie at the best B score; the first 3 in grid order are NOT "B's preferred
            # designs". The (speed, size) mix of every design within 0.5 of the best is.
            "B_first3_at_best_grid_order": [designs[int(i)] for i in np.argsort(-sb, kind="stable")[:3]],
            "B_near_best_speed_size_counts": sorted(
                [[float(k[0]), float(k[1]), int(v)] for k, v in __import__("collections").Counter(
                    (round(designs[int(i)]["speed"], 2), round(designs[int(i)]["size"], 2))
                    for i in np.where(sb >= sb.max() - 0.5)[0]).items()], key=lambda t: -t[2]),
            "n_alive_B": int(np.sum(sb > 0)), "steps": b["steps"],
            "A_tied_B_scores": {"min": float(sb[tied].min()), "q25": float(np.quantile(sb[tied], 0.25)),
                                "median": float(np.median(sb[tied])), "q75": float(np.quantile(sb[tied], 0.75)),
                                "max": float(sb[tied].max()), "share_below_20": float(np.mean(sb[tied] < 20)),
                                "share_zero": float(np.mean(sb[tied] == 0))}}
        out["arrays"][f"B_stage1_pop_{e}"] = sb.tolist()
    return out


def rule_one(args):
    """One rule seed. Writes its result to its own file (atomic) so an interrupted rule run
    can resume; a saved seed is reused only if commit, config hash and energies all match."""
    seed, cfg, energies, seed_dir, stamp = args
    path = os.path.join(seed_dir, f"{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old.get("stamp") == stamp and not stamp["quick"]:  # quick runs never reuse (code may have changed)
            return old
    t0 = time.process_time()
    sa = search2.a_search(cfg, seed, int(cfg["part2"]["search"]["stage2_sets"]),
                          int(cfg["part2"]["search"]["stage2_top"]))
    out = {"seed": seed, "stamp": stamp, "A_pop_winner": sa["designs"][sa["pop_winner"]], "levels": {}}
    # Same B-search and U2 streams as the main B (world id 0), as arm 4 will use in the final
    # runs (Section 10: arms use the main arm's streams). This also gives every energy level the
    # same random numbers, so the levels differ only in energy.
    wid = run2.WORLD_IDS["main"]
    for e in energies:
        env, wc = run2.b_world(cfg, f"energy{e}")
        sb = search2.b_search(cfg, seed, wid, env, wc)
        u2 = search2.u2_runs(cfg, seed, wid, env, wc, sb["winner"], sa["pop_winner"])
        out["levels"][str(e)] = {"B_winner": sa["designs"][sb["winner"]],
                                 "b_winner_total": u2["b_winner_total"], "a_winner_total": u2["a_winner_total"],
                                 "count": u2["count"], "b_winner": u2["b_winner"], "a_winner": u2["a_winner"]}
    out["process_seconds"] = time.process_time() - t0
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
    return out


def median_fraction(xs):
    """Exact median of a list of Fractions (mean of the middle two when the count is even)."""
    v = sorted(xs)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--rule", action="store_true")
    ap.add_argument("--round", type=int, choices=(1, 2), default=1,
                    help="rule round: 1 = seeds 2860-2869, mean gap (Section 9.3); "
                         "2 = seeds 2872-2881, median gap (2 Nov 2026, Section 17)")
    ap.add_argument("--energies", default=None,
                    help="optional; must equal the round's recorded levels (ROUND_LEVELS)")
    ap.add_argument("--probe-energies", default=",".join(str(e) for e in PROBE_ENERGIES))
    ap.add_argument("--probe-name", default="probe")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    over = {"part2": {"grid": {"points": 3}, "search": {"stage2_sets": 2, "stage2_top": 3}}} if args.quick else None
    cfg = config.load(ARM3, over)
    prov = config.provenance(cfg)
    if prov["git"]["dirty"] and not args.quick:
        raise SystemExit("uncommitted changes: commit first")
    outdir = os.path.join(ROOT, "results", "part2", "b_redesign" + ("_quick" if args.quick else ""))
    os.makedirs(outdir, exist_ok=True)
    if args.probe:
        pe = [int(e) for e in args.probe_energies.split(",")]
        with Pool(os.cpu_count()) as pool:
            res = pool.map(probe_one, [(s, cfg, pe) for s in PROBE_SEEDS])
        doc = {"step": args.probe_name, "seeds": list(PROBE_SEEDS), "energies": pe, "runs": res,
               "provenance": prov}
        path = os.path.join(outdir, f"{args.probe_name}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(doc, f, indent=1, default=float)
        os.replace(path + ".tmp", path)
        for e in pe:
            rows = [r["levels"][str(e)] for r in res]
            print(f"energy {e}: best B {[r['best_B'] for r in rows]}, top A design in B "
                  f"{[r['A_top5_B_scores'][0] for r in rows]} (percentile {[round(r['A_top1_B_percentile'], 2) for r in rows]}), "
                  f"tau {[round(r['tau_b'], 2) for r in rows]}, alive {[r['n_alive_B'] for r in rows]}")
            print("   most common (speed, size, count) within 0.5 of B's best:", [r["B_near_best_speed_size_counts"][:4] for r in rows])
            print("   A's tied-at-top designs in B:", [r["A_tied_B_scores"] for r in rows])
    if args.rule:
        energies = sorted(ROUND_LEVELS[args.round], reverse=True)
        if args.energies is not None and sorted({int(e) for e in args.energies.split(",")}, reverse=True) != energies:
            raise SystemExit(f"round {args.round}'s levels are fixed at {energies} (Section 17)")
        lo, hi = cfg["part2"]["seeds"]["b_redesign_rule" if args.round == 1 else "b_redesign_rule_round2"]
        all_seeds = tuple(range(lo, hi + 1))
        if args.round == 1:
            assert all_seeds == RULE_SEEDS
        seeds = all_seeds[:2] if args.quick else all_seeds
        tag = "rule" if args.round == 1 else "rule_round2"
        seed_dir = os.path.join(outdir, "rule_seeds" if args.round == 1 else "rule_round2_seeds")
        os.makedirs(seed_dir, exist_ok=True)
        stamp = {"commit": prov["git"]["commit"], "config_hash": prov["config_hash"], "energies": energies,
                 "quick": bool(args.quick), "round": args.round}
        with Pool(os.cpu_count()) as pool:
            res = []
            for r in pool.imap_unordered(rule_one, [(s, cfg, energies, seed_dir, stamp) for s in seeds]):
                res.append(r)
                print(f"seed {r['seed']} done ({len(res)}/{len(seeds)})", flush=True)
        res.sort(key=lambda r: r["seed"])
        summary, chosen = {}, None
        for e in energies:  # highest first, so the first that qualifies is the highest
            rows = [r["levels"][str(e)] for r in res]
            gaps = [Fraction(r["b_winner_total"] - r["a_winner_total"], r["count"]) for r in rows]
            gap = sum(gaps, Fraction(0)) / len(rows)
            med = median_fraction(gaps)
            bsc = sum((Fraction(r["b_winner_total"], r["count"]) for r in rows), Fraction(0)) / len(rows)
            # Round 1: mean gap >= 5 (Section 9.3). Round 2: median gap >= 5 (2 Nov 2026).
            # Both: mean B-winner U2 score >= 20.
            ok = (gap if args.round == 1 else med) >= 5 and bsc >= 20
            summary[str(e)] = {"mean_gap": float(gap), "median_gap": float(med),
                               "gaps_per_seed": [float(g) for g in gaps], "mean_B_winner": float(bsc),
                               "mean_A_winner": float(bsc - gap), "qualifies": bool(ok)}
            if ok and chosen is None:
                chosen = e
            print(f"energy {e}: B winner {float(bsc):.2f}, A winner {float(bsc - gap):.2f}, mean gap {float(gap):.2f}, "
                  f"median gap {float(med):.2f}{'  <- qualifies' if ok else ''}")
        doc = {"step": tag, "round": args.round, "rule": "mean gap >= 5" if args.round == 1 else "median gap >= 5",
               "seeds": list(seeds), "energies": energies, "summary": summary,
               "chosen_energy": chosen, "runs": res, "provenance": prov}
        path = os.path.join(outdir, f"{tag}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(doc, f, indent=1, default=float)
        os.replace(path + ".tmp", path)
        if chosen is None:
            raise SystemExit("no energy level qualifies; no setting selected")
        print(f"chosen energy: {chosen}")
        if not args.quick:
            with open(os.path.join(ROOT, "configs", "part2", "arm4.yaml"), "w") as f:
                f.write(f"# Written by scripts/part2/b_redesign.py --rule --round {args.round} "
                        f"from results/part2/b_redesign/{tag}.json.\n"
                        "# Do not edit by hand. Arm 4: world B with less energy per generation.\n"
                        f"extends: arm3.yaml\npart2:\n  arm4_energy: {chosen}\n"
                        f"  arm4_rule_round: {args.round}\n"
                        f"  arm4_rule_commit: {prov['git']['commit']}\n")
            print("wrote configs/part2/arm4.yaml")


if __name__ == "__main__":
    main()
