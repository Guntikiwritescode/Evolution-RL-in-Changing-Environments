"""Independent NPZ-to-outcome-table audit. Never modifies experimental results."""
import argparse
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    source = ROOT / "results/part2/final"
    manifest = json.loads((source / "run.json").read_text())
    cfg = manifest["provenance"]["config"]
    assert cfg["phases"]["train_generations"] == 100
    assert manifest["seeds"] == list(range(2000, 2200))
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config_hash = digest(cfg)
    frozen_hash = digest({k: cfg[k] for k in ("world", "env_A", "env_B", "blob_default", "mutation_sd", "phases")})
    assert config_hash == manifest["provenance"]["config_hash"]
    assert frozen_hash == manifest["frozen_hash"]
    commits = set()
    for seed in manifest["seeds"]:
        meta = json.loads((source / f"seed{seed}.json").read_text())
        done = json.loads((source / f"seed{seed}.done").read_text())
        assert meta["seed"] == seed
        assert meta["config_hash"] == meta["provenance"]["config_hash"] == done["config_hash"] == config_hash
        assert meta["frozen_hash"] == frozen_hash
        assert not meta["provenance"]["git"]["dirty"]
        assert meta["provenance"]["git"]["commit"] == done["commit"]
        commits.add(done["commit"])
    with (source / "analysis/outcomes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (source / "analysis/checks.csv").open(newline="") as handle:
        saved_checks = {int(row["seed"]): row for row in csv.DictReader(handle)}
    assert sorted(saved_checks) == manifest["seeds"]
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(row)
    assert sorted(grouped) == list(range(2000, 2200))
    missing, failures, checked_seeds = [], [], []
    checks = 0
    clone_array_checks = 0
    search_check_fields = 0
    count_rows = 0
    for seed, seed_rows in sorted(grouped.items()):
        path = source / f"seed{seed}.npz"
        if not path.exists():
            missing.append(seed)
            continue
        with np.load(path, allow_pickle=False) as arrays:
            meta = json.loads((source / f"seed{seed}.json").read_text())
            search = meta["search_A"]
            a_generations = arrays["A__GS_pop_frozen__g_gen"]
            a_window = (a_generations >= 10) & (a_generations <= 99)
            assert int(a_window.sum()) == 90
            a_mean = Fraction(int(arrays["A__GS_pop_frozen__g_population"][a_window].sum()), 90)
            controls = {
                "GS_pop_frozen_A_mean_10_99": float(a_mean),
                "u3_shortfall": float(Fraction(*search["pop_winner_stage2_exact"]) - a_mean),
                "pop_winner_stage2_score": search["pop_winner_stage2_score"],
                "steps_search_A": search["steps_stage1"] + search["steps_stage2"],
            }
            for field in ("tied_stage1_pop", "tied_stage1_food", "tied_stage2_pop", "tied_stage2_food"):
                controls[field] = search[field]
            for world, u2 in meta["u2"].items():
                controls[f"u2_{world}"] = float(Fraction(u2["b_winner_total"] - u2["a_winner_total"], u2["count"]))
                controls[f"steps_search_B_{world}"] = sum(meta["search_B"][world][k] for k in ("steps_stage1", "steps_stage2"))
            for key, value in controls.items():
                assert math.isclose(value, float(saved_checks[seed][key]), rel_tol=1e-12, abs_tol=1e-12), (seed, key, "search check table")
                search_check_fields += 1
            # Predation cannot distinguish same-size clones: main and arm 2
            # must agree over their common horizon, including individual logs.
            for group in ("GS_food_frozen", "GS_pop_frozen", "ET_cloneB", "ORACLE_B"):
                main_prefix = f"B__main__{group}__"
                arm2_prefix = f"B__arm2__{group}__"
                end = int(arrays[arm2_prefix + "g_gen"].max())
                for key in arrays.files:
                    if not key.startswith(arm2_prefix):
                        continue
                    field = key[len(arm2_prefix):]
                    generation_field = "g_gen" if field.startswith("g_") else "b_gen"
                    selected = arrays[main_prefix + generation_field] <= end
                    assert np.array_equal(arrays[main_prefix + field][selected], arrays[key]), (seed, group, field, "clone invariance")
                    clone_array_checks += 1
            expected_prefixes = {"B__" + r["world"] + "__" + r["group"] for r in seed_rows}
            assert len(expected_prefixes) == len(seed_rows), (seed, "duplicate outcome rows")
            actual_prefixes = {k[:-7] for k in arrays.files if k.startswith("B__") and k.endswith("__g_gen")}
            assert expected_prefixes == actual_prefixes, (seed, "table/raw group coverage")
            for row in seed_rows:
                prefix = "B__" + row["world"] + "__" + row["group"] + "__g_"
                fields = {k[len(prefix):]: arrays[k] for k in arrays.files if k.startswith(prefix)}
                generations = fields["gen"].astype(int)
                assert len(set(generations)) == len(generations), (seed, prefix, "duplicate generation")
                by_generation = {int(g): i for i, g in enumerate(generations)}
                idx = [by_generation[100 + g] for g in range(50)]
                pop = fields["population"][idx].astype(int)
                survivors = fields["survivors"][idx].astype(int)
                births = fields["births"][idx].astype(int)
                assert np.array_equal(pop[1:], (survivors + births)[:-1]), (seed, prefix, "population continuity")
                assert np.all(pop >= 0), (seed, prefix, "negative population")
                food = fields["food_eaten"][idx] + fields["prey_eaten"][idx]
                ratios = [float(food[g]) / int(pop[g]) for g in range(40, 50) if pop[g] > 0]
                food_count = int(row["world"][4:]) if row["world"].startswith("food") else 25
                reached = [g for g in range(1, 50) if pop[g] >= math.ceil(.9 * food_count)]
                computed = {
                    "P1_total": int(sum(pop[40:50])),
                    "P1": sum(int(p) for p in pop[40:50]) / 10,
                    "P2": bool(survivors[49] + births[49] == 0),
                    "S5": bool(pop[0] > 0),
                    "S6": sum(int(p) for p in pop[1:50]) / 49,
                    "S4": sum(ratios) / len(ratios) if ratios else float("nan"),
                    "S4_n": len(ratios),
                    "S1": reached[0] if reached else None,
                    "B0_population": int(pop[0]),
                    "prey_eaten_B": int(sum(fields["prey_eaten"][idx])),
                    "steps_B": int(sum(fields["creature_steps"][idx])),
                }
                if all(g in by_generation for g in range(290, 300)):
                    late = [by_generation[g] for g in range(290, 300)]
                    last = by_generation[299]
                    computed["late_P1_total"] = int(sum(fields["population"][late]))
                    computed["late_P2"] = bool(fields["survivors"][last] + fields["births"][last] == 0)
                for key in ("late_P1_total", "late_P2"):
                    if key not in computed:
                        computed[key] = None
                for key, value in computed.items():
                    saved = row[key]
                    if value is None:
                        match = saved == ""
                    elif isinstance(value, bool):
                        match = saved == str(value)
                    else:
                        number = float(saved)
                        match = (math.isnan(value) and math.isnan(number)) or math.isclose(value, number, rel_tol=1e-12, abs_tol=1e-12)
                    checks += 1
                    if not match:
                        failures.append({"seed": seed, "world": row["world"], "group": row["group"], "field": key, "computed": value, "saved": saved})
                count_rows += 1
        checked_seeds.append(seed)
    report = {"source": "results/part2/final", "checked_seeds": checked_seeds,
              "metadata_pairs_checked": 200, "recorded_commits": sorted(commits),
              "config_hash": config_hash, "frozen_hash": frozen_hash,
              "missing_seeds": missing, "rows": count_rows, "field_checks": checks,
              "clone_array_checks": clone_array_checks,
              "search_check_fields": search_check_fields,
              "failures": failures, "complete": not missing and not failures,
              "scope": "Outcome calculations and population continuity from the recorded trajectories."}
    target = ROOT / "results/audit/part2_trajectories_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "field_checks", "complete")}))
    print(f"{len(checked_seeds)} seeds checked, {len(missing)} missing, {len(failures)} mismatches")
    if failures or (missing and not args.allow_partial):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
