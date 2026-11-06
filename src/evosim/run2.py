"""Part 2: everything for one run seed (docs/PREREG_part2.md).

run_seed() does, in order:
1. the confirmatory A search (Section 3) and the B searches + U2 runs for the
   main B, arm 3 and arm 4 (Sections 3, 8, 9.2, 9.3);
2. A phase for every group that needs one (Section 4);
3. B phase of every group in every B world: main, arm 2, arm 3, arm 4 (confirmatory)
   and, on exploratory seeds, the suite, the mutation-size reruns, the longer
   B, ET_scratchB, the stage-1-only winners and the A-checks (Section 18);
4. on landscape seeds, the landscape check (Section 18.4).

It returns (meta, arrays): meta is JSON-safe; arrays holds every raw log.
"""
import copy
import hashlib
import json
import time

import numpy as np

from . import config, search2, sim2, streams, world
from .runner import Log, et_start_traits

FROZEN_SECTIONS = ("world", "env_A", "env_B", "blob_default", "mutation_sd", "phases")

# B world ids, used in stream keys. Never renumber.
WORLD_IDS = {"main": 0, "arm3": 1, "side400": 2, "side500": 3, "side600": 4, "side800": 5,
             "side1000": 6, "food15": 7, "food40": 8, "ticks100": 9, "uniform": 10,
             "cann_arm3": 11, "arm2": 12,
             # B redesign (1 Nov 2026): less energy per generation in B
             "energy400": 13, "energy350": 14, "energy300": 15, "energy250": 16, "energy200": 17,
             "energy150": 18, "energy100": 19, "arm4": 20}
def world_id(name):
    """Stream id of a B world. Energy levels not in WORLD_IDS get 1000 + energy."""
    if name in WORLD_IDS:
        return WORLD_IDS[name]
    if name.startswith("energy"):
        return 1000 + int(name[6:])
    raise KeyError(name)


CONFIRMATORY_GROUPS = ("ET", "ET_freezeB", "ET_cloneB", "GS_food_frozen", "GS_pop_frozen",
                       "GS_pop_mut", "SV_frozen", "ORACLE_B")
SUITE_GROUPS = CONFIRMATORY_GROUPS[:-1]
VARIED_GROUPS = ("ET", "ET_freezeB", "GS_pop_mut", "SV_frozen")


def frozen_hash(cfg):
    blob = json.dumps({k: cfg[k] for k in FROZEN_SECTIONS}, sort_keys=True,
                      separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def in_range(seed, rng_pair):
    return int(rng_pair[0]) <= int(seed) <= int(rng_pair[1])


def b_world(cfg, name):
    """(env_B, world_cfg_B) for a named B world."""
    p2 = cfg["part2"]
    env = copy.deepcopy(cfg["env_B"])
    wc = copy.deepcopy(cfg["world"])
    if name == "main":
        pass
    elif name == "arm2":
        wc["cannibalism"] = True
    elif name in ("arm3", "cann_arm3"):
        if p2.get("arm3_side") is None:
            raise ValueError("arm 3's side is not set (run the map-size rule first)")
        env["size"] = float(p2["arm3_side"])
        if name == "cann_arm3":
            wc["cannibalism"] = True
    elif name.startswith("side"):
        env["size"] = float(name[4:])
    elif name.startswith("food"):
        env["food"] = int(name[4:])
    elif name == "ticks100":
        wc["tick_cap"] = int(p2["suite_tick_cap"])
    elif name == "uniform":
        env["layout"] = "uniform"
    elif name.startswith("energy"):
        wc["energy_per_generation"] = float(name[6:])
    elif name == "arm4":
        if p2.get("arm4_energy") is None:
            raise ValueError("arm 4's energy is not set (run the B-redesign rule first)")
        wc["energy_per_generation"] = float(p2["arm4_energy"])
    else:
        raise KeyError(name)
    return env, wc


def suite_variants(cfg):
    """Suite variants to run (Section 18.4). The side equal to arm 3's is arm 3
    itself and is not run again."""
    p2 = cfg["part2"]
    a3 = p2.get("arm3_side")
    names = [f"side{int(s)}" for s in p2["suite_sides"] if a3 is None or float(s) != float(a3)]
    names += [f"food{int(f)}" for f in p2["suite_food"]] + ["ticks100", "uniform"]
    if a3 is not None:
        names.append("cann_arm3")
    return names


def sv_designs(cfg, seed, n):
    """SV_frozen: each trait log-uniform over the grid range, drawn separately per blob."""
    axes = search2.grid_points(cfg)
    u = streams.rng(seed, "sv_designs").random((n, len(world.TRAITS)))
    out = []
    for row in u:
        d = {}
        for k, t in enumerate(world.TRAITS):
            lo, hi = np.log(axes[t][0]), np.log(axes[t][-1])
            d[t] = float(np.exp(lo + row[k] * (hi - lo)))
        out.append(d)
    return out


def distinct_designs(pop):
    if world.pop_size(pop) == 0:
        return 0
    m = np.column_stack([pop[t] for t in world.TRAITS])
    return int(np.unique(m, axis=0).shape[0])


class _Timer:
    def __init__(self):
        self.parts = {}

    def run(self, label, fn, *a, **k):
        t0 = time.process_time()
        out = fn(*a, **k)
        self.parts[label] = self.parts.get(label, 0.0) + time.process_time() - t0
        return out


def _put(arrays, prefix, log):
    for k, v in log.arrays().items():
        arrays[f"{prefix}__{k}"] = v


def run_seed(seed, cfg, exploratory=None, landscape=None):
    """Everything for one run seed. exploratory / landscape default to the
    seed ranges in cfg['part2']['seeds'] (the pilot passes True)."""
    p2 = cfg["part2"]
    sd = {t: float(v) for t, v in cfg["mutation_sd"].items()}
    nc = int(p2["n_clones"])
    n_b = int(cfg["phases"]["transfer_generations"])
    if exploratory is None:
        exploratory = in_range(seed, p2["seeds"]["exploratory"])
    if landscape is None:
        landscape = in_range(seed, p2["seeds"]["landscape"])
    has_arm3 = p2.get("arm3_side") is not None
    has_arm4 = p2.get("arm4_energy") is not None
    # B worlds with their own B search and U2 runs; they also get the exploratory items (Section 9.3)
    searched = ["main"] + (["arm3"] if has_arm3 else []) + (["arm4"] if has_arm4 else [])
    T = _Timer()
    arrays, meta = {}, {"seed": int(seed), "exploratory": bool(exploratory),
                        "landscape": bool(landscape), "arm3_side": p2.get("arm3_side"),
                        "arm4_energy": p2.get("arm4_energy")}
    t_wall = time.time()

    # ---- 1. searches (confirmatory)
    n_sets, top = int(p2["search"]["stage2_sets"]), int(p2["search"]["stage2_top"])
    with streams.context("search_A"):
        sa = T.run("conf:search_A", search2.a_search, cfg, seed, n_sets, top)
    designs = sa["designs"]
    arrays["SA__stage1_pop_score"] = sa["stage1"]["pop_score"]
    arrays["SA__stage1_food_score"] = sa["stage1"]["food_score"]
    arrays["SA__stage1_worst"] = sa["stage1"]["worst"]
    arrays["SA__stage1_pops"] = sa["stage1"]["pops"]
    arrays["SA__tie_keys"] = sa["tie_keys"]
    arrays["SA__designs"] = np.array([[d[t] for t in world.TRAITS] for d in designs])
    meta["search_A"] = {k: sa[k] for k in (
        "top_pop", "top_food", "pop_winner", "food_winner", "pop_winner_stage2_score",
        "food_winner_stage2_score", "pop_winner_stage2_exact", "stage1_pop_winner", "stage1_food_winner",
        "tied_stage1_pop", "tied_stage1_food", "tied_stage2_pop", "tied_stage2_food",
        "n_sets", "top", "steps_stage1", "steps_stage2")}
    meta["search_A"]["stage2_pop_scores"] = {str(k): v for k, v in sa["stage2_pop_scores"].items()}
    meta["search_A"]["stage2_food_scores"] = {str(k): v for k, v in sa["stage2_food_scores"].items()}
    meta["designs"] = {"pop_winner": designs[sa["pop_winner"]],
                       "food_winner": designs[sa["food_winner"]],
                       "stage1_pop_winner": designs[sa["stage1_pop_winner"]],
                       "stage1_food_winner": designs[sa["stage1_food_winner"]]}

    meta["search_B"], meta["u2"] = {}, {}
    for w in searched:
        env_w, wc_w = b_world(cfg, w)
        wid = WORLD_IDS["main"]  # arms 3 and 4 use the main B's search and U2 streams (Section 10)
        with streams.context(f"search_B:{w}"):
            sb = T.run(f"conf:search_B_{w}", search2.b_search, cfg, seed, wid, env_w, wc_w)
        arrays[f"SB__{w}__stage1_pop_score"] = sb["stage1_pop_score"]
        arrays[f"SB__{w}__stage1_pops"] = sb["stage1_pops"]
        meta["search_B"][w] = {k: sb[k] for k in ("top", "winner", "winner_stage2_score",
                                                   "tied_stage1", "tied_stage2",
                                                   "steps_stage1", "steps_stage2")}
        meta["search_B"][w]["stage2_scores"] = {str(k): v for k, v in sb["stage2_scores"].items()}
        meta["designs"][f"B_winner_{w}"] = designs[sb["winner"]]
        with streams.context(f"u2:{w}"):
            meta["u2"][w] = T.run(f"conf:u2_{w}", search2.u2_runs, cfg, seed, wid, env_w, wc_w,
                                  sb["winner"], sa["pop_winner"])

    # ---- 2. A phase
    default = et_start_traits(cfg)
    a_groups = {
        "ET": ([default] * nc, sd),
        "GS_food_frozen": ([designs[sa["food_winner"]]] * nc, sim2.ZERO_SD),
        "GS_pop_frozen": ([designs[sa["pop_winner"]]] * nc, sim2.ZERO_SD),
        "GS_pop_mut": ([designs[sa["pop_winner"]]] * nc, sd),
    }
    with streams.context("sv"):
        a_groups["SV_frozen"] = (sv_designs(cfg, seed, nc), sim2.ZERO_SD)
    if exploratory:
        a_groups["GS_pop_s1"] = ([designs[sa["stage1_pop_winner"]]] * nc, sim2.ZERO_SD)
        a_groups["GS_food_s1"] = ([designs[sa["stage1_food_winner"]]] * nc, sim2.ZERO_SD)
        for f in p2["mutation_factors"]:
            tag = f"x{f:g}"
            a_groups[f"ET_{tag}"] = ([default] * nc, sim2.scaled_sd(sd, f))
            a_groups[f"GS_pop_mut_{tag}"] = ([designs[sa["pop_winner"]]] * nc, sim2.scaled_sd(sd, f))
    end_A = {}
    for g, (ds, sdg) in a_groups.items():
        log = Log()
        label = "conf" if g in CONFIRMATORY_GROUPS else "expl"
        with streams.context(f"deploy_A:{g}"):
            end_A[g] = T.run(f"{label}:A_{g}", sim2.deploy_A, ds, cfg, seed, sdg, log)
        _put(arrays, f"A__{g}", log)
        for t in world.TRAITS:
            arrays[f"ENDA__{g}__{t}"] = np.asarray(end_A[g][t])
    meta["distinct_designs_B0"] = {g: distinct_designs(end_A[g]) for g in ("ET", "GS_pop_mut", "SV_frozen")}

    with streams.context("clone_pick"):
        clone_idx = sim2.pick_clone_index(end_A["ET"], seed, 0)
    meta["clone_index"] = clone_idx
    meta["clone_design"] = ({t: float(end_A["ET"][t][clone_idx]) for t in world.TRAITS}
                            if clone_idx >= 0 else None)

    def cloned(pop, idx):
        return sim2.clones_of(pop, idx) if idx >= 0 else sim2.empty_like(pop)

    def b_starts(et_end, idx, gs_food, gs_pop, gs_mut, sv):
        return {"ET": (et_end, sd), "ET_freezeB": (et_end, sim2.ZERO_SD),
                "ET_cloneB": (cloned(et_end, idx), sim2.ZERO_SD),
                "GS_food_frozen": (gs_food, sim2.ZERO_SD), "GS_pop_frozen": (gs_pop, sim2.ZERO_SD),
                "GS_pop_mut": (gs_mut, sd), "SV_frozen": (sv, sim2.ZERO_SD)}

    starts = b_starts(end_A["ET"], clone_idx, end_A["GS_food_frozen"], end_A["GS_pop_frozen"],
                      end_A["GS_pop_mut"], end_A["SV_frozen"])

    def run_b(world_name, groups, n_gens, label, extra=None):
        env_w, wc_w = b_world(cfg, world_name)
        for g in groups:
            log = Log()
            ctx = streams.context(f"deploy_B:{world_name}:{g}")
            ctx.__enter__()
            if g == "ORACLE_B":
                src = {"arm3": "arm3", "cann_arm3": "arm3", "arm4": "arm4"}.get(world_name, "main")
                win = designs[meta["search_B"][src]["winner"]]
                T.run(f"{label}:B_{world_name}", sim2.start_in_B, [win] * nc, env_w, wc_w,
                      sim2.ZERO_SD, n_gens, cfg, seed, log)
            elif g == "ET_scratchB":
                T.run(f"{label}:B_{world_name}", sim2.start_in_B, [default] * nc, env_w, wc_w,
                      sd, n_gens, cfg, seed, log)
            else:
                pop0, sdg = (extra or starts)[g]
                T.run(f"{label}:B_{world_name}", sim2.deploy_B, pop0, env_w, wc_w, sdg, n_gens,
                      cfg, seed, log)
            ctx.__exit__(None, None, None)
            _put(arrays, f"B__{world_name}__{g}", log)

    # ---- 3. B phases
    long_b = int(p2["long_B_generations"]) if exploratory else n_b
    run_b("main", CONFIRMATORY_GROUPS, long_b, "conf")
    run_b("arm2", CONFIRMATORY_GROUPS, n_b, "conf")
    if has_arm3:
        run_b("arm3", CONFIRMATORY_GROUPS, long_b, "conf")
    if has_arm4:
        run_b("arm4", CONFIRMATORY_GROUPS, long_b, "conf")

    if exploratory:
        extra_starts = dict(starts)
        extra_starts["GS_pop_s1"] = (end_A["GS_pop_s1"], sim2.ZERO_SD)
        extra_starts["GS_food_s1"] = (end_A["GS_food_s1"], sim2.ZERO_SD)
        for w in searched:
            run_b(w, ("ET_scratchB",), n_b, "expl")
            run_b(w, ("GS_pop_s1", "GS_food_s1"), n_b, "expl", extra_starts)
        for v in suite_variants(cfg):
            run_b(v, VARIED_GROUPS if v == "cann_arm3" else SUITE_GROUPS, n_b, "expl")
        meta["clone_index_mut"] = {}
        for k, f in enumerate(p2["mutation_factors"], start=1):
            tag = f"x{f:g}"
            sdf = sim2.scaled_sd(sd, f)
            et_end = end_A[f"ET_{tag}"]
            with streams.context("clone_pick"):
                idx = sim2.pick_clone_index(et_end, seed, k)
            meta["clone_index_mut"][tag] = idx
            st = {f"ET_{tag}": (et_end, sdf), f"ET_freezeB_{tag}": (et_end, sim2.ZERO_SD),
                  f"ET_cloneB_{tag}": (cloned(et_end, idx), sim2.ZERO_SD),
                  f"GS_pop_mut_{tag}": (end_A[f"GS_pop_mut_{tag}"], sdf)}
            for w in searched:
                run_b(w, tuple(st), n_b, "expl", st)
        # A-checks (Section 4): continue ET's end-of-A population in A for 50 generations
        n_ac = int(p2["acheck_generations"])
        env_a = cfg["env_A"]
        n_a = int(cfg["phases"]["train_generations"])
        for name, pop0, sdg in (("mut", end_A["ET"], sd), ("frozen", end_A["ET"], sim2.ZERO_SD),
                                ("clone", cloned(end_A["ET"], clone_idx), sim2.ZERO_SD)):
            log = Log()
            with streams.context("acheck"):
                T.run("expl:acheck", sim2.run_streamed, copy.deepcopy(pop0), env_a, cfg["world"], sdg,
                      n_ac, seed, "acheck", (), log, n_a, "A")
            _put(arrays, f"ACHECK__{name}", log)

    # ---- 4. landscape check
    if landscape:
        meta["landscape_steps"] = {}
        for v in suite_variants(cfg):
            env_v, wc_v = b_world(cfg, v)
            with streams.context(f"landscape:{v}"):
                ls = T.run("expl:landscape", search2.landscape_stage1, cfg, seed, world_id(v), env_v, wc_v)
            arrays[f"LS__{v}__pop_score"] = ls["pop_score"]
            meta["landscape_steps"][v] = ls["steps"]

    meta["b_worlds"] = {w: dict(zip(("env_B", "world"), b_world(cfg, w)))
                        for w in ["main", "arm2"] + (["arm3"] if has_arm3 else [])
                        + (["arm4"] if has_arm4 else []) + (suite_variants(cfg) if exploratory else [])}
    meta["process_seconds"] = T.parts
    meta["process_seconds_conf"] = sum(v for k, v in T.parts.items() if k.startswith("conf:"))
    meta["process_seconds_expl"] = sum(v for k, v in T.parts.items() if k.startswith("expl:"))
    meta["wall_seconds"] = time.time() - t_wall
    meta["config_hash"] = config.config_hash(cfg)
    meta["frozen_hash"] = frozen_hash(cfg)
    return meta, arrays
