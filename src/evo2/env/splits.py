"""Loading the Part 3 environment splits (SPEC_part3.md, Section 3.4).

This is the only module that reads configs/part3/test_split.yaml, and it does
so only when called with final=True. A test scans the code to check that no
other module or script names that file, and that only evaluate.py calls
load_test_split.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from evo2.env.generator import EnvRanges, env_from_seed, params_from_dict, ranges_from_dict
from evo2.env.world import EnvParams, WorldConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "part3"
_TEST_FILE = "test_split.yaml"


class TestSplitLocked(RuntimeError):
    """Raised when the test split is requested without final=True."""


def _read(name: str, config_dir: Path | None = None) -> dict:
    with open((config_dir or CONFIG_DIR) / name) as fh:
        return yaml.safe_load(fh)


def load_world_config(mode: str, config_dir: Path | None = None) -> WorldConfig:
    """World constants for 'episodic' or 'ecological' mode."""
    if mode not in ("episodic", "ecological"):
        raise ValueError(f"unknown mode {mode!r}")
    d = _read("world.yaml", config_dir)
    return WorldConfig.from_dict({**d["common"], **d[mode]})


def load_train(config_dir: Path | None = None) -> tuple[EnvRanges, EnvParams]:
    """(train ranges, canonical environment)."""
    d = _read("train.yaml", config_dir)
    return ranges_from_dict(d["train_ranges"]), params_from_dict(d["canonical"])


def _seed_list(lohi) -> list[int]:
    lo, hi = lohi
    return list(range(int(lo), int(hi) + 1))


def load_validation(config_dir: Path | None = None) -> dict:
    """Validation environments: {'drawn': [...], 'corners': [...], 'episode_seeds': [...]}."""
    d = _read("validation.yaml", config_dir)
    if d["drawn"]["ranges"] != "train":
        raise ValueError("validation environments must be drawn from the train ranges")
    train_ranges, _ = load_train(config_dir)
    drawn = [env_from_seed("validation", s, train_ranges) for s in _seed_list(d["drawn"]["env_seeds"])]
    corners = [params_from_dict(c) for c in d["corners"]]
    return {"drawn": drawn, "corners": corners, "episode_seeds": _seed_list(d["episode_seeds"])}


def load_pilot_shift(config_dir: Path | None = None) -> dict:
    """Pilot-only stand-ins for T2 and T3 (configs/part3/pilot_shift.yaml):
    the same ranges, new environments drawn with pilot-evaluation seeds."""
    d = _read("pilot_shift.yaml", config_dir)
    out = {}
    for name in ("P2", "P3"):
        r = ranges_from_dict(d[name]["ranges"])
        out[name] = [env_from_seed("pilot_eval", s, r) for s in _seed_list(d[name]["env_seeds"])]
    out["episode_seeds"] = _seed_list(d["episode_seeds"])
    return out


def load_test_split(final: bool = False, config_dir: Path | None = None) -> dict:
    """The locked test set. Raises TestSplitLocked unless final is True."""
    if final is not True:
        raise TestSplitLocked(
            "the test split is locked: only scripts/part3/evaluate.py --final may load it")
    d = _read(_TEST_FILE, config_dir)
    out = {}
    for name in ("T1", "T2", "T3"):
        r = ranges_from_dict(d[name]["ranges"])
        out[name] = [env_from_seed("test_env", s, r) for s in _seed_list(d[name]["env_seeds"])]
    out["episode_seeds"] = _seed_list(d["episode_seeds"])
    return out
