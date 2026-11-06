"""Load configs and record provenance (config hash, git commit, versions, hardware)."""
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path, overrides=None):
    """Load a YAML config. A file may name a parent with 'extends: <file>'."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load(os.path.join(os.path.dirname(path), parent)), cfg)
    return _merge(cfg, overrides)


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        head = "none"
    try:
        # Output folders under results/ don't count: a run writes there while it runs,
        # and a restart must not be refused because of its own outputs (PREREG_part2 18.8).
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
                                    cwd=REPO_ROOT, capture_output=True, text=True,
                                    check=True).stdout.strip())
    except Exception:
        dirty = True
    return {"commit": head, "dirty": dirty}


def versions():
    import numba
    import numpy
    import scipy
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "numba": numba.__version__, "scipy": scipy.__version__, "pyyaml": yaml.__version__}


def hardware():
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "cpu_count": os.cpu_count()}


def provenance(cfg):
    return {"config": cfg, "config_hash": config_hash(cfg), "git": git_commit(),
            "versions": versions(), "hardware": hardware()}
