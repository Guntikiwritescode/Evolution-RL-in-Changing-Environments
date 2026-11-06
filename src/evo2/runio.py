"""Provenance, checkpoints and run folders for the Part 3 scripts.

Lessons built in (docs/MISTAKES.md): the clean-tree check uses the project's
own command, which leaves out results/ (#12, #34); files are written to a
temporary name and renamed, so a stopped write never looks finished (#26);
a run is complete only when done.json exists.
"""
from __future__ import annotations

import json
import os
import pickle
import platform
import subprocess
import time
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[2]


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    dirty = run("status", "--porcelain", "--", ".", ":(exclude)results").strip()
    return {"commit": run("rev-parse", "HEAD").strip(), "clean": dirty == "", "dirty_files": dirty.splitlines()}


def _gpu_info() -> list[str]:
    """GPU name, driver and memory from nvidia-smi, if there is one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _cuda_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        for part in out.split("|"):
            if "CUDA Version" in part:
                return part.split("CUDA Version:")[1].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return None


def enable_compile_cache(path: str | os.PathLike | None = None) -> str:
    """Keep compiled programs on disk and reuse them in later processes (JAX's
    persistent compilation cache). A program is reused only for the same
    computation, JAX version and device type, so results do not change; it
    only saves compile time. Folder: `path`, else $EVO2_JAX_CACHE, else
    ~/.cache/evo2-jax (outside the repo, so it never makes the tree dirty).
    Returns the folder used. Call before the first compile."""
    folder = Path(path or os.environ.get("EVO2_JAX_CACHE") or Path.home() / ".cache" / "evo2-jax")
    folder.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(folder))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return str(folder)


def hardware() -> dict:
    devs = jax.devices()
    try:
        platform_version = devs[0].client.platform_version
    except Exception:  # not every backend exposes it
        platform_version = None
    return {
        "gpus": _gpu_info(),
        "cuda_driver_version": _cuda_version(),
        "jax_platform_version": platform_version,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [f"{d.platform}:{getattr(d, 'device_kind', '?')}" for d in devs],
        "host": platform.node(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


def require_clean(purpose: str, quick: bool) -> dict:
    """Pilot, tuning and final runs must start from a clean, committed tree.
    Only --quick runs on test seeds may run from a dirty tree."""
    g = git_state()
    if not g["clean"] and not (quick and purpose == "test"):
        raise SystemExit(f"refusing to run a {purpose} run from a dirty tree: {g['dirty_files']}")
    return g


def write_atomic(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj):
    write_atomic(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())


def save_pickle(path: Path, obj):
    write_atomic(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def append_line(path: Path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
