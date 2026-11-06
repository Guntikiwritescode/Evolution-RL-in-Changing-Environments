"""The final job's failure paths (scripts/part3/pod/final_job.sh).

A fake python stands in for train.py, curves.py and evaluate.py: it records each call, and its
train.py finishes, fails (failed.json) or crashes depending on the seed.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "scripts/part3/pod/final_job.sh"

FAKE = r"""#!/bin/bash
echo "$*" >> "$CALLS"
script="$1"; shift
case "$script" in
  scripts/part3/train.py)
    while [ $# -gt 0 ]; do case "$1" in --method) m="$2"; shift;; --seed) s="$2"; shift;; --out) o="$2"; shift;; esac; shift; done
    d="$o/${m}_seed${s}"; mkdir -p "$d"
    case "$s" in
      2) echo '{}' > "$d/failed.json";;          # parameters stopped being finite
      3) exit 1;;                                  # crash
      *) echo '{}' > "$d/done.json";;
    esac;;
  scripts/part3/evaluate.py)
    echo '{"look": 1}' >> results/part3/test_access_log.jsonl;;
esac
exit 0
"""


def run_job(tmp_path, runs, before_log=None):
    code, out, logs = tmp_path / "code", tmp_path / "vol/final/p1", tmp_path / "logs"
    for p in (code, out, logs):
        p.mkdir(parents=True, exist_ok=True)
    py = tmp_path / "fakepy"
    py.write_text(FAKE)
    py.chmod(0o755)
    if before_log is not None:
        (out / "test_access_log.jsonl").write_text(before_log)
    env = {**os.environ, "OUT": str(out), "RUNS": runs, "PY": str(py), "LOGDIR": str(logs),
           "CALLS": str(tmp_path / "calls.txt")}
    r = subprocess.run(["bash", str(JOB)], cwd=code, env=env, capture_output=True, text=True)
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    return r, out, (logs / "status.txt").read_text(), calls


def test_failed_and_crashed_runs_are_left_out_and_the_rest_evaluated(tmp_path):
    r, out, status, calls = run_job(tmp_path, "oe:1 oe:2 oe:3 ppo_dr:4")
    assert r.returncode == 0, r.stderr
    good = f"{out}/oe_seed1 {out}/ppo_dr_seed4"
    curves = [c for c in calls if c.startswith("scripts/part3/curves.py")]
    evals = [c for c in calls if c.startswith("scripts/part3/evaluate.py")]
    assert curves == [f"scripts/part3/curves.py --runs {good}"]
    assert evals == [f"scripts/part3/evaluate.py --runs {good} --split test --final --adapt"]
    assert "run_failed oe 2" in status and "run_failed oe 3" in status
    assert "trained oe 1" in status and "trained ppo_dr 4" in status and "evaluation done" in status
    assert (out / "status.txt").read_text() == status           # the status is copied to the volume


def test_access_log_is_on_the_volume_and_only_appended(tmp_path):
    r, out, status, calls = run_job(tmp_path, "oe:1", before_log='{"look": 0}\n')
    assert r.returncode == 0, r.stderr
    lines = (out / "test_access_log.jsonl").read_text().splitlines()
    assert [json.loads(x)["look"] for x in lines] == [0, 1]       # an earlier pod's record is kept
    link = tmp_path / "code/results/part3/test_access_log.jsonl"
    assert link.is_symlink() and link.resolve() == (out / "test_access_log.jsonl").resolve()


def test_no_trained_run_means_no_evaluation_and_a_failure(tmp_path):
    r, out, status, calls = run_job(tmp_path, "oe:2 oe:3")
    assert r.returncode != 0
    assert not any(c.startswith(("scripts/part3/curves.py", "scripts/part3/evaluate.py")) for c in calls)
    assert "no run trained" in status


# The exploratory longer-training job (scripts/part3/pod/explore_job.sh).
EXPLORE = ROOT / "scripts/part3/pod/explore_job.sh"
FAKE_EXPLORE = FAKE.replace(
    "  scripts/part3/evaluate.py)",
    '  scripts/part3/curves.py)\n    case "$*" in *_seed4*) exit 1;; esac;;\n  scripts/part3/evaluate.py)')


def run_explore(tmp_path, runs, budget="1000"):
    code, out, logs = tmp_path / "code", tmp_path / "vol/explore_long/a", tmp_path / "logs"
    for p in (code, logs):
        p.mkdir(parents=True, exist_ok=True)
    py = tmp_path / "fakepy"
    py.write_text(FAKE_EXPLORE)
    py.chmod(0o755)
    env = {**os.environ, "OUT": str(out), "RUNS": runs, "PY": str(py), "LOGDIR": str(logs),
           "CALLS": str(tmp_path / "calls.txt")}
    if budget is not None:
        env["BUDGET"] = budget
    r = subprocess.run(["bash", str(EXPLORE)], cwd=code, env=env, capture_output=True, text=True)
    calls_f = tmp_path / "calls.txt"
    calls = calls_f.read_text().splitlines() if calls_f.exists() else []
    status_f = logs / "status.txt"
    return r, out, (status_f.read_text() if status_f.exists() else ""), calls


def test_explore_trains_to_the_budget_scores_curves_and_never_evaluates(tmp_path):
    r, out, status, calls = run_explore(tmp_path, "oe:1 oe:2 oe:3 oe:4 oe:5")
    assert r.returncode == 0, r.stderr
    trains = [c for c in calls if c.startswith("scripts/part3/train.py")]
    assert trains == [f"scripts/part3/train.py --method oe --purpose final --seed {s} --budget 1000 --out {out}"
                      for s in (1, 2, 3, 4, 5)]
    curves = [c for c in calls if c.startswith("scripts/part3/curves.py")]
    assert curves == [f"scripts/part3/curves.py --runs {out}/oe_seed{s}" for s in (1, 4, 5)]
    assert not any(c.startswith("scripts/part3/evaluate.py") for c in calls)      # test split never loaded
    assert not (tmp_path / "code/results").exists()
    for line in ("trained oe 1", "curves done oe 1", "run_failed oe 2", "run_failed oe 3",
                 "curves_failed oe 4", "curves done oe 5"):
        assert line in status
    assert (out / "status.txt").read_text() == status


def test_explore_refuses_without_a_whole_number_budget(tmp_path):
    for budget in (None, "", "1e9", "-5"):
        r, out, status, calls = run_explore(tmp_path / str(budget), "oe:1", budget=budget)
        assert r.returncode == 2 and calls == [] and "BUDGET" in status


def test_explore_fails_when_no_run_is_trained_and_scored(tmp_path):
    r, out, status, calls = run_explore(tmp_path, "oe:2 oe:4")
    assert r.returncode == 1
    assert "no run trained and scored" in status


# Addendum A (PREREG_part3.md Section 16): es_fixed gets the addendum's tuned settings in the final job,
# and the addendum job tunes first, writes the settings, then runs the final job.
def test_final_job_passes_the_addendum_settings_to_es_fixed_only(tmp_path, monkeypatch):
    monkeypatch.setenv("TUNED_EXTRA", "/vol/tuned_addendum.yaml")
    r, out, status, calls = run_job(tmp_path, "es_fixed:5000 es_dr:5001")
    assert r.returncode == 0, r.stderr
    trains = [c for c in calls if c.startswith("scripts/part3/train.py")]
    assert trains == [f"scripts/part3/train.py --method es_fixed --purpose final --seed 5000 --out {out} "
                      "--tuned-extra /vol/tuned_addendum.yaml",
                      f"scripts/part3/train.py --method es_dr --purpose final --seed 5001 --out {out}"]


def test_final_job_leaves_out_es_fixed_without_the_addendum_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("TUNED_EXTRA", raising=False)
    r, out, status, calls = run_job(tmp_path, "es_fixed:5000 es_dr:5001")
    assert r.returncode == 0, r.stderr
    assert "run_failed es_fixed 5000 (no TUNED_EXTRA)" in status
    assert [c for c in calls if c.startswith("scripts/part3/train.py")] == [
        f"scripts/part3/train.py --method es_dr --purpose final --seed 5001 --out {out}"]


ADDENDUM = ROOT / "scripts/part3/pod/addendum_job.sh"
FAKE_ADDENDUM = FAKE.replace("  scripts/part3/evaluate.py)", """  scripts/part3/tune.py)
    case "$*" in
      *--write-addendum*) [ -n "${TUNE_FAILS:-}${WRITE_FAILS:-}" ] && exit 1
                          while [ $# -gt 0 ]; do case "$1" in --to) t="$2"; shift;; esac; shift; done
                          echo "settings: {}" > "$t";;
      *) [ -n "${TUNE_FAILS:-}" ] && exit 1
         while [ $# -gt 0 ]; do case "$1" in --out) o="$2"; shift;; esac; shift; done
         mkdir -p "$o"; echo '{}' > "$o/tuning_es_fixed.json";;
    esac;;
  scripts/part3/evaluate.py)""")


def run_addendum(tmp_path, runs, tune_fails=False, write_fails=False):
    code, out, logs = tmp_path / "code", tmp_path / "vol/addendum/a", tmp_path / "logs"
    for p in (code / "scripts/part3/pod", logs):
        p.mkdir(parents=True, exist_ok=True)
    (code / "scripts/part3/pod/final_job.sh").write_text(JOB.read_text())
    py = tmp_path / "fakepy"
    py.write_text(FAKE_ADDENDUM)
    py.chmod(0o755)
    env = {**os.environ, "OUT": str(out), "RUNS": runs, "PY": str(py), "LOGDIR": str(logs),
           "CALLS": str(tmp_path / "calls.txt")}
    env.pop("TUNED_EXTRA", None)
    if tune_fails:
        env["TUNE_FAILS"] = "1"
    if write_fails:
        env["WRITE_FAILS"] = "1"
    r = subprocess.run(["bash", str(ADDENDUM)], cwd=code, env=env, capture_output=True, text=True)
    calls_f = tmp_path / "calls.txt"
    return r, out, (logs / "status.txt").read_text(), (calls_f.read_text().splitlines() if calls_f.exists() else [])


def test_addendum_tunes_writes_settings_then_runs_the_final_job(tmp_path):
    r, out, status, calls = run_addendum(tmp_path, "es_fixed:5000 es_dr:5000")
    assert r.returncode == 0, r.stderr
    assert calls[0] == f"scripts/part3/tune.py --methods es_fixed --out {out}/tuning"
    assert calls[1] == f"scripts/part3/tune.py --write-addendum {out}/tuning --to {out}/tuned_addendum.yaml"
    assert calls[2] == (f"scripts/part3/train.py --method es_fixed --purpose final --seed 5000 --out {out} "
                        f"--tuned-extra {out}/tuned_addendum.yaml")
    assert calls[3] == f"scripts/part3/train.py --method es_dr --purpose final --seed 5000 --out {out}"
    assert any(c.startswith("scripts/part3/evaluate.py") and "--final" in c for c in calls)
    for line in ("tuning done", "tuned settings written", "evaluation done"):
        assert line in status


def test_addendum_stops_before_training_if_tuning_fails(tmp_path):
    r, out, status, calls = run_addendum(tmp_path, "es_fixed:5000", tune_fails=True)
    assert r.returncode != 0
    assert calls == [f"scripts/part3/tune.py --methods es_fixed --out {out}/tuning"]
    assert "tuning failed" in status


def test_addendum_stops_before_training_if_the_settings_are_not_written(tmp_path):
    r, out, status, calls = run_addendum(tmp_path, "es_fixed:5000", write_fails=True)
    assert r.returncode != 0 and "FAILED: writing the tuned settings" in status
    assert not any(c.startswith("scripts/part3/train.py") for c in calls)
