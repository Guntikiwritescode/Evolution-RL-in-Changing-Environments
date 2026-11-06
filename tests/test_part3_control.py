"""The machine controller's stop rules, spend count, result copying and results check
(scripts/part3/pod/control.py). No network: copying is
tested against a local python http.server, the same server the pods run on port 8888."""
from __future__ import annotations

import functools
import http.server
import importlib.util
import json
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("control", ROOT / "scripts/part3/pod/control.py")
control = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(control)


def d(**kw):
    base = dict(status="ok", age_s=10.0, limit_s=100.0, spend=1.0, cap_stop=6.5, balance=5.0, floor=0.75,
                uploaded=True)
    base.update(kw)
    return control.decide(**base)


def test_money_stops_come_first():
    assert d(spend=6.5, status="all_done") == "stop_cap"
    assert d(spend=6.49) == "wait"
    assert d(balance=0.75, status="all_done") == "stop_balance"
    assert d(balance=0.76) == "wait"
    assert d(balance=None) == "wait"                     # an unread balance never stops nor passes the cap
    assert d(balance=None, spend=7.0) == "stop_cap"


def test_time_limit_then_job_state():
    assert d(age_s=100.1, status="all_done") == "stop_limit"
    assert d(age_s=100.0) == "wait"
    assert d(status="x\nFAILED: setup\n") == "stop_failed"
    assert d(status="trained oe 5000\nall_done\n") == "finish"
    assert d(status="waiting_for_code\n", uploaded=False) == "upload"
    assert d(status="waiting_for_code\n", uploaded=True) == "wait"


def test_own_spend_counts_pods_and_volumes_until_deleted():
    parts = {"a": {"price": 0.89, "pod_created_ts": 0.0, "pod_deleted_ts": 7200.0,
                   "volume_gb": 10, "volume_created_ts": 0.0, "volume_deleted_ts": 7300.0},
             "b": {"price": 0.89, "pod_created_ts": 3600.0, "pod_deleted_ts": None,
                   "volume_gb": 10, "volume_created_ts": 3600.0}}
    got = control.own_spend(parts, now=7200.0)
    vol = 10 * 0.07 / 730 * (7300 / 3600 + 1.0)
    assert got == pytest.approx(0.89 * 2 + 0.89 * 1 + vol)


def test_parse_listing_skips_parents_queries_and_unfinished_writes():
    page = ('<ul><li><a href="oe_seed5000/">oe_seed5000/</a></li><li><a href="status.txt">s</a></li>'
            '<li><a href="done.json.tmp">t</a></li><li><a href="../">up</a></li><li><a href="?C=M">q</a></li>'
            '<li><a href="a%20b.txt">x</a></li><li><a href="https://x.y/z">e</a></li></ul>')
    assert control.parse_listing(page) == ["oe_seed5000/", "status.txt", "a b.txt"]


def make_run(root: Path, budget: int, *, tenths=10, curves=True, reached=True):
    r = root / "oe_seed5000"
    (r / "snapshots").mkdir(parents=True)
    (r / "run.json").write_text(json.dumps({"budget_agent_steps": budget}))
    (r / "done.json").write_text(json.dumps({"reached_budget": reached, "agent_steps": budget + 3 if reached else 5}))
    (r / "policy.pkl").write_bytes(b"p")
    (r / "progress.jsonl").write_text("{}\n")
    for k in range(1, tenths + 1):
        (r / "snapshots" / f"{k}.pkl").write_bytes(b"s")
    if curves:
        (r / "curves.json").write_text(json.dumps({"quick": False, "points": [{"tenth": k} for k in range(1, 11)]}))
    return r


def test_check_part_passes_a_complete_run(tmp_path):
    make_run(tmp_path, 1000)
    assert control.check_part(tmp_path, "oe:5000", 1000) == []


@pytest.mark.parametrize("kw,expect", [(dict(tenths=9), "snapshots/10.pkl"), (dict(curves=False), "curves.json"),
                                       (dict(reached=False), "budget not reached")])
def test_check_part_names_what_is_missing(tmp_path, kw, expect):
    make_run(tmp_path, 1000, **kw)
    probs = control.check_part(tmp_path, "oe:5000", 1000)
    assert probs and any(expect in p for p in probs)


def test_check_part_catches_a_budget_other_than_the_plan(tmp_path):
    make_run(tmp_path, 1000)
    assert any("budget differs" in p for p in control.check_part(tmp_path, "oe:5000", 2000))


def test_check_part_catches_a_missing_run(tmp_path):
    assert any("oe_seed5001" in p for p in control.check_part(tmp_path, "oe:5001", 1000))


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A pod's /w/serve layout (results/ and logs/) behind python's http.server."""
    serve = tmp_path / "serve"
    (serve / "results/oe_seed5000/snapshots").mkdir(parents=True)
    (serve / "logs").mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(serve))
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(control, "proxy", lambda pod, port, path: f"http://127.0.0.1:{srv.server_port}/{path}")
    yield serve
    srv.shutdown()


def test_sync_copies_new_and_changed_files_and_skips_unchanged(server, tmp_path):
    res = server / "results/oe_seed5000"
    (res / "progress.jsonl").write_text("a\n")
    (res / "snapshots/1.pkl").write_bytes(b"one")
    (res / "done.json.tmp").write_text("half")
    (server / "logs/status.txt").write_text("boot\n")
    local = tmp_path / "local"
    assert control.sync("pod", local) == 3
    assert (local / "oe_seed5000/snapshots/1.pkl").read_bytes() == b"one"
    assert (local / "_logs/status.txt").read_text() == "boot\n"
    assert not (local / "oe_seed5000/done.json.tmp").exists()
    assert control.sync("pod", local) == 0                      # nothing changed
    (res / "progress.jsonl").write_text("a\nb\n")               # grew
    assert control.sync("pod", local) == 1
    assert (local / "oe_seed5000/progress.jsonl").read_text() == "a\nb\n"
    assert control.sync("pod", local, force=True) == 3          # the last copy fetches everything


def test_sync_reports_a_failed_listing(server, tmp_path, monkeypatch):
    monkeypatch.setattr(control, "proxy", lambda pod, port, path: "http://127.0.0.1:9/" + path)
    assert control.sync("pod", tmp_path / "local") == -1


def test_state_lock_is_exclusive(tmp_path, monkeypatch):
    """A create and a watch pass take turns on the state file (fcntl lock on STAGE_DIR/state.lock)."""
    import fcntl
    monkeypatch.setenv("STAGE_DIR", str(tmp_path))
    with control.locked():
        with open(tmp_path / "state.lock", "w") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with open(tmp_path / "state.lock", "w") as other:          # released afterwards
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _evaluated(run: Path, full=True):
    envs = [{"adaptation": {}} for _ in range(30 if full else 29)]
    (run / "eval_test.json").write_text(json.dumps({"split": "test", "envs": envs}))


def test_check_part_for_an_addendum_needs_evaluation_tuning_and_the_access_log(tmp_path):
    r = make_run(tmp_path, 1000)
    probs = control.check_part(tmp_path, "oe:5000", 1000, "addendum")
    for need in ("eval_test.json", "tuning/tuning_es_fixed.json", "tuned_addendum.yaml", "test_access_log.jsonl"):
        assert any(need in p for p in probs), need
    _evaluated(r)
    (tmp_path / "tuning").mkdir()
    (tmp_path / "tuning/tuning_es_fixed.json").write_text("{}")
    (tmp_path / "tuned_addendum.yaml").write_text("settings: {}\n")
    (tmp_path / "test_access_log.jsonl").write_text('{"look": 1}\n')
    assert control.check_part(tmp_path, "oe:5000", 1000, "addendum") == []
    _evaluated(r, full=False)
    assert any("full test evaluation" in p for p in control.check_part(tmp_path, "oe:5000", 1000, "addendum"))
    assert control.check_part(tmp_path, "oe:5000", 1000) == []     # explore_long parts are not evaluated


def test_volume_delete_needs_the_pod_deleted_first(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGE_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(control, "api", lambda method, path, obj=None: calls.append((method, path)) or (204 if method == "DELETE" else 404, None))
    (tmp_path / "state.json").write_text(json.dumps({"parts": {"a": {"pod_id": "p1", "pod_deleted_ts": None, "volume_id": "v1"}}}))
    with pytest.raises(SystemExit, match="pod is not deleted"):
        control.cmd_volume_delete("a")
    assert calls == []
    (tmp_path / "state.json").write_text(json.dumps({"parts": {"a": {"pod_id": "p1", "pod_deleted_ts": 1.0, "volume_id": "v1"}}}))
    assert control.cmd_volume_delete("a") is True
    assert calls[0] == ("DELETE", "/networkvolumes/v1")
    assert json.loads((tmp_path / "state.json").read_text())["parts"]["a"]["volume_deleted_ts"] is not None


def _stage(tmp_path, monkeypatch, volume):
    monkeypatch.setenv("STAGE_DIR", str(tmp_path))
    plan = {"stage": "s", "cap": 2.0, "cap_stop": 1.8, "balance_floor": 0.75, "price_max": 0.89,
            "gpu": "G", "volume_gb": 10, "volume": volume,
            "parts": {"a": {"job": "addendum", "runs": "oe:5000", "budget": 1000, "limit_s": 60}}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "code.tgz").write_bytes(b"code")
    return plan


def test_create_without_a_network_volume(tmp_path, monkeypatch):
    _stage(tmp_path, monkeypatch, "none")
    calls = []

    def api(method, path, obj=None):
        calls.append((method, path, obj))
        return (200, {"id": "pod1", "costPerHr": 0.89}) if path == "/pods" else (500, None)
    monkeypatch.setattr(control, "api", api)
    assert control.create_in("a", "EU-RO-1") is True
    assert [c[1] for c in calls] == ["/pods"]                      # no network volume is created
    body = calls[0][2]
    assert "networkVolumeId" not in body and body["volumeInGb"] == 0 and body["dataCenterIds"] == ["EU-RO-1"]
    assert body["env"]["VOLUME_KIND"] == "none"
    st = json.loads((tmp_path / "state.json").read_text())["parts"]["a"]
    assert st["pod_id"] == "pod1" and "volume_id" not in st


def test_create_with_a_network_volume_is_unchanged(tmp_path, monkeypatch):
    plan = _stage(tmp_path, monkeypatch, "network")
    body = control.pod_body(plan, "a", {"dc": "EU-RO-1", "volume_id": "v1"}, {"X": "1"})
    assert body["networkVolumeId"] == "v1" and body["volumeMountPath"] == "/workspace" and "volumeInGb" not in body


def test_finish_without_a_volume_retries_the_last_copy_and_deletes_no_volume(tmp_path, monkeypatch):
    plan = _stage(tmp_path, monkeypatch, "none")
    st = {"parts": {"a": {"pod_id": "pod1", "dc": "EU-RO-1", "volume": "none", "price": 0.89,
                          "pod_created_ts": 1.0, "pod_deleted_ts": None}}}
    (tmp_path / "state.json").write_text(json.dumps(st))
    copies, deleted = [], []
    monkeypatch.setattr(control, "LAST_COPY_WAIT_S", 0)
    monkeypatch.setattr(control, "sync", lambda pod, local, force=False: copies.append(pod) or (-1 if len(copies) < 3 else 5))
    monkeypatch.setattr(control, "api", lambda method, path, obj=None: deleted.append((method, path)) or ((204, None) if method == "DELETE" else (404, None)))
    control.finish(st, plan, "a", "finished")
    assert len(copies) == 3                                         # retried until a copy worked
    assert ("DELETE", "/pods/pod1") in deleted and not any("networkvolumes" in p for _, p in deleted)
