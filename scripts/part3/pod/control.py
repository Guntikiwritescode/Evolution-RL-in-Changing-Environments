"""Part 3 machine control: rent RunPod GPU machines for one stage, watch them, delete them.

Each stage uses the following controls:
- one network volume and one GPU pod per part, in a data center with stock, secure cloud only;
- the pod runs scripts/part3/pod/start_job.sh, accepts one upload (the code package, checked
  against its sha256 by recv.py) and serves its logs and results read-only on port 8888;
- results are copied off while the pod runs; every pod is deleted when its part finishes, at
  its time limit, when this stage's own spend reaches cap_stop, or when the account balance falls
  to balance_floor. Pod-side deletion failed in the deployment test, so `watch` must keep running
  while any pod is up to enforce these limits.

Stage folder (env STAGE_DIR, outside the repo): plan.json, code.tgz, state.json (written here),
results/<part>/ (copies). The key comes only from env RUNPOD_API_KEY; it is never written to disk
by this script.

Storage (plan "volume"): "network" (default) puts results on a network volume that outlives the pod;
"none" keeps them on the pod's own disk at /workspace, lost when the pod is deleted.
With "none", results are copied off every pass, and finish() retries the last copy
before it deletes the pod.

plan.json example:
  {"stage": "explore_long", "cap": 7.0, "cap_stop": 6.5, "balance_floor": 0.75,
   "price_max": 0.89, "gpu": "NVIDIA GeForce RTX 4090", "volume_gb": 10,
   "parts": {"a": {"job": "explore_long", "runs": "oe:5000", "budget": 1000000000, "limit_s": 8400}}}

Commands (python3 -I scripts/part3/pod/control.py ...):
  package COMMIT      build STAGE_DIR/code.tgz from GitHub at COMMIT and check it like the pod opens it
  balance             account balance and spend rate (GraphQL "myself")
  stock               data centers listing the GPU (GraphQL "dataCenters")
  create PART DC,DC   network volume in the first data center that can take the pod, then the pod
  watch MINUTES       every 90 s: upload code, copy results, apply the stop rules
  delete PART         delete the part's pod, then its volume if its results check passes
  volume-delete PART  delete a part's kept volume, only after its pod is deleted (results copied first)
  spend               this stage's own spend from state.json
"""
from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import html
import json
import os
import re
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
REPO_URL = "https://github.com/Guntikiwritescode/evolving-ai-v2"
POD_DIR = Path(__file__).resolve().parent
UA = "evolving-ai-pod-control/2.0"
VOLUME_PRICE_GB_MONTH = 0.07     # RunPod network volumes, $/GB/month
HOURS_PER_MONTH = 730.0
POLL_S = 90
MAX_UPLOAD_TRIES = 5
SPARSE = ["/*", "!/results/", "!/calibration/", "!/figures/", "!/paper/", "!/reference/"]


# ---------------------------------------------------------------- pure logic (tested offline)

def own_spend(parts: dict, now: float) -> float:
    """This stage's own spend: each pod's price x its running time, plus each volume's storage."""
    total = 0.0
    for p in parts.values():
        if p.get("pod_created_ts") is not None:
            end = p["pod_deleted_ts"] if p.get("pod_deleted_ts") is not None else now
            total += float(p["price"]) * max(0.0, end - p["pod_created_ts"]) / 3600.0
        if p.get("volume_created_ts") is not None:
            end = p["volume_deleted_ts"] if p.get("volume_deleted_ts") is not None else now
            hours = max(0.0, end - p["volume_created_ts"]) / 3600.0
            total += float(p["volume_gb"]) * VOLUME_PRICE_GB_MONTH * hours / HOURS_PER_MONTH
    return total


def decide(status: str, age_s: float, limit_s: float, spend: float, cap_stop: float,
           balance: float | None, floor: float, uploaded: bool) -> str:
    """What to do with one running pod. Order: money first, then the time limit, then the job."""
    if spend >= cap_stop:
        return "stop_cap"
    if balance is not None and balance <= floor:
        return "stop_balance"
    if age_s > limit_s:
        return "stop_limit"
    if "FAILED" in status:
        return "stop_failed"
    if "all_done" in status:
        return "finish"
    if "waiting_for_code" in status and not uploaded:
        return "upload"
    return "wait"


_HREF = re.compile(r'href="([^"]+)"')


def parse_listing(page: str) -> list[str]:
    """Entry names from a python http.server directory page (directories end with '/')."""
    out = []
    for h in _HREF.findall(page):
        h = html.unescape(h)
        if h.startswith(("?", "/", "#", "..")) or "://" in h:
            continue
        name = urllib.parse.unquote(h)
        if name.endswith(".tmp"):          # a write in progress (runio.write_atomic); copied once renamed
            continue
        out.append(name)
    return out


def check_run(run_dir: Path, budget: int, evaluated: bool = False) -> list[str]:
    """Problems with one copied run folder (empty list = complete). evaluated: the job also ran the
    final test evaluation (final, addendum), so eval_test.json must hold all 30 test environments."""
    probs = []
    if evaluated:
        f = run_dir / "eval_test.json"
        if not f.is_file():
            probs.append("missing eval_test.json")
        else:
            e = json.loads(f.read_text())
            if e.get("split") != "test" or len(e.get("envs", [])) != 30 \
                    or any("adaptation" not in r for r in e.get("envs", [])):
                probs.append("eval_test.json is not a full test evaluation with adaptation")
    for f in ("run.json", "done.json", "policy.pkl", "progress.jsonl", "curves.json"):
        if not (run_dir / f).is_file():
            probs.append(f"missing {f}")
    for k in range(1, 11):
        if not (run_dir / "snapshots" / f"{k}.pkl").is_file():
            probs.append(f"missing snapshots/{k}.pkl")
    if (run_dir / "failed.json").exists():
        probs.append("failed.json present")
    if (run_dir / "run.json").is_file() and json.loads((run_dir / "run.json").read_text()).get("budget_agent_steps") != budget:
        probs.append("run.json budget differs from the plan")
    if (run_dir / "done.json").is_file():
        d = json.loads((run_dir / "done.json").read_text())
        if not d.get("reached_budget") or d.get("agent_steps", 0) < budget:
            probs.append("done.json: budget not reached")
    if (run_dir / "curves.json").is_file():
        c = json.loads((run_dir / "curves.json").read_text())
        if [p.get("tenth") for p in c.get("points", [])] != list(range(1, 11)) or c.get("quick"):
            probs.append("curves.json does not have the 10 full points")
    return probs


def check_part(local: Path, runs: str, budget: int, job: str = "explore_long") -> list[str]:
    """Problems with a copied part. final and addendum parts were evaluated on the test split; an
    addendum part also holds its tuning summary, its tuned settings and the test-access record."""
    probs = []
    evaluated = job in ("final", "addendum")
    for r in runs.split():
        m, s = r.split(":")
        probs += [f"{m}_seed{s}: {x}" for x in check_run(local / f"{m}_seed{s}", budget, evaluated)]
    if job == "addendum":
        for f in ("tuning/tuning_es_fixed.json", "tuned_addendum.yaml"):
            if not (local / f).is_file():
                probs.append(f"missing {f}")
    if evaluated:
        f = local / "test_access_log.jsonl"
        if not f.is_file() or not f.read_text().strip():
            probs.append("missing test_access_log.jsonl (every look at the test split is recorded)")
    return probs


# ---------------------------------------------------------------- I/O

def stage_dir() -> Path:
    return Path(os.environ["STAGE_DIR"])


def load_plan() -> dict:
    return json.loads((stage_dir() / "plan.json").read_text())


@contextlib.contextmanager
def locked():
    """Exclusive lock on the stage's state: a watch pass and a create or delete never interleave, so
    neither can overwrite the other's record of a pod (added 4 Nov 2026, while a watcher was running)."""
    with open(stage_dir() / "state.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_state() -> dict:
    f = stage_dir() / "state.json"
    return json.loads(f.read_text()) if f.exists() else {"parts": {}}


def save_state(st: dict):
    f = stage_dir() / "state.json"
    tmp = f.with_name("state.json.tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True))
    os.replace(tmp, f)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def key() -> str:
    k = os.environ.get("RUNPOD_API_KEY", "")
    if not k:
        sys.exit("RUNPOD_API_KEY is not set")
    return k


def http(method: str, url: str, body: bytes | None = None, headers: dict | None = None, timeout: int = 60):
    """Returns (status code, bytes). Network errors give code 0 and are never counted as success."""
    req = urllib.request.Request(url, data=body, method=method, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:            # noqa: BLE001 - reported to the caller as code 0
        return 0, str(e).encode()


def api(method: str, path: str, obj=None):
    body = json.dumps(obj).encode() if obj is not None else None
    code, data = http(method, REST + path, body, {"Authorization": f"Bearer {key()}",
                                                  "Content-Type": "application/json"})
    try:
        return code, json.loads(data) if data else None
    except ValueError:
        return code, data.decode(errors="replace")


def gql(query: str):
    code, data = http("POST", GRAPHQL, json.dumps({"query": query}).encode(),
                      {"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    try:
        return code, json.loads(data)
    except ValueError:
        return code, data.decode(errors="replace")


def balance() -> tuple[float | None, float | None]:
    code, r = gql("query { myself { clientBalance currentSpendPerHr } }")
    try:
        m = r["data"]["myself"]
        return float(m["clientBalance"]), float(m["currentSpendPerHr"])
    except Exception:                 # noqa: BLE001
        print(f"{now_utc()}  balance read failed: {code} {str(r)[:200]}", flush=True)
        return None, None


def proxy(pod_id: str, port: int, path: str) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net/{path.lstrip('/')}"


# ---------------------------------------------------------------- commands

def cmd_package(commit: str):
    d = stage_dir()
    with tempfile.TemporaryDirectory() as t:
        code = Path(t) / "code"
        run = lambda *a, **k: subprocess.run(a, check=True, capture_output=True, text=True, **k)  # noqa: E731
        run("git", "clone", "-q", "--filter=blob:none", "--no-checkout", REPO_URL, str(code))
        run("git", "-C", str(code), "sparse-checkout", "set", "--no-cone", *SPARSE)
        run("git", "-C", str(code), "checkout", "-q", commit)
        full = run("git", "-C", str(code), "rev-parse", "HEAD").stdout.strip()
        out = d / "code.tgz"
        run("tar", "-czf", str(out), "-C", str(code), ".")          # files at the top
        # check it the way the pod opens it: tar -xzf code.tgz -C /w/code
        chk = Path(t) / "w/code"
        chk.mkdir(parents=True)
        with tarfile.open(out) as tf:
            tf.extractall(chk, filter="tar")
        need = ["requirements_part3.txt", "configs/part3/tuned.yaml", "scripts/part3/pod/start_job.sh",
                "scripts/part3/pod/explore_job.sh", "scripts/part3/pod/final_job.sh",
                "scripts/part3/pod/addendum_job.sh", "scripts/part3/train.py", "scripts/part3/tune.py",
                "scripts/part3/curves.py", "scripts/part3/evaluate.py"]
        missing = [f for f in need if not (chk / f).is_file()]
        head = run("git", "-C", str(chk), "rev-parse", "HEAD").stdout.strip()
        dirty = run("git", "-C", str(chk), "status", "--porcelain", "--", ".", ":(exclude)results").stdout
        if missing or head != full or dirty.strip():
            sys.exit(f"package check failed: missing {missing}, head {head} vs {full}, dirty {dirty!r}")
    data = out.read_bytes()
    print(json.dumps({"commit": full, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "checked": need}, indent=1))


def cmd_balance():
    b, rate = balance()
    print(json.dumps({"at": now_utc(), "balance": b, "spend_per_hour": rate}))


def cmd_stock():
    """Secure-cloud stock and price of the plan's GPU. RunPod's per-data-center stock field
    (dataCenters.gpuAvailability) came back empty for every GPU on 4 Nov 2026, so `create` tries
    data centers in turn instead."""
    plan = load_plan()
    q = ('query { gpuTypes(input: {id: "%s"}) { id lowestPrice(input: {gpuCount: 1, secureCloud: true}) '
         '{ stockStatus uninterruptablePrice } } }' % plan["gpu"])
    code, r = gql(q)
    try:
        lp = r["data"]["gpuTypes"][0]["lowestPrice"]
    except Exception:                 # noqa: BLE001
        sys.exit(f"stock query failed: {code} {str(r)[:500]}")
    print(json.dumps({"at": now_utc(), "gpu": plan["gpu"], "secure_stock": lp["stockStatus"],
                      "price_per_hour": lp["uninterruptablePrice"]}))


def cmd_create(part: str, dcs: str):
    """Try each data center in the comma-separated list: volume there, then the pod. If the pod
    cannot be created there, that volume is deleted (and the deletion confirmed) before the next."""
    for dc in dcs.split(","):
        with locked():
            ok = create_in(part, dc)
        if ok:
            return
    sys.exit(f"{part}: no listed data center could take the pod")


def pod_body(plan: dict, part: str, p: dict, env: dict) -> dict:
    """The REST body for a part's pod: with a network volume, or (volume "none") with none."""
    body = {"name": f"evolving-ai-p3-{plan['stage']}-{part}", "imageName": "ubuntu:24.04", "cloudType": "SECURE",
            "computeType": "GPU", "gpuTypeIds": [plan["gpu"]], "gpuCount": 1, "containerDiskInGb": 40,
            "dataCenterIds": [p["dc"]], "ports": ["8888/http", "8889/http"], "env": env,
            "dockerStartCmd": ["bash", "-c", 'echo "$START_B64" | base64 -d > /start_job.sh && bash /start_job.sh']}
    if plan.get("volume", "network") == "none":
        body["volumeInGb"] = 0
    else:
        body.update({"networkVolumeId": p["volume_id"], "volumeMountPath": "/workspace"})
    return body


def create_in(part: str, dc: str) -> bool:
    plan, st = load_plan(), load_state()
    spec = plan["parts"][part]
    p = st["parts"].get(part, {})
    no_volume = plan.get("volume", "network") == "none"
    if p.get("pod_id") and p.get("pod_deleted_ts") is None:
        sys.exit(f"{part} already has pod {p['pod_id']}")
    spent = own_spend({**st["parts"], **{f"earlier_{i}": q for i, q in enumerate(st.get("earlier_volumes", []))}}, time.time())
    if spent >= plan["cap_stop"]:
        sys.exit(f"stage spend ${spent:.2f} is at the stop level; not creating")
    if no_volume:
        p = {"dc": dc, "volume": "none"}
        st["parts"][part] = p
    elif p.get("volume_id") and p.get("volume_deleted_ts") is None and p.get("dc") != dc:
        sys.exit(f"{part} has volume {p['volume_id']} in {p.get('dc')}, not {dc}; delete it first")
    if not no_volume and (not p.get("volume_id") or p.get("volume_deleted_ts") is not None):
        code, v = api("POST", "/networkvolumes", {"name": f"evolving-ai-p3-{plan['stage']}-{part}",
                                                   "size": plan["volume_gb"], "dataCenterId": dc})
        if code not in (200, 201) or not isinstance(v, dict) or "id" not in v:
            sys.exit(f"volume for {part} in {dc} failed: {code} {v}")
        old = st.setdefault("earlier_volumes", [])
        if p.get("volume_id"):
            old.append(p)               # kept for the spend count
        p = {"volume_id": v["id"], "dc": dc, "volume_gb": plan["volume_gb"], "volume_created_ts": time.time(),
             "volume_created_at": now_utc()}
        st["parts"][part] = p
        save_state(st)
        print(f"{now_utc()}  {part} volume {v['id']} in {dc}", flush=True)
    code_tgz = stage_dir() / "code.tgz"
    token = secrets.token_hex(16)
    env = {"TOKEN": token, "CODE_SHA256": hashlib.sha256(code_tgz.read_bytes()).hexdigest(),
           "RECV_B64": base64.b64encode((POD_DIR / "recv.py").read_bytes()).decode(),
           "START_B64": base64.b64encode((POD_DIR / "start_job.sh").read_bytes()).decode(),
           "PART": part, "JOB": spec["job"], "METHODS": spec.get("methods", ""), "RUNS": spec.get("runs", ""),
           "BUDGET": str(spec.get("budget", "")), "LIMIT_S": str(spec["limit_s"]),
           "VOLUME_KIND": "none" if no_volume else "network"}
    code, r = api("POST", "/pods", pod_body(plan, part, p, env))
    if code not in (200, 201) or not isinstance(r, dict) or "id" not in r:
        print(f"{now_utc()}  pod for {part} in {dc} failed: {code} {str(r)[:300]}", flush=True)
        if no_volume:
            return False
        if not delete_volume(st, part):
            sys.exit(f"{part}: volume {p['volume_id']} could not be deleted; stopping")
        return False
    price = float(r.get("costPerHr") or r.get("adjustedCostPerHr") or plan["price_max"])
    p.update({"pod_id": r["id"], "token": token, "sha256": env["CODE_SHA256"], "price": price,
              "pod_created_ts": time.time(), "pod_created_at": now_utc(), "uploaded": False, "upload_tries": 0,
              "pod_deleted_ts": None, "end_reason": None})
    st["parts"][part] = p
    save_state(st)
    print(f"{now_utc()}  {part} pod {r['id']} at ${price:.3f}/h", flush=True)
    if price > plan["price_max"] + 1e-9:
        print(f"{part}: price ${price} is above the configured ${plan['price_max']}; deleting", flush=True)
        delete_pod(st, part, "price above the configured maximum")
        sys.exit(1)
    return True


def delete_pod(st: dict, part: str, why: str) -> bool:
    p = st["parts"][part]
    if p.get("pod_deleted_ts"):
        return True
    code, r = api("DELETE", f"/pods/{p['pod_id']}")
    gone, _ = api("GET", f"/pods/{p['pod_id']}")
    if code in (200, 204) or gone == 404:
        p.update({"pod_deleted_ts": time.time(), "pod_deleted_at": now_utc(), "end_reason": why,
                  "pod_delete_codes": [code, gone]})
        save_state(st)
        print(f"{now_utc()}  {part} pod {p['pod_id']} deleted ({why}; DELETE {code}, then GET {gone})", flush=True)
        return True
    print(f"{now_utc()}  {part} pod {p['pod_id']} DELETE FAILED: {code} {str(r)[:200]}; GET {gone}", flush=True)
    return False


def delete_volume(st: dict, part: str) -> bool:
    p = st["parts"][part]
    if p.get("volume_deleted_ts"):
        return True
    code, r = api("DELETE", f"/networkvolumes/{p['volume_id']}")
    gone, _ = api("GET", f"/networkvolumes/{p['volume_id']}")
    if code in (200, 204) or gone == 404:
        p.update({"volume_deleted_ts": time.time(), "volume_deleted_at": now_utc(),
                  "volume_delete_codes": [code, gone]})
        save_state(st)
        print(f"{now_utc()}  {part} volume {p['volume_id']} deleted (DELETE {code}, then GET {gone})", flush=True)
        return True
    print(f"{now_utc()}  {part} volume {p['volume_id']} DELETE FAILED: {code} {str(r)[:200]}", flush=True)
    return False


def sync(pod_id: str, local: Path, force: bool = False) -> int:
    """Copy every result file (and the logs) off the pod; a file is fetched when it is new, its size
    changed, or force is set. Returns the number of files fetched; -1 if the listing failed."""
    fetched = 0

    def walk(remote: str, dest: Path):
        nonlocal fetched
        code, page = http("GET", proxy(pod_id, 8888, remote), timeout=60)
        if code != 200:
            raise OSError(f"listing {remote}: {code}")
        for name in parse_listing(page.decode(errors="replace")):
            if name.endswith("/"):
                walk(remote + urllib.parse.quote(name), dest / name.rstrip("/"))
                continue
            url = proxy(pod_id, 8888, remote + urllib.parse.quote(name))
            target = dest / name
            if not force and target.exists():
                req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        size = int(r.headers.get("Content-Length", -1))
                except Exception:     # noqa: BLE001
                    size = -1
                if size == target.stat().st_size:
                    continue
            c, data = http("GET", url, timeout=300)
            if c != 200:
                raise OSError(f"fetch {remote}{name}: {c}")
            dest.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            fetched += 1

    try:
        walk("results/", local)
        walk("logs/", local / "_logs")
    except OSError as e:
        print(f"{now_utc()}  sync {pod_id}: {e}", flush=True)
        return -1
    return fetched


def status_text(pod_id: str) -> str:
    code, data = http("GET", proxy(pod_id, 8888, "logs/status.txt"), timeout=30)
    return data.decode(errors="replace") if code == 200 else ""


def upload(p: dict) -> bool:
    data = (stage_dir() / "code.tgz").read_bytes()
    if hashlib.sha256(data).hexdigest() != p["sha256"]:
        print("code.tgz changed since the pod was created; not uploading", flush=True)
        return False
    code, r = http("PUT", proxy(p["pod_id"], 8889, f"{p['token']}/code.tgz"), data,
                   {"Content-Type": "application/octet-stream"}, timeout=300)
    return code == 200 and b"received" in r


LAST_COPY_TRIES = 3
LAST_COPY_WAIT_S = 30


def finish(st: dict, plan: dict, part: str, why: str):
    """Last full copy, delete the pod, then delete the volume only if the copy checks out. Without a
    network volume the pod's disk is the only other copy, so a failed last copy is retried first."""
    p, spec = st["parts"][part], plan["parts"][part]
    local = stage_dir() / "results" / part
    tries = 1 if p.get("volume_id") else LAST_COPY_TRIES
    for t in range(tries):
        n = sync(p["pod_id"], local, force=True)
        if n >= 0:
            break
        if t + 1 < tries:
            time.sleep(LAST_COPY_WAIT_S)
    if not delete_pod(st, part, why):
        return
    probs = (check_part(local, spec.get("runs", ""), int(spec.get("budget", 0)), spec["job"]) if n >= 0
             else ["last copy failed"])
    p["check"] = probs
    save_state(st)
    if not p.get("volume_id"):
        print(f"{now_utc()}  {part}: no network volume (results only in {local}); problems: {probs[:5]}", flush=True)
    elif why == "finished" and not probs:
        delete_volume(st, part)
    else:
        print(f"{now_utc()}  {part}: volume {p['volume_id']} KEPT ({why}; problems: {probs[:5]})", flush=True)


def watch_pass(plan: dict) -> bool:
    """One pass over every part. Returns True while any pod is still up."""
    st = load_state()
    bal, rate = balance()
    t = time.time()
    spend = own_spend({**st["parts"], **{f"earlier_{i}": p for i, p in enumerate(st.get("earlier_volumes", []))}}, t)
    print(f"{now_utc()}  stage spend ${spend:.2f} (cap ${plan['cap']:.2f}, stop at ${plan['cap_stop']:.2f}); "
          f"balance {bal}; account spend/h {rate}", flush=True)
    up = False
    for part, p in sorted(st["parts"].items()):
        if not p.get("pod_id") or p.get("pod_deleted_ts"):
            continue
        g, info = api("GET", f"/pods/{p['pod_id']}")
        if g == 404:
            p.update({"pod_deleted_ts": t, "pod_deleted_at": now_utc(), "end_reason": "gone (404)"})
            save_state(st)
            print(f"{now_utc()}  {part} pod is gone (404); volume kept", flush=True)
            continue
        status = status_text(p["pod_id"])
        last = status.strip().splitlines()[-1] if status.strip() else "(no status yet)"
        action = decide(status, t - p["pod_created_ts"], plan["parts"][part]["limit_s"], spend,
                        plan["cap_stop"], bal, plan["balance_floor"], p.get("uploaded", False))
        if action == "upload":
            ok = upload(p)
            p["upload_tries"] = p.get("upload_tries", 0) + 1
            p["uploaded"] = ok
            save_state(st)
            print(f"{now_utc()}  {part} upload {'ok' if ok else 'FAILED'} (try {p['upload_tries']})", flush=True)
            if not ok and p["upload_tries"] >= MAX_UPLOAD_TRIES:
                finish(st, plan, part, "upload failed")
                continue
        elif action == "finish":
            finish(st, plan, part, "finished")
            continue
        elif action.startswith("stop_"):
            finish(st, plan, part, action)
            continue
        n = sync(p["pod_id"], stage_dir() / "results" / part) if p.get("uploaded") else 0
        print(f"{now_utc()}  {part} up {(t - p['pod_created_ts']) / 60:5.1f} min | {last} | copied {n}", flush=True)
        up = True
    return up


def cmd_watch(minutes: float):
    plan = load_plan()
    end = time.time() + minutes * 60
    while True:
        with locked():
            up = watch_pass(plan)
        if not up:
            print(f"{now_utc()}  no pod of this stage is up", flush=True)
            return
        if time.time() + POLL_S > end:
            return
        time.sleep(POLL_S)


def cmd_delete(part: str):
    with locked():
        plan, st = load_plan(), load_state()
        finish(st, plan, part, "deleted by hand")


def cmd_volume_delete(part: str) -> bool:
    """A volume kept by finish() (a failed run, a stop): deleted by hand once the copy has been checked."""
    with locked():
        st = load_state()
        p = st["parts"][part]
        if not p.get("pod_deleted_ts"):
            sys.exit(f"{part}: its pod is not deleted; delete the pod first")
        return delete_volume(st, part)


def cmd_spend():
    st = load_state()
    t = time.time()
    every = {**st["parts"], **{f"earlier_{i}": p for i, p in enumerate(st.get("earlier_volumes", []))}}
    out = {"at": now_utc(), "total": round(own_spend(every, t), 4), "parts": {}}
    for part, p in st["parts"].items():
        out["parts"][part] = {k: p.get(k) for k in ("dc", "price", "pod_created_at", "pod_deleted_at",
                                                    "volume_deleted_at", "end_reason", "check")}
        out["parts"][part]["cost"] = round(own_spend({part: p}, t), 4)
    print(json.dumps(out, indent=1))


def main():
    a = sys.argv[1:]
    cmds = {"package": lambda: cmd_package(a[1]), "balance": cmd_balance, "stock": cmd_stock,
            "create": lambda: cmd_create(a[1], a[2]), "watch": lambda: cmd_watch(float(a[1])),
            "delete": lambda: cmd_delete(a[1]), "volume-delete": lambda: cmd_volume_delete(a[1]),
            "spend": cmd_spend}
    if not a or a[0] not in cmds:
        sys.exit(__doc__)
    cmds[a[0]]()


if __name__ == "__main__":
    main()
