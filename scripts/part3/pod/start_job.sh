#!/bin/bash
# Fixed steps for an evolving-ai Part 3 pod (tuning or final runs).
# Everything this pod will do is written here at creation time. It accepts one upload
# (the code package, checked against CODE_SHA256), sets up Python and JAX, runs its job,
# keeps every result on the network volume (/workspace), and serves logs and results
# read-only on port 8888. It accepts no commands.
# Set at creation: TOKEN, CODE_SHA256, RECV_B64, PART (a name), JOB (selftest | tune | final | explore_long | addendum),
# METHODS (tune: space-separated methods), RUNS (final, explore_long, addendum: space-separated method:seed
# pairs; see final_job.sh, explore_job.sh, addendum_job.sh), BUDGET (explore_long: training agent-steps),
# LIMIT_S (seconds after boot when the pod deletes itself; 0 = no limit),
# VOLUME_KIND (network | none; none: /workspace is the pod's own disk, see control.py).
set -uo pipefail
VOL=/workspace
OUT="$VOL/$JOB/$PART"
mkdir -p /w/logs /w/serve "$OUT"
ln -sfn /w/logs /w/serve/logs
ln -sfn "$OUT" /w/serve/results
log() { echo "$(date -u +%FT%TZ) $*" >> /w/logs/status.txt; cp /w/logs/status.txt "$OUT/status.txt" 2>/dev/null; }
stop() { log "FAILED: $*"; sleep infinity; }
# Pod-side time limit: try to delete this pod through RunPod's API with the key RunPod gives every pod.
# The selftest job showed on 3 Nov 2026 that this does NOT work (empty reply, pod not deleted), so it
# is not a cost-safety: the session's own polling (control.py watch; rp2.py until 3 Nov) stops every pod (MISTAKES R16, R17).
# The attempt is kept only because it is harmless; it does not stop the job.
self_delete() {
  curl -s -m 30 -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
       -H "Authorization: Bearer $RUNPOD_API_KEY" > /w/logs/self_delete.txt 2>&1
  echo "exit $?" >> /w/logs/self_delete.txt
}
if [ "${LIMIT_S:-0}" -gt 0 ]; then
  ( sleep "$LIMIT_S"; log "time limit reached: deleting this pod"; cp /w/logs/* "$OUT/" 2>/dev/null; self_delete ) &
fi
log boot
if [ "${VOLUME_KIND:-network}" = "none" ]; then
  log "no network volume: results stay on this pod's disk until copied off"
else
  grep -qs " $VOL " /proc/mounts || stop no_network_volume
fi
export DEBIAN_FRONTEND=noninteractive
{ apt-get update -qq && apt-get install -y -qq python3 git curl ca-certificates; } > /w/logs/apt.log 2>&1 || stop apt
log apt_ok
(cd /w/serve && exec python3 -m http.server 8888 > /w/logs/http.log 2>&1) &
if [ "$JOB" = "selftest" ]; then
  log "selftest_waiting key_set=$([ -n "${RUNPOD_API_KEY:-}" ] && echo yes || echo no) pod_id_set=$([ -n "${RUNPOD_POD_ID:-}" ] && echo yes || echo no)"
  sleep infinity
fi
echo "$RECV_B64" | base64 -d > /w/recv.py
log waiting_for_code
python3 /w/recv.py || stop receive
log code_received
mkdir -p /w/code && tar -xzf /w/code.tgz -C /w/code || stop untar
curl -LsSf https://astral.sh/uv/install.sh | sh > /w/logs/uv.log 2>&1 || stop uv_install
export PATH="$HOME/.local/bin:$PATH"
cd /w/code
{ uv python install 3.13 && uv venv --python 3.13 venv3 \
  && uv pip install --python venv3/bin/python -r requirements_part3.txt \
  && uv pip install --python venv3/bin/python --no-deps -e . \
  && uv pip install --python venv3/bin/python "jax[cuda12]==0.11.2" ; } > /w/logs/setup.log 2>&1 || stop setup
git config --global --add safe.directory /w/code
{ venv3/bin/python -c "import sys, jax, jaxlib, flax, optax, evosax, gymnax, rliable; print(sys.version.split()[0], 'jax', jax.__version__, 'backend', jax.default_backend(), jax.devices())"
  echo "commit $(git rev-parse HEAD) dirty_lines $(git status --porcelain -- . ':(exclude)results' | wc -l)"
  nvidia-smi 2>&1 | head -15
  grep -m1 'model name' /proc/cpuinfo; nproc; free -g | head -2; df -h "$VOL" | tail -1; } > /w/logs/env.txt 2>&1
cp /w/logs/env.txt "$OUT/env.txt"
venv3/bin/python -c "import jax, sys; sys.exit(0 if jax.default_backend() == 'gpu' else 1)" || stop no_gpu_backend
log setup_ok
export EVO2_JAX_CACHE="$VOL/jax_cache"   # compiled programs kept on the volume (compile time only)
rc=0
if [ "$JOB" = "tune" ]; then
  # shellcheck disable=SC2086
  venv3/bin/python scripts/part3/tune.py --methods $METHODS --out "$OUT" > /w/logs/job.log 2>&1 || rc=$?
elif [ "$JOB" = "final" ]; then
  # train, training curves, test evaluation; failed runs are left out and reported (final_job.sh)
  OUT="$OUT" RUNS="$RUNS" PY=venv3/bin/python LOGDIR=/w/logs bash scripts/part3/pod/final_job.sh || rc=$?
elif [ "$JOB" = "addendum" ]; then
  # Addendum A: tune ES-fixed, then the final job for RUNS (addendum_job.sh)
  OUT="$OUT" RUNS="$RUNS" PY=venv3/bin/python LOGDIR=/w/logs bash scripts/part3/pod/addendum_job.sh || rc=$?
elif [ "$JOB" = "explore_long" ]; then
  # exploratory longer training plus validation curves; never loads the test split (explore_job.sh)
  OUT="$OUT" RUNS="$RUNS" BUDGET="${BUDGET:-}" PY=venv3/bin/python LOGDIR=/w/logs bash scripts/part3/pod/explore_job.sh || rc=$?
else
  stop "unknown job $JOB"
fi
cp /w/logs/*.log "$OUT/" 2>/dev/null
[ $rc -eq 0 ] && log all_done || log "FAILED: $JOB (exit $rc; see job.log)"
sleep infinity
