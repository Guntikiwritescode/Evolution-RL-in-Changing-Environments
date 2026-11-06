#!/bin/bash
# Exploratory longer-training check (docs/EXPLORE_part3_long.md). Decided on 4 Nov 2026,
# after the final results were seen, so everything it produces is labeled exploratory.
#
# Each run trains to BUDGET agent-steps with the seed of a final run. The seed fixes the random keys
# and chunk sizes depend only on progress toward the target, so the run takes exactly the final
# run's chunks until the final run's last chunk (just before 2e8), then keeps going. After training,
# its curve is scored on the 24 validation environments (curves.py, as for the final runs).
# The test split is never loaded: no evaluate.py call here.
#
# Called by start_job.sh (JOB=explore_long) from the code folder; tested by tests/test_part3_pod.py.
# Env: OUT (results folder on the volume), RUNS ("method:seed ..."), BUDGET (agent-steps),
#      PY (python), LOGDIR (status folder; default /w/logs).
# Exit 0 only if at least one run trained and its curve was scored.
set -uo pipefail
LOGDIR="${LOGDIR:-/w/logs}"
PY="${PY:-venv3/bin/python}"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOGDIR/status.txt"; cp "$LOGDIR/status.txt" "$OUT/status.txt" 2>/dev/null; }

mkdir -p "$OUT" || { log "FAILED: no output folder"; exit 2; }
case "${BUDGET:-}" in
  ''|*[!0-9]*) log "FAILED: BUDGET must be a whole number of agent-steps"; exit 2;;
esac

ok=0
for r in $RUNS; do
  m="${r%%:*}"; s="${r##*:}"; d="$OUT/${m}_seed${s}"
  # A run that crashes, or whose parameters stop being finite (failed.json), is reported; the others go on.
  if "$PY" scripts/part3/train.py --method "$m" --purpose final --seed "$s" --budget "$BUDGET" --out "$OUT" \
       >> "$LOGDIR/job.log" 2>&1 && [ -f "$d/done.json" ] && [ ! -f "$d/failed.json" ]; then
    log "trained $m $s"
    if "$PY" scripts/part3/curves.py --runs "$d" >> "$LOGDIR/job.log" 2>&1; then
      log "curves done $m $s"
      ok=$((ok + 1))
    else
      log "curves_failed $m $s (see job.log)"
    fi
  else
    log "run_failed $m $s (see job.log)"
  fi
done
[ "$ok" -gt 0 ] || { log "no run trained and scored"; exit 1; }
