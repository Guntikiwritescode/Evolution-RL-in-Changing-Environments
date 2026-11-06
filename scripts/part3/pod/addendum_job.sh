#!/bin/bash
# Addendum A (PREREG_part3.md Section 16, 4 Nov 2026): ES-fixed, tuned and run in one job.
#   1. tune es_fixed: 8 trials, seeds 4500-4507, a quarter of the training budget each, scored on the
#      24 validation environments only (tune.py, the procedure every core method had);
#   2. write the chosen settings to $OUT/tuned_addendum.yaml (tune.py --write-addendum);
#   3. the final job (final_job.sh) for RUNS: train, training curves, test evaluation with the full
#      adaptation; es_fixed runs use $OUT/tuned_addendum.yaml, every other run the committed tuned.yaml.
# Each step runs only if the one before it succeeded. Called by start_job.sh (JOB=addendum) from the
# code folder; tested by tests/test_part3_pod.py.
# Env: OUT, RUNS ("method:seed ..."), PY, LOGDIR (as final_job.sh).
set -uo pipefail
LOGDIR="${LOGDIR:-/w/logs}"
PY="${PY:-venv3/bin/python}"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOGDIR/status.txt"; cp "$LOGDIR/status.txt" "$OUT/status.txt" 2>/dev/null; }
mkdir -p "$OUT" || { log "FAILED: no output folder"; exit 2; }

"$PY" scripts/part3/tune.py --methods es_fixed --out "$OUT/tuning" >> "$LOGDIR/job.log" 2>&1 \
  || { rc=$?; log "tuning failed (exit $rc)"; exit "$rc"; }
log "tuning done"
"$PY" scripts/part3/tune.py --write-addendum "$OUT/tuning" --to "$OUT/tuned_addendum.yaml" >> "$LOGDIR/job.log" 2>&1 \
  && [ -f "$OUT/tuned_addendum.yaml" ] || { log "FAILED: writing the tuned settings"; exit 2; }
log "tuned settings written"
OUT="$OUT" RUNS="$RUNS" PY="$PY" LOGDIR="$LOGDIR" TUNED_EXTRA="$OUT/tuned_addendum.yaml" \
  bash scripts/part3/pod/final_job.sh
