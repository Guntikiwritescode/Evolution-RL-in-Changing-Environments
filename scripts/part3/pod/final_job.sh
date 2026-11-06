#!/bin/bash
# The final job (PREREG Sections 3, 8.4, 9): train each run, score its training curve on the
# validation environments, then evaluate it on the test split with the full adaptation.
# Called by start_job.sh (JOB=final) from the code folder; tested by tests/test_part3_pod.py.
# Env: OUT (results folder on the volume), RUNS ("method:seed ..."), PY (python),
#      LOGDIR (status folder; default /w/logs), TUNED_EXTRA (optional: the addendum's tuned
#      settings, passed to train.py for the addendum method es_fixed only; PREREG Section 16).
# Exit 0 only if at least one run trained and its curves and evaluation finished.
set -uo pipefail
LOGDIR="${LOGDIR:-/w/logs}"
PY="${PY:-venv3/bin/python}"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOGDIR/status.txt"; cp "$LOGDIR/status.txt" "$OUT/status.txt" 2>/dev/null; }

# Every look at the test split is recorded (evaluate.py appends a line to
# results/part3/test_access_log.jsonl before it loads the split). That path is linked to a file on
# the volume, so the record is on the volume from the first line and a relaunch only adds to it.
mkdir -p "$OUT" results/part3 && touch "$OUT/test_access_log.jsonl" \
  && ln -sfn "$OUT/test_access_log.jsonl" results/part3/test_access_log.jsonl \
  || { log "FAILED: access_log_link"; exit 2; }

dirs=()
for r in $RUNS; do
  m="${r%%:*}"; s="${r##*:}"; d="$OUT/${m}_seed${s}"
  extra=()
  if [ "$m" = "es_fixed" ]; then
    [ -n "${TUNED_EXTRA:-}" ] || { log "run_failed $m $s (no TUNED_EXTRA)"; continue; }
    extra=(--tuned-extra "$TUNED_EXTRA")
  fi
  # A run that crashes, or whose parameters stop being finite (train.py writes failed.json instead
  # of done.json), is reported and left out; the other runs go on. Replacements run as a new part.
  if "$PY" scripts/part3/train.py --method "$m" --purpose final --seed "$s" --out "$OUT" "${extra[@]}" >> "$LOGDIR/job.log" 2>&1 \
     && [ -f "$d/done.json" ] && [ ! -f "$d/failed.json" ]; then
    log "trained $m $s"
    dirs+=("$d")
  else
    log "run_failed $m $s (see job.log)"
  fi
done
[ ${#dirs[@]} -gt 0 ] || { log "no run trained"; exit 1; }
"$PY" scripts/part3/curves.py --runs "${dirs[@]}" >> "$LOGDIR/job.log" 2>&1 || { rc=$?; log "curves failed (exit $rc)"; exit "$rc"; }
log "curves done"
"$PY" scripts/part3/evaluate.py --runs "${dirs[@]}" --split test --final --adapt >> "$LOGDIR/job.log" 2>&1 \
  || { rc=$?; log "evaluation failed (exit $rc)"; exit "$rc"; }
log "evaluation done"
