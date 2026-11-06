"""Recheck retained data; never train or evaluate new policies. See REPRODUCE.md."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/audit/viability/rerun'
OUT.mkdir(parents=True, exist_ok=True)
scripts = [
    'audit_raw.py', 'audit_part2_trajectories.py', 'audit_part2_outcomes.py',
    'strict_review_checks.py', 'audit_prewrite_completion.py',
    'independent_reviews/raw_record_check.py',
    'independent_reviews/independent_check.py',
    'independent_reviews/verify_design_records.py',
    'independent_reviews/step_axis_reanalysis.py',
]
status = []
for script in scripts:
    log = OUT / (Path(script).stem + '.log')
    with log.open('w') as f:
        p = subprocess.run([sys.executable, str(ROOT/'scripts/report'/script)],
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    status.append({'script': script, 'returncode': p.returncode, 'log': str(log.relative_to(ROOT))})
    print(script, 'PASS' if p.returncode == 0 else 'FAIL', flush=True)
    (OUT/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    if p.returncode:
        raise SystemExit(p.returncode)
print('All existing-data reanalysis entry points completed. This is not a retraining or an acceptance test.')
