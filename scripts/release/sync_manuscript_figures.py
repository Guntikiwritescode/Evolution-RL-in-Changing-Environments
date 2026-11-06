"""Copy the seven regenerated figures only after checking their numerical inputs."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[2]
for p in sorted((ROOT/'manuscript/figure_data').glob('*.json')):
 fresh=ROOT/'results/publication'/p.name
 if json.loads(fresh.read_text())!=json.loads(p.read_text()):raise SystemExit(f'Numerical figure data changed: {p.name}')
 src=ROOT/'figures/publication'/p.with_suffix('.pdf').name
 if not src.is_file():raise SystemExit(f'Missing figure: {src.name}')
 shutil.copy2(src,ROOT/'manuscript/figures'/src.name)
 print(src.name)
