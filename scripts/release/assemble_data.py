"""Verify and extract research data from complete ZIPs or repository archive parts."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, tempfile, zipfile
ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def extract(stream, asset):
    with zipfile.ZipFile(stream) as z:
        manifest = json.loads(z.read('FILE_MANIFEST.json'))
        expected = {row['path'] for row in manifest}
        actual = {n for n in z.namelist() if n != 'FILE_MANIFEST.json' and not n.endswith('/')}
        roots = set(asset.get('allowed_roots', ['results']))
        if len(manifest) != asset['files'] or expected != actual:
            raise SystemExit('Archive manifest does not cover payload')
        for row in manifest:
            name = PurePosixPath(row['path'])
            if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0] not in roots:
                raise SystemExit('Unsafe member path')
            out = ROOT / str(name)
            b = z.read(str(name))
            if len(b) != row['bytes'] or hashlib.sha256(b).hexdigest() != row['sha256']:
                raise SystemExit(f'Payload mismatch: {name}')
            if out.exists() and digest(out) != row['sha256']:
                raise SystemExit(f'Refusing to overwrite different data: {name}')
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                out.write_bytes(b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', type=Path, default=ROOT / 'data_archives')
    args = ap.parse_args()
    assets = json.loads((ROOT / 'provenance/data_assets.json').read_text())
    for asset in assets:
        full = args.assets / asset['filename']
        if full.is_file():
            if digest(full) != asset['sha256']:
                raise SystemExit(f'Altered archive: {full.name}')
            extract(full, asset)
        else:
            if not asset.get('parts'):
                raise SystemExit(f'Missing archive: {full.name}')
            with tempfile.TemporaryFile() as joined:
                h = hashlib.sha256()
                for part in asset['parts']:
                    name = PurePosixPath(part['path'])
                    if name.is_absolute() or '..' in name.parts:
                        raise SystemExit('Unsafe archive-part path')
                    p = args.assets / str(name)
                    if not p.is_file() or p.stat().st_size != part['bytes'] or digest(p) != part['sha256']:
                        raise SystemExit(f'Missing or altered archive part: {name}')
                    with p.open('rb') as f:
                        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
                            h.update(b)
                            joined.write(b)
                if joined.tell() != asset['bytes'] or h.hexdigest() != asset['sha256']:
                    raise SystemExit(f'Archive checksum mismatch: {asset["filename"]}')
                joined.seek(0)
                extract(joined, asset)
        print(f'{asset["filename"]}: verified and assembled {asset["files"]} files', flush=True)

if __name__ == '__main__':
    main()
