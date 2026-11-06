"""Bibliography source failures must not erase the last usable bibliography."""
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("source", [None, "not valid JSON"])
def test_source_failure_preserves_existing_outputs(tmp_path, monkeypatch, source):
    path = Path(__file__).resolve().parents[1] / "scripts/report/build_bib.py"
    spec = importlib.util.spec_from_file_location("bib_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = tmp_path / "refs_sources"
    sources.mkdir()
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(module, "SRC_DIR", sources)
    monkeypatch.setattr(module, "REFS", [("required", "doi", "10.test/source", "Required title")])
    monkeypatch.setattr(module, "MANUAL", {})
    before = {"refs.bib": b"last complete bibliography\n", "refs_check.md": b"last successful check\n"}
    for name, data in before.items():
        (tmp_path / name).write_bytes(data)
    if source is not None:
        (sources / "required.crossref.json").write_text(source)
    assert module.build(offline=True) != 0
    assert {name: (tmp_path / name).read_bytes() for name in before} == before
