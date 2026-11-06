"""Checks of paper/v2/numbers.tex.

1. Macros against the generated reports (docs/RESULTS_part3.md, docs/RESULTS_part2.md,
   docs/EXPLORE_part3_long.md): the paper must say what the reports say.
2. Recomputed independently from the saved inputs: Holm-adjusted p-values from the raw p-values,
   and P(X > Y) from the Mann-Whitney U, with code written here, not imported.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import make_numbers as MN  # noqa: E402

LABEL = {"OE": "Oe", "OE-fixed": "OeFixed", "PPO-fixed": "PpoFixed", "PPO-DR": "PpoDr", "ES-DR": "EsDr"}


@pytest.fixture(scope="module")
def macros(tmp_path_factory):
    out = tmp_path_factory.mktemp("numbers") / "numbers.tex"
    MN.build(out, require_addendum=False)
    vals = {}
    for line in out.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}% ", line)
        if m:
            vals[m.group(1)] = m.group(2)
    return vals


def plain(v: str) -> str:
    """The number inside \\ensuremath{...}, as the reports print it."""
    m = re.fullmatch(r"\\ensuremath\{(.*)\}", v)
    return (m.group(1) if m else v).replace("{,}", ",")


def test_macro_names_unique_and_letters_only(macros):
    assert all(n.isalpha() for n in macros)
    assert len(macros) > 150


def test_h1_matches_report(macros):
    rep = (ROOT / "docs/RESULTS_part3.md").read_text()
    m = re.search(r"D = IQM\(OE\) − IQM\(PPO-DR\) = (\S+), 95% CI \[(\S+), (\S+)\]", rep)
    assert m, "H1 line not found in RESULTS_part3.md"
    assert (plain(macros["HOneD"]), plain(macros["HOneDLo"]), plain(macros["HOneDHi"])) == m.groups()
    v = re.search(r"\*\*Verdict: (\w[^.]*)\.\*\*", rep).group(1)
    assert macros["HOneVerdict"] == v


def test_iqm_table_matches_report(macros):
    rep = (ROOT / "docs/RESULTS_part3.md").read_text()
    rows = re.findall(r"^\| (OE|OE-fixed|PPO-fixed|PPO-DR|ES-DR) \| ([\d.]+) \| \[([\d.]+), ([\d.]+)\] \|$", rep, re.M)
    assert len(rows) == 5
    for name, v, lo, hi in rows:
        t = "IQM" + LABEL[name]
        assert (plain(macros[t]), plain(macros[t + "Lo"]), plain(macros[t + "Hi"])) == (v, lo, hi), name


def test_secondary_table_matches_report(macros):
    rep = (ROOT / "docs/RESULTS_part3.md").read_text()
    rows = re.findall(r"^\| S(\d) \| [^|]+\| [^|]+\| [^|]+\| ([\d.]+) \| ([\d.e-]+) \| ([\d.e-]+) \| "
                      r"(\S+) \[(\S+), (\S+)\] \| [^|]+\| ([^|]+) \|$", rep, re.M)
    assert len(rows) == 7
    for n, pxy, p, ph, d, lo, hi, result in rows:
        t = "Sec" + MN.word(int(n))
        assert plain(macros[t + "PXgtY"]) == pxy
        assert float(plain(macros[t + "P"])) == pytest.approx(float(p), rel=1e-3)
        assert float(plain(macros[t + "PHolm"])) == pytest.approx(float(ph), rel=1e-3)
        assert (plain(macros[t + "Diff"]), plain(macros[t + "DiffLo"]), plain(macros[t + "DiffHi"])) == (d, lo, hi)
        assert macros[t + "Result"] == result.strip()


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjustment, written here independently of src/evo2."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def test_holm_and_prob_recomputed(macros):
    sec = json.loads((ROOT / "results/part3/final/analysis.json").read_text())["secondary"]
    adj = holm([r["p"] for r in sec])
    for r, a in zip(sec, adj):
        t = "Sec" + MN.word(int(r["name"][1:]))
        assert float(plain(macros[t + "PHolm"])) == pytest.approx(a, rel=1e-3)
        assert float(plain(macros[t + "PXgtY"])) == pytest.approx(r["U"] / (r["n_x"] * r["n_y"]), abs=5e-4)


def test_part2_rows_match_report(macros):
    rep = (ROOT / "docs/RESULTS_part2.md").read_text()
    main = rep[rep.index("### Main arm"):]
    rows = re.findall(r"^\| (\d) \| [^|]+\| (\S+) \[(\S+), (\S+)\] \| (\S+) \[(\S+), (\S+)\] \|", main, re.M)[:8]
    assert len(rows) == 8

    def unsign(s):  # the report prints "+0.71"; the macro prints "0.71"
        return s.lstrip("+")
    for r in rows:
        t = "PTwoRow" + MN.word(int(r[0]))
        got = [plain(macros[t + x]) for x in ("POne", "POneLo", "POneHi", "PTwo", "PTwoLo", "PTwoHi")]
        assert got == [unsign(x) for x in r[1:]], r[0]


def test_long_gains_match_report(macros):
    rep = (ROOT / "docs/EXPLORE_part3_long.md").read_text()
    gains = re.findall(r"^\| 500[012] \|(?: [\d.]+ \|){10} (\S+) \|", rep, re.M)
    assert gains == [plain(macros["LongGain" + w]) for w in ("One", "Two", "Three")]
    assert macros["LongReading"] in rep


def test_number_format():
    assert MN.num(-0.0004) == r"\ensuremath{-0.000}"          # sign kept, as in the reports
    assert MN.num(1.23456, 2, sign=True) == r"\ensuremath{+1.23}"
    assert MN.pval(0.0001827) == r"\ensuremath{0.0001827}"
    assert MN.pval(9.54e-60) == r"\ensuremath{9.540\times 10^{-60}}"
    with pytest.raises(ValueError):
        MN.num(float("nan"))


def test_addendum_matches_report(macros):
    """Addendum A macros against docs/RESULTS_part3_addendum.md (skipped until the analysis exists)."""
    rep_path = ROOT / "docs/RESULTS_part3_addendum.md"
    if not (ROOT / "results/part3/addendum/analysis.json").exists() or not rep_path.exists():
        pytest.skip("addendum not analysed yet")
    rep = rep_path.read_text()
    m = re.search(r"D = IQM\(ES-DR\) − IQM\(ES-fixed\) = (\S+), 95% CI \[(\S+), (\S+)\]", rep)
    assert (plain(macros["AOneD"]), plain(macros["AOneDLo"]), plain(macros["AOneDHi"])) == m.groups()
    assert macros["AOneVerdict"] == re.search(r"## A1.*?\*\*Verdict: ([^*]+?)\.\*\*", rep, re.S).group(1)
    row = re.search(r"^\| ES-fixed \| ([\d.]+) \| \[([\d.]+), ([\d.]+)\] \|$", rep, re.M).groups()
    assert (plain(macros["IQMEsFixed"]), plain(macros["IQMEsFixedLo"]), plain(macros["IQMEsFixedHi"])) == row
    rows = re.findall(r"^\| A(\d) \| [^|]+\| [^|]+\| [^|]+\| ([\d.]+) \| ([\d.e-]+) \| ([\d.e-]+) \| "
                      r"(\S+) \[(\S+), (\S+)\] \| [^|]+\| ([^|]+) \|$", rep, re.M)
    assert [r[0] for r in rows] == ["2", "3"]
    for n, pxy, p, ph, d, lo, hi, result in rows:
        t = "Add" + MN.word(int(n))
        assert plain(macros[t + "PXgtY"]) == pxy
        assert float(plain(macros[t + "PHolm"])) == pytest.approx(float(ph), rel=1e-3)
        assert (plain(macros[t + "Diff"]), plain(macros[t + "DiffLo"]), plain(macros[t + "DiffHi"])) == (d, lo, hi)
        assert macros[t + "Result"] == result.strip()
    s = re.search(r"Variety effect for ES \(ES-DR − ES-fixed\): (\S+) \[(\S+), (\S+)\]; for OE \(OE − OE-fixed\): "
                  r"(\S+) \[(\S+), (\S+)\]; difference \(ES minus OE\): (\S+) \[(\S+), (\S+)\] \(([^)]+)\)", rep)
    names = [f"AFour{k}{e}" for k in ("VarietyEs", "VarietyOe", "Interaction") for e in ("", "Lo", "Hi")]
    assert [plain(macros[n]) for n in names] == list(s.groups()[:9])
    assert macros["AFourVerdict"] == s.group(10)

