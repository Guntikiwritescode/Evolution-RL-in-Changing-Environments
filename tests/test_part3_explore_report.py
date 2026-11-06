"""The reading rule of the exploratory longer-training check (scripts/part3/explore_long_report.py),
fixed on 4 Nov 2026 before its results existed."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rep", ROOT / "scripts/part3/explore_long_report.py")
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)


def test_gain_is_last_two_tenths_minus_first_two():
    s = [1, 3, 50, 50, 50, 50, 50, 50, 10, 14]
    assert rep.gain(s) == pytest.approx(12 - 2)
    with pytest.raises(ValueError):
        rep.gain(s[:9])


def test_reading_needs_every_run_clear_for_helped_and_none_for_did_not_help():
    c = rep.GAIN_CLEAR
    assert rep.GAIN_CLEAR == 4.3
    assert rep.reading([c + 0.01] * 3) == "more training helped"
    assert rep.reading([c] * 3) == "did not help"                  # equal to the threshold is not clear
    assert rep.reading([c + 1, c + 1, 0.0]) == "mixed"
    assert rep.reading([-9.0, 0.0, c]) == "did not help"


def test_first_difference_ignores_wall_time():
    a = [{"chunk": 1, "food": 3, "wall_seconds": 1.0}, {"chunk": 2, "food": 4, "wall_seconds": 2.0}]
    b = [{"chunk": 1, "food": 3, "wall_seconds": 9.0}, {"chunk": 2, "food": 5, "wall_seconds": 9.0}]
    assert rep.first_difference(a, b)[0] == 2
    assert rep.first_difference(a, [dict(x, wall_seconds=0) for x in a]) is None
