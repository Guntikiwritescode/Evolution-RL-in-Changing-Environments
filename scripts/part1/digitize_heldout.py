"""Read the paper's HELD-OUT curves off its figures, by pixel, for comparison only.

These numbers were never used for calibration or any setting. They are read after the
final Part 1 run, only to place the paper's curves next to the rebuilt ones.

Figures read:
  Fig 1 (p57) GS fitness over training      Fig 2 (p58) GS population over training
  Fig 6 (p61) ET + GS fitness in transfer   Fig 7 (p62) ET + GS population in transfer
  Fig 8 (p62) gap bar chart (bar tops)      Fig 9 (p63) ET + GS trait diversity in transfer

Method: same as scripts/part1/digitize_figures.py (frame from spines, axes from tick marks,
line traced by color, value read at each whole-generation vertex), with two changes needed
for lines that stop early (the GS lines in Figs 6, 7, 9 stop near transfer generation 26):
  1. Tracing starts at a given generation for each line (a hint), not in the right half of
     the plot, where the only orange left is the legend sample.
  2. A line is never extended past where it is drawn. An end is extended straight to the
     frame edge only if the traced line reaches within 1 generation of that edge (Figs 1-2,
     whose x axis starts at 0 and ends at 99 on the frame). Gaps of up to 2 generations
     inside a line are filled by straight interpolation and listed; longer gaps stay empty.
Output: results/part1/paper_heldout_digitized.json

Usage: python scripts/part1/digitize_heldout.py path/to/original_paper.pdf
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from digitize_figures import (BLUE, ORANGE, ROOT, _runs, axis_map, find_frame, render,  # noqa: E402
                              tick_centers)

# (name, pdf page, crop rows, x tick labels, y tick labels, {curve: (color, start-generation hint)}, gen range)
FIGS = [
    ("fig1_gs_fitness_train", 57, (1280, 2050), [0, 20, 40, 60, 80], [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
     {"gs": (ORANGE, 50)}, (0, 99)),
    ("fig2_gs_population_train", 58, (660, 1430), [0, 20, 40, 60, 80],
     [30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0], {"gs": (ORANGE, 70)}, (0, 99)),
    ("fig6_fitness_transfer", 61, (1230, 2080), [0, 10, 20, 30, 40, 50],
     [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8], {"et": (BLUE, 40), "gs": (ORANGE, 12)}, (0, 49)),
    ("fig7_population_transfer", 62, (360, 1200), [0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50],
     {"et": (BLUE, 40), "gs": (ORANGE, 12)}, (0, 49)),
    ("fig9_cv2_transfer", 63, (1490, 2340), [0, 10, 20, 30, 40, 50],
     [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], {"et": (BLUE, 40), "gs": (ORANGE, 12)}, (0, 49)),
]
# Fig 8 bar chart: crop rows, y tick labels
FIG8 = ("fig8_gap_bars", 62, (2080, 2950), [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])


def axes_for(crop):
    gray = crop.mean(axis=2)
    dark = gray < 80
    top, bottom, left, right = find_frame(dark)
    band = dark[bottom + 4: bottom + 12, left - 3: right + 4].any(axis=0)
    xt = [left - 3 + c for c in tick_centers(band)]
    bandy = dark[top - 3: bottom + 4, left - 12: left - 4].any(axis=1)
    yt = [top - 3 + r for r in tick_centers(bandy)]
    return (top, bottom, left, right), xt, yt


def trace_from(m, start, max_jump=40):
    """Follow one line left and right from column `start`, taking in each column the run of
    matching pixels closest to the last height. Stops a direction at the first column with no
    run within max_jump (the line has ended). Returns row per column (NaN = not traced)."""
    n = m.shape[1]
    runs = [[(a + b) / 2.0 for a, b in _runs(m[:, c])] for c in range(n)]
    y = np.full(n, np.nan)
    y[start] = runs[start][0]
    for step in (1, -1):
        last, c = y[start], start + step
        misses = 0
        while 0 <= c < n:
            close = [v for v in runs[c] if abs(v - last) <= max_jump]
            if close:
                best = min(close, key=lambda v: abs(v - last))
                y[c] = best
                last = best
                misses = 0
            else:
                misses += 1
                if misses > 6:  # a few columns of nothing: the line has ended (or is hidden)
                    break
            c += step
    return y


def digitize(img, y0, y1, xlabels, ylabels, curves, gen_range, color_tol=55.0):
    crop = img[y0:y1]
    (top, bottom, left, right), xt, yt = axes_for(crop)
    ax, bx, rx = axis_map(xt, xlabels)
    ay, by, ry = axis_map(sorted(yt, reverse=True), ylabels)
    out = {"frame_px": [int(top), int(bottom), int(left), int(right)], "x_ticks_px": xt, "y_ticks_px": yt,
           "tick_fit_max_residual": {"x": rx, "y": ry}, "curves": {}, "notes": {}}
    x0 = left + 3
    inner = crop[top + 3: bottom - 2, x0: right - 2]
    n = inner.shape[1]
    gx = ax * (np.arange(n) + x0) + bx
    g_lo, g_hi = gen_range
    for name, (col, hint) in curves.items():
        m = np.sqrt(((inner - np.array(col)) ** 2).sum(axis=2)) < color_tol
        # start column: nearest column to the hint generation with exactly one run of the color
        c_hint = int(round((hint - bx) / ax)) - x0
        cands = sorted((c for c in range(max(0, c_hint - 40), min(n, c_hint + 41)) if len(_runs(m[:, c])) == 1),
                       key=lambda c: abs(c - c_hint))
        if not cands:
            raise RuntimeError(f"{name}: no single-run column near generation {hint}")
        y = trace_from(m, cands[0])
        ok = np.where(~np.isnan(y))[0]
        series, extended, filled = {}, [], []
        for g in range(g_lo, g_hi + 1):
            c0 = int(round((g - bx) / ax)) - x0
            sel = [c for c in range(c0 - 1, c0 + 2) if 0 <= c < n and not np.isnan(y[c])]
            if sel:
                series[g] = float(ay * (np.median([y[c] for c in sel]) + top + 3) + by)
        # extend an end only where the traced line reaches within 1 generation of that frame edge
        edge_gen = {"lo": gx[0], "hi": gx[-1]}
        for g, cols, edge in ((g_lo, ok[:6], "lo"), (g_hi, ok[-6:], "hi")):
            reaches = abs(gx[cols[0] if edge == "lo" else cols[-1]] - edge_gen[edge]) <= 1.0
            if g not in series and len(cols) >= 2 and reaches and abs(edge_gen[edge] - g) <= 1.0:
                k, b0 = np.polyfit(gx[cols], y[cols], 1)
                series[g] = float(ay * (k * g + b0 + top + 3) + by)
                extended.append(g)
        known = sorted(series)
        for g in range(known[0], known[-1] + 1):
            if g in series:
                continue
            lo = max(k for k in known if k < g)
            hi = min(k for k in known if k > g)
            if hi - lo <= 3:  # gap of at most 2 generations
                series[g] = series[lo] + (series[hi] - series[lo]) * (g - lo) / (hi - lo)
                filled.append(g)
        out["curves"][name] = {str(g): series[g] for g in sorted(series)}
        out["notes"][name] = {"start_column_generation": float(gx[cands[0]]),
                              "first_generation_read": min(series), "last_generation_read": max(series),
                              "traced_x_range_generations": [float(gx[ok[0]]), float(gx[ok[-1]])],
                              "extended_to_edge": extended, "gap_filled": filled,
                              "missing_inside_range": [g for g in range(min(series), max(series) + 1)
                                                       if g not in series]}
    return out


def read_bars(img, y0, y1, ylabels):
    """Fig 8: top of each colored bar, mapped to the y axis."""
    crop = img[y0:y1]
    (top, bottom, left, right), _, yt = axes_for(crop)
    ay, by, ry = axis_map(sorted(yt, reverse=True), ylabels)
    inner = crop[top + 3: bottom - 2, left + 3: right - 2]
    res = {"frame_px": [int(top), int(bottom), int(left), int(right)], "y_ticks_px": yt,
           "tick_fit_max_residual": {"y": ry}}
    for name, col in (("et", BLUE), ("gs", ORANGE)):
        m = np.sqrt(((inner - np.array(col)) ** 2).sum(axis=2)) < 55
        cols = np.where(m.sum(axis=0) > 5)[0]
        mid = cols[len(cols) // 4]  # a quarter across: the black error bar runs down the middle
        rows = np.where(m[:, mid])[0]
        # error bar: the dark vertical run in the bar's center column (it crosses the bar top)
        center = (cols.min() + cols.max()) // 2
        dark_rows = np.where(inner[:, center - 1:center + 2].mean(axis=2).min(axis=1) < 80)[0]
        runs = _runs(np.isin(np.arange(inner.shape[0]), dark_rows))
        merged = []  # the gray zero line is drawn over the error bar: join runs split by <= 3 rows
        for r in runs:
            if merged and r[0] - merged[-1][1] <= 4:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        runs = merged
        top_row = rows.min()
        err = [r for r in runs if r[0] <= top_row <= r[1]]
        res[name] = {"bar_top": float(ay * (rows.min() + top + 3) + by),
                     "bar_bottom": float(ay * (rows.max() + top + 3) + by),
                     "bar_columns_px": [int(cols.min() + left + 3), int(cols.max() + left + 3)]}
        if err:
            r0, r1 = err[0]
            res[name]["error_bar_top"] = float(ay * (r0 + top + 3) + by)
            res[name]["error_bar_bottom"] = float(ay * (r1 + top + 3) + by)
            res[name]["error_bar_half_length"] = (res[name]["error_bar_top"] - res[name]["error_bar_bottom"]) / 2
    return res


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper", "original_paper.pdf")
    res = {"source_pdf": os.path.basename(pdf), "purpose": "comparison only; never used to set anything",
           "method": __doc__.strip().splitlines()[9:20], "figures": {}}
    with tempfile.TemporaryDirectory() as tmp:
        pages = {}
        for name, page, (y0, y1), xl, yl, curves, gr in FIGS:
            if page not in pages:
                pages[page] = render(pdf, page, tmp)
            r = digitize(pages[page], y0, y1, xl, yl, curves, gr)
            r.update(page=page, crop_rows=[y0, y1], x_labels=xl, y_labels=yl)
            res["figures"][name] = r
            print(name, "tick residuals", r["tick_fit_max_residual"],
                  {k: (v["first_generation_read"], v["last_generation_read"], len(r["curves"][k]))
                   for k, v in r["notes"].items()})
        name, page, (y0, y1), yl = FIG8
        if page not in pages:
            pages[page] = render(pdf, page, tmp)
        r = read_bars(pages[page], y0, y1, yl)
        r.update(page=page, crop_rows=[y0, y1], y_labels=yl)
        res["figures"][name] = r
        print(name, {k: r[k] for k in ("et", "gs")}, r["tick_fit_max_residual"])
    out = os.path.join(ROOT, "results", "part1", "paper_heldout_digitized.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
