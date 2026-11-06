"""Read the paper's ET training curves off its figures (Figs 1-5), by pixel.

Method, per figure:
1. Render the PDF page at 300 dpi (pdftoppm).
2. Find the plot frame: the longest dark horizontal/vertical lines (matplotlib spines).
3. Find the tick marks just outside the bottom and left spines, and pair them,
   in order, with the tick labels printed on the figure (typed in below from the image).
4. Fit a straight pixel -> value map for each axis from those ticks.
5. Pick pixels close to the line's color inside the frame; for each pixel column
   take the median height; convert to data units; read values at whole generations.
Output: calibration/digitized_paper_figures.json. These numbers are approximate
readings of printed charts, used only as calibration targets with stated tolerances.

Usage: python scripts/part1/digitize_figures.py path/to/original_paper.pdf
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Line colors (seaborn "colorblind" palette as rendered in the PDF)
BLUE = (1, 115, 178)
ORANGE = (222, 143, 5)
GREEN = (2, 158, 115)
PINK = (204, 120, 188)

# (name, pdf page, crop rows (y0, y1) in the 300-dpi render, x tick labels, y tick labels, curves)
FIGS = [
    ("fig1_et_fitness", 57, (1280, 2050), [0, 20, 40, 60, 80], [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
     {"et": BLUE}),
    ("fig2_et_population", 58, (660, 1430), [0, 20, 40, 60, 80],
     [30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0], {"et": BLUE}),
    ("fig3_et_traits", 59, (420, 1360), [0, 20, 40, 60, 80, 100, 120, 140], [0, 5, 10, 15, 20, 25, 30, 35],
     {"speed": BLUE, "sense": GREEN, "size": ORANGE, "reach": PINK}),
    ("fig4_et_cv2", 60, (420, 1190), [0, 20, 40, 60, 80], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], {"et": BLUE}),
    ("fig5_et_shannon", 60, (1740, 2510), [0, 20, 40, 60, 80], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
     {"et": BLUE}),
]


def render(pdf, page, outdir):
    stem = os.path.join(outdir, f"p{page}")
    subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", "300", "-png", pdf, stem], check=True)
    return np.asarray(Image.open(f"{stem}-{page}.png").convert("RGB")).astype(int)


def longest_run(mask_1d):
    best = cur = 0
    for v in mask_1d:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def find_frame(dark):
    """Plot frame from its spines: first find the two long vertical lines, then
    the rows that are dark across (almost) the whole span between them."""
    h, w = dark.shape
    cols = [c for c in range(w) if longest_run(dark[:, c]) > 0.4 * h]
    if not cols:
        raise RuntimeError("frame not found (no vertical spines)")
    left, right = min(cols), max(cols)
    span = dark[:, left:right + 1]
    rows = [r for r in range(h) if span[r].mean() > 0.9]
    if not rows:
        raise RuntimeError("frame not found (no horizontal spines)")
    return min(rows), max(rows), left, right  # top, bottom, left, right


def tick_centers(profile):
    """Centers of runs of True in a 1-D boolean array."""
    out, start = [], None
    for i, v in enumerate(list(profile) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start + i - 1) / 2.0)
            start = None
    return out


def axis_map(pixels, labels):
    if len(pixels) != len(labels):
        raise RuntimeError(f"found {len(pixels)} ticks, expected {len(labels)}: {pixels}")
    a, b = np.polyfit(pixels, labels, 1)
    resid = np.max(np.abs(np.polyval([a, b], pixels) - np.array(labels)))
    return a, b, float(resid)


def digitize(img, y0, y1, xlabels, ylabels, curves, gen_range, color_tol=55.0):
    crop = img[y0:y1]
    gray = crop.mean(axis=2)
    dark = gray < 80
    top, bottom, left, right = find_frame(dark)
    # x ticks: dark pixels in a band below the bottom spine
    band = dark[bottom + 4: bottom + 12, left - 3: right + 4].any(axis=0)
    xt = [left - 3 + c for c in tick_centers(band)]
    # y ticks: dark pixels in a band left of the left spine
    bandy = dark[top - 3: bottom + 4, left - 12: left - 4].any(axis=1)
    yt = [top - 3 + r for r in tick_centers(bandy)]
    ax, bx, rx = axis_map(xt, xlabels)
    ay, by, ry = axis_map(sorted(yt, reverse=True), ylabels)  # larger value = higher = smaller row
    out = {"frame_px": [int(top), int(bottom), int(left), int(right)],
           "x_ticks_px": xt, "y_ticks_px": yt, "tick_fit_max_residual": {"x": rx, "y": ry}, "curves": {}}
    inner = crop[top + 3: bottom - 2, left + 3: right - 2]
    for name, col in curves.items():
        d = np.sqrt(((inner - np.array(col)) ** 2).sum(axis=2))
        m = d < color_tol
        ypix_col = track_line(m)
        cols_x = np.arange(inner.shape[1]) + left + 3
        gx = ax * cols_x + bx  # generation at each column
        g_lo, g_hi = gen_range
        series, interpolated = {}, []
        for g in range(g_lo, g_hi + 1):
            # the line is a polyline with a vertex at each whole generation:
            # read the tracked height in the columns within 1 px of that vertex
            c0 = int(round((g - bx) / ax)) - (left + 3)
            sel = [c for c in range(c0 - 1, c0 + 2) if 0 <= c < len(ypix_col) and not np.isnan(ypix_col[c])]
            if sel:
                series[g] = float(ay * (np.median([ypix_col[c] for c in sel]) + top + 3) + by)
        # end points sit on the frame edge, where no column is read: extend the
        # line straight from its first/last 6 tracked columns
        ok = np.where(~np.isnan(ypix_col))[0]
        for g, cols in ((g_lo, ok[:6]), (g_hi, ok[-6:])):  # only used if no column sits on the vertex
            if g not in series and len(cols) >= 2:
                k, b0 = np.polyfit(gx[cols], ypix_col[cols], 1)
                series[g] = float(ay * (k * g + b0 + top + 3) + by)
                out.setdefault("extrapolated_gens", {}).setdefault(name, []).append(g)
        known = sorted(series)
        for g in range(known[0], known[-1] + 1):  # fill short gaps (e.g. where another line is drawn on top)
            if g not in series:
                lo = max(k for k in known if k < g)
                hi = min(k for k in known if k > g)
                series[g] = series[lo] + (series[hi] - series[lo]) * (g - lo) / (hi - lo)
                interpolated.append(g)
        out["curves"][name] = {str(g): series[g] for g in sorted(series)}
        out.setdefault("interpolated_gens", {})[name] = interpolated
    return out


def track_line(m, max_jump=40):
    """Follow one line across columns. In each column, matching pixels form one
    or more vertical runs (the line, plus e.g. a legend sample of the same color).
    Start from the column whose single run is longest-lived in the right half,
    then walk left and right, always taking the run closest to the last height.
    Returns the line's row (run center) per column, NaN where no run is close."""
    n = m.shape[1]
    runs = [[(a + b) / 2.0 for a, b in _runs(m[:, c])] for c in range(n)]
    single = [c for c in range(n // 2, n) if len(runs[c]) == 1]
    if not single:
        single = [c for c in range(n) if runs[c]]
    start = single[len(single) // 2]
    y = np.full(n, np.nan)
    y[start] = runs[start][0]
    for rng in (range(start + 1, n), range(start - 1, -1, -1)):
        last = y[start]
        for c in rng:
            if runs[c]:
                best = min(runs[c], key=lambda v: abs(v - last))
                if abs(best - last) <= max_jump:
                    y[c] = best
                    last = best
    return y


def _runs(col):
    out, start = [], None
    for i, v in enumerate(list(col) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper", "original_paper.pdf")
    res = {"source": os.path.relpath(pdf, ROOT), "method": __doc__.strip().splitlines()[2:12], "figures": {}}
    with tempfile.TemporaryDirectory() as tmp:
        pages = {}
        for name, page, (y0, y1), xl, yl, curves in FIGS:
            if page not in pages:
                pages[page] = render(pdf, page, tmp)
            r = digitize(pages[page], y0, y1, xl, yl, curves, (0, 149 if "traits" in name else 99))
            r.update(page=page, crop_rows=[y0, y1], x_labels=xl, y_labels=yl)
            res["figures"][name] = r
            print(name, "tick fit residuals", r["tick_fit_max_residual"],
                  {k: len(v) for k, v in r["curves"].items()})
    out = os.path.join(ROOT, "calibration", "digitized_paper_figures.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
