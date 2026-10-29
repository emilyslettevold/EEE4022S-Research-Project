#!/usr/bin/env python3
"""
stereo_support.py - how strong was the stereo (visual depth) constraint in each stereo-inertial run?

  python3 ~/vio_data/stereo_support.py fig8_3 orbit3 line1 rail170 lawnmower2 square2

Reads ~/vio_data/testD/NAME/NAME_SI_metrics.csv (the per-frame VINS internals) and prints, for frames where the
solver is running:
  stereo_med      median stereo-matched features per frame
  ratio_med       median stereo-matched / tracked
  weak_%          share of frames with fewer than --weak stereo matches (default 10)
  longest_weak_s  longest continuous stretch with fewer than --weak matches (s), and when it started
  guess_med       median fraction of window features with a guessed depth
Optionally also for the SO run (--so), to check both configurations saw the same scene.
"""
import argparse, csv
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="+")
ap.add_argument("--weak", type=int, default=10)
ap.add_argument("--so", action="store_true", help="also report the stereo-only runs")
a = ap.parse_args()
VD = Path.home() / "vio_data"


def col(rows, k):
    out = []
    for r in rows:
        try:
            out.append(float(r[k]))
        except (KeyError, ValueError, TypeError):
            out.append(np.nan)
    return np.array(out)


res = []
for name in a.names:
    for tag in (("SI", "SO") if a.so else ("SI",)):
        f = VD / "testD" / name / f"{name}_{tag}_metrics.csv"
        if not f.exists():
            print(f"{name} {tag}: no {f.name}"); continue
        rows = list(csv.DictReader(open(f)))
        t = col(rows, "t"); t = np.where(t > 1e11, t / 1e9, t)
        run = col(rows, "solver_flag") == 1
        t, ns, nf = t[run], col(rows, "n_stereo")[run], col(rows, "n_feat")[run]
        ng, nw = col(rows, "n_depth_guess")[run], col(rows, "n_win")[run]
        weak = ns < a.weak
        best, start, cur, cs = 0.0, np.nan, 0.0, None
        for i in range(1, len(t)):
            if weak[i] and weak[i - 1]:
                cs = t[i - 1] if cs is None else cs
                if t[i] - cs > best:
                    best, start = t[i] - cs, cs
            else:
                cs = None
        res.append(dict(run=f"{name} {tag}", stereo_med=f"{np.nanmedian(ns):.0f}",
                        ratio_med=f"{np.nanmedian(ns / np.maximum(nf, 1)):.2f}",
                        **{"weak_%": f"{100 * np.mean(weak):.1f}"},
                        longest_weak_s=f"{best:.1f}" + (f" (from {start - t[0]:.0f} s)" if best > 0 else ""),
                        guess_med=f"{np.nanmedian(ng / np.maximum(nw, 1)):.2f}"))
if res:
    cols = list(res[0])
    w = {c: max(len(c), *(len(r[c]) for r in res)) for c in cols}
    print("  ".join(c.rjust(w[c]) for c in cols))
    for r in res:
        print("  ".join(r[c].rjust(w[c]) for c in cols))
    print(f"\nweak = fewer than {a.weak} stereo-matched features in a frame; times from the first solved frame")
