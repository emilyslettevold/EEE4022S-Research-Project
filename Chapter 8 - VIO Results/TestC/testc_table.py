#!/usr/bin/env python3
"""
testc_table.py - Test C summary: for every run, the stage metrics AND the output, side by side, plus the change
from the baseline run of the same bag. Standard library only.

Usage (Lenovo, in the results folder, e.g. ~/vio_data/testC):
  python3 testc_table.py                      # every <bag>_<tag>.csv in this folder
  python3 testc_table.py --base base          # tag of the baseline run (default: base)

Output metrics (from <run>.csv):   closure_m (start-to-end; orbit/fig-8 return to their start), path_m,
                                   jumps (pose steps > 0.10 m), max_speed
Stage metrics (from _metrics.csv, median after initialisation):
  1 td_ms   2 n_feat, stereo_ratio   3 guess_frac, depth_med, cap_frac (depths at the 100 m bound)
  4 ba_norm   6 solve_ms, cost, outlier_frac
Writes testc_summary.csv.
"""
import argparse, csv, glob, math, os, statistics as st

def sec(x):
    v = float(x); return v / 1e9 if v > 1e12 else v

def output_metrics(path):
    P, V = [], []
    for r in csv.reader(open(path)):
        if len(r) < 4 or not r[0].strip():
            continue
        try:
            P.append([float(v) for v in r[1:4]])
            V.append([float(v) for v in r[8:11]] if len(r) >= 11 else [0, 0, 0])
        except ValueError:
            pass
    if not P:
        return {}
    steps = [math.dist(p, q) for p, q in zip(P, P[1:])]
    return {"poses": len(P), "closure_m": round(math.dist(P[0], P[-1]), 3), "path_m": round(sum(steps), 2),
            "jumps": sum(s > 0.10 for s in steps), "max_speed": round(max(math.hypot(*v) for v in V), 2)}

def stage_metrics(path):
    if not os.path.exists(path):
        return {}
    rows = [r for r in csv.DictReader(open(path)) if r.get("solver_flag", "").strip() == "1"]
    if not rows:
        return {"note": "never initialised"}
    def med(f):
        vals = [f(r) for r in rows]
        vals = [v for v in vals if v is not None]
        return round(st.median(vals), 3) if vals else ""
    def fl(r, k):
        try: return float(r[k])
        except (KeyError, ValueError): return None
    nwin = lambda r: max(fl(r, "n_win") or 0, 1)
    out = {
        "td_ms": med(lambda r: fl(r, "td_ms")),
        "n_feat": med(lambda r: fl(r, "n_feat")),
        "stereo_ratio": med(lambda r: (fl(r, "n_stereo") or 0) / max(fl(r, "n_feat") or 0, 1)),
        "guess_frac": med(lambda r: (fl(r, "n_depth_guess") or 0) / nwin(r)),
        "depth_med": med(lambda r: fl(r, "depth_med")),
        "ba_norm": med(lambda r: fl(r, "ba_norm")),
        "solve_ms": med(lambda r: fl(r, "solve_ms")),
        "cost": med(lambda r: fl(r, "cost")),
        "outlier_frac": med(lambda r: (fl(r, "n_outliers") or 0) / nwin(r)),
    }
    if "n_depth_cap" in rows[0]:
        out["cap_frac"] = med(lambda r: (fl(r, "n_depth_cap") or 0) / nwin(r))
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="base")
a = ap.parse_args()

runs = []
for f in sorted(glob.glob("*.csv")):
    if f.endswith("_metrics.csv") or f.startswith(("testc_summary", "vins_check")):
        continue
    name = f[:-4]
    if "_" not in name:
        continue
    bag, tag = name.rsplit("_", 1)
    runs.append({"bag": bag, "tag": tag, **output_metrics(f), **stage_metrics(f"{name}_metrics.csv")})
if not runs:
    raise SystemExit("no <bag>_<tag>.csv files here")

cols = ["bag", "tag", "closure_m", "jumps", "max_speed", "path_m", "td_ms", "n_feat", "stereo_ratio",
        "guess_frac", "cap_frac", "depth_med", "ba_norm", "solve_ms", "cost", "outlier_frac", "poses", "note"]
with open("testc_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(runs)

show = ["tag", "closure_m", "jumps", "max_speed", "td_ms", "stereo_ratio", "guess_frac", "cap_frac", "depth_med", "ba_norm", "solve_ms"]
for bag in sorted({r["bag"] for r in runs}):
    rs = sorted([r for r in runs if r["bag"] == bag], key=lambda r: (r["tag"] != a.base, r["tag"]))
    base = next((r for r in rs if r["tag"] == a.base), None)
    print(f"\n=== {bag}" + ("" if base else f"   (no '{a.base}' run - no deltas)"))
    print("".join(f"{c:>13}" for c in show))
    for r in rs:
        print("".join(f"{str(r.get(c, '')):>13}" for c in show) + (f"   {r['note']}" if r.get("note") else ""))
        if base and r is not base and isinstance(r.get("closure_m"), float) and isinstance(base.get("closure_m"), float):
            print(f"{'':>13}  closure change vs base: {r['closure_m'] - base['closure_m']:+.3f} m")
print("\nwrote testc_summary.csv")
