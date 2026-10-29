#!/usr/bin/env python3
"""
Print the numbers still marked [fill in] in the VIO framework selection section.

Run on the Mac, in vins_env, from ~/Desktop/tank_benchmark, after evaluate_all.sh:
    python3 report_numbers.py
    python3 report_numbers.py --runs results/accuracy/eval/runs_all.csv     # explicit path

Needs numpy and scipy only (both already used by evaluate.py).
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

SEQS = ["Structure_Easy", "HalfTank_Easy", "WholeTank_Medium", "WholeTank_Hard"]
METHODS = ["VINS", "ORB3"]
NAMES = {"VINS": "VINS-Fusion", "ORB3": "ORB-SLAM3"}


def f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/accuracy/eval/runs_all.csv")
    ap.add_argument("--min-rpe-pairs", type=int, default=20)
    ap.add_argument("--results", default="results/accuracy")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.runs)))
    g = defaultdict(list)
    for r in rows:
        g[(r["sequence"], r["method"])].append(r)

    def vals(s, m, key, ok_only=True):
        v = np.array([f(r.get(key)) for r in g[(s, m)] if (not ok_only or r["success"] == "1")])
        return v[np.isfinite(v)]

    print("== Runs found per sequence and framework (expect 10 each) ==")
    for s in SEQS:
        print(f"  {s:18s} " + "  ".join(f"{NAMES[m]}: {len(g[(s, m)])}" for m in METHODS))

    print("\n== Table 2: path length (m), from the ground truth ==")
    for s in SEQS:
        pl = [f(r["path_len_m"]) for m in METHODS for r in g[(s, m)]]
        src = {r.get("path_len_source", "?") for m in METHODS for r in g[(s, m)]}
        if pl:
            print(f"  {s:18s} {np.nanmedian(pl):7.1f} m   (source: {', '.join(sorted(src))})")

    print("\n== Table 5: drift = median ATE / path length (successful runs) ==")
    for s in SEQS:
        pl = np.nanmedian([f(r["path_len_m"]) for m in METHODS for r in g[(s, m)]] or [np.nan])
        for m in METHODS:
            v = vals(s, m, "ate_t_rmse")
            if len(v):
                print(f"  {s:18s} {NAMES[m]:12s} {np.median(v):.3f} m  ->  {100 * np.median(v) / pl:.2f} % of path")

    print(f"\n== Table 6: RPE translation (m), median over successful runs "
          f"(blank if fewer than {a.min_rpe_pairs} pairs) ==")
    print(f"  {'sequence':18s} {'framework':12s} {'1 m':>8s} {'2 m':>8s} {'5 m':>8s}   pairs at 1/2/5 m")
    for s in SEQS:
        for m in METHODS:
            if not vals(s, m, "ate_t_rmse").size:
                continue
            cells, ns = [], []
            for d in ("1", "2", "5"):
                v, n = vals(s, m, f"rpe_t_{d}m"), vals(s, m, f"rpe_n_{d}m")
                nn = int(np.median(n)) if len(n) else 0
                ns.append(str(nn))
                cells.append(f"{np.median(v):8.3f}" if len(v) and nn >= a.min_rpe_pairs else f"{'-':>8s}")
            print(f"  {s:18s} {NAMES[m]:12s} {' '.join(cells)}   {'/'.join(ns)}")

    print("\n== Section 5.3: CPU time (s) and peak memory (MB), median over all runs ==")
    for s in SEQS:
        for m in METHODS:
            c, mem = vals(s, m, "cpu_s", False), vals(s, m, "max_rss_mb", False)
            if len(c) or len(mem):
                print(f"  {s:18s} {NAMES[m]:12s} CPU {np.median(c) if len(c) else np.nan:8.1f} s   "
                      f"memory {np.median(mem) if len(mem) else np.nan:6.0f} MB")

    print("\n== Section 5.2: Mann-Whitney U test on ATE of successful runs (two-sided) ==")
    for s in SEQS:
        x, y = vals(s, "ORB3", "ate_t_rmse"), vals(s, "VINS", "ate_t_rmse")
        if len(x) >= 2 and len(y) >= 2:
            res = mannwhitneyu(x, y, alternative="two-sided")
            print(f"  {s:18s} n = {len(x)} ORB-SLAM3, {len(y)} VINS-Fusion   U = {res.statistic:.1f}   p = {res.pvalue:.4f}")
        else:
            print(f"  {s:18s} not tested (fewer than 2 successful runs for one framework)")


    # ------------------------------------------------------------------------------------------
    print("\n== Indicative only: whole-run error against the fused reference path (gt_full) ==")
    print("  Each successful run is aligned exactly as it was scored (to TankGT), then compared with")
    print("  gt_full (TankGT with gaps filled by AQUA-SLAM, which uses the DVL) over the WHOLE run.")
    print("  gt_full is not independent ground truth away from the structure, so report these as indicative.")
    from evaluate import load_tum, associate, umeyama_se3
    ev = os.path.join(a.results, "eval", "gt")
    for s in SEQS:
        gtp, gfp = os.path.join(ev, f"{s}_gt.txt"), os.path.join(ev, f"{s}_gt_full.txt")
        if not (os.path.exists(gtp) and os.path.exists(gfp)):
            print(f"  {s:18s} no gt_full for this sequence")
            continue
        t_g, p_g, _ = load_tum(gtp)
        t_f, p_f, _ = load_tum(gfp)
        # gt_full is stored in a different frame from TankGT (check_gtfull.py: ~170 deg rotation, 3-4 m shift).
        # Bring it into the TankGT frame with a rigid fit on the poses the two share.
        kg, kf = associate(t_g, t_f, 0.02)
        Rf, tf_ = umeyama_se3(p_f[kf], p_g[kg])
        p_f = p_f @ Rf.T + tf_
        fit = np.median(np.linalg.norm(p_f[kf] - p_g[kg], axis=1))
        ext = np.ptp(p_f, axis=0)
        print(f"  {s:18s} gt_full moved into the TankGT frame ({len(kg)} shared poses, residual median {fit:.3f} m)")
        print(f"  {'':18s} reference path extent x/y/z: {ext[0]:.1f} / {ext[1]:.1f} / {ext[2]:.1f} m")
        for m in METHODS:
            errs, exts = [], []
            for r in g[(s, m)]:
                if r["success"] != "1":
                    continue
                est = os.path.join(a.results, m, s, f"run_{int(r['run']):02d}", "est_cam_tum.txt")
                if not os.path.exists(est):
                    continue
                t_e, p_e, _ = load_tum(est)
                ig, ie = associate(t_g, t_e, 0.02)
                R, t = umeyama_se3(p_e[ie], p_g[ig])                      # same alignment as the score
                p_al = p_e @ R.T + t
                jf, je = associate(t_f, t_e, 0.02)
                if len(jf) < 10:
                    continue
                errs.append(np.sqrt(np.mean(np.sum((p_al[je] - p_f[jf]) ** 2, axis=1))))
                exts.append(np.ptp(p_al, axis=0))
            if errs:
                e = np.median(exts, axis=0)
                print(f"  {'':18s} {NAMES[m]:12s} whole-run RMSE {np.median(errs):6.2f} m "
                      f"[{np.percentile(errs, 25):.2f}-{np.percentile(errs, 75):.2f}]  "
                      f"(n = {len(errs)})   estimate extent x/y/z: {e[0]:.1f} / {e[1]:.1f} / {e[2]:.1f} m")
            else:
                print(f"  {'':18s} {NAMES[m]:12s} no successful runs")


if __name__ == "__main__":
    main()
