#!/usr/bin/env python3
"""
Does gt_full agree with TankGT where both exist? If it does, gt_full is in the same frame and
time base as the scored ground truth, and whole-run errors against it can be reported (as indicative).

Run from ~/Desktop/tank_benchmark after evaluate_all.sh:
    python3 check_gtfull.py
"""
import os
import numpy as np
from evaluate import load_tum, associate, umeyama_se3

EV = "results/accuracy/eval/gt"
for s in ["HalfTank_Easy", "WholeTank_Medium", "WholeTank_Hard"]:
    gp, fp = os.path.join(EV, f"{s}_gt.txt"), os.path.join(EV, f"{s}_gt_full.txt")
    if not (os.path.exists(gp) and os.path.exists(fp)):
        print(f"{s}: missing file"); continue
    tg, pg, _ = load_tum(gp)
    tf, pf, _ = load_tum(fp)
    ig, jf = associate(tg, tf, 0.02)
    d = np.linalg.norm(pg[ig] - pf[jf], axis=1)
    R, t = umeyama_se3(pf[jf], pg[ig])
    d_al = np.linalg.norm(pf[jf] @ R.T + t - pg[ig], axis=1)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    # best time shift between the two (in case the time bases differ)
    best = min(((np.sqrt(np.mean(np.linalg.norm(
        pg[associate(tg, tf + dt, 0.02)[0]] - pf[associate(tg, tf + dt, 0.02)[1]], axis=1) ** 2)), dt)
        for dt in np.arange(-2.0, 2.01, 0.05)), key=lambda x: x[0])
    print(f"{s}: {len(ig)} shared poses ({100 * len(ig) / len(tg):.0f}% of TankGT)")
    print(f"   difference as stored: median {np.median(d):.3f} m, max {d.max():.3f} m")
    print(f"   after best SE(3) fit: median {np.median(d_al):.3f} m (fit rotation {ang:.1f} deg, shift {np.linalg.norm(t):.2f} m)")
    print(f"   best time shift: {best[1]:+.2f} s gives RMSE {best[0]:.3f} m")
