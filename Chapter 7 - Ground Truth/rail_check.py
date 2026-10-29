#!/usr/bin/env python3
"""
rail_check.py - accuracy check of the AprilTag ground truth from the rail run.

The ROV is clamped to a straight rail and stopped every --spacing metres (default 0.5 m).
Because the ROV only translates along the rail, the ground truth should show:

  1. stop-to-stop distances equal to the rail spacing           -> scale / depth accuracy
  2. stop positions on a straight line                          -> lateral accuracy
  3. the same orientation at every stop                         -> rotation accuracy
  4. small scatter while the ROV is stopped                     -> precision (jitter)

None of these depend on where the rail is relative to the structure, or on the unknown
offset between the camera and the point you measured the stop distances to.

Input: gt_frames.csv from tag_gt.py.  Stops are found automatically (stationary periods),
or given explicitly with --windows (CSV: t_start_s,t_end_s per stop, times as in gt_frames.csv).

  python3 rail_check.py --frames gt_rail_run1/gt_frames.csv --spacing 0.5 \
      --nominal 2.5,2.0,1.5,1.0,0.5 --out rail_run1
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation


def load(path, accept=("ok",)):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} is empty")
    t = np.array([int(r["t_ns"]) for r in rows]) * 1e-9
    status = np.array([r["status"] for r in rows])
    ok = np.isin(status, list(accept))
    P = np.full((len(rows), 3), np.nan)
    Q = np.full((len(rows), 4), np.nan)
    rms = np.full(len(rows), np.nan)
    ntag = np.zeros(len(rows), int)
    for i, r in enumerate(rows):
        if r["tx"] != "":
            P[i] = [float(r[k]) for k in ("tx", "ty", "tz")]
            Q[i] = [float(r[k]) for k in ("qx", "qy", "qz", "qw")]
        if r["rms_px"] != "":
            rms[i] = float(r["rms_px"])
        ntag[i] = int(r["n_tags"] or 0)
    return t, status, ok, P, Q, rms, ntag


def find_stops(t, s, speed_thr, min_dur, merge_dist, trim):
    """Stationary segments of the along-rail coordinate s(t) (valid frames only)."""
    n = len(s)
    dt = np.median(np.diff(t)) if n > 1 else 1 / 30
    k = max(1, int(round(0.15 / dt)))                     # +-0.15 s difference window
    s_f = median_filter(s, size=max(3, 2 * k + 1), mode="nearest")
    v = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        v[i] = abs(s_f[b] - s_f[a]) / max(t[b] - t[a], 1e-6)
    still = v < speed_thr
    segs, i = [], 0
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and still[j + 1] and t[j + 1] - t[j] < 1.0:   # a >1 s gap breaks a segment
            j += 1
        segs.append([i, j])
        i = j + 1
    segs = [g for g in segs if t[g[1]] - t[g[0]] >= min_dur]
    merged = []
    for g in segs:                                         # pauses split by a twitch -> one stop
        if merged and abs(np.median(s[g[0]:g[1] + 1]) - np.median(s[merged[-1][0]:merged[-1][1] + 1])) < merge_dist:
            merged[-1][1] = g[1]
        else:
            merged.append(g)
    out = []
    for a, b in merged:                                    # drop settling at each end
        m = int((b - a + 1) * trim)
        out.append((a + m, b - m))
    return out, v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="gt_frames.csv from tag_gt.py")
    ap.add_argument("--spacing", type=float, default=0.5, help="m between stops along the rail")
    ap.add_argument("--nominal", help="stop distances, far to near, e.g. 2.5,2.0,1.5,0.5; if the count matches the "
                    "stops found, spacing errors are computed against these instead of --spacing")
    ap.add_argument("--windows", help="optional CSV t_start_s,t_end_s per stop (overrides auto detection)")
    ap.add_argument("--speed-thr", type=float, default=0.03, help="m/s; below this the ROV counts as stopped")
    ap.add_argument("--min-stop", type=float, default=1.0, help="s; shortest stop accepted")
    ap.add_argument("--trim", type=float, default=0.1, help="fraction of each stop dropped at both ends")
    ap.add_argument("--status", default="ok",
                    help="comma-separated statuses to use, e.g. ok,ambiguous to include frames flagged ambiguous")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t, status, ok, P, Q, rms, ntag = load(args.frames, tuple(args.status.split(",")))
    print(f"using frames with status: {args.status}")
    if ok.sum() < 10:
        sys.exit(f"only {ok.sum()} frames with status ok - fix tag_gt.py output first")
    iv = np.flatnonzero(ok)
    tv, Pv, Qv = t[iv], P[iv], Q[iv]

    # rail axis = principal direction of the valid positions; s increases away from the face
    c = Pv.mean(0)
    _, _, Vt = np.linalg.svd(Pv - c)
    axis = Vt[0]
    if np.corrcoef((Pv - c) @ axis, np.linalg.norm(Pv, axis=1))[0, 1] < 0:
        axis = -axis
    s = (Pv - c) @ axis

    if args.windows:
        with open(args.windows) as f:
            wins = [(float(r[0]), float(r[1])) for r in csv.reader(f) if r and not r[0].startswith("t")]
        stops = []
        for a, b in wins:
            idx = np.flatnonzero((tv >= a) & (tv <= b))
            if len(idx):
                stops.append((idx[0], idx[-1]))
        speed = None
    else:
        stops, speed = find_stops(tv, s, args.speed_thr, args.min_stop, 0.2 * args.spacing, args.trim)
    if len(stops) < 2:
        sys.exit("found fewer than 2 stops - use --windows or adjust --speed-thr / --min-stop")

    # per-stop statistics
    S = []
    for a, b in stops:
        sel = slice(a, b + 1)
        p = Pv[sel]
        R = Rotation.from_quat(Qv[sel])
        Rm = R.mean()
        ang = np.degrees((Rm.inv() * R).magnitude())
        t0, t1 = tv[a], tv[b]
        inwin = (t >= t0) & (t <= t1)
        S.append(dict(t0=t0, t1=t1, n=b - a + 1, p=p.mean(0), sd_mm=1e3 * p.std(0), R=Rm,
                      rot_jit_deg=float(np.sqrt((ang ** 2).mean())),
                      dist_face=float(np.linalg.norm(p.mean(0))), s=float(s[sel].mean()),
                      rms=float(np.nanmedian(rms[iv][sel])), ntag=float(np.median(ntag[iv][sel])),
                      gt_pct=100.0 * ok[inwin].mean()))
    S.sort(key=lambda d: -d["s"])                  # far -> near
    n = len(S)
    M = np.array([d["p"] for d in S])

    # 1. spacing
    nominal = [float(x) for x in args.nominal.split(",")] if args.nominal else None
    use_nom = bool(nominal) and len(nominal) == n
    steps = np.linalg.norm(np.diff(M, axis=0), axis=1)
    step_nom = -np.diff(nominal) if use_nom else np.full(n - 1, args.spacing)
    step_err = 1e3 * (steps - step_nom)
    # along-rail positions vs nominal: slope = scale factor, residuals = position error
    cM = M.mean(0)
    _, _, Vt2 = np.linalg.svd(M - cM)
    a2 = Vt2[0] if (M[0] - M[-1]) @ Vt2[0] > 0 else -Vt2[0]
    u = (M - cM) @ a2                               # along-line coordinate of each stop
    nom = (np.array(nominal) - nominal[0]) if use_nom else -args.spacing * np.arange(n)   # far stop first
    A = np.column_stack([nom, np.ones(n)])
    (scale, off), *_ = np.linalg.lstsq(A, u, rcond=None)
    pos_err_mm = 1e3 * (u - nom - np.mean(u - nom))   # vs exact spacing, best common offset removed
    # 2. straightness: perpendicular distance of stop means from the fitted line
    perp = (M - cM) - np.outer(u, a2)
    perp_mm = 1e3 * np.linalg.norm(perp, axis=1)
    # 3. orientation constancy relative to the mean orientation of all stops
    Rall = Rotation.from_quat(np.array([d["R"].as_quat() for d in S]))
    Rbar = Rall.mean()
    rot_dev = np.degrees((Rbar.inv() * Rall).magnitude())

    lines = []
    P_ = lines.append
    P_(f"Rail check: {args.frames}")
    P_(f"frames {len(t)}, used ({args.status}) {ok.sum()} ({100 * ok.mean():.1f}%), stops found {n}")
    if nominal and len(nominal) != n:
        P_(f"WARNING: --nominal has {len(nominal)} values but {n} stops were found")
    P_("")
    P_(" stop  t [s]          frames  dist to face centre [m]  jitter x/y/z [mm]    rot jitter [deg]  RMS [px]  tags  GT %")
    for k, d in enumerate(S):
        P_(f"  {k:2d}  {d['t0']:7.1f}-{d['t1']:7.1f}  {d['n']:5d}   {d['dist_face']:7.3f}"
           f"{'' if not nominal or k >= len(nominal) else f' ({nominal[k]:.2f})':>9}"
           f"   {d['sd_mm'][0]:5.1f} {d['sd_mm'][1]:5.1f} {d['sd_mm'][2]:5.1f}     {d['rot_jit_deg']:6.3f}"
           f"        {d['rms']:5.2f}    {d['ntag']:3.0f}  {d['gt_pct']:5.1f}")
    P_("")
    P_("1. Spacing " + ("(nominal from --nominal)" if use_nom else f"(nominal {args.spacing:.3f} m)"))
    for k in range(n - 1):
        P_(f"   stop {k}->{k + 1}: {steps[k]:.4f} m  (nominal {step_nom[k]:.3f})   error {step_err[k]:+6.1f} mm")
    P_(f"   scale factor (measured / nominal travel): {scale:.4f}"
       f"   -> {1e2 * (scale - 1):+.2f}% ; >1-2% points to wrong intrinsics or tag size")
    P_(f"   stop position error along rail, best common offset removed: "
       + " ".join(f"{e:+.1f}" for e in pos_err_mm) + f" mm  (RMS {np.sqrt((pos_err_mm ** 2).mean()):.1f} mm)")
    P_(f"2. Straightness: distance of stop means from fitted line: "
       + " ".join(f"{e:.1f}" for e in perp_mm) + f" mm  (RMS {np.sqrt((perp_mm ** 2).mean()):.1f} mm)")
    P_(f"3. Orientation: deviation of each stop from the mean orientation: "
       + " ".join(f"{e:.2f}" for e in rot_dev) + " deg")
    P_(f"4. Precision: median jitter {np.median([np.linalg.norm(d['sd_mm']) for d in S]):.1f} mm (3D), "
       f"{np.median([d['rot_jit_deg'] for d in S]):.3f} deg")
    if nominal and len(nominal) == n:
        off_ = np.array([d["dist_face"] for d in S]) - np.array(nominal)
        P_(f"   (info) camera-to-face distance minus nominal: " + " ".join(f"{1e3 * o:+.0f}" for o in off_) +
           " mm. A constant offset is just the camera sitting behind/in front of your measuring point;"
           " only the variation between stops is an error (covered by 1.)")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out, "rail_report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, "rail_stops.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stop", "t_start_s", "t_end_s", "frames", "x", "y", "z", "dist_face_m", "jit_x_mm", "jit_y_mm",
                    "jit_z_mm", "rot_jit_deg", "rms_px", "tags", "gt_pct", "pos_err_mm", "perp_mm", "rot_dev_deg"])
        for k, d in enumerate(S):
            w.writerow([k, f"{d['t0']:.3f}", f"{d['t1']:.3f}", d["n"], *np.round(d["p"], 5), round(d["dist_face"], 4),
                        *np.round(d["sd_mm"], 2), round(d["rot_jit_deg"], 4), round(d["rms"], 3), d["ntag"],
                        round(d["gt_pct"], 1), round(pos_err_mm[k], 2), round(perp_mm[k], 2), round(rot_dev[k], 3)])

    # ------------------------------------------------------------------ figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping figure")
        return
    BLUE, GREY, INK, INK2, SURF, GRID = "#2a78d6", "#b5b4ad", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), facecolor=SURF)
    for a in ax:
        a.set_facecolor(SURF)
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)

    # A: along-rail position over time, stops highlighted
    a = ax[0]
    instop = np.zeros(len(tv), bool)
    for d in S:
        instop |= (tv >= d["t0"]) & (tv <= d["t1"])
    dv = np.linalg.norm(Pv, axis=1)                # camera distance to the face centre
    a.plot(tv - t[0], dv, color=GREY, lw=1.0, label="all frames used")
    a.scatter(tv[instop] - t[0], dv[instop], s=4, color=BLUE, label="frames used for each stop", zorder=3)
    for k, d in enumerate(S):
        lab = f"{d['dist_face']:.2f} m" + (f"\n(nominal {nominal[k]:g} m)" if use_nom else "")
        a.annotate(lab, xy=((d["t0"] + d["t1"]) / 2 - t[0], d["dist_face"]), xytext=(0, 8),
                   textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK2)
    a.set_ylim(0, max(dv.max(), 0.5) + 0.45)
    a.set_xlabel("time since start [s]")
    a.set_ylabel("camera distance to face [m]")
    a.set_title("A. Stops detected in the rail run")
    a.legend(frameon=False, loc="upper right")

    # B: stop position error along the rail (best common offset removed), with jitter
    a = ax[1]
    xs = np.array([d["dist_face"] for d in S])
    sd = np.array([np.linalg.norm(d["sd_mm"]) for d in S])
    a.axhline(0, color=INK2, lw=0.8)
    a.errorbar(xs, pos_err_mm, yerr=sd, fmt="o", color=BLUE, ms=6, elinewidth=1.5, capsize=0)
    for x, e in zip(xs, pos_err_mm):
        a.annotate(f"{e:+.1f}", (x, e), textcoords="offset points", xytext=(8, -3), color=INK, fontsize=8)
    a.set_xlabel("camera distance to face [m]")
    a.set_ylabel("stop position error [mm]")
    a.set_title("B. Error of each stop along the rail vs nominal position (bars: 3D jitter, 1 sd)")
    a.invert_xaxis()

    # C: reprojection error vs distance
    a = ax[2]
    a.scatter(np.linalg.norm(Pv, axis=1), rms[iv], s=4, color=BLUE)
    a.set_xlabel("camera distance to face [m]")
    a.set_ylabel("RMS reprojection error [px]")
    a.set_title("C. Reprojection error of every accepted frame")
    a.invert_xaxis()
    a.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "rail_check.png"), dpi=150, facecolor=SURF)
    print(f"\nwrote {args.out}/rail_report.txt, rail_stops.csv, rail_check.png")


if __name__ == "__main__":
    main()
