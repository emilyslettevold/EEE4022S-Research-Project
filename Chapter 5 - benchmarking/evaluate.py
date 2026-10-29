#!/usr/bin/env python3
"""
Evaluate one run, or summarise many.

One run (appends a row to runs.csv):
    python3 evaluate.py run --est est_tum.txt --gt gt_tum.txt --gt-full gt_full_tum.txt \
        --method ORB3 --seq Structure_Easy --run 1 --out-csv results/runs.csv \
        --log log.txt --time time.txt --plot results/plots/ORB3_SE_run01

Summary table over all runs:
    python3 evaluate.py summarise --runs-csv results/runs.csv --out results/summary.csv

Design choices (why):
  * GT gaps are NOT interpolated. A GT pose is only matched to an estimate within --max-dt
    (default 20 ms, i.e. under half a 20 Hz frame), so HalfTank/WholeTank errors are measured
    only where TankGT actually observed tags.
  * SE(3) alignment (rotation + translation, no scale): stereo-inertial systems are metric,
    and this matches the Tank paper (eq. 13).
  * Translation AND rotation ATE (paper eqs. 14-15).
  * Coverage = fraction of GT poses that received an estimate. It exposes initialisation
    delays and tracking loss (ORB-SLAM3 only saves its largest map; VINS only outputs after
    initialisation), which ATE alone hides.
  * RPE over travelled distance, only inside continuous GT segments (use on Structure_*).
"""
import argparse
import csv
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as Rot

SEQ_ORDER = ["Structure_Easy", "Structure_Medium", "Structure_Hard", "HalfTank_Easy",
             "HalfTank_Medium", "HalfTank_Hard", "WholeTank_Medium", "WholeTank_Hard"]
LOG_PATTERNS = {
    "ORB3": {"track_fail": "Fail to track local map", "new_map": "Stored map with ID",   # printed once per EXTRA map (tracking lost -> new map)
             "loops": "Loop detected", "resets": "Active map reset",
             "no_motion_resets": "Not enough motion for initializing",
             "accel_gate_waits": "not enough acceleration",
             "fast_init_on": "Fast IMU initialization"},
    # VINS-Fusion's loop closure lives in a separate node; run_vins.sh checks it isn't running.
    "VINS": {"track_fail": "failure detection", "new_map": "system reboot"},
}


def load_tum(path):
    d = np.loadtxt(path, comments="#", ndmin=2)
    return d[:, 0], d[:, 1:4], Rot.from_quat(d[:, 4:8])


def associate(t_gt, t_est, max_dt):
    """For every GT stamp, the nearest estimate within max_dt. Returns index arrays."""
    j = np.clip(np.searchsorted(t_est, t_gt), 1, len(t_est) - 1)
    j -= (t_gt - t_est[j - 1]) < (t_est[j] - t_gt)
    ok = np.abs(t_est[j] - t_gt) <= max_dt
    return np.nonzero(ok)[0], j[ok]


def umeyama_se3(src, dst):
    """R, t minimising sum ||R src + t - dst||^2 (no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    H = (src - ms).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    return R, md - R @ ms


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def rpe_by_distance(t, p_gt, R_gt, p_est, R_est, delta, max_gap):
    """Relative pose error over 'delta' metres of GT travel, inside gap-free segments."""
    seg = np.concatenate([[0], np.cumsum(np.diff(t) > max_gap)])
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))])
    et, er = [], []
    for s in np.unique(seg):
        idx = np.nonzero(seg == s)[0]
        dd = dist[idx]
        for a_i, i in enumerate(idx):
            k = np.searchsorted(dd, dd[a_i] + delta)
            if k >= len(idx):
                break
            j = idx[k]
            dR_gt = R_gt[i].inv() * R_gt[j]
            dp_gt = R_gt[i].inv().apply(p_gt[j] - p_gt[i])
            dR_est = R_est[i].inv() * R_est[j]
            dp_est = R_est[i].inv().apply(p_est[j] - p_est[i])
            E_R = dR_gt.inv() * dR_est
            E_t = dR_gt.inv().apply(dp_est - dp_gt)
            et.append(np.linalg.norm(E_t))
            er.append(np.degrees(np.linalg.norm(E_R.as_rotvec())))
    return rmse(et), rmse(er), len(et)


def parse_time_file(path):
    out = {"wall_s": np.nan, "cpu_s": np.nan, "max_rss_mb": np.nan}
    if not path or not os.path.exists(path):
        return out
    txt = open(path).read()
    m = re.search(r"Elapsed \(wall clock\) time.*: (?:(\d+):)?(\d+):([\d.]+)", txt)
    if m:
        h = float(m.group(1) or 0)
        out["wall_s"] = h * 3600 + float(m.group(2)) * 60 + float(m.group(3))
    u = re.search(r"User time \(seconds\): ([\d.]+)", txt)
    s = re.search(r"System time \(seconds\): ([\d.]+)", txt)
    if u and s:
        out["cpu_s"] = float(u.group(1)) + float(s.group(1))
    r = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
    if r:
        out["max_rss_mb"] = int(r.group(1)) / 1024
    return out


def count_log(path, method):
    out = {k: np.nan for k in ("track_fail", "new_map", "loops", "resets", "no_motion_resets",
                               "accel_gate_waits", "fast_init_on", "exit_code")}
    if not path or not os.path.exists(path) or method not in LOG_PATTERNS:
        return out
    txt = open(path, errors="ignore").read()
    for k, pat in LOG_PATTERNS[method].items():
        out[k] = len(re.findall(re.escape(pat), txt, flags=re.IGNORECASE))
    m = re.findall(r"exit code (\d+)", txt)               # written by run_orbslam3.sh; 139 = segfault
    out["exit_code"] = int(m[-1]) if m else np.nan
    return out


def cmd_run(a):
    t_gt, p_gt, R_gt = load_tum(a.gt)
    try:
        t_est, p_est, R_est = load_tum(a.est)
    except (OSError, ValueError, IndexError):                        # crashed run / empty output
        t_est = np.array([])
    if len(t_est) >= 2:
        ig, ie = associate(t_gt, t_est, a.max_dt)
    else:
        ig = ie = np.array([], dtype=int)
    row = {"method": a.method, "sequence": a.seq, "run": a.run,
           "n_gt": len(t_gt), "n_est": len(t_est), "n_matched": len(ig),
           "coverage": len(ig) / len(t_gt)}

    if a.gt_full and os.path.exists(a.gt_full) and os.path.getsize(a.gt_full) > 100:
        pf = load_tum(a.gt_full)[1]
        path_len = float(np.linalg.norm(np.diff(pf, axis=0), axis=1).sum())
        row["path_len_source"] = "gt_full"
    else:
        # no GT_full in the bag: sum GT steps, bridging each gap with a straight line.
        # This is a LOWER bound on the true path, so the success threshold is conservative.
        path_len = float(np.linalg.norm(np.diff(p_gt, axis=0), axis=1).sum())
        row["path_len_source"] = "gt_with_gap_chords"
    row["path_len_m"] = path_len

    if len(ig) >= 10:
        Rg, tg = R_gt[ig], t_gt[ig]
        R, t = umeyama_se3(p_est[ie], p_gt[ig])
        p_al = p_est[ie] @ R.T + t
        R_al = Rot.from_matrix(R) * R_est[ie]
        e_t = np.linalg.norm(p_al - p_gt[ig], axis=1)
        e_r = np.degrees(np.linalg.norm((Rg.inv() * R_al).as_rotvec(), axis=1))
        row.update(ate_t_rmse=rmse(e_t), ate_t_mean=float(e_t.mean()), ate_t_median=float(np.median(e_t)),
                   ate_t_max=float(e_t.max()), ate_r_rmse_deg=rmse(e_r), ate_r_max_deg=float(e_r.max()))
        for d in a.rpe_deltas:
            rt, rr, n = rpe_by_distance(tg, p_gt[ig], Rg, p_est[ie], R_est[ie], d, a.max_gap)
            row[f"rpe_t_{d:g}m"], row[f"rpe_r_deg_{d:g}m"], row[f"rpe_n_{d:g}m"] = rt, rr, n
        if a.plot:
            plot(a.plot, tg, e_t, p_gt, p_al, ig, a.method, a.seq, a.max_gap)
    else:
        row.update(ate_t_rmse=np.nan, ate_t_mean=np.nan, ate_t_median=np.nan, ate_t_max=np.nan,
                   ate_r_rmse_deg=np.nan, ate_r_max_deg=np.nan)
        for d in a.rpe_deltas:
            row[f"rpe_t_{d:g}m"] = row[f"rpe_r_deg_{d:g}m"] = row[f"rpe_n_{d:g}m"] = np.nan

    row["success"] = int(row["coverage"] >= a.cov_min and
                         np.isfinite(row["ate_t_rmse"]) and row["ate_t_rmse"] <= a.drift_max * path_len)
    row.update(count_log(a.log, a.method))
    row.update(parse_time_file(a.time))

    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    new = not os.path.exists(a.out_csv)
    fields = list(row.keys())
    if not new:
        with open(a.out_csv) as fh:
            fields = next(csv.reader(fh))
        for k in row:
            if k not in fields:
                raise SystemExit(f"column {k} not in existing {a.out_csv}; use the same --rpe-deltas")
    with open(a.out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"{a.method} {a.seq} run {a.run}: ATE {row['ate_t_rmse']:.3f} m / "
          f"{row['ate_r_rmse_deg']:.2f} deg, coverage {100 * row['coverage']:.1f}%, success {row['success']}")


def plot(prefix, tg, e_t, p_gt, p_al, ig, method, seq, max_gap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    tt = tg - tg[0]
    brk = np.nonzero(np.diff(tt) > max_gap)[0] + 1                   # break the line at GT gaps
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.insert(tt, brk, np.nan), np.insert(e_t, brk, np.nan))
    ax.set(xlabel="Time [s]", ylabel="Position error [m]", title=f"{seq}: {method} ATE (gaps = no GT)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(prefix + "_ate_vs_time.png", dpi=200)
    plt.close(fig)
    g = p_gt[ig]
    ax_i = np.argsort(np.ptp(g, axis=0))[::-1][:2]                   # two widest axes = top-down view
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(g[:, ax_i[0]], g[:, ax_i[1]], ".", ms=2, label="GT (tag-visible)")
    ax.plot(p_al[:, ax_i[0]], p_al[:, ax_i[1]], ".", ms=2, label=method)
    ax.set_aspect("equal")
    ax.set(xlabel=f"axis {'xyz'[ax_i[0]]} [m]", ylabel=f"axis {'xyz'[ax_i[1]]} [m]", title=seq)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(prefix + "_topdown.png", dpi=200)
    plt.close(fig)


def cmd_summarise(a):
    with open(a.runs_csv) as fh:
        rows = list(csv.DictReader(fh))
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["sequence"]), []).append(r)

    def col(rs, k, only_ok=False):
        v = np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan for r in rs
                      if (not only_ok or r["success"] == "1")])
        return v[np.isfinite(v)]

    def med_iqr(v):
        if not len(v):
            return "n/a"
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        return f"{q2:.3f} [{q1:.3f}-{q3:.3f}]"

    out = []
    for (m, s), rs in sorted(groups.items(), key=lambda kv: (SEQ_ORDER.index(kv[0][1])
                                                             if kv[0][1] in SEQ_ORDER else 99, kv[0][0])):
        out.append({
            "sequence": s, "method": m, "runs": len(rs),
            "success_rate": f"{np.mean([r['success'] == '1' for r in rs]):.2f}",
            "ATE_t_m_median[IQR]_successful": med_iqr(col(rs, "ate_t_rmse", True)),
            "ATE_r_deg_median[IQR]_successful": med_iqr(col(rs, "ate_r_rmse_deg", True)),
            "ATE_t_m_median_all_runs": f"{np.median(col(rs, 'ate_t_rmse')):.3f}" if len(col(rs, 'ate_t_rmse')) else "n/a",
            "coverage_median": f"{np.median(col(rs, 'coverage')):.3f}",
            "track_fail_median": f"{np.median(col(rs, 'track_fail')):.0f}" if len(col(rs, 'track_fail')) else "n/a",
            "loops_detected_max": f"{col(rs, 'loops').max():.0f}" if len(col(rs, 'loops')) else "n/a",
            "map_resets_median": f"{np.median(col(rs, 'resets')):.0f}" if len(col(rs, 'resets')) else "n/a",
            "crashes": f"{int(sum(1 for r in rs if r.get('exit_code') not in (None, '', 'nan') and float(r['exit_code']) not in (0.0,)))}"
                       if any(r.get('exit_code') not in (None, '', 'nan') for r in rs) else "n/a",
            "wall_s_median": f"{np.median(col(rs, 'wall_s')):.1f}" if len(col(rs, 'wall_s')) else "n/a",
            "max_rss_mb_median": f"{np.median(col(rs, 'max_rss_mb')):.0f}" if len(col(rs, 'max_rss_mb')) else "n/a",
        })
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    keys = list(out[0].keys())
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in out:
        print("| " + " | ".join(str(r[k]) for k in keys) + " |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--est", required=True)
    r.add_argument("--gt", required=True)
    r.add_argument("--gt-full")
    r.add_argument("--method", required=True, help="ORB3 or VINS")
    r.add_argument("--seq", required=True)
    r.add_argument("--run", type=int, required=True)
    r.add_argument("--out-csv", required=True)
    r.add_argument("--log")
    r.add_argument("--time")
    r.add_argument("--plot")
    r.add_argument("--max-dt", type=float, default=0.02)
    r.add_argument("--max-gap", type=float, default=0.5, help="GT gap that breaks RPE segments/plots [s]")
    r.add_argument("--rpe-deltas", type=float, nargs="*", default=[1.0, 2.0, 5.0])
    r.add_argument("--cov-min", type=float, default=0.8)
    r.add_argument("--drift-max", type=float, default=0.10, help="fail if ATE > this fraction of path length")
    s = sub.add_parser("summarise")
    s.add_argument("--runs-csv", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_summarise(a)


if __name__ == "__main__":
    main()
