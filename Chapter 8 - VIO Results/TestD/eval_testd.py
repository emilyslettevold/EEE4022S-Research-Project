#!/usr/bin/env python3
"""
eval_testd.py - Test D: score every VINS-Fusion run against the AprilTag ground truth, and plot the trajectories.

Usage (Lenovo, evo-env active):
  python3 eval_testd.py --runs ~/vio_data/testD --out ~/vio_data/testD_eval \
      --config SO=~/vio_data/config_rov/testD_SO.yaml --config SI=~/vio_data/config_rov/testD_SI.yaml \
      --gt fig8_3=~/Desktop/ground_truth/VIO_valid/gt_figure_8_3/gt_tum.txt \
      --gt square2=<path to square2 gt_tum.txt>

  --runs    folder with <bag>_<tag>.csv files written by run_vins_testA.sh (e.g. fig8_3_SO.csv)
  --config  TAG=config.yaml : the VINS config each tag ran with (its body_T_cam0 converts IMU pose -> left camera)
  --gt      BAG=gt_tum.txt  : ground truth (TUM: t[s] x y z qx qy qz qw, left camera in the structure frame)
            bags without --gt are still plotted and get path length / start-to-end only
  --t-max-diff   max time difference when matching VIO to GT poses (default 0.02 s)
  --rpe-delta    distance for the relative pose error (default 1.0 m)

For each run it:
  1. converts the VINS body (IMU) pose to the left-camera pose:  T_W_cam0 = T_W_body * body_T_cam0
  2. matches VIO and GT poses by timestamp
  3. aligns VIO to GT with a rigid SE(3) fit (Umeyama, NO scale - scale error is part of what is measured)
  4. ATE: position error after alignment (RMSE, mean, median, max) + rotation error (RMSE, deg)
  5. RPE: error of the relative motion over every --rpe-delta metres of GT path (translation RMSE in m and as %
     of delta; rotation RMSE in deg)
  6. info: Sim(3) scale factor (how much bigger/smaller the VIO trajectory is), start-to-end and path length vs GT
Writes testD_results.csv, testD_results.md and three PNGs per recording: <rec>_plan.png, <rec>_side.png, <rec>_ate.png.
Standard library + numpy + scipy + matplotlib + pandas (all in evo-env).
"""
import argparse, csv, glob, math, os, re, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"GT": "#222222", "SO": "#1f77b4", "SI": "#d62728"}
FULL = {"GT": "Ground truth", "SO": "Stereo only", "SI": "Stereo-inertial"}


# ---------------------------------------------------------------- loading
def body_T_cam0(path):
    """Read body_T_cam0 from an OpenCV yaml without needing cv2."""
    txt = Path(path).expanduser().read_text()
    m = re.search(r"body_T_cam0:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]", txt, re.S)
    if not m:
        sys.exit(f"body_T_cam0 not found in {path}")
    v = [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    if len(v) != 16:
        sys.exit(f"body_T_cam0 in {path} has {len(v)} numbers, expected 16")
    return np.array(v).reshape(4, 4)


def load_vins(path, B_T_C):
    """VINS csv: t[ns], px,py,pz, qw,qx,qy,qz, vx,vy,vz  ->  (t[s], T_W_cam0 4x4 array)"""
    t, T = [], []
    for row in csv.reader(open(path)):
        if len(row) < 8 or not row[0].strip():
            continue
        try:
            v = [float(x) for x in row[:8]]
        except ValueError:
            continue
        TWB = np.eye(4)
        TWB[:3, :3] = R.from_quat([v[5], v[6], v[7], v[4]]).as_matrix()   # csv is w,x,y,z
        TWB[:3, 3] = v[1:4]
        t.append(v[0] / 1e9 if v[0] > 1e12 else v[0])
        T.append(TWB @ B_T_C)
    return np.array(t), np.array(T)


def load_tum(path):
    t, T = [], []
    for line in open(Path(path).expanduser()):
        if not line.strip() or line.startswith("#"):
            continue
        v = [float(x) for x in line.split()[:8]]
        M = np.eye(4)
        M[:3, :3] = R.from_quat(v[4:8]).as_matrix()
        M[:3, 3] = v[1:4]
        t.append(v[0]); T.append(M)
    return np.array(t), np.array(T)


# ---------------------------------------------------------------- maths
def associate(t_est, t_gt, max_dt):
    """index pairs (i_est, i_gt) of nearest timestamps within max_dt"""
    j = np.searchsorted(t_gt, t_est)
    pairs = []
    for i, k in enumerate(j):
        best = None
        for kk in (k - 1, k):
            if 0 <= kk < len(t_gt):
                d = abs(t_gt[kk] - t_est[i])
                if d <= max_dt and (best is None or d < best[1]):
                    best = (kk, d)
        if best:
            pairs.append((i, best[0]))
    return np.array(pairs, dtype=int).reshape(-1, 2)


def umeyama(src, dst, with_scale=False):
    """R, t, s minimising |dst - (s R src + t)|"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    Rm = U @ S @ Vt
    s = (D * np.diag(S)).sum() / xs.var(0).sum() if with_scale else 1.0
    return Rm, mu_d - s * Rm @ mu_s, s


def path_length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def rpe(T_gt, T_est, delta):
    """relative pose error over every `delta` metres of GT path"""
    P = T_gt[:, :3, 3]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    et, er = [], []
    for i in range(len(P)):
        j = np.searchsorted(s, s[i] + delta)
        if j >= len(P):
            break
        Dg = np.linalg.inv(T_gt[i]) @ T_gt[j]
        De = np.linalg.inv(T_est[i]) @ T_est[j]
        E = np.linalg.inv(Dg) @ De
        et.append(np.linalg.norm(E[:3, 3]))
        er.append(np.degrees(np.linalg.norm(R.from_matrix(E[:3, :3]).as_rotvec())))
    if not et:
        return None, None
    return float(np.sqrt(np.mean(np.square(et)))), float(np.sqrt(np.mean(np.square(er))))


# ---------------------------------------------------------------- main
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--config", action="append", default=[], help="TAG=config.yaml")
ap.add_argument("--gt", action="append", default=[], help="BAG=gt_tum.txt")
ap.add_argument("--t-max-diff", type=float, default=0.02)
ap.add_argument("--rpe-delta", type=float, default=1.0)
ap.add_argument("--summary", help="master CSV for all recordings: this run's rows replace any earlier rows for the same recording")
a = ap.parse_args()

cfg = {k: body_T_cam0(v) for k, v in (s.split("=", 1) for s in a.config)}
gts = {k: str(Path(v).expanduser()) for k, v in (s.split("=", 1) for s in a.gt)}
out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

runs = {}
for f in sorted(glob.glob(str(Path(a.runs).expanduser() / "*.csv"))):
    name = Path(f).stem
    if name.endswith("_metrics") or "_" not in name:
        continue
    bag, tag = name.rsplit("_", 1)
    if tag not in cfg:
        print(f"skip {name}: no --config for tag '{tag}'")
        continue
    runs.setdefault(bag, {})[tag] = f
if not runs:
    sys.exit("no <bag>_<tag>.csv runs found for the given --config tags")

rows = []
for bag, tagfiles in runs.items():
    gt = load_tum(gts[bag]) if bag in gts else None
    plot = {}
    if gt is not None:
        plot["GT"] = (gt[0], gt[1][:, :3, 3])
    errs = {}
    for tag, f in sorted(tagfiles.items()):
        t, T = load_vins(f, cfg[tag])
        P = T[:, :3, 3]
        # speed over ~0.5 s windows (per-frame speed is dominated by jitter)
        k = max(1, int(round(0.5 / np.median(np.diff(t))))) if len(t) > 2 else 1
        speed = np.linalg.norm(P[k:] - P[:-k], axis=1) / np.maximum(t[k:] - t[:-k], 1e-6) if len(t) > k else np.array([])
        r = {"recording": bag, "config": tag, "vio_poses": len(t),
             "vio_path_m": round(path_length(P), 3),
             "vio_start_end_m": round(float(np.linalg.norm(P[-1] - P[0])), 3),
             "max_speed_ms": round(float(np.nanmax(speed)), 2) if len(speed) else ""}
        r["diverged"] = "yes" if (len(speed) and np.nanmax(speed) > 3.0) or r["vio_path_m"] > 1000 else "no"   # ROV never exceeds ~1 m/s
        if gt is None:
            plot[tag] = (t, P)
            rows.append(r)
            continue
        tg, Tg = gt
        m = associate(t, tg, a.t_max_diff)
        r["matched"] = len(m)
        r["gt_poses"] = len(tg)
        if len(m) < 10:
            print(f"{bag}_{tag}: only {len(m)} poses matched GT - check the clocks (check_overlap.py)")
            plot[tag] = (t, P); rows.append(r); continue
        Te, Tgm = T[m[:, 0]], Tg[m[:, 1]]
        Rm, tv, _ = umeyama(Te[:, :3, 3], Tgm[:, :3, 3])
        A = np.eye(4); A[:3, :3] = Rm; A[:3, 3] = tv
        Ta = np.einsum("ij,njk->nik", A, Te)                      # aligned VIO camera poses
        e = np.linalg.norm(Ta[:, :3, 3] - Tgm[:, :3, 3], axis=1)
        rot = [np.degrees(np.linalg.norm(R.from_matrix(Tgm[k, :3, :3].T @ Ta[k, :3, :3]).as_rotvec())) for k in range(len(m))]
        _, _, s_sim = umeyama(Te[:, :3, 3], Tgm[:, :3, 3], with_scale=True)
        rpe_t, rpe_r = rpe(Tgm, Ta, a.rpe_delta)
        Pg = Tgm[:, :3, 3]
        r.update({
            "ate_rmse_m": round(float(np.sqrt(np.mean(e ** 2))), 4), "ate_mean_m": round(float(e.mean()), 4),
            "ate_median_m": round(float(np.median(e)), 4), "ate_max_m": round(float(e.max()), 4),
            "rot_rmse_deg": round(float(np.sqrt(np.mean(np.square(rot)))), 2),
            f"rpe_{a.rpe_delta:g}m_m": round(rpe_t, 4) if rpe_t is not None else "",
            "rpe_pct": round(100 * rpe_t / a.rpe_delta, 2) if rpe_t is not None else "",
            "rpe_rot_deg": round(rpe_r, 2) if rpe_r is not None else "",
            "sim3_scale": round(float(s_sim), 3),
            "gt_start_end_m": round(float(np.linalg.norm(Pg[-1] - Pg[0])), 3),
            "vio_start_end_matched_m": round(float(np.linalg.norm(Ta[-1, :3, 3] - Ta[0, :3, 3])), 3),
            "gt_path_matched_m": round(path_length(Pg), 3),
            "vio_path_matched_m": round(path_length(Ta[:, :3, 3]), 3),
        })
        rows.append(r)
        # whole VIO trajectory in the GT frame, for the plot
        Pall = (A[:3, :3] @ P.T).T + A[:3, 3]
        plot[tag] = (t, Pall)
        errs[tag] = (t[m[:, 0]], e)

    # ---- figures: plan view, side view, error over time - three separate PNGs
    ref = plot["GT"][1] if "GT" in plot else next(iter(plot.values()))[1]
    a1, a2, a3 = np.argsort(np.ptp(ref, axis=0))[::-1]              # plan view = the two axes with most spread
    lab = "xyz"

    def styled(name):
        diverged = any(r["recording"] == bag and r["config"] == name and r["diverged"] == "yes" for r in rows)
        return dict(color=COLORS.get(name), ls="--" if name == "GT" else "-", lw=2.2 if name == "GT" else 1.6,
                    zorder=5 if name == "GT" else 2,
                    label=FULL.get(name, name) + (" (diverged)" if diverged else "")), diverged

    def view(ax_i, ax_j, title, fname):
        fig, ax = plt.subplots(figsize=(8, 7))
        for name in sorted(plot, key=lambda k: k != "GT"):
            tt, PP = plot[name]
            st, diverged = styled(name)
            if diverged:     # keep the axes on the GT scale
                PP = PP[np.all(np.abs(PP - ref.mean(0)) < 3 * np.ptp(ref, axis=0).max() + 1, axis=1)]
            ax.plot(PP[:, ax_i], PP[:, ax_j], **st)
            if len(PP):
                ax.plot(PP[0, ax_i], PP[0, ax_j], "o", color=st["color"], ms=8, zorder=6)
        ax.set_xlabel(f"{lab[ax_i]} (m)", fontsize=14); ax.set_ylabel(f"{lab[ax_j]} (m)", fontsize=14)
        ax.set_title(f"{bag}: {title}", fontsize=15); ax.tick_params(labelsize=12)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)
        ax.legend(fontsize=14, markerscale=1.5, handlelength=3, loc="best", framealpha=0.9)
        fig.tight_layout(); fig.savefig(out / fname, dpi=150); plt.close(fig)
        print(f"plot: {out / fname}")

    view(a1, a2, "plan view (start marked ●)", f"{bag}_plan.png")
    view(a1, a3, "side view", f"{bag}_side.png")
    if errs:
        fig, ax = plt.subplots(figsize=(10, 5))
        t0 = plot["GT"][0][0]
        for name, (tt, e) in errs.items():
            ax.plot(tt - t0, e, color=COLORS.get(name), lw=1.3, label=FULL.get(name, name))
        ax.set_xlabel("time (s)", fontsize=14); ax.set_ylabel("position error after alignment (m)", fontsize=14)
        ax.set_title(f"{bag}: absolute trajectory error over time", fontsize=15); ax.tick_params(labelsize=12)
        ax.grid(alpha=0.3); ax.legend(fontsize=14, handlelength=3, framealpha=0.9)
        fig.tight_layout(); fig.savefig(out / f"{bag}_ate.png", dpi=150); plt.close(fig)
        print(f"plot: {out / (bag + '_ate.png')}")

cols = ["recording", "config", "diverged", "ate_rmse_m", "ate_median_m", "ate_max_m", "rot_rmse_deg",
        f"rpe_{a.rpe_delta:g}m_m", "rpe_pct", "rpe_rot_deg", "sim3_scale", "gt_start_end_m",
        "vio_start_end_matched_m", "gt_path_matched_m", "vio_path_matched_m", "matched", "gt_poses",
        "vio_poses", "vio_path_m", "vio_start_end_m", "max_speed_ms"]
with open(out / "testD_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    w.writerows({k: r.get(k, "") for k in cols} for r in rows)
short = ["recording", "config", "diverged", "ate_rmse_m", "ate_max_m", "rot_rmse_deg", "rpe_pct", "sim3_scale",
         "gt_start_end_m", "vio_start_end_matched_m", "matched"]
md = ["| " + " | ".join(short) + " |", "|" + "---|" * len(short)]
md += ["| " + " | ".join(str(r.get(k, "")) if r.get(k, "") != "" else "—" for k in short) + " |" for r in rows]
(out / "testD_results.md").write_text("\n".join(md) + "\n")
wid = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in short}
print("\n" + "  ".join(k.rjust(wid[k]) for k in short))
for r in rows:
    print("  ".join(str(r.get(k, "")).rjust(wid[k]) for k in short))
print(f"\nwrote {out / 'testD_results.csv'} and testD_results.md")

if a.summary:   # one table for every recording: replace this recording's rows, keep the others
    sp = Path(a.summary).expanduser(); sp.parent.mkdir(parents=True, exist_ok=True)
    done = {r["recording"] for r in rows}
    old = [r for r in csv.DictReader(open(sp))] if sp.exists() else []
    allrows = [r for r in old if r.get("recording") not in done] + [{k: r.get(k, "") for k in cols} for r in rows]
    allrows.sort(key=lambda r: (r["recording"], r["config"]))
    with open(sp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(allrows)
    md = ["| " + " | ".join(short) + " |", "|" + "---|" * len(short)]
    md += ["| " + " | ".join(str(r.get(k, "")) if str(r.get(k, "")) != "" else "\u2014" for k in short) + " |" for r in allrows]
    sp.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"updated summary {sp} ({len(allrows)} rows, {len({r['recording'] for r in allrows})} recordings)")
