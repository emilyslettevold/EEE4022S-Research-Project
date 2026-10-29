#!/usr/bin/env python3
"""
si_divergence.py - WHERE and WHY did the stereo-inertial (SI) run diverge?  Reads existing files only.

  source ~/evo-env/bin/activate
  python3 ~/vio_data/si_divergence.py square2
  python3 ~/vio_data/si_divergence.py rail170
  python3 ~/vio_data/si_divergence.py accuracyRun1 --imu ~/vio_data/accuracyRun1/dataset_half/imu0.csv
  python3 ~/vio_data/si_divergence.py lawnmower2 --onset 95      # force the time to study (s from first frame)

Files read
  ~/vio_data/testD/NAME/NAME_SI.csv          SI poses + velocity      (VINS output)
  ~/vio_data/testD/NAME/NAME_SI_metrics.csv  SI internals per frame  (metrics patch: n_feat, n_stereo, n_win,
                                             n_depth_guess, n_outliers, cost, iters, ba_norm, bg_norm, td_ms, ...)
  ~/vio_data/testD/NAME/NAME_SO.csv          stereo-only run: its speed and turn rate are the reference for
                                             what the ROV was really doing (frame-independent, so no alignment)
  ~/vio_data/NAME/dataset_half/imu0.csv      raw IMU: how much the accelerometer was actually excited
                                             (override with --imu)

Divergence onset = first time the SI speed (0.5 s window) exceeds 1 m/s and stays above it for 1 s
(the ROV never moves faster than about 1 m/s). If SI never diverges, the time of its largest error growth
is not knowable without ground truth, so the script reports the whole run and you can pass --onset.

Outputs (in ~/vio_data/testD_eval/NAME/)
  NAME_SI_divergence.png   six panels on one time axis, onset marked, the 10 s before it shaded:
                           1 speed: SI vs SO     2 raw IMU excitation     3 SI bias estimates
                           4 SI time offset td   5 features / stereo / outliers     6 solver cost and iterations
  NAME_SI_divergence.txt   the printed report: each quantity early in the run vs the 10 s before onset,
                           what the ROV was doing at onset, and which warning signs appeared first
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("name")
ap.add_argument("--imu", help="imu0.csv (default ~/vio_data/NAME/dataset_half/imu0.csv)")
ap.add_argument("--onset", type=float, help="study this time instead of the detected onset (s from first frame)")
ap.add_argument("--pre", type=float, default=10.0, help="seconds before onset to examine (default 10)")
a = ap.parse_args()

VD = Path.home() / "vio_data"
NAME = a.name
RUN = VD / "testD" / NAME
OUT = VD / "testD_eval" / NAME; OUT.mkdir(parents=True, exist_ok=True)
lines = []
def say(s=""):
    print(s); lines.append(s)


def sec(x):
    return x / 1e9 if x > 1e11 else x


def load_vins(f):
    t, P, Q, V = [], [], [], []
    for r in csv.reader(open(f)):
        try:
            v = [float(x) for x in r[:11]]
        except (ValueError, IndexError):
            continue
        if len(v) < 8:
            continue
        t.append(sec(v[0])); P.append(v[1:4]); Q.append([v[5], v[6], v[7], v[4]])   # csv is w,x,y,z
        V.append(v[8:11] if len(v) >= 11 else [np.nan] * 3)
    return np.array(t), np.array(P), np.array(Q), np.array(V)


def win_speed(t, P, w=0.5):
    k = max(1, int(round(w / np.median(np.diff(t)))))
    s = np.full(len(t), np.nan)
    s[k:] = np.linalg.norm(P[k:] - P[:-k], axis=1) / (t[k:] - t[:-k])
    return s


def turn_rate(t, Q, w=0.5):
    k = max(1, int(round(w / np.median(np.diff(t)))))
    r = np.full(len(t), np.nan)
    rel = R.from_quat(Q[:-k]).inv() * R.from_quat(Q[k:])
    r[k:] = np.degrees(rel.magnitude()) / (t[k:] - t[:-k])
    return r


def load_metrics(f):
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None
    M = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, TypeError):
                vals.append(np.nan)
        M[k] = np.array(vals)
    tk = "t" if "t" in M else list(M)[0]
    M["_t"] = np.array([sec(x) for x in M[tk]])
    return M


def load_imu(f):
    rows = [r for r in csv.reader(open(f)) if r and r[0][:1].isdigit()]
    d = np.array([[float(x) for x in r[:7]] for r in rows])
    return np.array([sec(x) for x in d[:, 0]]), d[:, 1:4], d[:, 4:7]       # EuRoC order: t, gyro xyz, acc xyz


# ------------------------------------------------------------------ load
fsi = RUN / f"{NAME}_SI.csv"
if not fsi.exists():
    sys.exit(f"missing {fsi}")
tS, PS, QS, VS = load_vins(fsi)
so = load_vins(RUN / f"{NAME}_SO.csv") if (RUN / f"{NAME}_SO.csv").exists() else None
M = load_metrics(RUN / f"{NAME}_SI_metrics.csv")
imu_f = Path(a.imu).expanduser() if a.imu else VD / NAME / "dataset_half" / "imu0.csv"
imu = load_imu(imu_f) if imu_f.exists() else None

pairs = VD / NAME / "dataset_half" / "pairs.csv"
t0 = float(next(csv.DictReader(open(pairs)))["stamp_ns"]) / 1e9 if pairs.exists() else tS[0]
rel = lambda x: x - t0

spS = win_speed(tS, PS)
say(f"=== {NAME}: stereo-inertial divergence analysis (times in s from the first camera frame)")
say(f"SI poses from {rel(tS[0]):.1f} s to {rel(tS[-1]):.1f} s")
if M is None:
    say(f"(no {NAME}_SI_metrics.csv - internal panels will be empty)")
if imu is None:
    say(f"(no IMU file at {imu_f} - pass --imu; excitation panel will be empty)")

# ------------------------------------------------------------------ onset
if a.onset is not None:
    onset = t0 + a.onset; how = "given with --onset"
else:
    fast = spS > 1.0
    k1 = max(1, int(round(1.0 / np.median(np.diff(tS)))))
    onset = None
    for i in np.flatnonzero(fast):
        if np.all(fast[i:i + k1]):
            onset = tS[i]; break
    how = "SI speed first stays above 1 m/s for 1 s"
    if onset is None:
        i = int(np.nanargmax(spS)); onset = tS[i]
        how = f"SI never diverged - using its highest speed ({spS[i]:.2f} m/s)"
say(f"onset: {rel(onset):.1f} s  ({how})")
pre = (onset - a.pre, onset)
init_end = tS[0] + 0.2 * (onset - tS[0])        # 'early' = first 20 % of the run before onset
early = (tS[0] + 2.0, max(init_end, tS[0] + 5.0))

# ------------------------------------------------------------------ what the ROV was doing (from SO, the reference)
say("\nWHAT THE ROV WAS DOING  (stereo-only run as reference, last %.0f s before onset)" % a.pre)
if so is not None:
    tO, PO, QO, _ = so
    spO, trO = win_speed(tO, PO), turn_rate(tO, QO)
    m = (tO >= pre[0]) & (tO < pre[1])
    if m.any():
        say(f"  speed      median {np.nanmedian(spO[m]):.2f} m/s, spread (std) {np.nanstd(spO[m]):.2f} m/s")
        say(f"  turn rate  median {np.nanmedian(trO[m]):.1f} deg/s, max {np.nanmax(trO[m]):.1f} deg/s")
        straight = np.nanmedian(trO[m]) < 5 and np.nanstd(spO[m]) < 0.05
        say("  -> " + ("slow, straight, near-constant-velocity motion" if straight else "turning and/or changing speed"))
else:
    say("  (no SO run)")

# ------------------------------------------------------------------ IMU excitation
exc_t = exc_a = exc_g = None
say("\nIMU EXCITATION  (raw imu0.csv)")
if imu is not None:
    ti, gy, ac = imu
    w = max(5, int(round(1.0 / np.median(np.diff(ti)))))           # 1 s windows
    n = len(ti) // w
    exc_t = np.array([ti[j * w:(j + 1) * w].mean() for j in range(n)])
    exc_a = np.array([np.linalg.norm(ac[j * w:(j + 1) * w].std(0)) for j in range(n)])   # accel variation, m/s^2
    exc_g = np.degrees(np.array([np.linalg.norm(gy[j * w:(j + 1) * w].mean(0)) for j in range(n)]))
    def wm(x, lo, hi):
        s = (exc_t >= lo) & (exc_t < hi)
        return np.nanmedian(x[s]) if s.any() else np.nan
    a_all, a_pre = np.nanmedian(exc_a), wm(exc_a, *pre)
    say(f"  accelerometer variation (1 s windows): whole run median {a_all:.3f} m/s^2, "
        f"10 s before onset {a_pre:.3f} m/s^2  ({100 * a_pre / a_all:.0f}% of the run median)")
    say(f"  rotation rate: whole run median {np.nanmedian(exc_g):.1f} deg/s, before onset {wm(exc_g, *pre):.1f} deg/s")

# ------------------------------------------------------------------ internals: early vs before onset
flags = []
if M is not None:
    tm = M["_t"]
    def med(key, lo, hi, f=lambda x: x):
        if key not in M:
            return np.nan
        s = (tm >= lo) & (tm < hi) & (M.get("solver_flag", np.ones_like(tm)) == 1)
        return np.nanmedian(f(M[key][s])) if s.any() else np.nan
    ratio = lambda num, den: (lambda lo, hi: med(num, lo, hi) / max(med(den, lo, hi), 1))
    rowsdef = [
        ("features tracked", lambda lo, hi: med("n_feat", lo, hi), "{:.0f}"),
        ("stereo-match ratio", ratio("n_stereo", "n_feat"), "{:.2f}"),
        ("window features used", lambda lo, hi: med("n_win", lo, hi), "{:.0f}"),
        ("guessed-depth fraction", ratio("n_depth_guess", "n_win"), "{:.2f}"),
        ("outlier fraction", ratio("n_outliers", "n_win"), "{:.2f}"),
        ("median feature depth (m)", lambda lo, hi: med("depth_med", lo, hi), "{:.2f}"),
        ("solver cost", lambda lo, hi: med("cost", lo, hi), "{:.3g}"),
        ("solver iterations", lambda lo, hi: med("iters", lo, hi), "{:.0f}"),
        ("accel bias |ba| (m/s^2)", lambda lo, hi: med("ba_norm", lo, hi), "{:.3f}"),
        ("gyro bias |bg| (rad/s)", lambda lo, hi: med("bg_norm", lo, hi), "{:.4f}"),
        ("time offset td (ms)", lambda lo, hi: med("td_ms", lo, hi), "{:.1f}"),
    ]
    say("\nSI INTERNALS  (median, solver running)        early in run    10 s before onset    after onset")
    post = (onset, onset + 5)
    vals = {}
    for lab, fn, fmt in rowsdef:
        e, p, q = fn(*early), fn(*pre), fn(*post)
        vals[lab] = (e, p)
        fm = lambda x: (fmt.format(x) if np.isfinite(x) else "-")
        say(f"  {lab:<28} {fm(e):>14}  {fm(p):>18}  {fm(q):>12}")

    # first warning sign: earliest time each indicator crosses its threshold before onset
    say("\nFIRST WARNING SIGNS BEFORE ONSET  (earliest crossing; an earlier crossing suggests a cause)")
    def first_cross(series, test, label):
        s = (tm >= tS[0] + 2) & (tm < onset)
        idx = np.flatnonzero(s & test(series))
        if idx.size:
            flags.append((tm[idx[0]], label))
    if "n_feat" in M:
        base = np.nanmedian(M["n_feat"][(tm >= early[0]) & (tm < early[1])])
        first_cross(M["n_feat"], lambda x: x < 0.5 * base, f"features fell below half the early level ({0.5*base:.0f})")
    if "n_stereo" in M and "n_feat" in M:
        first_cross(M["n_stereo"] / np.maximum(M["n_feat"], 1), lambda x: x < 0.15, "stereo-match ratio < 0.15")
    if "n_outliers" in M and "n_win" in M:
        first_cross(M["n_outliers"] / np.maximum(M["n_win"], 1), lambda x: x > 0.2, "outlier fraction > 0.2")
    if "ba_norm" in M:
        first_cross(M["ba_norm"], lambda x: x > 1.0, "accelerometer bias > 1 m/s^2")
        first_cross(M["ba_norm"], lambda x: x > 3.0, "accelerometer bias > 3 m/s^2")
    if "bg_norm" in M:
        first_cross(M["bg_norm"], lambda x: x > 0.05, "gyro bias > 0.05 rad/s (~3 deg/s)")
    if "td_ms" in M:
        td0 = np.nanmedian(M["td_ms"][(tm >= early[0]) & (tm < early[1])])
        first_cross(M["td_ms"], lambda x: np.abs(x - td0) > 10, f"td moved > 10 ms from {td0:.1f} ms")
    if "cost" in M:
        cb = np.nanmedian(M["cost"][(tm >= early[0]) & (tm < early[1])])
        first_cross(M["cost"], lambda x: x > 10 * cb, "solver cost > 10x its early level")
    first_cross(np.interp(tm, tS, spS), lambda x: x > 0.5, "SI speed > 0.5 m/s (faster than the ROV moves)")
    if flags:
        for tt, lab in sorted(flags):
            say(f"  {rel(tt):7.1f} s  ({rel(tt) - rel(onset):+6.1f} s)  {lab}")
    else:
        say("  none of the indicators crossed its threshold before onset")

say("\nHOW TO READ THIS")
say("  - features / stereo ratio drop first, IMU fine        -> the camera lost texture; the IMU had to carry the estimate")
say("  - features steady, accelerometer bias climbs first    -> IMU-camera inconsistency absorbed by the bias, then runaway")
say("  - low accelerometer variation + straight motion       -> poor excitation: bias, gravity and scale not observable")
say("  - td drifting                                         -> time offset being misused to explain the disagreement")
(OUT / f"{NAME}_SI_divergence.txt").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
ax[0].plot(rel(tS), spS, color="#d62728", lw=1.2, label="Stereo-inertial")
if so is not None:
    ax[0].plot(rel(so[0]), win_speed(so[0], so[1]), color="#1f77b4", lw=1.2, label="Stereo only")
ax[0].axhline(1.0, color="k", ls="--", lw=0.8); ax[0].set_ylim(0, 2.5); ax[0].set_ylabel("speed (m/s)")
if exc_t is not None:
    ax[1].plot(rel(exc_t), exc_a, color="#444", lw=1.2, label="accelerometer variation (m/s²)")
    a2 = ax[1].twinx(); a2.plot(rel(exc_t), exc_g, color="#e08a00", lw=1, label="rotation rate (°/s)")
    a2.set_ylabel("°/s", color="#e08a00"); a2.legend(loc="upper left", fontsize=10)
ax[1].set_ylabel("m/s²")
if M is not None:
    tm = rel(M["_t"])
    if "ba_norm" in M: ax[2].plot(tm, M["ba_norm"], color="#d62728", lw=1.2, label="|accelerometer bias| (m/s²)")
    if "bg_norm" in M:
        a3 = ax[2].twinx(); a3.plot(tm, np.degrees(M["bg_norm"]), color="#6a3d9a", lw=1, label="|gyro bias| (°/s)")
        a3.set_ylabel("°/s", color="#6a3d9a"); a3.legend(loc="upper left", fontsize=10)
    if "td_ms" in M: ax[3].plot(tm, M["td_ms"], color="#2e7d4f", lw=1.2, label="time offset td")
    if "n_feat" in M: ax[4].plot(tm, M["n_feat"], color="#1f77b4", lw=1.2, label="features tracked")
    if "n_stereo" in M: ax[4].plot(tm, M["n_stereo"], color="#1f77b4", ls=":", lw=1.2, label="stereo-matched")
    if "n_outliers" in M: ax[4].plot(tm, M["n_outliers"], color="#c0392b", lw=1, label="outliers rejected")
    if "cost" in M: ax[5].semilogy(tm, np.maximum(M["cost"], 1e-6), color="#444", lw=1, label="solver cost")
    if "iters" in M:
        a5 = ax[5].twinx(); a5.plot(tm, M["iters"], color="#e08a00", lw=0.8, alpha=0.7, label="iterations")
        a5.legend(loc="upper left", fontsize=10)
ax[2].set_ylabel("m/s²"); ax[3].set_ylabel("td (ms)"); ax[4].set_ylabel("count"); ax[5].set_ylabel("cost")
titles = ["Speed: stereo-inertial vs stereo only (ROV < ~1 m/s)", "Raw IMU excitation (1 s windows)",
          "Stereo-inertial bias estimates", "Camera–IMU time offset estimate", "Front-end: features and outliers",
          "Back-end solver"]
for axi, ti in zip(ax, titles):
    axi.axvspan(rel(pre[0]), rel(pre[1]), color="#f4c542", alpha=0.25, lw=0)
    axi.axvline(rel(onset), color="#d62728", ls="--", lw=1)
    axi.set_title(ti, loc="left", fontsize=12); axi.grid(alpha=0.3)
    if axi.get_legend_handles_labels()[0]:
        axi.legend(loc="upper right", fontsize=10)
ax[-1].set_xlabel("time from first camera frame (s)   — dashed red: divergence onset, yellow: 10 s before", fontsize=12)
fig.suptitle(f"{NAME}: why did stereo-inertial diverge?", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(OUT / f"{NAME}_SI_divergence.png", dpi=130)
print(f"\nwrote {OUT / (NAME + '_SI_divergence.png')} and {NAME}_SI_divergence.txt")
