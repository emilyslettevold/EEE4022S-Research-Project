#!/usr/bin/env python3
"""
diag_testd.py - pinpoint WHEN and WHY a Test D run went wrong, from the raw run files.

  source ~/evo-env/bin/activate
  python3 ~/vio_data/diag_testd.py lawnmower2
  python3 ~/vio_data/diag_testd.py square2

Reads (nothing is re-run):
  ~/vio_data/testD/NAME/NAME_SO.csv, NAME_SI.csv            VINS poses
  ~/vio_data/testD/NAME/NAME_SO_metrics.csv, _SI_metrics.csv per-frame internals (if the metrics patch logged them)
  ~/vio_data/NAME/gt_tum.txt                                 ground truth (only to show WHEN it is available)

Writes ~/vio_data/testD_eval/NAME/NAME_diag.png - one time axis (seconds from the first camera frame), panels:
  1. estimated speed (0.5 s windows) of SO and SI; the ROV never exceeds ~1 m/s, so spikes = jumps / divergence
  2. features tracked and stereo-matched per frame (from _metrics.csv)
  3. fraction of window features with a guessed depth (from _metrics.csv)
  4. any bias columns found in _metrics.csv (e.g. accelerometer bias) - SI only
  grey shading on every panel = no ground truth at that time
and prints the time of the first jump > 0.1 m, the first speed > 1 m/s, and the ground-truth coverage.
"""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: diag_testd.py NAME")
VD = Path.home() / "vio_data"
RUN = VD / "testD" / NAME
OUT = VD / "testD_eval" / NAME; OUT.mkdir(parents=True, exist_ok=True)
COL = {"SO": "#1f77b4", "SI": "#d62728"}
FULL = {"SO": "Stereo only", "SI": "Stereo-inertial"}


def sec(x):
    return x / 1e9 if x > 1e11 else x


def load_vins(f):
    t, P = [], []
    for r in csv.reader(open(f)):
        try:
            t.append(sec(float(r[0]))); P.append([float(v) for v in r[1:4]])
        except (ValueError, IndexError):
            pass
    return np.array(t), np.array(P)


def load_metrics(f):
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None
    keys = list(rows[0].keys())
    tkey = next((k for k in ("t", "stamp", "t_ns", "time", "timestamp") if k in keys), keys[0])
    out = {}
    for k in keys:
        v = []
        for r in rows:
            try:
                v.append(float(r[k]))
            except (ValueError, TypeError):
                v.append(np.nan)
        out[k] = np.array(v)
    out["_t"] = np.array([sec(x) for x in out[tkey]])
    return out


runs = {}
for tag in ("SO", "SI"):
    f = RUN / f"{NAME}_{tag}.csv"
    if f.exists():
        runs[tag] = (*load_vins(f), load_metrics(RUN / f"{NAME}_{tag}_metrics.csv"))
if not runs:
    sys.exit(f"no {NAME}_SO.csv / {NAME}_SI.csv in {RUN}")

gt_t = np.array([])
gtf = VD / NAME / "gt_tum.txt"
if gtf.exists():
    gt_t = np.array([float(l.split()[0]) for l in open(gtf) if l.strip() and not l.startswith("#")])
pairs = VD / NAME / "dataset_half" / "pairs.csv"
t0 = min(r[0][0] for r in runs.values())
if pairs.exists():
    st = [float(r["stamp_ns"]) for r in csv.DictReader(open(pairs))]
    t0 = st[0] / 1e9          # pairs.csv stamp_ns is always nanoseconds
t_end = max(r[0][-1] for r in runs.values())

# ---- printed summary
print(f"{NAME}: times in seconds from the first camera frame (recording length {t_end - t0:.1f} s)")
if len(gt_t):
    gaps = np.flatnonzero(np.diff(gt_t) > 0.2)
    print(f"ground truth: {len(gt_t)} poses, available {gt_t[0]-t0:.1f}-{gt_t[-1]-t0:.1f} s; "
          f"{len(gaps)} gaps > 0.2 s, longest {np.diff(gt_t).max():.1f} s")
    for g in sorted(gaps, key=lambda i: gt_t[i] - gt_t[i + 1])[:5]:
        print(f"   no GT {gt_t[g]-t0:6.1f} -> {gt_t[g+1]-t0:6.1f} s  ({gt_t[g+1]-gt_t[g]:.1f} s)")
for tag, (t, P, M) in runs.items():
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    k = max(1, int(round(0.5 / np.median(np.diff(t)))))
    spd = np.linalg.norm(P[k:] - P[:-k], axis=1) / (t[k:] - t[:-k])
    big = np.flatnonzero(step > 0.10)
    fast = np.flatnonzero(spd > 1.0)
    print(f"{FULL[tag]}: first pose {t[0]-t0:.1f} s; {len(big)} jumps > 0.1 m"
          + (f", first at {t[big[0]+1]-t0:.1f} s ({step[big[0]]:.2f} m)" if len(big) else "")
          + (f"; speed first > 1 m/s at {t[fast[0]+k]-t0:.1f} s" if len(fast) else "; speed never > 1 m/s"))
    if len(big):
        print("   largest jumps: " + ", ".join(f"{t[i+1]-t0:.1f} s ({step[i]:.2f} m)" for i in big[np.argsort(-step[big])][:5]))
    if M is None:
        print(f"   (no {NAME}_{tag}_metrics.csv - internals panels left empty)")

# ---- figure
bias_keys = sorted({k for r in runs.values() if r[2] is not None for k in r[2]
                    if any(s in k.lower() for s in ("ba", "bias", "bg"))} - {"_t"})
n = 4 if bias_keys else 3
fig, ax = plt.subplots(n, 1, figsize=(12, 2.6 * n), sharex=True)


def shade(a):
    if not len(gt_t):
        return
    edges = [(t0, gt_t[0])] + [(gt_t[i], gt_t[i + 1]) for i in np.flatnonzero(np.diff(gt_t) > 0.2)] + [(gt_t[-1], t_end)]
    for x0, x1 in edges:
        if x1 > x0:
            a.axvspan(x0 - t0, x1 - t0, color="#bbbbbb", alpha=0.35, lw=0)


for tag, (t, P, M) in runs.items():
    k = max(1, int(round(0.5 / np.median(np.diff(t)))))
    spd = np.linalg.norm(P[k:] - P[:-k], axis=1) / (t[k:] - t[:-k])
    ax[0].plot(t[k:] - t0, spd, color=COL[tag], lw=1, label=FULL[tag])
    if M is not None:
        tm = M["_t"] - t0
        if "n_feat" in M:
            ax[1].plot(tm, M["n_feat"], color=COL[tag], lw=1, label=f"{FULL[tag]}: tracked")
        if "n_stereo" in M:
            ax[1].plot(tm, M["n_stereo"], color=COL[tag], lw=1, ls=":", label=f"{FULL[tag]}: stereo-matched")
        if "n_depth_guess" in M and "n_win" in M:
            ax[2].plot(tm, M["n_depth_guess"] / np.maximum(M["n_win"], 1), color=COL[tag], lw=1, label=FULL[tag])
        if tag == "SI":
            for bk in bias_keys:
                if bk in M:
                    ax[3].plot(tm, M[bk], lw=1, label=bk)
ax[0].axhline(1.0, color="k", ls="--", lw=0.8); ax[0].set_ylim(0, 3)
ax[0].set_ylabel("speed (m/s)")
ax[1].set_ylabel("features per frame"); ax[2].set_ylabel("guessed-depth fraction")
if bias_keys:
    ax[3].set_ylabel("SI bias estimates")
for a in ax:
    shade(a); a.grid(alpha=0.3)
    if a.get_legend_handles_labels()[0]:
        a.legend(fontsize=10, loc="upper right", ncol=2)
ax[0].set_title(f"{NAME}: what happened over time (grey = no ground truth)", fontsize=14)
ax[-1].set_xlabel("time from first camera frame (s)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / f"{NAME}_diag.png", dpi=140)
print(f"\nwrote {OUT / (NAME + '_diag.png')}")
