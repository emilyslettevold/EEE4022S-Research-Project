#!/usr/bin/env python3
"""
fig_failures.py - the one figure that explains the Test D failures (report Figure D.2).

  source ~/evo-env/bin/activate
  python3 ~/vio_data/fig_failures.py square2:0-30 lawnmower2:40-95

One column per recording (time window in s from the first camera frame), three rows on a shared time axis:
  1. estimated speed, stereo only vs stereo-inertial (0.5 s windows) - the ROV never exceeds ~1 m/s
  2. stereo-matched features per frame (0.5 s median), both configurations
  3. stereo-inertial accelerometer bias |ba|
Grey shading: no tag detections for more than 5 s (structure out of view; from gt_tum.txt, used only for this).
Red dashed line: stereo-inertial divergence onset (speed stays above 1 m/s for 1 s), if it happens in the window.
Blue dotted line: largest stereo-only jump in the window, if larger than 0.5 m.
Reads ~/vio_data/testD/NAME/NAME_{SO,SI}.csv and NAME_{SO,SI}_metrics.csv. Nothing is re-run.
Writes ~/vio_data/testD_eval/fig_failures.png and .pdf
"""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VD = Path.home() / "vio_data"
COL = {"SO": "#1f77b4", "SI": "#d62728"}
FULL = {"SO": "Stereo only", "SI": "Stereo-inertial"}
sec = lambda x: x / 1e9 if x > 1e11 else x


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
    get = lambda k: np.array([float(r[k]) if r.get(k, "") not in ("", None) else np.nan for r in rows])
    M = {k: get(k) for k in ("t", "n_stereo", "ba_norm", "solver_flag") if rows and k in rows[0]}
    if "t" in M:
        M["t"] = np.array([sec(x) for x in M["t"]])
    return M


def speed(t, P, w=0.5):
    k = max(1, int(round(w / np.median(np.diff(t)))))
    s = np.full(len(t), np.nan)
    s[k:] = np.linalg.norm(P[k:] - P[:-k], axis=1) / (t[k:] - t[:-k])
    return s


def rolling_median(t, x, w=0.5):
    k = max(1, int(round(w / np.median(np.diff(t)))))
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        out[i] = np.nanmedian(x[max(0, i - k + 1): i + 1])
    return out


specs = []
for a in sys.argv[1:] or sys.exit(__doc__):
    name, _, win = a.partition(":")
    lo, hi = (float(v) for v in win.split("-")) if win else (None, None)
    specs.append((name, lo, hi))

fig, ax = plt.subplots(3, len(specs), figsize=(6.2 * len(specs), 8.2), sharex="col", squeeze=False)
for c, (name, lo, hi) in enumerate(specs):
    run = VD / "testD" / name
    pairs = VD / name / "dataset_half" / "pairs.csv"
    data = {tag: load_vins(run / f"{name}_{tag}.csv") for tag in ("SO", "SI") if (run / f"{name}_{tag}.csv").exists()}
    if not data:
        sys.exit(f"no runs in {run}")
    t0 = float(next(csv.DictReader(open(pairs)))["stamp_ns"]) / 1e9 if pairs.exists() else min(d[0][0] for d in data.values())
    lo = 0.0 if lo is None else lo
    hi = max(d[0][-1] for d in data.values()) - t0 if hi is None else hi

    for tag, (t, P) in data.items():
        ax[0, c].plot(t - t0, speed(t, P), color=COL[tag], lw=1.4, label=FULL[tag])
        M = load_metrics(run / f"{name}_{tag}_metrics.csv")
        if M and "n_stereo" in M:
            ok = M.get("solver_flag", np.ones_like(M["t"])) == 1
            ax[1, c].plot(M["t"][ok] - t0, rolling_median(M["t"][ok], M["n_stereo"][ok]), color=COL[tag], lw=1.4,
                          label=FULL[tag])
            if tag == "SI" and "ba_norm" in M:
                ax[2, c].plot(M["t"][ok] - t0, M["ba_norm"][ok], color=COL["SI"], lw=1.6, label=FULL["SI"])

    # events
    if "SI" in data:
        t, P = data["SI"]; s = speed(t, P); k1 = max(1, int(round(1 / np.median(np.diff(t)))))
        for i in np.flatnonzero(s > 1.0):
            if np.all(s[i:i + k1] > 1.0) and lo <= t[i] - t0 <= hi:
                for r in range(3):
                    ax[r, c].axvline(t[i] - t0, color=COL["SI"], ls="--", lw=1.2)
                ax[0, c].annotate(f"stereo-inertial diverges ({t[i]-t0:.1f} s)", (t[i] - t0, 1.35), fontsize=10,
                                  color=COL["SI"], ha="right", xytext=(-4, 0), textcoords="offset points")
                break
    if "SO" in data:
        t, P = data["SO"]; st = np.linalg.norm(np.diff(P, axis=0), axis=1); tt = t[1:] - t0
        m = (tt >= lo) & (tt <= hi)
        if m.any() and st[m].max() > 0.5:
            j = np.flatnonzero(m)[np.argmax(st[m])]
            for r in range(3):
                ax[r, c].axvline(tt[j], color=COL["SO"], ls=":", lw=1.5)
            ax[0, c].annotate(f"stereo-only jump {st[j]:.1f} m ({tt[j]:.1f} s)", (tt[j], 1.35), fontsize=10,
                              color=COL["SO"], ha="left", xytext=(4, 0), textcoords="offset points")
    gtf = VD / name / "gt_tum.txt"
    if gtf.exists():
        g = np.array([float(l.split()[0]) for l in open(gtf) if l.strip() and not l.startswith("#")])
        for i in np.flatnonzero(np.diff(g) > 5.0):
            for r in range(3):
                ax[r, c].axvspan(g[i] - t0, g[i + 1] - t0, color="#999999", alpha=0.22, lw=0,
                                 label="structure out of view" if r == 1 else None)

    ax[0, c].set_ylim(0, 1.5); ax[0, c].axhline(1.0, color="k", lw=0.6, ls="--")
    ax[0, c].set_title(name, fontsize=14)
    ax[2, c].set_xlim(lo, hi); ax[2, c].set_xlabel("time from first frame (s)", fontsize=12)
    for r in range(3):
        ax[r, c].grid(alpha=0.3); ax[r, c].tick_params(labelsize=11)
ax[0, 0].set_ylabel("estimated speed (m/s)", fontsize=12)
ax[1, 0].set_ylabel("stereo-matched features", fontsize=12)
ax[2, 0].set_ylabel("|accelerometer bias| (m/s²)", fontsize=12)
h, l = [], []
for a_ in ax[:, 0]:
    for hh, ll in zip(*a_.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=12, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
out = VD / "testD_eval" / "fig_failures"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf")
print(f"wrote {out}.png and {out}.pdf")
