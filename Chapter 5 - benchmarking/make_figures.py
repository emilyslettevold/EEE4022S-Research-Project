#!/usr/bin/env python3
"""
Make Figures 1-5 of the VIO framework selection section.

Run on the Mac, in vins_env, from ~/Desktop/tank_benchmark, after evaluate_all.sh:
    python3 make_figures.py
Output: results/figures/fig1_success_coverage.pdf ... fig5_memory.pdf (+ .png of each).
Use the PDFs in LaTeX: \\includegraphics[width=\\linewidth]{fig1_success_coverage.pdf}

  Fig 1  success rate and median coverage per sequence
  Fig 2  ATE of successful runs per sequence (log scale)
  Fig 3  WholeTank Medium top-down trajectories: GT and the median-ATE run of each framework
  Fig 4  position error over time, HalfTank Easy and WholeTank Medium, median-ATE run of each framework
  Fig 5  peak memory against sequence duration, with a straight-line fit per framework

Needs numpy, scipy, matplotlib (as evaluate.py).
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import load_tum, associate, umeyama_se3

SEQS = ["Structure_Easy", "HalfTank_Easy", "WholeTank_Medium", "WholeTank_Hard"]
LABEL = {"Structure_Easy": "Structure\nEasy", "HalfTank_Easy": "HalfTank\nEasy",
         "WholeTank_Medium": "WholeTank\nMedium", "WholeTank_Hard": "WholeTank\nHard"}
DURATION_S = {"Structure_Easy": 182.6, "HalfTank_Easy": 367.6, "WholeTank_Medium": 634.5, "WholeTank_Hard": 397.2}
METHODS = ["VINS", "ORB3"]
NAME = {"VINS": "VINS-Fusion", "ORB3": "ORB-SLAM3"}
COL = {"VINS": "#1f5fa8", "ORB3": "#d9822b"}     # same colour per framework in every figure
GT_COL = "#555555"
SHOW_UNSCORED = True        # Fig 3 (trajectories): also draw the unscored part of each estimate, faint and dashed

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def median_run(rows, ok_only=True):
    """Run whose ATE is the median of the (successful) runs; falls back to all runs with an ATE."""
    cand = [r for r in rows if (r["success"] == "1" or not ok_only) and np.isfinite(num(r["ate_t_rmse"]))]
    if not cand and ok_only:
        return median_run(rows, False)
    if not cand:
        return None
    cand.sort(key=lambda r: num(r["ate_t_rmse"]))
    return cand[(len(cand) - 1) // 2]


def aligned(res, gt_path, method, seq, run):
    """Whole estimated trajectory, aligned to GT with the same SE(3) fit as evaluate.py."""
    est = os.path.join(res, method, seq, f"run_{int(run):02d}", "est_cam_tum.txt")
    t_gt, p_gt, _ = load_tum(gt_path)
    t_e, p_e, _ = load_tum(est)
    ig, ie = associate(t_gt, t_e, 0.02)
    R, t = umeyama_se3(p_e[ie], p_gt[ig])
    return t_e, p_e @ R.T + t, t_gt, p_gt, ig, ie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/accuracy")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    ev = os.path.join(a.results, "eval")
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(ev, "runs_all.csv"))))
    g = defaultdict(list)
    for r in rows:
        g[(r["sequence"], r["method"])].append(r)
    x = np.arange(len(SEQS))
    w = 0.38

    # ---------------------------------------------------------------- Fig 1
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    for k, m in enumerate(METHODS):
        succ = [sum(r["success"] == "1" for r in g[(s, m)]) for s in SEQS]
        cov = [100 * np.median([num(r["coverage"]) for r in g[(s, m)]]) if g[(s, m)] else np.nan for s in SEQS]
        bars = ax.bar(x + (k - 0.5) * w, succ, w, color=COL[m], label=NAME[m])
        for b, n, c in zip(bars, succ, cov):
            ax.text(b.get_x() + b.get_width() / 2, n + 0.25, f"{n}/10\n{c:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set(xticks=x, xticklabels=[LABEL[s] for s in SEQS], ylabel="Successful runs (of 10)", ylim=(0, 13.5),
           yticks=range(0, 11, 2))
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_title("Success rate per sequence (label: successes and median coverage)", loc="left")
    save(fig, a.out, "fig1_success_coverage")

    # ---------------------------------------------------------------- Fig 2
    # one panel per sequence, each with its own linear axis starting at zero
    fig, axs = plt.subplots(1, len(SEQS), figsize=(7.4, 2.9))
    rng = np.random.default_rng(0)
    for ax, s in zip(axs, SEQS):
        top = 0
        for k, m in enumerate(METHODS):
            v = np.array([num(r["ate_t_rmse"]) for r in g[(s, m)] if r["success"] == "1"])
            v = v[np.isfinite(v)]
            if len(v) == 0:
                ax.text(k, 0.03, "no\nsuccessful\nruns", ha="center", va="bottom", fontsize=6.5, color=COL[m],
                        transform=ax.get_xaxis_transform())
                continue
            top = max(top, v.max())
            ax.boxplot(v, positions=[k], widths=0.55, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=COL[m], alpha=0.3, edgecolor=COL[m]),
                       medianprops=dict(color=COL[m], lw=1.8), whiskerprops=dict(color=COL[m]),
                       capprops=dict(color=COL[m]))
            ax.plot(k + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=2.8, color=COL[m], alpha=0.85)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, top * 1.2 if top > 0 else 1)
        ax.set_xticks([])
        ax.set_title(s.replace("_", " "), fontsize=8.5)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#999999")
    axs[0].set_ylabel("Position ATE (m)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COL[m], alpha=0.5, edgecolor=COL[m]) for m in METHODS]
    fig.legend(handles, [NAME[m] for m in METHODS], loc="center left", bbox_to_anchor=(1.0, 0.55),
               frameon=True, edgecolor="#999999", title="Framework", title_fontsize=8)
    fig.text(1.0, 0.25, "Box: median and IQR\nDots: individual runs\nEach panel has its\nown scale", fontsize=7,
             va="top", color="#444444")
    fig.tight_layout(w_pad=1.2)
    save(fig, a.out, "fig2_ate_distribution")

    # ---------------------------------------------------------------- Fig 3
    # Top-down view against the ground truth that was actually scored (TankGT, tag-visible poses only).
    # Solid: the estimate at the poses matched to TankGT, i.e. exactly what the ATE measures.
    # Dashed (SHOW_UNSCORED): the rest of the estimate, where no ground truth exists.
    # Vertical axis = the one with the smallest extent in TankGT (y); of the other two, the one with the
    # larger extent over everything drawn goes along the page, so the figure is landscape.
    seq = "WholeTank_Medium"
    gt = os.path.join(ev, "gt", f"{seq}_gt.txt")
    t_ref, p_ref, _ = load_tum(gt)
    vert = int(np.argmin(np.ptp(p_ref, axis=0)))
    runs = []
    for m in METHODS:
        r = median_run(g[(seq, m)])
        if r is not None:
            runs.append((m, r, aligned(a.results, gt, m, seq, r["run"])))
    allp = np.vstack([p_ref] + [d[1] for _, _, d in runs])
    hz = [i for i in range(3) if i != vert]
    hz.sort(key=lambda i: -np.ptp(allp[:, i]))                       # wider axis horizontal

    def broken(t, p, gap=0.5):
        k = np.nonzero(np.diff(t) > gap)[0] + 1
        return np.insert(p[:, hz[0]], k, np.nan), np.insert(p[:, hz[1]], k, np.nan)

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.plot(*broken(t_ref, p_ref), "-", color="#222222", lw=1.8, zorder=1, label="Ground truth (TankGT)")
    for m, r, (t_e, p_al, t_gt, _, ig, ie) in runs:
        if SHOW_UNSCORED:
            ax.plot(*broken(t_e, p_al), "--", color=COL[m], lw=1.3, alpha=0.75, dashes=(4, 2.5), zorder=0)
        ax.plot(*broken(t_gt[ig], p_al[ie]), "-", color=COL[m], lw=1.4, zorder=2,
                label=f"{NAME[m]}, scored (run {int(r['run'])}, ATE {num(r['ate_t_rmse']):.2f} m)")
    if SHOW_UNSCORED:
        ax.plot([], [], "--", color="#666666", lw=1.3, dashes=(4, 2.5), label="Not scored (no ground truth)")
    ax.set_aspect("equal")
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    ax.set(xlabel=f"{'xyz'[hz[0]]} (m)", ylabel=f"{'xyz'[hz[1]]} (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=True, edgecolor="#999999",
              borderpad=0.7, columnspacing=1.5)
    ax.set_title(f"WholeTank Medium, top-down view (vertical axis {'xyz'[vert]} not shown)", loc="left")
    fig.tight_layout()
    save(fig, a.out, "fig3_trajectories_wholetank_medium")

    # ---------------------------------------------------------------- Fig 4
    seqs4 = ["HalfTank_Easy", "WholeTank_Medium"]
    fig, axs = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=False)
    for axx, seq in zip(axs, seqs4):
        gt = os.path.join(ev, "gt", f"{seq}_gt.txt")
        t0 = load_tum(gt)[0][0]
        for m in METHODS:
            r = median_run(g[(seq, m)])
            if r is None:
                continue
            t_e, p_al, t_gt, p_gt, ig, ie = aligned(a.results, gt, m, seq, r["run"])
            tt = t_gt[ig] - t0
            e = np.linalg.norm(p_al[ie] - p_gt[ig], axis=1)
            brk = np.nonzero(np.diff(tt) > 0.5)[0] + 1
            tag = "" if r["success"] == "1" else ", failed run"
            axx.plot(np.insert(tt, brk, np.nan), np.insert(e, brk, np.nan), "-", color=COL[m], lw=1.0,
                     label=f"{NAME[m]} (run {int(r['run'])}{tag})")
            # mark stretches with ground truth but no estimate (tracking lost / not yet initialised)
            have = np.zeros(len(t_gt), bool)
            have[ig] = True
            miss = t_gt[~have] - t0
            if len(miss):
                yb = 0.03 if m == "VINS" else 0.08                            # separate rows per framework
                axx.plot(miss, np.full(len(miss), yb), "|", ms=4, color=COL[m], alpha=0.5,
                         transform=axx.get_xaxis_transform())
        axx.set_yscale("log")
        axx.set(ylabel="Position error (m)", xlim=(0, DURATION_S[seq]))
        axx.set_title(seq.replace("_", " ") + "  (ticks at bottom: ground truth available but no pose output)",
                      loc="left")
        axx.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="none", ncol=2)
    axs[-1].set_xlabel("Time from start of sequence (s)")
    fig.tight_layout()
    save(fig, a.out, "fig4_error_over_time")

    # ---------------------------------------------------------------- Fig 5
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for m in METHODS:
        xs, ys = [], []
        for s in SEQS:
            for r in g[(s, m)]:
                v = num(r.get("max_rss_mb"))
                if np.isfinite(v):
                    xs.append(DURATION_S[s] / 60)
                    ys.append(v)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 2:
            continue
        ax.plot(xs + np.random.default_rng(1).uniform(-0.08, 0.08, len(xs)), ys, "o", ms=3, alpha=0.6, color=COL[m])
        k, c = np.polyfit(xs, ys, 1)
        xx = np.linspace(0, xs.max() * 1.05, 50)
        ax.plot(xx, k * xx + c, "-", color=COL[m], lw=1.2, label=f"{NAME[m]}: {k:.0f} MB per minute")
    ax.set(xlabel="Sequence duration (min)", ylabel="Peak memory (MB)", xlim=(0, None), ylim=(0, None))
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Peak memory against sequence duration (dots: runs; lines: least-squares fit)", loc="left")
    save(fig, a.out, "fig5_memory")


if __name__ == "__main__":
    main()
