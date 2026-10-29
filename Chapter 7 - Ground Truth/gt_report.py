#!/usr/bin/env python3
"""
gt_report.py - one quality report + trajectory plots for ANY ground-truth run
(rail, orbit, square, lawnmower, figure-eight ...).

  python3 gt_report.py --frames gt_orbit1/gt_frames.csv --faces faces.csv --tagmap tagmap_all.csv \
      --name "orbit run 1" --out report_orbit1

Single-face runs (no faces.csv, e.g. the rail run) also work; the frame is then the face frame
(y up), and the top-down view is x-z.

Compare several runs in one table (reads the metrics.json each report writes):
  python3 gt_report.py --table report_orbit1 report_orbit2 report_square1 report_lawnmower1

Outputs in --out:
  report.txt        human-readable summary
  metrics.json      the same numbers, machine-readable (used by --table)
  trajectory_xy.png top-down view: position of the camera, structure outline, viewing direction
  trajectory_3d.png 3D view
  timeseries.png    x, y, z, reprojection error and faces in view against time

Metric definitions
  frames with a pose     : PnP produced a camera pose (status ok, ambiguous or high_rms)
  frames with GT         : pose passed every check (status ok) - this is the ground truth you use
  GT gaps                : stretches of time with no accepted pose (tags out of view, or rejected)
  precision (noise)      : estimated from second differences of consecutive accepted poses
                           (sigma = 1.4826*MAD(p[i-1] - 2p[i] + p[i+1]) / sqrt(6)). Valid while the
                           motion is smooth over 3 frames; it measures frame-to-frame jitter, not bias.
  predicted 1-sigma      : from the PnP covariance (sig_*_mm columns of gt_frames.csv)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#b5b4ad", "#fcfcfb", "#e6e5e0"
BLUE = "#2a78d6"


def fnum(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else np.nan


def read_faces(path):
    faces = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            R = Rotation.from_quat([float(r[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
            faces[int(r["face"])] = (R, np.array([float(r[k]) for k in ("tx", "ty", "tz")]))
    return faces


def read_tagmap(path):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        txt = f.read()
    delim = max([";", ",", "\t"], key=txt.splitlines()[0].count)
    rows = list(csv.reader(txt.splitlines(), delimiter=delim))
    h = [c.strip().lower() for c in rows[0]]
    for r in rows[1:]:
        if r and r[0].strip():
            d = dict(zip(h, r))
            out[int(float(d["id"]))] = int(float(d["face"]))
    return out


def segments(mask, t, max_dt):
    """Contiguous runs of True, split where consecutive accepted frames are > max_dt apart."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    segs, s0 = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1 or t[b] - t[a] > max_dt:
            segs.append((s0, a)); s0 = b
    segs.append((s0, idx[-1]))
    return segs


def noise_2nd_diff(P, Q, t, ok, frame):
    """Robust frame-to-frame noise from consecutive accepted triples."""
    i = np.flatnonzero(ok[1:-1] & ok[:-2] & ok[2:]) + 1
    i = i[(frame[i] - frame[i - 1] == 1) & (frame[i + 1] - frame[i] == 1) & (t[i + 1] - t[i - 1] < 0.2)]
    if len(i) < 10:
        return None
    d2 = P[i - 1] - 2 * P[i] + P[i + 1]
    s = 1.4826 * np.median(np.abs(d2 - np.median(d2, 0)), 0) / np.sqrt(6)
    Rm, R0, Rp = Rotation.from_quat(Q[i - 1]), Rotation.from_quat(Q[i]), Rotation.from_quat(Q[i + 1])
    a = (R0.inv() * Rm).as_rotvec() + (R0.inv() * Rp).as_rotvec()   # ~ 2nd difference of rotation
    sa = 1.4826 * np.median(np.abs(np.linalg.norm(a, axis=1))) / np.sqrt(6)
    return 1e3 * s, float(np.degrees(sa)), int(len(i))


def make_report(a):
    rows = list(csv.DictReader(open(a.frames, encoding="utf-8-sig")))
    if not rows:
        sys.exit(f"{a.frames} is empty")
    os.makedirs(a.out, exist_ok=True)
    n = len(rows)
    frame = np.array([int(r["frame"]) for r in rows])
    t = np.array([int(r["t_ns"]) for r in rows]) * 1e-9
    t0 = t[0]
    tt = t - t0
    st = np.array([r["status"] for r in rows])
    P = np.array([[fnum(r, k) for k in ("tx", "ty", "tz")] for r in rows])
    Q = np.array([[fnum(r, k) for k in ("qx", "qy", "qz", "qw")] for r in rows])
    rms = np.array([fnum(r, "rms_px") for r in rows])
    nfac = np.array([int(r["n_faces"] or 0) for r in rows])
    ntag = np.array([int(r["n_tags"] or 0) for r in rows])
    amb = np.array([fnum(r, "ambig_ratio") for r in rows])
    sig = np.array([[fnum(r, k) for k in ("sig_x_mm", "sig_y_mm", "sig_z_mm")] for r in rows])
    ids = [r["ids"].split() for r in rows]
    haspose = np.isin(st, ["ok", "ambiguous", "high_rms"])
    ok = st == "ok"
    dt_nom = float(np.median(np.diff(t))) if n > 1 else 1 / 30

    faces = read_faces(a.faces) if a.faces else None
    up = "z" if faces else "y"
    tag2face = read_tagmap(a.tagmap) if a.tagmap else None

    # horizontal coordinates and distance to the structure
    if up == "z":
        H = P[:, :2]; vert = P[:, 2]; hlab = ("x [m]", "y [m]"); vlab = "z (up) [m]"
        dist = np.linalg.norm(H, axis=1)
        dist_label = "horizontal distance to structure axis"
    else:
        H = P[:, [0, 2]]; vert = P[:, 1]; hlab = ("x [m]", "z, out of face [m]"); vlab = "y (up) [m]"
        dist = np.linalg.norm(P, axis=1)
        dist_label = "distance to face centre"

    segs = segments(ok, t, max(5 * dt_nom, 0.3))
    gaps = []
    edges = [(-1, -1)] + segs + [(n, n)]
    for (a0, a1), (b0, b1) in zip(edges[:-1], edges[1:]):
        ts = t0 if a1 < 0 else t[a1]
        te = t[-1] if b0 >= n else t[b0]
        if te - ts > max(5 * dt_nom, 0.3):
            gaps.append(te - ts)
    path = sum(np.linalg.norm(np.diff(P[s0:s1 + 1], axis=0), axis=1).sum() for s0, s1 in segs) if segs else 0.0
    nz = noise_2nd_diff(P, Q, t, ok, frame)
    nz1 = noise_2nd_diff(P, Q, t, ok & (nfac == 1), frame)
    nz2 = noise_2nd_diff(P, Q, t, ok & (nfac >= 2), frame)

    per_tag = {}
    if a.per_tag_from:
        try:
            per_tag = json.load(open(a.per_tag_from)).get("per_tag_rms_px_median", {})
        except (OSError, ValueError):
            per_tag = {}
    face_seen = {}
    if tag2face:
        for f in sorted(set(tag2face.values())):
            face_seen[f] = float(100 * np.mean([any(tag2face.get(int(i)) == f for i in L) for L in ids]))

    # start-to-end displacement: median pose over the first / last --endpoint-window seconds of ground truth
    endpoint = None
    if ok.sum() >= 2:
        io = np.flatnonzero(ok)
        w = a.endpoint_window
        s_idx = io[t[io] <= t[io[0]] + w]
        e_idx = io[t[io] >= t[io[-1]] - w]
        ps, pe = np.median(P[s_idx], 0), np.median(P[e_idx], 0)
        dvec = pe - ps
        if up == "z":
            dh, dv = float(np.linalg.norm(dvec[:2])), float(dvec[2])
        else:
            dh, dv = float(np.linalg.norm(dvec[[0, 2]])), float(dvec[1])
        Rs = Rotation.from_quat(Q[s_idx]).mean(); Re = Rotation.from_quat(Q[e_idx]).mean()
        endpoint = dict(t_start_s=round(float(t[io[0]] - t0), 2), t_end_s=round(float(t[io[-1]] - t0), 2),
                        frames_start=int(len(s_idx)), frames_end=int(len(e_idx)),
                        start_xyz_m=[round(float(v), 4) for v in ps], end_xyz_m=[round(float(v), 4) for v in pe],
                        distance_m=round(float(np.linalg.norm(dvec)), 4), horizontal_m=round(dh, 4),
                        vertical_m=round(dv, 4),
                        rotation_deg=round(float(np.degrees((Rs.inv() * Re).magnitude())), 2))

    single = haspose & (nfac == 1)
    M = {
        "name": a.name or os.path.basename(os.path.dirname(os.path.abspath(a.frames))),
        "frames": n, "duration_s": round(float(t[-1] - t0), 2), "fps": round(1 / dt_nom, 2),
        "status_counts": {s: int((st == s).sum()) for s in np.unique(st)},
        "pct_with_pose": round(100 * haspose.mean(), 1),
        "pct_with_gt": round(100 * ok.mean(), 1),
        "pct_frames_0_faces": round(100 * np.mean(nfac == 0), 1),
        "pct_frames_1_face": round(100 * np.mean(nfac == 1), 1),
        "pct_frames_2plus_faces": round(100 * np.mean(nfac >= 2), 1),
        "pct_single_face_frames_ambiguous": round(100 * np.mean(st[single] == "ambiguous"), 1) if single.any() else None,
        "gt_segments": len(segs),
        "gt_longest_continuous_s": round(max((t[s1] - t[s0] for s0, s1 in segs), default=0.0), 2),
        "gt_gaps": len(gaps), "gt_longest_gap_s": round(max(gaps, default=0.0), 2),
        "gt_time_without_s": round(float(sum(gaps)), 2),
        "gt_path_length_m": round(float(path), 3),
        "start_end": endpoint,
        "distance_label": dist_label,
        "distance_m_min_med_max": [round(float(np.nanmin(dist[ok])), 3), round(float(np.nanmedian(dist[ok])), 3),
                                   round(float(np.nanmax(dist[ok])), 3)] if ok.any() else None,
        "rms_px_median": round(float(np.nanmedian(rms[ok])), 3) if ok.any() else None,
        "rms_px_p95": round(float(np.nanpercentile(rms[ok], 95)), 3) if ok.any() else None,
        "tags_per_frame_median": float(np.median(ntag[ok])) if ok.any() else None,
        "noise_mm_xyz": [round(float(v), 2) for v in nz[0]] if nz else None,
        "noise_rot_deg": round(nz[1], 3) if nz else None,
        "noise_mm_xyz_1face": [round(float(v), 2) for v in nz1[0]] if nz1 else None,
        "noise_mm_xyz_2faces": [round(float(v), 2) for v in nz2[0]] if nz2 else None,
        "predicted_sigma_mm_xyz_median": [round(float(v), 2) for v in np.nanmedian(sig[ok], 0)] if ok.any() else None,
        "pct_time_each_face_in_view": face_seen or None,
        "frame": "structure (z up)" if up == "z" else "face (y up)",
    }
    json.dump(M, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)

    L = [f"Ground-truth report: {M['name']}", f"source: {a.frames}", ""]
    L.append(f"frames {n}, duration {M['duration_s']} s ({M['fps']} fps); coordinate frame: {M['frame']}")
    L.append(f"status: " + ", ".join(f"{k} {v}" for k, v in M["status_counts"].items()))
    L.append("")
    L.append("AVAILABILITY")
    L.append(f"  frames with a pose               {M['pct_with_pose']:5.1f} %")
    L.append(f"  frames with ground truth (ok)    {M['pct_with_gt']:5.1f} %")
    L.append(f"  faces in view: none {M['pct_frames_0_faces']}%, one {M['pct_frames_1_face']}%, "
             f"two or more {M['pct_frames_2plus_faces']}%")
    if M["pct_single_face_frames_ambiguous"] is not None:
        L.append(f"  single-face frames flagged ambiguous: {M['pct_single_face_frames_ambiguous']}%")
    L.append(f"  continuous GT segments {M['gt_segments']}, longest {M['gt_longest_continuous_s']} s; "
             f"gaps {M['gt_gaps']}, longest {M['gt_longest_gap_s']} s, total without GT {M['gt_time_without_s']} s")
    if face_seen:
        L.append("  share of frames each face is in view: " + ", ".join(f"face {f} {v:.0f}%" for f, v in face_seen.items()))
    L.append("")
    L.append("GEOMETRY")
    if M["distance_m_min_med_max"]:
        d = M["distance_m_min_med_max"]
        L.append(f"  {dist_label}: min {d[0]} m, median {d[1]} m, max {d[2]} m")
    L.append(f"  path length covered by GT: {M['gt_path_length_m']} m")
    if endpoint:
        e = endpoint
        L.append(f"  start to end (first vs last ground truth, median over {a.endpoint_window:g} s each): "
                 f"{e['distance_m']:.3f} m  (horizontal {e['horizontal_m']:.3f} m, vertical {e['vertical_m']:+.3f} m, "
                 f"rotation {e['rotation_deg']:.1f} deg)")
        L.append(f"     start at {e['t_start_s']} s: x/y/z {e['start_xyz_m'][0]:+.3f} / {e['start_xyz_m'][1]:+.3f} / "
                 f"{e['start_xyz_m'][2]:+.3f} m ({e['frames_start']} frames);  end at {e['t_end_s']} s: "
                 f"{e['end_xyz_m'][0]:+.3f} / {e['end_xyz_m'][1]:+.3f} / {e['end_xyz_m'][2]:+.3f} m ({e['frames_end']} frames)")
    L.append("")
    L.append("QUALITY (accepted frames)")
    L.append(f"  RMS reprojection error: median {M['rms_px_median']} px, 95th percentile {M['rms_px_p95']} px")
    L.append(f"  tags per frame (median): {M['tags_per_frame_median']}")
    if nz:
        L.append(f"  frame-to-frame noise x/y/z: {nz[0][0]:.1f} / {nz[0][1]:.1f} / {nz[0][2]:.1f} mm, rotation {nz[1]:.3f} deg"
                 f"   (from {nz[2]} consecutive triples)")
    if nz1:
        L.append(f"     one face in view:   {nz1[0][0]:.1f} / {nz1[0][1]:.1f} / {nz1[0][2]:.1f} mm")
    if nz2:
        L.append(f"     two faces in view:  {nz2[0][0]:.1f} / {nz2[0][1]:.1f} / {nz2[0][2]:.1f} mm")
    if M["predicted_sigma_mm_xyz_median"]:
        s = M["predicted_sigma_mm_xyz_median"]
        L.append(f"  predicted 1-sigma from PnP covariance (median): {s[0]} / {s[1]} / {s[2]} mm")
    if per_tag:
        med = np.median(list(per_tag.values()))
        bad = {k: v for k, v in per_tag.items() if v > 1.5 * med}
        L.append(f"  per-tag RMS: median {med:.2f} px" + (f"; tags above 1.5x median: {bad}" if bad else "; no outlier tags"))
    rep = "\n".join(L)
    open(os.path.join(a.out, "report.txt"), "w").write(rep + "\n")
    print(rep)

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "normal",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    amb_m = st == "ambiguous"

    def face_segments():
        """Face outlines: in the top-down plane (2 points each) and in 3D (4 corners)."""
        out = []
        if faces:
            for f, (R, tv) in faces.items():
                c = np.array([[-0.2, -0.3, 0], [0.2, -0.3, 0], [0.2, 0.3, 0], [-0.2, 0.3, 0]]) @ R.T + tv
                out.append((f, c, R[:, 2], tv))
        else:
            R, tv = np.eye(3), np.zeros(3)
            c = np.array([[-0.2, -0.3, 0], [0.2, -0.3, 0], [0.2, 0.3, 0], [-0.2, 0.3, 0]])
            out.append((0, c, R[:, 2], tv))
        return out

    def hv(p):
        return (p[..., 0], p[..., 1]) if up == "z" else (p[..., 0], p[..., 2])

    # top-down
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=SURF)
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for f, c, nrm, tv in face_segments():
        h0, h1 = hv(c[[0, 1]])
        ax.plot(h0, h1, color=INK, lw=3, solid_capstyle="butt", zorder=4)
        lx, ly = hv(tv + 0.12 * nrm)
        ax.text(lx, ly, f"F{f}", ha="center", va="center", fontsize=8, color=INK2, zorder=5)
    if amb_m.any():
        ax.scatter(*hv(P[amb_m]), s=3, color=MUTED, label="pose flagged ambiguous (one face, far)", zorder=2)
    ax.scatter(*hv(P[ok]), s=3, color=BLUE, label="ground truth (accepted)", zorder=3)
    k = np.flatnonzero(ok)[:: max(1, int(ok.sum() / 40))]
    if len(k):
        fwd = Rotation.from_quat(Q[k]).apply([0, 0, 1.0])
        fx, fy = hv(fwd)
        px, py = hv(P[k])
        ax.quiver(px, py, fx, fy, color=BLUE, angles="xy", scale_units="xy", scale=6, width=0.003, zorder=3)
    if ok.any():
        i0, i1 = np.flatnonzero(ok)[[0, -1]]
        ax.scatter(*hv(P[i0]), s=60, marker="o", facecolor=SURF, edgecolor=INK, lw=1.5, zorder=6)
        ax.annotate("start", hv(P[i0]), textcoords="offset points", xytext=(8, 4), color=INK)
        ax.scatter(*hv(P[i1]), s=60, marker="s", facecolor=SURF, edgecolor=INK, lw=1.5, zorder=6)
        ax.annotate("end", hv(P[i1]), textcoords="offset points", xytext=(8, 4), color=INK)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(hlab[0]); ax.set_ylabel(hlab[1])
    ax.set_title(f"{M['name']}: top-down trajectory (arrows: camera viewing direction)")
    ax.legend(loc="best", markerscale=4)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "trajectory_xy.png"), dpi=150, facecolor=SURF); plt.close(fig)

    # 3D
    fig = plt.figure(figsize=(8, 7), facecolor=SURF)
    ax = fig.add_subplot(projection="3d"); ax.set_facecolor(SURF)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def to3(p):
        return p if up == "z" else p[..., [0, 2, 1]]
    polys = [to3(c) for _, c, _, _ in face_segments()]
    ax.add_collection3d(Poly3DCollection(polys, facecolor="#d9d8d2", edgecolor=INK, lw=1.0, alpha=0.6))
    if amb_m.any():
        pa = to3(P[amb_m]); ax.scatter(pa[:, 0], pa[:, 1], pa[:, 2], s=2, color=MUTED, label="ambiguous", depthshade=False)
    po = to3(P[ok])
    if ok.any():
        ax.scatter(po[:, 0], po[:, 1], po[:, 2], s=2, color=BLUE, label="ground truth", depthshade=False)
    allp = np.vstack([po] + polys) if ok.any() else np.vstack(polys)
    mid = (np.nanmax(allp, 0) + np.nanmin(allp, 0)) / 2; rng_ = np.nanmax(np.nanmax(allp, 0) - np.nanmin(allp, 0)) / 2
    ax.set_xlim(mid[0] - rng_, mid[0] + rng_); ax.set_ylim(mid[1] - rng_, mid[1] + rng_); ax.set_zlim(mid[2] - rng_, mid[2] + rng_)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(hlab[0]); ax.set_ylabel(hlab[1]); ax.set_zlabel(vlab)
    ax.set_title(f"{M['name']}: 3D trajectory"); ax.legend(loc="upper left", markerscale=4)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "trajectory_3d.png"), dpi=150, facecolor=SURF); plt.close(fig)

    # time series
    names = ["x [m]", "y [m]", "z [m]"]
    fig, axs = plt.subplots(5, 1, figsize=(9, 10), sharex=True, facecolor=SURF,
                            gridspec_kw={"height_ratios": [1, 1, 1, 0.8, 0.6]})
    for j in range(3):
        ax = axs[j]
        if amb_m.any():
            ax.scatter(tt[amb_m], P[amb_m, j], s=2, color=MUTED, label="ambiguous" if j == 0 else None)
        ax.scatter(tt[ok], P[ok, j], s=2, color=BLUE, label="ground truth" if j == 0 else None)
        ax.set_ylabel(names[j])
    axs[0].legend(loc="upper right", markerscale=4, ncol=2)
    axs[0].set_title(f"{M['name']}: position, reprojection error and faces in view over time "
                     f"(grey bands: no ground truth)")
    axs[3].scatter(tt[haspose], rms[haspose], s=2, color=np.where(ok[haspose], BLUE, MUTED))
    axs[3].set_ylabel("RMS [px]"); axs[3].set_ylim(bottom=0)
    axs[4].step(tt, nfac, where="post", color=INK2, lw=1)
    axs[4].set_ylabel("faces"); axs[4].set_yticks([0, 1, 2]); axs[4].set_xlabel("time since start [s]")
    for ax in axs:
        ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for (a0, a1), (b0, b1) in zip(edges[:-1], edges[1:]):
            ts = 0.0 if a1 < 0 else tt[a1]
            te = tt[-1] if b0 >= n else tt[b0]
            if te - ts > max(5 * dt_nom, 0.3):
                ax.axvspan(ts, te, color=GRID, alpha=0.7, lw=0, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "timeseries.png"), dpi=150, facecolor=SURF); plt.close(fig)
    print(f"\nwrote {a.out}/report.txt, metrics.json, trajectory_xy.png, trajectory_3d.png, timeseries.png")


def make_table(dirs):
    rows = []
    for d in dirs:
        p = os.path.join(d, "metrics.json") if os.path.isdir(d) else d
        rows.append(json.load(open(p)))
    cols = [("run", lambda m: m["name"]), ("dur s", lambda m: m["duration_s"]),
            ("pose %", lambda m: m["pct_with_pose"]), ("GT %", lambda m: m["pct_with_gt"]),
            ("2-face %", lambda m: m["pct_frames_2plus_faces"]), ("longest gap s", lambda m: m["gt_longest_gap_s"]),
            ("dist med m", lambda m: (m["distance_m_min_med_max"] or [None] * 3)[1]),
            ("start-end m", lambda m: (m.get("start_end") or {}).get("distance_m", "-")),
            ("RMS px", lambda m: m["rms_px_median"]),
            ("noise x/y/z mm", lambda m: "/".join(f"{v:.1f}" for v in m["noise_mm_xyz"]) if m["noise_mm_xyz"] else "-"),
            ("noise deg", lambda m: m["noise_rot_deg"])]
    table = [[c[0] for c in cols]] + [[str(c[1](m)) for c in cols] for m in rows]
    w = [max(len(r[k]) for r in table) for k in range(len(cols))]
    for k, r in enumerate(table):
        print("  ".join(v.ljust(w[j]) for j, v in enumerate(r)))
        if k == 0:
            print("  ".join("-" * x for x in w))
    with open("gt_runs_table.csv", "w", newline="") as f:
        csv.writer(f).writerows(table)
    print("\nwrote gt_runs_table.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", help="gt_frames.csv from tag_gt.py")
    ap.add_argument("--faces", help="faces.csv from calibrate_faces.py (omit for single-face runs)")
    ap.add_argument("--tagmap", help="tag map, to report which faces were in view")
    ap.add_argument("--name", help="run name used in titles")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--endpoint-window", type=float, default=1.0,
                    help="s; start/end position = median of the first/last this many seconds of ground truth")
    ap.add_argument("--table", nargs="+", help="report folders (or metrics.json files) to compare in one table")
    a = ap.parse_args()
    if a.table:
        make_table(a.table); return
    if not a.frames or not a.out:
        ap.error("--frames and --out are required (or use --table)")
    s = os.path.join(os.path.dirname(os.path.abspath(a.frames)), "summary.json")
    a.per_tag_from = s if os.path.exists(s) else None
    make_report(a)


if __name__ == "__main__":
    main()
