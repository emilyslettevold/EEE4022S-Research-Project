#!/usr/bin/env python3
"""
calibrate_faces.py - estimate the pose of every face of the AprilTag structure (the "calibration
mode" step, as in TankGT), so that all faces share one coordinate frame.

Input : a recording that goes around the structure and sees at least one corner per pair of
        adjacent faces (two faces in view at once). A slow orbit is ideal.
Output: faces.csv  (one row per face: T_W_F as tx,ty,tz,qx,qy,qz,qw)  -> pass to tag_gt.py --faces
        face_calibration.txt (quality report)

Method
  1. Detect tags in frames where >= 2 faces are visible (each with >= --min-tags-per-face tags).
  2. Per face, solve PnP from that face alone -> relative pose between the two faces in that frame.
  3. Robust average of each face-pair transform, then chain them out from the reference face
     (spanning tree, pairs with most observations first) -> initial face poses.
  4. Bundle adjustment: all face poses (reference face fixed) and all camera poses refined together,
     minimising pixel reprojection error of every tag corner (robust soft-L1 loss).
  5. Re-express everything in a STRUCTURE FRAME:
        origin = centre of the face centres (box axis, at tag-centre height)
        z      = up (average of the faces' "up" axes)
        x      = horizontal direction of the reference face's outward normal
     so the trajectory's x-y plane is the pool floor plane (top-down view).

Tag map (--tagmap) must list every tag on every face:  id,face,x_m,y_m,size_m

Example:
  python3 calibrate_faces.py --video orbit_run1/left.mkv --camchain stereo_calib-camchain.yaml \
      --tagmap tagmap_all.csv --out faces.csv --min-margin 0 --max-hamming 1
"""
import argparse
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_gt import Undistorter, frame_source, load_camera, read_table, num, row_corners  # noqa: E402


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def inv(M):
    R, t = M[:3, :3], M[:3, 3]
    return T(R.T, -R.T @ t)


def pnp(obj, img, K):
    ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj, img, K, None, rv, tv)
    proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(1).mean()))
    return T(cv2.Rodrigues(rv)[0], tv.ravel()), rms


def robust_mean_T(Ts):
    Rs = Rotation.from_matrix(np.array([M[:3, :3] for M in Ts]))
    ts = np.array([M[:3, 3] for M in Ts])
    keep = np.ones(len(Ts), bool)
    for _ in range(3):
        Rm = Rs[keep].mean()
        ang = np.degrees((Rm.inv() * Rs).magnitude())
        tm = np.median(ts[keep], axis=0)
        dt = np.linalg.norm(ts - tm, axis=1)
        a_thr = max(3 * np.median(ang[keep]), 0.5)
        t_thr = max(3 * np.median(dt[keep]), 0.005)
        keep = (ang < a_thr) & (dt < t_thr)
        if keep.sum() < 3:
            keep[:] = True
            break
    return T(Rs[keep].mean().as_matrix(), np.median(ts[keep], axis=0)), int(keep.sum()), \
        float(np.median(ang[keep])), float(1e3 * np.median(dt[keep]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--images")
    ap.add_argument("--camchain", required=True)
    ap.add_argument("--cam", default="cam0")
    ap.add_argument("--tagmap", required=True, help="tag map with ALL faces")
    ap.add_argument("--ref-face", type=int, default=0, help="face whose normal defines the structure x axis")
    ap.add_argument("--out", default="faces.csv")
    ap.add_argument("--undistort", choices=["image", "points", "none"], default="image")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--every", type=int, default=2, help="look at every Nth frame")
    ap.add_argument("--max-frames", type=int, default=400, help="max two-face frames used in the adjustment")
    ap.add_argument("--min-tags-per-face", type=int, default=2)
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--max-hamming", type=int, default=0, help="0 = only perfectly decoded tags (fewer false detections)")
    ap.add_argument("--max-face-rms", type=float, default=2.0, help="px; per-face PnP worse than this is not used")
    ap.add_argument("--decimate", type=float, default=1.0)
    ap.add_argument("--no-outlier-rejection", action="store_true",
                    help="reproduce the original calibration: no false-detection rejection, single-stage robust "
                         "adjustment, error reported over ALL observations")
    ap.add_argument("--label", default="", help="name of this configuration in the summary table, e.g. A")
    ap.add_argument("--summary-csv", help="append one table row with the key results to this CSV")
    a = ap.parse_args()
    a.timestamps = None

    K0, D, model, size = load_camera(a.camchain, a.cam)
    und = Undistorter(K0, D, model, size, a.undistort)
    K = und.K
    tagF, tagface = {}, {}
    for r in read_table(a.tagmap, ["id", "face", "x_m", "y_m", "size_m"]):
        tid, f = int(num(r["id"])), int(num(r["face"]))
        tagF[tid] = row_corners(r)
        tagface[tid] = f
    faces = sorted(set(tagface.values()))
    if a.ref_face not in faces:
        sys.exit(f"--ref-face {a.ref_face} is not in the tag map (faces {faces})")
    print(f"tag map: {len(tagF)} tags on faces {faces}")
    det = Detector(families="tag36h11", nthreads=4, quad_decimate=a.decimate, refine_edges=1, decode_sharpening=0.25)

    # ---------------------------------------------------------------- 1. collect two-face frames
    obs = []          # per frame: {face: (ids, img corners Nx2 per tag)}
    n_seen = 0
    for i, stem, img in frame_source(a):
        if i < a.start or (a.end >= 0 and i > a.end) or (i - a.start) % a.every:
            continue
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = und.image(g)
        n_seen += 1
        per = defaultdict(dict)
        dup = set()
        for d in det.detect(g):
            if d.tag_id not in tagF or d.hamming > a.max_hamming or d.decision_margin < a.min_margin:
                continue
            f = tagface[d.tag_id]
            if d.tag_id in per[f]:
                dup.add(d.tag_id)
            per[f][d.tag_id] = und.points(np.asarray(d.corners, float))
        for tid in dup:
            per[tagface[tid]].pop(tid, None)
        good = {f: v for f, v in per.items() if len(v) >= a.min_tags_per_face}
        if len(good) >= 2:
            obs.append((i, good))
        if n_seen % 300 == 0:
            print(f"  {n_seen} frames read, {len(obs)} with two faces")
    print(f"{n_seen} frames read, {len(obs)} frames with >= 2 faces visible")
    if len(obs) < 10:
        sys.exit("too few frames with two faces in view - use a recording that passes the corners slowly")
    if len(obs) > a.max_frames:
        sel = np.linspace(0, len(obs) - 1, a.max_frames).astype(int)
        obs = [obs[k] for k in sel]

    # ---------------------------------------------------------------- 1b. reject false detections
    # a tag that does not fit the other tags of its own face in the same frame is a false detection
    # (pool tiles, reflections, ROV parts decoded as a tag ID) or a badly located one -> drop it
    n_rej = defaultdict(int)
    for i, per in ([] if a.no_outlier_rejection else obs):
        for f in list(per):
            tags = per[f]
            for _ in range(2):                       # up to two bad tags per face per frame
                ids = sorted(tags)
                if len(ids) < 3:
                    break
                worst, worst_e = None, 0.0
                for t in ids:                        # leave-one-out: does tag t agree with the pose of the others?
                    rest = [u for u in ids if u != t]
                    res = pnp(np.vstack([tagF[u] for u in rest]), np.vstack([tags[u] for u in rest]), K)
                    if res is None:
                        continue
                    pr, _ = cv2.projectPoints(tagF[t], cv2.Rodrigues(res[0][:3, :3])[0], res[0][:3, 3], K, None)
                    e = float(np.sqrt(((pr.reshape(-1, 2) - tags[t]) ** 2).sum(1)).mean())
                    if e > worst_e:
                        worst, worst_e = t, e
                if worst is not None and worst_e > 8.0:
                    tags.pop(worst); n_rej[f] += 1
                else:
                    break
            if len(tags) < a.min_tags_per_face:
                per.pop(f)
    obs = [(i, per) for i, per in obs if len(per) >= 2]
    if n_rej:
        print("tag detections rejected as not fitting their own face (false or badly located detections): "
              + ", ".join(f"face {f}: {n}" for f, n in sorted(n_rej.items())))

    # ---------------------------------------------------------------- 2-3. pairwise init + chaining
    pair_T = defaultdict(list)
    face_fit = defaultdict(list)
    tag_fit = defaultdict(list)
    for i, per in obs:
        sol = {}
        for f, tags in per.items():
            ids = sorted(tags)
            o3, o2 = np.vstack([tagF[t] for t in ids]), np.vstack([tags[t] for t in ids])
            res = pnp(o3, o2, K)
            if res:
                face_fit[f].append(res[1])
                rv, tv = cv2.Rodrigues(res[0][:3, :3])[0], res[0][:3, 3]
                pr, _ = cv2.projectPoints(o3, rv, tv, K, None)
                e = np.sqrt(((pr.reshape(-1, 2) - o2) ** 2).sum(1)).reshape(-1, 4).mean(1)
                for t, v in zip(ids, e):
                    tag_fit[t].append(v)
            if res and res[1] < a.max_face_rms:
                sol[f] = res[0]                       # T_C_F
        fs = sorted(sol)
        for x in range(len(fs)):
            for y in range(x + 1, len(fs)):
                fi, fj = fs[x], fs[y]
                pair_T[(fi, fj)].append(inv(sol[fi]) @ sol[fj])          # T_Fi_Fj
    print("\nhow well each face's own tags fit the tag map (single-face PnP, median RMS):")
    bad_faces = []
    fit_med, fit_use = {}, {}
    for f in faces:
        e = np.array(face_fit.get(f, []))
        if not len(e):
            print(f"  face {f}: never seen in a two-face frame"); continue
        frac = 100 * np.mean(e < a.max_face_rms)
        fit_med[f], fit_use[f] = float(np.median(e)), float(frac)
        print(f"  face {f}: {np.median(e):6.2f} px, {frac:5.1f}% of views usable (< {a.max_face_rms} px)")
        if frac < 50:
            bad_faces.append(f)
        if np.median(e) > 0.7 * a.max_face_rms:
            ft = sorted(t for t in tag_fit if tagface[t] == f)
            print("      per-tag error on this face [px]: " + ", ".join(f"{t}: {np.median(tag_fit[t]):.1f}" for t in ft))
    if bad_faces:
        print(f"\n!! faces {bad_faces}: most views do NOT fit the tag map. The tag positions/orientation for these faces\n"
              f"   are probably wrong (IDs in a different order, a strip upside down, wrong row spacing).\n"
              f"   Run measure_tagmap.py on this video (it also finds strips that are not flat), then calibrate again\n"
              f"   with the tagmap_measured.csv it writes. Faces that do not fit are still used, but weighted down.")
    pair_est = {}
    print("\nface pairs seen together:")
    for (fi, fj), Ts in sorted(pair_T.items()):
        if len(Ts) < 3:
            continue
        M, nk, ang, dt = robust_mean_T(Ts)
        pair_est[(fi, fj)] = (M, len(Ts))
        nrm = np.degrees(np.arccos(np.clip(M[:3, 2] @ np.array([0, 0, 1.0]), -1, 1)))
        print(f"  faces {fi}-{fj}: {len(Ts):4d} frames (spread {ang:.2f} deg, {dt:.1f} mm), angle between normals {nrm:.1f} deg")
    # BFS from ref face over pairs, most-observed first
    T_ref = {a.ref_face: np.eye(4)}
    q = deque([a.ref_face])
    edges = sorted(pair_est.items(), key=lambda kv: -kv[1][1])
    while q:
        f = q.popleft()
        for (fi, fj), (M, n) in edges:
            if fi == f and fj not in T_ref:
                T_ref[fj] = T_ref[f] @ M; q.append(fj)
            elif fj == f and fi not in T_ref:
                T_ref[fi] = T_ref[f] @ inv(M); q.append(fi)
    missing = [f for f in faces if f not in T_ref]
    if missing:
        print(f"\nWARNING: faces {missing} are never seen together with the others - they cannot be calibrated "
              f"from this recording and will be left out of faces.csv")
    cal = sorted(T_ref)
    if len(cal) < 2:
        sys.exit("\nfewer than two faces could be linked - fix the tag map first (see the message above about measure_tagmap.py)")
    fidx = {f: k for k, f in enumerate([f for f in cal if f != a.ref_face])}

    # ---------------------------------------------------------------- 4. bundle adjustment
    # initial camera poses from multi-face PnP with the chained face poses
    frames, cam0 = [], []
    for i, per in obs:
        pts3, pts2 = [], []
        for f, tags in per.items():
            if f not in T_ref:
                continue
            for t, c in tags.items():
                pts3.append(tagF[t] @ T_ref[f][:3, :3].T + T_ref[f][:3, 3]); pts2.append(c)
        if len(pts3) < 2:
            continue
        res = pnp(np.vstack(pts3), np.vstack(pts2), K)
        if res is None:
            continue
        frames.append({f: v for f, v in per.items() if f in T_ref}); cam0.append(res[0])        # T_C_W(ref)
    nF, nC = len(fidx), len(frames)
    rows = []                                          # (cam idx, face, tag id, img corners)
    for c, per in enumerate(frames):
        for f, tags in per.items():
            if f in T_ref:
                for t, uv in tags.items():
                    rows.append((c, f, t, uv))
    nObs = len(rows)
    ci = np.array([r[0] for r in rows]); fi_ = np.array([fidx.get(r[1], -1) for r in rows])
    XF = np.array([tagF[r[2]] for r in rows])          # nObs x 4 x 3
    UV = np.array([r[3] for r in rows])                # nObs x 4 x 2

    def pack(Tf, Tc):
        x = []
        for f in sorted(fidx, key=fidx.get):
            x += list(Rotation.from_matrix(Tf[f][:3, :3]).as_rotvec()) + list(Tf[f][:3, 3])
        for M in Tc:
            x += list(Rotation.from_matrix(M[:3, :3]).as_rotvec()) + list(M[:3, 3])
        return np.array(x)

    def resid(x):
        fp = x[:6 * nF].reshape(nF, 6); cp = x[6 * nF:].reshape(nC, 6)
        RF = np.concatenate([np.eye(3)[None], Rotation.from_rotvec(fp[:, :3]).as_matrix()]) if nF else np.eye(3)[None]
        tF = np.concatenate([np.zeros((1, 3)), fp[:, 3:]])
        RC = Rotation.from_rotvec(cp[:, :3]).as_matrix(); tC = cp[:, 3:]
        k = fi_ + 1                                    # 0 = reference face
        Xw = np.einsum("nij,nkj->nki", RF[k], XF) + tF[k][:, None, :]
        Xc = np.einsum("nij,nkj->nki", RC[ci], Xw) + tC[ci][:, None, :]
        z = np.clip(Xc[..., 2], 1e-3, None)
        u = K[0, 0] * Xc[..., 0] / z + K[0, 2]
        v = K[1, 1] * Xc[..., 1] / z + K[1, 2]
        return (np.stack([u, v], -1) - UV).ravel()

    S = lil_matrix((nObs * 8, 6 * nF + 6 * nC), dtype=int)
    for o in range(nObs):
        rr = slice(8 * o, 8 * o + 8)
        S[rr, 6 * nF + 6 * ci[o]: 6 * nF + 6 * ci[o] + 6] = 1
        if fi_[o] >= 0:
            S[rr, 6 * fi_[o]: 6 * fi_[o] + 6] = 1
    x0 = pack(T_ref, cam0)
    r0 = resid(x0)
    print(f"\nbundle adjustment: {nC} frames, {nObs} tag observations, {len(cal)} faces; "
          f"initial RMS {np.sqrt((r0 ** 2).reshape(-1, 2).sum(1).mean()):.3f} px")
    if a.no_outlier_rejection:      # original single-stage adjustment, all observations kept
        sol = least_squares(resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=1.0, x_scale="jac", method="trf",
                            max_nfev=200)
        r = sol.fun.reshape(-1, 4, 2)
        n_out = 0
        print(f"final RMS {np.sqrt((r ** 2).sum(-1).mean()):.3f} px on all {nObs} observations  ({sol.message})")
    else:
        e0 = np.sqrt((r0.reshape(-1, 4, 2) ** 2).sum(-1).mean(-1))
        good = np.repeat(e0 < max(10.0, 3 * np.median(e0)), 8)
        print(f"  stage 1 uses {good.sum() // 8} of {nObs} observations (initial error < {max(10.0, 3 * np.median(e0)):.1f} px)")
        sol = least_squares(lambda x: resid(x) * good, x0, jac_sparsity=S, x_scale="jac", method="trf", max_nfev=300)
        sol = least_squares(lambda x: resid(x) * good, sol.x, jac_sparsity=S, loss="soft_l1", f_scale=2.0, x_scale="jac",
                            method="trf", max_nfev=500)
        e1 = np.sqrt((resid(sol.x).reshape(-1, 4, 2) ** 2).sum(-1).mean(-1))
        thr = max(5.0, 5 * np.median(e1))
        inl = e1 < thr
        n_out = int(np.sum(~inl))
        print(f"  {n_out} of {nObs} observations rejected as outliers (> {thr:.1f} px after adjustment)")
        w = np.repeat(inl, 8)
        sol = least_squares(lambda x: resid(x) * w, sol.x, jac_sparsity=S, loss="soft_l1", f_scale=2.0, x_scale="jac",
                            method="trf", max_nfev=500)
        r = resid(sol.x).reshape(-1, 4, 2)[inl]
        rows = [row for row, k in zip(rows, inl) if k]
        print(f"final RMS {np.sqrt((r ** 2).sum(-1).mean()):.3f} px on {inl.sum()} inlier observations  ({sol.message})")
    fp = sol.x[:6 * nF].reshape(nF, 6)
    T_fin = {a.ref_face: np.eye(4)}
    for f, k in fidx.items():
        T_fin[f] = T(Rotation.from_rotvec(fp[k, :3]).as_matrix(), fp[k, 3:])

    # ---------------------------------------------------------------- 5. structure frame
    up = np.mean([T_fin[f][:3, 1] for f in cal], axis=0); up /= np.linalg.norm(up)
    origin = np.mean([T_fin[f][:3, 3] for f in cal], axis=0)
    nx = T_fin[a.ref_face][:3, 2] - (T_fin[a.ref_face][:3, 2] @ up) * up; nx /= np.linalg.norm(nx)
    R_S = np.vstack([nx, np.cross(up, nx), up])       # rows: S axes in old coordinates
    T_S = {f: T(R_S @ M[:3, :3], R_S @ (M[:3, 3] - origin)) for f, M in T_fin.items()}

    if os.path.dirname(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fo:
        fo.write("face,tx,ty,tz,qx,qy,qz,qw\n")
        for f in cal:
            q_ = Rotation.from_matrix(T_S[f][:3, :3]).as_quat()
            fo.write(f"{f},{T_S[f][0,3]:.6f},{T_S[f][1,3]:.6f},{T_S[f][2,3]:.6f},"
                     f"{q_[0]:.8f},{q_[1]:.8f},{q_[2]:.8f},{q_[3]:.8f}\n")

    # ---------------------------------------------------------------- report
    L = []
    L.append(f"Face calibration from {a.video or a.images}")
    L.append(f"frames with >= 2 faces used: {nC}; tag observations: {nObs} ({n_out} rejected as outliers); "
             f"final RMS {np.sqrt((r ** 2).sum(-1).mean()):.3f} px")
    L.append(f"structure frame: origin = centre of face centres, z up, x = face {a.ref_face} outward normal\n")
    L.append(" face   centre x,y,z [m]            heading of normal [deg]   tilt from vertical [deg]   "
             "dist from axis [m]   RMS [px]   tags seen")
    rface = defaultdict(list)
    for o, row in enumerate(rows):
        rface[row[1]].append(r[o])
    for f in cal:
        M = T_S[f]; n = M[:3, 2]
        head = np.degrees(np.arctan2(n[1], n[0]))
        tilt = np.degrees(np.arccos(np.clip(M[2, 1], -1, 1)))     # face "up" axis vs structure z
        rr = np.array(rface[f])
        L.append(f"  {f:3d}   {M[0,3]:+.4f} {M[1,3]:+.4f} {M[2,3]:+.4f}        {head:+8.2f}                 "
                 f"{tilt:6.2f}                {np.hypot(M[0,3], M[1,3]):.4f}         "
                 f"{np.sqrt((rr ** 2).sum(-1).mean()):.3f}     {len(rr)}")
    L.append("\n angle between face normals (adjacent faces of a square box should be 90 deg):")
    for x in range(len(cal)):
        for y in range(x + 1, len(cal)):
            c = np.degrees(np.arccos(np.clip(T_S[cal[x]][:3, 2] @ T_S[cal[y]][:3, 2], -1, 1)))
            L.append(f"   faces {cal[x]}-{cal[y]}: {c:7.2f} deg   (seen together in {len(pair_T.get((cal[x], cal[y]), []))} frames)")
    rep = "\n".join(L)
    print("\n" + rep)
    with open(os.path.splitext(a.out)[0] + "_calibration.txt", "w") as fo:
        fo.write(rep + "\n")
    fin = float(np.sqrt((r ** 2).sum(-1).mean()))
    worst = max(cal, key=lambda f: np.sqrt((np.array(rface[f]) ** 2).sum(-1).mean()))
    if fin > 3 or np.sqrt((np.array(rface[worst]) ** 2).sum(-1).mean()) > 5:
        print(f"\n!! calibration NOT trustworthy (overall {fin:.1f} px, face {worst} worst). Do not use this faces.csv;"
              f" fix the tag map for the faces flagged above first.")
    if a.summary_csv:
        new = not os.path.exists(a.summary_csv)
        prs = sorted(pair_T)
        with open(a.summary_csv, "a") as fo:
            if new:
                fo.write("label,tagmap,settings,fit_px_per_face,usable_pct_per_face,pair_frames,final_rms_px,"
                         "final_rms_px_per_face,rejected_detections,rejected_outliers,adjacent_angles_deg\n")
            ang = []
            for (fi, fj) in prs:
                if fi in T_S and fj in T_S:
                    ang.append(f"{fi}-{fj}:{np.degrees(np.arccos(np.clip(T_S[fi][:3, 2] @ T_S[fj][:3, 2], -1, 1))):.2f}")
            fo.write(",".join([
                a.label, os.path.basename(a.tagmap),
                f"max-face-rms {a.max_face_rms}; min-tags {a.min_tags_per_face}; max-hamming {a.max_hamming}; "
                f"{'no rejection' if a.no_outlier_rejection else 'rejection on'}",
                " / ".join(f"{fit_med.get(f, float('nan')):.2f}" for f in faces),
                " / ".join(f"{fit_use.get(f, float('nan')):.1f}" for f in faces),
                " ".join(f"{fi}-{fj}:{len(pair_T[(fi, fj)])}" for (fi, fj) in prs),
                f"{fin:.2f}",
                " / ".join(f"{np.sqrt((np.array(rface[f]) ** 2).sum(-1).mean()):.2f}" if f in rface else "-" for f in faces),
                str(sum(n_rej.values())), str(n_out), " ".join(ang)]) + "\n")
        print(f"appended row '{a.label}' to {a.summary_csv}")
    print(f"\nwrote {a.out}")
    sys.stdout.flush()
    os._exit(0)      # exit before the AprilTag detector is destroyed: avoids a harmless segfault on Linux


if __name__ == "__main__":
    main()
