#!/usr/bin/env python3
"""
measure_tagmap.py - measure the real 3D pose of every tag on every face from video, and compare it
with your tag map. Finds:
  - tags in the wrong position (IDs swapped / strips glued in a different order)
  - tags or strips glued upside down or sideways (in-plane rotation)
  - gluing offsets of a few mm
  - tags or strips that are NOT flat with the rest of the face: standing proud / sunken (z) or tilted
    (warped board, lifted strip) - things a flat tag map cannot represent

How:
  1. each tag's pose is estimated on its own from its 4 corners; tags seen together on a face give
     their relative pose (no tag-map positions needed) -> initial layout per face
  2. per-face bundle adjustment: every tag's 6-DoF pose on the face and every camera pose are refined
     together, minimising the pixel error of all tag corners (one tag fixed to set the frame)
  3. face frame: plane fitted through the tag centres; origin at their centroid; y = "up" of most tags

The tag map is used only for which face each ID is on and the tag size, and for the comparison.

Output: tagmap_measured.csv  id,face,x_m,y_m,z_m,size_m,rot_deg,tilt_x_deg,tilt_y_deg
        (use it directly with tag_gt.py, calibrate_faces.py, gt_report.py)

  python3 measure_tagmap.py --video orbit2/left.mkv --camchain stereo_calib-camchain.yaml \
      --tagmap tagmap_all.csv --out tagmap_measured.csv
"""
import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_gt import Undistorter, frame_source, load_camera, num, read_table, row_opt  # noqa: E402


def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def inv(M):
    return T(M[:3, :3].T, -M[:3, :3].T @ M[:3, 3])


def tag_pose(c, size, K):
    """Pose of one tag (T_C_tag) from its 4 corners with IPPE_SQUARE; returns (T, ambiguity ratio)."""
    h = size / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)      # TL, TR, BR, BL (y up)
    img = c[[3, 2, 1, 0]].astype(float)                                         # pupil: BL, BR, TR, TL
    n, rv, tv, err = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if n < 1:
        return None
    err = np.asarray(err).ravel()
    k = int(np.argmin(err))
    ratio = float(err[1 - k] / max(err[k], 1e-9)) if n > 1 else np.inf
    return T(cv2.Rodrigues(rv[k])[0], tv[k].ravel()), ratio


def local_corners(size):
    h = size / 2
    return np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)     # pupil order, tag frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--images")
    ap.add_argument("--camchain", required=True)
    ap.add_argument("--cam", default="cam0")
    ap.add_argument("--tagmap", required=True)
    ap.add_argument("--out", default="tagmap_measured.csv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--min-px", type=float, default=60.0, help="only use tags at least this many px across")
    ap.add_argument("--min-ambig", type=float, default=3.0, help="initial layout: single-tag poses must be unambiguous")
    ap.add_argument("--max-hamming", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=300, help="max frames per face in the bundle adjustment")
    ap.add_argument("--planar", action="store_true", help="force all tags flat in the face plane (no z/tilt)")
    a = ap.parse_args()
    a.timestamps = None

    K0, D, model, size = load_camera(a.camchain, a.cam)
    und = Undistorter(K0, D, model, size, "image")
    K = und.K
    tmap = {}
    for r in read_table(a.tagmap, ["id", "face", "x_m", "y_m", "size_m"]):
        tmap[int(num(r["id"]))] = dict(face=int(num(r["face"])), x=num(r["x_m"]), y=num(r["y_m"]), s=num(r["size_m"]),
                                      z=row_opt(r, "z_m"), rot=row_opt(r, "rot_deg"))
    faces = sorted({v["face"] for v in tmap.values()})
    det = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=1, decode_sharpening=0.25)

    rel = defaultdict(list)                 # (ta, tb) -> list of T_ta_tb
    face_frames = defaultdict(list)         # face -> list of {tid: corners}
    nfr = 0
    for i, stem, img in frame_source(a):
        if i < a.start or (a.end >= 0 and i > a.end) or (i - a.start) % a.every or img is None:
            continue
        g = und.image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
        nfr += 1
        per = defaultdict(dict)
        poses = {}
        dup = set()
        for d in det.detect(g):
            if d.tag_id not in tmap or d.hamming > a.max_hamming:
                continue
            c = np.asarray(d.corners, float)
            side = np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)])
            if side < a.min_px:
                continue
            f = tmap[d.tag_id]["face"]
            if d.tag_id in per[f]:
                dup.add(d.tag_id)
            per[f][d.tag_id] = c
            p = tag_pose(c, tmap[d.tag_id]["s"], K)
            if p is not None and p[1] >= a.min_ambig:
                poses[d.tag_id] = p[0]
        for t in dup:
            per[tmap[t]["face"]].pop(t, None); poses.pop(t, None)
        for f, tags in per.items():
            if len(tags) >= 2:
                face_frames[f].append(tags)
        ids = sorted(poses)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                ta, tb = ids[x], ids[y]
                if tmap[ta]["face"] == tmap[tb]["face"]:
                    rel[(ta, tb)].append(inv(poses[ta]) @ poses[tb])
        if nfr % 300 == 0:
            print(f"  {nfr} frames read")
    print(f"{nfr} frames read")

    out_rows, report = [], []
    for f in faces:
        fid = sorted(t for t, v in tmap.items() if v["face"] == f)
        # ---------------- 1. initial layout from pairwise single-tag poses (pose graph, spanning tree)
        pairs = {}
        for (ta, tb), L in rel.items():
            if ta in fid and len(L) >= 3:
                Rs = Rotation.from_matrix(np.array([M[:3, :3] for M in L]))
                Rm = Rs.mean()
                keep = np.degrees((Rm.inv() * Rs).magnitude()) < 10
                if keep.sum() < 3:
                    continue
                pairs[(ta, tb)] = (T(Rs[keep].mean().as_matrix(), np.median(np.array([M[:3, 3] for M in L])[keep], 0)),
                                   int(keep.sum()))
        found = sorted({t for p in pairs for t in p})
        report.append(f"\nFACE {f}: tags in map {fid}")
        if len(found) < 2:
            report.append("   not enough close-range views of this face - map values kept")
            for t in fid:
                v = tmap[t]
                out_rows.append((t, f, v["x"], v["y"], v["z"], v["s"], v["rot"], 0.0, 0.0))
            continue
        ref = max(found, key=lambda t: sum(n for (p, q), (_, n) in pairs.items() if t in (p, q)))
        P = {ref: np.eye(4)}
        edges = sorted(pairs.items(), key=lambda kv: -kv[1][1])
        grew = True
        while grew:
            grew = False
            for (ta, tb), (M, n) in edges:
                if ta in P and tb not in P:
                    P[tb] = P[ta] @ M; grew = True
                elif tb in P and ta not in P:
                    P[ta] = P[tb] @ inv(M); grew = True
        comp = sorted(P)

        # ---------------- 2. per-face bundle adjustment on pixel reprojection error
        frames = [{t: c for t, c in fr.items() if t in P} for fr in face_frames[f]]
        frames = [fr for fr in frames if len(fr) >= 2]
        if len(frames) > a.max_frames:
            frames = [frames[k] for k in np.linspace(0, len(frames) - 1, a.max_frames).astype(int)]
        Lc = {t: local_corners(tmap[t]["s"]) for t in comp}
        cams = []
        for fr in frames:
            o = np.vstack([Lc[t] @ P[t][:3, :3].T + P[t][:3, 3] for t in fr])
            im = np.vstack([fr[t] for t in fr])
            ok, rv, tv = cv2.solvePnP(o, im, K, None, flags=cv2.SOLVEPNP_SQPNP)
            rv, tv = cv2.solvePnPRefineLM(o, im, K, None, rv, tv)
            cams.append(np.concatenate([rv.ravel(), tv.ravel()]))
        others = [t for t in comp if t != ref]
        oi = {t: k for k, t in enumerate(others)}
        obs = [(c, t, fr[t]) for c, fr in enumerate(frames) for t in fr]
        ci = np.array([o[0] for o in obs]); ti = np.array([oi.get(o[1], -1) for o in obs])
        XL = np.array([Lc[o[1]] for o in obs]); UV = np.array([o[2] for o in obs])
        nT, nC = len(others), len(frames)

        def unpack(x):
            tp = x[:6 * nT].reshape(nT, 6)
            if a.planar:      # keep tags in the reference tag's plane: only x, y, in-plane rotation free
                tp = tp.copy(); tp[:, 0:2] = 0; tp[:, 5] = 0
            return tp, x[6 * nT:].reshape(nC, 6)

        def resid(x):
            tp, cp = unpack(x)
            RT = np.concatenate([np.eye(3)[None], Rotation.from_rotvec(tp[:, :3]).as_matrix()])
            tT = np.concatenate([np.zeros((1, 3)), tp[:, 3:]])
            RC = Rotation.from_rotvec(cp[:, :3]).as_matrix(); tC = cp[:, 3:]
            k = ti + 1
            Xf = np.einsum("nij,nkj->nki", RT[k], XL) + tT[k][:, None, :]
            Xc = np.einsum("nij,nkj->nki", RC[ci], Xf) + tC[ci][:, None, :]
            z = np.clip(Xc[..., 2], 1e-3, None)
            uv = np.stack([K[0, 0] * Xc[..., 0] / z + K[0, 2], K[1, 1] * Xc[..., 1] / z + K[1, 2]], -1)
            return (uv - UV).ravel()

        x0 = np.concatenate([np.concatenate([Rotation.from_matrix(P[t][:3, :3]).as_rotvec(), P[t][:3, 3]]) for t in others]
                            + [np.concatenate(cams)]) if others else np.concatenate(cams)
        S = lil_matrix((len(obs) * 8, 6 * nT + 6 * nC), dtype=int)
        for o in range(len(obs)):
            S[8 * o:8 * o + 8, 6 * nT + 6 * ci[o]:6 * nT + 6 * ci[o] + 6] = 1
            if ti[o] >= 0:
                S[8 * o:8 * o + 8, 6 * ti[o]:6 * ti[o] + 6] = 1
        r0 = resid(x0)
        sol = least_squares(resid, x0, jac_sparsity=S, method="trf", x_scale="jac", max_nfev=100)          # stage 1
        sol = least_squares(resid, sol.x, jac_sparsity=S, method="trf", x_scale="jac", loss="soft_l1",
                            f_scale=2.0, max_nfev=200)                                                     # stage 2
        rr = sol.fun.reshape(-1, 4, 2)
        tp, _ = unpack(sol.x)
        TT = {ref: np.eye(4)}
        for t in others:
            TT[t] = T(Rotation.from_rotvec(tp[oi[t], :3]).as_matrix(), tp[oi[t], 3:])
        per_tag_rms = {t: float(np.sqrt((rr[[k for k, o in enumerate(obs) if o[1] == t]] ** 2).sum(-1).mean())) for t in comp}

        # ---------------- 3. face frame from the tag centres
        C = np.array([TT[t][:3, 3] for t in comp])
        cen = C.mean(0)
        if len(comp) >= 3:
            n = np.linalg.svd(C - cen)[2][2]
        else:
            n = TT[ref][:3, 2]
        zs = np.mean([TT[t][:3, 2] for t in comp], 0)
        if n @ zs < 0:
            n = -n
        # majority in-plane orientation
        def inplane_angle(M, xa):
            v = M[:3, 0] - (M[:3, 0] @ n) * n
            return np.degrees(np.arctan2(np.cross(xa, v) @ n, xa @ v))
        xa = TT[ref][:3, 0] - (TT[ref][:3, 0] @ n) * n; xa /= np.linalg.norm(xa)
        ang = {t: inplane_angle(TT[t], xa) for t in comp}
        q = {t: int(np.round(ang[t] / 90)) % 4 for t in comp}
        mode = max(set(q.values()), key=list(q.values()).count)
        maj = np.radians([ang[t] for t in comp if q[t] == mode])
        phi = np.arctan2(np.mean(np.sin(maj)), np.mean(np.cos(maj)))
        xF = np.cos(phi) * xa + np.sin(phi) * np.cross(n, xa)
        R_RF = np.column_stack([xF, np.cross(n, xF), n])               # face axes in reference-tag frame
        T_FR = inv(T(R_RF, cen))

        report.append(f"   bundle adjustment: {nC} frames, {len(obs)} tag views, RMS {np.sqrt((r0.reshape(-1, 2) ** 2).sum(1).mean()):.2f}"
                      f" -> {np.sqrt((rr ** 2).sum(-1).mean()):.2f} px")
        report.append("    id   x, y [mm]          z [mm]  rot [deg]  tilt x/y [deg] | map x, y [mm]     rot | in-plane diff [mm]  RMS [px]  note")
        for t in sorted(comp):
            M = T_FR @ TT[t]
            x, y, z = M[:3, 3]
            rot, tx, ty = Rotation.from_matrix(M[:3, :3]).as_euler("ZXY", degrees=True)
            mv = tmap[t]
            dpos = 1e3 * np.hypot(x - mv["x"], y - mv["y"])
            notes = []
            if abs(wrapd(rot - mv["rot"])) > 20:
                notes.append(f"ROTATED {rot:+.0f} deg (map {mv['rot']:.0f})")
            if dpos > 25:
                near = min(fid, key=lambda u: np.hypot(x - tmap[u]["x"], y - tmap[u]["y"]))
                notes.append(f"WRONG POSITION - sits where the map has tag {near}" if near != t else "position off")
            elif dpos > 5:
                notes.append("offset (gluing)")
            if abs(z) > 0.003:
                notes.append(f"{'PROUD' if z > 0 else 'SUNKEN'} {1e3 * abs(z):.0f} mm")
            if np.hypot(tx, ty) > 1.5:
                notes.append(f"TILTED {np.hypot(tx, ty):.1f} deg")
            report.append(f"   {t:3d}  {1e3*x:+7.1f} {1e3*y:+7.1f}     {1e3*z:+5.1f}    {rot:+6.1f}    {tx:+5.1f} {ty:+5.1f}    | "
                          f"{1e3*mv['x']:+7.1f} {1e3*mv['y']:+7.1f}  {mv['rot']:+4.0f} |  {dpos:6.1f}            "
                          f"{per_tag_rms[t]:5.2f}   {'; '.join(notes)}")
            out_rows.append((t, f, x, y, z, mv["s"], rot, tx, ty))
        for t in fid:
            if t not in comp:
                v = tmap[t]
                report.append(f"   {t:3d}  not measured (never seen together with the others at close range) - map value kept")
                out_rows.append((t, f, v["x"], v["y"], v["z"], v["s"], v["rot"], 0.0, 0.0))

    print("\n".join(report))
    if os.path.dirname(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fo:
        fo.write("id,face,x_m,y_m,z_m,size_m,rot_deg,tilt_x_deg,tilt_y_deg\n")
        for t, f, x, y, z, s, r, tx, ty in sorted(out_rows):
            fo.write(f"{t},{f},{x:.4f},{y:.4f},{z:.4f},{s:.4f},{r:.2f},{tx:.2f},{ty:.2f}\n")
    print(f"\nwrote {a.out}")
    print("Positions are relative to the centre of each face's tags; z is height above the plane fitted through them.")
    sys.stdout.flush()
    os._exit(0)      # exit before the AprilTag detector is destroyed: avoids a harmless segfault on Linux


if __name__ == "__main__":
    main()
