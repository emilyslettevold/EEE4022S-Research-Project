#!/usr/bin/env python3
"""
tag_gt.py - AprilTag ground truth by multi-tag PnP against a fixed tag map.

For every frame:
  1. (optional) undistort the image to a pinhole camera using the Kalibr calibration
  2. detect tag36h11 tags (AprilTag 3 via pupil_apriltags)
  3. look up each detected tag in the tag map -> 4 corners in the structure frame W
  4. stack all corners and solve one PnP:  SQPnP (global init) -> Levenberg-Marquardt refine
  5. drop tags whose residual is far above the frame median, re-solve
  6. if only one face is visible, run IPPE to measure the planar ambiguity
  7. gate on tag count / RMS reprojection error / ambiguity and write the pose

Outputs (in --out):
  gt_tum.txt      accepted poses only, TUM format: t[s] tx ty tz qx qy qz qw  (pose of the camera in W)
  gt_frames.csv   every frame, with status and quality measures
  summary.json    counts, % of frames with ground truth, per-tag residuals

Conventions
  W  = structure frame.  F_k = face frame: origin at face centre, x right and y up as seen
       looking at the face, z out of the face.  Face 0 is W unless --faces says otherwise.
  C  = camera, OpenCV convention (x right, y down, z forward).
  Pose written = T_W_C (camera position and orientation in W).

Tag map CSV (--tagmap), one row per tag, as-built values in metres in its face frame:
    id,face,x_m,y_m,size_m
  x_m, y_m = tag centre;  size_m = black square edge (the AprilTag "tag size", e.g. 0.145).

Faces CSV (--faces, optional): T_W_F for each face,  face,tx,ty,tz,qx,qy,qz,qw
  Without it, face 0 is the identity and tags on any other face are ignored.

Example (rail run, one face, frames extracted with ffmpeg + timestamp CSV):
  python3 tag_gt.py --camchain camchain-underwater.yaml --cam cam0 \
      --tagmap tagmap.csv --images frames_left/ --timestamps left_timestamps.csv \
      --out gt_rail_run1 --debug-every 30
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation


# ----------------------------------------------------------------------------- camera
def load_camera(path, cam):
    with open(path) as f:
        cc = yaml.safe_load(f)
    if cam not in cc:
        sys.exit(f"{cam} not found in {path} (have: {list(cc)})")
    c = cc[cam]
    if c.get("camera_model", "pinhole") != "pinhole":
        sys.exit(f"camera_model '{c['camera_model']}' not supported (pinhole only)")
    fu, fv, pu, pv = [float(v) for v in c["intrinsics"]]
    K = np.array([[fu, 0, pu], [0, fv, pv], [0, 0, 1]], float)
    D = np.array(c.get("distortion_coeffs", []), float)
    model = c.get("distortion_model", "none")
    if model == "equi":
        model = "equidistant"
    if model not in ("radtan", "equidistant", "none"):
        sys.exit(f"distortion_model '{model}' not supported (radtan, equidistant)")
    w, h = [int(v) for v in c["resolution"]]
    return K, D, model, (w, h)


class Undistorter:
    """Maps raw images/points to an ideal pinhole camera with matrix self.K.

    mode 'image'  : undistort the whole image, detect on it (recommended; tag edges become straight)
    mode 'points' : detect on the raw image, undistort only the detected corners
    mode 'none'   : images are already rectified (distortion ignored)
    """

    def __init__(self, K, D, model, size, mode):
        self.mode, self.model, self.K0, self.D = mode, model, K, D
        if mode == "none" or model == "none" or not np.any(D):
            self.mode, self.K = "none", K.copy()
            return
        if model == "radtan":
            self.K = K.copy()                      # keep the calibrated focal length and centre
            if mode == "image":
                self.m1, self.m2 = cv2.initUndistortRectifyMap(K, D, None, self.K, size, cv2.CV_32FC1)
        else:  # equidistant
            self.K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D.reshape(4, 1), size, np.eye(3), balance=0.0)
            if mode == "image":
                self.m1, self.m2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D.reshape(4, 1), np.eye(3), self.K, size, cv2.CV_32FC1)

    def image(self, gray):
        if self.mode == "image":
            return cv2.remap(gray, self.m1, self.m2, cv2.INTER_LINEAR)
        return gray

    def points(self, pts):
        if self.mode != "points":
            return pts
        p = pts.reshape(-1, 1, 2).astype(np.float64)
        if self.model == "radtan":
            u = cv2.undistortPoints(p, self.K0, self.D, P=self.K)
        else:
            u = cv2.fisheye.undistortPoints(p, self.K0, self.D.reshape(4, 1), P=self.K)
        return u.reshape(-1, 2)


# ----------------------------------------------------------------------------- tag map
def tag_corners_face(x, y, size, rot_deg=0.0, z=0.0, tilt_x_deg=0.0, tilt_y_deg=0.0):
    """Corners of one tag in its face frame, in the order pupil_apriltags returns them:
    bottom-left, bottom-right, top-right, top-left of the tag itself (its own 'up').
      x, y, z   : tag centre on the face [m] (z = how far it stands proud of the face plane)
      rot_deg   : in-plane rotation, counter-clockwise looking at the face (180 = glued upside down)
      tilt_x/y  : small tilts out of the face plane about the tag's own x / y axes [deg]
    Orientation of the tag in the face frame: R = Rz(rot) * Rx(tilt_x) * Ry(tilt_y)."""
    h = size / 2.0
    local = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]], float)
    R = Rotation.from_euler("ZXY", [rot_deg, tilt_x_deg, tilt_y_deg], degrees=True).as_matrix()
    return local @ R.T + [x, y, z]


def row_opt(r, key):
    v = r.get(key, "")
    return num(v) if str(v).strip() not in ("", None) else 0.0


def row_rot(r):
    return row_opt(r, "rot_deg")


def row_corners(r):
    """Tag corners in the face frame from a tag-map row (optional columns: rot_deg, z_m, tilt_x_deg, tilt_y_deg)."""
    return tag_corners_face(num(r["x_m"]), num(r["y_m"]), num(r["size_m"]), row_opt(r, "rot_deg"),
                            row_opt(r, "z_m"), row_opt(r, "tilt_x_deg"), row_opt(r, "tilt_y_deg"))


def num(s):
    """Number from a CSV cell; accepts a decimal comma (Excel with a South African/European locale)."""
    return float(str(s).strip().replace(",", "."))


def read_table(path, required):
    """Read a CSV saved by Excel or a text editor: handles the UTF-8 BOM, ';' or ',' or tab
    separators, stray spaces/capitals in headers and blank rows."""
    if not os.path.exists(path):
        here = sorted(f for f in os.listdir(os.path.dirname(path) or ".") if f.lower().endswith(".csv"))
        sys.exit(f"file not found: {path}\n  CSV files in that folder: {here}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    first = text.splitlines()[0] if text.strip() else ""
    delim = max([";", ",", "\t"], key=first.count)
    rows = list(csv.reader(text.splitlines(), delimiter=delim))
    head = [h.strip().lower() for h in rows[0]]
    missing = [c for c in required if c not in head]
    if missing:
        sys.exit(f"{path}: missing column(s) {missing}; found header {rows[0]} (separator '{delim}')")
    out = []
    for r in rows[1:]:
        if not r or all(not c.strip() for c in r):
            continue
        out.append({h: (r[i].strip() if i < len(r) else "") for i, h in enumerate(head)})
    return out


def load_tag_map(tag_csv, faces_csv):
    faces = {0: (np.eye(3), np.zeros(3))}
    if faces_csv:
        faces = {}
        with open(faces_csv) as f:
            for r in csv.DictReader(f):
                R = Rotation.from_quat([float(r[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
                faces[int(r["face"])] = (R, np.array([float(r[k]) for k in ("tx", "ty", "tz")]))
    tags, skipped = {}, set()
    for r in read_table(tag_csv, ["id", "face", "x_m", "y_m", "size_m"]):
        tid, face = int(num(r["id"])), int(num(r["face"]))
        if face not in faces:
            skipped.add(face)
            continue
        x, y, s = num(r["x_m"]), num(r["y_m"]), num(r["size_m"])
        if not (0.01 < s < 1.0) or abs(x) > 2 or abs(y) > 2:
            sys.exit(f"tag {tid}: x={x}, y={y}, size={s} - values must be in metres (e.g. 0.145, not 145)")
        rot = row_rot(r)
        cF = row_corners(r)
        R, t = faces[face]
        tags[tid] = {"face": face, "F": cF, "W": cF @ R.T + t}
        print(f"  tag {tid:3d}: face {face}  x {x:+.4f}  y {y:+.4f}  size {s:.4f} m" + (f"  rot {rot:+.1f} deg" if rot else "")
              + (f"  z {1e3*row_opt(r, 'z_m'):+.1f} mm  tilt {row_opt(r, 'tilt_x_deg'):+.1f}/{row_opt(r, 'tilt_y_deg'):+.1f} deg"
                 if any(row_opt(r, k) for k in ('z_m', 'tilt_x_deg', 'tilt_y_deg')) else ""))
    if skipped:
        print(f"note: tags on faces {sorted(skipped)} ignored (no pose for those faces in --faces)")
    if not tags:
        sys.exit("tag map is empty")
    return tags, faces


# ----------------------------------------------------------------------------- frames
def frame_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"cannot open {args.video}")
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield i, None, img
            i += 1
    else:
        files = sorted(sum((glob.glob(os.path.join(args.images, e)) for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif*")), []))
        if not files:
            sys.exit(f"no images in {args.images}")
        for i, fn in enumerate(files):
            yield i, os.path.splitext(os.path.basename(fn))[0], cv2.imread(fn, cv2.IMREAD_UNCHANGED)


def load_timestamps(path, col):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows or col not in rows[0]:
        sys.exit(f"column '{col}' not in {path} (have: {list(rows[0]) if rows else []})")
    return [int(float(r[col])) for r in rows]


# ----------------------------------------------------------------------------- solving
def reproj(obj, img, rvec, tvec, K):
    proj, J = cv2.projectPoints(obj, rvec, tvec, K, None)
    return proj.reshape(-1, 2) - img, J


def solve_pnp(obj, img, K, refine=True):
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    if refine:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    return rvec.reshape(3, 1), tvec.reshape(3, 1)


def camera_centre(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec).ravel()


def position_sigma(obj, img, rvec, tvec, K):
    """1-sigma of the camera centre in W from sigma^2 (J^T J)^-1, propagated through c = -R^T t."""
    res, J = reproj(obj, img, rvec, tvec, K)
    J = J[:, :6]
    dof = 2 * len(obj) - 6
    if dof <= 0:
        return np.full(3, np.nan)
    s2 = float((res ** 2).sum()) / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)
    Jc = np.zeros((3, 6))
    c0, eps = camera_centre(rvec, tvec), 1e-6
    for k in range(3):
        d = np.zeros((3, 1)); d[k] = eps
        Jc[:, k] = (camera_centre(rvec + d, tvec) - c0) / eps
    R, _ = cv2.Rodrigues(rvec)
    Jc[:, 3:] = -R.T
    return np.sqrt(np.clip(np.diag(Jc @ cov @ Jc.T), 0, None))


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def process_frame(dets, tags, faces, K, und, args):
    """Returns a dict with status, pose and quality measures for one frame."""
    out = {"n_det": len(dets), "n_tags": 0, "n_faces": 0, "ids": "", "rms_px": np.nan,
           "max_tag_rms_px": np.nan, "dropped": "", "ambig_ratio": np.nan, "ippe_dangle_deg": np.nan,
           "pose": None, "sigma": np.full(3, np.nan), "per_tag": {}, "corners": []}
    seen = {}
    for d in dets:
        if d.tag_id not in tags or d.hamming > args.max_hamming or d.decision_margin < args.min_margin:
            continue
        if d.tag_id in seen:              # duplicate ID in one image -> at least one is wrong, drop both
            seen[d.tag_id] = None
            continue
        seen[d.tag_id] = und.points(np.asarray(d.corners, float))
    ids = [i for i, c in seen.items() if c is not None]
    out["corners"] = [(i, seen[i]) for i in ids]
    if not ids:
        out["status"] = "no_tags"
        return out

    dropped = []
    for _ in range(args.max_drop + 1):
        obj = np.vstack([tags[i]["W"] for i in ids])
        img = np.vstack([seen[i] for i in ids])
        if len(ids) < args.min_tags:
            break
        sol = solve_pnp(obj, img, K, refine=not args.no_refine)
        if sol is None:
            break
        rvec, tvec = sol
        res, _ = reproj(obj, img, rvec, tvec, K)
        per_tag = np.sqrt((res.reshape(-1, 4, 2) ** 2).sum(2).mean(1))
        worst = int(np.argmax(per_tag))
        med = float(np.median(per_tag))
        if (len(ids) > args.min_tags and len(dropped) < args.max_drop
                and per_tag[worst] > max(args.drop_factor * med, args.drop_floor)):
            dropped.append(ids.pop(worst))
            continue
        break
    out["n_tags"], out["ids"], out["dropped"] = len(ids), " ".join(map(str, ids)), " ".join(map(str, dropped))
    out["n_faces"] = len({tags[i]["face"] for i in ids})
    if len(ids) < args.min_tags:
        out["status"] = "few_tags"
        return out
    if sol is None:
        out["status"] = "solve_failed"
        return out

    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    out.update(rms_px=rms, max_tag_rms_px=float(per_tag.max()),
               per_tag=dict(zip(ids, per_tag.tolist())),
               sigma=position_sigma(obj, img, rvec, tvec, K))

    # planar ambiguity check (only meaningful when all points lie on one face)
    if out["n_faces"] == 1:
        face = tags[ids[0]]["face"]
        objF = np.vstack([tags[i]["F"] for i in ids]).copy()
        objF[:, 2] = 0.0                       # IPPE needs exactly coplanar points (diagnostic only)
        n, rv, tv, err = cv2.solvePnPGeneric(objF, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        err = np.asarray(err).ravel()
        if n >= 2 and err[0] > 0:
            order = np.argsort(err)
            out["ambig_ratio"] = float(err[order[1]] / err[order[0]])
            # does the final pose agree with IPPE's best solution? (compare R_C_F)
            R_CW, _ = cv2.Rodrigues(rvec)
            R_WF = faces[face][0]
            R_ippe, _ = cv2.Rodrigues(rv[order[0]])
            out["ippe_dangle_deg"] = rot_angle_deg(R_CW @ R_WF, R_ippe)

    R_CW, _ = cv2.Rodrigues(rvec)
    R_WC = R_CW.T
    out["pose"] = (R_WC, (-R_WC @ tvec).ravel())

    if rms > args.max_rms:
        out["status"] = "high_rms"
    elif out["n_faces"] == 1 and np.isfinite(out["ambig_ratio"]) and (
            out["ambig_ratio"] < args.min_ambig_ratio or out["ippe_dangle_deg"] > args.max_ippe_dangle):
        out["status"] = "ambiguous"
    else:
        out["status"] = "ok"
    return out


def draw_debug(gray, res, tags, K, path):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    for tid, c in res["corners"]:
        for k, p in enumerate(c):
            cv2.circle(vis, tuple(int(round(v)) for v in p), 6, (0, 200, 0), 2)
        cv2.putText(vis, str(tid), tuple(int(v) for v in c.mean(0)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        cv2.putText(vis, "0", tuple(int(v) + 8 for v in c[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    if res["pose"] is not None:
        R_WC, p = res["pose"]
        R_CW = R_WC.T
        rvec, _ = cv2.Rodrigues(R_CW)
        tvec = -R_CW @ p
        obj = np.vstack([t["W"] for t in tags.values()])
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        for q in proj.reshape(-1, 2):
            if np.all(np.isfinite(q)) and abs(q[0]) < 1e5 and abs(q[1]) < 1e5:
                q = tuple(int(round(v)) for v in q)
                cv2.drawMarker(vis, q, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    txt = f"{res['status']}  tags={res['n_tags']}  rms={res['rms_px']:.2f}px  ambig={res['ambig_ratio']:.1f}"
    cv2.putText(vis, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(path, vis)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of frames (sorted by file name)")
    src.add_argument("--video", help="video file (one camera)")
    ap.add_argument("--camchain", required=True, help="Kalibr camchain yaml (use the UNDERWATER one for pool runs)")
    ap.add_argument("--cam", default="cam0")
    ap.add_argument("--tagmap", required=True)
    ap.add_argument("--faces", help="face poses CSV (omit for a single-face run: face 0 = W)")
    ap.add_argument("--timestamps", help="CSV with one row per frame, in frame order")
    ap.add_argument("--ts-col", default="arrival_monotonic_ns")
    ap.add_argument("--undistort", choices=["image", "points", "none"], default="image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    # detector
    ap.add_argument("--decimate", type=float, default=1.0, help="quad_decimate (1.0 = full resolution corners)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-hamming", type=int, default=0)
    ap.add_argument("--min-margin", type=float, default=20.0, help="minimum decision_margin")
    # solver / gating
    ap.add_argument("--no-refine", action="store_true", help="SQPnP only, skip the LM refinement")
    ap.add_argument("--min-tags", type=int, default=2)
    ap.add_argument("--max-rms", type=float, default=1.0, help="px; set from your own data (see summary)")
    ap.add_argument("--min-ambig-ratio", type=float, default=1.5,
                    help="single-face frames: flag if IPPE 2nd/1st reprojection error is below this")
    ap.add_argument("--max-ippe-dangle", type=float, default=2.0,
                    help="deg; flag if the final pose disagrees with IPPE's best solution")
    ap.add_argument("--max-drop", type=int, default=2, help="max tags dropped as outliers per frame")
    ap.add_argument("--drop-factor", type=float, default=3.0)
    ap.add_argument("--drop-floor", type=float, default=1.0, help="px; never drop a tag below this RMS")
    ap.add_argument("--debug-every", type=int, default=0, help="save an overlay image every N frames")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dbg_dir = os.path.join(args.out, "debug")
    if args.debug_every:
        os.makedirs(dbg_dir, exist_ok=True)

    K0, D, model, size = load_camera(args.camchain, args.cam)
    und = Undistorter(K0, D, model, size, args.undistort)
    K = und.K
    tags, faces = load_tag_map(args.tagmap, args.faces)
    ts = load_timestamps(args.timestamps, args.ts_col) if args.timestamps else None
    det = Detector(families="tag36h11", nthreads=args.threads, quad_decimate=args.decimate,
                   quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    print(f"camera {args.cam}: {model}, {size[0]}x{size[1]}, fx={K0[0,0]:.1f} fy={K0[1,1]:.1f}; "
          f"undistort={und.mode}; {len(tags)} tags on faces {sorted({t['face'] for t in tags.values()})}")

    fields = ["frame", "t_ns", "status", "n_det", "n_tags", "n_faces", "ids", "dropped", "rms_px",
              "max_tag_rms_px", "ambig_ratio", "ippe_dangle_deg", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
              "sig_x_mm", "sig_y_mm", "sig_z_mm"]
    fcsv = open(os.path.join(args.out, "gt_frames.csv"), "w", newline="")
    wr = csv.DictWriter(fcsv, fieldnames=fields)
    wr.writeheader()
    ftum = open(os.path.join(args.out, "gt_tum.txt"), "w")
    ftum.write("# timestamp tx ty tz qx qy qz qw   (T_W_C, camera in structure frame)\n")

    counts, per_tag_all, rms_ok, n_frames, last_i = defaultdict(int), defaultdict(list), [], 0, -1
    for i, stem, img in frame_source(args):
        last_i = i
        if i < args.start or (args.end >= 0 and i > args.end) or (i - args.start) % args.every:
            continue
        if img is None:
            print(f"warning: frame {i} unreadable")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if (gray.shape[1], gray.shape[0]) != size:
            sys.exit(f"frame size {gray.shape[1]}x{gray.shape[0]} != calibration {size}")
        gray = und.image(gray)
        res = process_frame(det.detect(gray), tags, faces, K, und, args)

        if ts is not None:
            t_ns = ts[i] if i < len(ts) else None
        elif stem is not None and stem.isdigit():
            t_ns = int(stem)
        else:
            t_ns = None
        if t_ns is None:
            t_ns = int(round(i * 1e9 / 30.0))   # fallback: frame index at 30 fps

        n_frames += 1
        counts[res["status"]] += 1
        row = {k: res.get(k, "") for k in fields}
        row.update(frame=i, t_ns=t_ns)
        if res["pose"] is not None:
            R_WC, p = res["pose"]
            q = Rotation.from_matrix(R_WC).as_quat()
            row.update(tx=p[0], ty=p[1], tz=p[2], qx=q[0], qy=q[1], qz=q[2], qw=q[3])
            row.update(sig_x_mm=1e3 * res["sigma"][0], sig_y_mm=1e3 * res["sigma"][1], sig_z_mm=1e3 * res["sigma"][2])
            if res["status"] == "ok":
                ftum.write(f"{t_ns * 1e-9:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                           f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n")
                rms_ok.append(res["rms_px"])
                for tid, r in res["per_tag"].items():
                    per_tag_all[tid].append(r)
        for k in ("rms_px", "max_tag_rms_px", "ambig_ratio", "ippe_dangle_deg"):
            if isinstance(row[k], float) and np.isfinite(row[k]):
                row[k] = round(row[k], 4)
            elif isinstance(row[k], float):
                row[k] = ""
        wr.writerow(row)
        if args.debug_every and n_frames % args.debug_every == 1:
            draw_debug(gray, res, tags, K, os.path.join(dbg_dir, f"frame_{i:06d}.jpg"))
        if n_frames % 200 == 0:
            print(f"  {n_frames} frames, {counts['ok']} ok")

    fcsv.close(); ftum.close()
    if ts is not None and last_i + 1 != len(ts):
        print(f"WARNING: {last_i + 1} frames but {len(ts)} timestamps - check frame/timestamp alignment")

    summary = {
        "frames": n_frames, "status_counts": dict(counts),
        "percent_with_pose": round(100.0 * sum(counts[k] for k in ("ok", "ambiguous", "high_rms")) / max(n_frames, 1), 1),
        "percent_with_gt": round(100.0 * counts["ok"] / max(n_frames, 1), 1),
        "rms_px_ok_median": float(np.median(rms_ok)) if rms_ok else None,
        "rms_px_ok_p95": float(np.percentile(rms_ok, 95)) if rms_ok else None,
        "per_tag_rms_px_median": {int(k): round(float(np.median(v)), 3) for k, v in sorted(per_tag_all.items())},
        "settings": {k: v for k, v in vars(args).items()},
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("frames", "status_counts", "percent_with_pose", "percent_with_gt",
                                              "rms_px_ok_median", "rms_px_ok_p95", "per_tag_rms_px_median")}, indent=2))
    sys.stdout.flush()
    os._exit(0)      # exit before the AprilTag detector is destroyed: avoids a harmless segfault on Linux


if __name__ == "__main__":
    main()
