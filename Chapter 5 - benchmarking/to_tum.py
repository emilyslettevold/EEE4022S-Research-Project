#!/usr/bin/env python3
"""
Convert trajectories to TUM format (t[s] x y z qx qy qz qw) in the LEFT-CAMERA frame,
which is the frame of the Tank ground truth.

    # ORB-SLAM3 SaveTrajectoryEuRoC output (IMU/body pose, t in ns, space-separated)
    python3 to_tum.py --method orb  --input f_orb.txt --output est_tum.txt --seq Structure_Easy
    # VINS-Fusion vio.csv (IMU/body pose, t in ns, comma-separated, qw BEFORE qx)
    python3 to_tum.py --method vins --input vio.csv   --output est_tum.txt --seq Structure_Easy
    # Ground truth gt.csv / gt_full.csv (already left-camera pose, t in ns)
    python3 to_tum.py --method gt   --input gt.csv    --output gt_tum.txt

Both VIO frameworks report the IMU (body) pose. Comparing that directly with the camera
ground truth adds a ~0.24-0.28 m lever arm and a ~90 deg rotation to every error, so
every estimate is converted with T_W_C = T_W_B * T_imu_camera.
"""
import argparse
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot


def load(method, path):
    if method == "orb":
        d = np.loadtxt(path, comments="#", ndmin=2)
        t, p, q = d[:, 0], d[:, 1:4], d[:, 4:8]                     # qx qy qz qw
    elif method == "vins":
        d = np.genfromtxt(path, delimiter=",", usecols=range(8), comments="#")
        d = d[~np.isnan(d).any(axis=1)]
        t, p, q = d[:, 0], d[:, 1:4], d[:, [5, 6, 7, 4]]            # qw qx qy qz -> qx qy qz qw
    elif method == "gt":
        d = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
        t, p, q = d[:, 0], d[:, 1:4], d[:, 4:8]
    else:
        raise ValueError(method)
    if len(t) == 0:
        return t, p, q
    t = t / 1e9 if t[0] > 1e12 else t                                # ns -> s
    order = np.argsort(t, kind="stable")
    t, p, q = t[order], p[order], q[order]
    keep = np.append(np.diff(t) > 0, True)                           # drop duplicate stamps
    return t[keep], p[keep], q[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["orb", "vins", "gt"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params", default="benchmark_params.yaml")
    ap.add_argument("--seq", help="sequence name (selects the extrinsic set)")
    ap.add_argument("--set", help="override extrinsic set (A/B)")
    a = ap.parse_args()

    t, p, q = load(a.method, a.input) if os.path.getsize(a.input) > 0 else (np.array([]),) * 3
    if len(t) == 0:
        raise SystemExit(f"{a.input} contains no poses")
    if a.method in ("orb", "vins"):
        with open(a.params) as fh:
            P = yaml.safe_load(fh)
        s = a.set or P["sequences"][a.seq]
        T = np.array(P["extrinsics"][s], dtype=float)
        U, _, Vt = np.linalg.svd(T[:3, :3])                          # same orthonormalisation as make_configs
        R_bc, t_bc = U @ Vt, T[:3, 3]
        R_wb = Rot.from_quat(q)
        p = p + R_wb.apply(t_bc)
        q = (R_wb * Rot.from_matrix(R_bc)).as_quat()

    np.savetxt(a.output, np.column_stack([t, p, q]), fmt="%.9f",
               header="timestamp tx ty tz qx qy qz qw")
    print(f"{a.method}: {len(t)} poses, {t[-1] - t[0]:.1f} s -> {a.output}")


if __name__ == "__main__":
    main()
