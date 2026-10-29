"""
compare_kalibr.py - compare Kalibr camera-IMU results from several runs.

Each argument is a folder holding one run's outputs:
    stage2-camchain-imucam.yaml   (required)
    stage2-results-imucam.txt     (optional, for the normalised errors)

Usage:
    python compare_kalibr.py results_A_s2_window results_B_s6_window results_C_s6_full

The first folder is the reference; every other run is compared against it.
Needs numpy and pyyaml (pip install pyyaml).
"""
import re
import sys
from pathlib import Path

import numpy as np
import yaml


def load(d):
    d = Path(d)
    cc = yaml.safe_load(open(d / "stage2-camchain-imucam.yaml"))
    run = {"name": d.name}
    for cam in ("cam0", "cam1"):
        if cam in cc:
            run[cam] = {"T": np.array(cc[cam]["T_cam_imu"], dtype=float),
                        "shift": float(cc[cam].get("timeshift_cam_imu", float("nan")))}
    txt = d / "stage2-results-imucam.txt"
    run["err"] = {}
    if txt.exists():
        s = txt.read_text()
        for key, pat in [("reproj0", r"Reprojection error \(cam0\)[^:]*:\s*mean\s*([\d.eE+-]+)"),
                         ("reproj1", r"Reprojection error \(cam1\)[^:]*:\s*mean\s*([\d.eE+-]+)"),
                         ("gyro", r"Gyroscope error \(imu0\):\s*mean\s*([\d.eE+-]+)"),
                         ("accel", r"Accelerometer error \(imu0\):\s*mean\s*([\d.eE+-]+)")]:
            m = re.search(pat, s)
            run["err"][key] = float(m.group(1)) if m else float("nan")
    return run


def rot_angle_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def roll_pitch_yaw(R):
    """ZYX Euler angles (deg) of R, for a readable orientation summary."""
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arcsin(-np.clip(R[2, 0], -1, 1)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return roll, pitch, yaw


runs = [load(d) for d in sys.argv[1:]]
if not runs:
    sys.exit(__doc__)

print("\n=== Per run ===")
for r in runs:
    print(f"\n{r['name']}")
    for cam in ("cam0", "cam1"):
        if cam not in r:
            continue
        T = r[cam]["T"]
        p_imu_in_cam = T[:3, 3]
        cam_in_imu = -T[:3, :3].T @ T[:3, 3]
        print(f"  {cam}: timeshift {r[cam]['shift'] * 1e3:7.2f} ms   "
              f"cam position in IMU frame [m] {np.round(cam_in_imu, 4)}   "
              f"distance {np.linalg.norm(cam_in_imu) * 100:.1f} cm")
    if "cam0" in r and "cam1" in r:
        T01 = r["cam1"]["T"] @ np.linalg.inv(r["cam0"]["T"])
        print(f"  stereo baseline (cam0->cam1): {np.linalg.norm(T01[:3, 3]) * 100:.2f} cm")
    e = r["err"]
    if e:
        print(f"  reprojection mean px: cam0 {e['reproj0']:.3f}, cam1 {e['reproj1']:.3f}   "
              f"normalised gyro {e['gyro']:.2f}, accel {e['accel']:.2f}")

ref = runs[0]
print(f"\n=== Differences from reference: {ref['name']} ===")
print("  targets: rotation < ~1 deg, translation < a few mm, timeshift < ~1-2 ms")
for r in runs[1:]:
    print(f"\n{r['name']}")
    for cam in ("cam0", "cam1"):
        if cam not in r or cam not in ref:
            continue
        R0, R1 = ref[cam]["T"][:3, :3], r[cam]["T"][:3, :3]
        c0 = -R0.T @ ref[cam]["T"][:3, 3]
        c1 = -R1.T @ r[cam]["T"][:3, 3]
        print(f"  {cam}: rotation diff {rot_angle_deg(R0, R1):.3f} deg   "
              f"translation diff {np.linalg.norm(c1 - c0) * 1e3:.1f} mm   "
              f"timeshift diff {(r[cam]['shift'] - ref[cam]['shift']) * 1e3:+.2f} ms")
