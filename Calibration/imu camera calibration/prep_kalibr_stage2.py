#!/usr/bin/env python3
"""
Build a Kalibr camera-IMU dataset from:
  - left/right .mkv videos + timestamp CSVs from stereo_cam.py
  - EITHER --imu0: imu0.csv from extract_imu.py (already on the Pi clock,
    timesync-aligned - no offset needed)
  - OR --gyr/--acc: GYR/ACC CSVs from check_kalibr_imu_log.py
    (needs --offset-ns from find_overlap.py, or auto-detects it)

Writes:
  <out>/cam0/<pi_clock_ns>.png   (left, greyscale)
  <out>/cam1/<pi_clock_ns>.png   (right, greyscale)
  <out>/imu0.csv                 (200 Hz, IMU timestamps moved onto the Pi clock)
  alignment_check.png            (plot to confirm the clocks line up)

Run inside ~/pymavlink-env (needs: numpy pandas opencv-python matplotlib).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--left-video", required=True)
p.add_argument("--left-csv", required=True)
p.add_argument("--right-video", required=True)
p.add_argument("--right-csv", required=True)
p.add_argument("--imu0", default=None,
               help="imu0.csv from extract_imu.py (already on the Pi clock)")
p.add_argument("--gyr", default=None, help="*_GYR0.csv")
p.add_argument("--acc", default=None, help="*_ACC0.csv")
p.add_argument("--out", default="dataset")
p.add_argument("--imu-rate", type=float, default=200.0)
p.add_argument("--skip-start", type=float, default=1.0,
               help="seconds of frames to drop at the start (frame 0 is unreliable)")
p.add_argument("--stride", type=int, default=1,
               help="keep every Nth frame (2 = half the images, e.g. 7.5 Hz "
                    "from 15 fps) - saves disk space and Kalibr run time")
p.add_argument("--offset-ns", type=int, default=None,
               help="use this IMU->Pi clock offset instead of detecting it")
args = p.parse_args()

if args.imu0 and (args.gyr or args.acc):
    sys.exit("Use either --imu0, or --gyr and --acc - not both.")
if not args.imu0 and not (args.gyr and args.acc):
    sys.exit("Give --imu0, or both --gyr and --acc.")
if args.imu0 and args.offset_ns is not None:
    sys.exit("--imu0 is already on the Pi clock - do not pass --offset-ns.")

out = Path(args.out)
if out.exists():
    sys.exit(f"'{out}' already exists - delete or rename it first.")


# ---------------------------------------------------------------- cameras
def extract(video, csv, cam_name):
    ts = pd.read_csv(csv)
    t_ns = (ts["pts_ns"].astype(np.int64) + ts["base_time_ns"].astype(np.int64)).to_numpy()

    tmp = out / f"_tmp_{cam_name}"
    tmp.mkdir(parents=True)
    print(f"[{cam_name}] extracting frames from {video} ...")
    # greyscale straight from ffmpeg: ~3x smaller than colour PNGs
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", video,
                    "-fps_mode", "passthrough", "-pix_fmt", "gray",
                    str(tmp / "%06d.png")], check=True)
    files = sorted(tmp.glob("*.png"))

    if len(files) != len(t_ns):
        print(f"[{cam_name}] WARNING: {len(files)} frames in video but "
              f"{len(t_ns)} rows in CSV - using the first {min(len(files), len(t_ns))}")
    n = min(len(files), len(t_ns))
    files, t_ns = files[:n], t_ns[:n]

    dt = np.median(np.diff(t_ns)) * 1e-9
    n_skip = int(round(args.skip_start / dt))
    print(f"[{cam_name}] {n} frames at {1 / dt:.1f} fps, dropping first {n_skip}, "
          f"keeping every {args.stride} -> ~{1 / dt / args.stride:.1f} Hz")

    cam_dir = out / cam_name
    cam_dir.mkdir()
    kept_t, motion, prev = [], [], None
    for i, (f, t) in enumerate(zip(files[n_skip:], t_ns[n_skip:])):
        if i % args.stride:
            f.unlink()          # not kept - free the space straight away
            continue
        if kept_t and t <= kept_t[-1]:
            print(f"[{cam_name}] WARNING: timestamp not increasing at {t}, frame skipped")
            f.unlink()
            continue
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        f.rename(cam_dir / f"{t}.png")   # move, not copy - no second set of files
        small = cv2.resize(g, None, fx=0.25, fy=0.25).astype(np.float32)
        motion.append(0.0 if prev is None else float(np.mean(np.abs(small - prev))))
        prev = small
        kept_t.append(t)
    shutil.rmtree(tmp)
    return np.array(kept_t, dtype=np.int64), np.array(motion), dt


t_left, img_motion, frame_dt = extract(args.left_video, args.left_csv, "cam0")
# image change between frames i-1 and i describes motion halfway between them
t_left_mid = np.concatenate([[t_left[0]], (t_left[1:] + t_left[:-1]) // 2])
t_right, _, _ = extract(args.right_video, args.right_csv, "cam1")

# -------------------------------------------------------------------- IMU
if args.imu0:
    d = pd.read_csv(args.imu0)
    d.columns = ["t", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
    imu = d[["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]]
    imu_t_ns = d["t"].to_numpy(dtype=np.int64)
    gaps = int(np.sum(np.diff(imu_t_ns) > 2 * np.median(np.diff(imu_t_ns))))
    print(f"[imu] {len(imu)} samples from {args.imu0} "
          f"({1e9 / np.median(np.diff(imu_t_ns)):.0f} Hz, {gaps} gaps), already on Pi clock")
else:
    g = pd.read_csv(args.gyr)
    a = pd.read_csv(args.acc)
    period_us = 1e6 / args.imu_rate
    t0 = max(g.SampleUS.min(), a.SampleUS.min())
    t1 = min(g.SampleUS.max(), a.SampleUS.max())
    g = g[(g.SampleUS >= t0) & (g.SampleUS < t1)].copy()
    a = a[(a.SampleUS >= t0) & (a.SampleUS < t1)].copy()
    g["bin"] = ((g.SampleUS - t0) // period_us).astype(np.int64)
    a["bin"] = ((a.SampleUS - t0) // period_us).astype(np.int64)
    imu = (g.groupby("bin")[["GyrX", "GyrY", "GyrZ"]].mean()
           .join(a.groupby("bin")[["AccX", "AccY", "AccZ"]].mean(), how="inner"))
    missing = int(np.sum(np.diff(imu.index.to_numpy()) > 1))
    imu_t_ns = ((t0 + (imu.index.to_numpy() + 0.5) * period_us) * 1000).astype(np.int64)
    print(f"[imu] {len(imu)} samples at {args.imu_rate:.0f} Hz, {missing} missing bins")
gyro_mag = np.linalg.norm(imu[["GyrX", "GyrY", "GyrZ"]].to_numpy(), axis=1)

# -------------------------------------------------------------- alignment
STEP = 20_000_000  # 20 ms grid


def norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


cam_grid = np.arange(t_left[0], t_left[-1], STEP)
cam_sig = norm(np.interp(cam_grid, t_left_mid, img_motion))
imu_grid = np.arange(imu_t_ns[0], imu_t_ns[-1], STEP)
imu_sig = norm(np.interp(imu_grid, imu_t_ns, gyro_mag))

if args.imu0:
    offset = 0
    print("[align] --imu0: timestamps already on the Pi clock (timesync), offset 0")
elif args.offset_ns is not None:
    offset = args.offset_ns
    print(f"[align] using offset given on command line: {offset} ns")
else:
    if len(imu_sig) < len(cam_sig):
        sys.exit("IMU log is shorter than the video - the log must cover the whole recording.")
    corr = np.correlate(imu_sig, cam_sig, mode="valid") / len(cam_sig)
    k = int(np.argmax(corr))
    offset = int(cam_grid[0] - imu_grid[k])
    print(f"[align] best match: correlation {corr[k]:.2f} "
          f"(good: > 0.5), offset = {offset} ns")

    # Check the start and end halves separately to detect clock drift
    half = len(cam_sig) // 2
    offsets = []
    for name, sl in [("first half", slice(0, half)), ("second half", slice(half, None))]:
        seg = cam_sig[sl]
        seg_t0 = cam_grid[sl][0]
        c = np.correlate(imu_sig, seg, mode="valid")
        kk = int(np.argmax(c))
        offsets.append(int(seg_t0 - imu_grid[kk]))
        print(f"[align] {name}: offset = {offsets[-1]} ns")
    drift_ms = abs(offsets[1] - offsets[0]) / 1e6
    print(f"[align] start/end difference: {drift_ms:.1f} ms "
          f"(one frame = {frame_dt * 1e3:.0f} ms)")
    if drift_ms > frame_dt * 1e3:
        print("[align] WARNING: halves disagree by more than one frame - "
              "check the plot; clocks may drift or detection failed.")

# ------------------------------------------------------------ write files
imu_pi_ns = imu_t_ns + offset
cam_lo = min(t_left[0], t_right[0])
cam_hi = max(t_left[-1], t_right[-1])
keep = (imu_pi_ns >= cam_lo - 1_000_000_000) & (imu_pi_ns <= cam_hi + 1_000_000_000)
if imu_pi_ns[0] > cam_lo or imu_pi_ns[-1] < cam_hi:
    print("[write] WARNING: IMU does not cover all frames - removing uncovered frames")
    for cam in ["cam0", "cam1"]:
        for f in (out / cam).glob("*.png"):
            t = int(f.stem)
            if t < imu_pi_ns[0] or t > imu_pi_ns[-1]:
                f.unlink()

vals = imu[["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]].to_numpy()[keep]
df = pd.DataFrame(vals, columns=["omega_x", "omega_y", "omega_z",
                                 "alpha_x", "alpha_y", "alpha_z"])
df.insert(0, "timestamp", imu_pi_ns[keep])
df.to_csv(out / "imu0.csv", index=False)
print(f"[write] {out}/imu0.csv: {len(df)} rows")
print(f"[write] {out}/cam0: {len(list((out / 'cam0').glob('*.png')))} images")
print(f"[write] {out}/cam1: {len(list((out / 'cam1').glob('*.png')))} images")

# ------------------------------------------------------------------- plot
try:
    import matplotlib.pyplot as plt
    tt = (cam_grid - cam_grid[0]) * 1e-9
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tt, cam_sig, label="image change (left cam)")
    ax.plot(tt, norm(np.interp(cam_grid, imu_pi_ns, gyro_mag)),
            label="gyro magnitude (shifted onto Pi clock)", alpha=0.7)
    ax.set_xlabel("seconds since first kept frame")
    ax.set_ylabel("normalised")
    ax.legend()
    ax.set_title("Peaks should line up, especially the jerks at each end")
    fig.tight_layout()
    fig.savefig("alignment_check.png", dpi=120)
    print("[plot] saved alignment_check.png")
except ImportError:
    print("[plot] matplotlib not installed - skipping")