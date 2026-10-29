#!/usr/bin/env python3
"""
Find where a camera video sits inside a (longer) ArduSub IMU log, and the
clock offset between them.

It compares how much the image changes frame-to-frame with how fast the gyro
is rotating, and slides one over the other until they match best.

Usage (inside ~/pymavlink-env; needs numpy pandas opencv-python matplotlib):
    python find_overlap.py --video LEFT.mkv --csv left_timestamps.csv \
                           --gyr potential_imuCamCalib_GYR0.csv
Optional:
    --search-last 400   only search the last 400 s of the log (faster, safer)
"""
import argparse
import sys

import cv2
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
p.add_argument("--csv", required=True)
p.add_argument("--gyr", required=True)
p.add_argument("--search-last", type=float, default=None,
               help="only search the last N seconds of the IMU log")
p.add_argument("--skip-start", type=float, default=1.0,
               help="seconds of frames to ignore at the start")
p.add_argument("--grid-ms", type=float, default=20.0)
p.add_argument("--min-overlap", type=float, default=60.0,
               help="minimum seconds of video that must overlap the log")
args = p.parse_args()
STEP_NS = int(args.grid_ms * 1e6)

# ------------------------------------------------------------ camera side
ts = pd.read_csv(args.csv)
t_ns = (ts["pts_ns"].astype(np.int64) + ts["base_time_ns"].astype(np.int64)).to_numpy()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    sys.exit(f"Could not open {args.video}")
motion, prev, n = [], None, 0
print("Reading video (no frames are saved) ...")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 320.0 / g.shape[1]
    small = cv2.resize(g, None, fx=scale, fy=scale).astype(np.float32)
    motion.append(0.0 if prev is None else float(np.mean(np.abs(small - prev))))
    prev = small
    n += 1
cap.release()

if n != len(t_ns):
    print(f"WARNING: {n} frames in video, {len(t_ns)} rows in CSV - using first {min(n, len(t_ns))}")
m = min(n, len(t_ns))
t_ns, motion = t_ns[:m], np.array(motion[:m])
# the change between frame i-1 and i describes motion halfway between them
t_ns = np.concatenate([[t_ns[0]], (t_ns[1:] + t_ns[:-1]) // 2])

frame_dt = np.median(np.diff(t_ns)) * 1e-9
skip = int(round(args.skip_start / frame_dt))
t_ns, motion = t_ns[skip:], motion[skip:]
motion[0] = motion[1]
dts = np.diff(t_ns) * 1e-9
print(f"Video: {m} frames, {1 / frame_dt:.1f} fps, {(t_ns[-1] - t_ns[0]) * 1e-9:.1f} s long, "
      f"{int(np.sum(dts > 1.5 * frame_dt))} dropped-frame gaps")

# --------------------------------------------------------------- IMU side
print("Reading gyro CSV ...")
gy = pd.read_csv(args.gyr, usecols=["SampleUS", "GyrX", "GyrY", "GyrZ"])
log_t0_us = gy.SampleUS.iloc[0]
mag = np.sqrt(gy.GyrX ** 2 + gy.GyrY ** 2 + gy.GyrZ ** 2)
bins = ((gy.SampleUS - log_t0_us) // (STEP_NS / 1000)).astype(np.int64)
imu_mag = mag.groupby(bins).mean()
full = np.arange(imu_mag.index.max() + 1)
imu_mag = imu_mag.reindex(full).interpolate().to_numpy()
imu_grid_ns = (log_t0_us * 1000 + (full + 0.5) * STEP_NS).astype(np.int64)
log_len = len(imu_grid_ns) * args.grid_ms / 1000
print(f"IMU log: {log_len:.1f} s long")

cam_grid = np.arange(t_ns[0], t_ns[-1], STEP_NS)
cam_sig = np.interp(cam_grid, t_ns, motion)
cam_z = (cam_sig - cam_sig.mean()) / cam_sig.std()
N = len(cam_z)

start_idx = 0
if args.search_last is not None:
    start_idx = max(0, len(imu_mag) - int(args.search_last * 1000 / args.grid_ms))
search = imu_mag[start_idx:]
min_ov = int(args.min_overlap * 1000 / args.grid_ms)
if min_ov > N:
    min_ov = N
if len(search) < min_ov:
    sys.exit("The part of the log being searched is shorter than --min-overlap.")


def partial_pearson(sig, v):
    """
    Pearson correlation of template v against sig for every placement where
    at least min_ov samples overlap. The video is allowed to hang off either
    end of the log (e.g. the log was stopped before the video ended).
    Returns r, and for each placement the start index of v inside sig
    (may be negative or run past the end).
    """
    n, M = len(v), len(sig)
    P = n - min_ov                      # how far the video may hang off an end
    a = np.concatenate([np.zeros(P), sig, np.zeros(P)])
    mask = np.concatenate([np.zeros(P), np.ones(M), np.zeros(P)])
    J = len(a) - n + 1
    j = np.arange(J)

    Sxy = np.correlate(a, v, mode="valid")
    ca = np.cumsum(np.insert(a, 0, 0.0))
    ca2 = np.cumsum(np.insert(a ** 2, 0, 0.0))
    Sx = ca[j + n] - ca[j]
    Sxx = ca2[j + n] - ca2[j]

    lo = np.clip(P - j, 0, n)           # first video index with log data
    hi = np.clip(P + M - j, 0, n)       # one past last video index with log data
    L = (hi - lo).astype(float)
    cv = np.cumsum(np.insert(v, 0, 0.0))
    cv2 = np.cumsum(np.insert(v ** 2, 0, 0.0))
    Sy = cv[hi] - cv[lo]
    Syy = cv2[hi] - cv2[lo]

    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy - Sx * Sy / L
        vx = Sxx - Sx ** 2 / L
        vy = Syy - Sy ** 2 / L
        r = cov / np.sqrt(vx * vy)
    bad = (L < min_ov) | (vx < 1e-6 * L) | (vy < 1e-6 * L)
    r = np.where(bad, -1.0, r)
    return r, j - P, L


print("Searching for the best match (video may run past the ends of the log) ...")
r, pos, L = partial_pearson(search, cam_sig)
# Short overlaps match by chance more easily, so reward covering more video
score = np.where(r > 0, r * L / N, -1.0)
k = int(np.argmax(score))
best = r[k]
guard = int(5000 / args.grid_ms)
s2 = score.copy()
s2[max(0, k - guard):k + guard] = -1
k2 = int(np.argmax(s2))
second = r[k2]

k_abs = start_idx + int(pos[k])        # imu grid index of the video's first sample
offset_ns = int(cam_grid[0] - (imu_grid_ns[0] + k_abs * STEP_NS))
vid_start_s = (imu_grid_ns[0] - log_t0_us * 1000) * 1e-9 + k_abs * args.grid_ms / 1000
vid_len_s = (cam_grid[-1] - cam_grid[0]) * 1e-9
vid_end_s = vid_start_s + vid_len_s

# which part of the video actually has IMU data
v_lo = max(0, -k_abs)
v_hi = min(N, len(imu_mag) - k_abs)
cov_lo_s, cov_hi_s = v_lo * args.grid_ms / 1000, v_hi * args.grid_ms / 1000

print("\n=== Result ===")
print(f"Best match correlation:   {best:.2f}   (good: > 0.5), score {score[k]:.2f}")
print(f"Next-best elsewhere:      {second:.2f}, score {s2[k2]:.2f}   (good: score clearly lower)")
print(f"Video would span log time: {vid_start_s:.1f} s -> {vid_end_s:.1f} s "
      f"(log is {log_len:.1f} s)")
print(f"Video with IMU data:      video {cov_lo_s:.1f} s -> {cov_hi_s:.1f} s "
      f"({cov_hi_s - cov_lo_s:.1f} of {vid_len_s:.1f} s)")
if cov_hi_s - cov_lo_s < vid_len_s - 1:
    print("NOTE: part of the video has no IMU data (log started late or stopped early).")
print(f"Clock offset (Pi - IMU):  {offset_ns} ns")

# ------------------------------------------ drift check: halves of the overlap
print("\n=== Drift check (halves of the overlapping part, searched within +-3 s) ===")
local = int(3000 / args.grid_ms)
mid = (v_lo + v_hi) // 2
half_offsets = []
for name, a0, a1 in [("first half ", v_lo, mid), ("second half", mid, v_hi)]:
    seg = cam_sig[a0:a1]
    centre = k_abs + a0
    lo = max(0, centre - local)
    hi = min(len(imu_mag), centre + local + len(seg))
    if hi - lo < len(seg) or seg.std() < 1e-9:
        print(f"{name}: not enough data to check")
        continue
    seg_z = (seg - seg.mean()) / seg.std()
    region = imu_mag[lo:hi]
    num = np.correlate(region, seg_z, mode="valid")
    cs = np.cumsum(np.insert(region, 0, 0.0)); cs2 = np.cumsum(np.insert(region ** 2, 0, 0.0))
    nn = len(seg_z)
    sd = np.sqrt(np.maximum((cs2[nn:] - cs2[:-nn]) / nn - ((cs[nn:] - cs[:-nn]) / nn) ** 2, 0))
    rr = np.where(sd > 1e-3, num / (nn * np.maximum(sd, 1e-12)), -1)
    kk = lo + int(np.argmax(rr))
    off = int(cam_grid[a0] - (imu_grid_ns[0] + kk * STEP_NS))
    half_offsets.append(off)
    print(f"{name}: offset {off} ns, correlation {rr.max():.2f}")
if len(half_offsets) == 2:
    diff_ms = abs(half_offsets[1] - half_offsets[0]) / 1e6
    print(f"Difference: {diff_ms:.0f} ms   (one frame = {frame_dt * 1e3:.0f} ms, "
          f"search step = {args.grid_ms:.0f} ms)")
    if diff_ms > frame_dt * 1e3:
        print("WARNING: halves disagree by more than one frame - clocks may drift, or the match is wrong.")

if best < 0.5 or s2[k2] > 0.8 * score[k]:
    print("\nWARNING: match is weak or ambiguous - check the plot before using this offset.")

print("\nIf the plot looks right, use it with:")
print(f"    python prep_kalibr_stage2.py ... --offset-ns {offset_ns}")

# ------------------------------------------------------------------- plot
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    tl = (imu_grid_ns - log_t0_us * 1000) * 1e-9
    ax[0].plot(tl, imu_mag, lw=0.5)
    ax[0].axvspan(vid_start_s, vid_end_s, color="orange", alpha=0.3, label="video")
    ax[0].set_xlim(min(0, vid_start_s) - 10, max(log_len, vid_end_s) + 10)
    ax[0].axvline(log_len, color="red", ls="--", lw=1, label="log ends")
    ax[0].set_xlabel("time since log start (s)")
    ax[0].set_ylabel("|gyro| (rad/s)")
    ax[0].set_title("Whole IMU log - orange = where the video was matched")
    ax[0].legend()

    tv = (cam_grid - cam_grid[0]) * 1e-9
    imu_on_cam = np.full(N, np.nan)
    imu_on_cam[v_lo:v_hi] = imu_mag[k_abs + v_lo:k_abs + v_hi]
    ov = imu_on_cam[v_lo:v_hi]
    cz = cam_sig[v_lo:v_hi]
    ax[1].plot(tv, (cam_sig - cz.mean()) / cz.std(), label="image change (video)")
    ax[1].plot(tv, (imu_on_cam - ov.mean()) / ov.std(), label="|gyro| (shifted)", alpha=0.7)
    if v_hi < N:
        ax[1].axvline(tv[v_hi - 1], color="red", ls="--", lw=1, label="IMU log ends")
    if v_lo > 0:
        ax[1].axvline(tv[v_lo], color="red", ls="--", lw=1, label="IMU log starts")
    ax[1].set_xlabel("time since start of video (s)")
    ax[1].set_ylabel("normalised")
    ax[1].set_title("Zoomed in - the peaks should sit on top of each other")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("overlap_check.png", dpi=120)
    print("\nSaved overlap_check.png")
except ImportError:
    print("(matplotlib not installed - skipping plot)")