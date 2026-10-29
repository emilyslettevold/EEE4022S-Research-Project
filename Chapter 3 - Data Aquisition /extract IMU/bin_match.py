#!/usr/bin/env python3
"""
bin_match.py - which BIN does this recording belong to?

The timesync file places the recording at the right TIME in every BIN, so the
wrong BIN still "covers" the video. This checks the CONTENT instead: it
compares how much the left camera image changes frame to frame with how fast
the gyro says the ROV was rotating, at exactly the timesync alignment (no
searching). The right BIN correlates clearly; wrong BINs give ~0.

Usage:
    python bin_match.py RECORDING_FOLDER CANDIDATE_IMU_CSV [CANDIDATE_IMU_CSV ...]

Example (after extracting each BIN to /tmp/bincheck/*.csv):
    python bin_match.py ~/Desktop/CampsBay/TEST4/figure_8_1 /tmp/bincheck/*.csv

Needs numpy and opencv-python.
"""
import os
import sys

import cv2
import numpy as np

if len(sys.argv) < 3:
    sys.exit(__doc__)
rec, cands = sys.argv[1], sys.argv[2:]
TD = 0.0189          # Kalibr time shift (t_imu = t_cam + TD); small effect here

t_cam = np.loadtxt(os.path.join(rec, "left_timestamps.csv"), delimiter=",",
                   skiprows=1, usecols=2).astype(np.int64)

cap = cv2.VideoCapture(os.path.join(rec, "left.mkv"))
if not cap.isOpened():
    sys.exit("Could not open left.mkv")
motion, prev = [], None
print("Reading left.mkv ...")
while True:
    ok, f = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1]))).astype(np.float32)
    motion.append(np.nan if prev is None else float(np.mean(np.abs(s - prev))))
    prev = s
cap.release()
n = min(len(motion), len(t_cam))
motion = np.array(motion[:n])
# change between frame i-1 and i happens half way between them
t_mid = np.concatenate([[t_cam[0]], (t_cam[1:n] + t_cam[:n - 1]) // 2])
ok = ~np.isnan(motion)
t_mid, motion = t_mid[ok], motion[ok]
print(f"{n} frames, {(t_cam[n-1]-t_cam[0])/1e9:.1f} s\n")


def smooth(x, k=15):
    # 'valid' = no zero padding, so the ends of the signals cannot create a
    # fake correlation (both outputs shrink by k-1 samples equally)
    return np.convolve(x, np.ones(k) / k, mode="valid")


results = []
for c in cands:
    d = np.loadtxt(c, delimiter=",", skiprows=1)
    ti = d[:, 0].astype(np.int64)
    w = np.linalg.norm(d[:, 1:4], axis=1)
    tq = t_mid + int(TD * 1e9)
    cover = np.mean((tq >= ti[0]) & (tq <= ti[-1]))
    if cover < 0.95:
        print(f"{os.path.basename(c)}: IMU covers only {100*cover:.0f}% of the video - skipped")
        continue
    g = np.interp(tq, ti, w)
    if np.std(smooth(g)) < 1e-6:
        print(f"{os.path.basename(c)}: gyro constant during the video - cannot compare")
        continue
    r = np.corrcoef(smooth(motion), smooth(g))[0, 1]
    # also check a shifted version: a wrong BIN should not get better when shifted
    best_lag, best_r = 0, r
    for lag_s in np.arange(-10, 10.01, 0.25):
        gl = np.interp(t_mid + int((TD + lag_s) * 1e9), ti, w)
        rl = np.corrcoef(smooth(motion), smooth(gl))[0, 1]
        if rl > best_r:
            best_lag, best_r = lag_s, rl
    # gyro std DURING the video (not the whole extracted window)
    results.append((os.path.basename(c), r, best_lag, best_r, np.std(g)))

print(f"{'candidate':<20}{'r at timesync':>15}{'best lag (s)':>14}{'r at best':>11}{'gyro std*':>10}")
for name, r, lag, rb, s in sorted(results, key=lambda x: -x[1]):
    print(f"{name:<20}{r:>15.2f}{lag:>14.2f}{rb:>11.2f}{s:>10.3f}")
print("\n* gyro std = rotation-rate variation during the video only.")
print("Right BIN: clearly highest 'r at timesync' (typically > 0.4) AND best lag ~0.\n"
      "Wrong BINs: r near 0, and any 'best lag' is far from 0 (chance match).\n"
      "If every candidate has a tiny gyro std, the ROV barely rotated during the video\n"
      "and this test cannot decide - use the session start time / Log Browser instead.")
