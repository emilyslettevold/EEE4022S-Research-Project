#!/usr/bin/env python3
"""
imu_excitation.py - how much was the IMU actually excited on each recording?  Raw imu0.csv only.

  source ~/evo-env/bin/activate
  python3 ~/vio_data/imu_excitation.py fig8_3 orbit3 line1 rail170 square2 lawnmower2
  python3 ~/vio_data/imu_excitation.py square2=~/Desktop/CampsBay/TEST4/square2/dataset/imu0.csv   # explicit file

imu0.csv is looked for in ~/vio_data/NAME/dataset_half/, ~/Desktop/CampsBay/TEST4/NAME/dataset/,
~/Desktop/CampsBay/TEST4/NAME/ (first found). Columns: timestamp, gyro x y z (rad/s), accel x y z (m/s^2).

Columns of the table (also written to ~/vio_data/testD_eval/imu_excitation.csv):
  accel_noise     median spread of the RAW accelerometer samples in 1 s windows (m/s^2). Mostly sensor noise:
                  this is what si_divergence.py called "accelerometer variation". Not a measure of motion.
  accel_motion    accelerometer averaged over 0.5 s (removes most noise), then its spread over 5 s windows,
                  median (m/s^2). Low-frequency changes in specific force = real acceleration + gravity
                  re-projected by tilting. This is the excitation that matters for bias and scale.
  accel_motion_p90  same, 90th percentile (the most excited 10% of the recording)
  yaw_total       total rotation about the gravity direction (deg) - turning
  tilt_total      total rotation about horizontal axes (deg) - roll + pitch; needed to separate the
                  accelerometer bias from gravity
  tilt_range      largest angle between the gravity direction (0.5 s means) and its average (deg)
  turning_%       share of time with rotation rate above 5 deg/s
  duration_s
Gyro and accel are used as measured (bias not removed); a gyro bias of a few 0.1 deg/s adds a small amount
to yaw_total/tilt_total over a long recording - compare recordings, not absolute values.
"""
import csv, sys
from pathlib import Path
import numpy as np

H = Path.home()


def find(name):
    for p in (H / "vio_data" / name / "dataset_half" / "imu0.csv",
              H / "Desktop/CampsBay/TEST4" / name / "dataset" / "imu0.csv",
              H / "Desktop/CampsBay/TEST4" / name / "imu0.csv"):
        if p.exists():
            return p
    return None


def load(p):
    rows = [r for r in csv.reader(open(p)) if r and r[0][:1].isdigit()]
    d = np.array([[float(x) for x in r[:7]] for r in rows])
    t = d[:, 0] / 1e9 if d[0, 0] > 1e11 else d[:, 0]
    return t, d[:, 1:4], d[:, 4:7]


def block_means(t, X, w):
    """means of X over consecutive blocks of w seconds"""
    n = max(1, int(round(w / np.median(np.diff(t)))))
    m = len(t) // n
    return t[: m * n].reshape(m, n).mean(1), X[: m * n].reshape(m, n, 3).mean(1)


def analyse(name, p):
    t, g, a = load(p)
    dt = np.median(np.diff(t))
    # 1. raw spread in 1 s windows (noise floor)
    n1 = max(5, int(round(1.0 / dt))); m1 = len(t) // n1
    noise = np.median(np.linalg.norm(a[: m1 * n1].reshape(m1, n1, 3).std(1), axis=1))
    # 2. low-frequency excitation: 0.5 s means, spread over 5 s windows (10 means)
    tm, am = block_means(t, a, 0.5)
    k = 10; mm = len(tm) // k
    spread = np.linalg.norm(am[: mm * k].reshape(mm, k, 3).std(1), axis=1)
    # 3. rotation split about gravity (yaw) and horizontal (tilt), gravity from the 0.5 s accel means
    gdir = am / np.linalg.norm(am, axis=1, keepdims=True)
    gm = gdir.mean(0); gm /= np.linalg.norm(gm)
    tg, gyro_m = block_means(t, g, 0.5)
    L = min(len(gyro_m), len(gdir))
    w = gyro_m[:L]; up = gdir[:L]
    yaw_rate = np.abs(np.sum(w * up, axis=1))
    tilt_rate = np.linalg.norm(w - np.sum(w * up, axis=1, keepdims=True) * up, axis=1)
    yaw_total = np.degrees(np.sum(yaw_rate) * 0.5)
    tilt_total = np.degrees(np.sum(tilt_rate) * 0.5)
    tilt_range = np.degrees(np.max(np.arccos(np.clip(gdir @ gm, -1, 1))))
    turning = 100 * np.mean(np.degrees(np.linalg.norm(w, axis=1)) > 5)
    return dict(recording=name, duration_s=round(t[-1] - t[0], 1),
                accel_noise=round(float(noise), 3), accel_motion=round(float(np.median(spread)), 3),
                accel_motion_p90=round(float(np.percentile(spread, 90)), 3),
                yaw_total=round(float(yaw_total)), tilt_total=round(float(tilt_total)),
                tilt_range=round(float(tilt_range), 1), **{"turning_%": round(float(turning), 1)})


args = sys.argv[1:] or sys.exit(__doc__)
rows = []
for a_ in args:
    name, _, path = a_.partition("=")
    p = Path(path).expanduser() if path else find(name)
    if p is None or not p.exists():
        print(f"{name}: no imu0.csv found - give it as {name}=/path/to/imu0.csv"); continue
    rows.append(analyse(name, p))
if not rows:
    sys.exit("nothing analysed")
cols = ["recording", "duration_s", "accel_noise", "accel_motion", "accel_motion_p90",
        "yaw_total", "tilt_total", "tilt_range", "turning_%"]
wd = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
print("  ".join(c.rjust(wd[c]) for c in cols))
for r in rows:
    print("  ".join(str(r[c]).rjust(wd[c]) for c in cols))
out = H / "vio_data" / "testD_eval" / "imu_excitation.csv"
try:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=cols); w_.writeheader(); w_.writerows(rows)
    print(f"\nwrote {out}")
except OSError:
    pass
