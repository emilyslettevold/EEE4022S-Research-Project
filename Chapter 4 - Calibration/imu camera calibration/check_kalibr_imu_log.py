#!/usr/bin/env python3
"""
Check an ArduSub .BIN log is usable for Kalibr camera-IMU calibration.

Usage (inside your pymavlink venv):
    python check_kalibr_imu_log.py path/to/log.BIN
"""
import sys
from collections import Counter, defaultdict

import numpy as np
from pymavlink import mavutil

path = sys.argv[1]
INST = 0  # 0 = IMU1

log = mavutil.mavlink_connection(path)
types = Counter()
params = {}
data = defaultdict(list)   # 'IMU', 'GYR', 'ACC' -> rows
dropped = []               # DSF dropped-message counts
clips = []                 # VIBE clip counts

while True:
    m = log.recv_match()
    if m is None:
        break
    t = m.get_type()
    types[t] += 1
    if t == "PARM":
        params[m.Name] = m.Value
    elif t == "IMU" and getattr(m, "I", 0) == INST:
        data["IMU"].append([m.TimeUS, m.GyrX, m.GyrY, m.GyrZ,
                            m.AccX, m.AccY, m.AccZ])
    elif t == "GYR" and getattr(m, "I", 0) == INST:
        ts = getattr(m, "SampleUS", m.TimeUS)
        data["GYR"].append([ts, m.GyrX, m.GyrY, m.GyrZ])
    elif t == "ACC" and getattr(m, "I", 0) == INST:
        ts = getattr(m, "SampleUS", m.TimeUS)
        data["ACC"].append([ts, m.AccX, m.AccY, m.AccZ])
    elif t == "DSF":
        dropped.append(getattr(m, "Dp", 0))
    elif t == "VIBE" and getattr(m, "IMU", 0) == INST:
        clips.append(getattr(m, "Clip", getattr(m, "Clip0", 0)))

print("\n=== Message types in log ===")
for name, n in sorted(types.items()):
    print(f"{name:10s} {n}")

print("\n=== Settings recorded in the log ===")
for p in ["LOG_BITMASK", "LOG_DISARMED", "SCHED_LOOP_RATE",
          "INS_GYRO_FILTER", "INS_ACCEL_FILTER", "AHRS_ORIENTATION"]:
    print(f"{p:18s} {params.get(p, 'not in log')}")

print("\n=== Logging health ===")
print(f"Dropped log messages (DSF): {max(dropped) if dropped else 'no DSF msgs'}")
print(f"Accel clipping (VIBE):      {max(clips) if clips else 'no VIBE msgs'}")


def timing(name, rows):
    d = np.array(rows, dtype=float)
    t = d[:, 0] * 1e-6
    dt = np.diff(t)
    med = np.median(dt)
    gaps = np.where(dt > 1.5 * med)[0]
    print(f"\n=== {name} (instance {INST}) ===")
    print(f"Samples:       {len(d)}")
    print(f"Duration:      {t[-1] - t[0]:.1f} s")
    print(f"Median rate:   {1 / med:.1f} Hz")
    print(f"Gaps > 1.5x:   {len(gaps)}"
          + (f"  (largest {dt.max() * 1e3:.1f} ms)" if len(gaps) else ""))
    print(f"Backwards:     {int(np.sum(dt <= 0))}")
    return d


streams = {}
for name in ["IMU", "GYR", "ACC"]:
    if data[name]:
        streams[name] = timing(name, data[name])
    else:
        print(f"\n=== {name}: none found for instance {INST} ===")

# ---- Motion check, using the gyro from IMU (or GYR if IMU missing) ----
src = streams.get("GYR", streams.get("IMU"))
if src is None:
    sys.exit("\nNo gyro data found - log is not usable.")

t = (src[:, 0] - src[0, 0]) * 1e-6
w = src[:, 1:4]
wn = np.linalg.norm(w, axis=1)

print("\n=== Motion (what Kalibr needs to see) ===")
print(f"Peak |gyro| per axis: {np.abs(w).max(axis=0).round(2)} rad/s")
print(f"Time rotating > 0.3 rad/s: {100 * np.mean(wn > 0.3):.0f} % of the whole log")

# ---- Jerk groups: sharp spikes, grouped when they happen close together ----
JERK = max(1.0, 0.4 * wn.max())   # rad/s, what counts as a jerk
EVENT_GAP = 0.4                   # s, spikes closer than this are one jerk
GROUP_GAP = 6.0                   # s, jerks closer than this are one group

above = np.where(wn > JERK)[0]
events = []
if len(above):
    start = above[0]
    for a, b in zip(above[:-1], above[1:]):
        if t[b] - t[a] > EVENT_GAP:
            events.append((t[start], t[a], wn[start:a + 1].max()))
            start = b
    events.append((t[start], t[above[-1]], wn[start:above[-1] + 1].max()))

groups = []
for e in events:
    if groups and e[0] - groups[-1][-1][1] < GROUP_GAP:
        groups[-1].append(e)
    else:
        groups.append([e])

print(f"\n=== Jerk groups (spikes above {JERK:.2f} rad/s) ===")
if not groups:
    print("None found - no jerks, or they were too gentle.")
for g in groups:
    peaks = ", ".join(f"{e[2]:.1f}" for e in g)
    print(f"{len(g)} jerk(s) at {g[0][0]:.1f}-{g[-1][1]:.1f} s   peaks: {peaks} rad/s")

threes = [g for g in groups if len(g) >= 3]
if len(threes) >= 2:
    a, b = threes[0], threes[-1]
    print(f"\nLooks like your start and end markers: {a[0][0]:.1f} s and {b[0][0]:.1f} s")
    print(f"The video should cover roughly {a[0][0] - 6:.0f} s -> {b[-1][1] + 6:.0f} s of the log "
          f"({b[-1][1] - a[0][0] + 12:.0f} s long).")
    seg = (t >= a[-1][1]) & (t <= b[0][0])
    if seg.any():
        print("\n=== Calibration motion, between the two markers ===")
        print(f"Duration: {t[seg][-1] - t[seg][0]:.0f} s")
        print(f"Peak |gyro| per axis: {np.abs(w[seg]).max(axis=0).round(2)} rad/s "
              "(want > 1.0 on all three)")
        print(f"Time rotating > 0.3 rad/s: {100 * np.mean(wn[seg] > 0.3):.0f} % "
              "(want most of it)")
else:
    print("\nDid not find two groups of 3 jerks. Check the plot - the jerks may have been "
          "too gentle, or too close together to separate.")

# ---- Save CSVs ----
base = path.rsplit(".", 1)[0]
hdr = {"IMU": "TimeUS,GyrX,GyrY,GyrZ,AccX,AccY,AccZ",
       "GYR": "SampleUS,GyrX,GyrY,GyrZ",
       "ACC": "SampleUS,AccX,AccY,AccZ"}
for name, d in streams.items():
    out = f"{base}_{name}{INST}.csv"
    np.savetxt(out, d, delimiter=",", fmt="%.9g", header=hdr[name], comments="")
    print(f"Saved {out}")

# ---- Optional plot to eyeball jerks and motion ----
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
    ax[0].plot(t, w)
    ax[0].set_ylabel("gyro (rad/s)")
    ax[0].legend(["x", "y", "z"])
    ax[1].plot(t, src[:, 4:7]) if src.shape[1] >= 7 else None
    ax[1].set_ylabel("accel (m/s^2)")
    ax[1].set_xlabel("time since log start (s)")
    fig.tight_layout()
    fig.savefig(f"{base}_imu_plot.png", dpi=120)
    print(f"Saved {base}_imu_plot.png")
except ImportError:
    print("(matplotlib not installed - skipping plot)")