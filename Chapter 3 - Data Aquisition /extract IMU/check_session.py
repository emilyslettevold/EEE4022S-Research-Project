"""
check_session.py - verify extracted IMU against each recording.

Each session folder needs: left_timestamps.csv, right_timestamps.csv, imu0.csv
(imu0.csv produced by extract_imu.py).

Usage:
    python check_session.py multi_test/*/

Per session it reports:
  - whether the IMU starts before the first frame and ends after the last
  - IMU rate during the recording
  - which gyro axis moved most (roll -> GyrX, pitch -> GyrY, yaw -> GyrZ)
  - when the biggest jolt happened, in seconds after the first frame
    (compare with the video: the tap should be visible at that time)
Then checks that the sessions do not overlap each other.
"""
import csv
import statistics
import sys
from pathlib import Path


def read_frames(path):
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        return [int(row[2]) for row in r if row]  # arrival_monotonic_ns


def read_imu(path):
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        return [(int(row[0]), *map(float, row[1:7])) for row in r if row]


results = []
for d in map(Path, sys.argv[1:]):
    if not (d / "imu0.csv").exists():
        print(f"\n{d.name}: no imu0.csv - run extract_imu.py first")
        continue
    frames = read_frames(d / "left_timestamps.csv") + \
        read_frames(d / "right_timestamps.csv")
    imu = read_imu(d / "imu0.csv")
    f0, f1 = min(frames), max(frames)
    i0, i1 = imu[0][0], imu[-1][0]
    inside = [s for s in imu if f0 <= s[0] <= f1]
    dur = (f1 - f0) / 1e9

    print(f"\n{d.name}")
    print(f"  video length          {dur:.1f} s")
    before, after = (f0 - i0) / 1e9, (i1 - f1) / 1e9
    ok = before > 0 and after > 0
    print(f"  IMU starts {before:+.2f} s before first frame, "
          f"ends {after:+.2f} s after last  -> "
          f"{'COVERS VIDEO' if ok else 'DOES NOT COVER VIDEO'}")
    if len(inside) < 2:
        print("  too few IMU samples inside the video window")
        continue
    print(f"  IMU rate in video     {len(inside) / dur:.1f} Hz")

    names = ["GyrX (roll)", "GyrY (pitch)", "GyrZ (yaw)"]
    spread = [statistics.pstdev(s[1 + k] for s in inside) for k in range(3)]
    for n, v in zip(names, spread):
        print(f"    {n:<14} std {v:.3f} rad/s")
    print(f"  most motion on        {names[spread.index(max(spread))]}")

    mags = [(s[4] ** 2 + s[5] ** 2 + s[6] ** 2) ** 0.5 for s in inside]
    med = statistics.median(mags)
    k = max(range(len(mags)), key=lambda j: abs(mags[j] - med))
    print(f"  biggest jolt at       {(inside[k][0] - f0) / 1e9:.2f} s "
          f"after first frame ({abs(mags[k] - med):.1f} m/s^2)")
    results.append((d.name, f0, f1, i0, i1))

results.sort(key=lambda r: r[1])
print("\nOverlap check (in recording order):")
clean = True
for a, b in zip(results, results[1:]):
    gap = (b[1] - a[2]) / 1e9
    print(f"  {a[0]} -> {b[0]}: gap between videos {gap:.1f} s")
    if gap <= 0:
        clean = False
        print("    WARNING: videos overlap - timestamps are inconsistent")
print("  no overlaps" if clean else "  PROBLEM: see warnings above")
