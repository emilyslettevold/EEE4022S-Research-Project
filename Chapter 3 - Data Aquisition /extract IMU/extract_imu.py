"""
extract_imu.py - cut the IMU data for one recording out of an ArduSub BIN
and put it on the Pi clock (same clock as arrival_monotonic_ns).

Needs imu_logger.py in the same folder (for fc_to_pi_offset).

Usage (on the laptop, pymavlink venv active):
    python extract_imu.py 00000042.BIN SESSION_timesync.csv --out imu0.csv

Output columns (Kalibr / EuRoC imu0.csv format):
    timestamp [ns, Pi clock], omega_x/y/z [rad/s], alpha_x/y/z [m/s^2]
"""
import argparse
import csv

from pymavlink import mavutil

from imu_logger import fc_to_pi_offset

p = argparse.ArgumentParser()
p.add_argument("bin")
p.add_argument("timesync")
p.add_argument("--out", default="imu0.csv")
p.add_argument("--instance", type=int, default=0, help="IMU instance (0 = IMU1)")
p.add_argument("--pad", type=float, default=1.0, help="seconds kept before/after recording")
args = p.parse_args()

offset_ns, fc_start_ns, fc_end_ns = fc_to_pi_offset(args.timesync)
t0_us = fc_start_ns / 1000 - args.pad * 1e6
t1_us = fc_end_ns / 1000 + args.pad * 1e6

log = mavutil.mavlink_connection(args.bin)
rows = []
while True:
    m = log.recv_match(type="IMU")
    if m is None:
        break
    if getattr(m, "I", 0) != args.instance:
        continue
    if m.TimeUS < t0_us:
        continue
    if m.TimeUS > t1_us:
        break
    pi_ns = m.TimeUS * 1000 - offset_ns  # pi = fc - offset
    rows.append([pi_ns, m.GyrX, m.GyrY, m.GyrZ, m.AccX, m.AccY, m.AccZ])

if not rows:
    raise SystemExit("No IMU rows in window - is this the BIN from the same boot as the recording?")

with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["#timestamp", "omega_x", "omega_y", "omega_z", "alpha_x", "alpha_y", "alpha_z"])
    w.writerows(rows)

dts = sorted(rows[i + 1][0] - rows[i][0] for i in range(len(rows) - 1))
median_dt = dts[len(dts) // 2]
gaps = sum(1 for d in dts if d > 2 * median_dt)
dur = (rows[-1][0] - rows[0][0]) / 1e9
print(f"{len(rows)} samples over {dur:.2f} s -> {len(rows) / dur:.1f} Hz "
      f"(median dt {median_dt / 1e6:.2f} ms, {gaps} gaps > 2x median)")
print(f"offset used: {offset_ns} ns; wrote {args.out}")
