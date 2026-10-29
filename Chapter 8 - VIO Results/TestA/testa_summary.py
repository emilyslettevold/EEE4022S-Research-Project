#!/usr/bin/env python3
"""
testa_summary.py - summarise Test A (ORB-SLAM3 start-up on the pool bags). Standard library only.

Usage (Mac Terminal, NOT in the container):
  python3 testa_summary.py <results/testA folder> <folder holding euroc_<bag>/> [--vins rail1=rail1_base_metrics.csv ...]

For each run it reports, from log.txt, f_orb.txt and times.txt:
  first_pose_s   seconds from the first frame to the first pose ORB-SLAM3 kept (poses lost to a map reset are not saved)
  coverage_%     poses kept / frames in the bag
  start_to_end_m distance between the first and last kept pose (rail: compare with the 2 m travelled; loops: ~0)
  largest_step_mm biggest single frame-to-frame jump
  exit           program exit code (0 ok; 134 = abort at shutdown, normal for this build; 139 = crash)
  path_m         length of the kept trajectory
  maps           "New Map created" count
  resets         "Active map reset" count
  no_motion      "Not enough motion for initializing" count
  no_accel       "not enough acceleration" count
  lost           "Fail to track local map" count
  viba           "end VIBA" count (IMU initialisation steps completed; 2 = fully initialised)
  med_track_ms   median tracking time ORB-SLAM3 prints at the end, if present
Writes testA_summary.csv next to the results and prints a per-bag table.
--vins adds VINS-Fusion's time to initialise (first row with solver_flag = 1 in vins_metrics.csv).
"""
import argparse, csv, math, re, statistics as st
from pathlib import Path

PATTERNS = {"maps": "New Map created", "resets": "Active map reset",
            "no_motion": "Not enough motion", "no_accel": "not enough acceleration",
            "lost": "Fail to track local map", "viba": "end VIBA"}

def to_s(x):
    v = float(x)
    return v / 1e9 if v > 1e12 else v           # ORB-SLAM3 writes ns; accept s too

def read_traj(p):
    rows = []
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 4 and not parts[0].startswith("#"):
            try:
                rows.append((to_s(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                pass
    return rows

def run_stats(run_dir, frames):
    log = (run_dir / "log.txt").read_text(errors="ignore") if (run_dir / "log.txt").exists() else ""
    s = {k: log.count(v) for k, v in PATTERNS.items()}
    m = re.search(r"median tracking time:\s*([0-9.eE+-]+)", log)
    s["med_track_ms"] = round(float(m.group(1)) * 1000, 1) if m else ""
    exit_m = re.findall(r"exit code (\d+)", log)
    traj = read_traj(run_dir / "f_orb.txt")
    t0 = frames[0] if frames else None
    if traj and t0 is not None:
        s["first_pose_s"] = round(traj[0][0] - t0, 2)
        s["coverage_%"] = round(100 * len(traj) / len(frames), 1)
        steps = [math.dist(a[1:], b[1:]) for a, b in zip(traj, traj[1:])]
        s["path_m"] = round(sum(steps), 3)
        s["start_to_end_m"] = round(math.dist(traj[0][1:], traj[-1][1:]), 3)
        s["largest_step_mm"] = round(1000 * max(steps), 0) if steps else 0
    else:
        s.update(first_pose_s="", path_m="", start_to_end_m="", largest_step_mm="", **{"coverage_%": 0.0})
    s["exit"] = exit_m[-1] if exit_m else ""
    s["poses"] = len(traj)
    return s

def vins_init(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ""
    t0 = float(rows[0]["t"])
    for r in rows:
        if r["solver_flag"].strip() == "1":
            return round(float(r["t"]) - t0, 2)
    return "never"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("pool")
    ap.add_argument("--vins", nargs="*", default=[], help="bag=path/to/vins_metrics.csv")
    a = ap.parse_args()
    res, pool = Path(a.results), Path(a.pool)
    out, cols = [], ["tag", "bag", "run", "exit", "poses", "coverage_%", "first_pose_s", "start_to_end_m", "largest_step_mm", "path_m",
                     "maps", "resets", "no_motion", "no_accel", "lost", "viba", "med_track_ms"]
    for run_dir in sorted(res.glob("*/*/run_*")):
        tag, bag = run_dir.parent.parent.name, run_dir.parent.name
        tf = pool / f"euroc_{bag}" / "times.txt"
        frames = [to_s(x) for x in tf.read_text().split()] if tf.exists() else []
        if not frames:
            print(f"warning: no times.txt for {bag} at {tf}; coverage and first-pose time left blank")
        s = run_stats(run_dir, frames)
        out.append({"tag": tag, "bag": bag, "run": run_dir.name, **s})
    if not out:
        raise SystemExit(f"No runs found under {res}/<tag>/<bag>/run_XX")
    with open(res / "testA_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        w.writerows({k: r.get(k, "") for k in cols} for r in out)

    print(f"{'tag':<14}{'bag':<9}{'run':<8}{'exit':>5}{'cov %':>7}{'1st pose s':>11}{'start-end m':>12}{'max step mm':>12}"
          f"{'maps':>6}{'resets':>7}{'no_mot':>7}{'no_acc':>8}{'VIBA':>6}")
    for r in out:
        print(f"{r['tag']:<14}{r['bag']:<9}{r['run']:<8}{str(r['exit']):>5}{r['coverage_%']:>7}{str(r['first_pose_s']):>11}"
              f"{str(r['start_to_end_m']):>12}{str(r['largest_step_mm']):>12}{r['maps']:>6}{r['resets']:>7}{r['no_motion']:>7}{r['no_accel']:>8}{r['viba']:>6}")
    print("\nPer bag (median across runs):")
    groups = {}
    for r in out:
        groups.setdefault((r["tag"], r["bag"]), []).append(r)
    for (tag, bag), rs in groups.items():
        fp = [r["first_pose_s"] for r in rs if r["first_pose_s"] != ""]
        cov = [r["coverage_%"] for r in rs]
        print(f"  {tag:<14}{bag:<9} runs {len(rs)}   first pose {st.median(fp) if fp else 'none':>6} s   "
              f"coverage {st.median(cov):.1f}%  (range {min(cov):.1f}-{max(cov):.1f})   "
              f"runs with a reset: {sum(1 for r in rs if r['resets'] > 0)}/{len(rs)}")
    for spec in a.vins:
        bag, path = spec.split("=", 1)
        print(f"  VINS      {bag:<9} time to initialise {vins_init(path)} s")
    print(f"\nwrote {res / 'testA_summary.csv'}")

if __name__ == "__main__":
    main()
