#!/usr/bin/env python3
"""
testb_table.py - one table of the important numbers from every Test B run. Standard library only.

Usage (Lenovo, in the results folder):
  cd ~/vio_data/testB2 && python3 ../testb_table.py
  python3 ../testb_table.py --times-dir ~/vio_data/pool_testA      # where euroc_<bag>/times.txt live (default)

Reads every <bag>_<tag>.csv (+ _metrics.csv and _log.txt if present). Tags ending in 'n' are stereo-only runs.
Writes testb_table.csv and testb_table.md (a Markdown table you can paste into the report) and prints the table.

Columns
  coverage_%      poses output / frames recorded
  first_pose_s    seconds from the first recorded frame to the first pose
  start_end_m     distance first -> last pose (orbit: closure error; rail: travelled distance)
  rail_err_%      rail only: (start_end - 2.0 m) / 2.0 m
  path_m          total length of the estimated trajectory
  max_step_mm     largest frame-to-frame jump;  jumps = steps over 100 mm
  max_speed       largest estimated speed, m/s (a sign of divergence if large)
  restarts        "failure detection" count in the log
  stereo_ratio    median stereo-matched / tracked features      } only with the metrics patch
  guess_frac      median fraction of window features with a guessed depth  } (B4 code)
  solve_ms        median optimiser time per frame               }
"""
import argparse, csv, glob, math, os, statistics as st
from pathlib import Path

TRUE_RAIL = 2.0

def sec(x):
    v = float(x); return v / 1e9 if v > 1e12 else v

def traj(path):
    t, P, V = [], [], []
    for r in csv.reader(open(path)):
        if len(r) < 4 or not r[0].strip():
            continue
        try:
            t.append(sec(r[0])); P.append([float(v) for v in r[1:4]])
            V.append([float(v) for v in r[8:11]] if len(r) >= 11 else [0, 0, 0])
        except ValueError:
            pass
    return t, P, V

def stage(path):
    if not os.path.exists(path):
        return {}
    rows = [r for r in csv.DictReader(open(path)) if r.get("solver_flag", "").strip() == "1"]
    if not rows:
        return {}
    def f(r, k):
        try: return float(r[k])
        except (KeyError, ValueError, TypeError): return None
    def med(fn):
        v = [x for x in map(fn, rows) if x is not None]
        return round(st.median(v), 2) if v else ""
    return {"stereo_ratio": med(lambda r: (f(r, "n_stereo") or 0) / max(f(r, "n_feat") or 0, 1)),
            "guess_frac": med(lambda r: (f(r, "n_depth_guess") or 0) / max(f(r, "n_win") or 0, 1)),
            "solve_ms": med(lambda r: f(r, "solve_ms"))}

ap = argparse.ArgumentParser()
ap.add_argument("--times-dir", default=os.path.expanduser("~/vio_data/pool_testA"))
a = ap.parse_args()

rows = []
for f in sorted(glob.glob("*.csv")):
    if f.endswith("_metrics.csv") or f.startswith(("testb_table", "vins_check", "testc_summary")):
        continue
    name = f[:-4]
    if "_" not in name:
        continue
    bag, tag = name.rsplit("_", 1)
    t, P, V = traj(f)
    if not t:
        continue
    tf = Path(a.times_dir) / f"euroc_{bag}" / "times.txt"
    frames = [sec(x) for x in tf.read_text().split()] if tf.exists() else []
    steps = [math.dist(p, q) for p, q in zip(P, P[1:])]
    se = math.dist(P[0], P[-1])
    log = Path(f"{name}_log.txt")
    r = {"step": tag.rstrip("n"), "bag": bag, "mode": "stereo only" if tag.endswith("n") else "stereo-inertial",
         "coverage_%": round(100 * len(t) / len(frames), 1) if frames else "",
         "first_pose_s": round(t[0] - frames[0], 2) if frames else "",
         "start_end_m": round(se, 3),
         "rail_err_%": round(100 * (se - TRUE_RAIL) / TRUE_RAIL, 1) if bag.startswith("rail") else "",
         "path_m": round(sum(steps), 2),
         "max_step_mm": round(1000 * max(steps)) if steps else 0,
         "jumps": sum(s > 0.10 for s in steps),
         "max_speed": round(max(math.hypot(*v) for v in V), 2),
         "restarts": log.read_text(errors="ignore").count("failure detection") if log.exists() else "",
         **stage(f"{name}_metrics.csv")}
    rows.append(r)
if not rows:
    raise SystemExit("no <bag>_<tag>.csv files in this folder")
missing = sorted({r["bag"] for r in rows if r["coverage_%"] == ""})
if missing:
    print(f"note: no times.txt for {missing} under {a.times_dir} - coverage and first-pose left blank")

order = {"orbit1": 0, "rail1": 1}
rows.sort(key=lambda r: (r["step"], order.get(r["bag"], 9), r["mode"]))
cols = ["step", "bag", "mode", "coverage_%", "first_pose_s", "start_end_m", "rail_err_%", "path_m",
        "max_step_mm", "jumps", "max_speed", "restarts", "stereo_ratio", "guess_frac", "solve_ms"]
with open("testb_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    w.writerows({k: r.get(k, "") for k in cols} for r in rows)
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(str(r.get(k, "")) if r.get(k, "") != "" else "—" for k in cols) + " |" for r in rows]
Path("testb_table.md").write_text("\n".join(md) + "\n")

widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in cols}
print("  ".join(k.rjust(widths[k]) for k in cols))
for r in rows:
    print("  ".join(str(r.get(k, "")).rjust(widths[k]) for k in cols))
print("\nwrote testb_table.csv and testb_table.md")
