#!/usr/bin/env python3
"""
make_testc_configs.py - write every Test C config as a copy of the frozen baseline with ONE setting changed.

Usage (Lenovo, in ~/vio_data/config_rov):
  python3 make_testc_configs.py testb2_half.yaml        (baseline = Test B step B4)
  python3 make_testc_configs.py half_A0_kalibr.yaml --fullres-cam0 cam0_full.yaml --fullres-cam1 cam1_full.yaml

Writes  testc_<TAG>.yaml  for each manipulation and prints exactly what changed, so the diff is on record.
Stops with an error if a key it needs is missing from the baseline (nothing is guessed).
"""
import argparse, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("--fullres-cam0", help="1600x1200 cam0 intrinsics yaml (enables C2.1)")
ap.add_argument("--fullres-cam1", help="1600x1200 cam1 intrinsics yaml (enables C2.1)")
a = ap.parse_args()
base = Path(a.base).read_text()

def get(key):
    m = re.search(rf"^{key}:\s*([^\s#]+)", base, re.M)
    if not m:
        sys.exit(f"'{key}' not found in {a.base} - add it or tell Claude")
    return m.group(1).strip('"')

def setk(text, key, value):
    new, n = re.subn(rf"^({key}:\s*)([^\s#]+)", lambda m: f"{m.group(1)}{value}", text, flags=re.M)
    if n != 1:
        sys.exit(f"'{key}' found {n} times in {a.base}")
    return new

noise = {k: float(get(k)) for k in ("acc_n", "gyr_n", "acc_w", "gyr_w")}
td = float(get("td"))
solver_t, solver_it = float(get("max_solver_time")), int(float(get("max_num_iterations")))
max_cnt, kf_par = int(float(get("max_cnt"))), float(get("keyframe_parallax"))
min_dist = float(get("min_dist"))

variants = {   # TAG: (stage, what, {key: value})
    "base":      ("all",   "frozen baseline A0", {}),
    "td0":       ("1",     "C1.1 time offset fixed at 0 ms", {"estimate_td": 0, "td": 0.0}),
    "tdfix":     ("1",     f"C1.1 time offset fixed at Kalibr {td*1000:.1f} ms", {"estimate_td": 0}),
    "cnt75":     ("2",     "C2.3 feature budget 75", {"max_cnt": 75}),
    "cnt300":    ("2",     "C2.3 feature budget 300", {"max_cnt": 300}),
    "par5":      ("3",     "C3.2 keyframe parallax 5 px", {"keyframe_parallax": kf_par / 2}),
    "par20":     ("3",     "C3.2 keyframe parallax 20 px", {"keyframe_parallax": kf_par * 2}),
    "noimu":     ("4",     "C4.1 IMU off (stereo only)", {"imu": 0}),
    "noise01":   ("4",     "C4.2 IMU noise /10 (raw Allan values)", {k: f"{v/10:.6e}" for k, v in noise.items()}),
    "noise5":    ("4",     "C4.2 IMU noise x5", {k: f"{v*5:.6e}" for k, v in noise.items()}),
    "solvhalf":  ("6",     "C6.1 solver budget halved", {"max_solver_time": solver_t / 2, "max_num_iterations": max(1, solver_it // 2)}),
    "solvdbl":   ("6",     "C6.1 solver budget doubled", {"max_solver_time": solver_t * 2, "max_num_iterations": solver_it * 2}),
    "md30":      ("2",     f"C2.4 feature spacing min_dist {min_dist:g} -> 30 px", {"min_dist": 30}),
    "md30n5":    ("2+4",   "C2.4 follow-up: min_dist 30 AND IMU noise x5 (interaction check, not one-at-a-time)",
                  {"min_dist": 30, **{k: f"{v*5:.6e}" for k, v in noise.items()}}),
}
if a.fullres_cam0 and a.fullres_cam1:
    variants["fullres"] = ("2", "C2.1 full resolution 1600x1200",
                           {"image_width": 1600, "image_height": 1200, "min_dist": int(min_dist * 2),   # same spacing in the scene
                            "cam0_calib": f'"{a.fullres_cam0}"', "cam1_calib": f'"{a.fullres_cam1}"'})

for tag, (stage, what, changes) in variants.items():
    t = base
    for k, v in changes.items():
        t = setk(t, k, v)
    note = f"# Test C {tag}: stage {stage}: {what}  (generated from {a.base})\n"
    if t.startswith("%YAML"):          # the %YAML directive must stay the first line for OpenCV
        first, rest = t.split("\n", 1); t = first + "\n" + note + rest
    else:
        t = note + t
    Path(f"testc_{tag}.yaml").write_text(t)
    print(f"testc_{tag}.yaml   stage {stage:<4} {what:<45} " + ", ".join(f"{k}={v}" for k, v in changes.items()))
if "fullres" not in variants:
    print("\n(C2.1 full-resolution config not written: pass --fullres-cam0/--fullres-cam1 with your 1600x1200 camera yamls)")
