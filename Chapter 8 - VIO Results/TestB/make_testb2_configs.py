#!/usr/bin/env python3
"""
make_testb2_configs.py - configs for the RESTRUCTURED Test B: every step uses your rig settings (A0: tuned IMU
noise, td estimated online, g 9.79, min_dist, multiple_thread 0). Only resolution changes between steps.

Usage (Lenovo, in ~/vio_data/config_rov - the folder that already holds testb_final.yaml and testb_stock_full.yaml):
  python3 make_testb2_configs.py

Reads:
  testb_final.yaml       = A0 (half resolution, tuned)            -> settings source
  testb_stock_full.yaml  -> the full-resolution image size and camera calibration files only
Writes:
  testb2_full.yaml  / testb2_full_noimu.yaml    A0 settings at 1600x1200   (B0, B1)
  testb2_half.yaml  / testb2_half_noimu.yaml    A0 settings at 800x600     (B2, B3, B4) - identical to A0
At full resolution min_dist is doubled (15 -> 30 px) so features keep the same spacing in the scene;
otherwise B2 would change feature spacing as well as resolution.
"""
import re, sys
from pathlib import Path

SCALAR = r"^({k}:\s*)([^\s#!][^#\n]*?)(\s*(#.*)?)$"

def get(text, k):
    m = re.search(SCALAR.format(k=k), text, re.M)
    if not m:
        sys.exit(f"'{k}' not found")
    return m.group(2)

def setk(text, k, v):
    new, n = re.subn(SCALAR.format(k=k), lambda m: f"{m.group(1)}{v}{m.group(3)}", text, flags=re.M)
    if n != 1:
        sys.exit(f"'{k}' found {n} times")
    return new

def tag(text, note):
    first, rest = text.split("\n", 1) if text.startswith("%YAML") else ("", text)
    return (first + "\n" if first else "") + f"# Test B (restructured): {note}\n" + rest

for f in ("testb_final.yaml", "testb_stock_full.yaml"):
    if not Path(f).exists():
        sys.exit(f"{f} not found - run this in ~/vio_data/config_rov")
a0 = Path("testb_final.yaml").read_text()
full_src = Path("testb_stock_full.yaml").read_text()

if get(a0, "multiple_thread").strip() != "0":
    sys.exit("testb_final.yaml does not have multiple_thread 0 - is it really A0?")

full = a0
for k in ("image_width", "image_height", "cam0_calib", "cam1_calib"):
    full = setk(full, k, get(full_src, k))
md = float(get(a0, "min_dist"))
full = setk(full, "min_dist", int(md * 2) if md.is_integer() else md * 2)

out = {"testb2_full": (full, "A0 settings, full resolution 1600x1200 (B0, B1)"),
       "testb2_half": (a0, "A0 settings, half resolution 800x600 (B2-B4) = A0")}
for name, (txt, note) in out.items():
    Path(f"{name}.yaml").write_text(tag(txt, note))
    Path(f"{name}_noimu.yaml").write_text(tag(setk(txt, "imu", 0), note + ", IMU off (stereo only)"))
    print(f"wrote {name}.yaml and {name}_noimu.yaml")

print("\nfull vs half differ only in:")
for k in ("image_width", "image_height", "cam0_calib", "cam1_calib", "min_dist"):
    print(f"  {k:<14} {get(full, k):>32}   {get(a0, k)}")
print("\nshared rig settings (put these in your B0 description):")
for k in ("acc_n", "gyr_n", "acc_w", "gyr_w", "g_norm", "estimate_td", "td", "max_cnt", "F_threshold",
          "multiple_thread", "estimate_extrinsic"):
    try:
        print(f"  {k:<18} {get(a0, k)}")
    except SystemExit:
        pass
