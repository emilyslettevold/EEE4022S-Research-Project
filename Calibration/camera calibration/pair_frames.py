#!/usr/bin/env python3
"""
pair_frames.py

Pairs left/right camera frames by nearest arrival_monotonic_ns timestamp
and writes them into a Kalibr-ready dataset/cam0, dataset/cam1 folder
structure (filenames = synthetic increasing nanosecond timestamps).

Usage:
    python3 pair_frames.py \
        --left-csv left_timestamps.csv \
        --right-csv right_timestamps.csv \
        --left-frames frames_left \
        --right-frames frames_right \
        --out dataset \
        --max-offset-ms 20 \
        --frames 134 160 188 195 239 254 311 354 365 399 453 482 520 560 603 660 699 723
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-csv", required=True, help="left_timestamps CSV path")
    ap.add_argument("--right-csv", required=True, help="right_timestamps CSV path")
    ap.add_argument("--left-frames", required=True, help="dir of ffmpeg-extracted left PNGs")
    ap.add_argument("--right-frames", required=True, help="dir of ffmpeg-extracted right PNGs")
    ap.add_argument("--out", default="dataset", help="output dataset dir (will contain cam0/, cam1/)")
    ap.add_argument("--max-offset-ms", type=float, default=50.0,
                     help="reject a pairing if nearest match is farther than this (ms)")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                     help="optional list of left-frame indices to keep (0-indexed, matches "
                          "the 'frame' column). If omitted, every frame in left CSV is considered.")
    args = ap.parse_args()

    max_offset_ns = args.max_offset_ms * 1_000_000

    left = pd.read_csv(args.left_csv).sort_values("arrival_monotonic_ns").reset_index(drop=True)
    right = pd.read_csv(args.right_csv).sort_values("arrival_monotonic_ns").reset_index(drop=True)

    if args.frames is not None:
        left = left[left["frame"].isin(args.frames)].reset_index(drop=True)
        if left.empty:
            raise SystemExit("No rows matched --frames; check the indices you passed.")

    left_frames_dir = Path(args.left_frames)
    right_frames_dir = Path(args.right_frames)

    # ffmpeg's %06d output is 1-indexed; 'frame' column is 0-indexed
    left["img"] = left["frame"].apply(lambda f: left_frames_dir / f"{f + 1:06d}.png")
    right["img"] = right["frame"].apply(lambda f: right_frames_dir / f"{f + 1:06d}.png")

    out_cam0 = Path(args.out) / "cam0"
    out_cam1 = Path(args.out) / "cam1"
    out_cam0.mkdir(parents=True, exist_ok=True)
    out_cam1.mkdir(parents=True, exist_ok=True)

    ri = 0
    kept = 0
    skipped_offset = 0
    skipped_missing = 0

    for _, lrow in left.iterrows():
        while ri < len(right) - 1 and abs(
            right.loc[ri + 1, "arrival_monotonic_ns"] - lrow["arrival_monotonic_ns"]
        ) < abs(right.loc[ri, "arrival_monotonic_ns"] - lrow["arrival_monotonic_ns"]):
            ri += 1

        rrow = right.loc[ri]
        offset = abs(rrow["arrival_monotonic_ns"] - lrow["arrival_monotonic_ns"])
        if offset > max_offset_ns:
            skipped_offset += 1
            continue

        if not lrow["img"].exists() or not rrow["img"].exists():
            skipped_missing += 1
            continue

        ts = (kept + 1) * 1_000_000_000
        shutil.copy(lrow["img"], out_cam0 / f"{ts}.png")
        shutil.copy(rrow["img"], out_cam1 / f"{ts}.png")
        kept += 1

    print(f"Wrote {kept} pairs to {args.out}/cam0 and {args.out}/cam1")
    if skipped_offset:
        print(f"Skipped {skipped_offset} candidate(s): nearest match exceeded "
              f"{args.max_offset_ms} ms")
    if skipped_missing:
        print(f"Skipped {skipped_missing} candidate(s): image file(s) not found on disk")


if __name__ == "__main__":
    main()
    