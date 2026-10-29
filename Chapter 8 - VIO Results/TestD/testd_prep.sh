#!/usr/bin/env bash
# testd_prep.sh - prepare ONE held-out recording for Test D, everything named by NAME so nothing gets overwritten.
#
#   source ~/evo-env/bin/activate
#   ~/vio_data/testd_prep.sh NAME SRC [GT]
#
#   NAME  short name, e.g. line1, fig8_3, orbit2, square2, lawn1   (used for every folder and file)
#   SRC   folder with left.mkv, right.mkv, left_timestamps.csv, right_timestamps.csv, imu0.csv
#   GT    optional: that recording's gt_tum.txt (copied to ~/vio_data/NAME/gt_tum.txt)
#
# Makes:
#   SRC/dataset/                       full-res pairs + imu0.csv   (kept: the original, hard-linked frames)
#   ~/vio_data/NAME/dataset_half/      800x600 copy
#   ~/vio_data/NAME/NAME_half.bag      the bag VINS reads
#   ~/vio_data/NAME/timing.txt         IMU rate, IMU coverage, frame gaps (for the write-up)
#   ~/vio_data/NAME/gt_tum.txt         ground truth (if given) + overlap check
# It STOPS instead of overwriting anything that already exists. No trimming (Test D rule: gaps are reported).
set -euo pipefail
NAME=${1:?usage: testd_prep.sh NAME SRC [GT]}; SRC=${2:?usage: testd_prep.sh NAME SRC [GT]}; GT=${3:-}
VD=$HOME/vio_data; OUT=$VD/$NAME
PAIR=$VD/accuracyRun1/pair_frames_vio.py; RESIZE=$VD/accuracyRun1/resize_dataset.py
SRC=$(cd "$SRC" && pwd)
say() { echo; echo "=== [$NAME] $*"; }

say "checks"
for f in left.mkv right.mkv left_timestamps.csv right_timestamps.csv; do [[ -f $SRC/$f ]] || { echo "missing $SRC/$f"; exit 1; }; done
[[ -f $SRC/imu0.csv || -f $SRC/dataset/imu0.csv ]] || { echo "no imu0.csv in $SRC - make it with extract_imu.py first"; exit 1; }
for f in "$PAIR" "$RESIZE" "$VD/check_overlap.py"; do [[ -f $f ]] || { echo "missing $f"; exit 1; }; done
[[ -e $OUT/dataset_half || -e $OUT/${NAME}_half.bag ]] && { echo "$OUT already has dataset_half or ${NAME}_half.bag - delete them first if you really want to redo $NAME"; exit 1; }
python3 -c "import pandas" 2>/dev/null || { echo "pandas missing - run: source ~/evo-env/bin/activate"; exit 1; }
df -h ~ | tail -1
mkdir -p "$OUT"

if [[ -f $SRC/dataset/pairs.csv ]]; then
  say "SRC/dataset already exists - reusing it (no re-extraction)"
else
  say "1/5 extract frames"
  [[ -e $SRC/frames ]] && { echo "$SRC/frames exists from an earlier attempt - delete it: rm -r '$SRC/frames'"; exit 1; }
  mkdir -p "$SRC/frames/left" "$SRC/frames/right"
  ffmpeg -loglevel error -i "$SRC/left.mkv"  -fps_mode passthrough "$SRC/frames/left/%06d.png"
  ffmpeg -loglevel error -i "$SRC/right.mkv" -fps_mode passthrough "$SRC/frames/right/%06d.png"
  echo "frames: left $(ls "$SRC/frames/left" | wc -l), right $(ls "$SRC/frames/right" | wc -l)"
  say "2/5 pair frames (real arrival_monotonic_ns timestamps)"
  (cd "$SRC" && python3 "$PAIR" --left-csv left_timestamps.csv --right-csv right_timestamps.csv \
      --left-frames frames/left --right-frames frames/right --out dataset --max-offset-ms 20)
fi
[[ -f $SRC/dataset/imu0.csv ]] || cp "$SRC/imu0.csv" "$SRC/dataset/imu0.csv"

say "3/5 timing check (saved to $OUT/timing.txt)"
python3 - "$SRC/dataset" <<'EOF' | tee "$OUT/timing.txt"
import sys, pandas as pd, numpy as np
d = sys.argv[1]
imu = pd.read_csv(f'{d}/imu0.csv'); cam = pd.read_csv(f'{d}/pairs.csv').stamp_ns
t = imu.iloc[:, 0].to_numpy(np.int64); dt = np.diff(cam) / 1e6
print(f'pairs {len(cam)}, duration {(cam.iloc[-1]-cam.iloc[0])/1e9:.1f} s, IMU ~{1e9/np.median(np.diff(t)):.0f} Hz')
print(f'IMU starts {(cam.iloc[0]-t[0])/1e9:+.2f} s before camera, ends {(t[-1]-cam.iloc[-1])/1e9:+.2f} s after')
print('frame gaps > 100 ms:', [f'{g:.0f} ms at {(cam.iloc[i]-cam.iloc[0])/1e9:.1f} s' for i, g in enumerate(dt) if g > 100])
EOF

say "4/5 half-resolution copy"
python3 "$RESIZE" "$SRC/dataset" "$OUT/dataset_half"
ls "$OUT/dataset_half"
[[ -d $SRC/frames ]] && rm -r "$SRC/frames" && echo "deleted $SRC/frames (full-res images stay in $SRC/dataset)"

say "5/5 build $OUT/${NAME}_half.bag (Kalibr container)"
docker run --rm -v "$VD":/data --entrypoint bash kalibr:latest -c \
  "source /catkin_ws/devel/setup.bash && /catkin_ws/devel/lib/kalibr/kalibr_bagcreater --folder /data/$NAME/dataset_half --output-bag /data/$NAME/${NAME}_half.bag" \
  || { echo "bag build failed - build it by hand:"; echo "  docker run -it --rm -v ~/vio_data:/data kalibr:latest bash"
       echo "  source /catkin_ws/devel/setup.bash && /catkin_ws/devel/lib/kalibr/kalibr_bagcreater --folder /data/$NAME/dataset_half --output-bag /data/$NAME/${NAME}_half.bag"; exit 1; }
ls -lh "$OUT/${NAME}_half.bag"

if [[ -n $GT ]]; then
  say "ground truth"
  cp "$GT" "$OUT/gt_tum.txt"
  python3 "$VD/check_overlap.py" "$OUT/dataset_half/pairs.csv" "$OUT/gt_tum.txt"
fi
say "done. Next:  ~/vio_data/testd_run.sh $NAME"
