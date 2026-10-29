#!/usr/bin/env bash
# run_vins_testA.sh - VINS-Fusion side of Test A (+ C1.2): every bag x every config, one after another.
# Run INSIDE the VINS container (docker exec -it vins bash), from anywhere:
#
#   /data/run_vins_testA.sh
#   BAGS="rail1:/data/accuracyRun1/rail1_half.bag" CONFIGS="B:half_B_gravfix.yaml" /data/run_vins_testA.sh   # one run
#   BAGS="orbit1:/data/orbit1/orbit1_half.bag:12.5" ...      # optional 3rd field: start the bag 12.5 s in (C5.1)
#
# For each run it: starts vins_node (output saved to <name>_log.txt), plays the bag at RATE,
# stops the node, and moves vio.csv + vins_metrics.csv to $RES/<bag>_<cfg>.csv and _metrics.csv.
# Finished runs are skipped, so it can be restarted.
set -o pipefail      # no -u: ROS setup scripts use unset variables

CFG_DIR=${CFG_DIR:-/data/config_rov}
OUT=${OUT:-/data/output}                     # the output_path in your VINS configs
RES=${RES:-/data/testA_vins}
RATE=${RATE:-0.5}
CONFIGS=${CONFIGS:-"A:half_A_kalibr.yaml B:half_B_gravfix.yaml"}
BAGS=${BAGS:-"rail1:/data/accuracyRun1/rail1_half.bag orbit1:/data/orbit1/orbit1_half.bag fig8_1:/data/fig8_1/fig8_1_half.bag"}

source /opt/ros/kinetic/setup.bash
source /root/catkin_ws/devel/setup.bash
mkdir -p "$RES" "$OUT"

# checks before starting anything
for c in $CONFIGS; do [[ -f $CFG_DIR/${c#*:} ]] || { echo "Missing config: $CFG_DIR/${c#*:}"; exit 1; }; done
bagfile() { local r=${1#*:}; echo "${r%%:*}"; }          # name:path[:start_s]
bagstart() { local r=${1#*:}; [[ $r == *:* ]] && echo "${r##*:}" || echo 0; }
for b in $BAGS; do [[ -f $(bagfile "$b") ]] || { echo "Missing bag: $(bagfile "$b")"; exit 1; }; done

if ! rostopic list >/dev/null 2>&1; then
  echo "starting roscore"; setsid roscore >/dev/null 2>&1 < /dev/null & sleep 5   # setsid: Ctrl+C won't kill it
fi
NODE=""
stop_all() { echo; echo "stopped by user - this run is not saved"; [[ -n $NODE ]] && kill -9 $NODE 2>/dev/null
             rm -f "$OUT/vio.csv" "$OUT/vins_metrics.csv"; exit 130; }
trap stop_all INT TERM

for b in $BAGS; do
  BAG=${b%%:*}; BAGFILE=$(bagfile "$b"); START=$(bagstart "$b")
  for c in $CONFIGS; do
    TAG=${c%%:*}; CFG=$CFG_DIR/${c#*:}; NAME=${BAG}_${TAG}
    if [[ -s $RES/$NAME.csv ]]; then echo "skip $NAME (done)"; continue; fi
    rm -f "$OUT/vio.csv" "$OUT/vins_metrics.csv"
    echo "=== $NAME  config ${c#*:}  bag start ${START}s  started $(date +%H:%M:%S)"
    rosrun vins vins_node "$CFG" > "$RES/${NAME}_log.txt" 2>&1 &
    NODE=$!
    sleep 6                                   # let the node load before the bag starts
    rosbag play -q -r "$RATE" -s "$START" "$BAGFILE"
    PLAY=$?
    sleep 5                                   # let VINS finish the last frames
    kill -INT $NODE 2>/dev/null; sleep 2; kill -9 $NODE 2>/dev/null; wait $NODE 2>/dev/null
    if [[ $PLAY -ne 0 ]]; then
      echo "    bag playback failed (exit $PLAY) - run not saved"; rm -f "$OUT/vio.csv" "$OUT/vins_metrics.csv"; exit 1
    elif [[ -s $OUT/vio.csv ]]; then
      mv "$OUT/vio.csv" "$RES/$NAME.csv"
      [[ -f $OUT/vins_metrics.csv ]] && mv "$OUT/vins_metrics.csv" "$RES/${NAME}_metrics.csv"
      echo "    poses $(wc -l < "$RES/$NAME.csv")   failure restarts $(grep -c 'failure detection' "$RES/${NAME}_log.txt")"
    else
      echo "    NO vio.csv written - check $RES/${NAME}_log.txt"
    fi
  done
done
echo "done $(date +%H:%M:%S)"
