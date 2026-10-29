#!/usr/bin/env bash
# testb2_run.sh - run the whole restructured Test B ladder. Run on the LENOVO host (not in the container),
# from the folder holding vins_code.sh (~/vio_data):
#
#   ./testb2_run.sh            # all steps
#   ./testb2_run.sh B3 B4      # only these steps
#
# For each step: swaps the VINS code (vins_code.sh), rebuilds, then runs inside the 'vins' container:
#   orbit1, stereo-inertial      -> <step>     e.g. orbit1_B0.csv
#   rail1,  stereo only          -> <step>n    e.g. rail1_B0n.csv
#   orbit1, stereo only (B0, B4) -> <step>n    shows how much the IMU was carrying the stock pipeline
# Results go to /data/testB2 in the container (= ~/vio_data/testB2). Finished runs are skipped, so you can restart.
# It ends on B4 code (= your full patched code), so the container is left ready for Test C.
set -euo pipefail
CT=${CT:-vins}
RES=${RES:-/data/testB2}
RAIL_FULL=/data/accuracyRun1/rail1.bag;  RAIL_HALF=/data/accuracyRun1/rail1_half.bag
ORB_FULL=/data/orbit1/orbit1.bag;        ORB_HALF=/data/orbit1/orbit1_half.bag

#        step code res   orbit-stereo-only?
LADDER="B0:B0:full:yes B1:B1:full:no B2:B1:half:no B3:B3:half:no B4:B4:half:yes"
STEPS=${*:-B0 B1 B2 B3 B4}

run() {   # run <bags> <configs>
  docker exec -it "$CT" bash -c "RES=$RES BAGS='$1' CONFIGS='$2' /data/run_vins_testA.sh"
}

for s in $STEPS; do
  line=$(tr ' ' '\n' <<<"$LADDER" | grep "^$s:") || { echo "unknown step $s"; exit 1; }
  IFS=: read -r STEP CODE RESN ORBN <<<"$line"
  if [[ $RESN == full ]]; then RAIL=$RAIL_FULL; ORB=$ORB_FULL; else RAIL=$RAIL_HALF; ORB=$ORB_HALF; fi
  echo; echo "################ $STEP  (code $CODE, $RESN resolution)  $(date +%H:%M)"
  have=$(docker exec "$CT" cat /root/CODE_VERSION 2>/dev/null || echo none)
  if [[ $have != "$CODE" && ! ( $CODE == B4 && $have == patched ) ]]; then ./vins_code.sh "$CODE"; else echo "code already $have"; fi
  run "orbit1:$ORB" "$STEP:testb2_${RESN}.yaml"
  run "rail1:$RAIL" "${STEP}n:testb2_${RESN}_noimu.yaml"
  [[ $ORBN == yes ]] && run "orbit1:$ORB" "${STEP}n:testb2_${RESN}_noimu.yaml"
done
echo; echo "all done $(date +%H:%M) - container code: $(docker exec "$CT" cat /root/CODE_VERSION)"
echo "next:  cd ~/vio_data/testB2 && python3 ../testb_table.py"
