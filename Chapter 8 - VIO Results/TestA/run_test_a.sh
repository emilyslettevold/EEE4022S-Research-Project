#!/usr/bin/env bash
# run_test_a.sh - Test A: ORB-SLAM3 start-up on the pool bags. Run INSIDE the orbslam3_test container.
#
#   POOL=/results/pool ./run_test_a.sh                                  # stereo-inertial, default init, 3 bags x 3 runs
#   POOL=/results/pool MODE=stereo BAGS="rail1" RUNS=1 ./run_test_a.sh  # stereo-only control on the rail
#   POOL=/results/pool TAG=fastinit CFG_NAME=orb_si_fastinit.yaml BAGS="rail1" ./run_test_a.sh   # only after the fastInit patch
#
# Expects  $POOL/euroc_<bag>/{mav0/..., times.txt}  and  $POOL/configs/<CFG_NAME>
# Writes   $POOL/results/testA/<TAG>/<bag>/run_XX/{f_orb.txt, kf_orb.txt, log.txt, time.txt}
# Finished runs (marked DONE) are skipped, so it can be restarted after a crash.
# A run that never initialises saves no poses; that is a result, not an error.
set -uo pipefail

ORB_ROOT=${ORB_ROOT:-/workspace/ORB_SLAM3}
POOL=${POOL:?set POOL to the in-container folder holding euroc_<bag>/ and configs/}
RUNS=${RUNS:-3}
TIMEOUT=${TIMEOUT:-1200}             # seconds; a run that hangs (e.g. at shutdown with an empty map) is killed
MODE=${MODE:-si}                      # si = stereo-inertial, stereo = stereo only
BAGS=${BAGS:-"rail1 orbit1 fig8_1"}

if [[ $MODE == si ]]; then
  EXE=$ORB_ROOT/Examples/Stereo-Inertial/stereo_inertial_euroc
  CFG_NAME=${CFG_NAME:-orb_si_gravfix.yaml}; TAG=${TAG:-default}
else
  EXE=$ORB_ROOT/Examples/Stereo/stereo_euroc
  CFG_NAME=${CFG_NAME:-orb_stereo.yaml};     TAG=${TAG:-stereo}
fi
VOC=$ORB_ROOT/Vocabulary/ORBvoc.txt
CFG=$POOL/configs/$CFG_NAME
for f in "$EXE" "$VOC" "$CFG"; do [[ -e $f ]] || { echo "Not found: $f"; exit 1; }; done
[[ -x /usr/bin/time ]] || { echo "GNU time missing: apt-get install -y time"; exit 1; }

echo "mode=$MODE tag=$TAG config=$CFG_NAME runs=$RUNS bags: $BAGS"
for BAG in $BAGS; do
  DATA=$POOL/euroc_$BAG
  [[ -s $DATA/times.txt ]] || { echo "Not found: $DATA/times.txt"; exit 1; }
  NF=$(wc -l < "$DATA/times.txt")
  for r in $(seq -f "%02g" 1 "$RUNS"); do
    OUT=$POOL/results/testA/$TAG/$BAG/run_$r
    if [[ -e $OUT/DONE ]]; then echo "skip $TAG $BAG run $r (done)"; continue; fi
    mkdir -p "$OUT" && cd "$OUT" || exit 1
    cp "$CFG" config_used.yaml
    echo "=== $TAG $BAG run $r  started $(date +%H:%M:%S)"
    timeout -s KILL "$TIMEOUT" /usr/bin/time -v -o time.txt "$EXE" "$VOC" "$CFG" "$DATA" "$DATA/times.txt" orb > log.txt 2>&1
    rc=$?
    [[ $rc -eq 137 ]] && echo "TIMEOUT after ${TIMEOUT}s (killed)" >> log.txt
    echo "exit code $rc" >> log.txt
    touch DONE
    NP=$(wc -l < f_orb.txt 2>/dev/null || echo 0)
    echo "    exit $rc   poses $NP / frames $NF   resets $(grep -c 'Active map reset' log.txt)   'Not enough motion' $(grep -c 'Not enough motion' log.txt)"
  done
done
echo "done $(date +%H:%M:%S)"
