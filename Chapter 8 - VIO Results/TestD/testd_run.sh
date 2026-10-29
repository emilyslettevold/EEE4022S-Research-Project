#!/usr/bin/env bash
# testd_run.sh - run both frozen Test D configs on ONE prepared recording, then score and plot it.
#
#   source ~/evo-env/bin/activate
#   ~/vio_data/testd_run.sh NAME
#
# Needs ~/vio_data/NAME/NAME_half.bag (from testd_prep.sh); uses ~/vio_data/NAME/gt_tum.txt if it exists.
# Results:  ~/vio_data/testD/NAME/NAME_SO.csv, NAME_SI.csv (+ _metrics, _log)
#           ~/vio_data/testD_eval/NAME/  NAME_plan.png, NAME_side.png, NAME_ate.png, testD_results.csv/.md
#           ~/vio_data/testD_eval/all_results.csv/.md  - one table for every recording (this NAME's rows replaced)
# Finished runs are skipped, so it is safe to rerun.
set -euo pipefail
NAME=${1:?usage: testd_run.sh NAME}
VD=$HOME/vio_data
[[ -f $VD/$NAME/${NAME}_half.bag ]] || { echo "no $VD/$NAME/${NAME}_half.bag - run testd_prep.sh first"; exit 1; }
for c in testD_SO.yaml testD_SI.yaml; do [[ -f $VD/config_rov/$c ]] || { echo "missing config_rov/$c (Test D step 1)"; exit 1; }; done
(cd "$VD/config_rov" && md5sum -c testD_configs.md5) || { echo "Test D configs changed since they were frozen - stop"; exit 1; }

docker start vins >/dev/null
code=$(docker exec vins cat /root/CODE_VERSION)
[[ $code == B4 || $code == patched ]] || { echo "container is on code '$code', not B4 - run: cd ~/vio_data && ./vins_code.sh B4"; exit 1; }

echo "=== [$NAME] VINS: stereo only + stereo-inertial"
docker exec -it vins bash -c "RES=/data/testD/$NAME BAGS='$NAME:/data/$NAME/${NAME}_half.bag' CONFIGS='SO:testD_SO.yaml SI:testD_SI.yaml' /data/run_vins_testA.sh"

echo "=== [$NAME] scoring"
GTARG=(); [[ -f $VD/$NAME/gt_tum.txt ]] && GTARG=(--gt "$NAME=$VD/$NAME/gt_tum.txt") || echo "(no gt_tum.txt for $NAME - plots and distances only)"
python3 "$VD/eval_testd.py" --runs "$VD/testD/$NAME" --out "$VD/testD_eval/$NAME" \
  --config SO="$VD/config_rov/testD_SO.yaml" --config SI="$VD/config_rov/testD_SI.yaml" "${GTARG[@]}" \
  --summary "$VD/testD_eval/all_results.csv"
echo "=== [$NAME] done: open $VD/testD_eval/$NAME/"
