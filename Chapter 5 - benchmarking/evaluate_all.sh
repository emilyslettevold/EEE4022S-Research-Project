#!/usr/bin/env bash
# Convert and evaluate benchmark runs, one sequence at a time.
# Needs python3 with numpy, scipy, pyyaml, matplotlib (run on the Mac in vins_env).
#
#   DATA_ROOT=~/Desktop/ORBSLAM3/datasets/Tank/euroc ./evaluate_all.sh                          # every sequence
#   DATA_ROOT=... ONLY="Structure_Easy" ./evaluate_all.sh                                      # one sequence
#   DATA_ROOT=... ONLY="WholeTank_Medium WholeTank_Hard" PLOTS=all ./evaluate_all.sh           # plots for every run
#
# Output, per sequence (never overwritten by evaluating another sequence):
#   <RESULTS>/eval/<Sequence>/runs.csv       one row per run, both frameworks
#   <RESULTS>/eval/<Sequence>/summary.csv    ORB3 vs VINS for that sequence (also printed)
#   <RESULTS>/eval/<Sequence>/plots/         error-vs-time and top-down plots
# Combined over all evaluated sequences:
#   <RESULTS>/eval/runs_all.csv, <RESULTS>/eval/summary_all.csv
set -uo pipefail
BENCH=${BENCH:-$(cd "$(dirname "$0")" && pwd)}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the folder holding <Sequence>/gt.csv}
RESULTS=${RESULTS:-$BENCH/results/accuracy}
PARAMS=${PARAMS:-$BENCH/benchmark_params.yaml}
ONLY=${ONLY:-}
PLOTS=${PLOTS:-first}                                   # first = plot run_01 only; all = every run
EVAL=$RESULTS/eval
mkdir -p "$EVAL/gt"

# sequences present in the results tree (either framework)
SEQS=$(ls -d "$RESULTS"/ORB3/*/ "$RESULTS"/VINS/*/ 2>/dev/null | xargs -n1 basename | sort -u)

for SEQ in $SEQS; do
  [[ -n "$ONLY" && " $ONLY " != *" $SEQ "* ]] && continue
  [[ -f $DATA_ROOT/$SEQ/gt.csv ]] || { echo "skip $SEQ (no $DATA_ROOT/$SEQ/gt.csv)"; continue; }
  OUT=$EVAL/$SEQ; mkdir -p "$OUT/plots"; rm -f "$OUT/runs.csv"
  echo "================ $SEQ ================"

  GT=$EVAL/gt/${SEQ}_gt.txt; GTF=$EVAL/gt/${SEQ}_gt_full.txt
  [[ -f $GT ]] || python3 "$BENCH/to_tum.py" --method gt --input "$DATA_ROOT/$SEQ/gt.csv" --output "$GT"
  if [[ ! -f $GTF && -s $DATA_ROOT/$SEQ/gt_full.csv ]] && [[ $(grep -vc '^#' "$DATA_ROOT/$SEQ/gt_full.csv") -gt 0 ]]; then
    python3 "$BENCH/to_tum.py" --method gt --input "$DATA_ROOT/$SEQ/gt_full.csv" --output "$GTF"
  fi
  GTF_ARG=(); [[ -f $GTF ]] && GTF_ARG=(--gt-full "$GTF")

  for METHOD in ORB3 VINS; do
    for RUN in "$RESULTS/$METHOD/$SEQ"/run_[0-9][0-9]/; do         # only benchmark runs
      [[ -d $RUN ]] || continue
      r=$(basename "$RUN"); r=${r#run_}
      if [[ $METHOD == ORB3 ]]; then RAW=$RUN/f_orb.txt; FMT=orb; else RAW=$RUN/vio.csv; FMT=vins; fi
      EST=$RUN/est_cam_tum.txt; rm -f "$EST"
      [[ -s $RAW ]] && python3 "$BENCH/to_tum.py" --method $FMT --input "$RAW" --output "$EST" \
                         --seq "$SEQ" --params "$PARAMS" > /dev/null
      PLOT=""; [[ $PLOTS == all || $r == 01 ]] && PLOT=$OUT/plots/${METHOD}_run$r
      python3 "$BENCH/evaluate.py" run --est "$EST" --gt "$GT" ${GTF_ARG[@]+"${GTF_ARG[@]}"} \
        --method $METHOD --seq "$SEQ" --run $((10#$r)) --out-csv "$OUT/runs.csv" \
        --log "$RUN/log.txt" --time "$RUN/time.txt" ${PLOT:+--plot "$PLOT"}
    done
  done
  [[ -f $OUT/runs.csv ]] && python3 "$BENCH/evaluate.py" summarise --runs-csv "$OUT/runs.csv" --out "$OUT/summary.csv"
done

# combined table over every sequence that has been evaluated so far
first=1; rm -f "$EVAL/runs_all.csv"
for f in "$EVAL"/*/runs.csv; do
  [[ -f $f ]] || continue
  if [[ $first == 1 ]]; then cat "$f" > "$EVAL/runs_all.csv"; first=0; else tail -n +2 "$f" >> "$EVAL/runs_all.csv"; fi
done
if [[ -f $EVAL/runs_all.csv ]]; then
  echo "================ ALL EVALUATED SEQUENCES ================"
  python3 "$BENCH/evaluate.py" summarise --runs-csv "$EVAL/runs_all.csv" --out "$EVAL/summary_all.csv"
fi
