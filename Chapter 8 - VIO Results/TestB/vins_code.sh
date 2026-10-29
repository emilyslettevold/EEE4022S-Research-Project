#!/usr/bin/env bash
# vins_code.sh - put a chosen mix of STOCK / PATCHED VINS-Fusion source into the running 'vins' container
# and rebuild. Your repo is never modified: stock files come from git (HEAD), patched files from your
# working copy. Run on the LENOVO (not in the container).
#
#   ./vins_code.sh check     # show which files differ from stock (run this first)
#   ./vins_code.sh B0        # stock code
#   ./vins_code.sh B1        # + stereo sync tolerance 10 ms
#   ./vins_code.sh B3        # + stereo pyramid 4 levels
#   ./vins_code.sh B4        # + triangulation rewrite, depth bounds, INIT_DEPTH 1.5 (+ metrics logging) = your current code
#   ./vins_code.sh patched   # same as B4: put everything back
set -euo pipefail
REPO=${REPO:-$HOME/Desktop/VINS-Fusion/VINS-Fusion}
CT=${CT:-vins}
DST=/root/catkin_ws/src/VINS-Fusion
SYNC=vins_estimator/src/rosNodeTest.cpp
PYR=vins_estimator/src/featureTracker/feature_tracker.cpp
FM=vins_estimator/src/estimator/feature_manager.cpp
EST=vins_estimator/src/estimator/estimator.cpp
PAR=vins_estimator/src/estimator/parameters.cpp      # INIT_DEPTH 5.0 -> 1.5 (depth guess): part of the depth fix, B4
KNOWN="$SYNC $PYR $FM $EST $PAR"
cd "$REPO"
git rev-parse HEAD >/dev/null 2>&1 || { echo "$REPO is not a git repo"; exit 1; }

if [[ ${1:-} == check ]]; then
  echo "HEAD: $(git log --oneline -1)"
  echo "Modified source files (vs git HEAD):"
  git diff --stat HEAD -- '*.cpp' '*.h' '*.hpp'
  for f in $KNOWN; do
    if git diff --quiet HEAD -- "$f"; then echo "  WARNING: $f is NOT modified - its step will change nothing"; fi
  done
  others=$(git diff --name-only HEAD -- '*.cpp' '*.h' '*.hpp' | grep -v -F -x -f <(tr ' ' '\n' <<<"$KNOWN") || true)
  [[ -n $others ]] && echo "  NOTE: these other modified files stay PATCHED in every step: $others"
  exit 0
fi

case ${1:-} in
  B0)          stock="$SYNC $PYR $FM $EST $PAR" ;;
  B1)          stock="$PYR $FM $EST $PAR" ;;
  B3)          stock="$FM $EST $PAR" ;;
  B4|patched)  stock="" ;;
  *) echo "usage: $0 check|B0|B1|B3|B4|patched"; exit 1 ;;
esac
docker ps --format '{{.Names}}' | grep -qx "$CT" || docker start "$CT" >/dev/null
TMP=$(mktemp -d)
for f in $KNOWN; do
  mkdir -p "$TMP/$(dirname "$f")"
  if [[ " $stock " == *" $f "* ]]; then git show "HEAD:$f" > "$TMP/$f"; src=stock; else cp "$f" "$TMP/$f"; src=patched; fi
  docker cp "$TMP/$f" "$CT:$DST/$f"
  printf "  %-50s %-8s %s\n" "$f" "$src" "$(md5sum "$TMP/$f" | cut -c1-8)"
done
rm -rf "$TMP"
echo "building..."
docker exec "$CT" bash -c 'source /opt/ros/kinetic/setup.bash && cd /root/catkin_ws && catkin build vins -j4 2>&1 | tail -3'
echo "$1" | docker exec -i "$CT" tee /root/CODE_VERSION >/dev/null
echo "container now runs code version: $1"
