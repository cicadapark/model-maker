#!/bin/bash
# Repaint stored objects with this checkout's paint and finish: their pictures and shapes kept, so only the paint
# (and what follows from it) can differ. For a change to the paint alone (the touch-up's candidate views, 0.3.0),
# measured against the stored run of the same card, seeds and prompts.
#   tools/trial/repaint.sh STORED_BATCH OUT JOB_NAME...
# STORED_BATCH: a batch folder made before (its job folders hold picture.png, mask.png, pictures/, shape.npz).
# OUT: a new folder; OUT/batch gets a copy of each named job's pre-paint files, OUT/prompts.txt their prompts.
# The environment as for any batch (MODEL_MAKER_PYTHON, MODEL_MAKER_MODELS, MODEL_MAKER_SOURCES, ...); extra
# arguments for the batch in REPAINT_ARGS (e.g. "--require-lettering --require-question-judge").
set -euo pipefail
stored=$1 out=$2
shift 2
mkdir -p "$out/batch"
: > "$out/prompts.txt"
for name in "$@"; do
  from="$stored/$name" to="$out/batch/$name"
  [ -f "$from/shape.npz" ] || { echo "repaint: $from has no shape" >&2; exit 2; }
  mkdir -p "$to"
  for kept in job.json picture.png mask.png shape.npz shape-42.npz pictures; do
    [ -e "$from/$kept" ] && cp -a "$from/$kept" "$to/"
  done
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["prompt"])' "$from/job.json" >> "$out/prompts.txt"
done
here=$(cd "$(dirname "$0")/../.." && pwd)
{ git -C "$here" describe --always --dirty 2>/dev/null || cat "$here/REVISION"; } > "$out/revision.txt"
cd "$here"
# shellcheck disable=SC2086
./model-maker batch "$out/prompts.txt" --out "$out/batch" --stages paint,finish ${REPAINT_ARGS:-} > "$out/batch.log" 2>&1
