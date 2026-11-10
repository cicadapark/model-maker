#!/usr/bin/env bash
# Release 1's trial: every prompt of trial-prompts.txt (ten objects, six held in the hand) made end to end on this
# PC's NVIDIA card, stage by stage (each model loaded once), into OUT, with a time per object (OUT/times.csv) and a
# report per object (OUT/<object>/report/report.html). A stage already done for an object is kept, so the run can be
# started again.
#
#   tools/trial/run_trial.sh OUT [more model-maker batch options]
#
# The models and the view painter's code are found as ./model-maker finds them: MODEL_MAKER_MODELS,
# MODEL_MAKER_SOURCES and MODEL_MAKER_PYTHON (see docs/objects.md).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${1:?usage: run_trial.sh OUT [options]}"
shift
mkdir -p "$out"
cargo build --release --locked -p model-maker-simplify --manifest-path "$here/Cargo.toml"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
started=$(date +%s)
"$here/model-maker" batch "$here/tools/trial/trial-prompts.txt" --out "$out" "$@" 2>&1 | tee -a "$out/trial.log"
echo "the trial took $(( $(date +%s) - started )) s; times per object: $out/times.csv" | tee -a "$out/trial.log"
