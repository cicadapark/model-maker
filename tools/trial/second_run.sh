#!/usr/bin/env bash
# The first trial's follow-up on this PC's card, into OUT (each part's results beside it, OUT/summary.txt at the end):
#
#   1. paint at fewer steps: three objects of the first trial (FIRST) painted again at 30 and at 20 steps, from their
#      own pictures and shapes; their paint time, coverage and likeness against FIRST's 50 steps;
#   2. the picture words: ten prompts (those whose tries most often failed for a floor or a shadow, the chair and
#      the longbow) painted with each lead, four tries and one round each; how many tries pass;
#   3. the steps changed: the chair (seen from above), the round table (four shape seeds) and the longbow (facing the
#      viewer) made again, pictures, shape and checks, unpainted.
#
#   tools/trial/second_run.sh FIRST OUT
#
# The models and their code are found as ./model-maker finds them (MODEL_MAKER_MODELS, MODEL_MAKER_SOURCES,
# MODEL_MAKER_PYTHON). Each part keeps what it has done, so the script can be started again.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
first="${1:?usage: second_run.sh FIRST OUT}"
out="${2:?usage: second_run.sh FIRST OUT}"
mm="$here/model-maker"
mkdir -p "$out"
cargo build --release --locked -p model-maker-simplify --manifest-path "$here/Cargo.toml"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
log="$out/second-run.log"

timed() {
    local started
    started=$(date +%s)
    "$@" 2>&1 | tee -a "$log"
    echo "  ($(( $(date +%s) - started )) s)" | tee -a "$log"
}

# 1. Paint at 30 and 20 steps: each object's picture and shape copied, nothing painted.
paint_prompts="$out/paint-prompts.txt"
printf '%s\n' "a wooden barrel with iron hoops" "a clay jug" "an old iron sword" > "$paint_prompts"
verify=()
for steps in 30 20; do
    folder="$out/paint-$steps"
    for slug in wooden-barrel-iron-hoops clay-jug iron-sword; do
        mkdir -p "$folder/$slug"
        for item in job.json picture.png mask.png shape.npz pictures; do
            [ -e "$folder/$slug/$item" ] || cp -r "$first/$slug/$item" "$folder/$slug/$item"
        done
    done
    timed "$mm" batch "$paint_prompts" --out "$folder" --stages paint,finish --paint-steps "$steps" "${verify[@]}"
    verify=(--no-verify)
done

# 2. The picture words: each lead, four tries, one round.
picture_prompts="$out/picture-prompts.txt"
printf '%s\n' "a wooden treasure chest" "a round wooden shield with an iron boss" "a leather-bound book, closed" \
    "a clay jug" "a small round table" "a rusty iron lantern" "a candle in a brass holder" \
    "a wooden barrel with iron hoops" "an old looking chair" "a wooden longbow" > "$picture_prompts"
verify=()
for lead in floating isolated cutout; do
    timed "$mm" batch "$picture_prompts" --out "$out/lead-$lead" --stages pictures --lead "$lead" \
        --picture-rounds 1 --tries 4 "${verify[@]}"
    verify=(--no-verify)
done

# 3. The steps changed, unpainted.
fix_prompts="$out/fix-prompts.txt"
printf '%s\n' "an old looking chair" "a small round table" "a wooden longbow" > "$fix_prompts"
timed "$mm" batch "$fix_prompts" --out "$out/fixes" --stages pictures,shape,finish --shape-tries 4

{
    echo "== 1. paint steps (FIRST is 50 steps)"
    python3 -I "$here/tools/trial/summary.py" paint "$first" "$out/paint-30" "$out/paint-20"
    echo "== 2. picture words"
    python3 -I "$here/tools/trial/summary.py" pictures "$out/lead-floating" "$out/lead-isolated" "$out/lead-cutout"
    echo "== 3. the steps changed"
    python3 -I "$here/tools/trial/summary.py" checks "$out/fixes"
} | tee "$out/summary.txt"
