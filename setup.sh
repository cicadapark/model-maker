#!/usr/bin/env bash
# The one script that sets up a PC to make models from prompts, everything inside this folder:
#
#   ./setup.sh                  check this PC, then install what is missing (asks before each model's licence)
#   ./setup.sh --check-only     check this PC and say what is missing (changes nothing)
#
# Options (with a full setup):
#   --yes, --accept-licences    accept every model's licence without asking (each is still shown) and install the
#                               Rust toolchain into .tools/ without asking, for unattended runs
#   --hf-token-file FILE        a Hugging Face access token, for FLUX.1 [schnell] (the hub asks each account to accept
#                               its terms once; see docs/setup.md). Also HF_TOKEN_FILE, or the hub's own token file.
#   --skip-models               everything but the models (the checks, the bake and the report run without them)
#   --skip-tests                not the self-test at the end
#
# What it installs, all under this folder (delete the folder and it is gone):
#   .tools/uv, .tools/python    uv (a pinned release, its sha256 checked) and the CPython 3.11 it installs
#   .venv/                      the Python environment, from python/requirements.lock (every wheel's sha256 checked),
#                               then every package's licence checked (python/model_maker/check_licences.py)
#   .tools/rust/                Rust (rustup, pinned and checked), only when this PC has neither rustup nor a cargo
#                               as new as Cargo.toml's rust-version
#   target/release/             model-maker-simplify, built from this repository
#   sources/                    the view painter's own code at the commit registry/models.toml names
#   models/                     each model after its licence is shown and accepted, from its hub at its pinned
#                               revision, every file checked against the registry's sha256
# A second run checks what is there and carries on where the first stopped. See docs/setup.md.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

# What the pipeline needs, from what it runs on today.
readonly GPU_GOOD_MIB=16000      # the picture and the views models together
readonly GPU_TRY_MIB=12000       # with the models loaded one at a time: "try it"
readonly CUDA_MAJOR=12
readonly MEMORY_GOOD_GIB=24      # the computer's memory (loading the picture model peaked at 32 GB resident, much
                                 # of it the weights' files mapped into memory: docs/setup.md)
readonly MEMORY_LEAST_GIB=16
readonly DISK_GB=75              # model weights about 57 GB, the Python environment about 6 GB, builds
readonly PYTHON_VERSION=3.11.14  # the CPython uv installs for the environment (what the pipeline is tried on: 3.11)
readonly UV_VERSION=0.13.5
readonly UV_SHA256_X86_64=f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f
readonly UV_SHA256_AARCH64=4bb69cf95ed13f23713bb7bc09f241896ea0c75b7c557e615a20604b82f9c403
readonly RUSTUP_VERSION=1.28.2
readonly RUSTUP_SHA256_X86_64=20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c
readonly RUSTUP_SHA256_AARCH64=e3853c5a252fca15252d07cb23a1bdd9377a8c6f3efa01531109281ae47f841c

tools="$here/.tools"
venv="$here/.venv"
# uv's downloads, its Pythons and its cache stay in this folder too.
export UV_CACHE_DIR="$tools/uv-cache" UV_PYTHON_INSTALL_DIR="$tools/python" UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed
# A Rust installed by this script.
if [ -x "$tools/rust/cargo/bin/cargo" ]; then
    export RUSTUP_HOME="$tools/rust/rustup" CARGO_HOME="$tools/rust/cargo"
    export PATH="$CARGO_HOME/bin:$PATH"
fi

problems=0
rows=()
mode=setup
yes=0
skip_models=0
skip_tests=0
token_file=""

row() { rows+=("$1|$2|$3"); }
missing() { problems=$((problems + 1)); row "$1" "$2" "MISSING: $3"; }
step() { printf '\n== %s\n' "$1"; }
die() { echo "setup: $*" >&2; exit 1; }

check_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        missing "NVIDIA graphics card" "no nvidia-smi" "an NVIDIA card and its driver (there is no CPU path)"
        return
    fi
    local cards best_name="" best_mib=0 name mib
    if ! cards=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null) || [ -z "$cards" ]; then
        missing "NVIDIA graphics card" "nvidia-smi sees no card" "a working NVIDIA driver"
        return
    fi
    while IFS=, read -r name mib; do
        mib=${mib//[^0-9]/}
        if [ "${mib:-0}" -gt "$best_mib" ]; then best_mib=$mib; best_name=$(echo "$name" | sed 's/^ *//'); fi
    done <<< "$cards"
    local found="$best_name, $((best_mib / 1024)) GB"
    if [ "$best_mib" -ge "$GPU_GOOD_MIB" ]; then
        row "NVIDIA graphics card" "$found" "ok"
    elif [ "$best_mib" -ge "$GPU_TRY_MIB" ]; then
        row "NVIDIA graphics card" "$found" "ok (under 16 GB the view painter's blocks take turns: docs/setup.md)"
    else
        missing "NVIDIA graphics card" "$found" "a card with at least 12 GB"
    fi

    local cuda
    cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1)
    if [ -z "$cuda" ]; then
        missing "CUDA" "not reported" "a driver with CUDA ${CUDA_MAJOR}.x"
    elif [ "${cuda%%.*}" -ge "$CUDA_MAJOR" ]; then
        row "CUDA" "$cuda (driver)" "ok"
    else
        missing "CUDA" "$cuda" "a driver with CUDA ${CUDA_MAJOR}.x or newer"
    fi
}

# The computer's memory in GiB (MemTotal); SETUP_MEMINFO names another file in its place, for the tests.
memory_gib() {
    local total_kb
    total_kb=$(awk '/^MemTotal:/ { print $2 }' "${SETUP_MEMINFO:-/proc/meminfo}" 2>/dev/null || true)
    echo $(( ${total_kb:-0} / 1024 / 1024 ))
}

check_memory() {
    local gib
    gib=$(memory_gib)
    if [ "$gib" -ge "$MEMORY_GOOD_GIB" ]; then
        row "Memory" "$gib GB" "ok"
    elif [ "$gib" -ge "$MEMORY_LEAST_GIB" ]; then
        row "Memory" "$gib GB" "try it ($MEMORY_GOOD_GIB GB or more is advised; swap helps)"
    else
        missing "Memory" "$gib GB" "at least $MEMORY_LEAST_GIB GB ($MEMORY_GOOD_GIB GB or more is advised)"
    fi
}

check_disk() {
    local free_kb free_gb have_gb=0 need
    free_kb=$(df -Pk . | awk 'NR == 2 { print $4 }')
    free_gb=$((free_kb / 1024 / 1024))
    # What an earlier run already put here counts as done.
    if [ -d models ]; then have_gb=$(( $(du -sk models 2>/dev/null | cut -f1) / 1024 / 1024 )); fi
    need=$((DISK_GB - have_gb))
    [ "$need" -lt 5 ] && need=5
    if [ "$free_gb" -ge "$need" ]; then
        row "Free disk here" "$free_gb GB" "ok"
    else
        missing "Free disk here" "$free_gb GB" "at least $need GB free in this folder"
    fi
}

# Python: setup installs its own CPython 3.11 with uv, so the PC's own Python only has to run nothing at all.
check_python() {
    if [ -x "$venv/bin/python" ]; then
        row "Python" "$("$venv/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])') (.venv)" "ok"
    elif command -v python3 >/dev/null 2>&1; then
        row "Python" "$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])') (this PC's)" \
            "ok (setup installs its own $PYTHON_VERSION)"
    else
        row "Python" "none" "ok (setup installs its own $PYTHON_VERSION)"
    fi
}

# The Rust version Cargo.toml asks for (its rust-version).
rust_version_needed() {
    awk -F'"' '/^rust-version *=/ { print $2; exit }' "$here/Cargo.toml"
}

# A Rust that builds this repository: rustup (it installs the toolchain rust-toolchain.toml pins on first use), or a
# cargo at least as new as Cargo.toml's rust-version. A distribution's own older cargo is not one.
usable_rust() {
    command -v rustup >/dev/null 2>&1 && return 0
    command -v cargo >/dev/null 2>&1 || return 1
    local found needed
    found=$(cargo --version 2>/dev/null | awk '{ print $2 }')
    needed=$(rust_version_needed)
    [ -n "$found" ] && [ "$(printf '%s\n%s\n' "$needed" "$found" | sort -V | head -n 1)" = "$needed" ]
}

check_rust() {
    if command -v rustup >/dev/null 2>&1; then
        row "Rust (rustup)" "$(rustup --version 2>/dev/null | head -n 1)" "ok (it installs the pinned toolchain)"
    elif usable_rust; then
        row "Rust (cargo)" "$(cargo --version 2>/dev/null | head -n 1)" "ok"
    elif command -v cargo >/dev/null 2>&1; then
        row "Rust (cargo)" "$(cargo --version 2>/dev/null | head -n 1)" \
            "too old, no rustup (setup installs Rust $(rust_version_needed) into .tools/rust)"
    else
        row "Rust (cargo)" "none" "ok (setup installs its own into .tools/rust)"
    fi
    if command -v cc >/dev/null 2>&1 && command -v c++ >/dev/null 2>&1; then
        row "C and C++ compiler" "$(cc --version 2>/dev/null | head -n 1)" "ok"
    else
        missing "C and C++ compiler" "none" "gcc and g++ (on Debian and Ubuntu: build-essential), for meshoptimizer"
    fi
}

check_tool() {
    local label=$1 command=$2 needed=$3 version
    if command -v "$command" >/dev/null 2>&1; then
        # Its version number alone: the first line of --version can be longer than the table's column. A --version
        # that fails or prints no number must not end the check (set -e, pipefail): "(version unknown)".
        version=$("$command" --version 2>&1 | head -n 1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -n 1 || true)
        row "$label" "$command ${version:-(version unknown)}" "ok"
    else
        missing "$label" "none" "$needed"
    fi
}

# tesseract, optional: found, its version and its English data; not found or not usable, how to install it (not counted
# as missing: the pipeline runs without it and says what it did not judge). Version 5 or newer is needed (version 4
# has no Sauvola threshold). setup.sh installs no system package.
readonly TESSERACT_INSTALL="reads pictures for painted lettering: tesseract 5 or newer with English data (Debian 12, \
Ubuntu 24.04: sudo apt install tesseract-ocr tesseract-ocr-eng; Ubuntu 22.04 ships 4: docs/setup.md)"
check_tesseract() {
    local version major languages
    if ! command -v tesseract >/dev/null 2>&1; then
        row "tesseract" "none" "optional, recommended: $TESSERACT_INSTALL"
        return
    fi
    # Tesseract 3 and some builds of 4 print their version on their errors; a --version that fails or prints no
    # number must not end the check (set -e, pipefail).
    version=$(tesseract --version 2>&1 | head -n 1 | grep -oE '[0-9]+(\.[0-9]+)+' | head -n 1 || true)
    major=${version%%.*}
    languages=$(tesseract --list-langs 2>&1 || true)
    if [ -z "$version" ]; then
        row "tesseract" "tesseract (version unknown)" "optional, not used (no version found): $TESSERACT_INSTALL"
    elif [ "$major" -lt 5 ]; then
        row "tesseract" "tesseract $version" "optional, not used (too old): $TESSERACT_INSTALL"
    elif ! grep -qx 'eng' <<< "$(tr -d '\r' <<< "$languages")"; then
        row "tesseract" "tesseract $version, no English data" "optional, not used: $TESSERACT_INSTALL"
    else
        row "tesseract" "tesseract $version" "ok"
    fi
}

check_system() {
    local system machine
    system=$(uname -s)
    machine=$(uname -m)
    if [ "$system" = "Linux" ]; then
        if grep -qi microsoft /proc/version 2>/dev/null; then
            row "System" "Linux (WSL2), $machine" "should work, not yet tried by us"
        else
            row "System" "Linux, $machine" "ok"
        fi
    else
        missing "System" "$system" "Linux (or WSL2 on Windows)"
    fi
    if [ "$machine" != "x86_64" ]; then
        missing "Processor" "$machine" "x86_64 (the pinned PyTorch with CUDA is built for it)"
    fi
}

print_table() {
    local line check found result
    printf '\n%-22s %-44s %s\n' "Check" "Found" "Result"
    printf '%-22s %-44s %s\n' "-----" "-----" "------"
    for line in "${rows[@]}"; do
        IFS='|' read -r check found result <<< "$line"
        printf '%-22s %-44s %s\n' "$check" "${found:0:44}" "$result"
    done
    echo
}

checked() {
    check_system
    check_gpu
    check_memory
    check_disk
    check_python
    check_rust
    check_tool "git" git "git"
    check_tool "curl" curl "curl"
    check_tool "tar" tar "tar"
    check_tesseract
    print_table
}

check_only() {
    echo "model-maker: checking this PC (nothing is changed)"
    checked
    if [ "$problems" -eq 0 ]; then
        echo "Everything needed is here, or ./setup.sh installs it."
    else
        echo "$problems thing(s) missing; see MISSING above."
    fi
    echo "This checks the PC, not the models: once they are downloaded, .venv/bin/python -I setup/fetch_models.py"
    echo "--verify checks every model file against its sum. It does not see what else is using the card: on a 12 GB"
    echo "card, stop anything else on it before a make."
    [ "$problems" -eq 0 ]
}

# A pinned download, its sha256 checked before it is used.
fetched() {
    local url=$1 sha256=$2 into=$3
    curl --proto '=https' --tlsv1.2 -fsSL --retry 5 -o "$into.part" "$url"
    if ! echo "$sha256  $into.part" | sha256sum -c --status; then
        rm -f "$into.part"
        die "$url: its sha256 is not the one setup.sh names; refusing it"
    fi
    mv "$into.part" "$into"
}

arch_pick() { # the value for this processor: x86_64 first, aarch64 second
    case "$(uname -m)" in
        x86_64) echo "$1" ;;
        aarch64 | arm64) echo "$2" ;;
        *) die "no pinned download for $(uname -m)" ;;
    esac
}

install_uv() {
    local uv="$tools/uv/uv"
    if [ -x "$uv" ] && [ "$("$uv" --version | awk '{print $2}')" = "$UV_VERSION" ]; then
        echo "uv $UV_VERSION: already in .tools/uv"
        return
    fi
    local target sha
    target=$(arch_pick x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu)
    sha=$(arch_pick "$UV_SHA256_X86_64" "$UV_SHA256_AARCH64")
    mkdir -p "$tools/uv"
    fetched "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$target.tar.gz" "$sha" "$tools/uv.tar.gz"
    tar -xzf "$tools/uv.tar.gz" -C "$tools/uv" --strip-components=1
    rm -f "$tools/uv.tar.gz"
    echo "uv $UV_VERSION (MIT or Apache-2.0): installed in .tools/uv"
}

install_python() {
    local uv="$tools/uv/uv" started=$SECONDS
    if [ -x "$venv/bin/python" ] && [ "$("$venv/bin/python" -c 'import platform; print(platform.python_version())')" = "$PYTHON_VERSION" ]; then
        echo "Python $PYTHON_VERSION: .venv already there"
    else
        rm -rf "$venv"
        # CPython from the python-build-standalone builds, whose sums uv carries; they link libedit, not GNU readline,
        # and leave out gdbm, so nothing under the GPL comes with them.
        "$uv" python install --quiet "$PYTHON_VERSION"
        "$uv" venv --quiet --python "$PYTHON_VERSION" "$venv"
        echo "Python $PYTHON_VERSION: .venv made"
    fi
    # Exactly the lock: what is not in it is removed, and every wheel must match one of its sums.
    "$uv" pip sync --quiet --python "$venv/bin/python" --require-hashes --only-binary :all: python/requirements.lock
    "$uv" pip check --quiet --python "$venv/bin/python"
    echo "Python packages: python/requirements.lock installed ($(du -sh "$venv" | cut -f1)) in $((SECONDS - started)) s"
    "$venv/bin/python" -I python/model_maker/check_licences.py
}

install_rust() {
    if usable_rust; then
        return
    fi
    echo "There is no rustup on this PC, and no cargo as new as $(rust_version_needed). setup.sh can install"
    echo "rustup $RUSTUP_VERSION and the toolchain rust-toolchain.toml names into .tools/rust (this folder only;"
    echo "nothing in your home folder changes)."
    if [ "$yes" -ne 1 ]; then
        local answer
        printf 'Install Rust into .tools/rust? [y/N] '
        # The terminal when there is one, else what is piped in.
        { read -r answer < /dev/tty; } 2>/dev/null || read -r answer || answer=""
        case "$answer" in y | Y | yes) ;; *) die "model-maker-simplify needs Rust; install it (https://rustup.rs) and run setup again" ;; esac
    fi
    local target sha
    target=$(arch_pick x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu)
    sha=$(arch_pick "$RUSTUP_SHA256_X86_64" "$RUSTUP_SHA256_AARCH64")
    mkdir -p "$tools/rust"
    fetched "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/$target/rustup-init" "$sha" "$tools/rustup-init"
    chmod +x "$tools/rustup-init"
    export RUSTUP_HOME="$tools/rust/rustup" CARGO_HOME="$tools/rust/cargo"
    # Its note about PATH is for a shell's own setup, which is left alone: a second run of setup.sh finds it here.
    "$tools/rustup-init" -y --quiet --no-modify-path --profile minimal --default-toolchain none > /dev/null
    rm -f "$tools/rustup-init"
    export PATH="$CARGO_HOME/bin:$PATH"
    echo "rustup $RUSTUP_VERSION (MIT or Apache-2.0): installed in .tools/rust"
}

build_rust() {
    local started=$SECONDS
    # rust-toolchain.toml pins the toolchain; rustup installs it on first use.
    cargo build --release --locked --quiet -p model-maker-simplify
    echo "model-maker-simplify: built in $((SECONDS - started)) s (target/release/model-maker-simplify)"
}

fetch_models() {
    local args=()
    [ "$yes" -eq 1 ] && args+=(--yes)
    [ -n "$token_file" ] && args+=(--hf-token-file "$token_file")
    "$venv/bin/python" -I setup/fetch_models.py --models "$here/models" --sources "$here/sources" "${args[@]}"
}

self_test() {
    local started=$SECONDS
    # The checks, the bake, the export and the report on made-up shapes (no card, about 2 to 11 minutes). The tests
    # make made-up jobs whose lines ("fails: paint covers the surface", a bow refused) are what those tests expect,
    # so only the result is shown; on a failure, the end of the log.
    local log="$here/.tools/self-test.log" status=0
    echo "self-test: the Python tests (made-up objects, no card; their own output goes to .tools/self-test.log)"
    (cd python && "$venv/bin/python" -m unittest discover -q -s tests -t . > "$log" 2>&1) || status=$?
    grep -E "^Ran [0-9]+ tests|^OK" "$log" || true
    if [ "$status" -ne 0 ]; then
        tail -n 40 "$log"
        echo "self-test: the Python tests FAILED (the whole log: .tools/self-test.log)"
    fi
    if [ "$skip_models" -eq 0 ]; then
        # Every model file present and its sum the registry's, every stage's code importing, the card seen.
        "$venv/bin/python" -I setup/fetch_models.py --models "$here/models" --sources "$here/sources" --verify | tail -n 2
        "$venv/bin/python" -I setup/smoke.py "$here/models" "$here/sources"
        "$venv/bin/python" -I -c 'import torch; assert torch.cuda.is_available(), "PyTorch sees no card"; p = torch.cuda.get_device_properties(0); print(f"PyTorch {torch.__version__} sees {p.name}, {p.total_memory / 2**30:.0f} GB")'
    fi
    echo "self-test: $((SECONDS - started)) s"
}

setup() {
    local started=$SECONDS
    echo "model-maker: setting up this PC (everything goes into $here)"
    checked
    if [ "$problems" -ne 0 ]; then
        echo "$problems thing(s) missing; see MISSING above. Nothing was changed."
        exit 1
    fi
    step "uv and Python"
    install_uv
    install_python
    step "Rust and model-maker-simplify"
    install_rust
    build_rust
    if [ "$skip_models" -eq 0 ]; then
        step "The models and their code (each licence first)"
        fetch_models
    fi
    if [ "$skip_tests" -eq 0 ]; then
        step "Self-test"
        self_test
    fi
    echo
    echo "Set up in $((SECONDS - started)) s. Make an object:"
    echo "  ./model-maker make \"an old looking chair\""
    echo "  ./model-maker make \"an old iron sword\" --size 0.9"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --check-only) mode=check ;;
        --yes | --accept-licences) yes=1 ;;
        --hf-token-file) token_file="${2:?--hf-token-file needs a file}"; shift ;;
        --hf-token-file=*) token_file="${1#*=}" ;;
        --skip-models) skip_models=1 ;;
        --skip-tests) skip_tests=1 ;;
        -h | --help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "setup: unknown option $1 (see ./setup.sh --help)" >&2; exit 2 ;;
    esac
    shift
done

if [ "$mode" = check ]; then
    check_only
else
    setup
fi
