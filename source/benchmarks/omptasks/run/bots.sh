#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# Unified BOTS runner (RUN FROM run/ DIRECTORY) — STATIC VARIANT LISTS
#
# Usage:
#   ./bots.sh <benchmark> <bin-variant> <input>
#
# Examples:
#   ./bots.sh fft       bin-omp-tasks 262144
#   ./bots.sh strassen  bin-omp-tasks 512
#   ./bots.sh fib       bin-serial    35
#   ./bots.sh nqueens   bin-omp-tasks 14
#
# Notes:
# - <bin-variant> must be one of the statically-defined variants below (no bin scan).
# - We STRICT-validate inputs against known sets per benchmark.
# - We normalize some variant names before passing to run-<benchmark>.sh to match
#   runner expectations (e.g., if_clause -> if, single-omp-tasks -> single-omp).
# ==============================================================================

# -------- Config: common perf counters --------
PERF_EVENTS="cycles,cache-references,cache-misses,branch-instructions,task-clock,context-switches,minor-faults,major-faults,branch-misses,branches,instructions,page-faults,cpu-clock"

# -------- Bench catalog --------
declare -a BENCHES=(alignment fft fib floorplan health knapsack nqueens sort sparselu strassen uts concom)

# -------- Allowed inputs (STRICT) --------
declare -a ALIGNMENT_ALLOWED=("prot.20.aa" "prot.100.aa")
declare -a FFT_ALLOWED=("258064" "260100" "262144" "264196" "266256")
declare -a FIB_ALLOWED=("10" "35" "40" "50" "53")
declare -a FLOORPLAN_ALLOWED=("input.5" "input.15" "input.20")
declare -a HEALTH_ALLOWED=("test.input" "small.input" "medium.input" "large.input")
declare -a KNAPSACK_ALLOWED=("knapsack-016.input" "knapsack-036.input" "knapsack-040.input" "knapsack-044.input")
declare -a NQUEENS_ALLOWED=("10" "13" "14" "15" "16")
declare -a SORT_ALLOWED=("8388608" "33554432" "134217728" "536870912")
declare -a SPARSELU_ALLOWED=("25x25" "25x100" "75x100" "100x100")
declare -a STRASSEN_ALLOWED=("508" "510" "512" "514" "516")
declare -a UTS_ALLOWED=("test.input" "small.input" "medium.input" "large.input")
declare -a CONCOM_ALLOWED=("100000" "1000000" "5000000")

# -------- Allowed bin-variants per benchmark (STATIC) --------
# These mirror your ../bin/ listing exactly; no filesystem probing.

# alignment:
declare -a ALIGNMENT_BIN_ALLOWED=("bin-for-omp-tasks" "bin-for-omp-tasks-tied" "bin-serial" "bin-single-omp-tasks" "bin-single-omp-tasks-tied")

# fft:
declare -a FFT_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-tied" "bin-serial")

# fib:
declare -a FIB_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-if_clause" "bin-omp-tasks-if_clause-tied" "bin-omp-tasks-manual" "bin-omp-tasks-manual-tied" "bin-omp-tasks-tied" "bin-serial")

# floorplan:
declare -a FLOORPLAN_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-if_clause" "bin-omp-tasks-if_clause-tied" "bin-omp-tasks-manual" "bin-omp-tasks-manual-tied" "bin-omp-tasks-tied" "bin-serial")

# health:
declare -a HEALTH_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-if_clause" "bin-omp-tasks-if_clause-tied" "bin-omp-tasks-manual" "bin-omp-tasks-manual-tied" "bin-omp-tasks-tied" "bin-serial")

# knapsack:
declare -a KNAPSACK_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-if_clause" "bin-omp-tasks-if_clause-tied" "bin-omp-tasks-manual" "bin-omp-tasks-manual-tied" "bin-omp-tasks-tied")

# nqueens:
declare -a NQUEENS_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-if_clause" "bin-omp-tasks-if_clause-tied" "bin-omp-tasks-manual" "bin-omp-tasks-manual-tied" "bin-omp-tasks-tied" "bin-serial")

# sort:
declare -a SORT_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-tied" "bin-serial")

# sparselu:
declare -a SPARSELU_BIN_ALLOWED=("bin-for-omp-tasks" "bin-for-omp-tasks-tied" "bin-serial" "bin-single-omp-tasks" "bin-single-omp-tasks-tied")

# strassen:
declare -a STRASSEN_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-if_clause" "bin-omp-tasks-if_clause-tied" "bin-omp-tasks-manual" "bin-omp-tasks-manual-tied" "bin-omp-tasks-tied" "bin-serial")

# uts:
declare -a UTS_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-tied" "bin-serial")

# concom:
declare -a CONCOM_BIN_ALLOWED=("bin-omp-tasks" "bin-omp-tasks-tied")

# -------- Helpers --------
err() { echo "Error: $*" >&2; exit 1; }
lower() { tr '[:upper:]' '[:lower:]'; }

in_array() {
  local needle="$1"; shift
  local -a arr=("$@")
  for x in "${arr[@]}"; do [[ "$x" == "$needle" ]] && return 0; done
  return 1
}

allowed_inputs_for_bench() {
  case "$1" in
    alignment)  printf "%s\n" "${ALIGNMENT_ALLOWED[@]}";;
    fft)        printf "%s\n" "${FFT_ALLOWED[@]}";;
    fib)        printf "%s\n" "${FIB_ALLOWED[@]}";;
    floorplan)  printf "%s\n" "${FLOORPLAN_ALLOWED[@]}";;
    health)     printf "%s\n" "${HEALTH_ALLOWED[@]}";;
    knapsack)   printf "%s\n" "${KNAPSACK_ALLOWED[@]}";;
    nqueens)    printf "%s\n" "${NQUEENS_ALLOWED[@]}";;
    sort)       printf "%s\n" "${SORT_ALLOWED[@]}";;
    sparselu)   printf "%s\n" "${SPARSELU_ALLOWED[@]}";;
    strassen)   printf "%s\n" "${STRASSEN_ALLOWED[@]}";;
    uts)        printf "%s\n" "${UTS_ALLOWED[@]}";;
    concom)     printf "%s\n" "${CONCOM_ALLOWED[@]}";;
    *)          ;;
  esac
}

validate_input() {
  local b="$1" in="$2"
  mapfile -t allowed < <(allowed_inputs_for_bench "$b")
  in_array "$in" "${allowed[@]}" || {
    echo "Allowed inputs for '$b': ${allowed[*]}" >&2
    err "Invalid input '$in' for $b."
  }
}

allowed_bins_for_bench() {
  case "$1" in
    alignment)  printf "%s\n" "${ALIGNMENT_BIN_ALLOWED[@]}";;
    fft)        printf "%s\n" "${FFT_BIN_ALLOWED[@]}";;
    fib)        printf "%s\n" "${FIB_BIN_ALLOWED[@]}";;
    floorplan)  printf "%s\n" "${FLOORPLAN_BIN_ALLOWED[@]}";;
    health)     printf "%s\n" "${HEALTH_BIN_ALLOWED[@]}";;
    knapsack)   printf "%s\n" "${KNAPSACK_BIN_ALLOWED[@]}";;
    nqueens)    printf "%s\n" "${NQUEENS_BIN_ALLOWED[@]}";;
    sort)       printf "%s\n" "${SORT_BIN_ALLOWED[@]}";;
    sparselu)   printf "%s\n" "${SPARSELU_BIN_ALLOWED[@]}";;
    strassen)   printf "%s\n" "${STRASSEN_BIN_ALLOWED[@]}";;
    uts)        printf "%s\n" "${UTS_BIN_ALLOWED[@]}";;
    concom)     printf "%s\n" "${CONCOM_BIN_ALLOWED[@]}";;
    *)          ;;
  esac
}

validate_bin_for_bench() {
  local b="$1" bin="$2"
  [[ "$bin" == bin-* ]] || err "bin-variant must start with 'bin-'. Got '$bin'."
  mapfile -t allowed_bins < <(allowed_bins_for_bench "$b")
  in_array "$bin" "${allowed_bins[@]}" || {
    echo "Allowed bin-variants for '$b': ${allowed_bins[*]:-<none defined>}" >&2
    err "Unsupported bin-variant '$bin' for benchmark '$b'."
  }
}

# Normalize variant names to what run-<benchmark>.sh expects in -ver <...>
# Fixes:
#   - "*-if_clause"        -> "*-if"
#   - "*-if_clause-tied"   -> "*-if-tied"
#   - "single-omp-tasks"   -> "single-omp"
#   - "single-omp-tasks-tied" -> "single-omp-tied"
normalize_variant_for_runner() {
  local v="$1"
  v="${v/omp-tasks-if_clause-tied/omp-tasks-if-tied}"
  v="${v/omp-tasks-if_clause/omp-tasks-if}"
  v="${v/single-omp-tasks-tied/single-omp-tied}"
  v="${v/single-omp-tasks/single-omp}"
  echo "$v"
}

print_usage() {
  cat <<'EOF'
Usage:
  ./bots.sh <benchmark> <bin-variant> <input>

Benchmarks:
  alignment, fft, fib, floorplan, health, knapsack, nqueens, sort, sparselu, strassen, uts, concom

Notes:
- Variants are taken from static definitions in this script and must be passed as "bin-<variant>".
- Inputs are strictly validated against known sets.
- Some variant names are normalized for the runner (if_clause→if, single-omp-tasks→single-omp).
EOF
}

# -------- Args --------
if [[ $# -ne 3 ]]; then
  print_usage
  err "Expected 3 arguments. Got $#."
fi

BENCH_RAW="$1"
BIN_VARIANT_DIR="$2"
INPUT_RAW="$3"

BENCH="$(echo "$BENCH_RAW" | lower)"
INPUT="$INPUT_RAW"
VARIANT="${BIN_VARIANT_DIR#bin-}"

# Validate
in_array "$BENCH" "${BENCHES[@]}" || { echo "Supported benchmarks: ${BENCHES[*]}" >&2; err "Unsupported benchmark '$BENCH'."; }
validate_input "$BENCH" "$INPUT"
validate_bin_for_bench "$BENCH" "$BIN_VARIANT_DIR"

# Paths
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
RUN_SCRIPT="$RUN_DIR/run-${BENCH}.sh"
[[ -x "$RUN_SCRIPT" ]] || err "Run script not found or not executable: $RUN_SCRIPT"

# perf presence (warn only)
if ! command -v perf >/dev/null 2>&1; then
  echo "Warning: 'perf' not found in PATH. Command will fail unless perf is installed." >&2
fi

# Normalize variant for runner expectations
VARIANT_FOR_RUNNER="$(normalize_variant_for_runner "$VARIANT")"

# Execute
exec perf stat -e "$PERF_EVENTS" "$RUN_SCRIPT" -ver "$VARIANT_FOR_RUNNER" -i "$INPUT"
