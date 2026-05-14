#!/usr/bin/env bash
set -Eeuo pipefail

# PolyBench omnibus runner (BOTS-style)
# Default: threads = number of cores, perf = ON
# Usage:
#   ./run_bench.sh <benchmark> <dataset> [threads] [--perf|--no-perf] [--events=...]
# Examples:
#   ./run_bench.sh atax LARGE_DATASET
#   ./run_bench.sh atax LARGE_DATASET 8
#   ./run_bench.sh jacobi-2d MEDIUM_DATASET --no-perf
#   ./run_bench.sh gemm EXTRALARGE_DATASET --events=cycles,instructions

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BIN_DIR="$ROOT/bin"

declare -a BENCHES=(
  gemm gemver gesummv symm syr2k syrk trmm
  2mm 3mm atax bicg doitgen mvt
  cholesky durbin gramschmidt lu ludcmp trisolv
  correlation covariance deriche floyd-warshall nussinov
  adi fdtd-2d heat-3d jacobi-1d jacobi-2d seidel-2d
)

valid_dataset() {
  case "$1" in
    MINI_DATASET|SMALL_DATASET|MEDIUM_DATASET|LARGE_DATASET|EXTRALARGE_DATASET) return 0 ;;
    *) return 1 ;;
  esac
}

in_array() {
  local x="$1"; shift
  local y
  for y in "$@"; do [[ "$y" == "$x" ]] && return 0; done
  return 1
}

n_cores() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif [[ -r /proc/cpuinfo ]]; then
    grep -c '^processor' /proc/cpuinfo || echo 1
  else
    sysctl -n hw.ncpu 2>/dev/null || echo 1
  fi
}

die() { echo "Error: $*" >&2; exit 1; }

print_usage() {
  cat <<'EOF'
Usage:
  ./run_bench.sh <benchmark> <dataset> [threads] [--perf|--no-perf] [--events=EVT1,EVT2,...]

Benchmarks:
  gemm gemver gesummv symm syr2k syrk trmm
  2mm 3mm atax bicg doitgen mvt
  cholesky durbin gramschmidt lu ludcmp trisolv
  correlation covariance deriche floyd-warshall nussinov
  adi fdtd-2d heat-3d jacobi-1d jacobi-2d seidel-2d

Datasets:
  MINI_DATASET SMALL_DATASET MEDIUM_DATASET LARGE_DATASET EXTRALARGE_DATASET
EOF
}

[[ $# -ge 2 ]] || { print_usage; exit 2; }

BENCH="$1"; shift
DATASET="$1"; shift

in_array "$BENCH" "${BENCHES[@]}" || die "Unsupported benchmark '$BENCH'."
valid_dataset "$DATASET" || die "Invalid dataset '$DATASET'."

# Defaults
PERF_ON=1
EVENTS_DEFAULT="cycles,cache-references,cache-misses,branch-instructions,task-clock,context-switches,minor-faults,major-faults,branch-misses,branches,instructions,page-faults,cpu-clock"
PERF_EVENTS="$EVENTS_DEFAULT"
THREADS=""

# Parse remaining args: accept numeric threads, and flags in any order
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-perf) PERF_ON=0 ;;
    --perf)    PERF_ON=1 ;;
    --events=*) PERF_EVENTS="${1#*=}" ;;
    ''|*[!0-9]*) # not a pure number -> ignore unless starts with '-' we already handled
                  :
                  ;;
    *)          # pure number => threads
                THREADS="$1"
                ;;
  esac
  shift
done

# Threads default = number of cores
if [[ -z "${THREADS}" ]]; then
  THREADS="$(n_cores)"
fi
[[ "$THREADS" =~ ^[0-9]+$ && "$THREADS" -ge 1 ]] || die "Invalid threads '$THREADS'."

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# Executable path matches Makefile naming: <bench>_<dataset lowercased>
dataset_lc="$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')"
exe="$BIN_DIR/${BENCH}_${dataset_lc}"

# Build if missing
if [[ ! -x "$exe" ]]; then
  echo "==> Building $exe"
  make -C "$ROOT" -j "$BENCH" DATASET="$DATASET"
fi

echo "==> Running: $exe"
echo "    DATASET=$DATASET"
echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS"
if [[ "$PERF_ON" -eq 1 ]]; then
  if command -v perf >/dev/null 2>&1; then
    echo "    perf events: $PERF_EVENTS"
    exec perf stat -e "$PERF_EVENTS" "$exe"
  else
    echo "    Warning: 'perf' not found. Running without perf." >&2
    exec "$exe"
  fi
else
  exec "$exe"
fi

