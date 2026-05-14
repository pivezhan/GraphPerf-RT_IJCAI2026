#!/usr/bin/env bash
# PolyBench "smoke-all" runner with performance output + robust OpenMP binding.
# Runs ALL kernels with MINI_DATASET by default and prints perf like run_bench.sh.

# Do not stop on first failure; keep going to summarize.
set -u -o pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
RUN="$ROOT/run_bench.sh"
if [[ ! -x "$RUN" ]]; then
  echo "Error: run_bench.sh not found or not executable at $RUN" >&2
  exit 2
fi

# Full PolyBench list (newline-delimited).
read -r -d '' BENCHES_NL <<'EOF'
gemm
gemver
gesummv
symm
syr2k
syrk
trmm
2mm
3mm
atax
bicg
doitgen
mvt
cholesky
durbin
gramschmidt
lu
ludcmp
trisolv
correlation
covariance
deriche
floyd-warshall
nussinov
adi
fdtd-2d
heat-3d
jacobi-1d
jacobi-2d
seidel-2d
EOF
mapfile -t BENCHES <<< "$BENCHES_NL"

# Defaults.
DATASET="MINI_DATASET"
THREADS=""                 # let run_bench.sh choose nproc if empty
PERF_FLAG="--perf"         # show perf by default
EVENTS_ARG=()
BINDING="spread"           # spread across cores by default: spread|close|none

usage() {
  cat <<EOF
Usage:
  ./smoke_all_poly.sh [--dataset=MINI_DATASET] [--threads=N]
                      [--perf|--no-perf] [--events=EVT1,EVT2,...]
                      [--bind=spread|close|none]

Defaults:
  dataset = MINI_DATASET
  perf    = --perf
  bind    = spread  (sets OMP_PLACES=cores, OMP_PROC_BIND=spread)
Notes:
  - Script clears GOMP_CPU_AFFINITY, KMP_AFFINITY, and OMP_DYNAMIC to avoid single-core pinning.
  - Use --bind=none to leave binding to OpenMP defaults.
EOF
}

# Parse args.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset=*) DATASET="${1#*=}" ;;
    --threads=*) THREADS="${1#*=}" ;;
    --perf)      PERF_FLAG="--perf" ;;
    --no-perf)   PERF_FLAG="--no-perf" ;;
    --events=*)  EVENTS_ARG=("$1") ;;
    --bind=*)    BINDING="${1#*=}" ;;
    -h|--help)   usage; exit 0 ;;
    *)           echo "Warning: ignoring unknown option '$1'" >&2 ;;
  esac
  shift
done

# Validate dataset (runner also validates).
case "$DATASET" in
  MINI_DATASET|SMALL_DATASET|MEDIUM_DATASET|LARGE_DATASET|EXTRALARGE_DATASET) ;;
  *) echo "Error: invalid dataset '$DATASET'." >&2; exit 2;;
esac

# Prepare threads arg for the runner.
THREADS_ARG=()
if [[ -n "$THREADS" ]]; then
  if ! [[ "$THREADS" =~ ^[0-9]+$ && "$THREADS" -ge 1 ]]; then
    echo "Error: invalid threads '$THREADS'." >&2; exit 2
  fi
  THREADS_ARG=("$THREADS")
fi

# ---- OpenMP affinity hygiene (important when running with sudo) ----
# Clear conflicting vars that can pin everything to CPU 0.
unset GOMP_CPU_AFFINITY
unset KMP_AFFINITY
export OMP_DYNAMIC=FALSE

case "$BINDING" in
  spread)
    export OMP_PLACES=cores
    export OMP_PROC_BIND=spread
    ;;
  close)
    export OMP_PLACES=cores
    export OMP_PROC_BIND=close
    ;;
  none)
    unset OMP_PLACES
    unset OMP_PROC_BIND
    ;;
  *)
    echo "Error: unknown --bind=$BINDING (use spread|close|none)" >&2
    exit 2
    ;;
esac

PASS=0
FAIL=0
REPORT=()

echo "=== PolyBench smoke-all ==="
echo "DATASET : $DATASET"
echo "PERF    : ${PERF_FLAG#--}"
echo "BIND    : $BINDING"
[[ -n "${THREADS_ARG[*]:-}" ]] && echo "THREADS : ${THREADS_ARG[0]}"
echo

for b in "${BENCHES[@]}"; do
  echo "============================================================"
  echo "==> $b | DATASET=$DATASET | ${THREADS:+THREADS=$THREADS | }${PERF_FLAG#--}${EVENTS_ARG:+ | ${EVENTS_ARG[0]#--}}"
  echo "    OMP_PLACES=${OMP_PLACES-<unset>}  OMP_PROC_BIND=${OMP_PROC_BIND-<unset>}  OMP_DYNAMIC=${OMP_DYNAMIC-<unset>}"
  echo "------------------------------------------------------------"
  if "$RUN" "$b" "$DATASET" "${THREADS_ARG[@]}" "$PERF_FLAG" "${EVENTS_ARG[@]}"; then
    REPORT+=("[PASS] $b $DATASET")
    PASS=$((PASS+1))
  else
    REPORT+=("[FAIL] $b $DATASET")
    FAIL=$((FAIL+1))
  fi
  echo
done

echo "=================== SMOKE SUMMARY ==================="
for line in "${REPORT[@]}"; do echo "$line"; done
echo "-----------------------------------------------------"
echo "PASS: $PASS"
echo "FAIL: $FAIL"

exit $(( FAIL > 0 ? 1 : 0 ))
