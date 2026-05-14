#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# Smoke test: run all discovered variants with the first allowed input.
# RUN FROM omptasks/run/
#
# Usage:
#   ./smoke_all.sh
# ==============================================================================

err() { echo "Error: $*" >&2; exit 1; }

declare -a BENCHES=(alignment fft fib floorplan health knapsack nqueens sort sparselu strassen uts concom)

first_input_for() {
  case "$1" in
    alignment)  echo "prot.20.aa";;
    fft)        echo "258064";;
    fib)        echo "10";;
    floorplan)  echo "input.5";;
    health)     echo "test.input";;
    knapsack)   echo "knapsack-016.input";;
    nqueens)    echo "10";;
    sort)       echo "8388608";;
    sparselu)   echo "25x25";;
    strassen)   echo "508";;
    uts)        echo "test.input";;
    concom)     echo "100000";;
    *)          err "Unknown bench '$1'";;
  esac
}

discover_bin_variants() {
  local b="$1"
  local run_dir bin_dir patt
  run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
  bin_dir="$run_dir/../bin"
  patt="$bin_dir/${b}.gcc.*"

  shopt -s nullglob
  local -a files=( $patt )
  shopt -u nullglob

  local f v
  for f in "${files[@]}"; do
    v="${f##*.gcc.}"
    [[ -n "$v" ]] && echo "bin-${v}"
  done | sort -u
}

RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BOTS="$RUN_DIR/bots.sh"
[[ -x "$BOTS" ]] || err "bots.sh not found or not executable at $BOTS"

declare -i pass=0
declare -i fail=0
declare -a REPORT=()

for b in "${BENCHES[@]}"; do
  in0="$(first_input_for "$b")"
  mapfile -t variants < <(discover_bin_variants "$b")

  if [[ ${#variants[@]} -eq 0 ]]; then
    REPORT+=("[SKIP] $b : no variants discovered in ../bin")
    continue
  fi

  for variant in "${variants[@]}"; do
    printf "==> %s | %s | input=%s ... " "$b" "$variant" "$in0"
    if "$BOTS" "$b" "$variant" "$in0"; then
      echo "OK"
      REPORT+=("[PASS] $b $variant $in0")
      pass+=1
    else
      rc=$?
      echo "FAIL(rc=$rc)"
      REPORT+=("[FAIL] $b $variant $in0 rc=$rc")
      fail+=1
    fi
  done
done

echo
echo "=================== SMOKE SUMMARY ==================="
for line in "${REPORT[@]}"; do echo "$line"; done
echo "-----------------------------------------------------"
echo "PASS: $pass"
echo "FAIL: $fail"
[[ $fail -eq 0 ]] || exit 1
